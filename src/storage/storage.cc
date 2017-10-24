/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 */
#include <mxnet/storage.h>
#include <mshadow/tensor.h>
#include <dmlc/logging.h>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <array>
#include "./storage_manager.h"
#include "./naive_storage_manager.h"
#include "./pooled_storage_manager.h"
#include "./cpu_device_storage.h"
#include "./pinned_memory_storage.h"
#include "../common/cuda_utils.h"
#include "../common/lazy_alloc_array.h"

namespace mxnet {

// consider change storage as a pure abstract class
class StorageImpl : public Storage {
 public:
  Handle Alloc(size_t size, Context ctx) override;
  void Free(Handle handle) override;
  void DirectFree(Handle handle) override;
  Handle SharedAlloc(size_t size) override;
  Handle SharedRetrieve(const char* filename, size_t size) override;
  void SharedFree(Handle handle, bool unlink = true) override;
  StorageImpl() {}
  virtual ~StorageImpl() = default;

 private:
  static constexpr size_t kMaxNumberOfDevices = Context::kMaxDevType + 1;
  static constexpr size_t kMaxNumberOfDeviceIDs = Context::kMaxDevID + 1;
#if MXNET_USE_CUDA
  static int num_gpu_device;
#endif  // MXNET_USE_CUDA

  static void ActivateDevice(Context ctx) {
    switch (ctx.dev_type) {
      case Context::kCPU: break;
      case Context::kGPU:
      case Context::kCPUPinned: {
#if MXNET_USE_CUDA
          if (num_gpu_device > 0) {
            CUDA_CALL(cudaSetDevice(ctx.dev_id));
          }
#endif  // MXNET_USE_CUDA
          break;
        }
      default:
        LOG(FATAL) << "Unimplemented device";
    }
  }
  // internal storage managers
  std::array<common::LazyAllocArray<storage::StorageManager>,
             kMaxNumberOfDevices> storage_managers_;
};  // struct Storage::Impl
#if MXNET_USE_CUDA
int StorageImpl::num_gpu_device = 0;
#endif  // MXNET_USE_CUDA

Storage::Handle StorageImpl::Alloc(size_t size, Context ctx) {
  // space already recycled, ignore request
  Handle hd;
  hd.ctx = ctx;
  hd.size = size;
  auto&& device = storage_managers_.at(ctx.dev_type);
  std::shared_ptr<storage::StorageManager> manager = device.Get(
      ctx.dev_id, [ctx]() {
        storage::StorageManager *ptr = nullptr;
        switch (ctx.dev_type) {
          case Context::kCPU: {
            ptr = new storage::NaiveStorageManager<storage::CPUDeviceStorage>();
            break;
          }
          case Context::kCPUPinned: {
#if MXNET_USE_CUDA
            num_gpu_device = 0;
            cudaError_t e = cudaGetDeviceCount(&num_gpu_device);
            if (e != cudaSuccess) {
              num_gpu_device = 0;
            }
            if (num_gpu_device > 0) {
              ptr = new storage::NaiveStorageManager<storage::PinnedMemoryStorage>();
            } else {
              ptr = new storage::NaiveStorageManager<storage::CPUDeviceStorage>();
            }
#else
            ptr = new storage::NaiveStorageManager<storage::CPUDeviceStorage>();
#endif  // MXNET_USE_CUDA
            break;
          }
          case Context::kGPU: {
#if MXNET_USE_CUDA
            CUDA_CALL(cudaGetDeviceCount(&num_gpu_device));
            CHECK_GT(num_gpu_device, 0) << "GPU usage requires at least 1 GPU";
            ptr = new storage::GPUPooledStorageManager();
#else
            LOG(FATAL) << "Compile with USE_CUDA=1 to enable GPU usage";
#endif  // MXNET_USE_CUDA
            break;
          }
          default: LOG(FATAL) <<  "Unimplemented device " << ctx.dev_type;
        }
        return ptr;
      });
  this->ActivateDevice(ctx);
  hd.dptr = manager->Alloc(size);
  return hd;
}

Storage::Handle StorageImpl::SharedAlloc(size_t size) {
  const int MAX_FILENAME = 24;
  Handle hd;
  hd.ctx = Context::CPU(0);
  hd.size = size;

  char* filename = new char[MAX_FILENAME];
  int fid;
  for(int i = 0; i < 10; ++i) {
    snprintf(filename, MAX_FILENAME, "/mx_%08x_%08x", getpid(), std::rand());
    if ((fid = shm_open(filename, O_EXCL|O_CREAT|O_RDWR, 0666)) != -1) break;
    LOG(INFO) << filename;
  }
  if (fid == -1) {
    LOG(FATAL)
      << "Unabled to create shared memory."
      << ". shm_open failed with error " << strerror(errno);
  }
  ftruncate(fid, size);
  hd.dptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fid, 0);
  hd.filename = filename;

  return hd;
}

Storage::Handle StorageImpl::SharedRetrieve(const char* filename, size_t size) {
  Handle hd;
  hd.ctx = Context::CPU(0);
  hd.size = size;
  hd.filename = new char[strlen(filename)+1];
  strcpy(hd.filename, filename);
  int fid = shm_open(hd.filename, O_RDWR, 0666);
  CHECK_NE(fid, -1)
    << "Failed to open shared memory " << filename
    << ". shm_open failed with error " << strerror(errno);
  hd.dptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fid, 0);

  return hd;
}

void StorageImpl::SharedFree(Storage::Handle handle, bool unlink) {
  CHECK(handle.filename != nullptr);
  CHECK_EQ(munmap(handle.dptr, handle.size), 0)
    << "Failed to unmap shared memory " << handle.filename;
  if (unlink) {
    CHECK_EQ(shm_unlink(handle.filename), 0)
      << "Failed to unlink shared memory " << handle.filename;
  }
  delete handle.filename;
}

void StorageImpl::Free(Storage::Handle handle) {
  if (handle.filename != nullptr) {
    SharedFree(handle);
    return;
  }
  const Context &ctx = handle.ctx;
  auto&& device = storage_managers_.at(ctx.dev_type);
  std::shared_ptr<storage::StorageManager> manager = device.Get(
      ctx.dev_id, []() {
        LOG(FATAL) <<  "Cannot Free space to a device you have not allocated";
        return nullptr;
      });
  this->ActivateDevice(ctx);
  manager->Free(handle.dptr, handle.size);
}

void StorageImpl::DirectFree(Storage::Handle handle) {
  const Context &ctx = handle.ctx;
  auto&& device = storage_managers_.at(ctx.dev_type);
  std::shared_ptr<storage::StorageManager> manager = device.Get(
      ctx.dev_id, []() {
        LOG(FATAL) <<  "Cannot Free space to a device you have not allocated";
        return nullptr;
      });
  this->ActivateDevice(ctx);
  // directly free ths data.
  manager->DirectFree(handle.dptr, handle.size);
}

std::shared_ptr<Storage> Storage::_GetSharedRef() {
#ifdef __MXNET_JS__
  // dummy code needed for emscripten code to pass
  // do not know why, the new will be NULLPTR
  static int *q = new int();
#endif
  static std::shared_ptr<Storage> inst(new StorageImpl());
  return inst;
}

Storage* Storage::Get() {
  static Storage *ptr = _GetSharedRef().get();
  return ptr;
}
}  // namespace mxnet
