//===- SyclRuntimeWrappers.cpp - MLIR Sycl API wrapper library ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>
#include <map>
#include <mutex>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#ifdef _WIN32
#define SYCL_RUNTIME_EXPORT __declspec(dllexport)
#else
#define SYCL_RUNTIME_EXPORT
#endif // _WIN32

namespace {

template <typename F> auto catchAll(F &&func) {
  try {
    return func();
  } catch (const std::exception &e) {
    fprintf(stdout, "An exception was thrown: %s\n", e.what());
    fflush(stdout);
    abort();
  } catch (...) {
    fprintf(stdout, "An unknown exception was thrown\n");
    fflush(stdout);
    abort();
  }
}

#define L0_SAFE_CALL(call)                                                     \
  {                                                                            \
    ze_result_t status = (call);                                               \
    if (status != ZE_RESULT_SUCCESS) {                                         \
      fprintf(stdout, "L0 error %d\n", status);                                \
      fflush(stdout);                                                          \
      abort();                                                                 \
    }                                                                          \
  }

} // namespace

struct SpirvModule {
  ze_module_handle_t module = nullptr;
  ~SpirvModule();
};

namespace {
// Create a Map for the spirv module lookup
std::map<void *, SpirvModule> moduleCache;
std::mutex mutexLock;
} // namespace

SpirvModule::~SpirvModule() {
  L0_SAFE_CALL(zeModuleDestroy(SpirvModule::module));
}

struct ParamDesc {
  void *data;
  size_t size;

  bool operator==(const ParamDesc &rhs) const {
    return data == rhs.data && size == rhs.size;
  }

  bool operator!=(const ParamDesc &rhs) const { return !(*this == rhs); }
};

template <typename T> size_t countUntil(T *ptr, T &&elem) {
  assert(ptr);
  auto curr = ptr;
  while (*curr != elem) {
    ++curr;
  }
  return static_cast<size_t>(curr - ptr);
}

static sycl::device getDefaultDevice() {
  auto platformList = sycl::platform::get_platforms();
  for (const auto &platform : platformList) {
    auto platformName = platform.get_info<sycl::info::platform::name>();
    bool isLevelZero = platformName.find("Level-Zero") != std::string::npos;
    if (!isLevelZero)
      continue;

    return platform.get_devices()[0];
  }
}

struct GPUSYCLQUEUE {

  sycl::device syclDevice_;
  sycl::context syclContext_;
  sycl::queue syclQueue_;

  GPUSYCLQUEUE(sycl::property_list propList) {

    syclDevice_ = getDefaultDevice();
    syclContext_ = sycl::context(syclDevice_);
    syclQueue_ = sycl::queue(syclContext_, syclDevice_, propList);
  }

  GPUSYCLQUEUE(sycl::device *device, sycl::context *context,
               sycl::property_list propList) {
    syclDevice_ = *device;
    syclContext_ = *context;
    syclQueue_ = sycl::queue(syclContext_, syclDevice_, propList);
  }
  GPUSYCLQUEUE(sycl::device *device, sycl::property_list propList) {

    syclDevice_ = *device;
    syclContext_ = sycl::context(syclDevice_);
    syclQueue_ = sycl::queue(syclContext_, syclDevice_, propList);
  }

  GPUSYCLQUEUE(sycl::context *context, sycl::property_list propList) {

    syclDevice_ = getDefaultDevice();
    syclContext_ = *context;
    syclQueue_ = sycl::queue(syclContext_, syclDevice_, propList);
  }

}; // end of GPUSYCLQUEUE

static void *allocDeviceMemory(GPUSYCLQUEUE *queue, size_t size,
                               size_t alignment, bool isShared) {
  void *memPtr = nullptr;
  if (isShared) {
    memPtr = sycl::aligned_alloc_shared(alignment, size, queue->syclQueue_);
  } else {
    memPtr = sycl::aligned_alloc_device(alignment, size, queue->syclQueue_);
  }
  if (memPtr == nullptr) {
    throw std::runtime_error(
        "aligned_alloc_shared() failed to allocate memory!");
  }
  return memPtr;
}

static void deallocDeviceMemory(GPUSYCLQUEUE *queue, void *ptr) {
  sycl::free(ptr, queue->syclQueue_);
}

static ze_module_handle_t loadModule(GPUSYCLQUEUE *queue, const void *data,
                                     size_t dataSize) {
  assert(data);
  auto syclQueue = queue->syclQueue_;
  ze_module_handle_t zeModule;

  auto it = moduleCache.find((void *)data);
  // Check the map if the module is present/cached.
  if (it != moduleCache.end()) {
    return it->second.module;
  }

  ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                           nullptr,
                           ZE_MODULE_FORMAT_IL_SPIRV,
                           dataSize,
                           (const uint8_t *)data,
                           nullptr,
                           nullptr};
  auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      syclQueue.get_device());
  auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      syclQueue.get_context());
  L0_SAFE_CALL(zeModuleCreate(zeContext, zeDevice, &desc, &zeModule, nullptr));
  std::lock_guard<std::mutex> entryLock(mutexLock);
  moduleCache[(void *)data].module = zeModule;
  return zeModule;
}

static sycl::kernel *getKernel(GPUSYCLQUEUE *queue, ze_module_handle_t zeModule,
                               const char *name) {
  assert(zeModule);
  assert(name);
  auto syclQueue = queue->syclQueue_;
  ze_kernel_handle_t zeKernel;
  sycl::kernel *syclKernel;
  ze_kernel_desc_t desc = {};
  desc.pKernelName = name;

  L0_SAFE_CALL(zeKernelCreate(zeModule, &desc, &zeKernel));
  sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>(
          {zeModule}, syclQueue.get_context());

  auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {kernelBundle, zeKernel}, syclQueue.get_context());
  syclKernel = new sycl::kernel(kernel);
  return syclKernel;
}

static sycl::event enqueueKernel(sycl::queue queue, sycl::kernel *kernel,
                                 sycl::nd_range<3> NdRange, ParamDesc *params,
                                 size_t sharedMemBytes) {
  auto paramsCount = countUntil(params, ParamDesc{nullptr, 0});
  // The assumption is, if there is a param for the shared local memory,
  // then that will always be the last argument.
  if (sharedMemBytes) {
    paramsCount = paramsCount - 1;
  }
  sycl::event event = queue.submit([&](sycl::handler &cgh) {
    for (size_t i = 0; i < paramsCount; i++) {
      auto param = params[i];
      cgh.set_arg(static_cast<uint32_t>(i),
                  *(static_cast<void **>(param.data)));
    }
    if (sharedMemBytes) {
      // TODO: Handle other data types
      using share_mem_t =
          sycl::accessor<float, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>;
      share_mem_t local_buffer =
          share_mem_t(sharedMemBytes / sizeof(float), cgh);
      cgh.set_arg(paramsCount, local_buffer);
      cgh.parallel_for(NdRange, *kernel);
    } else {
      cgh.parallel_for(NdRange, *kernel);
    }
  });
  return event;
}

static void launchKernel(GPUSYCLQUEUE *queue, sycl::kernel *kernel,
                         size_t gridX, size_t gridY, size_t gridZ,
                         size_t blockX, size_t blockY, size_t blockZ,
                         size_t sharedMemBytes, ParamDesc *params) {
  auto syclQueue = queue->syclQueue_;
  auto syclGlobalRange =
      ::sycl::range<3>(blockZ * gridZ, blockY * gridY, blockX * gridX);
  auto syclLocalRange = ::sycl::range<3>(blockZ, blockY, blockX);
  sycl::nd_range<3> syclNdRange(
      sycl::nd_range<3>(syclGlobalRange, syclLocalRange));

  if (getenv("IMEX_ENABLE_PROFILING")) {
    auto executionTime = 0.0f;
    auto maxTime = 0.0f;
    auto minTime = FLT_MAX;
    auto rounds = 100;
    auto warmups = 3;

    if (getenv("IMEX_PROFILING_RUNS")) {
      auto runs = strtol(getenv("IMEX_PROFILING_RUNS"), NULL, 10L);
      if (runs)
        rounds = runs;
    }

    if (getenv("IMEX_PROFILING_WARMUPS")) {
      auto runs = strtol(getenv("IMEX_PROFILING_WARMUPS"), NULL, 10L);
      if (warmups)
        warmups = runs;
    }

    // warmups
    for (int r = 0; r < warmups; r++) {
      enqueueKernel(syclQueue, kernel, syclNdRange, params, sharedMemBytes);
    }

    for (int r = 0; r < rounds; r++) {
      sycl::event event =
          enqueueKernel(syclQueue, kernel, syclNdRange, params, sharedMemBytes);

      auto startTime = event.get_profiling_info<
          cl::sycl::info::event_profiling::command_start>();
      auto endTime = event.get_profiling_info<
          cl::sycl::info::event_profiling::command_end>();
      auto gap = float(endTime - startTime) / 1000000.0f;
      executionTime += gap;
      if (gap > maxTime)
        maxTime = gap;
      if (gap < minTime)
        minTime = gap;
    }

    fprintf(stdout,
            "the kernel execution time is (ms):"
            "avg: %.4f, min: %.4f, max: %.4f (over %d runs)\n",
            executionTime / rounds, minTime, maxTime, rounds);
  } else {
    enqueueKernel(syclQueue, kernel, syclNdRange, params, sharedMemBytes);
  }
}

// Wrappers

extern "C" SYCL_RUNTIME_EXPORT GPUSYCLQUEUE *gpuCreateStream(void *device,
                                                             void *context) {
  auto propList = sycl::property_list{};
  if (getenv("IMEX_ENABLE_PROFILING")) {
    propList = sycl::property_list{sycl::property::queue::enable_profiling()};
  }
  return catchAll([&]() {
    if (!device && !context) {
      return new GPUSYCLQUEUE(propList);
    } else if (device && context) {
      // TODO: Check if the pointers/address is valid and holds the correct
      // device and context
      return new GPUSYCLQUEUE(static_cast<sycl::device *>(device),
                              static_cast<sycl::context *>(context), propList);
    } else if (device && !context) {
      return new GPUSYCLQUEUE(static_cast<sycl::device *>(device), propList);
    } else {
      return new GPUSYCLQUEUE(static_cast<sycl::context *>(context), propList);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuStreamDestroy(GPUSYCLQUEUE *queue) {
  catchAll([&]() { delete queue; });
}

extern "C" SYCL_RUNTIME_EXPORT void *
gpuMemAlloc(GPUSYCLQUEUE *queue, size_t size, size_t alignment, bool isShared) {
  return catchAll([&]() {
    if (queue) {
      return allocDeviceMemory(queue, size, alignment, isShared);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuMemFree(GPUSYCLQUEUE *queue, void *ptr) {
  catchAll([&]() {
    if (queue && ptr) {
      deallocDeviceMemory(queue, ptr);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT ze_module_handle_t
gpuModuleLoad(GPUSYCLQUEUE *queue, const void *data, size_t dataSize) {
  return catchAll([&]() {
    if (queue) {
      return loadModule(queue, data, dataSize);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT sycl::kernel *
gpuKernelGet(GPUSYCLQUEUE *queue, ze_module_handle_t module, const char *name) {
  return catchAll([&]() {
    if (queue) {
      return getKernel(queue, module, name);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void
gpuLaunchKernel(GPUSYCLQUEUE *queue, sycl::kernel *kernel, size_t gridX,
                size_t gridY, size_t gridZ, size_t blockX, size_t blockY,
                size_t blockZ, size_t sharedMemBytes, void *params) {
  return catchAll([&]() {
    if (queue) {
      launchKernel(queue, kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
                   sharedMemBytes, static_cast<ParamDesc *>(params));
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuWait(GPUSYCLQUEUE *queue) {

  catchAll([&]() {
    if (queue) {
      queue->syclQueue_.wait();
    }
  });
}
