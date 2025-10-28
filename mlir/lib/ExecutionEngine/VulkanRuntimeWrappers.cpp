//===- VulkanRuntimeWrappers.cpp - MLIR Vulkan runner wrapper library -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C runtime wrappers around the VulkanRuntime.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include "VulkanRuntime.h"

// Explicitly export entry points to the vulkan-runtime-wrapper.

#ifdef _WIN32
#define VULKAN_WRAPPER_SYMBOL_EXPORT __declspec(dllexport)
#else
#define VULKAN_WRAPPER_SYMBOL_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

namespace {

class VulkanModule;

// Class to be a thing that can be returned from `mgpuModuleGetFunction`.
struct VulkanFunction {
  VulkanModule *module;
  std::string name;

  VulkanFunction(VulkanModule *module, const char *name)
      : module(module), name(name) {}
};

// Class to own a copy of the SPIR-V provided to `mgpuModuleLoad` and to manage
// allocation of pointers returned from `mgpuModuleGetFunction`.
class VulkanModule {
public:
  VulkanModule(const uint8_t *ptr, size_t sizeInBytes)
      : blob(ptr, ptr + sizeInBytes) {}
  ~VulkanModule() = default;

  VulkanFunction *getFunction(const char *name) {
    return functions.emplace_back(std::make_unique<VulkanFunction>(this, name))
        .get();
  }

  uint8_t *blobData() { return blob.data(); }
  size_t blobSizeInBytes() const { return blob.size(); }

private:
  std::vector<uint8_t> blob;
  std::vector<std::unique_ptr<VulkanFunction>> functions;
};

class VulkanRuntimeManager {
public:
  VulkanRuntimeManager() = default;
  VulkanRuntimeManager(const VulkanRuntimeManager &) = delete;
  VulkanRuntimeManager operator=(const VulkanRuntimeManager &) = delete;
  ~VulkanRuntimeManager() = default;

  void setResourceData(DescriptorSetIndex setIndex, BindingIndex bindIndex,
                       const VulkanHostMemoryBuffer &memBuffer) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setResourceData(setIndex, bindIndex, memBuffer);
  }

  void setEntryPoint(const char *entryPoint) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setEntryPoint(entryPoint);
  }

  void setNumWorkGroups(NumWorkGroups numWorkGroups) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setNumWorkGroups(numWorkGroups);
  }

  void setShaderModule(uint8_t *shader, uint32_t size) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setShaderModule(shader, size);
  }

  void runOnVulkan() {
    std::lock_guard<std::mutex> lock(mutex);
    if (failed(vulkanRuntime.initRuntime()) || failed(vulkanRuntime.run()) ||
        failed(vulkanRuntime.updateHostMemoryBuffers()) ||
        failed(vulkanRuntime.destroy())) {
      std::cerr << "runOnVulkan failed";
    }
  }

private:
  VulkanRuntime vulkanRuntime;
  std::mutex mutex;
};

} // namespace

template <typename T, int N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {

//===----------------------------------------------------------------------===//
//
// Wrappers intended for mlir-runner. Uses of GPU dialect operations get
// lowered to calls to these functions by GPUToLLVMConversionPass.
//
//===----------------------------------------------------------------------===//

VULKAN_WRAPPER_SYMBOL_EXPORT void *mgpuStreamCreate() {
  return new VulkanRuntimeManager();
}

VULKAN_WRAPPER_SYMBOL_EXPORT void mgpuStreamDestroy(void *vkRuntimeManager) {
  delete static_cast<VulkanRuntimeManager *>(vkRuntimeManager);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void mgpuStreamSynchronize(void *) {
  // Currently a no-op as the other operations are synchronous.
}

VULKAN_WRAPPER_SYMBOL_EXPORT void *mgpuModuleLoad(const void *data,
                                                  size_t gpuBlobSize) {
  // gpuBlobSize is the size of the data in bytes.
  return new VulkanModule(static_cast<const uint8_t *>(data), gpuBlobSize);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void mgpuModuleUnload(void *vkModule) {
  delete static_cast<VulkanModule *>(vkModule);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void *mgpuModuleGetFunction(void *vkModule,
                                                         const char *name) {
  if (!vkModule)
    abort();
  return static_cast<VulkanModule *>(vkModule)->getFunction(name);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void
mgpuLaunchKernel(void *vkKernel, size_t gridX, size_t gridY, size_t gridZ,
                 size_t /*blockX*/, size_t /*blockY*/, size_t /*blockZ*/,
                 size_t /*smem*/, void *vkRuntimeManager, void **params,
                 void ** /*extra*/, size_t paramsCount) {
  auto manager = static_cast<VulkanRuntimeManager *>(vkRuntimeManager);

  // GpuToLLVMConversionPass with the kernelBarePtrCallConv and
  // kernelIntersperseSizeCallConv options will set up the params array like:
  // { &memref_ptr0, &memref_size0, &memref_ptr1, &memref_size1, ... }
  const size_t paramsPerMemRef = 2;
  if (paramsCount % paramsPerMemRef != 0) {
    abort(); // This would indicate a serious calling convention mismatch.
  }
  const DescriptorSetIndex setIndex = 0;
  BindingIndex bindIndex = 0;
  for (size_t i = 0; i < paramsCount; i += paramsPerMemRef) {
    void *memrefBufferBasePtr = *static_cast<void **>(params[i + 0]);
    size_t memrefBufferSize = *static_cast<size_t *>(params[i + 1]);
    VulkanHostMemoryBuffer memBuffer{memrefBufferBasePtr,
                                     static_cast<uint32_t>(memrefBufferSize)};
    manager->setResourceData(setIndex, bindIndex, memBuffer);
    ++bindIndex;
  }

  manager->setNumWorkGroups(NumWorkGroups{static_cast<uint32_t>(gridX),
                                          static_cast<uint32_t>(gridY),
                                          static_cast<uint32_t>(gridZ)});

  auto function = static_cast<VulkanFunction *>(vkKernel);
  // Expected size should be in bytes.
  manager->setShaderModule(
      function->module->blobData(),
      static_cast<uint32_t>(function->module->blobSizeInBytes()));
  manager->setEntryPoint(function->name.c_str());

  manager->runOnVulkan();
}

//===----------------------------------------------------------------------===//
//
// Miscellaneous utility functions that can be directly used by tests.
//
//===----------------------------------------------------------------------===//

/// Fills the given 1D float memref with the given float value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource1DFloat(MemRefDescriptor<float, 1> *ptr, // NOLINT
                                 float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

/// Fills the given 2D float memref with the given float value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource2DFloat(MemRefDescriptor<float, 2> *ptr, // NOLINT
                                 float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1], value);
}

/// Fills the given 3D float memref with the given float value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource3DFloat(MemRefDescriptor<float, 3> *ptr, // NOLINT
                                 float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1] * ptr->sizes[2],
              value);
}

/// Fills the given 1D int memref with the given int value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource1DInt(MemRefDescriptor<int32_t, 1> *ptr, // NOLINT
                               int32_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

/// Fills the given 2D int memref with the given int value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource2DInt(MemRefDescriptor<int32_t, 2> *ptr, // NOLINT
                               int32_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1], value);
}

/// Fills the given 3D int memref with the given int value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource3DInt(MemRefDescriptor<int32_t, 3> *ptr, // NOLINT
                               int32_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1] * ptr->sizes[2],
              value);
}

/// Fills the given 1D int memref with the given int8 value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource1DInt8(MemRefDescriptor<int8_t, 1> *ptr, // NOLINT
                                int8_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

/// Fills the given 2D int memref with the given int8 value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource2DInt8(MemRefDescriptor<int8_t, 2> *ptr, // NOLINT
                                int8_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1], value);
}

/// Fills the given 3D int memref with the given int8 value.
VULKAN_WRAPPER_SYMBOL_EXPORT void
_mlir_ciface_fillResource3DInt8(MemRefDescriptor<int8_t, 3> *ptr, // NOLINT
                                int8_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1] * ptr->sizes[2],
              value);
}
}
