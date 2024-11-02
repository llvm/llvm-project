//===- vulkan-runtime-wrappers.cpp - MLIR Vulkan runner wrapper library ---===//
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

#include "VulkanRuntime.h"

// Explicitly export entry points to the vulkan-runtime-wrapper.

#ifdef _WIN32
#define VULKAN_WRAPPER_SYMBOL_EXPORT __declspec(dllexport)
#else
#define VULKAN_WRAPPER_SYMBOL_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

namespace {

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

template <typename T, uint32_t S>
void bindMemRef(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                BindingIndex bindIndex, MemRefDescriptor<T, S> *ptr) {
  uint32_t size = sizeof(T);
  for (unsigned i = 0; i < S; i++)
    size *= ptr->sizes[i];
  VulkanHostMemoryBuffer memBuffer{ptr->aligned, size};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

extern "C" {
/// Initializes `VulkanRuntimeManager` and returns a pointer to it.
VULKAN_WRAPPER_SYMBOL_EXPORT void *initVulkan() {
  return new VulkanRuntimeManager();
}

/// Deinitializes `VulkanRuntimeManager` by the given pointer.
VULKAN_WRAPPER_SYMBOL_EXPORT void deinitVulkan(void *vkRuntimeManager) {
  delete reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void runOnVulkan(void *vkRuntimeManager) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)->runOnVulkan();
}

VULKAN_WRAPPER_SYMBOL_EXPORT void setEntryPoint(void *vkRuntimeManager,
                                                const char *entryPoint) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setEntryPoint(entryPoint);
}

VULKAN_WRAPPER_SYMBOL_EXPORT void
setNumWorkGroups(void *vkRuntimeManager, uint32_t x, uint32_t y, uint32_t z) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setNumWorkGroups({x, y, z});
}

VULKAN_WRAPPER_SYMBOL_EXPORT void
setBinaryShader(void *vkRuntimeManager, uint8_t *shader, uint32_t size) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setShaderModule(shader, size);
}

/// Binds the given memref to the given descriptor set and descriptor
/// index.
#define DECLARE_BIND_MEMREF(size, type, typeName)                              \
  VULKAN_WRAPPER_SYMBOL_EXPORT void bindMemRef##size##D##typeName(             \
      void *vkRuntimeManager, DescriptorSetIndex setIndex,                     \
      BindingIndex bindIndex, MemRefDescriptor<type, size> *ptr) {             \
    bindMemRef<type, size>(vkRuntimeManager, setIndex, bindIndex, ptr);        \
  }

DECLARE_BIND_MEMREF(1, float, Float)
DECLARE_BIND_MEMREF(2, float, Float)
DECLARE_BIND_MEMREF(3, float, Float)
DECLARE_BIND_MEMREF(1, int32_t, Int32)
DECLARE_BIND_MEMREF(2, int32_t, Int32)
DECLARE_BIND_MEMREF(3, int32_t, Int32)
DECLARE_BIND_MEMREF(1, int16_t, Int16)
DECLARE_BIND_MEMREF(2, int16_t, Int16)
DECLARE_BIND_MEMREF(3, int16_t, Int16)
DECLARE_BIND_MEMREF(1, int8_t, Int8)
DECLARE_BIND_MEMREF(2, int8_t, Int8)
DECLARE_BIND_MEMREF(3, int8_t, Int8)
DECLARE_BIND_MEMREF(1, int16_t, Half)
DECLARE_BIND_MEMREF(2, int16_t, Half)
DECLARE_BIND_MEMREF(3, int16_t, Half)

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
