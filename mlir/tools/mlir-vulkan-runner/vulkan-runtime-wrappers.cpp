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

extern "C" {
/// Initializes `VulkanRuntimeManager` and returns a pointer to it.
void *initVulkan() { return new VulkanRuntimeManager(); }

/// Deinitializes `VulkanRuntimeManager` by the given pointer.
void deinitVulkan(void *vkRuntimeManager) {
  delete reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager);
}

void runOnVulkan(void *vkRuntimeManager) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)->runOnVulkan();
}

void setEntryPoint(void *vkRuntimeManager, const char *entryPoint) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setEntryPoint(entryPoint);
}

void setNumWorkGroups(void *vkRuntimeManager, uint32_t x, uint32_t y,
                      uint32_t z) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setNumWorkGroups({x, y, z});
}

void setBinaryShader(void *vkRuntimeManager, uint8_t *shader, uint32_t size) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setShaderModule(shader, size);
}

/// Binds the given 1D float memref to the given descriptor set and descriptor
/// index.
void bindMemRef1DFloat(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                       BindingIndex bindIndex,
                       MemRefDescriptor<float, 1> *ptr) {
  VulkanHostMemoryBuffer memBuffer{
      ptr->allocated, static_cast<uint32_t>(ptr->sizes[0] * sizeof(float))};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

/// Binds the given 2D float memref to the given descriptor set and descriptor
/// index.
void bindMemRef2DFloat(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                       BindingIndex bindIndex,
                       MemRefDescriptor<float, 2> *ptr) {
  VulkanHostMemoryBuffer memBuffer{
      ptr->allocated,
      static_cast<uint32_t>(ptr->sizes[0] * ptr->sizes[1] * sizeof(float))};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

/// Binds the given 3D float memref to the given descriptor set and descriptor
/// index.
void bindMemRef3DFloat(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                       BindingIndex bindIndex,
                       MemRefDescriptor<float, 3> *ptr) {
  VulkanHostMemoryBuffer memBuffer{
      ptr->allocated, static_cast<uint32_t>(ptr->sizes[0] * ptr->sizes[1] *
                                            ptr->sizes[2] * sizeof(float))};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

/// Binds the given 1D int memref to the given descriptor set and descriptor
/// index.
void bindMemRef1DInt(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                     BindingIndex bindIndex,
                     MemRefDescriptor<int32_t, 1> *ptr) {
  VulkanHostMemoryBuffer memBuffer{
      ptr->allocated, static_cast<uint32_t>(ptr->sizes[0] * sizeof(int32_t))};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

/// Binds the given 2D int memref to the given descriptor set and descriptor
/// index.
void bindMemRef2DInt(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                     BindingIndex bindIndex,
                     MemRefDescriptor<int32_t, 2> *ptr) {
  VulkanHostMemoryBuffer memBuffer{
      ptr->allocated,
      static_cast<uint32_t>(ptr->sizes[0] * ptr->sizes[1] * sizeof(int32_t))};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

/// Binds the given 3D int memref to the given descriptor set and descriptor
/// index.
void bindMemRef3DInt(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                     BindingIndex bindIndex,
                     MemRefDescriptor<int32_t, 3> *ptr) {
  VulkanHostMemoryBuffer memBuffer{
      ptr->allocated, static_cast<uint32_t>(ptr->sizes[0] * ptr->sizes[1] *
                                            ptr->sizes[2] * sizeof(int32_t))};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

/// Fills the given 1D float memref with the given float value.
void _mlir_ciface_fillResource1DFloat(MemRefDescriptor<float, 1> *ptr, // NOLINT
                                      float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

/// Fills the given 2D float memref with the given float value.
void _mlir_ciface_fillResource2DFloat(MemRefDescriptor<float, 2> *ptr, // NOLINT
                                      float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1], value);
}

/// Fills the given 3D float memref with the given float value.
void _mlir_ciface_fillResource3DFloat(MemRefDescriptor<float, 3> *ptr, // NOLINT
                                      float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1] * ptr->sizes[2],
              value);
}

/// Fills the given 1D int memref with the given int value.
void _mlir_ciface_fillResource1DInt(MemRefDescriptor<int32_t, 1> *ptr, // NOLINT
                                    int32_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

/// Fills the given 2D int memref with the given int value.
void _mlir_ciface_fillResource2DInt(MemRefDescriptor<int32_t, 2> *ptr, // NOLINT
                                    int32_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1], value);
}

/// Fills the given 3D int memref with the given int value.
void _mlir_ciface_fillResource3DInt(MemRefDescriptor<int32_t, 3> *ptr, // NOLINT
                                    int32_t value) {
  std::fill_n(ptr->allocated, ptr->sizes[0] * ptr->sizes[1] * ptr->sizes[2],
              value);
}
}
