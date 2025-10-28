//===- VulkanRuntime.cpp - MLIR Vulkan runtime ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares Vulkan runtime API.
//
//===----------------------------------------------------------------------===//

#ifndef VULKAN_RUNTIME_H
#define VULKAN_RUNTIME_H

#include "mlir/Support/LLVM.h"

#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

using namespace mlir;

using DescriptorSetIndex = uint32_t;
using BindingIndex = uint32_t;

/// Struct containing information regarding to a device memory buffer.
struct VulkanDeviceMemoryBuffer {
  BindingIndex bindingIndex{0};
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
  VkDescriptorBufferInfo bufferInfo{};
  VkBuffer hostBuffer{VK_NULL_HANDLE};
  VkDeviceMemory hostMemory{VK_NULL_HANDLE};
  VkBuffer deviceBuffer{VK_NULL_HANDLE};
  VkDeviceMemory deviceMemory{VK_NULL_HANDLE};
  uint32_t bufferSize{0};
};

/// Struct containing information regarding to a host memory buffer.
struct VulkanHostMemoryBuffer {
  /// Pointer to a host memory.
  void *ptr{nullptr};
  /// Size of a host memory in bytes.
  uint32_t size{0};
};

/// Struct containing the number of local workgroups to dispatch for each
/// dimension.
struct NumWorkGroups {
  uint32_t x{1};
  uint32_t y{1};
  uint32_t z{1};
};

/// Struct containing information regarding a descriptor set.
struct DescriptorSetInfo {
  /// Index of a descriptor set in descriptor sets.
  DescriptorSetIndex descriptorSet{0};
  /// Number of descriptors in a set.
  uint32_t descriptorSize{0};
  /// Type of a descriptor set.
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
};

/// VulkanHostMemoryBuffer mapped into a descriptor set and a binding.
using ResourceData = std::unordered_map<
    DescriptorSetIndex,
    std::unordered_map<BindingIndex, VulkanHostMemoryBuffer>>;

/// SPIR-V storage classes.
/// Note that this duplicates spirv::StorageClass but it keeps the Vulkan
/// runtime library detached from SPIR-V dialect, so we can avoid pick up lots
/// of dependencies.
enum class SPIRVStorageClass {
  Uniform = 2,
  StorageBuffer = 12,
};

/// StorageClass mapped into a descriptor set and a binding.
using ResourceStorageClassBindingMap =
    std::unordered_map<DescriptorSetIndex,
                       std::unordered_map<BindingIndex, SPIRVStorageClass>>;

/// Vulkan runtime.
/// The purpose of this class is to run SPIR-V compute shader on Vulkan
/// device.
/// Before the run, user must provide and set resource data with descriptors,
/// SPIR-V shader, number of work groups and entry point. After the creation of
/// VulkanRuntime, special methods must be called in the following
/// sequence: initRuntime(), run(), updateHostMemoryBuffers(), destroy();
/// each method in the sequence returns success or failure depends on the Vulkan
/// result code.
class VulkanRuntime {
public:
  explicit VulkanRuntime() = default;
  VulkanRuntime(const VulkanRuntime &) = delete;
  VulkanRuntime &operator=(const VulkanRuntime &) = delete;

  /// Sets needed data for Vulkan runtime.
  void setResourceData(const ResourceData &resData);
  void setResourceData(const DescriptorSetIndex desIndex,
                       const BindingIndex bindIndex,
                       const VulkanHostMemoryBuffer &hostMemBuffer);
  void setShaderModule(uint8_t *shader, uint32_t size);
  void setNumWorkGroups(const NumWorkGroups &numberWorkGroups);
  void setResourceStorageClassBindingMap(
      const ResourceStorageClassBindingMap &stClassData);
  void setEntryPoint(const char *entryPointName);

  /// Runtime initialization.
  LogicalResult initRuntime();

  /// Runs runtime.
  LogicalResult run();

  /// Updates host memory buffers.
  LogicalResult updateHostMemoryBuffers();

  /// Destroys all created vulkan objects and resources.
  LogicalResult destroy();

private:
  //===--------------------------------------------------------------------===//
  // Pipeline creation methods.
  //===--------------------------------------------------------------------===//

  LogicalResult createInstance();
  LogicalResult createDevice();
  LogicalResult getBestComputeQueue();
  LogicalResult createMemoryBuffers();
  LogicalResult createShaderModule();
  void initDescriptorSetLayoutBindingMap();
  LogicalResult createDescriptorSetLayout();
  LogicalResult createPipelineLayout();
  LogicalResult createComputePipeline();
  LogicalResult createDescriptorPool();
  LogicalResult allocateDescriptorSets();
  LogicalResult setWriteDescriptors();
  LogicalResult createCommandPool();
  LogicalResult createQueryPool();
  LogicalResult createComputeCommandBuffer();
  LogicalResult submitCommandBuffersToQueue();
  // Copy resources from host (staging buffer) to device buffer or from device
  // buffer to host buffer.
  LogicalResult copyResource(bool deviceToHost);

  //===--------------------------------------------------------------------===//
  // Helper methods.
  //===--------------------------------------------------------------------===//

  /// Maps storage class to a descriptor type.
  LogicalResult
  mapStorageClassToDescriptorType(SPIRVStorageClass storageClass,
                                  VkDescriptorType &descriptorType);

  /// Maps storage class to buffer usage flags.
  LogicalResult
  mapStorageClassToBufferUsageFlag(SPIRVStorageClass storageClass,
                                   VkBufferUsageFlagBits &bufferUsage);

  LogicalResult countDeviceMemorySize();

  //===--------------------------------------------------------------------===//
  // Vulkan objects.
  //===--------------------------------------------------------------------===//

  VkInstance instance{VK_NULL_HANDLE};
  VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
  VkDevice device{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};

  /// Specifies VulkanDeviceMemoryBuffers divided into sets.
  std::unordered_map<DescriptorSetIndex, std::vector<VulkanDeviceMemoryBuffer>>
      deviceMemoryBufferMap;

  /// Specifies shader module.
  VkShaderModule shaderModule{VK_NULL_HANDLE};

  /// Specifies layout bindings.
  std::unordered_map<DescriptorSetIndex,
                     std::vector<VkDescriptorSetLayoutBinding>>
      descriptorSetLayoutBindingMap;

  /// Specifies layouts of descriptor sets.
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
  VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};

  /// Specifies descriptor sets.
  std::vector<VkDescriptorSet> descriptorSets;

  /// Specifies a pool of descriptor set info, each descriptor set must have
  /// information such as type, index and amount of bindings.
  std::vector<DescriptorSetInfo> descriptorSetInfoPool;
  VkDescriptorPool descriptorPool{VK_NULL_HANDLE};

  /// Timestamp query.
  VkQueryPool queryPool{VK_NULL_HANDLE};
  // Number of nonoseconds for timestamp to increase 1
  float timestampPeriod{0.f};

  /// Computation pipeline.
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkCommandPool commandPool{VK_NULL_HANDLE};
  std::vector<VkCommandBuffer> commandBuffers;

  //===--------------------------------------------------------------------===//
  // Vulkan memory context.
  //===--------------------------------------------------------------------===//

  uint32_t queueFamilyIndex{0};
  VkQueueFamilyProperties queueFamilyProperties{};
  uint32_t hostMemoryTypeIndex{VK_MAX_MEMORY_TYPES};
  uint32_t deviceMemoryTypeIndex{VK_MAX_MEMORY_TYPES};
  VkDeviceSize memorySize{0};

  //===--------------------------------------------------------------------===//
  // Vulkan execution context.
  //===--------------------------------------------------------------------===//

  NumWorkGroups numWorkGroups;
  const char *entryPoint{nullptr};
  uint8_t *binary{nullptr};
  uint32_t binarySize{0};

  //===--------------------------------------------------------------------===//
  // Vulkan resource data and storage classes.
  //===--------------------------------------------------------------------===//

  ResourceData resourceData;
  ResourceStorageClassBindingMap resourceStorageClassData;
};
#endif
