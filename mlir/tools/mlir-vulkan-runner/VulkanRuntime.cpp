//===- VulkanRuntime.cpp - MLIR Vulkan runtime ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a library for running a module on a Vulkan device.
// Implements a Vulkan runtime.
//
//===----------------------------------------------------------------------===//

#include "VulkanRuntime.h"

#include <chrono>
#include <cstring>
// TODO: It's generally bad to access stdout/stderr in a library.
// Figure out a better way for error reporting.
#include <iomanip>
#include <iostream>

inline void emitVulkanError(const char *api, VkResult error) {
  std::cerr << " failed with error code " << error << " when executing " << api;
}

#define RETURN_ON_VULKAN_ERROR(result, api)                                    \
  if ((result) != VK_SUCCESS) {                                                \
    emitVulkanError(api, (result));                                            \
    return failure();                                                          \
  }

using namespace mlir;

void VulkanRuntime::setNumWorkGroups(const NumWorkGroups &numberWorkGroups) {
  numWorkGroups = numberWorkGroups;
}

void VulkanRuntime::setResourceStorageClassBindingMap(
    const ResourceStorageClassBindingMap &stClassData) {
  resourceStorageClassData = stClassData;
}

void VulkanRuntime::setResourceData(
    const DescriptorSetIndex desIndex, const BindingIndex bindIndex,
    const VulkanHostMemoryBuffer &hostMemBuffer) {
  resourceData[desIndex][bindIndex] = hostMemBuffer;
  resourceStorageClassData[desIndex][bindIndex] =
      SPIRVStorageClass::StorageBuffer;
}

void VulkanRuntime::setEntryPoint(const char *entryPointName) {
  entryPoint = entryPointName;
}

void VulkanRuntime::setResourceData(const ResourceData &resData) {
  resourceData = resData;
}

void VulkanRuntime::setShaderModule(uint8_t *shader, uint32_t size) {
  binary = shader;
  binarySize = size;
}

LogicalResult VulkanRuntime::mapStorageClassToDescriptorType(
    SPIRVStorageClass storageClass, VkDescriptorType &descriptorType) {
  switch (storageClass) {
  case SPIRVStorageClass::StorageBuffer:
    descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    break;
  case SPIRVStorageClass::Uniform:
    descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    break;
  }
  return success();
}

LogicalResult VulkanRuntime::mapStorageClassToBufferUsageFlag(
    SPIRVStorageClass storageClass, VkBufferUsageFlagBits &bufferUsage) {
  switch (storageClass) {
  case SPIRVStorageClass::StorageBuffer:
    bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    break;
  case SPIRVStorageClass::Uniform:
    bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    break;
  }
  return success();
}

LogicalResult VulkanRuntime::countDeviceMemorySize() {
  for (const auto &resourceDataMapPair : resourceData) {
    const auto &resourceDataMap = resourceDataMapPair.second;
    for (const auto &resourceDataBindingPair : resourceDataMap) {
      if (resourceDataBindingPair.second.size) {
        memorySize += resourceDataBindingPair.second.size;
      } else {
        std::cerr << "expected buffer size greater than zero for resource data";
        return failure();
      }
    }
  }
  return success();
}

LogicalResult VulkanRuntime::initRuntime() {
  if (!resourceData.size()) {
    std::cerr << "Vulkan runtime needs at least one resource";
    return failure();
  }
  if (!binarySize || !binary) {
    std::cerr << "binary shader size must be greater than zero";
    return failure();
  }
  if (failed(countDeviceMemorySize())) {
    return failure();
  }
  return success();
}

LogicalResult VulkanRuntime::destroy() {
  // According to Vulkan spec:
  // "To ensure that no work is active on the device, vkDeviceWaitIdle can be
  // used to gate the destruction of the device. Prior to destroying a device,
  // an application is responsible for destroying/freeing any Vulkan objects
  // that were created using that device as the first parameter of the
  // corresponding vkCreate* or vkAllocate* command."
  RETURN_ON_VULKAN_ERROR(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");

  // Free and destroy.
  vkFreeCommandBuffers(device, commandPool, commandBuffers.size(),
                       commandBuffers.data());
  vkDestroyQueryPool(device, queryPool, nullptr);
  vkDestroyCommandPool(device, commandPool, nullptr);
  vkFreeDescriptorSets(device, descriptorPool, descriptorSets.size(),
                       descriptorSets.data());
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  for (auto &descriptorSetLayout : descriptorSetLayouts) {
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
  }
  vkDestroyShaderModule(device, shaderModule, nullptr);

  // For each descriptor set.
  for (auto &deviceMemoryBufferMapPair : deviceMemoryBufferMap) {
    auto &deviceMemoryBuffers = deviceMemoryBufferMapPair.second;
    // For each descriptor binding.
    for (auto &memoryBuffer : deviceMemoryBuffers) {
      vkFreeMemory(device, memoryBuffer.deviceMemory, nullptr);
      vkFreeMemory(device, memoryBuffer.hostMemory, nullptr);
      vkDestroyBuffer(device, memoryBuffer.hostBuffer, nullptr);
      vkDestroyBuffer(device, memoryBuffer.deviceBuffer, nullptr);
    }
  }

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  return success();
}

LogicalResult VulkanRuntime::run() {
  // Create logical device, shader module and memory buffers.
  if (failed(createInstance()) || failed(createDevice()) ||
      failed(createMemoryBuffers()) || failed(createShaderModule())) {
    return failure();
  }

  // Descriptor bindings divided into sets. Each descriptor binding
  // must have a layout binding attached into a descriptor set layout.
  // Each layout set must be binded into a pipeline layout.
  initDescriptorSetLayoutBindingMap();
  if (failed(createDescriptorSetLayout()) || failed(createPipelineLayout()) ||
      // Each descriptor set must be allocated from a descriptor pool.
      failed(createComputePipeline()) || failed(createDescriptorPool()) ||
      failed(allocateDescriptorSets()) || failed(setWriteDescriptors()) ||
      // Create command buffer.
      failed(createCommandPool()) || failed(createQueryPool()) ||
      failed(createComputeCommandBuffer())) {
    return failure();
  }

  // Get working queue.
  vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

  if (failed(copyResource(/*deviceToHost=*/false)))
    return failure();

  auto submitStart = std::chrono::high_resolution_clock::now();
  // Submit command buffer into the queue.
  if (failed(submitCommandBuffersToQueue()))
    return failure();
  auto submitEnd = std::chrono::high_resolution_clock::now();

  RETURN_ON_VULKAN_ERROR(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
  auto execEnd = std::chrono::high_resolution_clock::now();

  auto submitDuration = std::chrono::duration_cast<std::chrono::microseconds>(
      submitEnd - submitStart);
  auto execDuration = std::chrono::duration_cast<std::chrono::microseconds>(
      execEnd - submitEnd);

  if (queryPool != VK_NULL_HANDLE) {
    uint64_t timestamps[2];
    RETURN_ON_VULKAN_ERROR(
        vkGetQueryPoolResults(
            device, queryPool, /*firstQuery=*/0, /*queryCount=*/2,
            /*dataSize=*/sizeof(timestamps),
            /*pData=*/reinterpret_cast<void *>(timestamps),
            /*stride=*/sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
        "vkGetQueryPoolResults");
    float microsec = (timestamps[1] - timestamps[0]) * timestampPeriod / 1000;
    std::cout << "Compute shader execution time: " << std::setprecision(3)
              << microsec << "us\n";
  }

  std::cout << "Command buffer submit time: " << submitDuration.count()
            << "us\nWait idle time: " << execDuration.count() << "us\n";

  return success();
}

LogicalResult VulkanRuntime::createInstance() {
  VkApplicationInfo applicationInfo = {};
  applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationInfo.pNext = nullptr;
  applicationInfo.pApplicationName = "MLIR Vulkan runtime";
  applicationInfo.applicationVersion = 0;
  applicationInfo.pEngineName = "mlir";
  applicationInfo.engineVersion = 0;
  applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

  VkInstanceCreateInfo instanceCreateInfo = {};
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pNext = nullptr;
  instanceCreateInfo.flags = 0;
  instanceCreateInfo.pApplicationInfo = &applicationInfo;
  instanceCreateInfo.enabledLayerCount = 0;
  instanceCreateInfo.ppEnabledLayerNames = nullptr;
  instanceCreateInfo.enabledExtensionCount = 0;
  instanceCreateInfo.ppEnabledExtensionNames = nullptr;

  RETURN_ON_VULKAN_ERROR(
      vkCreateInstance(&instanceCreateInfo, nullptr, &instance),
      "vkCreateInstance");
  return success();
}

LogicalResult VulkanRuntime::createDevice() {
  uint32_t physicalDeviceCount = 0;
  RETURN_ON_VULKAN_ERROR(
      vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr),
      "vkEnumeratePhysicalDevices");

  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  RETURN_ON_VULKAN_ERROR(vkEnumeratePhysicalDevices(instance,
                                                    &physicalDeviceCount,
                                                    physicalDevices.data()),
                         "vkEnumeratePhysicalDevices");

  RETURN_ON_VULKAN_ERROR(physicalDeviceCount ? VK_SUCCESS : VK_INCOMPLETE,
                         "physicalDeviceCount");

  // TODO: find the best device.
  physicalDevice = physicalDevices.front();
  if (failed(getBestComputeQueue()))
    return failure();

  const float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo deviceQueueCreateInfo = {};
  deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  deviceQueueCreateInfo.pNext = nullptr;
  deviceQueueCreateInfo.flags = 0;
  deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
  deviceQueueCreateInfo.queueCount = 1;
  deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

  // Structure specifying parameters of a newly created device.
  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.pNext = nullptr;
  deviceCreateInfo.flags = 0;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
  deviceCreateInfo.enabledLayerCount = 0;
  deviceCreateInfo.ppEnabledLayerNames = nullptr;
  deviceCreateInfo.enabledExtensionCount = 0;
  deviceCreateInfo.ppEnabledExtensionNames = nullptr;
  deviceCreateInfo.pEnabledFeatures = nullptr;

  RETURN_ON_VULKAN_ERROR(
      vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device),
      "vkCreateDevice");

  VkPhysicalDeviceMemoryProperties properties = {};
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);

  // Try to find memory type with following properties:
  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT bit specifies that memory allocated
  // with this type can be mapped for host access using vkMapMemory;
  // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT bit specifies that the host cache
  // management commands vkFlushMappedMemoryRanges and
  // vkInvalidateMappedMemoryRanges are not needed to flush host writes to the
  // device or make device writes visible to the host, respectively.
  for (uint32_t i = 0, e = properties.memoryTypeCount; i < e; ++i) {
    if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT &
         properties.memoryTypes[i].propertyFlags) &&
        (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT &
         properties.memoryTypes[i].propertyFlags) &&
        (memorySize <=
         properties.memoryHeaps[properties.memoryTypes[i].heapIndex].size)) {
      hostMemoryTypeIndex = i;
      break;
    }
  }

  // Find memory type memory type with VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT to be
  // used on the device. This will allow better performance access for GPU with
  // on device memory.
  for (uint32_t i = 0, e = properties.memoryTypeCount; i < e; ++i) {
    if ((VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT &
         properties.memoryTypes[i].propertyFlags) &&
        (memorySize <=
         properties.memoryHeaps[properties.memoryTypes[i].heapIndex].size)) {
      deviceMemoryTypeIndex = i;
      break;
    }
  }

  RETURN_ON_VULKAN_ERROR((hostMemoryTypeIndex == VK_MAX_MEMORY_TYPES ||
                          deviceMemoryTypeIndex == VK_MAX_MEMORY_TYPES)
                             ? VK_INCOMPLETE
                             : VK_SUCCESS,
                         "invalid memoryTypeIndex");
  return success();
}

LogicalResult VulkanRuntime::getBestComputeQueue() {
  uint32_t queueFamilyPropertiesCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      physicalDevice, &queueFamilyPropertiesCount, nullptr);

  std::vector<VkQueueFamilyProperties> familyProperties(
      queueFamilyPropertiesCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physicalDevice, &queueFamilyPropertiesCount, familyProperties.data());

  // VK_QUEUE_COMPUTE_BIT specifies that queues in this queue family support
  // compute operations. Try to find a compute-only queue first if possible.
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; ++i) {
    auto flags = familyProperties[i].queueFlags;
    if ((flags & VK_QUEUE_COMPUTE_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT)) {
      queueFamilyIndex = i;
      queueFamilyProperties = familyProperties[i];
      return success();
    }
  }

  // Otherwise use a queue that can also support graphics.
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; ++i) {
    auto flags = familyProperties[i].queueFlags;
    if ((flags & VK_QUEUE_COMPUTE_BIT)) {
      queueFamilyIndex = i;
      queueFamilyProperties = familyProperties[i];
      return success();
    }
  }

  std::cerr << "cannot find valid queue";
  return failure();
}

LogicalResult VulkanRuntime::createMemoryBuffers() {
  // For each descriptor set.
  for (const auto &resourceDataMapPair : resourceData) {
    std::vector<VulkanDeviceMemoryBuffer> deviceMemoryBuffers;
    const auto descriptorSetIndex = resourceDataMapPair.first;
    const auto &resourceDataMap = resourceDataMapPair.second;

    // For each descriptor binding.
    for (const auto &resourceDataBindingPair : resourceDataMap) {
      // Create device memory buffer.
      VulkanDeviceMemoryBuffer memoryBuffer;
      memoryBuffer.bindingIndex = resourceDataBindingPair.first;
      VkDescriptorType descriptorType = {};
      VkBufferUsageFlagBits bufferUsage = {};

      // Check that descriptor set has storage class map.
      const auto resourceStorageClassMapIt =
          resourceStorageClassData.find(descriptorSetIndex);
      if (resourceStorageClassMapIt == resourceStorageClassData.end()) {
        std::cerr
            << "cannot find storage class for resource in descriptor set: "
            << descriptorSetIndex;
        return failure();
      }

      // Check that specific descriptor binding has storage class.
      const auto &resourceStorageClassMap = resourceStorageClassMapIt->second;
      const auto resourceStorageClassIt =
          resourceStorageClassMap.find(resourceDataBindingPair.first);
      if (resourceStorageClassIt == resourceStorageClassMap.end()) {
        std::cerr
            << "cannot find storage class for resource with descriptor index: "
            << resourceDataBindingPair.first;
        return failure();
      }

      const auto resourceStorageClassBinding = resourceStorageClassIt->second;
      if (failed(mapStorageClassToDescriptorType(resourceStorageClassBinding,
                                                 descriptorType)) ||
          failed(mapStorageClassToBufferUsageFlag(resourceStorageClassBinding,
                                                  bufferUsage))) {
        std::cerr << "storage class for resource with descriptor binding: "
                  << resourceDataBindingPair.first
                  << " in the descriptor set: " << descriptorSetIndex
                  << " is not supported ";
        return failure();
      }

      // Set descriptor type for the specific device memory buffer.
      memoryBuffer.descriptorType = descriptorType;
      const auto bufferSize = resourceDataBindingPair.second.size;
      memoryBuffer.bufferSize = bufferSize;
      // Specify memory allocation info.
      VkMemoryAllocateInfo memoryAllocateInfo = {};
      memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      memoryAllocateInfo.pNext = nullptr;
      memoryAllocateInfo.allocationSize = bufferSize;
      memoryAllocateInfo.memoryTypeIndex = hostMemoryTypeIndex;

      // Allocate device memory.
      RETURN_ON_VULKAN_ERROR(vkAllocateMemory(device, &memoryAllocateInfo,
                                              nullptr,
                                              &memoryBuffer.hostMemory),
                             "vkAllocateMemory");
      memoryAllocateInfo.memoryTypeIndex = deviceMemoryTypeIndex;
      RETURN_ON_VULKAN_ERROR(vkAllocateMemory(device, &memoryAllocateInfo,
                                              nullptr,
                                              &memoryBuffer.deviceMemory),
                             "vkAllocateMemory");
      void *payload;
      RETURN_ON_VULKAN_ERROR(vkMapMemory(device, memoryBuffer.hostMemory, 0,
                                         bufferSize, 0,
                                         reinterpret_cast<void **>(&payload)),
                             "vkMapMemory");

      // Copy host memory into the mapped area.
      std::memcpy(payload, resourceDataBindingPair.second.ptr, bufferSize);
      vkUnmapMemory(device, memoryBuffer.hostMemory);

      VkBufferCreateInfo bufferCreateInfo = {};
      bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferCreateInfo.pNext = nullptr;
      bufferCreateInfo.flags = 0;
      bufferCreateInfo.size = bufferSize;
      bufferCreateInfo.usage = bufferUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      bufferCreateInfo.queueFamilyIndexCount = 1;
      bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
      RETURN_ON_VULKAN_ERROR(vkCreateBuffer(device, &bufferCreateInfo, nullptr,
                                            &memoryBuffer.hostBuffer),
                             "vkCreateBuffer");
      RETURN_ON_VULKAN_ERROR(vkCreateBuffer(device, &bufferCreateInfo, nullptr,
                                            &memoryBuffer.deviceBuffer),
                             "vkCreateBuffer");

      // Bind buffer and device memory.
      RETURN_ON_VULKAN_ERROR(vkBindBufferMemory(device, memoryBuffer.hostBuffer,
                                                memoryBuffer.hostMemory, 0),
                             "vkBindBufferMemory");
      RETURN_ON_VULKAN_ERROR(vkBindBufferMemory(device,
                                                memoryBuffer.deviceBuffer,
                                                memoryBuffer.deviceMemory, 0),
                             "vkBindBufferMemory");

      // Update buffer info.
      memoryBuffer.bufferInfo.buffer = memoryBuffer.deviceBuffer;
      memoryBuffer.bufferInfo.offset = 0;
      memoryBuffer.bufferInfo.range = VK_WHOLE_SIZE;
      deviceMemoryBuffers.push_back(memoryBuffer);
    }

    // Associate device memory buffers with a descriptor set.
    deviceMemoryBufferMap[descriptorSetIndex] = deviceMemoryBuffers;
  }
  return success();
}

LogicalResult VulkanRuntime::copyResource(bool deviceToHost) {
  VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      nullptr,
      commandPool,
      VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      1,
  };
  VkCommandBuffer commandBuffer;
  RETURN_ON_VULKAN_ERROR(vkAllocateCommandBuffers(device,
                                                  &commandBufferAllocateInfo,
                                                  &commandBuffer),
                         "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo commandBufferBeginInfo = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      nullptr,
      0,
      nullptr,
  };
  RETURN_ON_VULKAN_ERROR(
      vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo),
      "vkBeginCommandBuffer");

  for (const auto &deviceMemoryBufferMapPair : deviceMemoryBufferMap) {
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
    const auto &deviceMemoryBuffers = deviceMemoryBufferMapPair.second;
    for (const auto &memBuffer : deviceMemoryBuffers) {
      VkBufferCopy copy = {0, 0, memBuffer.bufferSize};
      if (deviceToHost)
        vkCmdCopyBuffer(commandBuffer, memBuffer.deviceBuffer,
                        memBuffer.hostBuffer, 1, &copy);
      else
        vkCmdCopyBuffer(commandBuffer, memBuffer.hostBuffer,
                        memBuffer.deviceBuffer, 1, &copy);
    }
  }

  RETURN_ON_VULKAN_ERROR(vkEndCommandBuffer(commandBuffer),
                         "vkEndCommandBuffer");
  VkSubmitInfo submitInfo = {
      VK_STRUCTURE_TYPE_SUBMIT_INFO,
      nullptr,
      0,
      nullptr,
      nullptr,
      1,
      &commandBuffer,
      0,
      nullptr,
  };
  submitInfo.pCommandBuffers = &commandBuffer;
  RETURN_ON_VULKAN_ERROR(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE),
                         "vkQueueSubmit");
  RETURN_ON_VULKAN_ERROR(vkQueueWaitIdle(queue), "vkQueueWaitIdle");

  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  return success();
}

LogicalResult VulkanRuntime::createShaderModule() {
  VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
  shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCreateInfo.pNext = nullptr;
  shaderModuleCreateInfo.flags = 0;
  // Set size in bytes.
  shaderModuleCreateInfo.codeSize = binarySize;
  // Set pointer to the binary shader.
  shaderModuleCreateInfo.pCode = reinterpret_cast<uint32_t *>(binary);
  RETURN_ON_VULKAN_ERROR(vkCreateShaderModule(device, &shaderModuleCreateInfo,
                                              nullptr, &shaderModule),
                         "vkCreateShaderModule");
  return success();
}

void VulkanRuntime::initDescriptorSetLayoutBindingMap() {
  for (const auto &deviceMemoryBufferMapPair : deviceMemoryBufferMap) {
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
    const auto &deviceMemoryBuffers = deviceMemoryBufferMapPair.second;
    const auto descriptorSetIndex = deviceMemoryBufferMapPair.first;

    // Create a layout binding for each descriptor.
    for (const auto &memBuffer : deviceMemoryBuffers) {
      VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
      descriptorSetLayoutBinding.binding = memBuffer.bindingIndex;
      descriptorSetLayoutBinding.descriptorType = memBuffer.descriptorType;
      descriptorSetLayoutBinding.descriptorCount = 1;
      descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      descriptorSetLayoutBinding.pImmutableSamplers = nullptr;
      descriptorSetLayoutBindings.push_back(descriptorSetLayoutBinding);
    }
    descriptorSetLayoutBindingMap[descriptorSetIndex] =
        descriptorSetLayoutBindings;
  }
}

LogicalResult VulkanRuntime::createDescriptorSetLayout() {
  for (const auto &deviceMemoryBufferMapPair : deviceMemoryBufferMap) {
    const auto descriptorSetIndex = deviceMemoryBufferMapPair.first;
    const auto &deviceMemoryBuffers = deviceMemoryBufferMapPair.second;
    // Each descriptor in a descriptor set must be the same type.
    VkDescriptorType descriptorType =
        deviceMemoryBuffers.front().descriptorType;
    const uint32_t descriptorSize = deviceMemoryBuffers.size();
    const auto descriptorSetLayoutBindingIt =
        descriptorSetLayoutBindingMap.find(descriptorSetIndex);

    if (descriptorSetLayoutBindingIt == descriptorSetLayoutBindingMap.end()) {
      std::cerr << "cannot find layout bindings for the set with number: "
                << descriptorSetIndex;
      return failure();
    }

    const auto &descriptorSetLayoutBindings =
        descriptorSetLayoutBindingIt->second;
    // Create descriptor set layout.
    VkDescriptorSetLayout descriptorSetLayout = {};
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};

    descriptorSetLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pNext = nullptr;
    descriptorSetLayoutCreateInfo.flags = 0;
    // Amount of descriptor bindings in a layout set.
    descriptorSetLayoutCreateInfo.bindingCount =
        descriptorSetLayoutBindings.size();
    descriptorSetLayoutCreateInfo.pBindings =
        descriptorSetLayoutBindings.data();
    RETURN_ON_VULKAN_ERROR(
        vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo,
                                    nullptr, &descriptorSetLayout),
        "vkCreateDescriptorSetLayout");

    descriptorSetLayouts.push_back(descriptorSetLayout);
    descriptorSetInfoPool.push_back(
        {descriptorSetIndex, descriptorSize, descriptorType});
  }
  return success();
}

LogicalResult VulkanRuntime::createPipelineLayout() {
  // Associate descriptor sets with a pipeline layout.
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.pNext = nullptr;
  pipelineLayoutCreateInfo.flags = 0;
  pipelineLayoutCreateInfo.setLayoutCount = descriptorSetLayouts.size();
  pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();
  pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
  pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;
  RETURN_ON_VULKAN_ERROR(vkCreatePipelineLayout(device,
                                                &pipelineLayoutCreateInfo,
                                                nullptr, &pipelineLayout),
                         "vkCreatePipelineLayout");
  return success();
}

LogicalResult VulkanRuntime::createComputePipeline() {
  VkPipelineShaderStageCreateInfo stageInfo = {};
  stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageInfo.pNext = nullptr;
  stageInfo.flags = 0;
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = shaderModule;
  // Set entry point.
  stageInfo.pName = entryPoint;
  stageInfo.pSpecializationInfo = nullptr;

  VkComputePipelineCreateInfo computePipelineCreateInfo = {};
  computePipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.pNext = nullptr;
  computePipelineCreateInfo.flags = 0;
  computePipelineCreateInfo.stage = stageInfo;
  computePipelineCreateInfo.layout = pipelineLayout;
  computePipelineCreateInfo.basePipelineHandle = nullptr;
  computePipelineCreateInfo.basePipelineIndex = 0;
  RETURN_ON_VULKAN_ERROR(vkCreateComputePipelines(device, nullptr, 1,
                                                  &computePipelineCreateInfo,
                                                  nullptr, &pipeline),
                         "vkCreateComputePipelines");
  return success();
}

LogicalResult VulkanRuntime::createDescriptorPool() {
  std::vector<VkDescriptorPoolSize> descriptorPoolSizes;
  for (const auto &descriptorSetInfo : descriptorSetInfoPool) {
    // For each descriptor set populate descriptor pool size.
    VkDescriptorPoolSize descriptorPoolSize = {};
    descriptorPoolSize.type = descriptorSetInfo.descriptorType;
    descriptorPoolSize.descriptorCount = descriptorSetInfo.descriptorSize;
    descriptorPoolSizes.push_back(descriptorPoolSize);
  }

  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
  descriptorPoolCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.pNext = nullptr;
  descriptorPoolCreateInfo.flags = 0;
  descriptorPoolCreateInfo.maxSets = descriptorPoolSizes.size();
  descriptorPoolCreateInfo.poolSizeCount = descriptorPoolSizes.size();
  descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes.data();
  RETURN_ON_VULKAN_ERROR(vkCreateDescriptorPool(device,
                                                &descriptorPoolCreateInfo,
                                                nullptr, &descriptorPool),
                         "vkCreateDescriptorPool");
  return success();
}

LogicalResult VulkanRuntime::allocateDescriptorSets() {
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  // Size of descriptor sets and descriptor layout sets is the same.
  descriptorSets.resize(descriptorSetLayouts.size());
  descriptorSetAllocateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.pNext = nullptr;
  descriptorSetAllocateInfo.descriptorPool = descriptorPool;
  descriptorSetAllocateInfo.descriptorSetCount = descriptorSetLayouts.size();
  descriptorSetAllocateInfo.pSetLayouts = descriptorSetLayouts.data();
  RETURN_ON_VULKAN_ERROR(vkAllocateDescriptorSets(device,
                                                  &descriptorSetAllocateInfo,
                                                  descriptorSets.data()),
                         "vkAllocateDescriptorSets");
  return success();
}

LogicalResult VulkanRuntime::setWriteDescriptors() {
  if (descriptorSets.size() != descriptorSetInfoPool.size()) {
    std::cerr << "Each descriptor set must have descriptor set information";
    return failure();
  }
  // For each descriptor set.
  auto descriptorSetIt = descriptorSets.begin();
  // Each descriptor set is associated with descriptor set info.
  for (const auto &descriptorSetInfo : descriptorSetInfoPool) {
    // For each device memory buffer in the descriptor set.
    const auto &deviceMemoryBuffers =
        deviceMemoryBufferMap[descriptorSetInfo.descriptorSet];
    for (const auto &memoryBuffer : deviceMemoryBuffers) {
      // Structure describing descriptor sets to write to.
      VkWriteDescriptorSet wSet = {};
      wSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      wSet.pNext = nullptr;
      // Descriptor set.
      wSet.dstSet = *descriptorSetIt;
      wSet.dstBinding = memoryBuffer.bindingIndex;
      wSet.dstArrayElement = 0;
      wSet.descriptorCount = 1;
      wSet.descriptorType = memoryBuffer.descriptorType;
      wSet.pImageInfo = nullptr;
      wSet.pBufferInfo = &memoryBuffer.bufferInfo;
      wSet.pTexelBufferView = nullptr;
      vkUpdateDescriptorSets(device, 1, &wSet, 0, nullptr);
    }
    // Increment descriptor set iterator.
    ++descriptorSetIt;
  }
  return success();
}

LogicalResult VulkanRuntime::createCommandPool() {
  VkCommandPoolCreateInfo commandPoolCreateInfo = {};
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.pNext = nullptr;
  commandPoolCreateInfo.flags = 0;
  commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
  RETURN_ON_VULKAN_ERROR(vkCreateCommandPool(device, &commandPoolCreateInfo,
                                             /*pAllocator=*/nullptr,
                                             &commandPool),
                         "vkCreateCommandPool");
  return success();
}

LogicalResult VulkanRuntime::createQueryPool() {
  // Return directly if timestamp query is not supported.
  if (queueFamilyProperties.timestampValidBits == 0)
    return success();

  // Get timestamp period for this physical device.
  VkPhysicalDeviceProperties deviceProperties = {};
  vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
  timestampPeriod = deviceProperties.limits.timestampPeriod;

  // Create query pool.
  VkQueryPoolCreateInfo queryPoolCreateInfo = {};
  queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  queryPoolCreateInfo.pNext = nullptr;
  queryPoolCreateInfo.flags = 0;
  queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  queryPoolCreateInfo.queryCount = 2;
  queryPoolCreateInfo.pipelineStatistics = 0;
  RETURN_ON_VULKAN_ERROR(vkCreateQueryPool(device, &queryPoolCreateInfo,
                                           /*pAllocator=*/nullptr, &queryPool),
                         "vkCreateQueryPool");

  return success();
}

LogicalResult VulkanRuntime::createComputeCommandBuffer() {
  VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
  commandBufferAllocateInfo.sType =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  commandBufferAllocateInfo.pNext = nullptr;
  commandBufferAllocateInfo.commandPool = commandPool;
  commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferAllocateInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  RETURN_ON_VULKAN_ERROR(vkAllocateCommandBuffers(device,
                                                  &commandBufferAllocateInfo,
                                                  &commandBuffer),
                         "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo commandBufferBeginInfo = {};
  commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  commandBufferBeginInfo.pNext = nullptr;
  commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  commandBufferBeginInfo.pInheritanceInfo = nullptr;

  // Commands begin.
  RETURN_ON_VULKAN_ERROR(
      vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo),
      "vkBeginCommandBuffer");

  if (queryPool != VK_NULL_HANDLE)
    vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout, 0, descriptorSets.size(),
                          descriptorSets.data(), 0, nullptr);
  // Get a timestamp before invoking the compute shader.
  if (queryPool != VK_NULL_HANDLE)
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        queryPool, 0);
  vkCmdDispatch(commandBuffer, numWorkGroups.x, numWorkGroups.y,
                numWorkGroups.z);
  // Get another timestamp after invoking the compute shader.
  if (queryPool != VK_NULL_HANDLE)
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        queryPool, 1);

  // Commands end.
  RETURN_ON_VULKAN_ERROR(vkEndCommandBuffer(commandBuffer),
                         "vkEndCommandBuffer");

  commandBuffers.push_back(commandBuffer);
  return success();
}

LogicalResult VulkanRuntime::submitCommandBuffersToQueue() {
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.pNext = nullptr;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.pWaitSemaphores = nullptr;
  submitInfo.pWaitDstStageMask = nullptr;
  submitInfo.commandBufferCount = commandBuffers.size();
  submitInfo.pCommandBuffers = commandBuffers.data();
  submitInfo.signalSemaphoreCount = 0;
  submitInfo.pSignalSemaphores = nullptr;
  RETURN_ON_VULKAN_ERROR(vkQueueSubmit(queue, 1, &submitInfo, nullptr),
                         "vkQueueSubmit");
  return success();
}

LogicalResult VulkanRuntime::updateHostMemoryBuffers() {
  // First copy back the data to the staging buffer.
  (void)copyResource(/*deviceToHost=*/true);

  // For each descriptor set.
  for (auto &resourceDataMapPair : resourceData) {
    auto &resourceDataMap = resourceDataMapPair.second;
    auto &deviceMemoryBuffers =
        deviceMemoryBufferMap[resourceDataMapPair.first];
    // For each device memory buffer in the set.
    for (auto &deviceMemoryBuffer : deviceMemoryBuffers) {
      if (resourceDataMap.count(deviceMemoryBuffer.bindingIndex)) {
        void *payload;
        auto &hostMemoryBuffer =
            resourceDataMap[deviceMemoryBuffer.bindingIndex];
        RETURN_ON_VULKAN_ERROR(vkMapMemory(device,
                                           deviceMemoryBuffer.hostMemory, 0,
                                           hostMemoryBuffer.size, 0,
                                           reinterpret_cast<void **>(&payload)),
                               "vkMapMemory");
        std::memcpy(hostMemoryBuffer.ptr, payload, hostMemoryBuffer.size);
        vkUnmapMemory(device, deviceMemoryBuffer.hostMemory);
      }
    }
  }
  return success();
}
