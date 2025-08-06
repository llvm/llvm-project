//===- LevelZeroRuntimeWrappers.cpp - MLIR Level Zero (L0) wrapper library-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements wrappers around the Level Zero (L0) runtime library with C linkage
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"

#include "level_zero/ze_api.h"
#include <cassert>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>

namespace {
template <typename F>
auto catchAll(F &&func) {
  try {
    return func();
  } catch (const std::exception &e) {
    std::cerr << "An exception was thrown: " << e.what() << std::endl;
    std::abort();
  } catch (...) {
    std::cerr << "An unknown exception was thrown." << std::endl;
    std::abort();
  }
}

#define L0_SAFE_CALL(call)                                                     \
  {                                                                            \
    ze_result_t status = (call);                                               \
    if (status != ZE_RESULT_SUCCESS) {                                         \
      const char *errorString;                                                 \
      zeDriverGetLastErrorDescription(NULL, &errorString);                     \
      std::cerr << "L0 error " << status << ": " << errorString << std::endl;  \
      std::abort();                                                            \
    }                                                                          \
  }
} // namespace

//===----------------------------------------------------------------------===//
// L0 RT context & device setters
//===----------------------------------------------------------------------===//

// Returns the L0 driver handle for the given index. Default index is 0
// (i.e., returns the first driver handle of the available drivers).

static ze_driver_handle_t getDriver(uint32_t idx = 0) {
  ze_init_driver_type_desc_t driver_type = {};
  driver_type.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
  driver_type.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
  driver_type.pNext = nullptr;
  uint32_t driverCount{0};
  thread_local static std::vector<ze_driver_handle_t> drivers;
  thread_local static bool isDriverInitialised{false};
  if (isDriverInitialised && idx < drivers.size())
    return drivers[idx];
  L0_SAFE_CALL(zeInitDrivers(&driverCount, nullptr, &driver_type));
  if (!driverCount)
    throw std::runtime_error("No L0 drivers found.");
  drivers.resize(driverCount);
  L0_SAFE_CALL(zeInitDrivers(&driverCount, drivers.data(), &driver_type));
  if (idx >= driverCount)
    throw std::runtime_error((llvm::Twine("Requested driver idx out-of-bound, "
                                          "number of availabe drivers: ") +
                              std::to_string(driverCount))
                                 .str());
  isDriverInitialised = true;
  return drivers[idx];
}

static ze_device_handle_t getDevice(const uint32_t driverIdx = 0,
                                    const int32_t devIdx = 0) {
  thread_local static ze_device_handle_t l0Device;
  thread_local int32_t currDevIdx{-1};
  thread_local uint32_t currDriverIdx{0};
  if (currDriverIdx == driverIdx && currDevIdx == devIdx)
    return l0Device;
  auto driver = getDriver(driverIdx);
  uint32_t deviceCount{0};
  L0_SAFE_CALL(zeDeviceGet(driver, &deviceCount, nullptr));
  if (!deviceCount)
    throw std::runtime_error("getDevice failed: did not find L0 device.");
  if (static_cast<int>(deviceCount) < devIdx + 1)
    throw std::runtime_error("getDevice failed: devIdx out-of-bounds.");
  std::vector<ze_device_handle_t> devices(deviceCount);
  L0_SAFE_CALL(zeDeviceGet(driver, &deviceCount, devices.data()));
  l0Device = devices[devIdx];
  currDriverIdx = driverIdx;
  currDevIdx = devIdx;
  return l0Device;
}

// Returns the default L0 context of the defult driver.
static ze_context_handle_t getContext(ze_driver_handle_t driver) {
  thread_local static ze_context_handle_t context;
  thread_local static bool isContextInitialised{false};
  if (isContextInitialised)
    return context;
  ze_context_desc_t ctxtDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  L0_SAFE_CALL(zeContextCreate(driver, &ctxtDesc, &context));
  isContextInitialised = true;
  return context;
}

//===----------------------------------------------------------------------===//
// L0 RT helper structs
//===----------------------------------------------------------------------===//

struct ZeContextDeleter {
  void operator()(ze_context_handle_t ctx) const {
    if (ctx)
      L0_SAFE_CALL(zeContextDestroy(ctx));
  }
};

struct ZeCommandListDeleter {
  void operator()(ze_command_list_handle_t cmdList) const {
    if (cmdList)
      L0_SAFE_CALL(zeCommandListDestroy(cmdList));
  }
};
using UniqueZeContext =
    std::unique_ptr<std::remove_pointer<ze_context_handle_t>::type,
                    ZeContextDeleter>;
using UniqueZeCommandList =
    std::unique_ptr<std::remove_pointer<ze_command_list_handle_t>::type,
                    ZeCommandListDeleter>;
struct L0RTContextWrapper {
  ze_driver_handle_t driver{nullptr};
  ze_device_handle_t device{nullptr};
  UniqueZeContext context;
  // Usually, one immediate command list with ordinal 0 suffices for
  // both copy and compute ops, but leaves HW underutilized.
  UniqueZeCommandList immCmdListCompute;
  // Copy engines can be used for both memcpy and memset, but
  // they have limitations for memset pattern size (e.g., 1 byte).
  UniqueZeCommandList immCmdListCopy;
  uint32_t copyEngineMaxMemoryFillPatternSize{-1u};

  L0RTContextWrapper() = default;
  L0RTContextWrapper(const uint32_t driverIdx = 0, const int32_t devIdx = 0)
      : driver(getDriver(driverIdx)), device(getDevice(devIdx)) {
    // Create context
    ze_context_handle_t ctx = getContext(driver);
    context.reset(ctx);

    // Determine ordinals
    uint32_t computeEngineOrdinal = -1u, copyEngineOrdinal = -1u;
    ze_device_properties_t deviceProperties{};
    L0_SAFE_CALL(zeDeviceGetProperties(device, &deviceProperties));
    uint32_t queueGroupCount = 0;
    L0_SAFE_CALL(zeDeviceGetCommandQueueGroupProperties(
        device, &queueGroupCount, nullptr));
    std::vector<ze_command_queue_group_properties_t> queueGroupProperties(
        queueGroupCount);
    L0_SAFE_CALL(zeDeviceGetCommandQueueGroupProperties(
        device, &queueGroupCount, queueGroupProperties.data()));

    for (uint32_t queueGroupIdx = 0; queueGroupIdx < queueGroupCount;
         ++queueGroupIdx) {
      const auto &group = queueGroupProperties[queueGroupIdx];
      if (group.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
        computeEngineOrdinal = queueGroupIdx;
      else if (group.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) {
        copyEngineOrdinal = queueGroupIdx;
        copyEngineMaxMemoryFillPatternSize = group.maxMemoryFillPatternSize;
      }
      if (copyEngineOrdinal != -1u && computeEngineOrdinal != -1u)
        break;
    }

    // Fallback to the default queue if no dedicated copy queue is available.
    if (copyEngineOrdinal == -1u)
      copyEngineOrdinal = computeEngineOrdinal;

    assert(copyEngineOrdinal != -1u && computeEngineOrdinal != -1u &&
           "Expected two engines to be available.");

    // Create copy command list
    ze_command_queue_desc_t cmdQueueDesc{
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        nullptr,
        copyEngineOrdinal, // ordinal
        0,                 // index (assume one physical engine in the group)
        0,                 // flags
        ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

    ze_command_list_handle_t rawCmdListCopy = nullptr;
    L0_SAFE_CALL(zeCommandListCreateImmediate(context.get(), device,
                                              &cmdQueueDesc, &rawCmdListCopy));
    immCmdListCopy.reset(rawCmdListCopy);

    // Create compute command list
    cmdQueueDesc.ordinal = computeEngineOrdinal;
    ze_command_list_handle_t rawCmdListCompute = nullptr;
    L0_SAFE_CALL(zeCommandListCreateImmediate(
        context.get(), device, &cmdQueueDesc, &rawCmdListCompute));
    immCmdListCompute.reset(rawCmdListCompute);
  }
  L0RTContextWrapper(const L0RTContextWrapper &) = delete;
  L0RTContextWrapper &operator=(const L0RTContextWrapper &) = delete;
  // Allow move
  L0RTContextWrapper(L0RTContextWrapper &&) noexcept = default;
  L0RTContextWrapper &operator=(L0RTContextWrapper &&) noexcept = default;
  ~L0RTContextWrapper() = default;
};

struct ZeEventDeleter {
  void operator()(ze_event_handle_t event) const {
    if (event)
      L0_SAFE_CALL(zeEventDestroy(event));
  }
};

struct ZeEventPoolDeleter {
  void operator()(ze_event_pool_handle_t pool) const {
    if (pool)
      L0_SAFE_CALL(zeEventPoolDestroy(pool));
  }
};

using UniqueZeEvent =
    std::unique_ptr<std::remove_pointer<ze_event_handle_t>::type,
                    ZeEventDeleter>;
using UniqueZeEventPool =
    std::unique_ptr<std::remove_pointer<ze_event_pool_handle_t>::type,
                    ZeEventPoolDeleter>;

// L0 only supports pre-determined sizes of event pools,
// implement a runtime data structure to avoid running out of events.

struct DynamicEventPool {
  constexpr static size_t numEventsPerPool{128};

  std::vector<UniqueZeEventPool> eventPools;
  std::vector<UniqueZeEvent> availableEvents;
  std::unordered_map<ze_event_handle_t, UniqueZeEvent> takenEvents;

  // Limit the number of events to avoid running out of memory.
  // The limit is set to 32K events, which should be sufficient for most use
  // cases.
  size_t maxEventsCount{32768}; // 32K events
  size_t currentEventsLimit{0};
  size_t currentEventsCnt{0};
  L0RTContextWrapper *rtCtx;

  DynamicEventPool(L0RTContextWrapper *rtCtx) : rtCtx(rtCtx) {
    createNewPool(numEventsPerPool);
  }

  DynamicEventPool(const DynamicEventPool &) = delete;
  DynamicEventPool &operator=(const DynamicEventPool &) = delete;

  // Allow move
  DynamicEventPool(DynamicEventPool &&) noexcept = default;
  DynamicEventPool &operator=(DynamicEventPool &&) noexcept = default;

  ~DynamicEventPool() {
    assert(takenEvents.empty() && "Some events were not released");
  }

  void createNewPool(size_t numEvents) {
    ze_event_pool_desc_t eventPoolDesc = {};
    eventPoolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    eventPoolDesc.count = numEvents;

    ze_event_pool_handle_t rawPool = nullptr;
    L0_SAFE_CALL(zeEventPoolCreate(rtCtx->context.get(), &eventPoolDesc, 1,
                                   &rtCtx->device, &rawPool));

    eventPools.emplace_back(UniqueZeEventPool(rawPool));
    currentEventsLimit += numEvents;
  }

  ze_event_handle_t takeEvent() {
    ze_event_handle_t rawEvent = nullptr;

    if (!availableEvents.empty()) {
      // Reuse one
      auto uniqueEvent = std::move(availableEvents.back());
      availableEvents.pop_back();
      rawEvent = uniqueEvent.get();
      takenEvents[rawEvent] = std::move(uniqueEvent);
    } else {
      if (currentEventsCnt >= maxEventsCount) {
        throw std::runtime_error("DynamicEventPool: reached max events limit");
      }
      if (currentEventsCnt == currentEventsLimit)
        createNewPool(numEventsPerPool);

      ze_event_desc_t eventDesc = {
          ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr,
          static_cast<uint32_t>(currentEventsCnt % numEventsPerPool),
          ZE_EVENT_SCOPE_FLAG_DEVICE, ZE_EVENT_SCOPE_FLAG_HOST};

      ze_event_handle_t newEvent = nullptr;
      L0_SAFE_CALL(
          zeEventCreate(eventPools.back().get(), &eventDesc, &newEvent));

      takenEvents[newEvent] = UniqueZeEvent(newEvent);
      rawEvent = newEvent;
      currentEventsCnt++;
    }

    return rawEvent;
  }

  void releaseEvent(ze_event_handle_t event) {
    auto it = takenEvents.find(event);
    assert(it != takenEvents.end() &&
           "Attempting to release unknown or already released event");

    L0_SAFE_CALL(zeEventHostReset(event));
    availableEvents.emplace_back(std::move(it->second));
    takenEvents.erase(it);
  }
};

L0RTContextWrapper &getRtContext() {
  thread_local static L0RTContextWrapper rtContext(0);
  return rtContext;
}

DynamicEventPool &getDynamicEventPool() {
  thread_local static DynamicEventPool dynEventPool{&getRtContext()};
  return dynEventPool;
}

struct StreamWrapper {
  // avoid event pointer invalidations
  std::deque<ze_event_handle_t> implicitEventStack;
  DynamicEventPool &dynEventPool;

  StreamWrapper(DynamicEventPool &dynEventPool) : dynEventPool(dynEventPool) {}
  ~StreamWrapper() { sync(); }

  ze_event_handle_t *getLastImplicitEventPtr() {
    // Assume current implicit events will not be used after `sync`.
    return implicitEventStack.size() ? &implicitEventStack.back() : nullptr;
  }

  void sync(ze_event_handle_t explicitEvent = nullptr) {
    ze_event_handle_t syncEvent{nullptr};
    if (!explicitEvent) {
      ze_event_handle_t *lastImplicitEventPtr = getLastImplicitEventPtr();
      syncEvent = lastImplicitEventPtr ? *lastImplicitEventPtr : nullptr;
    } else {
      syncEvent = explicitEvent;
    }
    if (syncEvent)
      L0_SAFE_CALL(zeEventHostSynchronize(
          syncEvent, std::numeric_limits<uint64_t>::max()));
    // All of the "implicit" events were signaled and are of no use, release
    // them. "explicit" event must be "released" via mgpuEventDestroy
    for (auto event : implicitEventStack)
      dynEventPool.releaseEvent(event);
    implicitEventStack.clear();
  }

  template <typename Func>
  void enqueueOp(Func &&op) {
    ze_event_handle_t newImplicitEvent = dynEventPool.takeEvent();
    ze_event_handle_t *lastImplicitEventPtr = getLastImplicitEventPtr();
    const uint32_t numWaitEvents = lastImplicitEventPtr ? 1 : 0;
    std::forward<Func>(op)(newImplicitEvent, numWaitEvents,
                           lastImplicitEventPtr);
    implicitEventStack.push_back(newImplicitEvent);
  }
};

static ze_module_handle_t loadModule(const void *data, size_t dataSize) {
  assert(data);
  ze_module_handle_t zeModule;
  ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                           nullptr,
                           ZE_MODULE_FORMAT_IL_SPIRV,
                           dataSize,
                           (const uint8_t *)data,
                           nullptr,
                           nullptr};
  ze_module_build_log_handle_t buildLogHandle;
  ze_result_t result =
      zeModuleCreate(getRtContext().context.get(), getRtContext().device, &desc,
                     &zeModule, &buildLogHandle);
  if (result != ZE_RESULT_SUCCESS) {
    std::cerr << "Error creating module, error code: " << result << std::endl;
    size_t logSize = 0;
    L0_SAFE_CALL(zeModuleBuildLogGetString(buildLogHandle, &logSize, nullptr));
    std::string buildLog(" ", logSize);
    L0_SAFE_CALL(
        zeModuleBuildLogGetString(buildLogHandle, &logSize, buildLog.data()));
    std::cerr << "Build log:\n" << buildLog << std::endl;
    std::abort();
  }
  return zeModule;
}

//===----------------------------------------------------------------------===//
// L0 Wrappers definition
//===----------------------------------------------------------------------===//

extern "C" StreamWrapper *mgpuStreamCreate() {
  return new StreamWrapper(getDynamicEventPool());
}

extern "C" void mgpuStreamSynchronize(StreamWrapper *stream) {
  if (stream)
    stream->sync();
}

extern "C" void mgpuStreamDestroy(StreamWrapper *stream) { delete stream; }

extern "C" void mgpuStreamWaitEvent(StreamWrapper *stream,
                                    ze_event_handle_t event) {
  assert(stream && "Invalid stream");
  assert(event && "Invalid event");
  stream->sync(event);
}

extern "C" ze_event_handle_t mgpuEventCreate() {
  return getDynamicEventPool().takeEvent();
}

extern "C" void mgpuEventDestroy(ze_event_handle_t event) {
  return getDynamicEventPool().releaseEvent(event);
}

extern "C" void mgpuEventSynchronize(ze_event_handle_t event) {
  L0_SAFE_CALL(
      zeEventHostSynchronize(event, std::numeric_limits<uint64_t>::max()));
  L0_SAFE_CALL(zeEventHostReset(event));
}

extern "C" void mgpuEventRecord(ze_event_handle_t event,
                                StreamWrapper *stream) {
  L0_SAFE_CALL(zeCommandListAppendSignalEvent(
      getRtContext().immCmdListCopy.get(), event));
  L0_SAFE_CALL(zeCommandListAppendSignalEvent(
      getRtContext().immCmdListCompute.get(), event));
}

extern "C" void *mgpuMemAlloc(uint64_t size, StreamWrapper *stream,
                              bool isShared) {
  return catchAll([&]() {
    void *memPtr = nullptr;
    constexpr size_t alignment{64};
    ze_device_mem_alloc_desc_t deviceDesc = {};
    deviceDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    if (isShared) {
      ze_host_mem_alloc_desc_t hostDesc = {};
      hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
      L0_SAFE_CALL(zeMemAllocShared(getRtContext().context.get(), &deviceDesc,
                                    &hostDesc, size, alignment,
                                    getRtContext().device, &memPtr));
    } else {
      L0_SAFE_CALL(zeMemAllocDevice(getRtContext().context.get(), &deviceDesc,
                                    size, alignment, getRtContext().device,
                                    &memPtr));
    }
    if (!memPtr)
      throw std::runtime_error("mem allocation failed!");
    return memPtr;
  });
}

extern "C" void mgpuMemFree(void *ptr, StreamWrapper *stream) {
  stream->sync();
  if (ptr)
    L0_SAFE_CALL(zeMemFree(getRtContext().context.get(), ptr));
}

extern "C" void mgpuMemcpy(void *dst, void *src, size_t sizeBytes,
                           StreamWrapper *stream) {
  stream->enqueueOp([&](ze_event_handle_t newEvent, uint32_t numWaitEvents,
                        ze_event_handle_t *waitEvents) {
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(
        getRtContext().immCmdListCopy.get(), dst, src, sizeBytes, newEvent,
        numWaitEvents, waitEvents));
  });
}

template <typename PATTERN_TYPE>
void mgpuMemset(void *dst, PATTERN_TYPE value, size_t count,
                StreamWrapper *stream) {
  L0RTContextWrapper &rtContext = getRtContext();
  auto listType =
      rtContext.copyEngineMaxMemoryFillPatternSize >= sizeof(PATTERN_TYPE)
          ? rtContext.immCmdListCopy.get()
          : rtContext.immCmdListCompute.get();
  stream->enqueueOp([&](ze_event_handle_t newEvent, uint32_t numWaitEvents,
                        ze_event_handle_t *waitEvents) {
    L0_SAFE_CALL(zeCommandListAppendMemoryFill(
        listType, dst, &value, sizeof(PATTERN_TYPE),
        count * sizeof(PATTERN_TYPE), newEvent, numWaitEvents, waitEvents));
  });
}
extern "C" void mgpuMemset32(void *dst, unsigned int value, size_t count,
                             StreamWrapper *stream) {
  mgpuMemset<unsigned int>(dst, value, count, stream);
}

extern "C" void mgpuMemset16(void *dst, unsigned short value, size_t count,
                             StreamWrapper *stream) {
  mgpuMemset<unsigned short>(dst, value, count, stream);
}

extern "C" ze_module_handle_t mgpuModuleLoad(const void *data,
                                             size_t gpuBlobSize) {
  return catchAll([&]() { return loadModule(data, gpuBlobSize); });
}

extern "C" ze_kernel_handle_t mgpuModuleGetFunction(ze_module_handle_t module,
                                                    const char *name) {
  assert(module && name);
  ze_kernel_handle_t zeKernel;
  ze_kernel_desc_t desc = {};
  desc.pKernelName = name;
  L0_SAFE_CALL(zeKernelCreate(module, &desc, &zeKernel));
  return zeKernel;
}

extern "C" void mgpuLaunchKernel(ze_kernel_handle_t kernel, size_t gridX,
                                 size_t gridY, size_t gridZ, size_t blockX,
                                 size_t blockY, size_t blockZ,
                                 size_t sharedMemBytes, StreamWrapper *stream,
                                 void **params, void ** /*extra*/,
                                 size_t paramsCount) {

  if (sharedMemBytes > 0) {
    paramsCount = paramsCount - 1; // Last param is shared memory size
    L0_SAFE_CALL(
        zeKernelSetArgumentValue(kernel, paramsCount, sharedMemBytes, nullptr));
  }
  for (size_t i = 0; i < paramsCount; ++i)
    L0_SAFE_CALL(zeKernelSetArgumentValue(kernel, static_cast<uint32_t>(i),
                                          sizeof(void *), params[i]));
  L0_SAFE_CALL(zeKernelSetGroupSize(kernel, blockX, blockY, blockZ));
  ze_group_count_t dispatch;
  dispatch.groupCountX = static_cast<uint32_t>(gridX);
  dispatch.groupCountY = static_cast<uint32_t>(gridY);
  dispatch.groupCountZ = static_cast<uint32_t>(gridZ);
  stream->enqueueOp([&](ze_event_handle_t newEvent, uint32_t numWaitEvents,
                        ze_event_handle_t *waitEvents) {
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(
        getRtContext().immCmdListCompute.get(), kernel, &dispatch, newEvent,
        numWaitEvents, waitEvents));
  });
}

extern "C" void mgpuModuleUnload(ze_module_handle_t module) {
  L0_SAFE_CALL(zeModuleDestroy(module));
}

extern "C" void mgpuSetDefaultDevice(int32_t devIdx) {
  catchAll([&]() {
    // For now, a user must ensure that streams and events complete
    // and are destroyed before switching a device.
    getRtContext() = L0RTContextWrapper(devIdx);
    getDynamicEventPool() = DynamicEventPool(&getRtContext());
  });
}
