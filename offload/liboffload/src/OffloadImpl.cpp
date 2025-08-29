//===- ol_impl.cpp - Implementation of the new LLVM/Offload API ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the definitions of the new LLVM/Offload API entry points. See
// new-api/API/README.md for more information.
//
//===----------------------------------------------------------------------===//

#include "OffloadImpl.hpp"
#include "Helpers.hpp"
#include "OffloadPrint.hpp"
#include "PluginManager.h"
#include "llvm/Support/FormatVariadic.h"
#include <OffloadAPI.h>

#include <mutex>

// TODO: Some plugins expect to be linked into libomptarget which defines these
// symbols to implement ompt callbacks. The least invasive workaround here is to
// define them in libLLVMOffload as false/null so they are never used. In future
// it would be better to allow the plugins to implement callbacks without
// pulling in details from libomptarget.
#ifdef OMPT_SUPPORT
namespace llvm::omp::target {
namespace ompt {
bool Initialized = false;
ompt_get_callback_t lookupCallbackByCode = nullptr;
ompt_function_lookup_t lookupCallbackByName = nullptr;
} // namespace ompt
} // namespace llvm::omp::target
#endif

using namespace llvm::omp::target;
using namespace llvm::omp::target::plugin;
using namespace error;

// Handle type definitions. Ideally these would be 1:1 with the plugins, but
// we add some additional data here for now to avoid churn in the plugin
// interface.
struct ol_device_impl_t {
  ol_device_impl_t(int DeviceNum, GenericDeviceTy *Device,
                   ol_platform_handle_t Platform, InfoTreeNode &&DevInfo)
      : DeviceNum(DeviceNum), Device(Device), Platform(Platform),
        Info(std::forward<InfoTreeNode>(DevInfo)) {}

  ~ol_device_impl_t() {
    assert(!OutstandingQueues.size() &&
           "Device object dropped with outstanding queues");
  }

  int DeviceNum;
  GenericDeviceTy *Device;
  ol_platform_handle_t Platform;
  InfoTreeNode Info;

  llvm::SmallVector<__tgt_async_info *> OutstandingQueues;
  std::mutex OutstandingQueuesMutex;

  /// If the device has any outstanding queues that are now complete, remove it
  /// from the list and return it.
  ///
  /// Queues may be added to the outstanding queue list by olDestroyQueue if
  /// they are destroyed but not completed.
  __tgt_async_info *getOutstandingQueue() {
    // Not locking the `size()` access is fine here - In the worst case we
    // either miss a queue that exists or loop through an empty array after
    // taking the lock. Both are sub-optimal but not that bad.
    if (OutstandingQueues.size()) {
      std::lock_guard<std::mutex> Lock(OutstandingQueuesMutex);

      // As queues are pulled and popped from this list, longer running queues
      // naturally bubble to the start of the array. Hence looping backwards.
      for (auto Q = OutstandingQueues.rbegin(); Q != OutstandingQueues.rend();
           Q++) {
        if (!Device->hasPendingWork(*Q)) {
          auto OutstandingQueue = *Q;
          *Q = OutstandingQueues.back();
          OutstandingQueues.pop_back();
          return OutstandingQueue;
        }
      }
    }
    return nullptr;
  }

  /// Complete all pending work for this device and perform any needed cleanup.
  ///
  /// After calling this function, no liboffload functions should be called with
  /// this device handle.
  llvm::Error destroy() {
    llvm::Error Result = Plugin::success();
    for (auto Q : OutstandingQueues)
      if (auto Err = Device->synchronize(Q, /*Release=*/true))
        Result = llvm::joinErrors(std::move(Result), std::move(Err));
    OutstandingQueues.clear();
    return Result;
  }
};

struct ol_platform_impl_t {
  ol_platform_impl_t(std::unique_ptr<GenericPluginTy> Plugin,
                     ol_platform_backend_t BackendType)
      : Plugin(std::move(Plugin)), BackendType(BackendType) {}
  std::unique_ptr<GenericPluginTy> Plugin;
  llvm::SmallVector<std::unique_ptr<ol_device_impl_t>> Devices;
  ol_platform_backend_t BackendType;

  /// Complete all pending work for this platform and perform any needed
  /// cleanup.
  ///
  /// After calling this function, no liboffload functions should be called with
  /// this platform handle.
  llvm::Error destroy() {
    llvm::Error Result = Plugin::success();
    for (auto &D : Devices)
      if (auto Err = D->destroy())
        Result = llvm::joinErrors(std::move(Result), std::move(Err));

    if (auto Res = Plugin->deinit())
      Result = llvm::joinErrors(std::move(Result), std::move(Res));

    return Result;
  }
};

struct ol_queue_impl_t {
  ol_queue_impl_t(__tgt_async_info *AsyncInfo, ol_device_handle_t Device)
      : AsyncInfo(AsyncInfo), Device(Device), Id(IdCounter++) {}
  __tgt_async_info *AsyncInfo;
  ol_device_handle_t Device;
  // A unique identifier for the queue
  size_t Id;
  static std::atomic<size_t> IdCounter;
};
std::atomic<size_t> ol_queue_impl_t::IdCounter(0);

struct ol_event_impl_t {
  ol_event_impl_t(void *EventInfo, ol_device_handle_t Device,
                  ol_queue_handle_t Queue)
      : EventInfo(EventInfo), Device(Device), QueueId(Queue->Id), Queue(Queue) {
  }
  // EventInfo may be null, in which case the event should be considered always
  // complete
  void *EventInfo;
  ol_device_handle_t Device;
  size_t QueueId;
  // Events may outlive the queue - don't assume this is always valid.
  // It is provided only to implement OL_EVENT_INFO_QUEUE. Use QueueId to check
  // for queue equality instead.
  ol_queue_handle_t Queue;
};

struct ol_program_impl_t {
  ol_program_impl_t(plugin::DeviceImageTy *Image,
                    std::unique_ptr<llvm::MemoryBuffer> ImageData,
                    const __tgt_device_image &DeviceImage)
      : Image(Image), ImageData(std::move(ImageData)),
        DeviceImage(DeviceImage) {}
  plugin::DeviceImageTy *Image;
  std::unique_ptr<llvm::MemoryBuffer> ImageData;
  std::mutex SymbolListMutex;
  __tgt_device_image DeviceImage;
  llvm::StringMap<std::unique_ptr<ol_symbol_impl_t>> KernelSymbols;
  llvm::StringMap<std::unique_ptr<ol_symbol_impl_t>> GlobalSymbols;
};

struct ol_symbol_impl_t {
  ol_symbol_impl_t(const char *Name, GenericKernelTy *Kernel)
      : PluginImpl(Kernel), Kind(OL_SYMBOL_KIND_KERNEL), Name(Name) {}
  ol_symbol_impl_t(const char *Name, GlobalTy &&Global)
      : PluginImpl(Global), Kind(OL_SYMBOL_KIND_GLOBAL_VARIABLE), Name(Name) {}
  std::variant<GenericKernelTy *, GlobalTy> PluginImpl;
  ol_symbol_kind_t Kind;
  llvm::StringRef Name;
};

namespace llvm {
namespace offload {

struct AllocInfo {
  ol_device_handle_t Device;
  ol_alloc_type_t Type;
};

// Global shared state for liboffload
struct OffloadContext;
// This pointer is non-null if and only if the context is valid and fully
// initialized
static std::atomic<OffloadContext *> OffloadContextVal;
std::mutex OffloadContextValMutex;
struct OffloadContext {
  OffloadContext(OffloadContext &) = delete;
  OffloadContext(OffloadContext &&) = delete;
  OffloadContext &operator=(OffloadContext &) = delete;
  OffloadContext &operator=(OffloadContext &&) = delete;

  bool TracingEnabled = false;
  bool ValidationEnabled = true;
  DenseMap<void *, AllocInfo> AllocInfoMap{};
  std::mutex AllocInfoMapMutex{};
  SmallVector<ol_platform_impl_t, 4> Platforms{};
  size_t RefCount;

  ol_device_handle_t HostDevice() {
    // The host platform is always inserted last
    return Platforms.back().Devices[0].get();
  }

  static OffloadContext &get() {
    assert(OffloadContextVal);
    return *OffloadContextVal;
  }
};

// If the context is uninited, then we assume tracing is disabled
bool isTracingEnabled() {
  return isOffloadInitialized() && OffloadContext::get().TracingEnabled;
}
bool isValidationEnabled() { return OffloadContext::get().ValidationEnabled; }
bool isOffloadInitialized() { return OffloadContextVal != nullptr; }

template <typename HandleT> Error olDestroy(HandleT Handle) {
  delete Handle;
  return Error::success();
}

constexpr ol_platform_backend_t pluginNameToBackend(StringRef Name) {
  if (Name == "amdgpu") {
    return OL_PLATFORM_BACKEND_AMDGPU;
  } else if (Name == "cuda") {
    return OL_PLATFORM_BACKEND_CUDA;
  } else {
    return OL_PLATFORM_BACKEND_UNKNOWN;
  }
}

// Every plugin exports this method to create an instance of the plugin type.
#define PLUGIN_TARGET(Name) extern "C" GenericPluginTy *createPlugin_##Name();
#include "Shared/Targets.def"

Error initPlugins(OffloadContext &Context) {
  // Attempt to create an instance of each supported plugin.
#define PLUGIN_TARGET(Name)                                                    \
  do {                                                                         \
    Context.Platforms.emplace_back(ol_platform_impl_t{                         \
        std::unique_ptr<GenericPluginTy>(createPlugin_##Name()),               \
        pluginNameToBackend(#Name)});                                          \
  } while (false);
#include "Shared/Targets.def"

  // Preemptively initialize all devices in the plugin
  for (auto &Platform : Context.Platforms) {
    // Do not use the host plugin - it isn't supported.
    if (Platform.BackendType == OL_PLATFORM_BACKEND_UNKNOWN)
      continue;
    auto Err = Platform.Plugin->init();
    [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
    for (auto DevNum = 0; DevNum < Platform.Plugin->number_of_devices();
         DevNum++) {
      if (Platform.Plugin->init_device(DevNum) == OFFLOAD_SUCCESS) {
        auto Device = &Platform.Plugin->getDevice(DevNum);
        auto Info = Device->obtainInfoImpl();
        if (auto Err = Info.takeError())
          return Err;
        Platform.Devices.emplace_back(std::make_unique<ol_device_impl_t>(
            DevNum, Device, &Platform, std::move(*Info)));
      }
    }
  }

  // Add the special host device
  auto &HostPlatform = Context.Platforms.emplace_back(
      ol_platform_impl_t{nullptr, OL_PLATFORM_BACKEND_HOST});
  HostPlatform.Devices.emplace_back(
      std::make_unique<ol_device_impl_t>(-1, nullptr, nullptr, InfoTreeNode{}));
  Context.HostDevice()->Platform = &HostPlatform;

  Context.TracingEnabled = std::getenv("OFFLOAD_TRACE");
  Context.ValidationEnabled = !std::getenv("OFFLOAD_DISABLE_VALIDATION");

  return Plugin::success();
}

Error olInit_impl() {
  std::lock_guard<std::mutex> Lock(OffloadContextValMutex);

  if (isOffloadInitialized()) {
    OffloadContext::get().RefCount++;
    return Plugin::success();
  }

  // Use a temporary to ensure that entry points querying OffloadContextVal do
  // not get a partially initialized context
  auto *NewContext = new OffloadContext{};
  Error InitResult = initPlugins(*NewContext);
  OffloadContextVal.store(NewContext);
  OffloadContext::get().RefCount++;

  return InitResult;
}

Error olShutDown_impl() {
  std::lock_guard<std::mutex> Lock(OffloadContextValMutex);

  if (--OffloadContext::get().RefCount != 0)
    return Error::success();

  llvm::Error Result = Error::success();
  auto *OldContext = OffloadContextVal.exchange(nullptr);

  for (auto &P : OldContext->Platforms) {
    // Host plugin is nullptr and has no deinit
    if (!P.Plugin || !P.Plugin->is_initialized())
      continue;

    if (auto Res = P.destroy())
      Result = llvm::joinErrors(std::move(Result), std::move(Res));
  }

  delete OldContext;
  return Result;
}

Error olGetPlatformInfoImplDetail(ol_platform_handle_t Platform,
                                  ol_platform_info_t PropName, size_t PropSize,
                                  void *PropValue, size_t *PropSizeRet) {
  InfoWriter Info(PropSize, PropValue, PropSizeRet);
  bool IsHost = Platform->BackendType == OL_PLATFORM_BACKEND_HOST;

  switch (PropName) {
  case OL_PLATFORM_INFO_NAME:
    return Info.writeString(IsHost ? "Host" : Platform->Plugin->getName());
  case OL_PLATFORM_INFO_VENDOR_NAME:
    // TODO: Implement this
    return Info.writeString("Unknown platform vendor");
  case OL_PLATFORM_INFO_VERSION: {
    return Info.writeString(formatv("v{0}.{1}.{2}", OL_VERSION_MAJOR,
                                    OL_VERSION_MINOR, OL_VERSION_PATCH)
                                .str());
  }
  case OL_PLATFORM_INFO_BACKEND: {
    return Info.write<ol_platform_backend_t>(Platform->BackendType);
  }
  default:
    return createOffloadError(ErrorCode::INVALID_ENUMERATION,
                              "getPlatformInfo enum '%i' is invalid", PropName);
  }

  return Error::success();
}

Error olGetPlatformInfo_impl(ol_platform_handle_t Platform,
                             ol_platform_info_t PropName, size_t PropSize,
                             void *PropValue) {
  return olGetPlatformInfoImplDetail(Platform, PropName, PropSize, PropValue,
                                     nullptr);
}

Error olGetPlatformInfoSize_impl(ol_platform_handle_t Platform,
                                 ol_platform_info_t PropName,
                                 size_t *PropSizeRet) {
  return olGetPlatformInfoImplDetail(Platform, PropName, 0, nullptr,
                                     PropSizeRet);
}

Error olGetDeviceInfoImplDetail(ol_device_handle_t Device,
                                ol_device_info_t PropName, size_t PropSize,
                                void *PropValue, size_t *PropSizeRet) {
  assert(Device != OffloadContext::get().HostDevice());
  InfoWriter Info(PropSize, PropValue, PropSizeRet);

  auto makeError = [&](ErrorCode Code, StringRef Err) {
    std::string ErrBuffer;
    llvm::raw_string_ostream(ErrBuffer) << PropName << ": " << Err;
    return Plugin::error(ErrorCode::UNIMPLEMENTED, ErrBuffer.c_str());
  };

  // These are not implemented by the plugin interface
  switch (PropName) {
  case OL_DEVICE_INFO_PLATFORM:
    return Info.write<void *>(Device->Platform);

  case OL_DEVICE_INFO_TYPE:
    return Info.write<ol_device_type_t>(OL_DEVICE_TYPE_GPU);

  case OL_DEVICE_INFO_SINGLE_FP_CONFIG:
  case OL_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    ol_device_fp_capability_flags_t flags{0};
    flags |= OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT |
             OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
             OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
             OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF |
             OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN |
             OL_DEVICE_FP_CAPABILITY_FLAG_DENORM |
             OL_DEVICE_FP_CAPABILITY_FLAG_FMA;
    return Info.write(flags);
  }

  case OL_DEVICE_INFO_HALF_FP_CONFIG:
    return Info.write<ol_device_fp_capability_flags_t>(0);

  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
    return Info.write<uint32_t>(1);

  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
    return Info.write<uint32_t>(0);

  // None of the existing plugins specify a limit on a single allocation,
  // so return the global memory size instead
  case OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    [[fallthrough]];
  // AMD doesn't provide the global memory size (trivially) with the device info
  // struct, so use the plugin interface
  case OL_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    uint64_t Mem;
    if (auto Err = Device->Device->getDeviceMemorySize(Mem))
      return Err;
    return Info.write<uint64_t>(Mem);
  } break;

  default:
    break;
  }

  if (PropName >= OL_DEVICE_INFO_LAST)
    return createOffloadError(ErrorCode::INVALID_ENUMERATION,
                              "getDeviceInfo enum '%i' is invalid", PropName);

  auto EntryOpt = Device->Info.get(static_cast<DeviceInfo>(PropName));
  if (!EntryOpt)
    return makeError(ErrorCode::UNIMPLEMENTED,
                     "plugin did not provide a response for this information");
  auto Entry = *EntryOpt;

  // Retrieve properties from the plugin interface
  switch (PropName) {
  case OL_DEVICE_INFO_NAME:
  case OL_DEVICE_INFO_PRODUCT_NAME:
  case OL_DEVICE_INFO_VENDOR:
  case OL_DEVICE_INFO_DRIVER_VERSION: {
    // String values
    if (!std::holds_alternative<std::string>(Entry->Value))
      return makeError(ErrorCode::BACKEND_FAILURE,
                       "plugin returned incorrect type");
    return Info.writeString(std::get<std::string>(Entry->Value).c_str());
  }

  case OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
  case OL_DEVICE_INFO_MAX_WORK_SIZE:
  case OL_DEVICE_INFO_VENDOR_ID:
  case OL_DEVICE_INFO_NUM_COMPUTE_UNITS:
  case OL_DEVICE_INFO_ADDRESS_BITS:
  case OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
  case OL_DEVICE_INFO_MEMORY_CLOCK_RATE: {
    // Uint32 values
    if (!std::holds_alternative<uint64_t>(Entry->Value))
      return makeError(ErrorCode::BACKEND_FAILURE,
                       "plugin returned incorrect type");
    auto Value = std::get<uint64_t>(Entry->Value);
    if (Value > std::numeric_limits<uint32_t>::max())
      return makeError(ErrorCode::BACKEND_FAILURE,
                       "plugin returned out of range device info");
    return Info.write(static_cast<uint32_t>(Value));
  }

  case OL_DEVICE_INFO_MAX_WORK_SIZE_PER_DIMENSION:
  case OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION: {
    // {x, y, z} triples
    ol_dimensions_t Out{0, 0, 0};

    auto getField = [&](StringRef Name, uint32_t &Dest) {
      if (auto F = Entry->get(Name)) {
        if (!std::holds_alternative<size_t>((*F)->Value))
          return makeError(
              ErrorCode::BACKEND_FAILURE,
              "plugin returned incorrect type for dimensions element");
        Dest = std::get<size_t>((*F)->Value);
      } else
        return makeError(ErrorCode::BACKEND_FAILURE,
                         "plugin didn't provide all values for dimensions");
      return Plugin::success();
    };

    if (auto Res = getField("x", Out.x))
      return Res;
    if (auto Res = getField("y", Out.y))
      return Res;
    if (auto Res = getField("z", Out.z))
      return Res;

    return Info.write(Out);
  }

  default:
    llvm_unreachable("Unimplemented device info");
  }
}

Error olGetDeviceInfoImplDetailHost(ol_device_handle_t Device,
                                    ol_device_info_t PropName, size_t PropSize,
                                    void *PropValue, size_t *PropSizeRet) {
  assert(Device == OffloadContext::get().HostDevice());
  InfoWriter Info(PropSize, PropValue, PropSizeRet);

  constexpr auto uint32_max = std::numeric_limits<uint32_t>::max();

  switch (PropName) {
  case OL_DEVICE_INFO_PLATFORM:
    return Info.write<void *>(Device->Platform);
  case OL_DEVICE_INFO_TYPE:
    return Info.write<ol_device_type_t>(OL_DEVICE_TYPE_HOST);
  case OL_DEVICE_INFO_NAME:
    return Info.writeString("Virtual Host Device");
  case OL_DEVICE_INFO_PRODUCT_NAME:
    return Info.writeString("Virtual Host Device");
  case OL_DEVICE_INFO_VENDOR:
    return Info.writeString("Liboffload");
  case OL_DEVICE_INFO_DRIVER_VERSION:
    return Info.writeString(LLVM_VERSION_STRING);
  case OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return Info.write<uint32_t>(1);
  case OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION:
    return Info.write<ol_dimensions_t>(ol_dimensions_t{1, 1, 1});
  case OL_DEVICE_INFO_MAX_WORK_SIZE:
    return Info.write<uint32_t>(uint32_max);
  case OL_DEVICE_INFO_MAX_WORK_SIZE_PER_DIMENSION:
    return Info.write<ol_dimensions_t>(
        ol_dimensions_t{uint32_max, uint32_max, uint32_max});
  case OL_DEVICE_INFO_VENDOR_ID:
    return Info.write<uint32_t>(0);
  case OL_DEVICE_INFO_NUM_COMPUTE_UNITS:
    return Info.write<uint32_t>(1);
  case OL_DEVICE_INFO_SINGLE_FP_CONFIG:
  case OL_DEVICE_INFO_DOUBLE_FP_CONFIG:
    return Info.write<ol_device_fp_capability_flags_t>(
        OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT |
        OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST |
        OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO |
        OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF |
        OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN |
        OL_DEVICE_FP_CAPABILITY_FLAG_DENORM | OL_DEVICE_FP_CAPABILITY_FLAG_FMA);
  case OL_DEVICE_INFO_HALF_FP_CONFIG:
    return Info.write<ol_device_fp_capability_flags_t>(0);
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
    return Info.write<uint32_t>(1);
  case OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
    return Info.write<uint32_t>(0);
  case OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
  case OL_DEVICE_INFO_MEMORY_CLOCK_RATE:
  case OL_DEVICE_INFO_ADDRESS_BITS:
    return Info.write<uint32_t>(std::numeric_limits<uintptr_t>::digits);
  case OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
  case OL_DEVICE_INFO_GLOBAL_MEM_SIZE:
    return Info.write<uint64_t>(0);
  default:
    return createOffloadError(ErrorCode::INVALID_ENUMERATION,
                              "getDeviceInfo enum '%i' is invalid", PropName);
  }

  return Error::success();
}

Error olGetDeviceInfo_impl(ol_device_handle_t Device, ol_device_info_t PropName,
                           size_t PropSize, void *PropValue) {
  if (Device == OffloadContext::get().HostDevice())
    return olGetDeviceInfoImplDetailHost(Device, PropName, PropSize, PropValue,
                                         nullptr);
  return olGetDeviceInfoImplDetail(Device, PropName, PropSize, PropValue,
                                   nullptr);
}

Error olGetDeviceInfoSize_impl(ol_device_handle_t Device,
                               ol_device_info_t PropName, size_t *PropSizeRet) {
  if (Device == OffloadContext::get().HostDevice())
    return olGetDeviceInfoImplDetailHost(Device, PropName, 0, nullptr,
                                         PropSizeRet);
  return olGetDeviceInfoImplDetail(Device, PropName, 0, nullptr, PropSizeRet);
}

Error olIterateDevices_impl(ol_device_iterate_cb_t Callback, void *UserData) {
  for (auto &Platform : OffloadContext::get().Platforms) {
    for (auto &Device : Platform.Devices) {
      if (!Callback(Device.get(), UserData)) {
        break;
      }
    }
  }

  return Error::success();
}

TargetAllocTy convertOlToPluginAllocTy(ol_alloc_type_t Type) {
  switch (Type) {
  case OL_ALLOC_TYPE_DEVICE:
    return TARGET_ALLOC_DEVICE;
  case OL_ALLOC_TYPE_HOST:
    return TARGET_ALLOC_HOST;
  case OL_ALLOC_TYPE_MANAGED:
  default:
    return TARGET_ALLOC_SHARED;
  }
}

Error olMemAlloc_impl(ol_device_handle_t Device, ol_alloc_type_t Type,
                      size_t Size, void **AllocationOut) {
  auto Alloc =
      Device->Device->dataAlloc(Size, nullptr, convertOlToPluginAllocTy(Type));
  if (!Alloc)
    return Alloc.takeError();

  *AllocationOut = *Alloc;
  {
    std::lock_guard<std::mutex> Lock(OffloadContext::get().AllocInfoMapMutex);
    OffloadContext::get().AllocInfoMap.insert_or_assign(
        *Alloc, AllocInfo{Device, Type});
  }
  return Error::success();
}

Error olMemFree_impl(void *Address) {
  ol_device_handle_t Device;
  ol_alloc_type_t Type;
  {
    std::lock_guard<std::mutex> Lock(OffloadContext::get().AllocInfoMapMutex);
    if (!OffloadContext::get().AllocInfoMap.contains(Address))
      return createOffloadError(ErrorCode::INVALID_ARGUMENT,
                                "address is not a known allocation");

    auto AllocInfo = OffloadContext::get().AllocInfoMap.at(Address);
    Device = AllocInfo.Device;
    Type = AllocInfo.Type;
    OffloadContext::get().AllocInfoMap.erase(Address);
  }

  if (auto Res =
          Device->Device->dataDelete(Address, convertOlToPluginAllocTy(Type)))
    return Res;

  return Error::success();
}

Error olCreateQueue_impl(ol_device_handle_t Device, ol_queue_handle_t *Queue) {
  auto CreatedQueue = std::make_unique<ol_queue_impl_t>(nullptr, Device);

  auto OutstandingQueue = Device->getOutstandingQueue();
  if (OutstandingQueue) {
    // The queue is empty, but we still need to sync it to release any temporary
    // memory allocations or do other cleanup.
    if (auto Err =
            Device->Device->synchronize(OutstandingQueue, /*Release=*/false))
      return Err;
    CreatedQueue->AsyncInfo = OutstandingQueue;
  } else if (auto Err =
                 Device->Device->initAsyncInfo(&(CreatedQueue->AsyncInfo))) {
    return Err;
  }

  *Queue = CreatedQueue.release();
  return Error::success();
}

Error olDestroyQueue_impl(ol_queue_handle_t Queue) {
  auto *Device = Queue->Device;
  // This is safe; as soon as olDestroyQueue is called it is not possible to add
  // any more work to the queue, so if it's finished now it will remain finished
  // forever.
  auto Res = Device->Device->hasPendingWork(Queue->AsyncInfo);
  if (!Res)
    return Res.takeError();

  if (!*Res) {
    // The queue is complete, so sync it and throw it back into the pool.
    if (auto Err = Device->Device->synchronize(Queue->AsyncInfo,
                                               /*Release=*/true))
      return Err;
  } else {
    // The queue still has outstanding work. Store it so we can check it later.
    std::lock_guard<std::mutex> Lock(Device->OutstandingQueuesMutex);
    Device->OutstandingQueues.push_back(Queue->AsyncInfo);
  }

  return olDestroy(Queue);
}

Error olSyncQueue_impl(ol_queue_handle_t Queue) {
  // Host plugin doesn't have a queue set so it's not safe to call synchronize
  // on it, but we have nothing to synchronize in that situation anyway.
  if (Queue->AsyncInfo->Queue) {
    // We don't need to release the queue and we would like the ability for
    // other offload threads to submit work concurrently, so pass "false" here
    // so we don't release the underlying queue object.
    if (auto Err = Queue->Device->Device->synchronize(Queue->AsyncInfo, false))
      return Err;
  }

  return Error::success();
}

Error olWaitEvents_impl(ol_queue_handle_t Queue, ol_event_handle_t *Events,
                        size_t NumEvents) {
  auto *Device = Queue->Device->Device;

  for (size_t I = 0; I < NumEvents; I++) {
    auto *Event = Events[I];

    if (!Event)
      return Plugin::error(ErrorCode::INVALID_NULL_HANDLE,
                           "olWaitEvents asked to wait on a NULL event");

    // Do nothing if the event is for this queue or the event is always complete
    if (Event->QueueId == Queue->Id || !Event->EventInfo)
      continue;

    if (auto Err = Device->waitEvent(Event->EventInfo, Queue->AsyncInfo))
      return Err;
  }

  return Error::success();
}

Error olGetQueueInfoImplDetail(ol_queue_handle_t Queue,
                               ol_queue_info_t PropName, size_t PropSize,
                               void *PropValue, size_t *PropSizeRet) {
  InfoWriter Info(PropSize, PropValue, PropSizeRet);

  switch (PropName) {
  case OL_QUEUE_INFO_DEVICE:
    return Info.write<ol_device_handle_t>(Queue->Device);
  case OL_QUEUE_INFO_EMPTY: {
    auto Pending = Queue->Device->Device->hasPendingWork(Queue->AsyncInfo);
    if (auto Err = Pending.takeError())
      return Err;
    return Info.write<bool>(!*Pending);
  }
  default:
    return createOffloadError(ErrorCode::INVALID_ENUMERATION,
                              "olGetQueueInfo enum '%i' is invalid", PropName);
  }

  return Error::success();
}

Error olGetQueueInfo_impl(ol_queue_handle_t Queue, ol_queue_info_t PropName,
                          size_t PropSize, void *PropValue) {
  return olGetQueueInfoImplDetail(Queue, PropName, PropSize, PropValue,
                                  nullptr);
}

Error olGetQueueInfoSize_impl(ol_queue_handle_t Queue, ol_queue_info_t PropName,
                              size_t *PropSizeRet) {
  return olGetQueueInfoImplDetail(Queue, PropName, 0, nullptr, PropSizeRet);
}

Error olSyncEvent_impl(ol_event_handle_t Event) {
  // No event info means that this event was complete on creation
  if (!Event->EventInfo)
    return Plugin::success();

  if (auto Res = Event->Device->Device->syncEvent(Event->EventInfo))
    return Res;

  return Error::success();
}

Error olDestroyEvent_impl(ol_event_handle_t Event) {
  if (Event->EventInfo)
    if (auto Res = Event->Device->Device->destroyEvent(Event->EventInfo))
      return Res;

  return olDestroy(Event);
}

Error olGetEventInfoImplDetail(ol_event_handle_t Event,
                               ol_event_info_t PropName, size_t PropSize,
                               void *PropValue, size_t *PropSizeRet) {
  InfoWriter Info(PropSize, PropValue, PropSizeRet);
  auto Queue = Event->Queue;

  switch (PropName) {
  case OL_EVENT_INFO_QUEUE:
    return Info.write<ol_queue_handle_t>(Queue);
  case OL_EVENT_INFO_IS_COMPLETE: {
    // No event info means that this event was complete on creation
    if (!Event->EventInfo)
      return Info.write<bool>(true);

    auto Res = Queue->Device->Device->isEventComplete(Event->EventInfo,
                                                      Queue->AsyncInfo);
    if (auto Err = Res.takeError())
      return Err;
    return Info.write<bool>(*Res);
  }
  default:
    return createOffloadError(ErrorCode::INVALID_ENUMERATION,
                              "olGetEventInfo enum '%i' is invalid", PropName);
  }

  return Error::success();
}

Error olGetEventInfo_impl(ol_event_handle_t Event, ol_event_info_t PropName,
                          size_t PropSize, void *PropValue) {

  return olGetEventInfoImplDetail(Event, PropName, PropSize, PropValue,
                                  nullptr);
}

Error olGetEventInfoSize_impl(ol_event_handle_t Event, ol_event_info_t PropName,
                              size_t *PropSizeRet) {
  return olGetEventInfoImplDetail(Event, PropName, 0, nullptr, PropSizeRet);
}

Error olCreateEvent_impl(ol_queue_handle_t Queue, ol_event_handle_t *EventOut) {
  auto Pending = Queue->Device->Device->hasPendingWork(Queue->AsyncInfo);
  if (auto Err = Pending.takeError())
    return Err;

  *EventOut = new ol_event_impl_t(nullptr, Queue->Device, Queue);
  if (!*Pending)
    // Queue is empty, don't record an event and consider the event always
    // complete
    return Plugin::success();

  if (auto Res = Queue->Device->Device->createEvent(&(*EventOut)->EventInfo))
    return Res;

  if (auto Res = Queue->Device->Device->recordEvent((*EventOut)->EventInfo,
                                                    Queue->AsyncInfo))
    return Res;

  return Plugin::success();
}

Error olMemcpy_impl(ol_queue_handle_t Queue, void *DstPtr,
                    ol_device_handle_t DstDevice, const void *SrcPtr,
                    ol_device_handle_t SrcDevice, size_t Size) {
  auto Host = OffloadContext::get().HostDevice();
  if (DstDevice == Host && SrcDevice == Host) {
    if (!Queue) {
      std::memcpy(DstPtr, SrcPtr, Size);
      return Error::success();
    } else {
      return createOffloadError(
          ErrorCode::INVALID_ARGUMENT,
          "ane of DstDevice and SrcDevice must be a non-host device if "
          "queue is specified");
    }
  }

  // If no queue is given the memcpy will be synchronous
  auto QueueImpl = Queue ? Queue->AsyncInfo : nullptr;

  if (DstDevice == Host) {
    if (auto Res =
            SrcDevice->Device->dataRetrieve(DstPtr, SrcPtr, Size, QueueImpl))
      return Res;
  } else if (SrcDevice == Host) {
    if (auto Res =
            DstDevice->Device->dataSubmit(DstPtr, SrcPtr, Size, QueueImpl))
      return Res;
  } else {
    if (auto Res = SrcDevice->Device->dataExchange(SrcPtr, *DstDevice->Device,
                                                   DstPtr, Size, QueueImpl))
      return Res;
  }

  return Error::success();
}

Error olMemFill_impl(ol_queue_handle_t Queue, void *Ptr, size_t PatternSize,
                     const void *PatternPtr, size_t FillSize) {
  return Queue->Device->Device->dataFill(Ptr, PatternPtr, PatternSize, FillSize,
                                         Queue->AsyncInfo);
}

Error olCreateProgram_impl(ol_device_handle_t Device, const void *ProgData,
                           size_t ProgDataSize, ol_program_handle_t *Program) {
  // Make a copy of the program binary in case it is released by the caller.
  auto ImageData = MemoryBuffer::getMemBufferCopy(
      StringRef(reinterpret_cast<const char *>(ProgData), ProgDataSize));

  auto DeviceImage = __tgt_device_image{
      const_cast<char *>(ImageData->getBuffer().data()),
      const_cast<char *>(ImageData->getBuffer().data()) + ProgDataSize, nullptr,
      nullptr};

  ol_program_handle_t Prog =
      new ol_program_impl_t(nullptr, std::move(ImageData), DeviceImage);

  auto Res =
      Device->Device->loadBinary(Device->Device->Plugin, &Prog->DeviceImage);
  if (!Res) {
    delete Prog;
    return Res.takeError();
  }
  assert(*Res != nullptr && "loadBinary returned nullptr");

  Prog->Image = *Res;
  *Program = Prog;

  return Error::success();
}

Error olDestroyProgram_impl(ol_program_handle_t Program) {
  auto &Device = Program->Image->getDevice();
  if (auto Err = Device.unloadBinary(Program->Image))
    return Err;

  auto &LoadedImages = Device.LoadedImages;
  LoadedImages.erase(
      std::find(LoadedImages.begin(), LoadedImages.end(), Program->Image));

  return olDestroy(Program);
}

Error olCalculateOptimalOccupancy_impl(ol_device_handle_t Device,
                                       ol_symbol_handle_t Kernel,
                                       size_t DynamicMemSize,
                                       size_t *GroupSize) {
  if (Kernel->Kind != OL_SYMBOL_KIND_KERNEL)
    return createOffloadError(ErrorCode::SYMBOL_KIND,
                              "provided symbol is not a kernel");
  auto *KernelImpl = std::get<GenericKernelTy *>(Kernel->PluginImpl);

  auto Res = KernelImpl->maxGroupSize(*Device->Device, DynamicMemSize);
  if (auto Err = Res.takeError())
    return Err;

  *GroupSize = *Res;

  return Error::success();
}

Error olLaunchKernel_impl(ol_queue_handle_t Queue, ol_device_handle_t Device,
                          ol_symbol_handle_t Kernel, const void *ArgumentsData,
                          size_t ArgumentsSize,
                          const ol_kernel_launch_size_args_t *LaunchSizeArgs) {
  auto *DeviceImpl = Device->Device;
  if (Queue && Device != Queue->Device) {
    return createOffloadError(
        ErrorCode::INVALID_DEVICE,
        "device specified does not match the device of the given queue");
  }

  if (Kernel->Kind != OL_SYMBOL_KIND_KERNEL)
    return createOffloadError(ErrorCode::SYMBOL_KIND,
                              "provided symbol is not a kernel");

  auto *QueueImpl = Queue ? Queue->AsyncInfo : nullptr;
  AsyncInfoWrapperTy AsyncInfoWrapper(*DeviceImpl, QueueImpl);
  KernelArgsTy LaunchArgs{};
  LaunchArgs.NumTeams[0] = LaunchSizeArgs->NumGroups.x;
  LaunchArgs.NumTeams[1] = LaunchSizeArgs->NumGroups.y;
  LaunchArgs.NumTeams[2] = LaunchSizeArgs->NumGroups.z;
  LaunchArgs.ThreadLimit[0] = LaunchSizeArgs->GroupSize.x;
  LaunchArgs.ThreadLimit[1] = LaunchSizeArgs->GroupSize.y;
  LaunchArgs.ThreadLimit[2] = LaunchSizeArgs->GroupSize.z;
  LaunchArgs.DynCGroupMem = LaunchSizeArgs->DynSharedMemory;

  KernelLaunchParamsTy Params;
  Params.Data = const_cast<void *>(ArgumentsData);
  Params.Size = ArgumentsSize;
  LaunchArgs.ArgPtrs = reinterpret_cast<void **>(&Params);
  // Don't do anything with pointer indirection; use arg data as-is
  LaunchArgs.Flags.IsCUDA = true;

  auto *KernelImpl = std::get<GenericKernelTy *>(Kernel->PluginImpl);
  auto Err = KernelImpl->launch(*DeviceImpl, LaunchArgs.ArgPtrs, nullptr,
                                LaunchArgs, AsyncInfoWrapper);

  AsyncInfoWrapper.finalize(Err);
  if (Err)
    return Err;

  return Error::success();
}

Error olGetSymbol_impl(ol_program_handle_t Program, const char *Name,
                       ol_symbol_kind_t Kind, ol_symbol_handle_t *Symbol) {
  auto &Device = Program->Image->getDevice();

  std::lock_guard<std::mutex> Lock(Program->SymbolListMutex);

  switch (Kind) {
  case OL_SYMBOL_KIND_KERNEL: {
    auto &Kernel = Program->KernelSymbols[Name];
    if (!Kernel) {
      auto KernelImpl = Device.constructKernel(Name);
      if (!KernelImpl)
        return KernelImpl.takeError();

      if (auto Err = KernelImpl->init(Device, *Program->Image))
        return Err;

      Kernel = std::make_unique<ol_symbol_impl_t>(KernelImpl->getName(),
                                                  &*KernelImpl);
    }

    *Symbol = Kernel.get();
    return Error::success();
  }
  case OL_SYMBOL_KIND_GLOBAL_VARIABLE: {
    auto &Global = Program->GlobalSymbols[Name];
    if (!Global) {
      GlobalTy GlobalObj{Name};
      if (auto Res =
              Device.Plugin.getGlobalHandler().getGlobalMetadataFromDevice(
                  Device, *Program->Image, GlobalObj))
        return Res;

      Global = std::make_unique<ol_symbol_impl_t>(GlobalObj.getName().c_str(),
                                                  std::move(GlobalObj));
    }

    *Symbol = Global.get();
    return Error::success();
  }
  default:
    return createOffloadError(ErrorCode::INVALID_ENUMERATION,
                              "getSymbol kind enum '%i' is invalid", Kind);
  }
}

Error olGetSymbolInfoImplDetail(ol_symbol_handle_t Symbol,
                                ol_symbol_info_t PropName, size_t PropSize,
                                void *PropValue, size_t *PropSizeRet) {
  InfoWriter Info(PropSize, PropValue, PropSizeRet);

  auto CheckKind = [&](ol_symbol_kind_t Required) {
    if (Symbol->Kind != Required) {
      std::string ErrBuffer;
      llvm::raw_string_ostream(ErrBuffer)
          << PropName << ": Expected a symbol of Kind " << Required
          << " but given a symbol of Kind " << Symbol->Kind;
      return Plugin::error(ErrorCode::SYMBOL_KIND, ErrBuffer.c_str());
    }
    return Plugin::success();
  };

  switch (PropName) {
  case OL_SYMBOL_INFO_KIND:
    return Info.write<ol_symbol_kind_t>(Symbol->Kind);
  case OL_SYMBOL_INFO_GLOBAL_VARIABLE_ADDRESS:
    if (auto Err = CheckKind(OL_SYMBOL_KIND_GLOBAL_VARIABLE))
      return Err;
    return Info.write<void *>(std::get<GlobalTy>(Symbol->PluginImpl).getPtr());
  case OL_SYMBOL_INFO_GLOBAL_VARIABLE_SIZE:
    if (auto Err = CheckKind(OL_SYMBOL_KIND_GLOBAL_VARIABLE))
      return Err;
    return Info.write<size_t>(std::get<GlobalTy>(Symbol->PluginImpl).getSize());
  default:
    return createOffloadError(ErrorCode::INVALID_ENUMERATION,
                              "olGetSymbolInfo enum '%i' is invalid", PropName);
  }

  return Error::success();
}

Error olGetSymbolInfo_impl(ol_symbol_handle_t Symbol, ol_symbol_info_t PropName,
                           size_t PropSize, void *PropValue) {

  return olGetSymbolInfoImplDetail(Symbol, PropName, PropSize, PropValue,
                                   nullptr);
}

Error olGetSymbolInfoSize_impl(ol_symbol_handle_t Symbol,
                               ol_symbol_info_t PropName, size_t *PropSizeRet) {
  return olGetSymbolInfoImplDetail(Symbol, PropName, 0, nullptr, PropSizeRet);
}

Error olLaunchHostFunction_impl(ol_queue_handle_t Queue,
                                ol_host_function_cb_t Callback,
                                void *UserData) {
  return Queue->Device->Device->enqueueHostCall(Callback, UserData,
                                                Queue->AsyncInfo);
}

} // namespace offload
} // namespace llvm
