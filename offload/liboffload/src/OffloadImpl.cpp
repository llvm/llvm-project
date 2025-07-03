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
  int DeviceNum;
  GenericDeviceTy *Device;
  ol_platform_handle_t Platform;
  InfoTreeNode Info;
};

struct ol_platform_impl_t {
  ol_platform_impl_t(std::unique_ptr<GenericPluginTy> Plugin,
                     ol_platform_backend_t BackendType)
      : Plugin(std::move(Plugin)), BackendType(BackendType) {}
  std::unique_ptr<GenericPluginTy> Plugin;
  std::vector<ol_device_impl_t> Devices;
  ol_platform_backend_t BackendType;
};

struct ol_queue_impl_t {
  ol_queue_impl_t(__tgt_async_info *AsyncInfo, ol_device_handle_t Device)
      : AsyncInfo(AsyncInfo), Device(Device) {}
  __tgt_async_info *AsyncInfo;
  ol_device_handle_t Device;
};

struct ol_event_impl_t {
  ol_event_impl_t(void *EventInfo, ol_queue_handle_t Queue)
      : EventInfo(EventInfo), Queue(Queue) {}
  void *EventInfo;
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
  __tgt_device_image DeviceImage;
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
  SmallVector<ol_platform_impl_t, 4> Platforms{};
  size_t RefCount;

  ol_device_handle_t HostDevice() {
    // The host platform is always inserted last
    return &Platforms.back().Devices[0];
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
        Platform.Devices.emplace_back(DevNum, Device, &Platform,
                                      std::move(*Info));
      }
    }
  }

  // Add the special host device
  auto &HostPlatform = Context.Platforms.emplace_back(
      ol_platform_impl_t{nullptr, OL_PLATFORM_BACKEND_HOST});
  HostPlatform.Devices.emplace_back(-1, nullptr, nullptr, InfoTreeNode{});
  Context.HostDevice()->Platform = &HostPlatform;

  Context.TracingEnabled = std::getenv("OFFLOAD_TRACE");
  Context.ValidationEnabled = !std::getenv("OFFLOAD_DISABLE_VALIDATION");

  return Plugin::success();
}

Error olInit_impl() {
  std::lock_guard<std::mutex> Lock{OffloadContextValMutex};

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
  std::lock_guard<std::mutex> Lock{OffloadContextValMutex};

  if (--OffloadContext::get().RefCount != 0)
    return Error::success();

  llvm::Error Result = Error::success();
  auto *OldContext = OffloadContextVal.exchange(nullptr);

  for (auto &P : OldContext->Platforms) {
    // Host plugin is nullptr and has no deinit
    if (!P.Plugin)
      continue;

    if (auto Res = P.Plugin->deinit())
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

  // Find the info if it exists under any of the given names
  auto getInfoString =
      [&](std::vector<std::string> Names) -> llvm::Expected<const char *> {
    for (auto &Name : Names) {
      if (auto Entry = Device->Info.get(Name)) {
        if (!std::holds_alternative<std::string>((*Entry)->Value))
          return makeError(ErrorCode::BACKEND_FAILURE,
                           "plugin returned incorrect type");
        return std::get<std::string>((*Entry)->Value).c_str();
      }
    }

    return makeError(ErrorCode::UNIMPLEMENTED,
                     "plugin did not provide a response for this information");
  };

  auto getInfoXyz =
      [&](std::vector<std::string> Names) -> llvm::Expected<ol_dimensions_t> {
    for (auto &Name : Names) {
      if (auto Entry = Device->Info.get(Name)) {
        auto Node = *Entry;
        ol_dimensions_t Out{0, 0, 0};

        auto getField = [&](StringRef Name, uint32_t &Dest) {
          if (auto F = Node->get(Name)) {
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

        return Out;
      }
    }

    return makeError(ErrorCode::UNIMPLEMENTED,
                     "plugin did not provide a response for this information");
  };

  switch (PropName) {
  case OL_DEVICE_INFO_PLATFORM:
    return Info.write<void *>(Device->Platform);
  case OL_DEVICE_INFO_TYPE:
    return Info.write<ol_device_type_t>(OL_DEVICE_TYPE_GPU);
  case OL_DEVICE_INFO_NAME:
    return Info.writeString(getInfoString({"Device Name"}));
  case OL_DEVICE_INFO_VENDOR:
    return Info.writeString(getInfoString({"Vendor Name"}));
  case OL_DEVICE_INFO_DRIVER_VERSION:
    return Info.writeString(
        getInfoString({"CUDA Driver Version", "HSA Runtime Version"}));
  case OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return Info.write(getInfoXyz({"Workgroup Max Size per Dimension" /*AMD*/,
                                  "Maximum Block Dimensions" /*CUDA*/}));
  default:
    return createOffloadError(ErrorCode::INVALID_ENUMERATION,
                              "getDeviceInfo enum '%i' is invalid", PropName);
  }

  return Error::success();
}

Error olGetDeviceInfoImplDetailHost(ol_device_handle_t Device,
                                    ol_device_info_t PropName, size_t PropSize,
                                    void *PropValue, size_t *PropSizeRet) {
  assert(Device == OffloadContext::get().HostDevice());
  InfoWriter Info(PropSize, PropValue, PropSizeRet);

  switch (PropName) {
  case OL_DEVICE_INFO_PLATFORM:
    return Info.write<void *>(Device->Platform);
  case OL_DEVICE_INFO_TYPE:
    return Info.write<ol_device_type_t>(OL_DEVICE_TYPE_HOST);
  case OL_DEVICE_INFO_NAME:
    return Info.writeString("Virtual Host Device");
  case OL_DEVICE_INFO_VENDOR:
    return Info.writeString("Liboffload");
  case OL_DEVICE_INFO_DRIVER_VERSION:
    return Info.writeString(LLVM_VERSION_STRING);
  case OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return Info.write<ol_dimensions_t>(ol_dimensions_t{1, 1, 1});
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
      if (!Callback(&Device, UserData)) {
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
  OffloadContext::get().AllocInfoMap.insert_or_assign(*Alloc,
                                                      AllocInfo{Device, Type});
  return Error::success();
}

Error olMemFree_impl(void *Address) {
  if (!OffloadContext::get().AllocInfoMap.contains(Address))
    return createOffloadError(ErrorCode::INVALID_ARGUMENT,
                              "address is not a known allocation");

  auto AllocInfo = OffloadContext::get().AllocInfoMap.at(Address);
  auto Device = AllocInfo.Device;
  auto Type = AllocInfo.Type;

  if (auto Res =
          Device->Device->dataDelete(Address, convertOlToPluginAllocTy(Type)))
    return Res;

  OffloadContext::get().AllocInfoMap.erase(Address);

  return Error::success();
}

Error olCreateQueue_impl(ol_device_handle_t Device, ol_queue_handle_t *Queue) {
  auto CreatedQueue = std::make_unique<ol_queue_impl_t>(nullptr, Device);
  if (auto Err = Device->Device->initAsyncInfo(&(CreatedQueue->AsyncInfo)))
    return Err;

  *Queue = CreatedQueue.release();
  return Error::success();
}

Error olDestroyQueue_impl(ol_queue_handle_t Queue) { return olDestroy(Queue); }

Error olWaitQueue_impl(ol_queue_handle_t Queue) {
  // Host plugin doesn't have a queue set so it's not safe to call synchronize
  // on it, but we have nothing to synchronize in that situation anyway.
  if (Queue->AsyncInfo->Queue) {
    if (auto Err = Queue->Device->Device->synchronize(Queue->AsyncInfo))
      return Err;
  }

  // Recreate the stream resource so the queue can be reused
  // TODO: Would be easier for the synchronization to (optionally) not release
  // it to begin with.
  if (auto Res = Queue->Device->Device->initAsyncInfo(&Queue->AsyncInfo))
    return Res;

  return Error::success();
}

Error olWaitEvent_impl(ol_event_handle_t Event) {
  if (auto Res = Event->Queue->Device->Device->syncEvent(Event->EventInfo))
    return Res;

  return Error::success();
}

Error olDestroyEvent_impl(ol_event_handle_t Event) {
  if (auto Res = Event->Queue->Device->Device->destroyEvent(Event->EventInfo))
    return Res;

  return olDestroy(Event);
}

ol_event_handle_t makeEvent(ol_queue_handle_t Queue) {
  auto EventImpl = std::make_unique<ol_event_impl_t>(nullptr, Queue);
  if (auto Res = Queue->Device->Device->createEvent(&EventImpl->EventInfo)) {
    llvm::consumeError(std::move(Res));
    return nullptr;
  }

  if (auto Res = Queue->Device->Device->recordEvent(EventImpl->EventInfo,
                                                    Queue->AsyncInfo)) {
    llvm::consumeError(std::move(Res));
    return nullptr;
  }

  return EventImpl.release();
}

Error olMemcpy_impl(ol_queue_handle_t Queue, void *DstPtr,
                    ol_device_handle_t DstDevice, const void *SrcPtr,
                    ol_device_handle_t SrcDevice, size_t Size,
                    ol_event_handle_t *EventOut) {
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

  if (EventOut)
    *EventOut = makeEvent(Queue);

  return Error::success();
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

Error olGetKernel_impl(ol_program_handle_t Program, const char *KernelName,
                       ol_kernel_handle_t *Kernel) {

  auto &Device = Program->Image->getDevice();
  auto KernelImpl = Device.constructKernel(KernelName);
  if (!KernelImpl)
    return KernelImpl.takeError();

  if (auto Err = KernelImpl->init(Device, *Program->Image))
    return Err;

  *Kernel = &*KernelImpl;

  return Error::success();
}

Error olLaunchKernel_impl(ol_queue_handle_t Queue, ol_device_handle_t Device,
                          ol_kernel_handle_t Kernel, const void *ArgumentsData,
                          size_t ArgumentsSize,
                          const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                          ol_event_handle_t *EventOut) {
  auto *DeviceImpl = Device->Device;
  if (Queue && Device != Queue->Device) {
    return createOffloadError(
        ErrorCode::INVALID_DEVICE,
        "device specified does not match the device of the given queue");
  }

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

  auto *KernelImpl = reinterpret_cast<GenericKernelTy *>(Kernel);
  auto Err = KernelImpl->launch(*DeviceImpl, LaunchArgs.ArgPtrs, nullptr,
                                LaunchArgs, AsyncInfoWrapper);

  AsyncInfoWrapper.finalize(Err);
  if (Err)
    return Err;

  if (EventOut)
    *EventOut = makeEvent(Queue);

  return Error::success();
}

} // namespace offload
} // namespace llvm
