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
                   ol_platform_handle_t Platform)
      : DeviceNum(DeviceNum), Device(Device), Platform(Platform) {}
  int DeviceNum;
  GenericDeviceTy *Device;
  ol_platform_handle_t Platform;
};

struct ol_platform_impl_t {
  ol_platform_impl_t(std::unique_ptr<GenericPluginTy> Plugin,
                     std::vector<ol_device_impl_t> Devices,
                     ol_platform_backend_t BackendType)
      : Plugin(std::move(Plugin)), Devices(Devices), BackendType(BackendType) {}
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

using AllocInfoMapT = DenseMap<void *, AllocInfo>;
AllocInfoMapT &allocInfoMap() {
  static AllocInfoMapT AllocInfoMap{};
  return AllocInfoMap;
}

using PlatformVecT = SmallVector<ol_platform_impl_t, 4>;
PlatformVecT &Platforms() {
  static PlatformVecT Platforms;
  return Platforms;
}

ol_device_handle_t HostDevice() {
  // The host platform is always inserted last
  return &Platforms().back().Devices[0];
}

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

void initPlugins() {
  // Attempt to create an instance of each supported plugin.
#define PLUGIN_TARGET(Name)                                                    \
  do {                                                                         \
    Platforms().emplace_back(ol_platform_impl_t{                               \
        std::unique_ptr<GenericPluginTy>(createPlugin_##Name()),               \
        {},                                                                    \
        pluginNameToBackend(#Name)});                                          \
  } while (false);
#include "Shared/Targets.def"

  // Preemptively initialize all devices in the plugin
  for (auto &Platform : Platforms()) {
    // Do not use the host plugin - it isn't supported.
    if (Platform.BackendType == OL_PLATFORM_BACKEND_UNKNOWN)
      continue;
    auto Err = Platform.Plugin->init();
    [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
    for (auto DevNum = 0; DevNum < Platform.Plugin->number_of_devices();
         DevNum++) {
      if (Platform.Plugin->init_device(DevNum) == OFFLOAD_SUCCESS) {
        Platform.Devices.emplace_back(ol_device_impl_t{
            DevNum, &Platform.Plugin->getDevice(DevNum), &Platform});
      }
    }
  }

  // Add the special host device
  auto &HostPlatform = Platforms().emplace_back(
      ol_platform_impl_t{nullptr,
                         {ol_device_impl_t{-1, nullptr, nullptr}},
                         OL_PLATFORM_BACKEND_HOST});
  HostDevice()->Platform = &HostPlatform;

  offloadConfig().TracingEnabled = std::getenv("OFFLOAD_TRACE");
  offloadConfig().ValidationEnabled =
      !std::getenv("OFFLOAD_DISABLE_VALIDATION");
}

// TODO: We can properly reference count here and manage the resources in a more
// clever way
Error olInit_impl() {
  static std::once_flag InitFlag;
  std::call_once(InitFlag, initPlugins);

  return Error::success();
}
Error olShutDown_impl() { return Error::success(); }

Error olGetPlatformInfoImplDetail(ol_platform_handle_t Platform,
                                  ol_platform_info_t PropName, size_t PropSize,
                                  void *PropValue, size_t *PropSizeRet) {
  ReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);
  bool IsHost = Platform->BackendType == OL_PLATFORM_BACKEND_HOST;

  switch (PropName) {
  case OL_PLATFORM_INFO_NAME:
    return ReturnValue(IsHost ? "Host" : Platform->Plugin->getName());
  case OL_PLATFORM_INFO_VENDOR_NAME:
    // TODO: Implement this
    return ReturnValue("Unknown platform vendor");
  case OL_PLATFORM_INFO_VERSION: {
    return ReturnValue(formatv("v{0}.{1}.{2}", OL_VERSION_MAJOR,
                               OL_VERSION_MINOR, OL_VERSION_PATCH)
                           .str()
                           .c_str());
  }
  case OL_PLATFORM_INFO_BACKEND: {
    return ReturnValue(Platform->BackendType);
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

  ReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  // Find the info if it exists under any of the given names
  auto GetInfo = [&](std::vector<std::string> Names) {
    if (Device == HostDevice())
      return std::string("Host");

    if (!Device->Device)
      return std::string("");

    auto Info = Device->Device->obtainInfoImpl();
    if (auto Err = Info.takeError())
      return std::string("");

    for (auto Name : Names) {
      if (auto Entry = Info->get(Name))
        return (*Entry)->Value;
    }

    return std::string("");
  };

  switch (PropName) {
  case OL_DEVICE_INFO_PLATFORM:
    return ReturnValue(Device->Platform);
  case OL_DEVICE_INFO_TYPE:
    return Device == HostDevice() ? ReturnValue(OL_DEVICE_TYPE_HOST)
                                  : ReturnValue(OL_DEVICE_TYPE_GPU);
  case OL_DEVICE_INFO_NAME:
    return ReturnValue(GetInfo({"Device Name"}).c_str());
  case OL_DEVICE_INFO_VENDOR:
    return ReturnValue(GetInfo({"Vendor Name"}).c_str());
  case OL_DEVICE_INFO_DRIVER_VERSION:
    return ReturnValue(
        GetInfo({"CUDA Driver Version", "HSA Runtime Version"}).c_str());
  default:
    return createOffloadError(ErrorCode::INVALID_ENUMERATION,
                              "getDeviceInfo enum '%i' is invalid", PropName);
  }

  return Error::success();
}

Error olGetDeviceInfo_impl(ol_device_handle_t Device, ol_device_info_t PropName,
                           size_t PropSize, void *PropValue) {
  return olGetDeviceInfoImplDetail(Device, PropName, PropSize, PropValue,
                                   nullptr);
}

Error olGetDeviceInfoSize_impl(ol_device_handle_t Device,
                               ol_device_info_t PropName, size_t *PropSizeRet) {
  return olGetDeviceInfoImplDetail(Device, PropName, 0, nullptr, PropSizeRet);
}

Error olIterateDevices_impl(ol_device_iterate_cb_t Callback, void *UserData) {
  for (auto &Platform : Platforms()) {
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
  allocInfoMap().insert_or_assign(*Alloc, AllocInfo{Device, Type});
  return Error::success();
}

Error olMemFree_impl(void *Address) {
  if (!allocInfoMap().contains(Address))
    return createOffloadError(ErrorCode::INVALID_ARGUMENT,
                              "address is not a known allocation");

  auto AllocInfo = allocInfoMap().at(Address);
  auto Device = AllocInfo.Device;
  auto Type = AllocInfo.Type;

  if (auto Res =
          Device->Device->dataDelete(Address, convertOlToPluginAllocTy(Type)))
    return Res;

  allocInfoMap().erase(Address);

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
  if (DstDevice == HostDevice() && SrcDevice == HostDevice()) {
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

  if (DstDevice == HostDevice()) {
    if (auto Res =
            SrcDevice->Device->dataRetrieve(DstPtr, SrcPtr, Size, QueueImpl))
      return Res;
  } else if (SrcDevice == HostDevice()) {
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

  Prog->Image = *Res;
  *Program = Prog;

  return Error::success();
}

Error olDestroyProgram_impl(ol_program_handle_t Program) {
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
