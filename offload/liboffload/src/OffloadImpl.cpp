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

using namespace llvm;
using namespace llvm::omp::target::plugin;

// Handle type definitions. Ideally these would be 1:1 with the plugins
struct ol_device_handle_t_ {
  int DeviceNum;
  GenericDeviceTy &Device;
  ol_platform_handle_t Platform;
};

struct ol_platform_handle_t_ {
  std::unique_ptr<GenericPluginTy> Plugin;
  std::vector<ol_device_handle_t_> Devices;
};

struct ol_queue_handle_t_ {
  __tgt_async_info *AsyncInfo;
  ol_device_handle_t Device;
  std::atomic_uint32_t RefCount;
};

struct ol_event_handle_t_ {
  void *EventInfo;
  ol_queue_handle_t Queue;
  ol_device_handle_t Device;
  std::atomic_uint32_t RefCount;
};

struct ol_program_handle_t_ {
  llvm::omp::target::plugin::DeviceImageTy *Image;
  std::unique_ptr<MemoryBuffer> ImageData;
  std::atomic_uint32_t RefCount;
};

struct OffloadArguments {
  static constexpr size_t MaxParamBytes = 4000u;
  using args_t = std::array<char, MaxParamBytes>;
  using args_size_t = std::vector<size_t>;
  using args_index_t = std::vector<void *>;
  args_t Storage;
  args_size_t ParamSizes;
  args_index_t Indices;
  args_size_t OffsetPerIndex;

  std::uint32_t ImplicitOffsetArgs[3] = {0, 0, 0};

  // NOTE:
  // This implementation is an exact copy of the CUDA adapter's argument
  // implementation. Even though it was designed for CUDA, the design of
  // libomptarget means it should work for other plugins as they will expect
  // the same argument layout.
  OffloadArguments() {
    // Place the implicit offset index at the end of the indicies collection
    Indices.emplace_back(&ImplicitOffsetArgs);
  }

  /// Add an argument to the kernel.
  /// If the argument existed before, it is replaced.
  /// Otherwise, it is added.
  /// Gaps are filled with empty arguments.
  /// Implicit offset argument is kept at the back of the indices collection.
  void addArg(size_t Index, size_t Size, const void *Arg,
              size_t LocalSize = 0) {
    if (Index + 2 > Indices.size()) {
      // Move implicit offset argument index with the end
      Indices.resize(Index + 2, Indices.back());
      // Ensure enough space for the new argument
      ParamSizes.resize(Index + 1);
      OffsetPerIndex.resize(Index + 1);
    }
    ParamSizes[Index] = Size;
    // calculate the insertion point on the array
    size_t InsertPos = std::accumulate(std::begin(ParamSizes),
                                       std::begin(ParamSizes) + Index, 0);
    // Update the stored value for the argument
    std::memcpy(&Storage[InsertPos], Arg, Size);
    Indices[Index] = &Storage[InsertPos];
    OffsetPerIndex[Index] = LocalSize;
  }

  void addLocalArg(size_t Index, size_t Size) {
    size_t LocalOffset = this->getLocalSize();

    // maximum required alignment is the size of the largest vector type
    const size_t MaxAlignment = sizeof(double) * 16;

    // for arguments smaller than the maximum alignment simply align to the
    // size of the argument
    const size_t Alignment = std::min(MaxAlignment, Size);

    // align the argument
    size_t AlignedLocalOffset = LocalOffset;
    size_t Pad = LocalOffset % Alignment;
    if (Pad != 0) {
      AlignedLocalOffset += Alignment - Pad;
    }

    addArg(Index, sizeof(size_t), (const void *)&(AlignedLocalOffset),
           Size + (AlignedLocalOffset - LocalOffset));
  }

  void setImplicitOffset(size_t Size, std::uint32_t *ImplicitOffset) {
    assert(Size == sizeof(std::uint32_t) * 3);
    std::memcpy(ImplicitOffsetArgs, ImplicitOffset, Size);
  }

  void clearLocalSize() {
    std::fill(std::begin(OffsetPerIndex), std::end(OffsetPerIndex), 0);
  }

  const args_index_t &getIndices() const noexcept { return Indices; }

  uint32_t getLocalSize() const {
    return std::accumulate(std::begin(OffsetPerIndex), std::end(OffsetPerIndex),
                           0);
  }

  const char *getStorage() const noexcept { return Storage.data(); }
};

struct ol_kernel_handle_t_ {
  ol_program_handle_t Program;
  std::atomic_uint32_t RefCount;
  GenericKernelTy *KernelImpl;
  OffloadArguments Args;
};

using PlatformVecT = SmallVector<ol_platform_handle_t_, 4>;
PlatformVecT &Platforms() {
  static PlatformVecT Platforms;
  return Platforms;
}

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

// Every plugin exports this method to create an instance of the plugin type.
#define PLUGIN_TARGET(Name) extern "C" GenericPluginTy *createPlugin_##Name();
#include "Shared/Targets.def"

void initPlugins() {
  // Attempt to create an instance of each supported plugin.
#define PLUGIN_TARGET(Name)                                                    \
  do {                                                                         \
    Platforms().emplace_back(ol_platform_handle_t_{                            \
        std::unique_ptr<GenericPluginTy>(createPlugin_##Name()), {}});         \
  } while (false);
#include "Shared/Targets.def"

  // Preemptively initialize all devices in the plugin so we can just return
  // them from deviceGet
  for (auto &Platform : Platforms()) {
    auto Err = Platform.Plugin->init();
    [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
    for (auto DevNum = 0; DevNum < Platform.Plugin->number_of_devices();
         DevNum++) {
      if (Platform.Plugin->init_device(DevNum) == OFFLOAD_SUCCESS) {
        Platform.Devices.emplace_back(ol_device_handle_t_{
            DevNum, Platform.Plugin->getDevice(DevNum), &Platform});
      }
    }
  }

  offloadConfig().TracingEnabled = std::getenv("OFFLOAD_TRACE");
}

// TODO: We can properly reference count here and manage the resources in a more
// clever way
ol_impl_result_t olInit_impl() {
  static std::once_flag InitFlag;
  std::call_once(InitFlag, initPlugins);

  return OL_SUCCESS;
}
ol_impl_result_t olShutDown_impl() { return OL_SUCCESS; }

ol_impl_result_t olGetPlatformCount_impl(uint32_t *NumPlatforms) {
  *NumPlatforms = Platforms().size();
  return OL_SUCCESS;
}

ol_impl_result_t olGetPlatform_impl(uint32_t NumEntries,
                                    ol_platform_handle_t *PlatformsOut) {
  if (NumEntries > Platforms().size()) {
    return {OL_ERRC_INVALID_SIZE,
            std::string{formatv("{0} platform(s) available but {1} requested.",
                                Platforms().size(), NumEntries)}};
  }

  for (uint32_t PlatformIndex = 0; PlatformIndex < NumEntries;
       PlatformIndex++) {
    PlatformsOut[PlatformIndex] = &(Platforms())[PlatformIndex];
  }

  return OL_SUCCESS;
}

ol_impl_result_t olGetPlatformInfoImplDetail(ol_platform_handle_t Platform,
                                             ol_platform_info_t PropName,
                                             size_t PropSize, void *PropValue,
                                             size_t *PropSizeRet) {
  ReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  switch (PropName) {
  case OL_PLATFORM_INFO_NAME:
    return ReturnValue(Platform->Plugin->getName());
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
    auto PluginName = Platform->Plugin->getName();
    if (PluginName == StringRef("CUDA")) {
      return ReturnValue(OL_PLATFORM_BACKEND_CUDA);
    } else if (PluginName == StringRef("AMDGPU")) {
      return ReturnValue(OL_PLATFORM_BACKEND_AMDGPU);
    } else {
      return ReturnValue(OL_PLATFORM_BACKEND_UNKNOWN);
    }
  }
  default:
    return OL_ERRC_INVALID_ENUMERATION;
  }

  return OL_SUCCESS;
}

ol_impl_result_t olGetPlatformInfo_impl(ol_platform_handle_t Platform,
                                        ol_platform_info_t PropName,
                                        size_t PropSize, void *PropValue) {
  return olGetPlatformInfoImplDetail(Platform, PropName, PropSize, PropValue,
                                     nullptr);
}

ol_impl_result_t olGetPlatformInfoSize_impl(ol_platform_handle_t Platform,
                                            ol_platform_info_t PropName,
                                            size_t *PropSizeRet) {
  return olGetPlatformInfoImplDetail(Platform, PropName, 0, nullptr,
                                     PropSizeRet);
}

ol_impl_result_t olGetDeviceCount_impl(ol_platform_handle_t Platform,
                                       uint32_t *pNumDevices) {
  *pNumDevices = static_cast<uint32_t>(Platform->Devices.size());

  return OL_SUCCESS;
}

ol_impl_result_t olGetDevice_impl(ol_platform_handle_t Platform,
                                  uint32_t NumEntries,
                                  ol_device_handle_t *Devices) {
  if (NumEntries > Platform->Devices.size()) {
    return OL_ERRC_INVALID_SIZE;
  }

  for (uint32_t DeviceIndex = 0; DeviceIndex < NumEntries; DeviceIndex++) {
    Devices[DeviceIndex] = &(Platform->Devices[DeviceIndex]);
  }

  return OL_SUCCESS;
}

ol_impl_result_t olGetDeviceInfoImplDetail(ol_device_handle_t Device,
                                           ol_device_info_t PropName,
                                           size_t PropSize, void *PropValue,
                                           size_t *PropSizeRet) {

  ReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  InfoQueueTy DevInfo;
  if (auto Err = Device->Device.obtainInfoImpl(DevInfo))
    return OL_ERRC_OUT_OF_RESOURCES;

  // Find the info if it exists under any of the given names
  auto GetInfo = [&DevInfo](std::vector<std::string> Names) {
    for (auto Name : Names) {
      auto InfoKeyMatches = [&](const InfoQueueTy::InfoQueueEntryTy &Info) {
        return Info.Key == Name;
      };
      auto Item = std::find_if(DevInfo.getQueue().begin(),
                               DevInfo.getQueue().end(), InfoKeyMatches);

      if (Item != std::end(DevInfo.getQueue())) {
        return Item->Value;
      }
    }

    return std::string("");
  };

  switch (PropName) {
  case OL_DEVICE_INFO_PLATFORM:
    return ReturnValue(Device->Platform);
  case OL_DEVICE_INFO_TYPE:
    return ReturnValue(OL_DEVICE_TYPE_GPU);
  case OL_DEVICE_INFO_NAME:
    return ReturnValue(GetInfo({"Device Name"}).c_str());
  case OL_DEVICE_INFO_VENDOR:
    return ReturnValue(GetInfo({"Vendor Name"}).c_str());
  case OL_DEVICE_INFO_DRIVER_VERSION:
    return ReturnValue(
        GetInfo({"CUDA Driver Version", "HSA Runtime Version"}).c_str());
  default:
    return OL_ERRC_INVALID_ENUMERATION;
  }

  return OL_SUCCESS;
}

ol_impl_result_t olGetDeviceInfo_impl(ol_device_handle_t Device,
                                      ol_device_info_t PropName,
                                      size_t PropSize, void *PropValue) {
  return olGetDeviceInfoImplDetail(Device, PropName, PropSize, PropValue,
                                   nullptr);
}

ol_impl_result_t olGetDeviceInfoSize_impl(ol_device_handle_t Device,
                                          ol_device_info_t PropName,
                                          size_t *PropSizeRet) {
  return olGetDeviceInfoImplDetail(Device, PropName, 0, nullptr, PropSizeRet);
}

TargetAllocTy convertOlToPluginAllocTy(ol_alloc_type_t Type) {
  switch (Type) {
  case OL_ALLOC_TYPE_DEVICE:
    return TARGET_ALLOC_DEVICE;
  case OL_ALLOC_TYPE_HOST:
    return TARGET_ALLOC_HOST;
  case OL_ALLOC_TYPE_SHARED:
  default:
    return TARGET_ALLOC_SHARED;
  }
}

ol_impl_result_t olMemAlloc_impl(ol_device_handle_t Device,
                                 ol_alloc_type_t Type, size_t Size, size_t,
                                 void **AllocationOut) {
  auto Alloc =
      Device->Device.dataAlloc(Size, nullptr, convertOlToPluginAllocTy(Type));
  if (!Alloc) {
    return {OL_ERRC_OUT_OF_RESOURCES,
            formatv("Could not create allocation on device {0}", Device).str()};
  }

  *AllocationOut = *Alloc;
  return OL_SUCCESS;
}

ol_impl_result_t olMemFree_impl(ol_device_handle_t Device, ol_alloc_type_t Type,
                                void *Address) {
  auto Res = Device->Device.dataDelete(Address, convertOlToPluginAllocTy(Type));
  if (Res) {
    return {OL_ERRC_OUT_OF_RESOURCES, "Could not free allocation"};
  }
  return OL_SUCCESS;
}

ol_impl_result_t olCreateQueue_impl(ol_device_handle_t Device,
                                    ol_queue_handle_t *Queue) {
  auto CreatedQueue = std::make_unique<ol_queue_handle_t_>();
  auto Err = Device->Device.initAsyncInfo(&(CreatedQueue->AsyncInfo));
  if (Err) {
    return OL_ERRC_OUT_OF_RESOURCES;
  }
  // TODO: Check error
  CreatedQueue->Device = Device;
  CreatedQueue->RefCount = 1;
  *Queue = CreatedQueue.release();
  return OL_SUCCESS;
}

ol_impl_result_t olRetainQueue_impl(ol_queue_handle_t Queue) {
  Queue->RefCount++;
  return OL_SUCCESS;
}

ol_impl_result_t olReleaseQueue_impl(ol_queue_handle_t Queue) {
  Queue->RefCount--;
  return OL_SUCCESS;
}

ol_impl_result_t olFinishQueue_impl(ol_queue_handle_t Queue) {
  // Host plugin doesn't have a queue set so it's not safe to call synchronize
  // on it, but we have nothing to synchronize in that situation anyway.
  if (Queue->AsyncInfo->Queue) {
    auto Err = Queue->Device->Device.synchronize(Queue->AsyncInfo);
    if (Err) {
      return OL_ERRC_OUT_OF_RESOURCES;
    }
  }

  // Recreate the stream resource so the queue can be reused
  // TODO: Would be easier for the synchronization to (optionally) not release
  // it to begin with.
  auto Res = Queue->Device->Device.initAsyncInfo(&Queue->AsyncInfo);
  if (Res) {
    return OL_ERRC_OUT_OF_RESOURCES;
  }

  return OL_SUCCESS;
}

ol_impl_result_t olWaitEvent_impl(ol_event_handle_t Event) {
  auto Res = Event->Device->Device.syncEvent(Event->EventInfo);
  if (Res) {
    return OL_ERRC_OUT_OF_RESOURCES;
  }
  return OL_SUCCESS;
}

ol_impl_result_t olRetainEvent_impl(ol_event_handle_t Event) {
  Event->RefCount++;
  return OL_SUCCESS;
}

ol_impl_result_t olReleaseEvent_impl(ol_event_handle_t Event) {
  Event->RefCount--;
  return OL_SUCCESS;
}

ol_event_handle_t makeEvent(ol_queue_handle_t Queue) {
  auto EventImpl = std::make_unique<ol_event_handle_t_>();
  EventImpl->Queue = Queue;
  auto Res = Queue->Device->Device.createEvent(&EventImpl->EventInfo);
  if (Res) {
    return nullptr;
  }
  Res =
      Queue->Device->Device.recordEvent(EventImpl->EventInfo, Queue->AsyncInfo);
  if (Res) {
    return nullptr;
  }

  return EventImpl.release();
}

ol_impl_result_t olEnqueueDataWrite_impl(ol_queue_handle_t Queue, void *SrcPtr,
                                         void *DstPtr, size_t Size,
                                         ol_event_handle_t *EventOut) {
  auto &DeviceImpl = Queue->Device->Device;

  auto Res = DeviceImpl.dataSubmit(DstPtr, SrcPtr, Size, Queue->AsyncInfo);

  if (Res) {
    return OL_ERRC_OUT_OF_RESOURCES;
  }

  if (EventOut) {
    *EventOut = makeEvent(Queue);
  }

  return OL_SUCCESS;
}

ol_impl_result_t olEnqueueDataRead_impl(ol_queue_handle_t Queue, void *SrcPtr,
                                        void *DstPtr, size_t Size,
                                        ol_event_handle_t *EventOut) {
  auto &DeviceImpl = Queue->Device->Device;

  auto Res = DeviceImpl.dataRetrieve(DstPtr, SrcPtr, Size, Queue->AsyncInfo);

  if (Res) {
    return OL_ERRC_OUT_OF_RESOURCES;
  }

  if (EventOut) {
    *EventOut = makeEvent(Queue);
  }

  return OL_SUCCESS;
}

ol_impl_result_t olEnqueueDataCopy_impl(ol_queue_handle_t Queue, void *SrcPtr,
                                        void *DstPtr,
                                        ol_device_handle_t DstDevice,
                                        size_t Size,
                                        ol_event_handle_t *EventOut) {
  auto &DeviceImpl = Queue->Device->Device;

  auto Res = DeviceImpl.dataExchange(SrcPtr, DstDevice->Device, DstPtr, Size,
                                     Queue->AsyncInfo);

  if (Res) {
    return OL_ERRC_OUT_OF_RESOURCES;
  }

  if (EventOut) {
    *EventOut = makeEvent(Queue);
  }

  return OL_SUCCESS;
}

ol_impl_result_t olCreateProgram_impl(ol_device_handle_t Device, void *ProgData,
                                      size_t ProgDataSize,
                                      ol_program_handle_t *Program) {
  auto ImageData = MemoryBuffer::getMemBufferCopy(
      StringRef(reinterpret_cast<char *>(ProgData), ProgDataSize));
  __tgt_device_image DeviceImage{(char *) ImageData->getBuffer().data(),
                                 ((char *)ImageData->getBuffer().data()) +
                                     ProgDataSize - 1,
                                 nullptr, nullptr};

  ol_program_handle_t Prog = new ol_program_handle_t_();

  auto Res = Device->Device.loadBinary(Device->Device.Plugin, &DeviceImage);
  if (!Res)
    return OL_ERRC_INVALID_VALUE;

  Prog->Image = *Res;
  Prog->RefCount = 1;
  Prog->ImageData = std::move(ImageData);
  *Program = Prog;

  return OL_SUCCESS;
}

ol_impl_result_t olRetainProgram_impl(ol_program_handle_t Program) {
  ++Program->RefCount;
  return OL_SUCCESS;
}

ol_impl_result_t olReleaseProgram_impl(ol_program_handle_t Program) {
  if (--Program->RefCount == 0) {
    delete Program;
  }
  return OL_SUCCESS;
}

ol_impl_result_t olCreateKernel_impl(ol_program_handle_t Program,
                                     const char *KernelName,
                                     ol_kernel_handle_t *Kernel) {

  auto &Device = Program->Image->getDevice();
  auto KernelImpl = Device.constructKernel(KernelName);
  if (!KernelImpl) {
    return OL_ERRC_OUT_OF_RESOURCES;
  }

  auto Err = KernelImpl->init(Device, *Program->Image);
  if (Err) {
    return OL_ERRC_OUT_OF_RESOURCES;
  }

  ol_kernel_handle_t CreatedKernel = new ol_kernel_handle_t_();
  CreatedKernel->Program = Program;
  CreatedKernel->RefCount = 1;
  CreatedKernel->KernelImpl = &*KernelImpl;
  *Kernel = CreatedKernel;

  return OL_SUCCESS;
}

ol_impl_result_t olRetainKernel_impl(ol_kernel_handle_t Kernel) {
  Kernel->RefCount++;
  return OL_SUCCESS;
}

ol_impl_result_t olReleaseKernel_impl(ol_kernel_handle_t Kernel) {
  if (--Kernel->RefCount == 0) {
    delete Kernel;
  }
  return OL_SUCCESS;
}

ol_impl_result_t olSetKernelArgValue_impl(ol_kernel_handle_t Kernel,
                                          uint32_t Index, size_t Size,
                                          void *ArgData) {
  Kernel->Args.addArg(Index, Size, ArgData);

  return OL_SUCCESS;
}

ol_impl_result_t olEnqueueKernelLaunch_impl(ol_queue_handle_t Queue,
                                            ol_kernel_handle_t Kernel,
                                            const size_t *GlobalWorkSize,
                                            ol_event_handle_t *EventOut) {
  auto &DeviceImpl = Queue->Device->Device;

  AsyncInfoWrapperTy AsyncInfoWrapper(DeviceImpl, Queue->AsyncInfo);

  KernelArgsTy LaunchArgs{};
  LaunchArgs.NumArgs = Kernel->Args.getIndices().size() - 1; // TODO
  LaunchArgs.NumTeams[0] = GlobalWorkSize[0];
  LaunchArgs.NumTeams[1] = 1;
  LaunchArgs.NumTeams[2] = 1;
  LaunchArgs.ThreadLimit[0] = 1;
  LaunchArgs.ThreadLimit[1] = 1;
  LaunchArgs.ThreadLimit[2] = 1;

  LaunchArgs.ArgPtrs = (void **)Kernel->Args.getStorage();

  // TODO: Verify this
  auto ArgOffsets = std::vector<ptrdiff_t>(LaunchArgs.NumArgs, 0ul);

  auto Err = Kernel->KernelImpl->launch(
      DeviceImpl, (void **)Kernel->Args.getStorage(), ArgOffsets.data(),
      LaunchArgs, AsyncInfoWrapper);

  AsyncInfoWrapper.finalize(Err);
  if (Err) {
    return OL_ERRC_OUT_OF_RESOURCES;
  }

  if (EventOut) {
    *EventOut = makeEvent(Queue);
  }

  return OL_SUCCESS;
}
