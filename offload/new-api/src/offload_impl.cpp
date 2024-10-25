//===- offload_impl.cpp - Implementation of the new LLVM/Offload API ------===//
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

#include "offload_impl.hpp"
#include "PluginManager.h"
#include "helpers.hpp"
#include "llvm/Support/FormatVariadic.h"
#include <offload_api.h>

#include <mutex>

using namespace llvm;
using namespace llvm::omp::target::plugin;

// Handle type definitions. Ideally these would be 1:1 with the plugins
struct offload_device_handle_t_ {
  int DeviceNum;
  GenericDeviceTy &Device;
  offload_platform_handle_t Platform;
};

struct offload_platform_handle_t_ {
  std::unique_ptr<GenericPluginTy> Plugin;
  std::vector<offload_device_handle_t_> Devices;
};

std::vector<offload_platform_handle_t_> &Platforms() {
  static std::vector<offload_platform_handle_t_> Platforms;
  return Platforms;
}

// Every plugin exports this method to create an instance of the plugin type.
#define PLUGIN_TARGET(Name) extern "C" GenericPluginTy *createPlugin_##Name();
#include "Shared/Targets.def"

void initPlugins() {
  // Attempt to create an instance of each supported plugin.
#define PLUGIN_TARGET(Name)                                                    \
  do {                                                                         \
    Platforms().emplace_back(offload_platform_handle_t_{                       \
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
        Platform.Devices.emplace_back(offload_device_handle_t_{
            DevNum, Platform.Plugin->getDevice(DevNum), &Platform});
      }
    }
  }
}

// TODO: We can properly reference count here and manage the resources in a more
// clever way
offload_impl_result_t offloadInit_impl() {
  static std::once_flag InitFlag;
  std::call_once(InitFlag, initPlugins);

  return OFFLOAD_RESULT_SUCCESS;
}
offload_impl_result_t offloadShutDown_impl() { return OFFLOAD_RESULT_SUCCESS; }

offload_impl_result_t offloadPlatformGetCount_impl(uint32_t *NumPlatforms) {
  // It is expected that offloadPlatformGet is the first function to be called.
  // In future it may make sense to have a specific entry point for Offload
  // initialization, or expose explicit initialization of plugins.
  *NumPlatforms = Platforms().size();
  return OFFLOAD_RESULT_SUCCESS;
}

offload_impl_result_t
offloadPlatformGet_impl(uint32_t NumEntries,
                        offload_platform_handle_t *PlatformsOut) {
  if (NumEntries > Platforms().size()) {
    return {OFFLOAD_ERRC_INVALID_SIZE,
            std::string{formatv("{0} platform(s) available but {1} requested.",
                                Platforms().size(), NumEntries)}};
  }

  for (uint32_t PlatformIndex = 0; PlatformIndex < NumEntries;
       PlatformIndex++) {
    PlatformsOut[PlatformIndex] = &(Platforms())[PlatformIndex];
  }

  return OFFLOAD_RESULT_SUCCESS;
}

offload_impl_result_t offloadPlatformGetInfoImplDetail(
    offload_platform_handle_t Platform, offload_platform_info_t PropName,
    size_t PropSize, void *PropValue, size_t *PropSizeRet) {
  ReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  switch (PropName) {
  case OFFLOAD_PLATFORM_INFO_NAME:
    return ReturnValue(Platform->Plugin->getName());
  case OFFLOAD_PLATFORM_INFO_VENDOR_NAME:
    // TODO: Implement this
    return ReturnValue("Unknown platform vendor");
  case OFFLOAD_PLATFORM_INFO_VERSION: {
    // TODO: Implement this
    return ReturnValue("v0.0.0");
  }
  case OFFLOAD_PLATFORM_INFO_BACKEND: {
    auto PluginName = Platform->Plugin->getName();
    if (PluginName == StringRef("CUDA")) {
      return ReturnValue(OFFLOAD_PLATFORM_BACKEND_CUDA);
    } else if (PluginName == StringRef("AMDGPU")) {
      return ReturnValue(OFFLOAD_PLATFORM_BACKEND_AMDGPU);
    } else {
      return ReturnValue(OFFLOAD_PLATFORM_BACKEND_UNKNOWN);
    }
  }
  default:
    return OFFLOAD_ERRC_INVALID_ENUMERATION;
  }

  return OFFLOAD_RESULT_SUCCESS;
}

offload_impl_result_t
offloadPlatformGetInfo_impl(offload_platform_handle_t Platform,
                            offload_platform_info_t PropName, size_t PropSize,
                            void *PropValue) {
  return offloadPlatformGetInfoImplDetail(Platform, PropName, PropSize,
                                          PropValue, nullptr);
}

offload_impl_result_t
offloadPlatformGetInfoSize_impl(offload_platform_handle_t Platform,
                                offload_platform_info_t PropName,
                                size_t *PropSizeRet) {
  return offloadPlatformGetInfoImplDetail(Platform, PropName, 0, nullptr,
                                          PropSizeRet);
}

offload_impl_result_t
offloadDeviceGetCount_impl(offload_platform_handle_t Platform,
                           uint32_t *pNumDevices) {
  *pNumDevices = static_cast<uint32_t>(Platform->Devices.size());

  return OFFLOAD_RESULT_SUCCESS;
}

offload_impl_result_t offloadDeviceGet_impl(offload_platform_handle_t Platform,
                                            uint32_t NumEntries,
                                            offload_device_handle_t *Devices) {
  if (NumEntries > Platform->Devices.size()) {
    return OFFLOAD_ERRC_INVALID_SIZE;
  }

  for (uint32_t DeviceIndex = 0; DeviceIndex < NumEntries; DeviceIndex++) {
    Devices[DeviceIndex] = &(Platform->Devices[DeviceIndex]);
  }

  return OFFLOAD_RESULT_SUCCESS;
}

offload_impl_result_t
offloadDeviceGetInfoImplDetail(offload_device_handle_t Device,
                               offload_device_info_t PropName, size_t PropSize,
                               void *PropValue, size_t *PropSizeRet) {

  ReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  InfoQueueTy DevInfo;
  if (auto Err = Device->Device.obtainInfoImpl(DevInfo))
    return OFFLOAD_ERRC_OUT_OF_RESOURCES;

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
  case OFFLOAD_DEVICE_INFO_PLATFORM:
    return ReturnValue(Device->Platform);
  case OFFLOAD_DEVICE_INFO_TYPE:
    return ReturnValue(OFFLOAD_DEVICE_TYPE_GPU);
  case OFFLOAD_DEVICE_INFO_NAME:
    return ReturnValue(GetInfo({"Device Name"}).c_str());
  case OFFLOAD_DEVICE_INFO_VENDOR:
    return ReturnValue(GetInfo({"Vendor Name"}).c_str());
  case OFFLOAD_DEVICE_INFO_DRIVER_VERSION:
    return ReturnValue(
        GetInfo({"CUDA Driver Version", "HSA Runtime Version"}).c_str());
  default:
    return OFFLOAD_ERRC_INVALID_ENUMERATION;
  }

  return OFFLOAD_RESULT_SUCCESS;
}

offload_impl_result_t offloadDeviceGetInfo_impl(offload_device_handle_t Device,
                                                offload_device_info_t PropName,
                                                size_t PropSize,
                                                void *PropValue) {
  return offloadDeviceGetInfoImplDetail(Device, PropName, PropSize, PropValue,
                                        nullptr);
}

offload_impl_result_t
offloadDeviceGetInfoSize_impl(offload_device_handle_t Device,
                              offload_device_info_t PropName,
                              size_t *PropSizeRet) {
  return offloadDeviceGetInfoImplDetail(Device, PropName, 0, nullptr,
                                        PropSizeRet);
}
