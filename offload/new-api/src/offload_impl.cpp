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

offload_impl_result_t
offloadPlatformGet_impl(uint32_t NumEntries,
                        offload_platform_handle_t *phPlatforms,
                        uint32_t *pNumPlatforms) {
  // It is expected that offloadPlatformGet is the first function to be called.
  // In future it may make sense to have a specific entry point for Offload
  // initialization, or expose explicit initialization of plugins.
  static std::once_flag InitFlag;
  std::call_once(InitFlag, initPlugins);

  if (NumEntries > Platforms().size()) {
    return {OFFLOAD_ERRC_INVALID_SIZE,
            std::string{formatv("{0} platform(s) available but {1} requested.",
                                Platforms().size(), NumEntries)}};
  }

  if (phPlatforms) {
    for (uint32_t PlatformIndex = 0; PlatformIndex < NumEntries;
         PlatformIndex++) {
      phPlatforms[PlatformIndex] = &(Platforms())[PlatformIndex];
    }
  }

  if (pNumPlatforms) {
    *pNumPlatforms = Platforms().size();
  }

  return OFFLOAD_SUCCESS;
}

offload_impl_result_t
offloadPlatformGetInfo_impl(offload_platform_handle_t hPlatform,
                            offload_platform_info_t propName, size_t propSize,
                            void *pPropValue, size_t *pPropSizeRet) {
  ReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case OFFLOAD_PLATFORM_INFO_NAME:
    return ReturnValue(hPlatform->Plugin->getName());
  case OFFLOAD_PLATFORM_INFO_VENDOR_NAME:
    // TODO: Implement this
    return ReturnValue("Unknown platform vendor");
  case OFFLOAD_PLATFORM_INFO_VERSION: {
    // TODO: Implement this
    return ReturnValue("v0.0.0");
  }
  case OFFLOAD_PLATFORM_INFO_BACKEND: {
    auto PluginName = hPlatform->Plugin->getName();
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

  return OFFLOAD_SUCCESS;
}

offload_impl_result_t offloadDeviceGet_impl(offload_platform_handle_t hPlatform,
                                            offload_device_type_t,
                                            uint32_t NumEntries,
                                            offload_device_handle_t *phDevices,
                                            uint32_t *pNumDevices) {

  if (phDevices) {
    for (uint32_t DeviceIndex = 0; DeviceIndex < NumEntries; DeviceIndex++) {
      phDevices[DeviceIndex] = &(hPlatform->Devices[DeviceIndex]);
    }
  }

  if (pNumDevices) {
    *pNumDevices = static_cast<uint32_t>(hPlatform->Devices.size());
  }

  return OFFLOAD_SUCCESS;
}

offload_impl_result_t offloadDeviceGetInfo_impl(offload_device_handle_t hDevice,
                                                offload_device_info_t propName,
                                                size_t propSize,
                                                void *pPropValue,
                                                size_t *pPropSizeRet) {

  ReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  InfoQueueTy DevInfo;
  if (auto Err = hDevice->Device.obtainInfoImpl(DevInfo))
    return OFFLOAD_ERRC_OUT_OF_RESOURCES;

  // Find the info if it exists under any of the given names
  auto GetInfo = [&DevInfo](std::vector<std::string> Names) {
    for (auto Name : Names) {
      auto InfoKeyMatches = [&](const InfoQueueTy::InfoQueueEntryTy &info) {
        return info.Key == Name;
      };
      auto Item = std::find_if(DevInfo.getQueue().begin(),
                               DevInfo.getQueue().end(), InfoKeyMatches);

      if (Item != std::end(DevInfo.getQueue())) {
        return Item->Value;
      }
    }

    return std::string("");
  };

  switch (propName) {
  case OFFLOAD_DEVICE_INFO_PLATFORM:
    return ReturnValue(hDevice->Platform);
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

  return OFFLOAD_SUCCESS;
}
