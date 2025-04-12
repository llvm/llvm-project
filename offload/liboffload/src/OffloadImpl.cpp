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
