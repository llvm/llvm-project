//===- DeviceManager.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceManager.h"
#include "PluginManager.h"
#include "openacc.h"

// OpenACC 3.4, sec. 2.3.1 "Modifying and Retrieving ICV Values"
// Each host thread needs its own value, thus these are `thread_local`.
//
// The DeviceManager owns these ICVs and they should not be accessible outside
// and are thus static.
namespace llvm::acc::target::icv {
/// OpenACC 3.4, sec. 2.3 "Internal Control Variables"
/// "acc-current-device-num-var - controls which device of the selected type is
/// used."
/// TODO can we use PerThreadTable here?
static thread_local std::array<DeviceManagerTy::DeviceIdTy,
                               AccDeviceNumConcreteTypes>
    AccCurrentDeviceNumVar = {0};
/// OpenACC 3.4, sec. 2.3 "Internal Control Variables"
/// "acc-current-device-type-var - controls which type of device is used."
static thread_local acc_device_t AccCurrentDeviceTypeVar = acc_device_default;
/// The device type to use when the default is asked for. Initially we set it to
/// none. When the plugins get initialized we will set the default to one of the
/// target device types we have available.
static acc_device_t AccCurrentDefaultDeviceTypeVar = acc_device_none;

} // namespace llvm::acc::target::icv

namespace llvm::acc::target {
DeviceManagerTy *DM = nullptr;
} // namespace llvm::acc::target

using namespace llvm::acc::target;

static const char *accDeviceToStr(acc_device_t DeviceType) {
  switch (DeviceType) {
  case acc_device_nvidia:
    return "nvidia";
  case acc_device_amd:
    return "amd";
  case acc_device_spirv:
    return "spirv";
  case acc_device_none:
    return "<none>";
  case acc_device_default:
    return "<default>";
  case acc_device_host:
    return "<host>";
  case acc_device_not_host:
    return "<not_host>";
  default:
    return "<unknown>";
  }
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     acc_device_t DeviceType) {
  return OS << accDeviceToStr(DeviceType) << " ("
            << static_cast<int>(DeviceType) << ")";
}

DeviceManagerTy::SingleDeviceTypeMapTy &
DeviceManagerTy::getSingleDeviceTypeMap(acc_device_t DeviceType) {
  return PMDeviceMap[DeviceType - AccDeviceTypeOffset];
}

void DeviceManagerTy::init() {
  refreshDeviceMapping(/*UpdateDeviceType=*/true);
}

void DeviceManagerTy::deinit() {}

void DeviceManagerTy::refreshDeviceMapping(bool UpdateDeviceType) {
  assert(this);

  for (int DeviceTypeInt = acc_device_concrete_type_begin;
       DeviceTypeInt < acc_device_concrete_type_end; DeviceTypeInt++)
    getSingleDeviceTypeMap(acc_device_nvidia).resize(0);

  auto ExclusiveDevicesAccessor = PM->getExclusiveDevicesAccessor();
  for (DeviceTy &Device : PM->devices(ExclusiveDevicesAccessor)) {
    if (Device.RTL->getTripleArch() == llvm::Triple::nvptx64) {
      getSingleDeviceTypeMap(acc_device_nvidia).push_back(Device.DeviceID);
    } else if (Device.RTL->getTripleArch() == llvm::Triple::amdgcn) {
      getSingleDeviceTypeMap(acc_device_amd).push_back(Device.DeviceID);
    } else if (Device.RTL->getTripleArch() == llvm::Triple::spirv64) {
      getSingleDeviceTypeMap(acc_device_spirv).push_back(Device.DeviceID);
    }
  }

  ODBG() << "Refreshed OpenACC devices:";
  for (int DeviceTypeInt = acc_device_concrete_type_begin;
       DeviceTypeInt < acc_device_concrete_type_end; DeviceTypeInt++) {
    acc_device_t DeviceType = static_cast<acc_device_t>(DeviceTypeInt);
    unsigned Num = getSingleDeviceTypeMap(DeviceType).size();
    ODBG() << "  Type " << DeviceType;
    for (unsigned I = 0; I < Num; I++) {
      ODBG() << "    OpenACC Device #" << I << " -> PM Device #"
             << getSingleDeviceTypeMap(DeviceType)[I];
    }
  }

  if (UpdateDeviceType) {
    // Set the default current device type to a device we have available in the
    // below order of priority.
    auto CheckType = [&](acc_device_t Type) {
      if (getSingleDeviceTypeMap(Type).size() > 0) {
        assert(Type >= acc_device_concrete_type_begin &&
               Type < acc_device_concrete_type_end &&
               "We should only set AccCurrentDefaultDeviceTypeVar to a "
               "concrete type");
        ODBG() << "Updating AccCurrentDefaultDeviceTypeVar to " << Type;
        icv::AccCurrentDefaultDeviceTypeVar = Type;
        return true;
      }
      return false;
    };
    false || CheckType(acc_device_nvidia) || CheckType(acc_device_amd) ||
        CheckType(acc_device_spirv) || CheckType(acc_device_host);
  }
}

int DeviceManagerTy::getPMDeviceId(acc_device_t DeviceType) {
  ODBG() << "Getting device for " << DeviceType;
  if (DeviceType == acc_device_none) {
    DeviceType = icv::AccCurrentDeviceTypeVar;
    ODBG() << "Correcting to current type " << DeviceType;
  }
  if (DeviceType == acc_device_default) {
    ODBG() << "Corrected to value of AccCurrentDefaultDeviceTypeVar: "
           << icv::AccCurrentDefaultDeviceTypeVar;
    DeviceType = icv::AccCurrentDefaultDeviceTypeVar;
  }
  ODBG() << "Current device has id " << icv::AccCurrentDeviceNumVar[DeviceType];
  checkICVs();
  return getSingleDeviceTypeMap(
      DeviceType)[icv::AccCurrentDeviceNumVar[DeviceType]];
}

int DeviceManagerTy::getPMDeviceId() {
  ODBG() << "Getting current device, type " << icv::AccCurrentDeviceTypeVar;
  checkICVs();
  return getPMDeviceId(icv::AccCurrentDeviceTypeVar);
}

int DeviceManagerTy::getDeviceId(acc_device_t DeviceType) {
  checkICVs();
  return icv::AccCurrentDeviceNumVar[DeviceType];
}

void DeviceManagerTy::checkICVs() {
  ODBG() << "acc-current-device-type = " << icv::AccCurrentDeviceTypeVar;
  for (int DeviceTypeInt = acc_device_concrete_type_begin;
       DeviceTypeInt < acc_device_concrete_type_end; DeviceTypeInt++) {
    acc_device_t DeviceType = static_cast<acc_device_t>(DeviceTypeInt);
    ODBG() << "acc-current-device-num[" << DeviceType
           << "] = " << icv::AccCurrentDeviceNumVar[DeviceType];
  }
  ODBG() << "acc-current-device-type = " << icv::AccCurrentDeviceTypeVar;
  assert(icv::AccCurrentDeviceTypeVar == acc_device_default ||
         (icv::AccCurrentDeviceTypeVar >= acc_device_concrete_type_begin &&
          icv::AccCurrentDeviceTypeVar < acc_device_concrete_type_end));
  acc_device_t DeviceType = icv::AccCurrentDeviceTypeVar;
  if (DeviceType == acc_device_default) {
    DeviceType = icv::AccCurrentDefaultDeviceTypeVar;
    ODBG() << "Corrected to value of AccCurrentDefaultDeviceTypeVar: "
           << icv::AccCurrentDefaultDeviceTypeVar;
  }
  ODBG() << icv::AccCurrentDeviceNumVar[DeviceType];
  assert(icv::AccCurrentDeviceNumVar[DeviceType] <
         static_cast<int64_t>(getSingleDeviceTypeMap(DeviceType).size()));
}

int DeviceManagerTy::getNumDevices(acc_device_t DeviceType) {
  checkICVs();
  return getSingleDeviceTypeMap(DeviceType).size();
}

void DeviceManagerTy::setAllDeviceId(int DevNum) {
  for (auto &CurrDevNum : icv::AccCurrentDeviceNumVar) {
    CurrDevNum = DevNum;
  }
  checkICVs();
}

void DeviceManagerTy::setDeviceId(acc_device_t DeviceType, int DevNum) {
  icv::AccCurrentDeviceNumVar[DeviceType] = DevNum;
  checkICVs();
}

void DeviceManagerTy::setDeviceId(int DevNum) {
  setDeviceId(icv::AccCurrentDeviceTypeVar, DevNum);
  checkICVs();
}

acc_device_t DeviceManagerTy::getDeviceType() {
  checkICVs();
  return icv::AccCurrentDeviceTypeVar;
}

void DeviceManagerTy::setDeviceType(acc_device_t DeviceType) {
  icv::AccCurrentDeviceTypeVar = DeviceType;
  checkICVs();
}

size_t DeviceManagerTy::getDeviceProperty(int, acc_device_t,
                                          acc_device_property_t) {
  REPORT_FATAL() << "device properties not yet implemented";
  return 0;
}

const char *DeviceManagerTy::getDevicePropertyString(int, acc_device_t,
                                                     acc_device_property_t) {
  REPORT_FATAL() << "device properties not yet implemented";
  return "";
}

llvm::Expected<DeviceTy &> DeviceManagerTy::getDevice(acc_device_t DeviceType) {
  return PM->getDevice(getPMDeviceId(DeviceType));
}

llvm::Expected<DeviceTy &> DeviceManagerTy::getDevice() {
  return PM->getDevice(getPMDeviceId());
}
