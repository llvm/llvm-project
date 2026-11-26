//===- DeviceManager.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceManager.h"
#include "AccPrint.h"
#include "Debug.h"
#include "PluginManager.h"
#include "RuntimeState.h"
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
static thread_local std::array<DeviceManagerTy::DeviceIdTy,
                               NumConcreteDeviceTypes>
    AccCurrentDeviceNumVar = {0};
/// OpenACC 3.4, sec. 2.3 "Internal Control Variables"
/// "acc-current-device-type-var - controls which type of device is used."
/// No value stands for acc_device_default, which is the initial one.
static thread_local std::optional<ConcreteDeviceType> AccCurrentDeviceTypeVar;
/// The device type to use when the default is asked for. Initially there is
/// none. When the plugins get initialized we will set the default to one of the
/// target device types we have available.
static std::optional<ConcreteDeviceType> DefaultDeviceType;

/// The current device number of a concrete device type.
static DeviceManagerTy::DeviceIdTy &currentDeviceNum(ConcreteDeviceType Type) {
  return AccCurrentDeviceNumVar[getIndex(Type)];
}

/// The device type that the current thread uses: the one it selected, or the
/// default one. None if there are no devices.
static std::optional<ConcreteDeviceType> currentDeviceType() {
  return AccCurrentDeviceTypeVar ? AccCurrentDeviceTypeVar : DefaultDeviceType;
}

} // namespace llvm::acc::target::icv

namespace llvm::acc::target {
DeviceManagerTy *DM = nullptr;
} // namespace llvm::acc::target

using namespace llvm::acc::target;

/// The concrete device type that \p DeviceType stands for, if there is one.
/// acc_device_default is the default device type and acc_device_current the
/// device type of the current thread, so there is none if there are no devices
/// at all. acc_device_none, acc_device_host and acc_device_not_host never
/// stand for a concrete device type. What they mean is up to the caller.
static std::optional<ConcreteDeviceType> resolve(acc_device_t DeviceType) {
  switch (DeviceType) {
  case acc_device_default:
    return icv::DefaultDeviceType;
  case acc_device_current:
    return icv::currentDeviceType();
  default:
    return toConcreteDeviceType(DeviceType);
  }
}

static llvm::Error noDevicesError(acc_device_t DeviceType) {
  std::string Msg;
  llvm::raw_string_ostream OS(Msg);
  OS << "there are no devices of type " << DeviceType;
  return llvm::createStringError(llvm::inconvertibleErrorCode(), Msg);
}

DeviceManagerTy::SingleDeviceTypeMapTy &
DeviceManagerTy::getSingleDeviceTypeMap(ConcreteDeviceType Type) {
  return PMDeviceMap[getIndex(Type)];
}

void DeviceManagerTy::init() {
  refreshDeviceMapping(/*UpdateDeviceType=*/true);
  // TODO: Apply the environment variables ACC_DEVICE_TYPE and ACC_DEVICE_NUM
  // to the ICVs (OpenACC 3.4, sec. 2.3.1 and chapter 4).
}

void DeviceManagerTy::deinit() {}

void DeviceManagerTy::refreshDeviceMapping(bool UpdateDeviceType) {
  for (SingleDeviceTypeMapTy &Map : PMDeviceMap)
    Map.clear();

  auto ExclusiveDevicesAccessor = PM->getExclusiveDevicesAccessor();
  for (DeviceTy &Device : PM->devices(ExclusiveDevicesAccessor)) {
    for (const ConcreteDeviceTypeInfo &Info : ConcreteDeviceTypes)
      if (Info.Backend == Device.getBackend())
        getSingleDeviceTypeMap(Info.Type).push_back(Device.DeviceID);
  }

  ODBG() << "Refreshed OpenACC devices:";
  for (const ConcreteDeviceTypeInfo &Info : ConcreteDeviceTypes) {
    const SingleDeviceTypeMapTy &Map = getSingleDeviceTypeMap(Info.Type);
    ODBG() << "  Type " << Info.AccType;
    for (unsigned I = 0; I < Map.size(); I++)
      ODBG() << "    OpenACC Device #" << I << " -> PM Device #" << Map[I];
  }

  if (UpdateDeviceType) {
    // Set the default current device type to a device we have available in the
    // below order of priority.  Without any device, there is no default device
    // type and device operations fail with an error.
    icv::DefaultDeviceType = std::nullopt;
    for (const ConcreteDeviceTypeInfo &Info : ConcreteDeviceTypes) {
      if (getSingleDeviceTypeMap(Info.Type).empty())
        continue;
      icv::DefaultDeviceType = Info.Type;
      break;
    }
  }
  dumpICVs();
}

int DeviceManagerTy::getPMDeviceId(ConcreteDeviceType Type) {
  const SingleDeviceTypeMapTy &Map = getSingleDeviceTypeMap(Type);
  DeviceIdTy DeviceNum = icv::currentDeviceNum(Type);
  if (DeviceNum < 0 || static_cast<size_t>(DeviceNum) >= Map.size())
    return -1;
  ODBG() << "Device " << DeviceNum << " of type " << toAccDeviceType(Type)
         << " is PM device " << Map[DeviceNum];
  return Map[DeviceNum];
}

int DeviceManagerTy::getPMDeviceId(acc_device_t DeviceType) {
  // The device of a device operation without a device type is the current one.
  if (DeviceType == acc_device_none)
    DeviceType = acc_device_current;
  std::optional<ConcreteDeviceType> Type = resolve(DeviceType);
  return Type ? getPMDeviceId(*Type) : -1;
}

int DeviceManagerTy::getPMDeviceId() {
  return getPMDeviceId(acc_device_current);
}

int DeviceManagerTy::getDeviceNum(acc_device_t DeviceType) {
  // OpenACC 3.4, sec. 3.2.5: if there are no devices of this type, or the type
  // is not supported, return -1.
  std::optional<ConcreteDeviceType> Type = resolve(DeviceType);
  if (!Type || getSingleDeviceTypeMap(*Type).empty())
    return -1;
  return icv::currentDeviceNum(*Type);
}

void DeviceManagerTy::dumpICVs() {
  ODBG_OS([&](llvm::raw_ostream &OS) {
    OS << "acc-current-device-type="
       << (icv::AccCurrentDeviceTypeVar
               ? toAccDeviceType(*icv::AccCurrentDeviceTypeVar)
               : acc_device_default)
       << ", default device type="
       << (icv::DefaultDeviceType ? toAccDeviceType(*icv::DefaultDeviceType)
                                  : acc_device_none)
       << ", acc-current-device-num={";
    for (const ConcreteDeviceTypeInfo &Info : ConcreteDeviceTypes)
      OS << (&Info == ConcreteDeviceTypes.begin() ? "" : ", ") << Info.AccType
         << ":" << icv::currentDeviceNum(Info.Type);
    OS << "}";
  });
}

int DeviceManagerTy::getNumDevices(acc_device_t DeviceType) {
  if (DeviceType == acc_device_not_host) {
    int Num = 0;
    for (const ConcreteDeviceTypeInfo &Info : ConcreteDeviceTypes)
      if (Info.Type != ConcreteDeviceType::host)
        Num += getSingleDeviceTypeMap(Info.Type).size();
    return Num;
  }
  // There is no device of the types that do not resolve.
  std::optional<ConcreteDeviceType> Type = resolve(DeviceType);
  return Type ? getSingleDeviceTypeMap(*Type).size() : 0;
}

void DeviceManagerTy::checkAndSetDeviceNum(ConcreteDeviceType Type, int DevNum,
                                           bool IgnoreNonExistent) {
  size_t NumDevices = getSingleDeviceTypeMap(Type).size();
  if (NumDevices == 0 && IgnoreNonExistent)
    return;
  if (DevNum < 0 || static_cast<size_t>(DevNum) >= NumDevices)
    REPORT_FATAL() << "Invalid device number " << DevNum << " for device type "
                   << toAccDeviceType(Type) << ", of which there are "
                   << NumDevices;
  icv::currentDeviceNum(Type) = DevNum;
}

void DeviceManagerTy::setAllDeviceNum(int DevNum) {
  for (const ConcreteDeviceTypeInfo &Info : ConcreteDeviceTypes)
    checkAndSetDeviceNum(Info.Type, DevNum, /*IgnoreNonExistent=*/true);
  dumpICVs();
}

void DeviceManagerTy::setDeviceNum(acc_device_t DeviceType, int DevNum) {
  // OpenACC 3.4, sec. 3.2.4 and 2.14.3: If the value of the device number is
  // negative, the runtime will revert to its default behavior, which is
  // implementation-defined.
  if (DevNum < 0)
    DevNum = 0;

  // If the value of the device type is zero, or the device_type clause of the
  // set directive does not appear, the selected device number will be used for
  // all device types.
  if (DeviceType == acc_device_none) {
    setAllDeviceNum(DevNum);
    return;
  }

  // Selecting a device implies selecting its device type.
  if (DeviceType == acc_device_current) {
    std::optional<ConcreteDeviceType> Current = resolve(DeviceType);
    if (!Current)
      REPORT_FATAL() << "Cannot select device " << DevNum
                     << " of the current device type, there are no devices";
    DeviceType = toAccDeviceType(*Current);
  }
  setDeviceType(DeviceType);
  std::optional<ConcreteDeviceType> Type = icv::currentDeviceType();
  if (!Type)
    REPORT_FATAL() << "Cannot select device " << DevNum
                   << " of the device type " << DeviceType;
  checkAndSetDeviceNum(*Type, DevNum);
  dumpICVs();
}

acc_device_t DeviceManagerTy::getDeviceType() {
  // OpenACC 3.4, sec. 3.1: acc_device_default is never returned.
  std::optional<ConcreteDeviceType> Type = icv::currentDeviceType();
  return Type ? toAccDeviceType(*Type) : acc_device_none;
}

void DeviceManagerTy::setDeviceType(acc_device_t DeviceType) {
  // The device types that do not name a device type select the default one.
  if (DeviceType == acc_device_none || DeviceType == acc_device_default) {
    icv::AccCurrentDeviceTypeVar = std::nullopt;
  } else {
    // A set of types, as acc_device_not_host and acc_device_current, cannot be
    // selected.
    if (DeviceType == acc_device_not_host || DeviceType == acc_device_current)
      REPORT_FATAL() << "The device type " << DeviceType
                     << " cannot be selected";
    std::optional<ConcreteDeviceType> Type = toConcreteDeviceType(DeviceType);
    if (!Type || getSingleDeviceTypeMap(*Type).empty())
      REPORT_FATAL() << "There are no devices of type " << DeviceType;
    icv::AccCurrentDeviceTypeVar = Type;
  }
  dumpICVs();
}

llvm::Expected<DeviceTy &> DeviceManagerTy::getDevice(acc_device_t DeviceType) {
  int PMDeviceId = getPMDeviceId(DeviceType);
  if (PMDeviceId < 0)
    return noDevicesError(DeviceType);
  return PM->getDevice(PMDeviceId);
}

llvm::Expected<DeviceTy &> DeviceManagerTy::getDevice(acc_device_t DeviceType,
                                                      int DeviceNum) {
  std::optional<ConcreteDeviceType> Type = resolve(DeviceType);
  if (!Type)
    return noDevicesError(DeviceType);
  SingleDeviceTypeMapTy &Map = getSingleDeviceTypeMap(*Type);
  if (DeviceNum < 0 || static_cast<size_t>(DeviceNum) >= Map.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid device number %d for device type %d", DeviceNum,
        static_cast<int>(toAccDeviceType(*Type)));
  return PM->getDevice(Map[DeviceNum]);
}

llvm::Expected<DeviceTy &> DeviceManagerTy::getDevice() {
  return getDevice(acc_device_none);
}

DeviceTy &DeviceManagerTy::getDeviceOrAbort(acc_device_t DeviceType) {
  llvm::Expected<DeviceTy &> DeviceOrErr = getDevice(DeviceType);
  if (!DeviceOrErr)
    REPORT_FATAL() << "Failed to get device: "
                   << toString(DeviceOrErr.takeError());
  return *DeviceOrErr;
}

DeviceTy &DeviceManagerTy::getDeviceOrAbort(acc_device_t DeviceType,
                                            int DeviceNum) {
  llvm::Expected<DeviceTy &> DeviceOrErr = getDevice(DeviceType, DeviceNum);
  if (!DeviceOrErr)
    REPORT_FATAL() << "Failed to get device: "
                   << toString(DeviceOrErr.takeError());
  return *DeviceOrErr;
}
