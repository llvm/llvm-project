//===- DeviceManager.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ACC_OFFLOAD_DEVICE_MANAGER_H_
#define LLVM_ACC_OFFLOAD_DEVICE_MANAGER_H_

#include "omptarget.h"
#include "openacc.h"
#include "llvm/Support/ErrorHandling.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace llvm::acc::target {

/// The device types that this runtime can have devices of. The values are the
/// indices of the arrays that hold one entry per concrete device type. Unlike
/// acc_device_t, a value of this type is always a concrete device type (i.e. no
/// acc_device_none, acc_device_default and the like).
enum class ConcreteDeviceType { nvidia, radeon, intelgpu, host };

/// A concrete device type, the acc_device_t that stands for it, and the
/// liboffload backend that provides its devices.
struct ConcreteDeviceTypeInfo {
  ConcreteDeviceType Type;
  acc_device_t AccType;
  ol_platform_backend_t Backend;
};

/// The single list of the concrete device types, in the order of
/// ConcreteDeviceType. It is also the order of priority when the default device
/// type is chosen, so the host is only the default if there is nothing else.
inline constexpr std::array ConcreteDeviceTypes = {
    ConcreteDeviceTypeInfo{ConcreteDeviceType::nvidia, acc_device_nvidia,
                           OL_PLATFORM_BACKEND_CUDA},
    ConcreteDeviceTypeInfo{ConcreteDeviceType::radeon, acc_device_radeon,
                           OL_PLATFORM_BACKEND_AMDGPU},
    ConcreteDeviceTypeInfo{ConcreteDeviceType::intelgpu, acc_device_intelgpu,
                           OL_PLATFORM_BACKEND_LEVEL_ZERO},
    ConcreteDeviceTypeInfo{ConcreteDeviceType::host, acc_device_host,
                           OL_PLATFORM_BACKEND_HOST},
};
inline constexpr size_t NumConcreteDeviceTypes = ConcreteDeviceTypes.size();

/// The index of \p Type in the arrays with one entry per concrete device type.
constexpr size_t getIndex(ConcreteDeviceType Type) {
  return static_cast<size_t>(Type);
}

/// The acc_device_t that stands for \p Type.
constexpr acc_device_t toAccDeviceType(ConcreteDeviceType Type) {
  return ConcreteDeviceTypes[getIndex(Type)].AccType;
}

/// The concrete device type that \p DeviceType is, if it is one.
constexpr std::optional<ConcreteDeviceType>
toConcreteDeviceType(acc_device_t DeviceType) {
  for (const ConcreteDeviceTypeInfo &Info : ConcreteDeviceTypes)
    if (Info.AccType == DeviceType)
      return Info.Type;
  return std::nullopt;
}

/// Maintains the OpenACC device mapping.
class DeviceManagerTy {
public:
  using DeviceIdTy = int64_t;

private:
  using SingleDeviceTypeMapTy = llvm::SmallVector<DeviceIdTy, 8>;
  using AllDeviceTypeMap =
      std::array<SingleDeviceTypeMapTy, NumConcreteDeviceTypes>;
  AllDeviceTypeMap PMDeviceMap;

  SingleDeviceTypeMapTy &getSingleDeviceTypeMap(ConcreteDeviceType Type);

  /// Set the current device number of \p DeviceType. If \p IgnoreNonExistent is
  /// true, do nothing when there are no devices of that type; otherwise report
  /// a fatal error if \p DeviceNum is out of range.
  void checkAndSetDeviceNum(ConcreteDeviceType Type, int DeviceNum,
                            bool IgnoreNonExistent = false);

  /// Get the PluginManager device ID for the current device of \p Type, or -1
  /// if the current device number is not a device of \p Type.
  int getPMDeviceId(ConcreteDeviceType Type);

  /// Set the current device number of every device type that has devices.
  void setAllDeviceNum(int DeviceNum);

  /// Print the ICVs of the calling thread, and the default device type.
  void dumpICVs();

public:
  void init();
  void deinit();

  /// Refreshes the device mapping according to the currently initialized
  /// devices in the PluginManager.
  void refreshDeviceMapping(bool UpdateDeviceType);

  /// Entry points for OpenACC user APIs.
  /// {
  /// acc_get_device_num. Returns -1 if there are no devices of \p DeviceType.
  int getDeviceNum(acc_device_t DeviceType);
  /// acc_get_num_devices.
  int getNumDevices(acc_device_t DeviceType);
  /// acc_set_device_num, and the device_num clause of the set directive: select
  /// device \p DeviceNum of type \p DeviceType. A negative number selects the
  /// default one, and acc_device_none selects the number for all device types.
  void setDeviceNum(acc_device_t DeviceType, int DeviceNum);

  /// acc_get_device_type. Never returns acc_device_default.
  acc_device_t getDeviceType();
  /// acc_set_device_type.
  void setDeviceType(acc_device_t DeviceType);
  /// }

  /// Get the PluginManager device ID for the current device of type \p
  /// DeviceType.
  int getPMDeviceId(acc_device_t DeviceType);
  /// Get the PluginManager device ID for the current device of the current
  /// device type.
  int getPMDeviceId();

  /// Get the device corresponding to type \p DeviceType and number \p
  /// DeviceNum.
  llvm::Expected<DeviceTy &> getDevice(acc_device_t DeviceType, int DeviceNum);
  /// Get the current device corresponding to type \p DeviceType.
  llvm::Expected<DeviceTy &> getDevice(acc_device_t DeviceType);
  /// Get the current device.
  llvm::Expected<DeviceTy &> getDevice();

  /// Like getDevice, but report a fatal error if there is no such device.
  DeviceTy &getDeviceOrAbort(acc_device_t DeviceType = acc_device_default);
  DeviceTy &getDeviceOrAbort(acc_device_t DeviceType, int DeviceNum);
};

extern DeviceManagerTy *DM;
} // namespace llvm::acc::target

#endif // LLVM_ACC_OFFLOAD_DEVICE_MANAGER_H_
