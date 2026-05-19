//===- DeviceManager.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ACC_OFFLOAD_DEVICE_MANAGER_H_
#define LLVM_ACC_OFFLOAD_DEVICE_MANAGER_H_

#include "include/openacc.h"
#include "omptarget.h"
#include <array>
#include <cstddef>

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     acc_device_t DeviceType);

namespace llvm::acc::target {

constexpr int AccDeviceNumConcreteTypes =
    acc_device_concrete_type_end - acc_device_concrete_type_begin;
constexpr int AccDeviceTypeOffset = acc_device_concrete_type_begin;

class DeviceManagerTy {
public:
  using DeviceIdTy = int64_t;

private:
  using SingleDeviceTypeMapTy = llvm::SmallVector<DeviceIdTy, 8>;
  using AllDeviceTypeMap =
      std::array<SingleDeviceTypeMapTy, AccDeviceNumConcreteTypes>;
  AllDeviceTypeMap PMDeviceMap;

  SingleDeviceTypeMapTy &getSingleDeviceTypeMap(acc_device_t DeviceType);

public:
  void init();
  void deinit();

  // Refreshes the device mapping according to the
  void refreshDeviceMapping(bool UpdateDeviceType);

  // Entry points for ACC APIs.
  int getDeviceId(acc_device_t DeviceType);
  int getNumDevices(acc_device_t DeviceType);

  void setAllDeviceId(int DeviceId);
  void setDeviceId(acc_device_t DeviceType, int DeviceId);
  void setDeviceId(int DeviceId);

  acc_device_t getDeviceType();
  void setDeviceType(acc_device_t DeviceType);

  size_t getDeviceProperty(int DeviceId, acc_device_t DeviceType,
                           acc_device_property_t DeviceProperty);
  const char *getDevicePropertyString(int DeviceId, acc_device_t DeviceType,
                                      acc_device_property_t DeviceProperty);

  // Verification.
  void checkICVs();

  // Obtaining the device ID for use with PluginManager.
  int getPMDeviceId(acc_device_t DeviceType);
  int getPMDeviceId();

  llvm::Expected<DeviceTy &> getDevice(acc_device_t DeviceType);
  llvm::Expected<DeviceTy &> getDevice();
};

extern DeviceManagerTy *DM;
} // namespace llvm::acc::target

#endif // LLVM_ACC_OFFLOAD_DEVICE_MANAGER_H_
