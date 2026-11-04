//===- MockOMP.cpp - Mock OMP functions for testing ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mock implementations for OMP functions (libomptarget not available in tests).
//
//===----------------------------------------------------------------------===//

// Mock device count for testing
static int MockNumDevices = 4;
static const char *MockDeviceUIDs[] = {"device-0", "device-1", "device-2",
                                       "device-3"};

extern "C" int omp_get_num_devices() { return MockNumDevices; }

extern "C" const char *omp_get_uid_from_device(int DeviceNum) {
  if (DeviceNum >= 0 && DeviceNum < MockNumDevices)
    return MockDeviceUIDs[DeviceNum];
  return "";
}
