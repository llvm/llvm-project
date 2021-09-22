//===- Configuration.cpp - OpenMP device configuration interface -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the data object of the constant device environment and the
// query API.
//
//===----------------------------------------------------------------------===//

#include "Configuration.h"
#include "State.h"
#include "Types.h"

using namespace _OMP;

struct DeviceEnvironmentTy {
  uint32_t DebugLevel;
  uint32_t NumDevices;
  uint32_t DeviceNum;
  uint64_t DynamicMemSize;
};

#pragma omp declare target

extern uint32_t __omp_rtl_debug_kind;

// TOOD: We want to change the name as soon as the old runtime is gone.
DeviceEnvironmentTy CONSTANT(omptarget_device_environment)
    __attribute__((used));

uint32_t config::getDebugLevel() {
  return __omp_rtl_debug_kind & omptarget_device_environment.DebugLevel;
}

uint32_t config::getNumDevices() {
  return omptarget_device_environment.NumDevices;
}

uint32_t config::getDeviceNum() {
  return omptarget_device_environment.DeviceNum;
}

uint64_t config::getDynamicMemorySize() {
  return omptarget_device_environment.DynamicMemSize;
}

bool config::isDebugMode(config::DebugLevel Level) {
  return config::getDebugLevel() > Level;
}

#pragma omp end declare target
