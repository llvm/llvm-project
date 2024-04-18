//===---- device_environment.h - OpenMP GPU device environment ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Global device environment
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_DEVICE_ENVIRONMENT_H_
#define _OMPTARGET_DEVICE_ENVIRONMENT_H_

// deviceRTL uses <stdint> and DeviceRTL uses explicit definitions

struct DeviceEnvironmentTy {
  uint32_t DebugKind;
  uint32_t NumDevices;
  uint32_t DeviceNum;
  uint32_t DynamicMemSize;
  uint64_t ClockFrequency;
};

#endif
