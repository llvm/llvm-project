//===--- Configuration.h - OpenMP device configuration interface -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// API to query the global (constant) device environment.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_CONFIGURATION_H
#define OMPTARGET_CONFIGURATION_H

#include "Types.h"

namespace _OMP {
namespace config {

enum DebugKind : uint32_t {
  Assertion = 1U << 0,
  FunctionTracing = 1U << 1,
};

/// Return the number of devices in the system, same number as returned on the
/// host by omp_get_num_devices.
uint32_t getNumDevices();

/// Return the number of devices in the system, same number as returned on the
/// host by omp_get_num_devices.
uint32_t getDeviceNum();

/// Return the user choosen debug level.
uint32_t getDebugKind();

/// Return the amount of dynamic shared memory that was allocated at launch.
uint64_t getDynamicMemorySize();

bool isDebugMode(DebugKind Level);

} // namespace config
} // namespace _OMP

#endif
