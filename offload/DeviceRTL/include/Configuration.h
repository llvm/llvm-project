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

#include "Shared/Environment.h"

#include "DeviceTypes.h"

namespace ompx {
namespace config {

/// Return the number of devices in the system, same number as returned on the
/// host by omp_get_num_devices.
OMP_ATTRS uint32_t getNumDevices();

/// Return the device number in the system for omp_get_device_num.
OMP_ATTRS uint32_t getDeviceNum();

/// Return the user choosen debug level.
OMP_ATTRS uint32_t getDebugKind();

/// Return if teams oversubscription is assumed
OMP_ATTRS uint32_t getAssumeTeamsOversubscription();

/// Return if threads oversubscription is assumed
OMP_ATTRS uint32_t getAssumeThreadsOversubscription();

/// Return the amount of dynamic shared memory that was allocated at launch.
OMP_ATTRS uint64_t getDynamicMemorySize();

/// Returns the cycles per second of the device's fixed frequency clock.
OMP_ATTRS uint64_t getClockFrequency();

/// Returns the pointer to the beginning of the indirect call table.
OMP_ATTRS void *getIndirectCallTablePtr();

/// Returns the size of the indirect call table.
OMP_ATTRS uint64_t getIndirectCallTableSize();

/// Returns the size of the indirect call table.
OMP_ATTRS uint64_t getHardwareParallelism();

/// Return if debugging is enabled for the given debug kind.
OMP_ATTRS bool isDebugMode(DeviceDebugKind Level);

/// Indicates if this kernel may require thread-specific states, or if it was
/// explicitly disabled by the user.
OMP_ATTRS bool mayUseThreadStates();

/// Indicates if this kernel may require data environments for nested
/// parallelism, or if it was explicitly disabled by the user.
OMP_ATTRS bool mayUseNestedParallelism();

} // namespace config
} // namespace ompx

#endif
