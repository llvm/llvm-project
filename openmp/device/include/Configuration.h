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
uint32_t getNumDevices();

/// Return the device number in the system for omp_get_device_num.
uint32_t getDeviceNum();

/// Return the user chosen debug level.
uint32_t getDebugKind();

/// Return if teams oversubscription is assumed
uint32_t getAssumeTeamsOversubscription();

/// Return if threads oversubscription is assumed
uint32_t getAssumeThreadsOversubscription();

/// Return the amount of dynamic shared memory that was allocated at launch.
uint64_t getDynamicMemorySize();

/// Returns the cycles per second of the device's fixed frequency clock.
uint64_t getClockFrequency();

/// Returns the pointer to the beginning of the indirect call table.
void *getIndirectCallTablePtr();

/// Returns the size of the indirect call table.
uint64_t getIndirectCallTableSize();

/// Returns the size of the indirect call table.
uint64_t getHardwareParallelism();

/// Return if debugging is enabled for the given debug kind.
bool isDebugMode(DeviceDebugKind Level);

/// Indicates if this kernel may require thread-specific states, or if it was
/// explicitly disabled by the user.
bool mayUseThreadStates();

/// Indicates if this kernel may require data environments for nested
/// parallelism, or if it was explicitly disabled by the user.
bool mayUseNestedParallelism();

/// Returns true if the current thread should enter the generic state machine.
/// On some architectures, some threads should not enter the state machine to
/// avoid warp-level barrier forwarding issues during initialization.
/// On other architectures, all threads must enter the state machine to satisfy
/// the requirements of workgroup synchronization.
static inline bool shouldEnterStateMachine(bool IsSPMD);

} // namespace config
} // namespace ompx

#include "Mapping.h"

namespace ompx {
namespace config {

static inline bool shouldEnterStateMachine(bool IsSPMD) {
#if defined(__NVPTX__) || defined(__AMDGPU__)
  // This check is important for NVIDIA Pascal (but not Volta) and AMD
  // GPU. In those cases, a single thread can apparently satisfy a barrier on
  // behalf of all threads in the same warp. Thus, it would not be safe for
  // other threads in the main thread's warp to reach the first
  // synchronize::threads call in genericStateMachine before the main thread
  // reaches its corresponding synchronize::threads call: that would permit all
  // active worker threads to proceed before the main thread has actually set
  // state::ParallelRegionFn, and then they would immediately quit without
  // doing any work.  mapping::getMaxTeamThreads() does not include any of the
  // main thread's warp, so none of its threads can ever be active worker
  // threads.
  return mapping::getThreadIdInBlock() < mapping::getMaxTeamThreads(IsSPMD);
#else
  // On other architectures (e.g., Intel GPUs) all threads must enter the state
  // machine to satisfy the requirements of workgroup of synchronize::threads
  // call in genericStateMachine. Otherwise, the workers will wait on the
  // call to synchronize::threads forever and never proceed.
  (void)IsSPMD;
  return true;
#endif
}

} // namespace config
} // namespace ompx

#endif
