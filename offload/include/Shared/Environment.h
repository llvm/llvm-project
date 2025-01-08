//===-- Shared/Environment.h - OpenMP GPU environments ------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Environments shared between host and device.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_ENVIRONMENT_H
#define OMPTARGET_SHARED_ENVIRONMENT_H

#include <stdint.h>

#ifdef OMPTARGET_DEVICE_RUNTIME
#include "DeviceTypes.h"
#else
#include "SourceInfo.h"

using IdentTy = ident_t;
#endif

#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"

enum class DeviceDebugKind : uint32_t {
  Assertion = 1U << 0,
  FunctionTracing = 1U << 1,
  CommonIssues = 1U << 2,
  AllocationTracker = 1U << 3,
};

struct DeviceEnvironmentTy {
  uint32_t DeviceDebugKind;
  uint32_t NumDevices;
  uint32_t DeviceNum;
  uint32_t DynamicMemSize;
  uint64_t ClockFrequency;
  uintptr_t IndirectCallTable;
  uint64_t IndirectCallTableSize;
  uint64_t HardwareParallelism;
};

struct DeviceMemoryPoolTy {
  void *Ptr;
  uint64_t Size;
};

struct DeviceMemoryPoolTrackingTy {
  uint64_t NumAllocations;
  uint64_t AllocationTotal;
  uint64_t AllocationMin;
  uint64_t AllocationMax;

  void combine(DeviceMemoryPoolTrackingTy &Other) {
    NumAllocations += Other.NumAllocations;
    AllocationTotal += Other.AllocationTotal;
    AllocationMin = AllocationMin > Other.AllocationMin ? Other.AllocationMin
                                                        : AllocationMin;
    AllocationMax = AllocationMax < Other.AllocationMax ? Other.AllocationMax
                                                        : AllocationMax;
  }
};

// NOTE: Please don't change the order of those members as their indices are
// used in the middle end. Always add the new data member at the end.
// Different from KernelEnvironmentTy below, this structure contains members
// that might be modified at runtime.
struct DynamicEnvironmentTy {
  /// Current indentation level for the function trace. Only accessed by thread
  /// 0.
  uint16_t DebugIndentionLevel;
};

// NOTE: Please don't change the order of those members as their indices are
// used in the middle end. Always add the new data member at the end.
struct ConfigurationEnvironmentTy {
  uint8_t UseGenericStateMachine = 2;
  uint8_t MayUseNestedParallelism = 2;
  llvm::omp::OMPTgtExecModeFlags ExecMode = llvm::omp::OMP_TGT_EXEC_MODE_SPMD;
  // Information about (legal) launch configurations.
  //{
  int32_t MinThreads = -1;
  int32_t MaxThreads = -1;
  int32_t MinTeams = -1;
  int32_t MaxTeams = -1;
  int32_t ReductionDataSize = 0;
  int32_t ReductionBufferLength = 0;
  //}
};

// NOTE: Please don't change the order of those members as their indices are
// used in the middle end. Always add the new data member at the end.
struct KernelEnvironmentTy {
  ConfigurationEnvironmentTy Configuration;
  IdentTy *Ident = nullptr;
  DynamicEnvironmentTy *DynamicEnv = nullptr;
};

struct KernelLaunchEnvironmentTy {
  uint32_t ReductionCnt = 0;
  uint32_t ReductionIterCnt = 0;
  void *ReductionBuffer = nullptr;
};

#endif // OMPTARGET_SHARED_ENVIRONMENT_H
