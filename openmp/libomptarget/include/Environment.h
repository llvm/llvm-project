//===------------ Environment.h - OpenMP GPU environments --------- C++ -*-===//
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

#ifndef _OMPTARGET_ENVIRONMENT_H_
#define _OMPTARGET_ENVIRONMENT_H_

#ifdef OMPTARGET_DEVICE_RUNTIME
#include "Types.h"
#else
#include "SourceInfo.h"

#include <cstdint>

using IdentTy = ident_t;
#endif

#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"

struct DeviceEnvironmentTy {
  uint32_t DebugKind;
  uint32_t NumDevices;
  uint32_t DeviceNum;
  uint32_t DynamicMemSize;
  uint64_t ClockFrequency;
  uintptr_t IndirectCallTable;
  uint64_t IndirectCallTableSize;
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
  uint8_t UseGenericStateMachine;
  uint8_t MayUseNestedParallelism;
  llvm::omp::OMPTgtExecModeFlags ExecMode;
};

// NOTE: Please don't change the order of those members as their indices are
// used in the middle end. Always add the new data member at the end.
struct KernelEnvironmentTy {
  ConfigurationEnvironmentTy Configuration;
  IdentTy *Ident;
  DynamicEnvironmentTy *DynamicEnv;
};

#endif // _OMPTARGET_ENVIRONMENT_H_
