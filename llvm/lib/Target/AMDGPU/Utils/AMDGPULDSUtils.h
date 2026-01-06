//===-- AMDGPULDSUtils.h - AMDGPU LDS utilities ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for computing LDS usage and limits for an AMDGPU function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H

#include <cstdint>
#include <utility>

namespace llvm {

class AMDGPUSubtarget;
class Function;
class IRBuilderBase;
class Module;
class TargetMachine;
class Value;

namespace AMDGPU {

/// Get workitem id for dimension N (0,1,2).
Value *getWorkitemID(IRBuilderBase &Builder, Module &M,
                     const AMDGPUSubtarget &ST, unsigned N);

/// Compute linear thread id within a workgroup.
Value *buildLinearThreadId(IRBuilderBase &Builder, Module &M,
                           const AMDGPUSubtarget &ST);

struct AMDGPULDSBudget {
  uint32_t currentUsage = 0;
  uint32_t limit = 0;
  unsigned maxOccupancy = 0;
  bool promotable = false;
  bool disabledDueToLocalArg = false;
  bool disabledDueToExternDynShared = false;
};

AMDGPULDSBudget computeLDSBudget(const Function &F, const TargetMachine &TM);

} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H
