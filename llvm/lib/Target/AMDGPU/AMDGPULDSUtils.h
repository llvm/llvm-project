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

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPULDSUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPULDSUTILS_H

#include "llvm/Support/Alignment.h"
#include <cstdint>

namespace llvm {

class Function;
class IRBuilderBase;
class TargetMachine;
class Value;

namespace AMDGPU {

/// Compute linear thread id within a workgroup.
Value *buildLinearThreadId(IRBuilderBase &Builder, const TargetMachine &TM);

struct AMDGPULDSBudget {
  uint64_t CurrentUsage = 0;
  uint64_t Limit = 0;
  unsigned MaxOccupancy = 0;
  bool Promotable = false;
  bool DisabledDueToLocalArg = false;
  bool DisabledDueToExternDynShared = false;

  /// Reserve an allocation while allowing for any possible leading padding.
  bool tryReserve(uint64_t AllocSize, Align Alignment);
};

AMDGPULDSBudget computeLDSBudget(const Function &F, const TargetMachine &TM);

} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPULDSUTILS_H
