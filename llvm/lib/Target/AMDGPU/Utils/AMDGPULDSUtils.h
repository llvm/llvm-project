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

namespace llvm {

class Function;
class TargetMachine;

struct AMDGPULDSBudget {
  uint32_t currentUsage = 0;
  uint32_t limit = 0;
  unsigned maxOccupancy = 0;
  bool promotable = false;
  bool disabledDueToLocalArg = false;
  bool disabledDueToExternDynShared = false;
};

AMDGPULDSBudget computeLDSBudget(const Function &F, const TargetMachine &TM);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H
