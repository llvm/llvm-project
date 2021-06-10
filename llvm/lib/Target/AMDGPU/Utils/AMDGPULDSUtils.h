//===- AMDGPULDSUtils.h - LDS related helper functions -*- C++ -*----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGPU LDS related helper utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H

#include "AMDGPU.h"

namespace llvm {

class ConstantExpr;

namespace AMDGPU {

bool isKernelCC(const Function *Func);

Align getAlign(DataLayout const &DL, const GlobalVariable *GV);

/// \returns true if a given global variable \p GV (or its global users) appear
/// as an use within some instruction (either from kernel or from non-kernel).
bool hasUserInstruction(const GlobalValue *GV);

/// \returns true if an LDS global requres lowering to a module LDS structure
/// if \p F is not given. If \p F is given it must be a kernel and function
/// \returns true if an LDS global is directly used from that kernel and it
/// is safe to replace its uses with a kernel LDS structure member.
bool shouldLowerLDSToStruct(const GlobalVariable &GV,
                            const Function *F = nullptr);

std::vector<GlobalVariable *> findVariablesToLower(Module &M,
                                                   const Function *F = nullptr);

SmallPtrSet<GlobalValue *, 32> getUsedList(Module &M);

/// Replace all uses of constant \p C with instructions in \p F.
void replaceConstantUsesInFunction(ConstantExpr *C, const Function *F);
} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H
