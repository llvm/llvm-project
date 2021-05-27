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

namespace AMDGPU {

bool isKernelCC(const Function *Func);

Align getAlign(DataLayout const &DL, const GlobalVariable *GV);

/// \returns true if an LDS global requres lowering to a module LDS structure
/// if \p F is not given. If \p F is given it must be a kernel and function
/// \returns true if an LDS global is directly used from that kernel and it
/// is safe to replace its uses with a kernel LDS structure member.
/// \p UsedList contains a union of llvm.used and llvm.compiler.used variables
/// which do not count as a use.
bool shouldLowerLDSToStruct(const SmallPtrSetImpl<GlobalValue *> &UsedList,
                            const GlobalVariable &GV,
                            const Function *F = nullptr);

std::vector<GlobalVariable *>
findVariablesToLower(Module &M, const SmallPtrSetImpl<GlobalValue *> &UsedList,
                     const Function *F = nullptr);

SmallPtrSet<GlobalValue *, 32> getUsedList(Module &M);

/// \returns true if all uses of \p U end up in a function \p F.
bool isUsedOnlyFromFunction(const User *U, const Function *F);

} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H
