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

bool isKernelCC(Function *Func);

Align getAlign(DataLayout const &DL, const GlobalVariable *GV);

bool userRequiresLowering(const SmallPtrSetImpl<GlobalValue *> &UsedList,
                          User *InitialUser);

std::vector<GlobalVariable *>
findVariablesToLower(Module &M, const SmallPtrSetImpl<GlobalValue *> &UsedList);

SmallPtrSet<GlobalValue *, 32> getUsedList(Module &M);

} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H
