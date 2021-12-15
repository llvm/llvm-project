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

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"

namespace llvm {

class ConstantExpr;

namespace AMDGPU {

bool isKernelCC(const Function *Func);

Align getAlign(DataLayout const &DL, const GlobalVariable *GV);

std::vector<GlobalVariable *> findVariablesToLower(Module &M,
                                                   const Function *F = nullptr);

/// Replace all uses of constant \p C with instructions in \p F.
void replaceConstantUsesInFunction(ConstantExpr *C, const Function *F);
} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULDSUTILS_H
