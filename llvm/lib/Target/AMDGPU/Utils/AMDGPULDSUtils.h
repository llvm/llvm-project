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

/// Collect reachable callees for each kernel defined in the module \p M and
/// return collected callees at \p KernelToCallees.
void collectReachableCallees(
    Module &M,
    DenseMap<Function *, SmallPtrSet<Function *, 8>> &KernelToCallees);

/// For the given LDS global \p GV, visit all its users and collect all
/// non-kernel functions within which \p GV is used and return collected list of
/// such non-kernel functions.
SmallPtrSet<Function *, 8> collectNonKernelAccessorsOfLDS(GlobalVariable *GV);

/// Collect all the instructions where user \p U belongs to. \p U could be
/// instruction itself or it could be a constant expression which is used within
/// an instruction. If \p CollectKernelInsts is true, collect instructions only
/// from kernels, otherwise collect instructions only from non-kernel functions.
DenseMap<Function *, SmallPtrSet<Instruction *, 8>>
getFunctionToInstsMap(User *U, bool CollectKernelInsts);

bool isKernelCC(const Function *Func);

Align getAlign(DataLayout const &DL, const GlobalVariable *GV);

/// \returns true if a given global variable \p GV (or its global users) appear
/// as an use within some instruction (either from kernel or from non-kernel).
bool hasUserInstruction(const GlobalValue *GV);

/// \returns true if an LDS global requires lowering to a module LDS structure
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
