//===- AMDGPUMemoryUtils.h - Memory related helper functions -*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUMEMORYUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUMEMORYUTILS_H

#include <vector>

namespace llvm {

struct Align;
class AAResults;
class ConstantExpr;
class DataLayout;
class Function;
class GlobalVariable;
class LoadInst;
class MemoryDef;
class MemorySSA;
class Module;
class Value;

namespace AMDGPU {

Align getAlign(DataLayout const &DL, const GlobalVariable *GV);

bool isLDSVariableToLower(const GlobalVariable &GV);
std::vector<GlobalVariable *> findLDSVariablesToLower(Module &M,
                                                      const Function *F);

/// Replace all uses of constant \p C with instructions in \p F.
void replaceConstantUsesInFunction(ConstantExpr *C, const Function *F);

/// Given a \p Def clobbering a load from \p Ptr according to the MSSA check
/// if this is actually a memory update or an artificial clobber to facilitate
/// ordering constraints.
bool isReallyAClobber(const Value *Ptr, MemoryDef *Def, AAResults *AA);

/// Check is a \p Load is clobbered in its function.
bool isClobberedInFunction(const LoadInst *Load, MemorySSA *MSSA,
                           AAResults *AA);

} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUMEMORYUTILS_H
