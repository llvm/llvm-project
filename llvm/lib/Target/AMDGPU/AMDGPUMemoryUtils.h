//===- AMDGPUMemoryUtils.h - Memory related helper functions -*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUMEMORYUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUMEMORYUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {

struct Align;
class AAResults;
class DataLayout;
class GlobalVariable;
class LoadInst;
class MemoryDef;
class MemorySSA;
class Value;
class Function;
class CallGraph;
class Module;
class TargetExtType;

namespace AMDGPU {

using FunctionVariableMap = DenseMap<Function *, DenseSet<GlobalVariable *>>;
using VariableFunctionMap = DenseMap<GlobalVariable *, DenseSet<Function *>>;

Align getAlign(const DataLayout &DL, const GlobalVariable *GV);

// Get the synthetic aperture number for the given address space, or None (0)
// if the address space does not have one.
unsigned tryGetSyntheticApertureNumber(unsigned AS);

// Copy metadata onto a load widened to read a superset of Source's bytes. Only
// value-independent metadata is copied; metadata describing the loaded value
// (!range, !noundef, !nofpclass, !tbaa, ...) is dropped.
void copyMetadataForWidenedLoad(LoadInst &Dest, const LoadInst &Source);

// If GV is a named-barrier return its type. Otherwise return nullptr.
TargetExtType *isNamedBarrier(const GlobalVariable &GV);

bool isDynamicLDS(const GlobalVariable &GV);
bool isLDSVariableToLower(const GlobalVariable &GV);

struct GVUsesInfoTy {
  FunctionVariableMap DirectAccess;
  FunctionVariableMap IndirectAccess;
};

/// Iterates over all GlobalVariables in \p M, and whenever \p Filter returns
/// true, replace all constant users of the GV with instructions.
bool eliminateGVConstantExprUsesFromAllInstructions(
    Module &M, function_ref<bool(const GlobalVariable &)> Filter);

/// Finds uses of Global Variables on a per-function basis.
/// \param CG \p M Call Graph
/// \param M Module
/// \param Filter Function that returns true for GVs that need to be considered.
/// \param Kernels[out] Maps kernels to global variables used by that kernel.
/// \param Functions[out] Maps functions to global variables used by that
/// function.
void getUsesOfGVByFunction(const CallGraph &CG, Module &M,
                           function_ref<bool(const GlobalVariable &)> Filter,
                           FunctionVariableMap &Kernels,
                           FunctionVariableMap &Functions);

/// Collects all uses of Global Variables in \p M using
/// \ref getUsesOfGVByFunction.
/// \param CG \p M Call Graph
/// \param M Module
/// \param Filter Filter for \ref getUsesOfGVByFunction - only GVs for which the
/// filter returns true will be considered.
/// \returns Uses of GVs that were found within each function, sorted by
/// direct and indirect accesses.
GVUsesInfoTy
getTransitiveUsesOfGV(const CallGraph &CG, Module &M,
                      function_ref<bool(const GlobalVariable &)> Filter);

/// Collects all uses of LDS Global Variables in \p M using
/// \ref getUsesOfGVByFunction, with \ref isLDSVariableToLower as the filter.
/// \param CG \p M Call Graph
/// \param M Module
/// \returns Uses of LDS GVs that need lowering that were found within each
/// function, sorted by direct and indirect accesses.
GVUsesInfoTy getTransitiveUsesOfLDSForLowering(const CallGraph &CG, Module &M);

/// Strip FnAttr attribute from any functions where we may have
/// introduced its use.
void removeFnAttrFromReachable(CallGraph &CG, Function *KernelRoot,
                               ArrayRef<StringRef> FnAttrs);

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
