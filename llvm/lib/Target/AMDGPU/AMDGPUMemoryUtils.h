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
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/DebugLoc.h"
#include <optional>

namespace llvm {

struct Align;
class AAResults;
class DataLayout;
class GlobalVariable;
class Instruction;
class LoadInst;
class MemoryDef;
class MemorySSA;
class MMRAMetadata;
class Value;
class Function;
class CallGraph;
class Module;
class TargetExtType;

namespace AMDGPU {

static constexpr unsigned NamedBarrierTypeSizeInBytes = 1;

using FunctionVariableMap = MapVector<Function *, DenseSet<GlobalVariable *>>;
using VariableFunctionMap = DenseMap<GlobalVariable *, DenseSet<Function *>>;

Align getAlign(const DataLayout &DL, const GlobalVariable *GV);

// Get the synthetic aperture number for the given address space, or None (0)
// if the address space does not have one.
unsigned getSyntheticApertureNumber(unsigned AS);

// Copy metadata onto a load widened to read a superset of Source's bytes. Only
// value-independent metadata is copied; metadata describing the loaded value
// (!range, !noundef, !nofpclass, !tbaa, ...) is dropped.
void copyMetadataForWidenedLoad(LoadInst &Dest, const LoadInst &Source);

// If GV is a named-barrier return its type. Otherwise return nullptr.
TargetExtType *isNamedBarrier(const GlobalVariable &GV);

/// \returns how many named barriers are declared by \p GV.
unsigned getNumNamedBarriersDeclared(const DataLayout &DL,
                                     const GlobalVariable &GV);

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

/// The distinct address spaces supported by the AMDGPU target for
/// atomic memory operation. Can be ORed together.
enum class SIAtomicAddrSpace {
  NONE = 0u,
  GLOBAL = 1u << 0,
  LDS = 1u << 1,
  SCRATCH = 1u << 2,
  GDS = 1u << 3,
  OTHER = 1u << 4,

  /// The address spaces that can be accessed by a FLAT instruction.
  FLAT = GLOBAL | LDS | SCRATCH,

  /// The address spaces that support atomic instructions.
  ATOMIC = GLOBAL | LDS | SCRATCH | GDS,

  /// All address spaces.
  ALL = GLOBAL | LDS | SCRATCH | GDS | OTHER,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ ALL)
};

/// \return Return a bit set of the address spaces accessed by \p AS.
SIAtomicAddrSpace toSIAtomicAddrSpace(unsigned AS);

/// Reads \p I's MMRAs to parse the "amdgpu-synchronize-as" MMRA.
/// If this tag isn't present, or if it has no meaningful values, returns
/// \p ATOMIC, otherwise returns the address spaces specified by the MD.
/// Does not diagnose unexpected MMRA tag suffixes.
SIAtomicAddrSpace getSynchronizeAddrSpace(const Instruction &I);

/// Parses the values for the "amdgpu-synchronize-as" MMRA.
/// If this tag isn't present, or if it has no meaningful values, returns
/// \p ATOMIC, otherwise returns the address spaces specified by the MD.
/// If \p DiagFn is non-null, unexpected suffixes for the MMRA tag are reported
/// as warnings to \p DiagFn's context, located at \p DL.
SIAtomicAddrSpace getSynchronizeAddrSpace(const MMRAMetadata &MMRA,
                                          const Function *DiagFn = nullptr,
                                          const DebugLoc &DL = {});

} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUMEMORYUTILS_H
