//===- VPlanAnalysis.h - Various Analyses working on VPlan ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANANALYSIS_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/IR/Type.h"

namespace llvm {

class LLVMContext;
class VPValue;
class VPBlendRecipe;
class VPInstruction;
class VPWidenRecipe;
class VPWidenCallRecipe;
class VPWidenIntOrFpInductionRecipe;
class VPWidenMemoryRecipe;
struct VPWidenSelectRecipe;
class VPReplicateRecipe;
class VPRecipeBase;
class VPlan;
class Value;
class TargetTransformInfo;
class Type;

/// An analysis for type-inference for VPValues.
/// It infers the scalar type for a given VPValue by bottom-up traversing
/// through defining recipes until root nodes with known types are reached (e.g.
/// live-ins or load recipes). The types are then propagated top down through
/// operations.
/// Note that the analysis caches the inferred types. A new analysis object must
/// be constructed once a VPlan has been modified in a way that invalidates any
/// of the previously inferred types.
class VPTypeAnalysis {
  DenseMap<const VPValue *, Type *> CachedTypes;
  /// Type of the canonical induction variable. Used for all VPValues without
  /// any underlying IR value (like the vector trip count or the backedge-taken
  /// count).
  Type *CanonicalIVTy;
  LLVMContext &Ctx;

  Type *inferScalarTypeForRecipe(const VPBlendRecipe *R);
  Type *inferScalarTypeForRecipe(const VPInstruction *R);
  Type *inferScalarTypeForRecipe(const VPWidenCallRecipe *R);
  Type *inferScalarTypeForRecipe(const VPWidenRecipe *R);
  Type *inferScalarTypeForRecipe(const VPWidenIntOrFpInductionRecipe *R);
  Type *inferScalarTypeForRecipe(const VPWidenMemoryRecipe *R);
  Type *inferScalarTypeForRecipe(const VPWidenSelectRecipe *R);
  Type *inferScalarTypeForRecipe(const VPReplicateRecipe *R);

public:
  VPTypeAnalysis(const VPlan &Plan);

  /// Infer the type of \p V. Returns the scalar type of \p V.
  Type *inferScalarType(const VPValue *V);

  /// Return the LLVMContext used by the analysis.
  LLVMContext &getContext() { return Ctx; }
};

// Collect a VPlan's ephemeral recipes (those used only by an assume).
void collectEphemeralRecipesForVPlan(VPlan &Plan,
                                     DenseSet<VPRecipeBase *> &EphRecipes);

/// A struct that represents some properties of the register usage
/// of a loop.
struct VPRegisterUsage {
  /// Holds the number of loop invariant values that are used in the loop.
  /// The key is ClassID of target-provided register class.
  SmallMapVector<unsigned, unsigned, 4> LoopInvariantRegs;
  /// Holds the maximum number of concurrent live intervals in the loop.
  /// The key is ClassID of target-provided register class.
  SmallMapVector<unsigned, unsigned, 4> MaxLocalUsers;

  /// Check if any of the tracked live intervals exceeds the number of
  /// available registers for the target. If non-zero, OverrideMaxNumRegs
  /// is used in place of the target's number of registers.
  bool exceedsMaxNumRegs(const TargetTransformInfo &TTI,
                         unsigned OverrideMaxNumRegs = 0) const;
};

/// Estimate the register usage for \p Plan and vectorization factors in \p VFs
/// by calculating the highest number of values that are live at a single
/// location as a rough estimate. Returns the register usage for each VF in \p
/// VFs.
SmallVector<VPRegisterUsage, 8> calculateRegisterUsageForPlan(
    VPlan &Plan, ArrayRef<ElementCount> VFs, const TargetTransformInfo &TTI,
    const SmallPtrSetImpl<const Value *> &ValuesToIgnore);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANANALYSIS_H
