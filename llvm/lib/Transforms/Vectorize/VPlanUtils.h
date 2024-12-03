//===- VPlanUtils.h - VPlan-related utilities -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANUTILS_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANUTILS_H

#include "VPlan.h"

namespace llvm {
class ScalarEvolution;
class SCEV;
} // namespace llvm

namespace llvm::vputils {
/// Returns true if only the first lane of \p Def is used.
bool onlyFirstLaneUsed(const VPValue *Def);

/// Returns true if only the first part of \p Def is used.
bool onlyFirstPartUsed(const VPValue *Def);

/// Get or create a VPValue that corresponds to the expansion of \p Expr. If \p
/// Expr is a SCEVConstant or SCEVUnknown, return a VPValue wrapping the live-in
/// value. Otherwise return a VPExpandSCEVRecipe to expand \p Expr. If \p Plan's
/// pre-header already contains a recipe expanding \p Expr, return it. If not,
/// create a new one.
VPValue *getOrCreateVPValueForSCEVExpr(VPlan &Plan, const SCEV *Expr,
                                       ScalarEvolution &SE);

/// Return the SCEV expression for \p V. Returns SCEVCouldNotCompute if no
/// SCEV expression could be constructed.
const SCEV *getSCEVExprForVPValue(VPValue *V, ScalarEvolution &SE);

/// Returns true if \p VPV is uniform after vectorization.
inline bool isUniformAfterVectorization(const VPValue *VPV) {
  // A value defined outside the vector region must be uniform after
  // vectorization inside a vector region.
  if (VPV->isDefinedOutsideLoopRegions())
    return true;
  const VPRecipeBase *Def = VPV->getDefiningRecipe();
  assert(Def && "Must have definition for value defined inside vector region");
  if (auto *Rep = dyn_cast<VPReplicateRecipe>(Def))
    return Rep->isUniform();
  if (auto *GEP = dyn_cast<VPWidenGEPRecipe>(Def))
    return all_of(GEP->operands(), isUniformAfterVectorization);
  if (auto *VPI = dyn_cast<VPInstruction>(Def))
    return VPI->isSingleScalar() || VPI->isVectorToScalar();
  return false;
}

/// Return true if \p V is a header mask in \p Plan.
bool isHeaderMask(const VPValue *V, VPlan &Plan);

/// Checks if \p V is uniform across all VF lanes and UF parts. It is considered
/// as such if it is either loop invariant (defined outside the vector region)
/// or its operand is known to be uniform across all VFs and UFs (e.g.
/// VPDerivedIV or VPCanonicalIVPHI).
bool isUniformAcrossVFsAndUFs(VPValue *V);

} // end namespace llvm::vputils

#endif
