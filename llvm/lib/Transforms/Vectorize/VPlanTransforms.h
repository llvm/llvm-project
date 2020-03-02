//===- VPlanTransforms.h - Utility VPlan to VPlan transforms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utility VPlan to VPlan transformations.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H

#include "VPlan.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"

namespace llvm {

class VPlanTransforms {

public:
  /// Replaces the VPInstructions in \p Plan with corresponding
  /// widen recipes.
  static void VPInstructionsToVPRecipes(
      Loop *OrigLoop, VPlanPtr &Plan,
      LoopVectorizationLegality::InductionList &Inductions,
      SmallPtrSetImpl<Instruction *> &DeadInstructions);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H
