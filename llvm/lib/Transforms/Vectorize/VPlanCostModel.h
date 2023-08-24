//===- SiFive_VPlanCostModel.cpp - Vectorizer Cost Model ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// VPlan-based cost model
///
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Instruction.h"

#include "VPlan.h"
#include "VPlanValue.h"

namespace llvm {
class Type;
class TargetTransformInfo;
class LoopVectorizationCostModel;

class VPlanCostModel {
public:
  explicit VPlanCostModel(const TargetTransformInfo &TTI,
                          llvm::LLVMContext &Context,
                          LoopVectorizationCostModel &CM)
      : TTI(TTI), Context(Context), CM(CM) {}

  /// Return cost of the VPlan for a given \p VF
  InstructionCost expectedCost(const VPlan &Plan, ElementCount VF, bool &IsVec);

private:
  /// Return individual cost of the \p VPBasicBlock for a given \p VF
  InstructionCost getCost(const VPBlockBase *Block, ElementCount VF,
                          bool &IsVec);

  /// Return individual cost of the \p Recipe for a given \p VF
  InstructionCost getCost(const VPRecipeBase *Recipe, ElementCount VF,
                          bool &IsVec);

  /// Return individual cost of the \p Recipe for a given \p VF
  InstructionCost getLegacyInstructionCost(Instruction *I, ElementCount VF);

  InstructionCost getMemoryOpCost(const VPWidenMemoryInstructionRecipe *VPWMIR,
                                  ElementCount VF);

  /// Return cost of the individual memory operation of a instruction \p I of a
  /// given type \p Ty
  InstructionCost getMemoryOpCost(const Instruction *I, Type *Ty,
                                  bool IsConsecutive, bool IsMasked,
                                  bool IsReverse);

  Type *getElementType(const VPRecipeBase *Recipe, unsigned N) const;
  Type *getReturnElementType(const VPRecipeBase *Recipe) const;
  Type *truncateToMinimalBitwidth(Type *ValTy, Instruction *I) const;

  /// Vector target information.
  const TargetTransformInfo &TTI;

  LLVMContext &Context;

  /// FIXME: Legacy model is only here during our transition to the vplan-based
  /// model
  LoopVectorizationCostModel &CM;

  /// Use same cost kind in the cost model
  const TargetTransformInfo::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
};
} // namespace llvm
