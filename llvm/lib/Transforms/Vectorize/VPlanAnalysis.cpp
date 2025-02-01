//===- VPlanAnalysis.cpp - Various Analyses working on VPlan ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPlanAnalysis.h"
#include "VPlan.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

using namespace llvm;

#define DEBUG_TYPE "vplan"

Type *VPTypeAnalysis::inferScalarTypeForRecipe(const VPBlendRecipe *R) {
  Type *ResTy = inferScalarType(R->getIncomingValue(0));
  for (unsigned I = 1, E = R->getNumIncomingValues(); I != E; ++I) {
    VPValue *Inc = R->getIncomingValue(I);
    assert(inferScalarType(Inc) == ResTy &&
           "different types inferred for different incoming values");
    CachedTypes[Inc] = ResTy;
  }
  return ResTy;
}

Type *VPTypeAnalysis::inferScalarTypeForRecipe(const VPInstruction *R) {
  // Set the result type from the first operand, check if the types for all
  // other operands match and cache them.
  auto SetResultTyFromOp = [this, R]() {
    Type *ResTy = inferScalarType(R->getOperand(0));
    for (unsigned Op = 1; Op != R->getNumOperands(); ++Op) {
      VPValue *OtherV = R->getOperand(Op);
      assert(inferScalarType(OtherV) == ResTy &&
             "different types inferred for different operands");
      CachedTypes[OtherV] = ResTy;
    }
    return ResTy;
  };

  unsigned Opcode = R->getOpcode();
  if (Instruction::isBinaryOp(Opcode) || Instruction::isUnaryOp(Opcode))
    return SetResultTyFromOp();

  switch (Opcode) {
  case Instruction::Select: {
    Type *ResTy = inferScalarType(R->getOperand(1));
    VPValue *OtherV = R->getOperand(2);
    assert(inferScalarType(OtherV) == ResTy &&
           "different types inferred for different operands");
    CachedTypes[OtherV] = ResTy;
    return ResTy;
  }
  case Instruction::ICmp:
  case VPInstruction::ActiveLaneMask:
    assert(inferScalarType(R->getOperand(0)) ==
               inferScalarType(R->getOperand(1)) &&
           "different types inferred for different operands");
    return IntegerType::get(Ctx, 1);
  case VPInstruction::ComputeReductionResult: {
    auto *PhiR = cast<VPReductionPHIRecipe>(R->getOperand(0));
    auto *OrigPhi = cast<PHINode>(PhiR->getUnderlyingValue());
    return OrigPhi->getType();
  }
  case VPInstruction::ExplicitVectorLength:
    return Type::getIntNTy(Ctx, 32);
  case VPInstruction::FirstOrderRecurrenceSplice:
  case VPInstruction::Not:
  case VPInstruction::ResumePhi:
  case VPInstruction::CalculateTripCountMinusVF:
  case VPInstruction::CanonicalIVIncrementForPart:
  case VPInstruction::AnyOf:
    return SetResultTyFromOp();
  case VPInstruction::ExtractFirstActive:
  case VPInstruction::ExtractFromEnd: {
    Type *BaseTy = inferScalarType(R->getOperand(0));
    if (auto *VecTy = dyn_cast<VectorType>(BaseTy))
      return VecTy->getElementType();
    return BaseTy;
  }
  case VPInstruction::LogicalAnd:
    assert(inferScalarType(R->getOperand(0))->isIntegerTy(1) &&
           inferScalarType(R->getOperand(1))->isIntegerTy(1) &&
           "LogicalAnd operands should be bool");
    return IntegerType::get(Ctx, 1);
  case VPInstruction::Broadcast:
  case VPInstruction::PtrAdd:
    // Return the type based on the pointer argument (i.e. first operand).
    return inferScalarType(R->getOperand(0));
  case VPInstruction::BranchOnCond:
  case VPInstruction::BranchOnCount:
    return Type::getVoidTy(Ctx);
  default:
    break;
  }
  // Type inference not implemented for opcode.
  LLVM_DEBUG({
    dbgs() << "LV: Found unhandled opcode for: ";
    R->getVPSingleValue()->dump();
  });
  llvm_unreachable("Unhandled opcode!");
}

Type *VPTypeAnalysis::inferScalarTypeForRecipe(const VPWidenRecipe *R) {
  unsigned Opcode = R->getOpcode();
  if (Instruction::isBinaryOp(Opcode) || Instruction::isShift(Opcode) ||
      Instruction::isBitwiseLogicOp(Opcode)) {
    Type *ResTy = inferScalarType(R->getOperand(0));
    assert(ResTy == inferScalarType(R->getOperand(1)) &&
           "types for both operands must match for binary op");
    CachedTypes[R->getOperand(1)] = ResTy;
    return ResTy;
  }

  switch (Opcode) {
  case Instruction::ICmp:
  case Instruction::FCmp:
    return IntegerType::get(Ctx, 1);
  case Instruction::FNeg:
  case Instruction::Freeze:
    return inferScalarType(R->getOperand(0));
  default:
    break;
  }

  // Type inference not implemented for opcode.
  LLVM_DEBUG({
    dbgs() << "LV: Found unhandled opcode for: ";
    R->getVPSingleValue()->dump();
  });
  llvm_unreachable("Unhandled opcode!");
}

Type *VPTypeAnalysis::inferScalarTypeForRecipe(const VPWidenCallRecipe *R) {
  auto &CI = *cast<CallInst>(R->getUnderlyingInstr());
  return CI.getType();
}

Type *VPTypeAnalysis::inferScalarTypeForRecipe(const VPWidenMemoryRecipe *R) {
  assert((isa<VPWidenLoadRecipe, VPWidenLoadEVLRecipe>(R)) &&
         "Store recipes should not define any values");
  return cast<LoadInst>(&R->getIngredient())->getType();
}

Type *VPTypeAnalysis::inferScalarTypeForRecipe(const VPWidenSelectRecipe *R) {
  Type *ResTy = inferScalarType(R->getOperand(1));
  VPValue *OtherV = R->getOperand(2);
  assert(inferScalarType(OtherV) == ResTy &&
         "different types inferred for different operands");
  CachedTypes[OtherV] = ResTy;
  return ResTy;
}

Type *VPTypeAnalysis::inferScalarTypeForRecipe(const VPReplicateRecipe *R) {
  unsigned Opcode = R->getUnderlyingInstr()->getOpcode();

  if (Instruction::isBinaryOp(Opcode) || Instruction::isShift(Opcode) ||
      Instruction::isBitwiseLogicOp(Opcode)) {
    Type *ResTy = inferScalarType(R->getOperand(0));
    assert(ResTy == inferScalarType(R->getOperand(1)) &&
           "inferred types for operands of binary op don't match");
    CachedTypes[R->getOperand(1)] = ResTy;
    return ResTy;
  }

  if (Instruction::isCast(Opcode))
    return R->getUnderlyingInstr()->getType();

  switch (Opcode) {
  case Instruction::Call: {
    unsigned CallIdx = R->getNumOperands() - (R->isPredicated() ? 2 : 1);
    return cast<Function>(R->getOperand(CallIdx)->getLiveInIRValue())
        ->getReturnType();
  }
  case Instruction::Select: {
    Type *ResTy = inferScalarType(R->getOperand(1));
    assert(ResTy == inferScalarType(R->getOperand(2)) &&
           "inferred types for operands of select op don't match");
    CachedTypes[R->getOperand(2)] = ResTy;
    return ResTy;
  }
  case Instruction::ICmp:
  case Instruction::FCmp:
    return IntegerType::get(Ctx, 1);
  case Instruction::Alloca:
  case Instruction::ExtractValue:
    return R->getUnderlyingInstr()->getType();
  case Instruction::Freeze:
  case Instruction::FNeg:
  case Instruction::GetElementPtr:
    return inferScalarType(R->getOperand(0));
  case Instruction::Load:
    return cast<LoadInst>(R->getUnderlyingInstr())->getType();
  case Instruction::Store:
    // FIXME: VPReplicateRecipes with store opcodes still define a result
    // VPValue, so we need to handle them here. Remove the code here once this
    // is modeled accurately in VPlan.
    return Type::getVoidTy(Ctx);
  default:
    break;
  }
  // Type inference not implemented for opcode.
  LLVM_DEBUG({
    dbgs() << "LV: Found unhandled opcode for: ";
    R->getVPSingleValue()->dump();
  });
  llvm_unreachable("Unhandled opcode");
}

Type *VPTypeAnalysis::inferScalarType(const VPValue *V) {
  if (Type *CachedTy = CachedTypes.lookup(V))
    return CachedTy;

  if (V->isLiveIn()) {
    if (auto *IRValue = V->getLiveInIRValue())
      return IRValue->getType();
    // All VPValues without any underlying IR value (like the vector trip count
    // or the backedge-taken count) have the same type as the canonical IV.
    return CanonicalIVTy;
  }

  Type *ResultTy =
      TypeSwitch<const VPRecipeBase *, Type *>(V->getDefiningRecipe())
          .Case<VPActiveLaneMaskPHIRecipe, VPCanonicalIVPHIRecipe,
                VPFirstOrderRecurrencePHIRecipe, VPReductionPHIRecipe,
                VPWidenPointerInductionRecipe, VPEVLBasedIVPHIRecipe,
                VPScalarPHIRecipe>([this](const auto *R) {
            // Handle header phi recipes, except VPWidenIntOrFpInduction
            // which needs special handling due it being possibly truncated.
            // TODO: consider inferring/caching type of siblings, e.g.,
            // backedge value, here and in cases below.
            return inferScalarType(R->getStartValue());
          })
          .Case<VPWidenIntOrFpInductionRecipe, VPDerivedIVRecipe>(
              [](const auto *R) { return R->getScalarType(); })
          .Case<VPReductionRecipe, VPPredInstPHIRecipe, VPWidenPHIRecipe,
                VPScalarIVStepsRecipe, VPWidenGEPRecipe, VPVectorPointerRecipe,
                VPReverseVectorPointerRecipe, VPWidenCanonicalIVRecipe,
                VPPartialReductionRecipe>([this](const VPRecipeBase *R) {
            return inferScalarType(R->getOperand(0));
          })
          .Case<VPBlendRecipe, VPInstruction, VPWidenRecipe, VPWidenEVLRecipe,
                VPReplicateRecipe, VPWidenCallRecipe, VPWidenMemoryRecipe,
                VPWidenSelectRecipe>(
              [this](const auto *R) { return inferScalarTypeForRecipe(R); })
          .Case<VPWidenIntrinsicRecipe>([](const VPWidenIntrinsicRecipe *R) {
            return R->getResultType();
          })
          .Case<VPInterleaveRecipe>([V](const VPInterleaveRecipe *R) {
            // TODO: Use info from interleave group.
            return V->getUnderlyingValue()->getType();
          })
          .Case<VPWidenCastRecipe>(
              [](const VPWidenCastRecipe *R) { return R->getResultType(); })
          .Case<VPScalarCastRecipe>(
              [](const VPScalarCastRecipe *R) { return R->getResultType(); })
          .Case<VPExpandSCEVRecipe>([](const VPExpandSCEVRecipe *R) {
            return R->getSCEV()->getType();
          })
          .Case<VPReductionRecipe>([this](const auto *R) {
            return inferScalarType(R->getChainOp());
          });

  assert(ResultTy && "could not infer type for the given VPValue");
  CachedTypes[V] = ResultTy;
  return ResultTy;
}

void llvm::collectEphemeralRecipesForVPlan(
    VPlan &Plan, DenseSet<VPRecipeBase *> &EphRecipes) {
  // First, collect seed recipes which are operands of assumes.
  SmallVector<VPRecipeBase *> Worklist;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getVectorLoopRegion()->getEntry()))) {
    for (VPRecipeBase &R : *VPBB) {
      auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
      if (!RepR || !match(RepR->getUnderlyingInstr(),
                          PatternMatch::m_Intrinsic<Intrinsic::assume>()))
        continue;
      Worklist.push_back(RepR);
      EphRecipes.insert(RepR);
    }
  }

  // Process operands of candidates in worklist and add them to the set of
  // ephemeral recipes, if they don't have side-effects and are only used by
  // other ephemeral recipes.
  while (!Worklist.empty()) {
    VPRecipeBase *Cur = Worklist.pop_back_val();
    for (VPValue *Op : Cur->operands()) {
      auto *OpR = Op->getDefiningRecipe();
      if (!OpR || OpR->mayHaveSideEffects() || EphRecipes.contains(OpR))
        continue;
      if (any_of(Op->users(), [EphRecipes](VPUser *U) {
            auto *UR = dyn_cast<VPRecipeBase>(U);
            return !UR || !EphRecipes.contains(UR);
          }))
        continue;
      EphRecipes.insert(OpR);
      Worklist.push_back(OpR);
    }
  }
}

template void DomTreeBuilder::Calculate<DominatorTreeBase<VPBlockBase, false>>(
    DominatorTreeBase<VPBlockBase, false> &DT);

bool VPDominatorTree::properlyDominates(const VPRecipeBase *A,
                                        const VPRecipeBase *B) {
  if (A == B)
    return false;

  auto LocalComesBefore = [](const VPRecipeBase *A, const VPRecipeBase *B) {
    for (auto &R : *A->getParent()) {
      if (&R == A)
        return true;
      if (&R == B)
        return false;
    }
    llvm_unreachable("recipe not found");
  };
  const VPBlockBase *ParentA = A->getParent();
  const VPBlockBase *ParentB = B->getParent();
  if (ParentA == ParentB)
    return LocalComesBefore(A, B);

#ifndef NDEBUG
  auto GetReplicateRegion = [](VPRecipeBase *R) -> VPRegionBlock * {
    auto *Region = dyn_cast_or_null<VPRegionBlock>(R->getParent()->getParent());
    if (Region && Region->isReplicator()) {
      assert(Region->getNumSuccessors() == 1 &&
             Region->getNumPredecessors() == 1 && "Expected SESE region!");
      assert(R->getParent()->size() == 1 &&
             "A recipe in an original replicator region must be the only "
             "recipe in its block");
      return Region;
    }
    return nullptr;
  };
  assert(!GetReplicateRegion(const_cast<VPRecipeBase *>(A)) &&
         "No replicate regions expected at this point");
  assert(!GetReplicateRegion(const_cast<VPRecipeBase *>(B)) &&
         "No replicate regions expected at this point");
#endif
  return Base::properlyDominates(ParentA, ParentB);
}
