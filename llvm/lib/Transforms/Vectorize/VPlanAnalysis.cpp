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
#include "VPlanHelpers.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;

#define DEBUG_TYPE "vplan"

VPTypeAnalysis::VPTypeAnalysis(const VPlan &Plan) : Ctx(Plan.getContext()) {
  if (auto LoopRegion = Plan.getVectorLoopRegion()) {
    if (const auto *CanIV = dyn_cast<VPCanonicalIVPHIRecipe>(
            &LoopRegion->getEntryBasicBlock()->front())) {
      CanonicalIVTy = CanIV->getScalarType();
      return;
    }
  }

  // If there's no canonical IV, retrieve the type from the trip count
  // expression.
  auto *TC = Plan.getTripCount();
  if (TC->isLiveIn()) {
    CanonicalIVTy = TC->getLiveInIRValue()->getType();
    return;
  }
  CanonicalIVTy = cast<VPExpandSCEVRecipe>(TC)->getSCEV()->getType();
}

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
  case Instruction::ExtractElement:
  case Instruction::Freeze:
  case VPInstruction::ReductionStartVector:
  case VPInstruction::ResumeForEpilogue:
    return inferScalarType(R->getOperand(0));
  case Instruction::Select: {
    Type *ResTy = inferScalarType(R->getOperand(1));
    VPValue *OtherV = R->getOperand(2);
    assert(inferScalarType(OtherV) == ResTy &&
           "different types inferred for different operands");
    CachedTypes[OtherV] = ResTy;
    return ResTy;
  }
  case Instruction::ICmp:
  case Instruction::FCmp:
  case VPInstruction::ActiveLaneMask:
    assert(inferScalarType(R->getOperand(0)) ==
               inferScalarType(R->getOperand(1)) &&
           "different types inferred for different operands");
    return IntegerType::get(Ctx, 1);
  case VPInstruction::ComputeAnyOfResult:
    return inferScalarType(R->getOperand(1));
  case VPInstruction::ComputeFindIVResult:
  case VPInstruction::ComputeReductionResult: {
    return inferScalarType(R->getOperand(0));
  }
  case VPInstruction::ExplicitVectorLength:
    return Type::getIntNTy(Ctx, 32);
  case Instruction::PHI:
    // Infer the type of first operand only, as other operands of header phi's
    // may lead to infinite recursion.
    return inferScalarType(R->getOperand(0));
  case VPInstruction::FirstOrderRecurrenceSplice:
  case VPInstruction::Not:
  case VPInstruction::CalculateTripCountMinusVF:
  case VPInstruction::CanonicalIVIncrementForPart:
  case VPInstruction::AnyOf:
  case VPInstruction::BuildStructVector:
  case VPInstruction::BuildVector:
    return SetResultTyFromOp();
  case VPInstruction::ExtractLane:
    return inferScalarType(R->getOperand(1));
  case VPInstruction::FirstActiveLane:
    return Type::getIntNTy(Ctx, 64);
  case VPInstruction::ExtractLastElement:
  case VPInstruction::ExtractLastLanePerPart:
  case VPInstruction::ExtractPenultimateElement: {
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
  case VPInstruction::WidePtrAdd:
    // Return the type based on first operand.
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
  case Instruction::ExtractValue: {
    assert(R->getNumOperands() == 2 && "expected single level extractvalue");
    auto *StructTy = cast<StructType>(inferScalarType(R->getOperand(0)));
    auto *CI = cast<ConstantInt>(R->getOperand(1)->getLiveInIRValue());
    return StructTy->getTypeAtIndex(CI->getZExtValue());
  }
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
                VPWidenPointerInductionRecipe, VPEVLBasedIVPHIRecipe>(
              [this](const auto *R) {
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
                VPVectorEndPointerRecipe, VPWidenCanonicalIVRecipe,
                VPPartialReductionRecipe>([this](const VPRecipeBase *R) {
            return inferScalarType(R->getOperand(0));
          })
          // VPInstructionWithType must be handled before VPInstruction.
          .Case<VPInstructionWithType, VPWidenIntrinsicRecipe,
                VPWidenCastRecipe>(
              [](const auto *R) { return R->getResultType(); })
          .Case<VPBlendRecipe, VPInstruction, VPWidenRecipe, VPReplicateRecipe,
                VPWidenCallRecipe, VPWidenMemoryRecipe, VPWidenSelectRecipe>(
              [this](const auto *R) { return inferScalarTypeForRecipe(R); })
          .Case<VPInterleaveBase>([V](const auto *R) {
            // TODO: Use info from interleave group.
            return V->getUnderlyingValue()->getType();
          })
          .Case<VPExpandSCEVRecipe>([](const VPExpandSCEVRecipe *R) {
            return R->getSCEV()->getType();
          })
          .Case<VPReductionRecipe>([this](const auto *R) {
            return inferScalarType(R->getChainOp());
          })
          .Case<VPExpressionRecipe>([this](const auto *R) {
            return inferScalarType(R->getOperandOfResultType());
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

bool VPRegisterUsage::exceedsMaxNumRegs(const TargetTransformInfo &TTI,
                                        unsigned OverrideMaxNumRegs) const {
  return any_of(MaxLocalUsers, [&TTI, &OverrideMaxNumRegs](auto &LU) {
    return LU.second > (OverrideMaxNumRegs > 0
                            ? OverrideMaxNumRegs
                            : TTI.getNumberOfRegisters(LU.first));
  });
}

SmallVector<VPRegisterUsage, 8> llvm::calculateRegisterUsageForPlan(
    VPlan &Plan, ArrayRef<ElementCount> VFs, const TargetTransformInfo &TTI,
    const SmallPtrSetImpl<const Value *> &ValuesToIgnore) {
  // Each 'key' in the map opens a new interval. The values
  // of the map are the index of the 'last seen' usage of the
  // VPValue that is the key.
  using IntervalMap = SmallDenseMap<VPValue *, unsigned, 16>;

  // Maps indices to recipes.
  SmallVector<VPRecipeBase *, 64> Idx2Recipe;
  // Marks the end of each interval.
  IntervalMap EndPoint;
  // Saves the list of VPValues that are used in the loop.
  SmallPtrSet<VPValue *, 8> Ends;
  // Saves the list of values that are used in the loop but are defined outside
  // the loop (not including non-recipe values such as arguments and
  // constants).
  SmallSetVector<VPValue *, 8> LoopInvariants;
  LoopInvariants.insert(&Plan.getVectorTripCount());

  // We scan the loop in a topological order in order and assign a number to
  // each recipe. We use RPO to ensure that defs are met before their users. We
  // assume that each recipe that has in-loop users starts an interval. We
  // record every time that an in-loop value is used, so we have a list of the
  // first occurences of each recipe and last occurrence of each VPValue.
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      LoopRegion);
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    if (!VPBB->getParent())
      break;
    for (VPRecipeBase &R : *VPBB) {
      Idx2Recipe.push_back(&R);

      // Save the end location of each USE.
      for (VPValue *U : R.operands()) {
        auto *DefR = U->getDefiningRecipe();

        // Ignore non-recipe values such as arguments, constants, etc.
        // FIXME: Might need some motivation why these values are ignored. If
        // for example an argument is used inside the loop it will increase the
        // register pressure (so shouldn't we add it to LoopInvariants).
        if (!DefR && (!U->getLiveInIRValue() ||
                      !isa<Instruction>(U->getLiveInIRValue())))
          continue;

        // If this recipe is outside the loop then record it and continue.
        if (!DefR) {
          LoopInvariants.insert(U);
          continue;
        }

        // Overwrite previous end points.
        EndPoint[U] = Idx2Recipe.size();
        Ends.insert(U);
      }
    }
    if (VPBB == LoopRegion->getExiting()) {
      // VPWidenIntOrFpInductionRecipes are used implicitly at the end of the
      // exiting block, where their increment will get materialized eventually.
      for (auto &R : LoopRegion->getEntryBasicBlock()->phis()) {
        if (auto *WideIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&R)) {
          EndPoint[WideIV] = Idx2Recipe.size();
          Ends.insert(WideIV);
        }
      }
    }
  }

  // Saves the list of intervals that end with the index in 'key'.
  using VPValueList = SmallVector<VPValue *, 2>;
  SmallDenseMap<unsigned, VPValueList, 16> TransposeEnds;

  // Next, we transpose the EndPoints into a multi map that holds the list of
  // intervals that *end* at a specific location.
  for (auto &Interval : EndPoint)
    TransposeEnds[Interval.second].push_back(Interval.first);

  SmallPtrSet<VPValue *, 8> OpenIntervals;
  SmallVector<VPRegisterUsage, 8> RUs(VFs.size());
  SmallVector<SmallMapVector<unsigned, unsigned, 4>, 8> MaxUsages(VFs.size());

  LLVM_DEBUG(dbgs() << "LV(REG): Calculating max register usage:\n");

  VPTypeAnalysis TypeInfo(Plan);

  const auto &TTICapture = TTI;
  auto GetRegUsage = [&TTICapture](Type *Ty, ElementCount VF) -> unsigned {
    if (Ty->isTokenTy() || !VectorType::isValidElementType(Ty) ||
        (VF.isScalable() &&
         !TTICapture.isElementTypeLegalForScalableVector(Ty)))
      return 0;
    return TTICapture.getRegUsageForType(VectorType::get(Ty, VF));
  };

  // We scan the instructions linearly and record each time that a new interval
  // starts, by placing it in a set. If we find this value in TransposEnds then
  // we remove it from the set. The max register usage is the maximum register
  // usage of the recipes of the set.
  for (unsigned int Idx = 0, Sz = Idx2Recipe.size(); Idx < Sz; ++Idx) {
    VPRecipeBase *R = Idx2Recipe[Idx];

    // Remove all of the VPValues that end at this location.
    VPValueList &List = TransposeEnds[Idx];
    for (VPValue *ToRemove : List)
      OpenIntervals.erase(ToRemove);

    // Ignore recipes that are never used within the loop and do not have side
    // effects.
    if (none_of(R->definedValues(),
                [&Ends](VPValue *Def) { return Ends.count(Def); }) &&
        !R->mayHaveSideEffects())
      continue;

    // Skip recipes for ignored values.
    // TODO: Should mark recipes for ephemeral values that cannot be removed
    // explictly in VPlan.
    if (isa<VPSingleDefRecipe>(R) &&
        ValuesToIgnore.contains(
            cast<VPSingleDefRecipe>(R)->getUnderlyingValue()))
      continue;

    // For each VF find the maximum usage of registers.
    for (unsigned J = 0, E = VFs.size(); J < E; ++J) {
      // Count the number of registers used, per register class, given all open
      // intervals.
      // Note that elements in this SmallMapVector will be default constructed
      // as 0. So we can use "RegUsage[ClassID] += n" in the code below even if
      // there is no previous entry for ClassID.
      SmallMapVector<unsigned, unsigned, 4> RegUsage;

      for (auto *VPV : OpenIntervals) {
        // Skip values that weren't present in the original loop.
        // TODO: Remove after removing the legacy
        // LoopVectorizationCostModel::calculateRegisterUsage
        if (isa<VPVectorPointerRecipe, VPVectorEndPointerRecipe,
                VPBranchOnMaskRecipe>(VPV))
          continue;

        if (VFs[J].isScalar() ||
            isa<VPCanonicalIVPHIRecipe, VPReplicateRecipe, VPDerivedIVRecipe,
                VPEVLBasedIVPHIRecipe, VPScalarIVStepsRecipe>(VPV) ||
            (isa<VPInstruction>(VPV) && vputils::onlyScalarValuesUsed(VPV)) ||
            (isa<VPReductionPHIRecipe>(VPV) &&
             (cast<VPReductionPHIRecipe>(VPV))->isInLoop())) {
          unsigned ClassID =
              TTI.getRegisterClassForType(false, TypeInfo.inferScalarType(VPV));
          // FIXME: The target might use more than one register for the type
          // even in the scalar case.
          RegUsage[ClassID] += 1;
        } else {
          // The output from scaled phis and scaled reductions actually has
          // fewer lanes than the VF.
          unsigned ScaleFactor =
              vputils::getVFScaleFactor(VPV->getDefiningRecipe());
          ElementCount VF = VFs[J].divideCoefficientBy(ScaleFactor);
          LLVM_DEBUG(if (VF != VFs[J]) {
            dbgs() << "LV(REG): Scaled down VF from " << VFs[J] << " to " << VF
                   << " for " << *R << "\n";
          });

          Type *ScalarTy = TypeInfo.inferScalarType(VPV);
          unsigned ClassID = TTI.getRegisterClassForType(true, ScalarTy);
          RegUsage[ClassID] += GetRegUsage(ScalarTy, VF);
        }
      }

      for (const auto &Pair : RegUsage) {
        auto &Entry = MaxUsages[J][Pair.first];
        Entry = std::max(Entry, Pair.second);
      }
    }

    LLVM_DEBUG(dbgs() << "LV(REG): At #" << Idx << " Interval # "
                      << OpenIntervals.size() << '\n');

    // Add used VPValues defined by the current recipe to the list of open
    // intervals.
    for (VPValue *DefV : R->definedValues())
      if (Ends.contains(DefV))
        OpenIntervals.insert(DefV);
  }

  // We also search for instructions that are defined outside the loop, but are
  // used inside the loop. We need this number separately from the max-interval
  // usage number because when we unroll, loop-invariant values do not take
  // more register.
  VPRegisterUsage RU;
  for (unsigned Idx = 0, End = VFs.size(); Idx < End; ++Idx) {
    // Note that elements in this SmallMapVector will be default constructed
    // as 0. So we can use "Invariant[ClassID] += n" in the code below even if
    // there is no previous entry for ClassID.
    SmallMapVector<unsigned, unsigned, 4> Invariant;

    for (auto *In : LoopInvariants) {
      // FIXME: The target might use more than one register for the type
      // even in the scalar case.
      bool IsScalar = vputils::onlyScalarValuesUsed(In);

      ElementCount VF = IsScalar ? ElementCount::getFixed(1) : VFs[Idx];
      unsigned ClassID = TTI.getRegisterClassForType(
          VF.isVector(), TypeInfo.inferScalarType(In));
      Invariant[ClassID] += GetRegUsage(TypeInfo.inferScalarType(In), VF);
    }

    LLVM_DEBUG({
      dbgs() << "LV(REG): VF = " << VFs[Idx] << '\n';
      dbgs() << "LV(REG): Found max usage: " << MaxUsages[Idx].size()
             << " item\n";
      for (const auto &pair : MaxUsages[Idx]) {
        dbgs() << "LV(REG): RegisterClass: "
               << TTI.getRegisterClassName(pair.first) << ", " << pair.second
               << " registers\n";
      }
      dbgs() << "LV(REG): Found invariant usage: " << Invariant.size()
             << " item\n";
      for (const auto &pair : Invariant) {
        dbgs() << "LV(REG): RegisterClass: "
               << TTI.getRegisterClassName(pair.first) << ", " << pair.second
               << " registers\n";
      }
    });

    RU.LoopInvariantRegs = Invariant;
    RU.MaxLocalUsers = MaxUsages[Idx];
    RUs[Idx] = RU;
  }

  return RUs;
}
