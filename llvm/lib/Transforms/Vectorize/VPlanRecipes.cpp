//===- VPlanRecipes.cpp - Implementations for VPlan recipes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains implementations for different VPlan recipes.
///
//===----------------------------------------------------------------------===//

#include "LoopVectorizationPlanner.h"
#include "VPlan.h"
#include "VPlanAnalysis.h"
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "VPlanUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <cassert>

using namespace llvm;

using VectorParts = SmallVector<Value *, 2>;

#define LV_NAME "loop-vectorize"
#define DEBUG_TYPE LV_NAME

bool VPRecipeBase::mayWriteToMemory() const {
  switch (getVPDefID()) {
  case VPExpressionSC:
    return cast<VPExpressionRecipe>(this)->mayReadOrWriteMemory();
  case VPInstructionSC:
    return cast<VPInstruction>(this)->opcodeMayReadOrWriteFromMemory();
  case VPInterleaveSC:
    return cast<VPInterleaveRecipe>(this)->getNumStoreOperands() > 0;
  case VPWidenStoreEVLSC:
  case VPWidenStoreSC:
    return true;
  case VPReplicateSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayWriteToMemory();
  case VPWidenCallSC:
    return !cast<VPWidenCallRecipe>(this)
                ->getCalledScalarFunction()
                ->onlyReadsMemory();
  case VPWidenIntrinsicSC:
    return cast<VPWidenIntrinsicRecipe>(this)->mayWriteToMemory();
  case VPCanonicalIVPHISC:
  case VPBranchOnMaskSC:
  case VPFirstOrderRecurrencePHISC:
  case VPReductionPHISC:
  case VPScalarIVStepsSC:
  case VPPredInstPHISC:
    return false;
  case VPBlendSC:
  case VPReductionEVLSC:
  case VPReductionSC:
  case VPVectorPointerSC:
  case VPWidenCanonicalIVSC:
  case VPWidenCastSC:
  case VPWidenGEPSC:
  case VPWidenIntOrFpInductionSC:
  case VPWidenLoadEVLSC:
  case VPWidenLoadSC:
  case VPWidenPHISC:
  case VPWidenSC:
  case VPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getVPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayWriteToMemory()) &&
           "underlying instruction may write to memory");
    return false;
  }
  default:
    return true;
  }
}

bool VPRecipeBase::mayReadFromMemory() const {
  switch (getVPDefID()) {
  case VPExpressionSC:
    return cast<VPExpressionRecipe>(this)->mayReadOrWriteMemory();
  case VPInstructionSC:
    return cast<VPInstruction>(this)->opcodeMayReadOrWriteFromMemory();
  case VPWidenLoadEVLSC:
  case VPWidenLoadSC:
    return true;
  case VPReplicateSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayReadFromMemory();
  case VPWidenCallSC:
    return !cast<VPWidenCallRecipe>(this)
                ->getCalledScalarFunction()
                ->onlyWritesMemory();
  case VPWidenIntrinsicSC:
    return cast<VPWidenIntrinsicRecipe>(this)->mayReadFromMemory();
  case VPBranchOnMaskSC:
  case VPFirstOrderRecurrencePHISC:
  case VPPredInstPHISC:
  case VPScalarIVStepsSC:
  case VPWidenStoreEVLSC:
  case VPWidenStoreSC:
    return false;
  case VPBlendSC:
  case VPReductionEVLSC:
  case VPReductionSC:
  case VPVectorPointerSC:
  case VPWidenCanonicalIVSC:
  case VPWidenCastSC:
  case VPWidenGEPSC:
  case VPWidenIntOrFpInductionSC:
  case VPWidenPHISC:
  case VPWidenSC:
  case VPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getVPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayReadFromMemory()) &&
           "underlying instruction may read from memory");
    return false;
  }
  default:
    return true;
  }
}

bool VPRecipeBase::mayHaveSideEffects() const {
  switch (getVPDefID()) {
  case VPExpressionSC:
    return cast<VPExpressionRecipe>(this)->mayHaveSideEffects();
  case VPDerivedIVSC:
  case VPFirstOrderRecurrencePHISC:
  case VPPredInstPHISC:
  case VPVectorEndPointerSC:
    return false;
  case VPInstructionSC:
    return mayWriteToMemory();
  case VPWidenCallSC: {
    Function *Fn = cast<VPWidenCallRecipe>(this)->getCalledScalarFunction();
    return mayWriteToMemory() || !Fn->doesNotThrow() || !Fn->willReturn();
  }
  case VPWidenIntrinsicSC:
    return cast<VPWidenIntrinsicRecipe>(this)->mayHaveSideEffects();
  case VPBlendSC:
  case VPReductionEVLSC:
  case VPReductionSC:
  case VPScalarIVStepsSC:
  case VPVectorPointerSC:
  case VPWidenCanonicalIVSC:
  case VPWidenCastSC:
  case VPWidenGEPSC:
  case VPWidenIntOrFpInductionSC:
  case VPWidenPHISC:
  case VPWidenPointerInductionSC:
  case VPWidenSC:
  case VPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getVPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayHaveSideEffects()) &&
           "underlying instruction has side-effects");
    return false;
  }
  case VPInterleaveSC:
    return mayWriteToMemory();
  case VPWidenLoadEVLSC:
  case VPWidenLoadSC:
  case VPWidenStoreEVLSC:
  case VPWidenStoreSC:
    assert(
        cast<VPWidenMemoryRecipe>(this)->getIngredient().mayHaveSideEffects() ==
            mayWriteToMemory() &&
        "mayHaveSideffects result for ingredient differs from this "
        "implementation");
    return mayWriteToMemory();
  case VPReplicateSC: {
    auto *R = cast<VPReplicateRecipe>(this);
    return R->getUnderlyingInstr()->mayHaveSideEffects();
  }
  default:
    return true;
  }
}

void VPRecipeBase::insertBefore(VPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  InsertPos->getParent()->insert(this, InsertPos->getIterator());
}

void VPRecipeBase::insertBefore(VPBasicBlock &BB,
                                iplist<VPRecipeBase>::iterator I) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(I == BB.end() || I->getParent() == &BB);
  BB.insert(this, I);
}

void VPRecipeBase::insertAfter(VPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  InsertPos->getParent()->insert(this, std::next(InsertPos->getIterator()));
}

void VPRecipeBase::removeFromParent() {
  assert(getParent() && "Recipe not in any VPBasicBlock");
  getParent()->getRecipeList().remove(getIterator());
  Parent = nullptr;
}

iplist<VPRecipeBase>::iterator VPRecipeBase::eraseFromParent() {
  assert(getParent() && "Recipe not in any VPBasicBlock");
  return getParent()->getRecipeList().erase(getIterator());
}

void VPRecipeBase::moveAfter(VPRecipeBase *InsertPos) {
  removeFromParent();
  insertAfter(InsertPos);
}

void VPRecipeBase::moveBefore(VPBasicBlock &BB,
                              iplist<VPRecipeBase>::iterator I) {
  removeFromParent();
  insertBefore(BB, I);
}

InstructionCost VPRecipeBase::cost(ElementCount VF, VPCostContext &Ctx) {
  // Get the underlying instruction for the recipe, if there is one. It is used
  // to
  //   * decide if cost computation should be skipped for this recipe,
  //   * apply forced target instruction cost.
  Instruction *UI = nullptr;
  if (auto *S = dyn_cast<VPSingleDefRecipe>(this))
    UI = dyn_cast_or_null<Instruction>(S->getUnderlyingValue());
  else if (auto *IG = dyn_cast<VPInterleaveRecipe>(this))
    UI = IG->getInsertPos();
  else if (auto *WidenMem = dyn_cast<VPWidenMemoryRecipe>(this))
    UI = &WidenMem->getIngredient();

  InstructionCost RecipeCost;
  if (UI && Ctx.skipCostComputation(UI, VF.isVector())) {
    RecipeCost = 0;
  } else {
    RecipeCost = computeCost(VF, Ctx);
    if (UI && ForceTargetInstructionCost.getNumOccurrences() > 0 &&
        RecipeCost.isValid())
      RecipeCost = InstructionCost(ForceTargetInstructionCost);
  }

  LLVM_DEBUG({
    dbgs() << "Cost of " << RecipeCost << " for VF " << VF << ": ";
    dump();
  });
  return RecipeCost;
}

InstructionCost VPRecipeBase::computeCost(ElementCount VF,
                                          VPCostContext &Ctx) const {
  llvm_unreachable("subclasses should implement computeCost");
}

bool VPRecipeBase::isPhi() const {
  return (getVPDefID() >= VPFirstPHISC && getVPDefID() <= VPLastPHISC) ||
         (isa<VPInstruction>(this) &&
          cast<VPInstruction>(this)->getOpcode() == Instruction::PHI) ||
         isa<VPIRPhi>(this);
}

bool VPRecipeBase::isScalarCast() const {
  auto *VPI = dyn_cast<VPInstruction>(this);
  return VPI && Instruction::isCast(VPI->getOpcode());
}

InstructionCost
VPPartialReductionRecipe::computeCost(ElementCount VF,
                                      VPCostContext &Ctx) const {
  std::optional<unsigned> Opcode;
  VPValue *Op = getOperand(0);
  VPRecipeBase *OpR = Op->getDefiningRecipe();

  // If the partial reduction is predicated, a select will be operand 0
  using namespace llvm::VPlanPatternMatch;
  if (match(getOperand(1), m_Select(m_VPValue(), m_VPValue(Op), m_VPValue()))) {
    OpR = Op->getDefiningRecipe();
  }

  Type *InputTypeA = nullptr, *InputTypeB = nullptr;
  TTI::PartialReductionExtendKind ExtAType = TTI::PR_None,
                                  ExtBType = TTI::PR_None;

  auto GetExtendKind = [](VPRecipeBase *R) {
    if (!R)
      return TTI::PR_None;
    auto *WidenCastR = dyn_cast<VPWidenCastRecipe>(R);
    if (!WidenCastR)
      return TTI::PR_None;
    if (WidenCastR->getOpcode() == Instruction::CastOps::ZExt)
      return TTI::PR_ZeroExtend;
    if (WidenCastR->getOpcode() == Instruction::CastOps::SExt)
      return TTI::PR_SignExtend;
    return TTI::PR_None;
  };

  // Pick out opcode, type/ext information and use sub side effects from a widen
  // recipe.
  auto HandleWiden = [&](VPWidenRecipe *Widen) {
    if (match(Widen,
              m_Binary<Instruction::Sub>(m_SpecificInt(0), m_VPValue(Op)))) {
      Widen = dyn_cast<VPWidenRecipe>(Op->getDefiningRecipe());
    }
    Opcode = Widen->getOpcode();
    VPRecipeBase *ExtAR = Widen->getOperand(0)->getDefiningRecipe();
    VPRecipeBase *ExtBR = Widen->getOperand(1)->getDefiningRecipe();
    InputTypeA = Ctx.Types.inferScalarType(ExtAR ? ExtAR->getOperand(0)
                                                 : Widen->getOperand(0));
    InputTypeB = Ctx.Types.inferScalarType(ExtBR ? ExtBR->getOperand(0)
                                                 : Widen->getOperand(1));
    ExtAType = GetExtendKind(ExtAR);
    ExtBType = GetExtendKind(ExtBR);
  };

  if (isa<VPWidenCastRecipe>(OpR)) {
    InputTypeA = Ctx.Types.inferScalarType(OpR->getOperand(0));
    ExtAType = GetExtendKind(OpR);
  } else if (isa<VPReductionPHIRecipe>(OpR)) {
    auto RedPhiOp1R = getOperand(1)->getDefiningRecipe();
    if (isa<VPWidenCastRecipe>(RedPhiOp1R)) {
      InputTypeA = Ctx.Types.inferScalarType(RedPhiOp1R->getOperand(0));
      ExtAType = GetExtendKind(RedPhiOp1R);
    } else if (auto Widen = dyn_cast<VPWidenRecipe>(RedPhiOp1R))
      HandleWiden(Widen);
  } else if (auto Widen = dyn_cast<VPWidenRecipe>(OpR)) {
    HandleWiden(Widen);
  } else if (auto Reduction = dyn_cast<VPPartialReductionRecipe>(OpR)) {
    return Reduction->computeCost(VF, Ctx);
  }
  auto *PhiType = Ctx.Types.inferScalarType(getOperand(1));
  return Ctx.TTI.getPartialReductionCost(getOpcode(), InputTypeA, InputTypeB,
                                         PhiType, VF, ExtAType, ExtBType,
                                         Opcode, Ctx.CostKind);
}

void VPPartialReductionRecipe::execute(VPTransformState &State) {
  auto &Builder = State.Builder;

  assert(getOpcode() == Instruction::Add &&
         "Unhandled partial reduction opcode");

  Value *BinOpVal = State.get(getOperand(1));
  Value *PhiVal = State.get(getOperand(0));
  assert(PhiVal && BinOpVal && "Phi and Mul must be set");

  Type *RetTy = PhiVal->getType();

  CallInst *V = Builder.CreateIntrinsic(
      RetTy, Intrinsic::experimental_vector_partial_reduce_add,
      {PhiVal, BinOpVal}, nullptr, "partial.reduce");

  State.set(this, V);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPPartialReductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                     VPSlotTracker &SlotTracker) const {
  O << Indent << "PARTIAL-REDUCE ";
  printAsOperand(O, SlotTracker);
  O << " = " << Instruction::getOpcodeName(getOpcode()) << " ";
  printOperands(O, SlotTracker);
}
#endif

FastMathFlags VPIRFlags::getFastMathFlags() const {
  assert(OpType == OperationType::FPMathOp &&
         "recipe doesn't have fast math flags");
  FastMathFlags Res;
  Res.setAllowReassoc(FMFs.AllowReassoc);
  Res.setNoNaNs(FMFs.NoNaNs);
  Res.setNoInfs(FMFs.NoInfs);
  Res.setNoSignedZeros(FMFs.NoSignedZeros);
  Res.setAllowReciprocal(FMFs.AllowReciprocal);
  Res.setAllowContract(FMFs.AllowContract);
  Res.setApproxFunc(FMFs.ApproxFunc);
  return Res;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPSingleDefRecipe::dump() const { VPDef::dump(); }
#endif

template <unsigned PartOpIdx>
VPValue *
VPUnrollPartAccessor<PartOpIdx>::getUnrollPartOperand(VPUser &U) const {
  if (U.getNumOperands() == PartOpIdx + 1)
    return U.getOperand(PartOpIdx);
  return nullptr;
}

template <unsigned PartOpIdx>
unsigned VPUnrollPartAccessor<PartOpIdx>::getUnrollPart(VPUser &U) const {
  if (auto *UnrollPartOp = getUnrollPartOperand(U))
    return cast<ConstantInt>(UnrollPartOp->getLiveInIRValue())->getZExtValue();
  return 0;
}

namespace llvm {
template class VPUnrollPartAccessor<2>;
template class VPUnrollPartAccessor<3>;
}

VPInstruction::VPInstruction(unsigned Opcode, ArrayRef<VPValue *> Operands,
                             const VPIRFlags &Flags, DebugLoc DL,
                             const Twine &Name)
    : VPRecipeWithIRFlags(VPDef::VPInstructionSC, Operands, Flags, DL),
      VPIRMetadata(), Opcode(Opcode), Name(Name.str()) {
  assert(flagsValidForOpcode(getOpcode()) &&
         "Set flags not supported for the provided opcode");
  assert((getNumOperandsForOpcode(Opcode) == -1u ||
          getNumOperandsForOpcode(Opcode) == getNumOperands()) &&
         "number of operands does not match opcode");
}

#ifndef NDEBUG
unsigned VPInstruction::getNumOperandsForOpcode(unsigned Opcode) {
  if (Instruction::isUnaryOp(Opcode) || Instruction::isCast(Opcode))
    return 1;

  if (Instruction::isBinaryOp(Opcode))
    return 2;

  switch (Opcode) {
  case VPInstruction::StepVector:
    return 0;
  case Instruction::Alloca:
  case Instruction::ExtractValue:
  case Instruction::Freeze:
  case Instruction::Load:
  case VPInstruction::AnyOf:
  case VPInstruction::BranchOnCond:
  case VPInstruction::CalculateTripCountMinusVF:
  case VPInstruction::CanonicalIVIncrementForPart:
  case VPInstruction::ExplicitVectorLength:
  case VPInstruction::ExtractLastElement:
  case VPInstruction::ExtractPenultimateElement:
  case VPInstruction::FirstActiveLane:
  case VPInstruction::Not:
    return 1;
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::Store:
  case VPInstruction::ActiveLaneMask:
  case VPInstruction::BranchOnCount:
  case VPInstruction::ComputeReductionResult:
  case VPInstruction::FirstOrderRecurrenceSplice:
  case VPInstruction::LogicalAnd:
  case VPInstruction::PtrAdd:
  case VPInstruction::WideIVStep:
    return 2;
  case Instruction::Select:
  case VPInstruction::ComputeAnyOfResult:
  case VPInstruction::ReductionStartVector:
    return 3;
  case VPInstruction::ComputeFindIVResult:
    return 4;
  case Instruction::Call:
  case Instruction::GetElementPtr:
  case Instruction::PHI:
  case Instruction::Switch:
    // Cannot determine the number of operands from the opcode.
    return -1u;
  }
  llvm_unreachable("all cases should be handled above");
}
#endif

bool VPInstruction::doesGeneratePerAllLanes() const {
  return Opcode == VPInstruction::PtrAdd && !vputils::onlyFirstLaneUsed(this);
}

bool VPInstruction::canGenerateScalarForFirstLane() const {
  if (Instruction::isBinaryOp(getOpcode()) || Instruction::isCast(getOpcode()))
    return true;
  if (isSingleScalar() || isVectorToScalar())
    return true;
  switch (Opcode) {
  case Instruction::Freeze:
  case Instruction::ICmp:
  case Instruction::PHI:
  case Instruction::Select:
  case VPInstruction::BranchOnCond:
  case VPInstruction::BranchOnCount:
  case VPInstruction::CalculateTripCountMinusVF:
  case VPInstruction::CanonicalIVIncrementForPart:
  case VPInstruction::PtrAdd:
  case VPInstruction::ExplicitVectorLength:
  case VPInstruction::AnyOf:
    return true;
  default:
    return false;
  }
}

Value *VPInstruction::generatePerLane(VPTransformState &State,
                                      const VPLane &Lane) {
  IRBuilderBase &Builder = State.Builder;

  assert(getOpcode() == VPInstruction::PtrAdd &&
         "only PtrAdd opcodes are supported for now");
  return Builder.CreatePtrAdd(State.get(getOperand(0), Lane),
                              State.get(getOperand(1), Lane), Name);
}

/// Create a conditional branch using \p Cond branching to the successors of \p
/// VPBB. Note that the first successor is always forward (i.e. not created yet)
/// while the second successor may already have been created (if it is a header
/// block and VPBB is a latch).
static BranchInst *createCondBranch(Value *Cond, VPBasicBlock *VPBB,
                                    VPTransformState &State) {
  // Replace the temporary unreachable terminator with a new conditional
  // branch, hooking it up to backward destination (header) for latch blocks
  // now, and to forward destination(s) later when they are created.
  // Second successor may be backwards - iff it is already in VPBB2IRBB.
  VPBasicBlock *SecondVPSucc = cast<VPBasicBlock>(VPBB->getSuccessors()[1]);
  BasicBlock *SecondIRSucc = State.CFG.VPBB2IRBB.lookup(SecondVPSucc);
  BasicBlock *IRBB = State.CFG.VPBB2IRBB[VPBB];
  BranchInst *CondBr = State.Builder.CreateCondBr(Cond, IRBB, SecondIRSucc);
  // First successor is always forward, reset it to nullptr
  CondBr->setSuccessor(0, nullptr);
  IRBB->getTerminator()->eraseFromParent();
  return CondBr;
}

Value *VPInstruction::generate(VPTransformState &State) {
  IRBuilderBase &Builder = State.Builder;

  if (Instruction::isBinaryOp(getOpcode())) {
    bool OnlyFirstLaneUsed = vputils::onlyFirstLaneUsed(this);
    Value *A = State.get(getOperand(0), OnlyFirstLaneUsed);
    Value *B = State.get(getOperand(1), OnlyFirstLaneUsed);
    auto *Res =
        Builder.CreateBinOp((Instruction::BinaryOps)getOpcode(), A, B, Name);
    if (auto *I = dyn_cast<Instruction>(Res))
      applyFlags(*I);
    return Res;
  }

  switch (getOpcode()) {
  case VPInstruction::Not: {
    Value *A = State.get(getOperand(0));
    return Builder.CreateNot(A, Name);
  }
  case Instruction::ExtractElement: {
    assert(State.VF.isVector() && "Only extract elements from vectors");
    if (getOperand(1)->isLiveIn()) {
      unsigned IdxToExtract =
          cast<ConstantInt>(getOperand(1)->getLiveInIRValue())->getZExtValue();
      return State.get(getOperand(0), VPLane(IdxToExtract));
    }
    Value *Vec = State.get(getOperand(0));
    Value *Idx = State.get(getOperand(1), /*IsScalar=*/true);
    return Builder.CreateExtractElement(Vec, Idx, Name);
  }
  case Instruction::Freeze: {
    Value *Op = State.get(getOperand(0), vputils::onlyFirstLaneUsed(this));
    return Builder.CreateFreeze(Op, Name);
  }
  case Instruction::FCmp:
  case Instruction::ICmp: {
    bool OnlyFirstLaneUsed = vputils::onlyFirstLaneUsed(this);
    Value *A = State.get(getOperand(0), OnlyFirstLaneUsed);
    Value *B = State.get(getOperand(1), OnlyFirstLaneUsed);
    return Builder.CreateCmp(getPredicate(), A, B, Name);
  }
  case Instruction::PHI: {
    llvm_unreachable("should be handled by VPPhi::execute");
  }
  case Instruction::Select: {
    bool OnlyFirstLaneUsed = vputils::onlyFirstLaneUsed(this);
    Value *Cond = State.get(getOperand(0), OnlyFirstLaneUsed);
    Value *Op1 = State.get(getOperand(1), OnlyFirstLaneUsed);
    Value *Op2 = State.get(getOperand(2), OnlyFirstLaneUsed);
    return Builder.CreateSelect(Cond, Op1, Op2, Name);
  }
  case VPInstruction::ActiveLaneMask: {
    // Get first lane of vector induction variable.
    Value *VIVElem0 = State.get(getOperand(0), VPLane(0));
    // Get the original loop tripcount.
    Value *ScalarTC = State.get(getOperand(1), VPLane(0));

    // If this part of the active lane mask is scalar, generate the CMP directly
    // to avoid unnecessary extracts.
    if (State.VF.isScalar())
      return Builder.CreateCmp(CmpInst::Predicate::ICMP_ULT, VIVElem0, ScalarTC,
                               Name);

    auto *Int1Ty = Type::getInt1Ty(Builder.getContext());
    auto *PredTy = VectorType::get(Int1Ty, State.VF);
    return Builder.CreateIntrinsic(Intrinsic::get_active_lane_mask,
                                   {PredTy, ScalarTC->getType()},
                                   {VIVElem0, ScalarTC}, nullptr, Name);
  }
  case VPInstruction::FirstOrderRecurrenceSplice: {
    // Generate code to combine the previous and current values in vector v3.
    //
    //   vector.ph:
    //     v_init = vector(..., ..., ..., a[-1])
    //     br vector.body
    //
    //   vector.body
    //     i = phi [0, vector.ph], [i+4, vector.body]
    //     v1 = phi [v_init, vector.ph], [v2, vector.body]
    //     v2 = a[i, i+1, i+2, i+3];
    //     v3 = vector(v1(3), v2(0, 1, 2))

    auto *V1 = State.get(getOperand(0));
    if (!V1->getType()->isVectorTy())
      return V1;
    Value *V2 = State.get(getOperand(1));
    return Builder.CreateVectorSplice(V1, V2, -1, Name);
  }
  case VPInstruction::CalculateTripCountMinusVF: {
    unsigned UF = getParent()->getPlan()->getUF();
    Value *ScalarTC = State.get(getOperand(0), VPLane(0));
    Value *Step = createStepForVF(Builder, ScalarTC->getType(), State.VF, UF);
    Value *Sub = Builder.CreateSub(ScalarTC, Step);
    Value *Cmp = Builder.CreateICmp(CmpInst::Predicate::ICMP_UGT, ScalarTC, Step);
    Value *Zero = ConstantInt::get(ScalarTC->getType(), 0);
    return Builder.CreateSelect(Cmp, Sub, Zero);
  }
  case VPInstruction::ExplicitVectorLength: {
    // TODO: Restructure this code with an explicit remainder loop, vsetvli can
    // be outside of the main loop.
    Value *AVL = State.get(getOperand(0), /*IsScalar*/ true);
    // Compute EVL
    assert(AVL->getType()->isIntegerTy() &&
           "Requested vector length should be an integer.");

    assert(State.VF.isScalable() && "Expected scalable vector factor.");
    Value *VFArg = State.Builder.getInt32(State.VF.getKnownMinValue());

    Value *EVL = State.Builder.CreateIntrinsic(
        State.Builder.getInt32Ty(), Intrinsic::experimental_get_vector_length,
        {AVL, VFArg, State.Builder.getTrue()});
    return EVL;
  }
  case VPInstruction::CanonicalIVIncrementForPart: {
    unsigned Part = getUnrollPart(*this);
    auto *IV = State.get(getOperand(0), VPLane(0));
    assert(Part != 0 && "Must have a positive part");
    // The canonical IV is incremented by the vectorization factor (num of
    // SIMD elements) times the unroll part.
    Value *Step = createStepForVF(Builder, IV->getType(), State.VF, Part);
    return Builder.CreateAdd(IV, Step, Name, hasNoUnsignedWrap(),
                             hasNoSignedWrap());
  }
  case VPInstruction::BranchOnCond: {
    Value *Cond = State.get(getOperand(0), VPLane(0));
    auto *Br = createCondBranch(Cond, getParent(), State);
    applyMetadata(*Br);
    return Br;
  }
  case VPInstruction::BranchOnCount: {
    // First create the compare.
    Value *IV = State.get(getOperand(0), /*IsScalar*/ true);
    Value *TC = State.get(getOperand(1), /*IsScalar*/ true);
    Value *Cond = Builder.CreateICmpEQ(IV, TC);
    return createCondBranch(Cond, getParent(), State);
  }
  case VPInstruction::Broadcast: {
    return Builder.CreateVectorSplat(
        State.VF, State.get(getOperand(0), /*IsScalar*/ true), "broadcast");
  }
  case VPInstruction::BuildStructVector: {
    // For struct types, we need to build a new 'wide' struct type, where each
    // element is widened, i.e., we create a struct of vectors.
    auto *StructTy =
        cast<StructType>(State.TypeAnalysis.inferScalarType(getOperand(0)));
    Value *Res = PoisonValue::get(toVectorizedTy(StructTy, State.VF));
    for (const auto &[LaneIndex, Op] : enumerate(operands())) {
      for (unsigned FieldIndex = 0; FieldIndex != StructTy->getNumElements();
           FieldIndex++) {
        Value *ScalarValue =
            Builder.CreateExtractValue(State.get(Op, true), FieldIndex);
        Value *VectorValue = Builder.CreateExtractValue(Res, FieldIndex);
        VectorValue =
            Builder.CreateInsertElement(VectorValue, ScalarValue, LaneIndex);
        Res = Builder.CreateInsertValue(Res, VectorValue, FieldIndex);
      }
    }
    return Res;
  }
  case VPInstruction::BuildVector: {
    auto *ScalarTy = State.TypeAnalysis.inferScalarType(getOperand(0));
    auto NumOfElements = ElementCount::getFixed(getNumOperands());
    Value *Res = PoisonValue::get(toVectorizedTy(ScalarTy, NumOfElements));
    for (const auto &[Idx, Op] : enumerate(operands()))
      Res = State.Builder.CreateInsertElement(Res, State.get(Op, true),
                                              State.Builder.getInt32(Idx));
    return Res;
  }
  case VPInstruction::ReductionStartVector: {
    if (State.VF.isScalar())
      return State.get(getOperand(0), true);
    IRBuilderBase::FastMathFlagGuard FMFG(Builder);
    Builder.setFastMathFlags(getFastMathFlags());
    // If this start vector is scaled then it should produce a vector with fewer
    // elements than the VF.
    ElementCount VF = State.VF.divideCoefficientBy(
        cast<ConstantInt>(getOperand(2)->getLiveInIRValue())->getZExtValue());
    auto *Iden = Builder.CreateVectorSplat(VF, State.get(getOperand(1), true));
    Constant *Zero = Builder.getInt32(0);
    return Builder.CreateInsertElement(Iden, State.get(getOperand(0), true),
                                       Zero);
  }
  case VPInstruction::ComputeAnyOfResult: {
    // FIXME: The cross-recipe dependency on VPReductionPHIRecipe is temporary
    // and will be removed by breaking up the recipe further.
    auto *PhiR = cast<VPReductionPHIRecipe>(getOperand(0));
    auto *OrigPhi = cast<PHINode>(PhiR->getUnderlyingValue());
    Value *ReducedPartRdx = State.get(getOperand(2));
    for (unsigned Idx = 3; Idx < getNumOperands(); ++Idx)
      ReducedPartRdx = Builder.CreateBinOp(
          (Instruction::BinaryOps)RecurrenceDescriptor::getOpcode(
              RecurKind::AnyOf),
          State.get(getOperand(Idx)), ReducedPartRdx, "bin.rdx");
    return createAnyOfReduction(Builder, ReducedPartRdx,
                                State.get(getOperand(1), VPLane(0)), OrigPhi);
  }
  case VPInstruction::ComputeFindIVResult: {
    // FIXME: The cross-recipe dependency on VPReductionPHIRecipe is temporary
    // and will be removed by breaking up the recipe further.
    auto *PhiR = cast<VPReductionPHIRecipe>(getOperand(0));
    // Get its reduction variable descriptor.
    RecurKind RK = PhiR->getRecurrenceKind();
    assert(RecurrenceDescriptor::isFindIVRecurrenceKind(RK) &&
           "Unexpected reduction kind");
    assert(!PhiR->isInLoop() &&
           "In-loop FindLastIV reduction is not supported yet");

    // The recipe's operands are the reduction phi, the start value, the
    // sentinel value, followed by one operand for each part of the reduction.
    unsigned UF = getNumOperands() - 3;
    Value *ReducedPartRdx = State.get(getOperand(3));
    RecurKind MinMaxKind;
    bool IsSigned = RecurrenceDescriptor::isSignedRecurrenceKind(RK);
    if (RecurrenceDescriptor::isFindLastIVRecurrenceKind(RK))
      MinMaxKind = IsSigned ? RecurKind::SMax : RecurKind::UMax;
    else
      MinMaxKind = IsSigned ? RecurKind::SMin : RecurKind::UMin;
    for (unsigned Part = 1; Part < UF; ++Part)
      ReducedPartRdx = createMinMaxOp(Builder, MinMaxKind, ReducedPartRdx,
                                      State.get(getOperand(3 + Part)));

    Value *Start = State.get(getOperand(1), true);
    Value *Sentinel = getOperand(2)->getLiveInIRValue();
    return createFindLastIVReduction(Builder, ReducedPartRdx, RK, Start,
                                     Sentinel);
  }
  case VPInstruction::ComputeReductionResult: {
    // FIXME: The cross-recipe dependency on VPReductionPHIRecipe is temporary
    // and will be removed by breaking up the recipe further.
    auto *PhiR = cast<VPReductionPHIRecipe>(getOperand(0));
    // Get its reduction variable descriptor.

    RecurKind RK = PhiR->getRecurrenceKind();
    assert(!RecurrenceDescriptor::isFindIVRecurrenceKind(RK) &&
           "should be handled by ComputeFindIVResult");

    // The recipe's operands are the reduction phi, followed by one operand for
    // each part of the reduction.
    unsigned UF = getNumOperands() - 1;
    VectorParts RdxParts(UF);
    for (unsigned Part = 0; Part < UF; ++Part)
      RdxParts[Part] = State.get(getOperand(1 + Part), PhiR->isInLoop());

    IRBuilderBase::FastMathFlagGuard FMFG(Builder);
    if (hasFastMathFlags())
      Builder.setFastMathFlags(getFastMathFlags());

    // Reduce all of the unrolled parts into a single vector.
    Value *ReducedPartRdx = RdxParts[0];
    if (PhiR->isOrdered()) {
      ReducedPartRdx = RdxParts[UF - 1];
    } else {
      // Floating-point operations should have some FMF to enable the reduction.
      for (unsigned Part = 1; Part < UF; ++Part) {
        Value *RdxPart = RdxParts[Part];
        if (RecurrenceDescriptor::isMinMaxRecurrenceKind(RK))
          ReducedPartRdx = createMinMaxOp(Builder, RK, ReducedPartRdx, RdxPart);
        else
          ReducedPartRdx = Builder.CreateBinOp(
              (Instruction::BinaryOps)RecurrenceDescriptor::getOpcode(RK),
              RdxPart, ReducedPartRdx, "bin.rdx");
      }
    }

    // Create the reduction after the loop. Note that inloop reductions create
    // the target reduction in the loop using a Reduction recipe.
    if (State.VF.isVector() && !PhiR->isInLoop()) {
      // TODO: Support in-order reductions based on the recurrence descriptor.
      // All ops in the reduction inherit fast-math-flags from the recurrence
      // descriptor.
      ReducedPartRdx = createSimpleReduction(Builder, ReducedPartRdx, RK);
    }

    return ReducedPartRdx;
  }
  case VPInstruction::ExtractLastElement:
  case VPInstruction::ExtractPenultimateElement: {
    unsigned Offset = getOpcode() == VPInstruction::ExtractLastElement ? 1 : 2;
    Value *Res;
    if (State.VF.isVector()) {
      assert(Offset <= State.VF.getKnownMinValue() &&
             "invalid offset to extract from");
      // Extract lane VF - Offset from the operand.
      Res = State.get(getOperand(0), VPLane::getLaneFromEnd(State.VF, Offset));
    } else {
      assert(Offset <= 1 && "invalid offset to extract from");
      Res = State.get(getOperand(0));
    }
    if (isa<ExtractElementInst>(Res))
      Res->setName(Name);
    return Res;
  }
  case VPInstruction::LogicalAnd: {
    Value *A = State.get(getOperand(0));
    Value *B = State.get(getOperand(1));
    return Builder.CreateLogicalAnd(A, B, Name);
  }
  case VPInstruction::PtrAdd: {
    assert(vputils::onlyFirstLaneUsed(this) &&
           "can only generate first lane for PtrAdd");
    Value *Ptr = State.get(getOperand(0), VPLane(0));
    Value *Addend = State.get(getOperand(1), VPLane(0));
    return Builder.CreatePtrAdd(Ptr, Addend, Name, getGEPNoWrapFlags());
  }
  case VPInstruction::AnyOf: {
    Value *Res = State.get(getOperand(0));
    for (VPValue *Op : drop_begin(operands()))
      Res = Builder.CreateOr(Res, State.get(Op));
    return State.VF.isScalar() ? Res : Builder.CreateOrReduce(Res);
  }
  case VPInstruction::FirstActiveLane: {
    if (getNumOperands() == 1) {
      Value *Mask = State.get(getOperand(0));
      return Builder.CreateCountTrailingZeroElems(Builder.getInt64Ty(), Mask,
                                                  true, Name);
    }
    // If there are multiple operands, create a chain of selects to pick the
    // first operand with an active lane and add the number of lanes of the
    // preceding operands.
    Value *RuntimeVF =
        getRuntimeVF(State.Builder, State.Builder.getInt64Ty(), State.VF);
    unsigned LastOpIdx = getNumOperands() - 1;
    Value *Res = nullptr;
    for (int Idx = LastOpIdx; Idx >= 0; --Idx) {
      Value *TrailingZeros = Builder.CreateCountTrailingZeroElems(
          Builder.getInt64Ty(), State.get(getOperand(Idx)), true, Name);
      Value *Current = Builder.CreateAdd(
          Builder.CreateMul(RuntimeVF, Builder.getInt64(Idx)), TrailingZeros);
      if (Res) {
        Value *Cmp = Builder.CreateICmpNE(TrailingZeros, RuntimeVF);
        Res = Builder.CreateSelect(Cmp, Current, Res);
      } else {
        Res = Current;
      }
    }

    return Res;
  }
  default:
    llvm_unreachable("Unsupported opcode for instruction");
  }
}

InstructionCost VPInstruction::computeCost(ElementCount VF,
                                           VPCostContext &Ctx) const {
  if (Instruction::isBinaryOp(getOpcode())) {
    Type *ResTy = Ctx.Types.inferScalarType(this);
    if (!vputils::onlyFirstLaneUsed(this))
      ResTy = toVectorTy(ResTy, VF);

    if (!getUnderlyingValue()) {
      switch (getOpcode()) {
      case Instruction::FMul:
        return Ctx.TTI.getArithmeticInstrCost(getOpcode(), ResTy, Ctx.CostKind);
      default:
        // TODO: Compute cost for VPInstructions without underlying values once
        // the legacy cost model has been retired.
        return 0;
      }
    }

    assert(!doesGeneratePerAllLanes() &&
           "Should only generate a vector value or single scalar, not scalars "
           "for all lanes.");
    return Ctx.TTI.getArithmeticInstrCost(getOpcode(), ResTy, Ctx.CostKind);
  }

  switch (getOpcode()) {
  case Instruction::ExtractElement: {
    // Add on the cost of extracting the element.
    auto *VecTy = toVectorTy(Ctx.Types.inferScalarType(getOperand(0)), VF);
    return Ctx.TTI.getVectorInstrCost(Instruction::ExtractElement, VecTy,
                                      Ctx.CostKind);
  }
  case VPInstruction::AnyOf: {
    auto *VecTy = toVectorTy(Ctx.Types.inferScalarType(this), VF);
    return Ctx.TTI.getArithmeticReductionCost(
        Instruction::Or, cast<VectorType>(VecTy), std::nullopt, Ctx.CostKind);
  }
  case VPInstruction::FirstActiveLane: {
    // Calculate the cost of determining the lane index.
    auto *PredTy = toVectorTy(Ctx.Types.inferScalarType(getOperand(0)), VF);
    IntrinsicCostAttributes Attrs(Intrinsic::experimental_cttz_elts,
                                  Type::getInt64Ty(Ctx.LLVMCtx),
                                  {PredTy, Type::getInt1Ty(Ctx.LLVMCtx)});
    return Ctx.TTI.getIntrinsicInstrCost(Attrs, Ctx.CostKind);
  }
  case VPInstruction::FirstOrderRecurrenceSplice: {
    assert(VF.isVector() && "Scalar FirstOrderRecurrenceSplice?");
    SmallVector<int> Mask(VF.getKnownMinValue());
    std::iota(Mask.begin(), Mask.end(), VF.getKnownMinValue() - 1);
    Type *VectorTy = toVectorTy(Ctx.Types.inferScalarType(this), VF);

    return Ctx.TTI.getShuffleCost(TargetTransformInfo::SK_Splice,
                                  cast<VectorType>(VectorTy),
                                  cast<VectorType>(VectorTy), Mask,
                                  Ctx.CostKind, VF.getKnownMinValue() - 1);
  }
  case VPInstruction::ActiveLaneMask: {
    Type *ArgTy = Ctx.Types.inferScalarType(getOperand(0));
    Type *RetTy = toVectorTy(Type::getInt1Ty(Ctx.LLVMCtx), VF);
    IntrinsicCostAttributes Attrs(Intrinsic::get_active_lane_mask, RetTy,
                                  {ArgTy, ArgTy});
    return Ctx.TTI.getIntrinsicInstrCost(Attrs, Ctx.CostKind);
  }
  case VPInstruction::ExplicitVectorLength: {
    Type *Arg0Ty = Ctx.Types.inferScalarType(getOperand(0));
    Type *I32Ty = Type::getInt32Ty(Ctx.LLVMCtx);
    Type *I1Ty = Type::getInt1Ty(Ctx.LLVMCtx);
    IntrinsicCostAttributes Attrs(Intrinsic::experimental_get_vector_length,
                                  I32Ty, {Arg0Ty, I32Ty, I1Ty});
    return Ctx.TTI.getIntrinsicInstrCost(Attrs, Ctx.CostKind);
  }
  case VPInstruction::ExtractPenultimateElement:
    if (VF == ElementCount::getScalable(1))
      return InstructionCost::getInvalid();
  LLVM_FALLTHROUGH;
  default:
    // TODO: Compute cost other VPInstructions once the legacy cost model has
    // been retired.
    assert(!getUnderlyingValue() &&
           "unexpected VPInstruction witht underlying value");
    return 0;
  }
}

bool VPInstruction::isVectorToScalar() const {
  return getOpcode() == VPInstruction::ExtractLastElement ||
         getOpcode() == VPInstruction::ExtractPenultimateElement ||
         getOpcode() == Instruction::ExtractElement ||
         getOpcode() == VPInstruction::FirstActiveLane ||
         getOpcode() == VPInstruction::ComputeAnyOfResult ||
         getOpcode() == VPInstruction::ComputeFindIVResult ||
         getOpcode() == VPInstruction::ComputeReductionResult ||
         getOpcode() == VPInstruction::AnyOf;
}

bool VPInstruction::isSingleScalar() const {
  return getOpcode() == Instruction::PHI || isScalarCast();
}

void VPInstruction::execute(VPTransformState &State) {
  assert(!State.Lane && "VPInstruction executing an Lane");
  IRBuilderBase::FastMathFlagGuard FMFGuard(State.Builder);
  assert(flagsValidForOpcode(getOpcode()) &&
         "Set flags not supported for the provided opcode");
  if (hasFastMathFlags())
    State.Builder.setFastMathFlags(getFastMathFlags());
  bool GeneratesPerFirstLaneOnly = canGenerateScalarForFirstLane() &&
                                   (vputils::onlyFirstLaneUsed(this) ||
                                    isVectorToScalar() || isSingleScalar());
  bool GeneratesPerAllLanes = doesGeneratePerAllLanes();
  if (GeneratesPerAllLanes) {
    for (unsigned Lane = 0, NumLanes = State.VF.getFixedValue();
         Lane != NumLanes; ++Lane) {
      Value *GeneratedValue = generatePerLane(State, VPLane(Lane));
      assert(GeneratedValue && "generatePerLane must produce a value");
      State.set(this, GeneratedValue, VPLane(Lane));
    }
    return;
  }

  Value *GeneratedValue = generate(State);
  if (!hasResult())
    return;
  assert(GeneratedValue && "generate must produce a value");
  assert((((GeneratedValue->getType()->isVectorTy() ||
            GeneratedValue->getType()->isStructTy()) ==
           !GeneratesPerFirstLaneOnly) ||
          State.VF.isScalar()) &&
         "scalar value but not only first lane defined");
  State.set(this, GeneratedValue,
            /*IsScalar*/ GeneratesPerFirstLaneOnly);
}

bool VPInstruction::opcodeMayReadOrWriteFromMemory() const {
  if (Instruction::isBinaryOp(getOpcode()) || Instruction::isCast(getOpcode()))
    return false;
  switch (getOpcode()) {
  case Instruction::ExtractElement:
  case Instruction::Freeze:
  case Instruction::FCmp:
  case Instruction::ICmp:
  case Instruction::Select:
  case VPInstruction::AnyOf:
  case VPInstruction::BuildStructVector:
  case VPInstruction::BuildVector:
  case VPInstruction::CalculateTripCountMinusVF:
  case VPInstruction::CanonicalIVIncrementForPart:
  case VPInstruction::ExtractLastElement:
  case VPInstruction::ExtractPenultimateElement:
  case VPInstruction::FirstActiveLane:
  case VPInstruction::FirstOrderRecurrenceSplice:
  case VPInstruction::LogicalAnd:
  case VPInstruction::Not:
  case VPInstruction::PtrAdd:
  case VPInstruction::WideIVStep:
  case VPInstruction::StepVector:
  case VPInstruction::ReductionStartVector:
    return false;
  default:
    return true;
  }
}

bool VPInstruction::onlyFirstLaneUsed(const VPValue *Op) const {
  assert(is_contained(operands(), Op) && "Op must be an operand of the recipe");
  if (Instruction::isBinaryOp(getOpcode()) || Instruction::isCast(getOpcode()))
    return vputils::onlyFirstLaneUsed(this);

  switch (getOpcode()) {
  default:
    return false;
  case Instruction::ExtractElement:
    return Op == getOperand(1);
  case Instruction::PHI:
    return true;
  case Instruction::FCmp:
  case Instruction::ICmp:
  case Instruction::Select:
  case Instruction::Or:
  case Instruction::Freeze:
    // TODO: Cover additional opcodes.
    return vputils::onlyFirstLaneUsed(this);
  case VPInstruction::ActiveLaneMask:
  case VPInstruction::ExplicitVectorLength:
  case VPInstruction::CalculateTripCountMinusVF:
  case VPInstruction::CanonicalIVIncrementForPart:
  case VPInstruction::BranchOnCount:
  case VPInstruction::BranchOnCond:
  case VPInstruction::Broadcast:
  case VPInstruction::ReductionStartVector:
    return true;
  case VPInstruction::PtrAdd:
    return Op == getOperand(0) || vputils::onlyFirstLaneUsed(this);
  case VPInstruction::ComputeAnyOfResult:
  case VPInstruction::ComputeFindIVResult:
    return Op == getOperand(1);
  };
  llvm_unreachable("switch should return");
}

bool VPInstruction::onlyFirstPartUsed(const VPValue *Op) const {
  assert(is_contained(operands(), Op) && "Op must be an operand of the recipe");
  if (Instruction::isBinaryOp(getOpcode()))
    return vputils::onlyFirstPartUsed(this);

  switch (getOpcode()) {
  default:
    return false;
  case Instruction::FCmp:
  case Instruction::ICmp:
  case Instruction::Select:
    return vputils::onlyFirstPartUsed(this);
  case VPInstruction::BranchOnCount:
  case VPInstruction::BranchOnCond:
  case VPInstruction::CanonicalIVIncrementForPart:
    return true;
  };
  llvm_unreachable("switch should return");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPInstruction::dump() const {
  VPSlotTracker SlotTracker(getParent()->getPlan());
  print(dbgs(), "", SlotTracker);
}

void VPInstruction::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT" << (isSingleScalar() ? "-SCALAR" : "") << " ";

  if (hasResult()) {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  switch (getOpcode()) {
  case VPInstruction::Not:
    O << "not";
    break;
  case VPInstruction::SLPLoad:
    O << "combined load";
    break;
  case VPInstruction::SLPStore:
    O << "combined store";
    break;
  case VPInstruction::ActiveLaneMask:
    O << "active lane mask";
    break;
  case VPInstruction::ExplicitVectorLength:
    O << "EXPLICIT-VECTOR-LENGTH";
    break;
  case VPInstruction::FirstOrderRecurrenceSplice:
    O << "first-order splice";
    break;
  case VPInstruction::BranchOnCond:
    O << "branch-on-cond";
    break;
  case VPInstruction::CalculateTripCountMinusVF:
    O << "TC > VF ? TC - VF : 0";
    break;
  case VPInstruction::CanonicalIVIncrementForPart:
    O << "VF * Part +";
    break;
  case VPInstruction::BranchOnCount:
    O << "branch-on-count";
    break;
  case VPInstruction::Broadcast:
    O << "broadcast";
    break;
  case VPInstruction::BuildStructVector:
    O << "buildstructvector";
    break;
  case VPInstruction::BuildVector:
    O << "buildvector";
    break;
  case VPInstruction::ExtractLastElement:
    O << "extract-last-element";
    break;
  case VPInstruction::ExtractPenultimateElement:
    O << "extract-penultimate-element";
    break;
  case VPInstruction::ComputeAnyOfResult:
    O << "compute-anyof-result";
    break;
  case VPInstruction::ComputeFindIVResult:
    O << "compute-find-iv-result";
    break;
  case VPInstruction::ComputeReductionResult:
    O << "compute-reduction-result";
    break;
  case VPInstruction::LogicalAnd:
    O << "logical-and";
    break;
  case VPInstruction::PtrAdd:
    O << "ptradd";
    break;
  case VPInstruction::AnyOf:
    O << "any-of";
    break;
  case VPInstruction::FirstActiveLane:
    O << "first-active-lane";
    break;
  case VPInstruction::ReductionStartVector:
    O << "reduction-start-vector";
    break;
  default:
    O << Instruction::getOpcodeName(getOpcode());
  }

  printFlags(O);
  printOperands(O, SlotTracker);

  if (auto DL = getDebugLoc()) {
    O << ", !dbg ";
    DL.print(O);
  }
}
#endif

void VPInstructionWithType::execute(VPTransformState &State) {
  State.setDebugLocFrom(getDebugLoc());
  if (isScalarCast()) {
    Value *Op = State.get(getOperand(0), VPLane(0));
    Value *Cast = State.Builder.CreateCast(Instruction::CastOps(getOpcode()),
                                           Op, ResultTy);
    State.set(this, Cast, VPLane(0));
    return;
  }
  switch (getOpcode()) {
  case VPInstruction::StepVector: {
    Value *StepVector =
        State.Builder.CreateStepVector(VectorType::get(ResultTy, State.VF));
    State.set(this, StepVector);
    break;
  }
  default:
    llvm_unreachable("opcode not implemented yet");
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPInstructionWithType::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT" << (isSingleScalar() ? "-SCALAR" : "") << " ";
  printAsOperand(O, SlotTracker);
  O << " = ";

  switch (getOpcode()) {
  case VPInstruction::WideIVStep:
    O << "wide-iv-step ";
    printOperands(O, SlotTracker);
    break;
  case VPInstruction::StepVector:
    O << "step-vector " << *ResultTy;
    break;
  default:
    assert(Instruction::isCast(getOpcode()) && "unhandled opcode");
    O << Instruction::getOpcodeName(getOpcode()) << " ";
    printOperands(O, SlotTracker);
    O << " to " << *ResultTy;
  }
}
#endif

void VPPhi::execute(VPTransformState &State) {
  State.setDebugLocFrom(getDebugLoc());
  PHINode *NewPhi = State.Builder.CreatePHI(
      State.TypeAnalysis.inferScalarType(this), 2, getName());
  unsigned NumIncoming = getNumIncoming();
  if (getParent() != getParent()->getPlan()->getScalarPreheader()) {
    // TODO: Fixup all incoming values of header phis once recipes defining them
    // are introduced.
    NumIncoming = 1;
  }
  for (unsigned Idx = 0; Idx != NumIncoming; ++Idx) {
    Value *IncV = State.get(getIncomingValue(Idx), VPLane(0));
    BasicBlock *PredBB = State.CFG.VPBB2IRBB.at(getIncomingBlock(Idx));
    NewPhi->addIncoming(IncV, PredBB);
  }
  State.set(this, NewPhi, VPLane(0));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPPhi::print(raw_ostream &O, const Twine &Indent,
                  VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT" << (isSingleScalar() ? "-SCALAR" : "") << " ";
  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printPhiOperands(O, SlotTracker);
}
#endif

VPIRInstruction *VPIRInstruction ::create(Instruction &I) {
  if (auto *Phi = dyn_cast<PHINode>(&I))
    return new VPIRPhi(*Phi);
  return new VPIRInstruction(I);
}

void VPIRInstruction::execute(VPTransformState &State) {
  assert(!isa<VPIRPhi>(this) && getNumOperands() == 0 &&
         "PHINodes must be handled by VPIRPhi");
  // Advance the insert point after the wrapped IR instruction. This allows
  // interleaving VPIRInstructions and other recipes.
  State.Builder.SetInsertPoint(I.getParent(), std::next(I.getIterator()));
}

InstructionCost VPIRInstruction::computeCost(ElementCount VF,
                                             VPCostContext &Ctx) const {
  // The recipe wraps an existing IR instruction on the border of VPlan's scope,
  // hence it does not contribute to the cost-modeling for the VPlan.
  return 0;
}

void VPIRInstruction::extractLastLaneOfFirstOperand(VPBuilder &Builder) {
  assert(isa<PHINode>(getInstruction()) &&
         "can only update exiting operands to phi nodes");
  assert(getNumOperands() > 0 && "must have at least one operand");
  VPValue *Exiting = getOperand(0);
  if (Exiting->isLiveIn())
    return;

  Exiting = Builder.createNaryOp(VPInstruction::ExtractLastElement, {Exiting});
  setOperand(0, Exiting);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPIRInstruction::print(raw_ostream &O, const Twine &Indent,
                            VPSlotTracker &SlotTracker) const {
  O << Indent << "IR " << I;
}
#endif

void VPIRPhi::execute(VPTransformState &State) {
  PHINode *Phi = &getIRPhi();
  for (const auto &[Idx, Op] : enumerate(operands())) {
    VPValue *ExitValue = Op;
    auto Lane = vputils::isSingleScalar(ExitValue)
                    ? VPLane::getFirstLane()
                    : VPLane::getLastLaneForVF(State.VF);
    VPBlockBase *Pred = getParent()->getPredecessors()[Idx];
    auto *PredVPBB = Pred->getExitingBasicBlock();
    BasicBlock *PredBB = State.CFG.VPBB2IRBB[PredVPBB];
    // Set insertion point in PredBB in case an extract needs to be generated.
    // TODO: Model extracts explicitly.
    State.Builder.SetInsertPoint(PredBB, PredBB->getFirstNonPHIIt());
    Value *V = State.get(ExitValue, VPLane(Lane));
    // If there is no existing block for PredBB in the phi, add a new incoming
    // value. Otherwise update the existing incoming value for PredBB.
    if (Phi->getBasicBlockIndex(PredBB) == -1)
      Phi->addIncoming(V, PredBB);
    else
      Phi->setIncomingValueForBlock(PredBB, V);
  }

  // Advance the insert point after the wrapped IR instruction. This allows
  // interleaving VPIRInstructions and other recipes.
  State.Builder.SetInsertPoint(Phi->getParent(), std::next(Phi->getIterator()));
}

void VPPhiAccessors::removeIncomingValueFor(VPBlockBase *IncomingBlock) const {
  VPRecipeBase *R = const_cast<VPRecipeBase *>(getAsRecipe());
  assert(R->getNumOperands() == R->getParent()->getNumPredecessors() &&
         "Number of phi operands must match number of predecessors");
  unsigned Position = R->getParent()->getIndexForPredecessor(IncomingBlock);
  R->removeOperand(Position);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPPhiAccessors::printPhiOperands(raw_ostream &O,
                                      VPSlotTracker &SlotTracker) const {
  interleaveComma(enumerate(getAsRecipe()->operands()), O,
                  [this, &O, &SlotTracker](auto Op) {
                    O << "[ ";
                    Op.value()->printAsOperand(O, SlotTracker);
                    O << ", ";
                    getIncomingBlock(Op.index())->printAsOperand(O);
                    O << " ]";
                  });
}
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPIRPhi::print(raw_ostream &O, const Twine &Indent,
                    VPSlotTracker &SlotTracker) const {
  VPIRInstruction::print(O, Indent, SlotTracker);

  if (getNumOperands() != 0) {
    O << " (extra operand" << (getNumOperands() > 1 ? "s" : "") << ": ";
    interleaveComma(
        enumerate(operands()), O, [this, &O, &SlotTracker](auto Op) {
          Op.value()->printAsOperand(O, SlotTracker);
          O << " from ";
          getParent()->getPredecessors()[Op.index()]->printAsOperand(O);
        });
    O << ")";
  }
}
#endif

VPIRMetadata::VPIRMetadata(Instruction &I, LoopVersioning *LVer)
    : VPIRMetadata(I) {
  if (!LVer || !isa<LoadInst, StoreInst>(&I))
    return;
  const auto &[AliasScopeMD, NoAliasMD] = LVer->getNoAliasMetadataFor(&I);
  if (AliasScopeMD)
    Metadata.emplace_back(LLVMContext::MD_alias_scope, AliasScopeMD);
  if (NoAliasMD)
    Metadata.emplace_back(LLVMContext::MD_noalias, NoAliasMD);
}

void VPIRMetadata::applyMetadata(Instruction &I) const {
  for (const auto &[Kind, Node] : Metadata)
    I.setMetadata(Kind, Node);
}

void VPWidenCallRecipe::execute(VPTransformState &State) {
  assert(State.VF.isVector() && "not widening");
  assert(Variant != nullptr && "Can't create vector function.");

  FunctionType *VFTy = Variant->getFunctionType();
  // Add return type if intrinsic is overloaded on it.
  SmallVector<Value *, 4> Args;
  for (const auto &I : enumerate(args())) {
    Value *Arg;
    // Some vectorized function variants may also take a scalar argument,
    // e.g. linear parameters for pointers. This needs to be the scalar value
    // from the start of the respective part when interleaving.
    if (!VFTy->getParamType(I.index())->isVectorTy())
      Arg = State.get(I.value(), VPLane(0));
    else
      Arg = State.get(I.value(), onlyFirstLaneUsed(I.value()));
    Args.push_back(Arg);
  }

  auto *CI = cast_or_null<CallInst>(getUnderlyingValue());
  SmallVector<OperandBundleDef, 1> OpBundles;
  if (CI)
    CI->getOperandBundlesAsDefs(OpBundles);

  CallInst *V = State.Builder.CreateCall(Variant, Args, OpBundles);
  applyFlags(*V);
  applyMetadata(*V);
  V->setCallingConv(Variant->getCallingConv());

  if (!V->getType()->isVoidTy())
    State.set(this, V);
}

InstructionCost VPWidenCallRecipe::computeCost(ElementCount VF,
                                               VPCostContext &Ctx) const {
  return Ctx.TTI.getCallInstrCost(nullptr, Variant->getReturnType(),
                                  Variant->getFunctionType()->params(),
                                  Ctx.CostKind);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenCallRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-CALL ";

  Function *CalledFn = getCalledScalarFunction();
  if (CalledFn->getReturnType()->isVoidTy())
    O << "void ";
  else {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  O << "call";
  printFlags(O);
  O << " @" << CalledFn->getName() << "(";
  interleaveComma(args(), O, [&O, &SlotTracker](VPValue *Op) {
    Op->printAsOperand(O, SlotTracker);
  });
  O << ")";

  O << " (using library function";
  if (Variant->hasName())
    O << ": " << Variant->getName();
  O << ")";
}
#endif

void VPWidenIntrinsicRecipe::execute(VPTransformState &State) {
  assert(State.VF.isVector() && "not widening");

  SmallVector<Type *, 2> TysForDecl;
  // Add return type if intrinsic is overloaded on it.
  if (isVectorIntrinsicWithOverloadTypeAtArg(VectorIntrinsicID, -1, State.TTI))
    TysForDecl.push_back(VectorType::get(getResultType(), State.VF));
  SmallVector<Value *, 4> Args;
  for (const auto &I : enumerate(operands())) {
    // Some intrinsics have a scalar argument - don't replace it with a
    // vector.
    Value *Arg;
    if (isVectorIntrinsicWithScalarOpAtArg(VectorIntrinsicID, I.index(),
                                           State.TTI))
      Arg = State.get(I.value(), VPLane(0));
    else
      Arg = State.get(I.value(), onlyFirstLaneUsed(I.value()));
    if (isVectorIntrinsicWithOverloadTypeAtArg(VectorIntrinsicID, I.index(),
                                               State.TTI))
      TysForDecl.push_back(Arg->getType());
    Args.push_back(Arg);
  }

  // Use vector version of the intrinsic.
  Module *M = State.Builder.GetInsertBlock()->getModule();
  Function *VectorF =
      Intrinsic::getOrInsertDeclaration(M, VectorIntrinsicID, TysForDecl);
  assert(VectorF &&
         "Can't retrieve vector intrinsic or vector-predication intrinsics.");

  auto *CI = cast_or_null<CallInst>(getUnderlyingValue());
  SmallVector<OperandBundleDef, 1> OpBundles;
  if (CI)
    CI->getOperandBundlesAsDefs(OpBundles);

  CallInst *V = State.Builder.CreateCall(VectorF, Args, OpBundles);

  applyFlags(*V);
  applyMetadata(*V);

  if (!V->getType()->isVoidTy())
    State.set(this, V);
}

InstructionCost VPWidenIntrinsicRecipe::computeCost(ElementCount VF,
                                                    VPCostContext &Ctx) const {
  // Some backends analyze intrinsic arguments to determine cost. Use the
  // underlying value for the operand if it has one. Otherwise try to use the
  // operand of the underlying call instruction, if there is one. Otherwise
  // clear Arguments.
  // TODO: Rework TTI interface to be independent of concrete IR values.
  SmallVector<const Value *> Arguments;
  for (const auto &[Idx, Op] : enumerate(operands())) {
    auto *V = Op->getUnderlyingValue();
    if (!V) {
      if (auto *UI = dyn_cast_or_null<CallBase>(getUnderlyingValue())) {
        Arguments.push_back(UI->getArgOperand(Idx));
        continue;
      }
      Arguments.clear();
      break;
    }
    Arguments.push_back(V);
  }

  Type *RetTy = toVectorizedTy(Ctx.Types.inferScalarType(this), VF);
  SmallVector<Type *> ParamTys;
  for (unsigned I = 0; I != getNumOperands(); ++I)
    ParamTys.push_back(
        toVectorTy(Ctx.Types.inferScalarType(getOperand(I)), VF));

  // TODO: Rework TTI interface to avoid reliance on underlying IntrinsicInst.
  FastMathFlags FMF = hasFastMathFlags() ? getFastMathFlags() : FastMathFlags();
  IntrinsicCostAttributes CostAttrs(
      VectorIntrinsicID, RetTy, Arguments, ParamTys, FMF,
      dyn_cast_or_null<IntrinsicInst>(getUnderlyingValue()),
      InstructionCost::getInvalid(), &Ctx.TLI);
  return Ctx.TTI.getIntrinsicInstrCost(CostAttrs, Ctx.CostKind);
}

StringRef VPWidenIntrinsicRecipe::getIntrinsicName() const {
  return Intrinsic::getBaseName(VectorIntrinsicID);
}

bool VPWidenIntrinsicRecipe::onlyFirstLaneUsed(const VPValue *Op) const {
  assert(is_contained(operands(), Op) && "Op must be an operand of the recipe");
  return all_of(enumerate(operands()), [this, &Op](const auto &X) {
    auto [Idx, V] = X;
    return V != Op || isVectorIntrinsicWithScalarOpAtArg(getVectorIntrinsicID(),
                                                         Idx, nullptr);
  });
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenIntrinsicRecipe::print(raw_ostream &O, const Twine &Indent,
                                   VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-INTRINSIC ";
  if (ResultTy->isVoidTy()) {
    O << "void ";
  } else {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  O << "call";
  printFlags(O);
  O << getIntrinsicName() << "(";

  interleaveComma(operands(), O, [&O, &SlotTracker](VPValue *Op) {
    Op->printAsOperand(O, SlotTracker);
  });
  O << ")";
}
#endif

void VPHistogramRecipe::execute(VPTransformState &State) {
  IRBuilderBase &Builder = State.Builder;

  Value *Address = State.get(getOperand(0));
  Value *IncAmt = State.get(getOperand(1), /*IsScalar=*/true);
  VectorType *VTy = cast<VectorType>(Address->getType());

  // The histogram intrinsic requires a mask even if the recipe doesn't;
  // if the mask operand was omitted then all lanes should be executed and
  // we just need to synthesize an all-true mask.
  Value *Mask = nullptr;
  if (VPValue *VPMask = getMask())
    Mask = State.get(VPMask);
  else
    Mask =
        Builder.CreateVectorSplat(VTy->getElementCount(), Builder.getInt1(1));

  // If this is a subtract, we want to invert the increment amount. We may
  // add a separate intrinsic in future, but for now we'll try this.
  if (Opcode == Instruction::Sub)
    IncAmt = Builder.CreateNeg(IncAmt);
  else
    assert(Opcode == Instruction::Add && "only add or sub supported for now");

  State.Builder.CreateIntrinsic(Intrinsic::experimental_vector_histogram_add,
                                {VTy, IncAmt->getType()},
                                {Address, IncAmt, Mask});
}

InstructionCost VPHistogramRecipe::computeCost(ElementCount VF,
                                               VPCostContext &Ctx) const {
  // FIXME: Take the gather and scatter into account as well. For now we're
  //        generating the same cost as the fallback path, but we'll likely
  //        need to create a new TTI method for determining the cost, including
  //        whether we can use base + vec-of-smaller-indices or just
  //        vec-of-pointers.
  assert(VF.isVector() && "Invalid VF for histogram cost");
  Type *AddressTy = Ctx.Types.inferScalarType(getOperand(0));
  VPValue *IncAmt = getOperand(1);
  Type *IncTy = Ctx.Types.inferScalarType(IncAmt);
  VectorType *VTy = VectorType::get(IncTy, VF);

  // Assume that a non-constant update value (or a constant != 1) requires
  // a multiply, and add that into the cost.
  InstructionCost MulCost =
      Ctx.TTI.getArithmeticInstrCost(Instruction::Mul, VTy, Ctx.CostKind);
  if (IncAmt->isLiveIn()) {
    ConstantInt *CI = dyn_cast<ConstantInt>(IncAmt->getLiveInIRValue());

    if (CI && CI->getZExtValue() == 1)
      MulCost = TTI::TCC_Free;
  }

  // Find the cost of the histogram operation itself.
  Type *PtrTy = VectorType::get(AddressTy, VF);
  Type *MaskTy = VectorType::get(Type::getInt1Ty(Ctx.LLVMCtx), VF);
  IntrinsicCostAttributes ICA(Intrinsic::experimental_vector_histogram_add,
                              Type::getVoidTy(Ctx.LLVMCtx),
                              {PtrTy, IncTy, MaskTy});

  // Add the costs together with the add/sub operation.
  return Ctx.TTI.getIntrinsicInstrCost(ICA, Ctx.CostKind) + MulCost +
         Ctx.TTI.getArithmeticInstrCost(Opcode, VTy, Ctx.CostKind);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPHistogramRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-HISTOGRAM buckets: ";
  getOperand(0)->printAsOperand(O, SlotTracker);

  if (Opcode == Instruction::Sub)
    O << ", dec: ";
  else {
    assert(Opcode == Instruction::Add);
    O << ", inc: ";
  }
  getOperand(1)->printAsOperand(O, SlotTracker);

  if (VPValue *Mask = getMask()) {
    O << ", mask: ";
    Mask->printAsOperand(O, SlotTracker);
  }
}

void VPWidenSelectRecipe::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-SELECT ";
  printAsOperand(O, SlotTracker);
  O << " = select ";
  printFlags(O);
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(1)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(2)->printAsOperand(O, SlotTracker);
  O << (isInvariantCond() ? " (condition is loop invariant)" : "");
}
#endif

void VPWidenSelectRecipe::execute(VPTransformState &State) {
  // The condition can be loop invariant but still defined inside the
  // loop. This means that we can't just use the original 'cond' value.
  // We have to take the 'vectorized' value and pick the first lane.
  // Instcombine will make this a no-op.
  auto *InvarCond =
      isInvariantCond() ? State.get(getCond(), VPLane(0)) : nullptr;

  Value *Cond = InvarCond ? InvarCond : State.get(getCond());
  Value *Op0 = State.get(getOperand(1));
  Value *Op1 = State.get(getOperand(2));
  Value *Sel = State.Builder.CreateSelect(Cond, Op0, Op1);
  State.set(this, Sel);
  if (auto *I = dyn_cast<Instruction>(Sel)) {
    if (isa<FPMathOperator>(I))
      applyFlags(*I);
    applyMetadata(*I);
  }
}

InstructionCost VPWidenSelectRecipe::computeCost(ElementCount VF,
                                                 VPCostContext &Ctx) const {
  SelectInst *SI = cast<SelectInst>(getUnderlyingValue());
  bool ScalarCond = getOperand(0)->isDefinedOutsideLoopRegions();
  Type *ScalarTy = Ctx.Types.inferScalarType(this);
  Type *VectorTy = toVectorTy(Ctx.Types.inferScalarType(this), VF);

  VPValue *Op0, *Op1;
  using namespace llvm::VPlanPatternMatch;
  if (!ScalarCond && ScalarTy->getScalarSizeInBits() == 1 &&
      (match(this, m_LogicalAnd(m_VPValue(Op0), m_VPValue(Op1))) ||
       match(this, m_LogicalOr(m_VPValue(Op0), m_VPValue(Op1))))) {
    // select x, y, false --> x & y
    // select x, true, y --> x | y
    const auto [Op1VK, Op1VP] = Ctx.getOperandInfo(Op0);
    const auto [Op2VK, Op2VP] = Ctx.getOperandInfo(Op1);

    SmallVector<const Value *, 2> Operands;
    if (all_of(operands(),
               [](VPValue *Op) { return Op->getUnderlyingValue(); }))
      Operands.append(SI->op_begin(), SI->op_end());
    bool IsLogicalOr = match(this, m_LogicalOr(m_VPValue(Op0), m_VPValue(Op1)));
    return Ctx.TTI.getArithmeticInstrCost(
        IsLogicalOr ? Instruction::Or : Instruction::And, VectorTy,
        Ctx.CostKind, {Op1VK, Op1VP}, {Op2VK, Op2VP}, Operands, SI);
  }

  Type *CondTy = Ctx.Types.inferScalarType(getOperand(0));
  if (!ScalarCond)
    CondTy = VectorType::get(CondTy, VF);

  CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
  if (auto *Cmp = dyn_cast<CmpInst>(SI->getCondition()))
    Pred = Cmp->getPredicate();
  return Ctx.TTI.getCmpSelInstrCost(
      Instruction::Select, VectorTy, CondTy, Pred, Ctx.CostKind,
      {TTI::OK_AnyValue, TTI::OP_None}, {TTI::OK_AnyValue, TTI::OP_None}, SI);
}

VPIRFlags::FastMathFlagsTy::FastMathFlagsTy(const FastMathFlags &FMF) {
  AllowReassoc = FMF.allowReassoc();
  NoNaNs = FMF.noNaNs();
  NoInfs = FMF.noInfs();
  NoSignedZeros = FMF.noSignedZeros();
  AllowReciprocal = FMF.allowReciprocal();
  AllowContract = FMF.allowContract();
  ApproxFunc = FMF.approxFunc();
}

#if !defined(NDEBUG)
bool VPIRFlags::flagsValidForOpcode(unsigned Opcode) const {
  switch (OpType) {
  case OperationType::OverflowingBinOp:
    return Opcode == Instruction::Add || Opcode == Instruction::Sub ||
           Opcode == Instruction::Mul ||
           Opcode == VPInstruction::VPInstruction::CanonicalIVIncrementForPart;
  case OperationType::Trunc:
    return Opcode == Instruction::Trunc;
  case OperationType::DisjointOp:
    return Opcode == Instruction::Or;
  case OperationType::PossiblyExactOp:
    return Opcode == Instruction::AShr;
  case OperationType::GEPOp:
    return Opcode == Instruction::GetElementPtr ||
           Opcode == VPInstruction::PtrAdd;
  case OperationType::FPMathOp:
    return Opcode == Instruction::FAdd || Opcode == Instruction::FMul ||
           Opcode == Instruction::FSub || Opcode == Instruction::FNeg ||
           Opcode == Instruction::FDiv || Opcode == Instruction::FRem ||
           Opcode == Instruction::FCmp || Opcode == Instruction::Select ||
           Opcode == VPInstruction::WideIVStep ||
           Opcode == VPInstruction::ReductionStartVector ||
           Opcode == VPInstruction::ComputeReductionResult;
  case OperationType::NonNegOp:
    return Opcode == Instruction::ZExt;
    break;
  case OperationType::Cmp:
    return Opcode == Instruction::FCmp || Opcode == Instruction::ICmp;
  case OperationType::Other:
    return true;
  }
  llvm_unreachable("Unknown OperationType enum");
}
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPIRFlags::printFlags(raw_ostream &O) const {
  switch (OpType) {
  case OperationType::Cmp:
    O << " " << CmpInst::getPredicateName(getPredicate());
    break;
  case OperationType::DisjointOp:
    if (DisjointFlags.IsDisjoint)
      O << " disjoint";
    break;
  case OperationType::PossiblyExactOp:
    if (ExactFlags.IsExact)
      O << " exact";
    break;
  case OperationType::OverflowingBinOp:
    if (WrapFlags.HasNUW)
      O << " nuw";
    if (WrapFlags.HasNSW)
      O << " nsw";
    break;
  case OperationType::Trunc:
    if (TruncFlags.HasNUW)
      O << " nuw";
    if (TruncFlags.HasNSW)
      O << " nsw";
    break;
  case OperationType::FPMathOp:
    getFastMathFlags().print(O);
    break;
  case OperationType::GEPOp:
    if (GEPFlags.isInBounds())
      O << " inbounds";
    else if (GEPFlags.hasNoUnsignedSignedWrap())
      O << " nusw";
    if (GEPFlags.hasNoUnsignedWrap())
      O << " nuw";
    break;
  case OperationType::NonNegOp:
    if (NonNegFlags.NonNeg)
      O << " nneg";
    break;
  case OperationType::Other:
    break;
  }
  O << " ";
}
#endif

void VPWidenRecipe::execute(VPTransformState &State) {
  auto &Builder = State.Builder;
  switch (Opcode) {
  case Instruction::Call:
  case Instruction::Br:
  case Instruction::PHI:
  case Instruction::GetElementPtr:
  case Instruction::Select:
    llvm_unreachable("This instruction is handled by a different recipe.");
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::FNeg:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    // Just widen unops and binops.
    SmallVector<Value *, 2> Ops;
    for (VPValue *VPOp : operands())
      Ops.push_back(State.get(VPOp));

    Value *V = Builder.CreateNAryOp(Opcode, Ops);

    if (auto *VecOp = dyn_cast<Instruction>(V)) {
      applyFlags(*VecOp);
      applyMetadata(*VecOp);
    }

    // Use this vector value for all users of the original instruction.
    State.set(this, V);
    break;
  }
  case Instruction::ExtractValue: {
    assert(getNumOperands() == 2 && "expected single level extractvalue");
    Value *Op = State.get(getOperand(0));
    auto *CI = cast<ConstantInt>(getOperand(1)->getLiveInIRValue());
    Value *Extract = Builder.CreateExtractValue(Op, CI->getZExtValue());
    State.set(this, Extract);
    break;
  }
  case Instruction::Freeze: {
    Value *Op = State.get(getOperand(0));
    Value *Freeze = Builder.CreateFreeze(Op);
    State.set(this, Freeze);
    break;
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    // Widen compares. Generate vector compares.
    bool FCmp = Opcode == Instruction::FCmp;
    Value *A = State.get(getOperand(0));
    Value *B = State.get(getOperand(1));
    Value *C = nullptr;
    if (FCmp) {
      // Propagate fast math flags.
      C = Builder.CreateFCmpFMF(
          getPredicate(), A, B,
          dyn_cast_or_null<Instruction>(getUnderlyingValue()));
    } else {
      C = Builder.CreateICmp(getPredicate(), A, B);
    }
    if (auto *I = dyn_cast<Instruction>(C))
      applyMetadata(*I);
    State.set(this, C);
    break;
  }
  default:
    // This instruction is not vectorized by simple widening.
    LLVM_DEBUG(dbgs() << "LV: Found an unhandled opcode : "
                      << Instruction::getOpcodeName(Opcode));
    llvm_unreachable("Unhandled instruction!");
  } // end of switch.

#if !defined(NDEBUG)
  // Verify that VPlan type inference results agree with the type of the
  // generated values.
  assert(VectorType::get(State.TypeAnalysis.inferScalarType(this), State.VF) ==
             State.get(this)->getType() &&
         "inferred type and type from generated instructions do not match");
#endif
}

InstructionCost VPWidenRecipe::computeCost(ElementCount VF,
                                           VPCostContext &Ctx) const {
  switch (Opcode) {
  case Instruction::FNeg: {
    Type *VectorTy = toVectorTy(Ctx.Types.inferScalarType(this), VF);
    return Ctx.TTI.getArithmeticInstrCost(
        Opcode, VectorTy, Ctx.CostKind,
        {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
        {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None});
  }

  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
    // More complex computation, let the legacy cost-model handle this for now.
    return Ctx.getLegacyCost(cast<Instruction>(getUnderlyingValue()), VF);
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    VPValue *RHS = getOperand(1);
    // Certain instructions can be cheaper to vectorize if they have a constant
    // second vector operand. One example of this are shifts on x86.
    TargetTransformInfo::OperandValueInfo RHSInfo = Ctx.getOperandInfo(RHS);

    if (RHSInfo.Kind == TargetTransformInfo::OK_AnyValue &&
        getOperand(1)->isDefinedOutsideLoopRegions())
      RHSInfo.Kind = TargetTransformInfo::OK_UniformValue;
    Type *VectorTy = toVectorTy(Ctx.Types.inferScalarType(this), VF);
    Instruction *CtxI = dyn_cast_or_null<Instruction>(getUnderlyingValue());

    SmallVector<const Value *, 4> Operands;
    if (CtxI)
      Operands.append(CtxI->value_op_begin(), CtxI->value_op_end());
    return Ctx.TTI.getArithmeticInstrCost(
        Opcode, VectorTy, Ctx.CostKind,
        {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
        RHSInfo, Operands, CtxI, &Ctx.TLI);
  }
  case Instruction::Freeze: {
    // This opcode is unknown. Assume that it is the same as 'mul'.
    Type *VectorTy = toVectorTy(Ctx.Types.inferScalarType(this), VF);
    return Ctx.TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy,
                                          Ctx.CostKind);
  }
  case Instruction::ExtractValue: {
    return Ctx.TTI.getInsertExtractValueCost(Instruction::ExtractValue,
                                             Ctx.CostKind);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Instruction *CtxI = dyn_cast_or_null<Instruction>(getUnderlyingValue());
    Type *VectorTy = toVectorTy(Ctx.Types.inferScalarType(getOperand(0)), VF);
    return Ctx.TTI.getCmpSelInstrCost(
        Opcode, VectorTy, CmpInst::makeCmpResultType(VectorTy), getPredicate(),
        Ctx.CostKind, {TTI::OK_AnyValue, TTI::OP_None},
        {TTI::OK_AnyValue, TTI::OP_None}, CtxI);
  }
  default:
    llvm_unreachable("Unsupported opcode for instruction");
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenRecipe::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN ";
  printAsOperand(O, SlotTracker);
  O << " = " << Instruction::getOpcodeName(Opcode);
  printFlags(O);
  printOperands(O, SlotTracker);
}
#endif

void VPWidenCastRecipe::execute(VPTransformState &State) {
  auto &Builder = State.Builder;
  /// Vectorize casts.
  assert(State.VF.isVector() && "Not vectorizing?");
  Type *DestTy = VectorType::get(getResultType(), State.VF);
  VPValue *Op = getOperand(0);
  Value *A = State.get(Op);
  Value *Cast = Builder.CreateCast(Instruction::CastOps(Opcode), A, DestTy);
  State.set(this, Cast);
  if (auto *CastOp = dyn_cast<Instruction>(Cast)) {
    applyFlags(*CastOp);
    applyMetadata(*CastOp);
  }
}

InstructionCost VPWidenCastRecipe::computeCost(ElementCount VF,
                                               VPCostContext &Ctx) const {
  // TODO: In some cases, VPWidenCastRecipes are created but not considered in
  // the legacy cost model, including truncates/extends when evaluating a
  // reduction in a smaller type.
  if (!getUnderlyingValue())
    return 0;
  // Computes the CastContextHint from a recipes that may access memory.
  auto ComputeCCH = [&](const VPRecipeBase *R) -> TTI::CastContextHint {
    if (VF.isScalar())
      return TTI::CastContextHint::Normal;
    if (isa<VPInterleaveRecipe>(R))
      return TTI::CastContextHint::Interleave;
    if (const auto *ReplicateRecipe = dyn_cast<VPReplicateRecipe>(R))
      return ReplicateRecipe->isPredicated() ? TTI::CastContextHint::Masked
                                             : TTI::CastContextHint::Normal;
    const auto *WidenMemoryRecipe = dyn_cast<VPWidenMemoryRecipe>(R);
    if (WidenMemoryRecipe == nullptr)
      return TTI::CastContextHint::None;
    if (!WidenMemoryRecipe->isConsecutive())
      return TTI::CastContextHint::GatherScatter;
    if (WidenMemoryRecipe->isReverse())
      return TTI::CastContextHint::Reversed;
    if (WidenMemoryRecipe->isMasked())
      return TTI::CastContextHint::Masked;
    return TTI::CastContextHint::Normal;
  };

  VPValue *Operand = getOperand(0);
  TTI::CastContextHint CCH = TTI::CastContextHint::None;
  // For Trunc/FPTrunc, get the context from the only user.
  if ((Opcode == Instruction::Trunc || Opcode == Instruction::FPTrunc) &&
      !hasMoreThanOneUniqueUser() && getNumUsers() > 0) {
    if (auto *StoreRecipe = dyn_cast<VPRecipeBase>(*user_begin()))
      CCH = ComputeCCH(StoreRecipe);
  }
  // For Z/Sext, get the context from the operand.
  else if (Opcode == Instruction::ZExt || Opcode == Instruction::SExt ||
           Opcode == Instruction::FPExt) {
    if (Operand->isLiveIn())
      CCH = TTI::CastContextHint::Normal;
    else if (Operand->getDefiningRecipe())
      CCH = ComputeCCH(Operand->getDefiningRecipe());
  }

  auto *SrcTy =
      cast<VectorType>(toVectorTy(Ctx.Types.inferScalarType(Operand), VF));
  auto *DestTy = cast<VectorType>(toVectorTy(getResultType(), VF));
  // Arm TTI will use the underlying instruction to determine the cost.
  return Ctx.TTI.getCastInstrCost(
      Opcode, DestTy, SrcTy, CCH, Ctx.CostKind,
      dyn_cast_if_present<Instruction>(getUnderlyingValue()));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenCastRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-CAST ";
  printAsOperand(O, SlotTracker);
  O << " = " << Instruction::getOpcodeName(Opcode);
  printFlags(O);
  printOperands(O, SlotTracker);
  O << " to " << *getResultType();
}
#endif

InstructionCost VPHeaderPHIRecipe::computeCost(ElementCount VF,
                                               VPCostContext &Ctx) const {
  return Ctx.TTI.getCFInstrCost(Instruction::PHI, Ctx.CostKind);
}

/// A helper function that returns an integer or floating-point constant with
/// value C.
static Constant *getSignedIntOrFpConstant(Type *Ty, int64_t C) {
  return Ty->isIntegerTy() ? ConstantInt::getSigned(Ty, C)
                           : ConstantFP::get(Ty, C);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenIntOrFpInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << " = WIDEN-INDUCTION  ";
  printOperands(O, SlotTracker);

  if (auto *TI = getTruncInst())
    O << " (truncated to " << *TI->getType() << ")";
}
#endif

bool VPWidenIntOrFpInductionRecipe::isCanonical() const {
  // The step may be defined by a recipe in the preheader (e.g. if it requires
  // SCEV expansion), but for the canonical induction the step is required to be
  // 1, which is represented as live-in.
  if (getStepValue()->getDefiningRecipe())
    return false;
  auto *StepC = dyn_cast<ConstantInt>(getStepValue()->getLiveInIRValue());
  auto *StartC = dyn_cast<ConstantInt>(getStartValue()->getLiveInIRValue());
  auto *CanIV = cast<VPCanonicalIVPHIRecipe>(&*getParent()->begin());
  return StartC && StartC->isZero() && StepC && StepC->isOne() &&
         getScalarType() == CanIV->getScalarType();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPDerivedIVRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << " = DERIVED-IV ";
  getStartValue()->printAsOperand(O, SlotTracker);
  O << " + ";
  getOperand(1)->printAsOperand(O, SlotTracker);
  O << " * ";
  getStepValue()->printAsOperand(O, SlotTracker);
}
#endif

void VPScalarIVStepsRecipe::execute(VPTransformState &State) {
  // Fast-math-flags propagate from the original induction instruction.
  IRBuilder<>::FastMathFlagGuard FMFG(State.Builder);
  if (hasFastMathFlags())
    State.Builder.setFastMathFlags(getFastMathFlags());

  /// Compute scalar induction steps. \p ScalarIV is the scalar induction
  /// variable on which to base the steps, \p Step is the size of the step.

  Value *BaseIV = State.get(getOperand(0), VPLane(0));
  Value *Step = State.get(getStepValue(), VPLane(0));
  IRBuilderBase &Builder = State.Builder;

  // Ensure step has the same type as that of scalar IV.
  Type *BaseIVTy = BaseIV->getType()->getScalarType();
  assert(BaseIVTy == Step->getType() && "Types of BaseIV and Step must match!");

  // We build scalar steps for both integer and floating-point induction
  // variables. Here, we determine the kind of arithmetic we will perform.
  Instruction::BinaryOps AddOp;
  Instruction::BinaryOps MulOp;
  if (BaseIVTy->isIntegerTy()) {
    AddOp = Instruction::Add;
    MulOp = Instruction::Mul;
  } else {
    AddOp = InductionOpcode;
    MulOp = Instruction::FMul;
  }

  // Determine the number of scalars we need to generate for each unroll
  // iteration.
  bool FirstLaneOnly = vputils::onlyFirstLaneUsed(this);
  // Compute the scalar steps and save the results in State.
  Type *IntStepTy =
      IntegerType::get(BaseIVTy->getContext(), BaseIVTy->getScalarSizeInBits());
  Type *VecIVTy = nullptr;
  Value *UnitStepVec = nullptr, *SplatStep = nullptr, *SplatIV = nullptr;
  if (!FirstLaneOnly && State.VF.isScalable()) {
    VecIVTy = VectorType::get(BaseIVTy, State.VF);
    UnitStepVec =
        Builder.CreateStepVector(VectorType::get(IntStepTy, State.VF));
    SplatStep = Builder.CreateVectorSplat(State.VF, Step);
    SplatIV = Builder.CreateVectorSplat(State.VF, BaseIV);
  }

  unsigned StartLane = 0;
  unsigned EndLane = FirstLaneOnly ? 1 : State.VF.getKnownMinValue();
  if (State.Lane) {
    StartLane = State.Lane->getKnownLane();
    EndLane = StartLane + 1;
  }
  Value *StartIdx0;
  if (getUnrollPart(*this) == 0)
    StartIdx0 = ConstantInt::get(IntStepTy, 0);
  else {
    StartIdx0 = State.get(getOperand(2), true);
    if (getUnrollPart(*this) != 1) {
      StartIdx0 =
          Builder.CreateMul(StartIdx0, ConstantInt::get(StartIdx0->getType(),
                                                        getUnrollPart(*this)));
    }
    StartIdx0 = Builder.CreateSExtOrTrunc(StartIdx0, IntStepTy);
  }

  if (!FirstLaneOnly && State.VF.isScalable()) {
    auto *SplatStartIdx = Builder.CreateVectorSplat(State.VF, StartIdx0);
    auto *InitVec = Builder.CreateAdd(SplatStartIdx, UnitStepVec);
    if (BaseIVTy->isFloatingPointTy())
      InitVec = Builder.CreateSIToFP(InitVec, VecIVTy);
    auto *Mul = Builder.CreateBinOp(MulOp, InitVec, SplatStep);
    auto *Add = Builder.CreateBinOp(AddOp, SplatIV, Mul);
    State.set(this, Add);
    // It's useful to record the lane values too for the known minimum number
    // of elements so we do those below. This improves the code quality when
    // trying to extract the first element, for example.
  }

  if (BaseIVTy->isFloatingPointTy())
    StartIdx0 = Builder.CreateSIToFP(StartIdx0, BaseIVTy);

  for (unsigned Lane = StartLane; Lane < EndLane; ++Lane) {
    Value *StartIdx = Builder.CreateBinOp(
        AddOp, StartIdx0, getSignedIntOrFpConstant(BaseIVTy, Lane));
    // The step returned by `createStepForVF` is a runtime-evaluated value
    // when VF is scalable. Otherwise, it should be folded into a Constant.
    assert((State.VF.isScalable() || isa<Constant>(StartIdx)) &&
           "Expected StartIdx to be folded to a constant when VF is not "
           "scalable");
    auto *Mul = Builder.CreateBinOp(MulOp, StartIdx, Step);
    auto *Add = Builder.CreateBinOp(AddOp, BaseIV, Mul);
    State.set(this, Add, VPLane(Lane));
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPScalarIVStepsRecipe::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << " = SCALAR-STEPS ";
  printOperands(O, SlotTracker);
}
#endif

void VPWidenGEPRecipe::execute(VPTransformState &State) {
  assert(State.VF.isVector() && "not widening");
  auto *GEP = cast<GetElementPtrInst>(getUnderlyingInstr());
  // Construct a vector GEP by widening the operands of the scalar GEP as
  // necessary. We mark the vector GEP 'inbounds' if appropriate. A GEP
  // results in a vector of pointers when at least one operand of the GEP
  // is vector-typed. Thus, to keep the representation compact, we only use
  // vector-typed operands for loop-varying values.

  if (areAllOperandsInvariant()) {
    // If we are vectorizing, but the GEP has only loop-invariant operands,
    // the GEP we build (by only using vector-typed operands for
    // loop-varying values) would be a scalar pointer. Thus, to ensure we
    // produce a vector of pointers, we need to either arbitrarily pick an
    // operand to broadcast, or broadcast a clone of the original GEP.
    // Here, we broadcast a clone of the original.
    //
    // TODO: If at some point we decide to scalarize instructions having
    //       loop-invariant operands, this special case will no longer be
    //       required. We would add the scalarization decision to
    //       collectLoopScalars() and teach getVectorValue() to broadcast
    //       the lane-zero scalar value.
    SmallVector<Value *> Ops;
    for (unsigned I = 0, E = getNumOperands(); I != E; I++)
      Ops.push_back(State.get(getOperand(I), VPLane(0)));

    auto *NewGEP = State.Builder.CreateGEP(GEP->getSourceElementType(), Ops[0],
                                           ArrayRef(Ops).drop_front(), "",
                                           getGEPNoWrapFlags());
    Value *Splat = State.Builder.CreateVectorSplat(State.VF, NewGEP);
    State.set(this, Splat);
  } else {
    // If the GEP has at least one loop-varying operand, we are sure to
    // produce a vector of pointers unless VF is scalar.
    // The pointer operand of the new GEP. If it's loop-invariant, we
    // won't broadcast it.
    auto *Ptr = isPointerLoopInvariant() ? State.get(getOperand(0), VPLane(0))
                                         : State.get(getOperand(0));

    // Collect all the indices for the new GEP. If any index is
    // loop-invariant, we won't broadcast it.
    SmallVector<Value *, 4> Indices;
    for (unsigned I = 1, E = getNumOperands(); I < E; I++) {
      VPValue *Operand = getOperand(I);
      if (isIndexLoopInvariant(I - 1))
        Indices.push_back(State.get(Operand, VPLane(0)));
      else
        Indices.push_back(State.get(Operand));
    }

    // Create the new GEP. Note that this GEP may be a scalar if VF == 1,
    // but it should be a vector, otherwise.
    auto *NewGEP = State.Builder.CreateGEP(GEP->getSourceElementType(), Ptr,
                                           Indices, "", getGEPNoWrapFlags());
    assert((State.VF.isScalar() || NewGEP->getType()->isVectorTy()) &&
           "NewGEP is not a pointer vector");
    State.set(this, NewGEP);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenGEPRecipe::print(raw_ostream &O, const Twine &Indent,
                             VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-GEP ";
  O << (isPointerLoopInvariant() ? "Inv" : "Var");
  for (size_t I = 0; I < getNumOperands() - 1; ++I)
    O << "[" << (isIndexLoopInvariant(I) ? "Inv" : "Var") << "]";

  O << " ";
  printAsOperand(O, SlotTracker);
  O << " = getelementptr";
  printFlags(O);
  printOperands(O, SlotTracker);
}
#endif

static Type *getGEPIndexTy(bool IsScalable, bool IsReverse, bool IsUnitStride,
                           unsigned CurrentPart, IRBuilderBase &Builder) {
  // Use i32 for the gep index type when the value is constant,
  // or query DataLayout for a more suitable index type otherwise.
  const DataLayout &DL = Builder.GetInsertBlock()->getDataLayout();
  return !IsUnitStride || (IsScalable && (IsReverse || CurrentPart > 0))
             ? DL.getIndexType(Builder.getPtrTy(0))
             : Builder.getInt32Ty();
}

void VPVectorEndPointerRecipe::execute(VPTransformState &State) {
  auto &Builder = State.Builder;
  unsigned CurrentPart = getUnrollPart(*this);
  bool IsUnitStride = Stride == 1 || Stride == -1;
  Type *IndexTy = getGEPIndexTy(State.VF.isScalable(), /*IsReverse*/ true,
                                IsUnitStride, CurrentPart, Builder);

  // The wide store needs to start at the last vector element.
  Value *RunTimeVF = State.get(getVFValue(), VPLane(0));
  if (IndexTy != RunTimeVF->getType())
    RunTimeVF = Builder.CreateZExtOrTrunc(RunTimeVF, IndexTy);
  // NumElt = Stride * CurrentPart * RunTimeVF
  Value *NumElt = Builder.CreateMul(
      ConstantInt::get(IndexTy, Stride * (int64_t)CurrentPart), RunTimeVF);
  // LastLane = Stride * (RunTimeVF - 1)
  Value *LastLane = Builder.CreateSub(RunTimeVF, ConstantInt::get(IndexTy, 1));
  if (Stride != 1)
    LastLane = Builder.CreateMul(ConstantInt::get(IndexTy, Stride), LastLane);
  Value *Ptr = State.get(getOperand(0), VPLane(0));
  Value *ResultPtr =
      Builder.CreateGEP(IndexedTy, Ptr, NumElt, "", getGEPNoWrapFlags());
  ResultPtr = Builder.CreateGEP(IndexedTy, ResultPtr, LastLane, "",
                                getGEPNoWrapFlags());

  State.set(this, ResultPtr, /*IsScalar*/ true);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPVectorEndPointerRecipe::print(raw_ostream &O, const Twine &Indent,
                                     VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << " = vector-end-pointer";
  printFlags(O);
  printOperands(O, SlotTracker);
}
#endif

void VPVectorPointerRecipe::execute(VPTransformState &State) {
  auto &Builder = State.Builder;
  unsigned CurrentPart = getUnrollPart(*this);
  Type *IndexTy = getGEPIndexTy(State.VF.isScalable(), /*IsReverse*/ false,
                                /*IsUnitStride*/ true, CurrentPart, Builder);
  Value *Ptr = State.get(getOperand(0), VPLane(0));

  Value *Increment = createStepForVF(Builder, IndexTy, State.VF, CurrentPart);
  Value *ResultPtr =
      Builder.CreateGEP(IndexedTy, Ptr, Increment, "", getGEPNoWrapFlags());

  State.set(this, ResultPtr, /*IsScalar*/ true);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPVectorPointerRecipe::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << " = vector-pointer ";

  printOperands(O, SlotTracker);
}
#endif

void VPBlendRecipe::execute(VPTransformState &State) {
  assert(isNormalized() && "Expected blend to be normalized!");
  // We know that all PHIs in non-header blocks are converted into
  // selects, so we don't have to worry about the insertion order and we
  // can just use the builder.
  // At this point we generate the predication tree. There may be
  // duplications since this is a simple recursive scan, but future
  // optimizations will clean it up.

  unsigned NumIncoming = getNumIncomingValues();

  // Generate a sequence of selects of the form:
  // SELECT(Mask3, In3,
  //        SELECT(Mask2, In2,
  //               SELECT(Mask1, In1,
  //                      In0)))
  // Note that Mask0 is never used: lanes for which no path reaches this phi and
  // are essentially undef are taken from In0.
  bool OnlyFirstLaneUsed = vputils::onlyFirstLaneUsed(this);
  Value *Result = nullptr;
  for (unsigned In = 0; In < NumIncoming; ++In) {
    // We might have single edge PHIs (blocks) - use an identity
    // 'select' for the first PHI operand.
    Value *In0 = State.get(getIncomingValue(In), OnlyFirstLaneUsed);
    if (In == 0)
      Result = In0; // Initialize with the first incoming value.
    else {
      // Select between the current value and the previous incoming edge
      // based on the incoming mask.
      Value *Cond = State.get(getMask(In), OnlyFirstLaneUsed);
      Result = State.Builder.CreateSelect(Cond, In0, Result, "predphi");
    }
  }
  State.set(this, Result, OnlyFirstLaneUsed);
}

InstructionCost VPBlendRecipe::computeCost(ElementCount VF,
                                           VPCostContext &Ctx) const {
  // Handle cases where only the first lane is used the same way as the legacy
  // cost model.
  if (vputils::onlyFirstLaneUsed(this))
    return Ctx.TTI.getCFInstrCost(Instruction::PHI, Ctx.CostKind);

  Type *ResultTy = toVectorTy(Ctx.Types.inferScalarType(this), VF);
  Type *CmpTy = toVectorTy(Type::getInt1Ty(Ctx.Types.getContext()), VF);
  return (getNumIncomingValues() - 1) *
         Ctx.TTI.getCmpSelInstrCost(Instruction::Select, ResultTy, CmpTy,
                                    CmpInst::BAD_ICMP_PREDICATE, Ctx.CostKind);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPBlendRecipe::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "BLEND ";
  printAsOperand(O, SlotTracker);
  O << " =";
  if (getNumIncomingValues() == 1) {
    // Not a User of any mask: not really blending, this is a
    // single-predecessor phi.
    O << " ";
    getIncomingValue(0)->printAsOperand(O, SlotTracker);
  } else {
    for (unsigned I = 0, E = getNumIncomingValues(); I < E; ++I) {
      O << " ";
      getIncomingValue(I)->printAsOperand(O, SlotTracker);
      if (I == 0)
        continue;
      O << "/";
      getMask(I)->printAsOperand(O, SlotTracker);
    }
  }
}
#endif

void VPReductionRecipe::execute(VPTransformState &State) {
  assert(!State.Lane && "Reduction being replicated.");
  Value *PrevInChain = State.get(getChainOp(), /*IsScalar*/ true);
  RecurKind Kind = getRecurrenceKind();
  assert(!RecurrenceDescriptor::isAnyOfRecurrenceKind(Kind) &&
         "In-loop AnyOf reductions aren't currently supported");
  // Propagate the fast-math flags carried by the underlying instruction.
  IRBuilderBase::FastMathFlagGuard FMFGuard(State.Builder);
  State.Builder.setFastMathFlags(getFastMathFlags());
  Value *NewVecOp = State.get(getVecOp());
  if (VPValue *Cond = getCondOp()) {
    Value *NewCond = State.get(Cond, State.VF.isScalar());
    VectorType *VecTy = dyn_cast<VectorType>(NewVecOp->getType());
    Type *ElementTy = VecTy ? VecTy->getElementType() : NewVecOp->getType();

    Value *Start = getRecurrenceIdentity(Kind, ElementTy, getFastMathFlags());
    if (State.VF.isVector())
      Start = State.Builder.CreateVectorSplat(VecTy->getElementCount(), Start);

    Value *Select = State.Builder.CreateSelect(NewCond, NewVecOp, Start);
    NewVecOp = Select;
  }
  Value *NewRed;
  Value *NextInChain;
  if (IsOrdered) {
    if (State.VF.isVector())
      NewRed =
          createOrderedReduction(State.Builder, Kind, NewVecOp, PrevInChain);
    else
      NewRed = State.Builder.CreateBinOp(
          (Instruction::BinaryOps)RecurrenceDescriptor::getOpcode(Kind),
          PrevInChain, NewVecOp);
    PrevInChain = NewRed;
    NextInChain = NewRed;
  } else {
    PrevInChain = State.get(getChainOp(), /*IsScalar*/ true);
    NewRed = createSimpleReduction(State.Builder, NewVecOp, Kind);
    if (RecurrenceDescriptor::isMinMaxRecurrenceKind(Kind))
      NextInChain = createMinMaxOp(State.Builder, Kind, NewRed, PrevInChain);
    else
      NextInChain = State.Builder.CreateBinOp(
          (Instruction::BinaryOps)RecurrenceDescriptor::getOpcode(Kind), NewRed,
          PrevInChain);
  }
  State.set(this, NextInChain, /*IsScalar*/ true);
}

void VPReductionEVLRecipe::execute(VPTransformState &State) {
  assert(!State.Lane && "Reduction being replicated.");

  auto &Builder = State.Builder;
  // Propagate the fast-math flags carried by the underlying instruction.
  IRBuilderBase::FastMathFlagGuard FMFGuard(Builder);
  Builder.setFastMathFlags(getFastMathFlags());

  RecurKind Kind = getRecurrenceKind();
  Value *Prev = State.get(getChainOp(), /*IsScalar*/ true);
  Value *VecOp = State.get(getVecOp());
  Value *EVL = State.get(getEVL(), VPLane(0));

  Value *Mask;
  if (VPValue *CondOp = getCondOp())
    Mask = State.get(CondOp);
  else
    Mask = Builder.CreateVectorSplat(State.VF, Builder.getTrue());

  Value *NewRed;
  if (isOrdered()) {
    NewRed = createOrderedReduction(Builder, Kind, VecOp, Prev, Mask, EVL);
  } else {
    NewRed = createSimpleReduction(Builder, VecOp, Kind, Mask, EVL);
    if (RecurrenceDescriptor::isMinMaxRecurrenceKind(Kind))
      NewRed = createMinMaxOp(Builder, Kind, NewRed, Prev);
    else
      NewRed = Builder.CreateBinOp(
          (Instruction::BinaryOps)RecurrenceDescriptor::getOpcode(Kind), NewRed,
          Prev);
  }
  State.set(this, NewRed, /*IsScalar*/ true);
}

InstructionCost VPReductionRecipe::computeCost(ElementCount VF,
                                               VPCostContext &Ctx) const {
  RecurKind RdxKind = getRecurrenceKind();
  Type *ElementTy = Ctx.Types.inferScalarType(this);
  auto *VectorTy = cast<VectorType>(toVectorTy(ElementTy, VF));
  unsigned Opcode = RecurrenceDescriptor::getOpcode(RdxKind);
  FastMathFlags FMFs = getFastMathFlags();
  std::optional<FastMathFlags> OptionalFMF =
      ElementTy->isFloatingPointTy() ? std::make_optional(FMFs) : std::nullopt;

  // TODO: Support any-of reductions.
  assert(
      (!RecurrenceDescriptor::isAnyOfRecurrenceKind(RdxKind) ||
       ForceTargetInstructionCost.getNumOccurrences() > 0) &&
      "Any-of reduction not implemented in VPlan-based cost model currently.");

  // Note that TTI should model the cost of moving result to the scalar register
  // and the BinOp cost in the getMinMaxReductionCost().
  if (RecurrenceDescriptor::isMinMaxRecurrenceKind(RdxKind)) {
    Intrinsic::ID Id = getMinMaxReductionIntrinsicOp(RdxKind);
    return Ctx.TTI.getMinMaxReductionCost(Id, VectorTy, FMFs, Ctx.CostKind);
  }

  // Note that TTI should model the cost of moving result to the scalar register
  // and the BinOp cost in the getArithmeticReductionCost().
  return Ctx.TTI.getArithmeticReductionCost(Opcode, VectorTy, OptionalFMF,
                                            Ctx.CostKind);
}

VPExpressionRecipe::VPExpressionRecipe(
    ExpressionTypes ExpressionType,
    ArrayRef<VPSingleDefRecipe *> ExpressionRecipes)
    : VPSingleDefRecipe(VPDef::VPExpressionSC, {}, {}),
      ExpressionRecipes(SetVector<VPSingleDefRecipe *>(
                            ExpressionRecipes.begin(), ExpressionRecipes.end())
                            .takeVector()),
      ExpressionType(ExpressionType) {
  assert(!ExpressionRecipes.empty() && "Nothing to combine?");
  assert(
      none_of(ExpressionRecipes,
              [](VPSingleDefRecipe *R) { return R->mayHaveSideEffects(); }) &&
      "expression cannot contain recipes with side-effects");

  // Maintain a copy of the expression recipes as a set of users.
  SmallPtrSet<VPUser *, 4> ExpressionRecipesAsSetOfUsers;
  for (auto *R : ExpressionRecipes)
    ExpressionRecipesAsSetOfUsers.insert(R);

  // Recipes in the expression, except the last one, must only be used by
  // (other) recipes inside the expression. If there are other users, external
  // to the expression, use a clone of the recipe for external users.
  for (VPSingleDefRecipe *R : ExpressionRecipes) {
    if (R != ExpressionRecipes.back() &&
        any_of(R->users(), [&ExpressionRecipesAsSetOfUsers](VPUser *U) {
          return !ExpressionRecipesAsSetOfUsers.contains(U);
        })) {
      // There are users outside of the expression. Clone the recipe and use the
      // clone those external users.
      VPSingleDefRecipe *CopyForExtUsers = R->clone();
      R->replaceUsesWithIf(CopyForExtUsers, [&ExpressionRecipesAsSetOfUsers](
                                                VPUser &U, unsigned) {
        return !ExpressionRecipesAsSetOfUsers.contains(&U);
      });
      CopyForExtUsers->insertBefore(R);
    }
    if (R->getParent())
      R->removeFromParent();
  }

  // Internalize all external operands to the expression recipes. To do so,
  // create new temporary VPValues for all operands defined by a recipe outside
  // the expression. The original operands are added as operands of the
  // VPExpressionRecipe itself.
  for (auto *R : ExpressionRecipes) {
    for (const auto &[Idx, Op] : enumerate(R->operands())) {
      auto *Def = Op->getDefiningRecipe();
      if (Def && ExpressionRecipesAsSetOfUsers.contains(Def))
        continue;
      addOperand(Op);
      LiveInPlaceholders.push_back(new VPValue());
      R->setOperand(Idx, LiveInPlaceholders.back());
    }
  }
}

void VPExpressionRecipe::decompose() {
  for (auto *R : ExpressionRecipes)
    R->insertBefore(this);

  for (const auto &[Idx, Op] : enumerate(operands()))
    LiveInPlaceholders[Idx]->replaceAllUsesWith(Op);

  replaceAllUsesWith(ExpressionRecipes.back());
  ExpressionRecipes.clear();
}

InstructionCost VPExpressionRecipe::computeCost(ElementCount VF,
                                                VPCostContext &Ctx) const {
  Type *RedTy = Ctx.Types.inferScalarType(this);
  auto *SrcVecTy = cast<VectorType>(
      toVectorTy(Ctx.Types.inferScalarType(getOperand(0)), VF));
  assert(RedTy->isIntegerTy() &&
         "VPExpressionRecipe only supports integer types currently.");
  switch (ExpressionType) {
  case ExpressionTypes::ExtendedReduction: {
    unsigned Opcode = RecurrenceDescriptor::getOpcode(
        cast<VPReductionRecipe>(ExpressionRecipes[1])->getRecurrenceKind());
    return Ctx.TTI.getExtendedReductionCost(
        Opcode,
        cast<VPWidenCastRecipe>(ExpressionRecipes.front())->getOpcode() ==
            Instruction::ZExt,
        RedTy, SrcVecTy, std::nullopt, Ctx.CostKind);
  }
  case ExpressionTypes::MulAccReduction:
    return Ctx.TTI.getMulAccReductionCost(false, RedTy, SrcVecTy, Ctx.CostKind);

  case ExpressionTypes::ExtMulAccReduction:
    return Ctx.TTI.getMulAccReductionCost(
        cast<VPWidenCastRecipe>(ExpressionRecipes.front())->getOpcode() ==
            Instruction::ZExt,
        RedTy, SrcVecTy, Ctx.CostKind);
  }
  llvm_unreachable("Unknown VPExpressionRecipe::ExpressionTypes enum");
}

bool VPExpressionRecipe::mayReadOrWriteMemory() const {
  return any_of(ExpressionRecipes, [](VPSingleDefRecipe *R) {
    return R->mayReadFromMemory() || R->mayWriteToMemory();
  });
}

bool VPExpressionRecipe::mayHaveSideEffects() const {
  assert(
      none_of(ExpressionRecipes,
              [](VPSingleDefRecipe *R) { return R->mayHaveSideEffects(); }) &&
      "expression cannot contain recipes with side-effects");
  return false;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

void VPExpressionRecipe::print(raw_ostream &O, const Twine &Indent,
                               VPSlotTracker &SlotTracker) const {
  O << Indent << "EXPRESSION ";
  printAsOperand(O, SlotTracker);
  O << " = ";
  auto *Red = cast<VPReductionRecipe>(ExpressionRecipes.back());
  unsigned Opcode = RecurrenceDescriptor::getOpcode(Red->getRecurrenceKind());

  switch (ExpressionType) {
  case ExpressionTypes::ExtendedReduction: {
    getOperand(1)->printAsOperand(O, SlotTracker);
    O << " +";
    O << " reduce." << Instruction::getOpcodeName(Opcode) << " (";
    getOperand(0)->printAsOperand(O, SlotTracker);
    Red->printFlags(O);

    auto *Ext0 = cast<VPWidenCastRecipe>(ExpressionRecipes[0]);
    O << Instruction::getOpcodeName(Ext0->getOpcode()) << " to "
      << *Ext0->getResultType();
    if (Red->isConditional()) {
      O << ", ";
      Red->getCondOp()->printAsOperand(O, SlotTracker);
    }
    O << ")";
    break;
  }
  case ExpressionTypes::MulAccReduction:
  case ExpressionTypes::ExtMulAccReduction: {
    getOperand(getNumOperands() - 1)->printAsOperand(O, SlotTracker);
    O << " + ";
    O << "reduce."
      << Instruction::getOpcodeName(
             RecurrenceDescriptor::getOpcode(Red->getRecurrenceKind()))
      << " (";
    O << "mul";
    bool IsExtended = ExpressionType == ExpressionTypes::ExtMulAccReduction;
    auto *Mul = cast<VPWidenRecipe>(IsExtended ? ExpressionRecipes[2]
                                               : ExpressionRecipes[0]);
    Mul->printFlags(O);
    if (IsExtended)
      O << "(";
    getOperand(0)->printAsOperand(O, SlotTracker);
    if (IsExtended) {
      auto *Ext0 = cast<VPWidenCastRecipe>(ExpressionRecipes[0]);
      O << " " << Instruction::getOpcodeName(Ext0->getOpcode()) << " to "
        << *Ext0->getResultType() << "), (";
    } else {
      O << ", ";
    }
    getOperand(1)->printAsOperand(O, SlotTracker);
    if (IsExtended) {
      auto *Ext1 = cast<VPWidenCastRecipe>(ExpressionRecipes[1]);
      O << " " << Instruction::getOpcodeName(Ext1->getOpcode()) << " to "
        << *Ext1->getResultType() << ")";
    }
    if (Red->isConditional()) {
      O << ", ";
      Red->getCondOp()->printAsOperand(O, SlotTracker);
    }
    O << ")";
    break;
  }
  }
}

void VPReductionRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "REDUCE ";
  printAsOperand(O, SlotTracker);
  O << " = ";
  getChainOp()->printAsOperand(O, SlotTracker);
  O << " +";
  printFlags(O);
  O << " reduce."
    << Instruction::getOpcodeName(
           RecurrenceDescriptor::getOpcode(getRecurrenceKind()))
    << " (";
  getVecOp()->printAsOperand(O, SlotTracker);
  if (isConditional()) {
    O << ", ";
    getCondOp()->printAsOperand(O, SlotTracker);
  }
  O << ")";
}

void VPReductionEVLRecipe::print(raw_ostream &O, const Twine &Indent,
                                 VPSlotTracker &SlotTracker) const {
  O << Indent << "REDUCE ";
  printAsOperand(O, SlotTracker);
  O << " = ";
  getChainOp()->printAsOperand(O, SlotTracker);
  O << " +";
  printFlags(O);
  O << " vp.reduce."
    << Instruction::getOpcodeName(
           RecurrenceDescriptor::getOpcode(getRecurrenceKind()))
    << " (";
  getVecOp()->printAsOperand(O, SlotTracker);
  O << ", ";
  getEVL()->printAsOperand(O, SlotTracker);
  if (isConditional()) {
    O << ", ";
    getCondOp()->printAsOperand(O, SlotTracker);
  }
  O << ")";
}

#endif

/// A helper function to scalarize a single Instruction in the innermost loop.
/// Generates a sequence of scalar instances for lane \p Lane. Uses the VPValue
/// operands from \p RepRecipe instead of \p Instr's operands.
static void scalarizeInstruction(const Instruction *Instr,
                                 VPReplicateRecipe *RepRecipe,
                                 const VPLane &Lane, VPTransformState &State) {
  assert((!Instr->getType()->isAggregateType() ||
          canVectorizeTy(Instr->getType())) &&
         "Expected vectorizable or non-aggregate type.");

  // Does this instruction return a value ?
  bool IsVoidRetTy = Instr->getType()->isVoidTy();

  Instruction *Cloned = Instr->clone();
  if (!IsVoidRetTy) {
    Cloned->setName(Instr->getName() + ".cloned");
#if !defined(NDEBUG)
    // Verify that VPlan type inference results agree with the type of the
    // generated values.
    assert(State.TypeAnalysis.inferScalarType(RepRecipe) == Cloned->getType() &&
           "inferred type and type from generated instructions do not match");
#endif
  }

  RepRecipe->applyFlags(*Cloned);
  RepRecipe->applyMetadata(*Cloned);

  if (auto DL = RepRecipe->getDebugLoc())
    State.setDebugLocFrom(DL);

  // Replace the operands of the cloned instructions with their scalar
  // equivalents in the new loop.
  for (const auto &I : enumerate(RepRecipe->operands())) {
    auto InputLane = Lane;
    VPValue *Operand = I.value();
    if (vputils::isSingleScalar(Operand))
      InputLane = VPLane::getFirstLane();
    Cloned->setOperand(I.index(), State.get(Operand, InputLane));
  }

  // Place the cloned scalar in the new loop.
  State.Builder.Insert(Cloned);

  State.set(RepRecipe, Cloned, Lane);

  // If we just cloned a new assumption, add it the assumption cache.
  if (auto *II = dyn_cast<AssumeInst>(Cloned))
    State.AC->registerAssumption(II);

  assert(
      (RepRecipe->getParent()->getParent() ||
       !RepRecipe->getParent()->getPlan()->getVectorLoopRegion() ||
       all_of(RepRecipe->operands(),
              [](VPValue *Op) { return Op->isDefinedOutsideLoopRegions(); })) &&
      "Expected a recipe is either within a region or all of its operands "
      "are defined outside the vectorized region.");
}

void VPReplicateRecipe::execute(VPTransformState &State) {
  Instruction *UI = getUnderlyingInstr();

  if (!State.Lane) {
    assert(IsSingleScalar && "VPReplicateRecipes outside replicate regions "
                             "must have already been unrolled");
    scalarizeInstruction(UI, this, VPLane(0), State);
    return;
  }

  assert((State.VF.isScalar() || !isSingleScalar()) &&
         "uniform recipe shouldn't be predicated");
  assert(!State.VF.isScalable() && "Can't scalarize a scalable vector");
  scalarizeInstruction(UI, this, *State.Lane, State);
  // Insert scalar instance packing it into a vector.
  if (State.VF.isVector() && shouldPack()) {
    Value *WideValue =
        State.Lane->isFirstLane()
            ? PoisonValue::get(VectorType::get(UI->getType(), State.VF))
            : State.get(this);
    State.set(this, State.packScalarIntoVectorizedValue(this, WideValue,
                                                        *State.Lane));
  }
}

bool VPReplicateRecipe::shouldPack() const {
  // Find if the recipe is used by a widened recipe via an intervening
  // VPPredInstPHIRecipe. In this case, also pack the scalar values in a vector.
  return any_of(users(), [](const VPUser *U) {
    if (auto *PredR = dyn_cast<VPPredInstPHIRecipe>(U))
      return any_of(PredR->users(), [PredR](const VPUser *U) {
        return !U->usesScalars(PredR);
      });
    return false;
  });
}

InstructionCost VPReplicateRecipe::computeCost(ElementCount VF,
                                               VPCostContext &Ctx) const {
  Instruction *UI = cast<Instruction>(getUnderlyingValue());
  // VPReplicateRecipe may be cloned as part of an existing VPlan-to-VPlan
  // transform, avoid computing their cost multiple times for now.
  Ctx.SkipCostComputation.insert(UI);

  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  Type *ResultTy = Ctx.Types.inferScalarType(this);
  switch (UI->getOpcode()) {
  case Instruction::GetElementPtr:
    // We mark this instruction as zero-cost because the cost of GEPs in
    // vectorized code depends on whether the corresponding memory instruction
    // is scalarized or not. Therefore, we handle GEPs with the memory
    // instruction cost.
    return 0;
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    auto Op2Info = Ctx.getOperandInfo(getOperand(1));
    SmallVector<const Value *, 4> Operands(UI->operand_values());
    return Ctx.TTI.getArithmeticInstrCost(
               UI->getOpcode(), ResultTy, CostKind,
               {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
               Op2Info, Operands, UI, &Ctx.TLI) *
           (isSingleScalar() ? 1 : VF.getFixedValue());
  }
  }

  return Ctx.getLegacyCost(UI, VF);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPReplicateRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << (IsSingleScalar ? "CLONE " : "REPLICATE ");

  if (!getUnderlyingInstr()->getType()->isVoidTy()) {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }
  if (auto *CB = dyn_cast<CallBase>(getUnderlyingInstr())) {
    O << "call";
    printFlags(O);
    O << "@" << CB->getCalledFunction()->getName() << "(";
    interleaveComma(make_range(op_begin(), op_begin() + (getNumOperands() - 1)),
                    O, [&O, &SlotTracker](VPValue *Op) {
                      Op->printAsOperand(O, SlotTracker);
                    });
    O << ")";
  } else {
    O << Instruction::getOpcodeName(getUnderlyingInstr()->getOpcode());
    printFlags(O);
    printOperands(O, SlotTracker);
  }

  if (shouldPack())
    O << " (S->V)";
}
#endif

void VPBranchOnMaskRecipe::execute(VPTransformState &State) {
  assert(State.Lane && "Branch on Mask works only on single instance.");

  VPValue *BlockInMask = getOperand(0);
  Value *ConditionBit = State.get(BlockInMask, *State.Lane);

  // Replace the temporary unreachable terminator with a new conditional branch,
  // whose two destinations will be set later when they are created.
  auto *CurrentTerminator = State.CFG.PrevBB->getTerminator();
  assert(isa<UnreachableInst>(CurrentTerminator) &&
         "Expected to replace unreachable terminator with conditional branch.");
  auto CondBr =
      State.Builder.CreateCondBr(ConditionBit, State.CFG.PrevBB, nullptr);
  CondBr->setSuccessor(0, nullptr);
  CurrentTerminator->eraseFromParent();
}

InstructionCost VPBranchOnMaskRecipe::computeCost(ElementCount VF,
                                                  VPCostContext &Ctx) const {
  // The legacy cost model doesn't assign costs to branches for individual
  // replicate regions. Match the current behavior in the VPlan cost model for
  // now.
  return 0;
}

void VPPredInstPHIRecipe::execute(VPTransformState &State) {
  assert(State.Lane && "Predicated instruction PHI works per instance.");
  Instruction *ScalarPredInst =
      cast<Instruction>(State.get(getOperand(0), *State.Lane));
  BasicBlock *PredicatedBB = ScalarPredInst->getParent();
  BasicBlock *PredicatingBB = PredicatedBB->getSinglePredecessor();
  assert(PredicatingBB && "Predicated block has no single predecessor.");
  assert(isa<VPReplicateRecipe>(getOperand(0)) &&
         "operand must be VPReplicateRecipe");

  // By current pack/unpack logic we need to generate only a single phi node: if
  // a vector value for the predicated instruction exists at this point it means
  // the instruction has vector users only, and a phi for the vector value is
  // needed. In this case the recipe of the predicated instruction is marked to
  // also do that packing, thereby "hoisting" the insert-element sequence.
  // Otherwise, a phi node for the scalar value is needed.
  if (State.hasVectorValue(getOperand(0))) {
    Value *VectorValue = State.get(getOperand(0));
    InsertElementInst *IEI = cast<InsertElementInst>(VectorValue);
    PHINode *VPhi = State.Builder.CreatePHI(IEI->getType(), 2);
    VPhi->addIncoming(IEI->getOperand(0), PredicatingBB); // Unmodified vector.
    VPhi->addIncoming(IEI, PredicatedBB); // New vector with inserted element.
    if (State.hasVectorValue(this))
      State.reset(this, VPhi);
    else
      State.set(this, VPhi);
    // NOTE: Currently we need to update the value of the operand, so the next
    // predicated iteration inserts its generated value in the correct vector.
    State.reset(getOperand(0), VPhi);
  } else {
    if (vputils::onlyFirstLaneUsed(this) && !State.Lane->isFirstLane())
      return;

    Type *PredInstType = State.TypeAnalysis.inferScalarType(getOperand(0));
    PHINode *Phi = State.Builder.CreatePHI(PredInstType, 2);
    Phi->addIncoming(PoisonValue::get(ScalarPredInst->getType()),
                     PredicatingBB);
    Phi->addIncoming(ScalarPredInst, PredicatedBB);
    if (State.hasScalarValue(this, *State.Lane))
      State.reset(this, Phi, *State.Lane);
    else
      State.set(this, Phi, *State.Lane);
    // NOTE: Currently we need to update the value of the operand, so the next
    // predicated iteration inserts its generated value in the correct vector.
    State.reset(getOperand(0), Phi, *State.Lane);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPPredInstPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << Indent << "PHI-PREDICATED-INSTRUCTION ";
  printAsOperand(O, SlotTracker);
  O << " = ";
  printOperands(O, SlotTracker);
}
#endif

InstructionCost VPWidenMemoryRecipe::computeCost(ElementCount VF,
                                                 VPCostContext &Ctx) const {
  Type *Ty = toVectorTy(getLoadStoreType(&Ingredient), VF);
  const Align Alignment = getLoadStoreAlignment(&Ingredient);
  unsigned AS = cast<PointerType>(Ctx.Types.inferScalarType(getAddr()))
                    ->getAddressSpace();
  unsigned Opcode = isa<VPWidenLoadRecipe, VPWidenLoadEVLRecipe>(this)
                        ? Instruction::Load
                        : Instruction::Store;

  if (!Consecutive) {
    // TODO: Using the original IR may not be accurate.
    // Currently, ARM will use the underlying IR to calculate gather/scatter
    // instruction cost.
    const Value *Ptr = getLoadStorePointerOperand(&Ingredient);
    assert(!Reverse &&
           "Inconsecutive memory access should not have the order.");
    return Ctx.TTI.getAddressComputationCost(Ty) +
           Ctx.TTI.getGatherScatterOpCost(Opcode, Ty, Ptr, IsMasked, Alignment,
                                          Ctx.CostKind, &Ingredient);
  }

  InstructionCost Cost = 0;
  if (IsMasked) {
    Cost +=
        Ctx.TTI.getMaskedMemoryOpCost(Opcode, Ty, Alignment, AS, Ctx.CostKind);
  } else {
    TTI::OperandValueInfo OpInfo = Ctx.getOperandInfo(
        isa<VPWidenLoadRecipe, VPWidenLoadEVLRecipe>(this) ? getOperand(0)
                                                           : getOperand(1));
    Cost += Ctx.TTI.getMemoryOpCost(Opcode, Ty, Alignment, AS, Ctx.CostKind,
                                    OpInfo, &Ingredient);
  }
  if (!Reverse)
    return Cost;

  return Cost += Ctx.TTI.getShuffleCost(
             TargetTransformInfo::SK_Reverse, cast<VectorType>(Ty),
             cast<VectorType>(Ty), {}, Ctx.CostKind, 0);
}

void VPWidenLoadRecipe::execute(VPTransformState &State) {
  Type *ScalarDataTy = getLoadStoreType(&Ingredient);
  auto *DataTy = VectorType::get(ScalarDataTy, State.VF);
  const Align Alignment = getLoadStoreAlignment(&Ingredient);
  bool CreateGather = !isConsecutive();

  auto &Builder = State.Builder;
  Value *Mask = nullptr;
  if (auto *VPMask = getMask()) {
    // Mask reversal is only needed for non-all-one (null) masks, as reverse
    // of a null all-one mask is a null mask.
    Mask = State.get(VPMask);
    if (isReverse())
      Mask = Builder.CreateVectorReverse(Mask, "reverse");
  }

  Value *Addr = State.get(getAddr(), /*IsScalar*/ !CreateGather);
  Value *NewLI;
  if (CreateGather) {
    NewLI = Builder.CreateMaskedGather(DataTy, Addr, Alignment, Mask, nullptr,
                                       "wide.masked.gather");
  } else if (Mask) {
    NewLI =
        Builder.CreateMaskedLoad(DataTy, Addr, Alignment, Mask,
                                 PoisonValue::get(DataTy), "wide.masked.load");
  } else {
    NewLI = Builder.CreateAlignedLoad(DataTy, Addr, Alignment, "wide.load");
  }
  applyMetadata(*cast<Instruction>(NewLI));
  if (Reverse)
    NewLI = Builder.CreateVectorReverse(NewLI, "reverse");
  State.set(this, NewLI);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenLoadRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN ";
  printAsOperand(O, SlotTracker);
  O << " = load ";
  printOperands(O, SlotTracker);
}
#endif

/// Use all-true mask for reverse rather than actual mask, as it avoids a
/// dependence w/o affecting the result.
static Instruction *createReverseEVL(IRBuilderBase &Builder, Value *Operand,
                                     Value *EVL, const Twine &Name) {
  VectorType *ValTy = cast<VectorType>(Operand->getType());
  Value *AllTrueMask =
      Builder.CreateVectorSplat(ValTy->getElementCount(), Builder.getTrue());
  return Builder.CreateIntrinsic(ValTy, Intrinsic::experimental_vp_reverse,
                                 {Operand, AllTrueMask, EVL}, nullptr, Name);
}

void VPWidenLoadEVLRecipe::execute(VPTransformState &State) {
  Type *ScalarDataTy = getLoadStoreType(&Ingredient);
  auto *DataTy = VectorType::get(ScalarDataTy, State.VF);
  const Align Alignment = getLoadStoreAlignment(&Ingredient);
  bool CreateGather = !isConsecutive();

  auto &Builder = State.Builder;
  CallInst *NewLI;
  Value *EVL = State.get(getEVL(), VPLane(0));
  Value *Addr = State.get(getAddr(), !CreateGather);
  Value *Mask = nullptr;
  if (VPValue *VPMask = getMask()) {
    Mask = State.get(VPMask);
    if (isReverse())
      Mask = createReverseEVL(Builder, Mask, EVL, "vp.reverse.mask");
  } else {
    Mask = Builder.CreateVectorSplat(State.VF, Builder.getTrue());
  }

  if (CreateGather) {
    NewLI =
        Builder.CreateIntrinsic(DataTy, Intrinsic::vp_gather, {Addr, Mask, EVL},
                                nullptr, "wide.masked.gather");
  } else {
    NewLI = Builder.CreateIntrinsic(DataTy, Intrinsic::vp_load,
                                    {Addr, Mask, EVL}, nullptr, "vp.op.load");
  }
  NewLI->addParamAttr(
      0, Attribute::getWithAlignment(NewLI->getContext(), Alignment));
  applyMetadata(*NewLI);
  Instruction *Res = NewLI;
  if (isReverse())
    Res = createReverseEVL(Builder, Res, EVL, "vp.reverse");
  State.set(this, Res);
}

InstructionCost VPWidenLoadEVLRecipe::computeCost(ElementCount VF,
                                                  VPCostContext &Ctx) const {
  if (!Consecutive || IsMasked)
    return VPWidenMemoryRecipe::computeCost(VF, Ctx);

  // We need to use the getMaskedMemoryOpCost() instead of getMemoryOpCost()
  // here because the EVL recipes using EVL to replace the tail mask. But in the
  // legacy model, it will always calculate the cost of mask.
  // TODO: Using getMemoryOpCost() instead of getMaskedMemoryOpCost when we
  // don't need to compare to the legacy cost model.
  Type *Ty = toVectorTy(getLoadStoreType(&Ingredient), VF);
  const Align Alignment = getLoadStoreAlignment(&Ingredient);
  unsigned AS = getLoadStoreAddressSpace(&Ingredient);
  InstructionCost Cost = Ctx.TTI.getMaskedMemoryOpCost(
      Instruction::Load, Ty, Alignment, AS, Ctx.CostKind);
  if (!Reverse)
    return Cost;

  return Cost + Ctx.TTI.getShuffleCost(
                    TargetTransformInfo::SK_Reverse, cast<VectorType>(Ty),
                    cast<VectorType>(Ty), {}, Ctx.CostKind, 0);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenLoadEVLRecipe::print(raw_ostream &O, const Twine &Indent,
                                 VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN ";
  printAsOperand(O, SlotTracker);
  O << " = vp.load ";
  printOperands(O, SlotTracker);
}
#endif

void VPWidenStoreRecipe::execute(VPTransformState &State) {
  VPValue *StoredVPValue = getStoredValue();
  bool CreateScatter = !isConsecutive();
  const Align Alignment = getLoadStoreAlignment(&Ingredient);

  auto &Builder = State.Builder;

  Value *Mask = nullptr;
  if (auto *VPMask = getMask()) {
    // Mask reversal is only needed for non-all-one (null) masks, as reverse
    // of a null all-one mask is a null mask.
    Mask = State.get(VPMask);
    if (isReverse())
      Mask = Builder.CreateVectorReverse(Mask, "reverse");
  }

  Value *StoredVal = State.get(StoredVPValue);
  if (isReverse()) {
    // If we store to reverse consecutive memory locations, then we need
    // to reverse the order of elements in the stored value.
    StoredVal = Builder.CreateVectorReverse(StoredVal, "reverse");
    // We don't want to update the value in the map as it might be used in
    // another expression. So don't call resetVectorValue(StoredVal).
  }
  Value *Addr = State.get(getAddr(), /*IsScalar*/ !CreateScatter);
  Instruction *NewSI = nullptr;
  if (CreateScatter)
    NewSI = Builder.CreateMaskedScatter(StoredVal, Addr, Alignment, Mask);
  else if (Mask)
    NewSI = Builder.CreateMaskedStore(StoredVal, Addr, Alignment, Mask);
  else
    NewSI = Builder.CreateAlignedStore(StoredVal, Addr, Alignment);
  applyMetadata(*NewSI);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenStoreRecipe::print(raw_ostream &O, const Twine &Indent,
                               VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN store ";
  printOperands(O, SlotTracker);
}
#endif

void VPWidenStoreEVLRecipe::execute(VPTransformState &State) {
  VPValue *StoredValue = getStoredValue();
  bool CreateScatter = !isConsecutive();
  const Align Alignment = getLoadStoreAlignment(&Ingredient);

  auto &Builder = State.Builder;

  CallInst *NewSI = nullptr;
  Value *StoredVal = State.get(StoredValue);
  Value *EVL = State.get(getEVL(), VPLane(0));
  if (isReverse())
    StoredVal = createReverseEVL(Builder, StoredVal, EVL, "vp.reverse");
  Value *Mask = nullptr;
  if (VPValue *VPMask = getMask()) {
    Mask = State.get(VPMask);
    if (isReverse())
      Mask = createReverseEVL(Builder, Mask, EVL, "vp.reverse.mask");
  } else {
    Mask = Builder.CreateVectorSplat(State.VF, Builder.getTrue());
  }
  Value *Addr = State.get(getAddr(), !CreateScatter);
  if (CreateScatter) {
    NewSI = Builder.CreateIntrinsic(Type::getVoidTy(EVL->getContext()),
                                    Intrinsic::vp_scatter,
                                    {StoredVal, Addr, Mask, EVL});
  } else {
    NewSI = Builder.CreateIntrinsic(Type::getVoidTy(EVL->getContext()),
                                    Intrinsic::vp_store,
                                    {StoredVal, Addr, Mask, EVL});
  }
  NewSI->addParamAttr(
      1, Attribute::getWithAlignment(NewSI->getContext(), Alignment));
  applyMetadata(*NewSI);
}

InstructionCost VPWidenStoreEVLRecipe::computeCost(ElementCount VF,
                                                   VPCostContext &Ctx) const {
  if (!Consecutive || IsMasked)
    return VPWidenMemoryRecipe::computeCost(VF, Ctx);

  // We need to use the getMaskedMemoryOpCost() instead of getMemoryOpCost()
  // here because the EVL recipes using EVL to replace the tail mask. But in the
  // legacy model, it will always calculate the cost of mask.
  // TODO: Using getMemoryOpCost() instead of getMaskedMemoryOpCost when we
  // don't need to compare to the legacy cost model.
  Type *Ty = toVectorTy(getLoadStoreType(&Ingredient), VF);
  const Align Alignment = getLoadStoreAlignment(&Ingredient);
  unsigned AS = getLoadStoreAddressSpace(&Ingredient);
  InstructionCost Cost = Ctx.TTI.getMaskedMemoryOpCost(
      Instruction::Store, Ty, Alignment, AS, Ctx.CostKind);
  if (!Reverse)
    return Cost;

  return Cost + Ctx.TTI.getShuffleCost(
                    TargetTransformInfo::SK_Reverse, cast<VectorType>(Ty),
                    cast<VectorType>(Ty), {}, Ctx.CostKind, 0);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenStoreEVLRecipe::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN vp.store ";
  printOperands(O, SlotTracker);
}
#endif

static Value *createBitOrPointerCast(IRBuilderBase &Builder, Value *V,
                                     VectorType *DstVTy, const DataLayout &DL) {
  // Verify that V is a vector type with same number of elements as DstVTy.
  auto VF = DstVTy->getElementCount();
  auto *SrcVecTy = cast<VectorType>(V->getType());
  assert(VF == SrcVecTy->getElementCount() && "Vector dimensions do not match");
  Type *SrcElemTy = SrcVecTy->getElementType();
  Type *DstElemTy = DstVTy->getElementType();
  assert((DL.getTypeSizeInBits(SrcElemTy) == DL.getTypeSizeInBits(DstElemTy)) &&
         "Vector elements must have same size");

  // Do a direct cast if element types are castable.
  if (CastInst::isBitOrNoopPointerCastable(SrcElemTy, DstElemTy, DL)) {
    return Builder.CreateBitOrPointerCast(V, DstVTy);
  }
  // V cannot be directly casted to desired vector type.
  // May happen when V is a floating point vector but DstVTy is a vector of
  // pointers or vice-versa. Handle this using a two-step bitcast using an
  // intermediate Integer type for the bitcast i.e. Ptr <-> Int <-> Float.
  assert((DstElemTy->isPointerTy() != SrcElemTy->isPointerTy()) &&
         "Only one type should be a pointer type");
  assert((DstElemTy->isFloatingPointTy() != SrcElemTy->isFloatingPointTy()) &&
         "Only one type should be a floating point type");
  Type *IntTy =
      IntegerType::getIntNTy(V->getContext(), DL.getTypeSizeInBits(SrcElemTy));
  auto *VecIntTy = VectorType::get(IntTy, VF);
  Value *CastVal = Builder.CreateBitOrPointerCast(V, VecIntTy);
  return Builder.CreateBitOrPointerCast(CastVal, DstVTy);
}

/// Return a vector containing interleaved elements from multiple
/// smaller input vectors.
static Value *interleaveVectors(IRBuilderBase &Builder, ArrayRef<Value *> Vals,
                                const Twine &Name) {
  unsigned Factor = Vals.size();
  assert(Factor > 1 && "Tried to interleave invalid number of vectors");

  VectorType *VecTy = cast<VectorType>(Vals[0]->getType());
#ifndef NDEBUG
  for (Value *Val : Vals)
    assert(Val->getType() == VecTy && "Tried to interleave mismatched types");
#endif

  // Scalable vectors cannot use arbitrary shufflevectors (only splats), so
  // must use intrinsics to interleave.
  if (VecTy->isScalableTy()) {
    assert(Factor <= 8 && "Unsupported interleave factor for scalable vectors");
    VectorType *InterleaveTy =
        VectorType::get(VecTy->getElementType(),
                        VecTy->getElementCount().multiplyCoefficientBy(Factor));
    return Builder.CreateIntrinsic(InterleaveTy,
                                   getInterleaveIntrinsicID(Factor), Vals,
                                   /*FMFSource=*/nullptr, Name);
  }

  // Fixed length. Start by concatenating all vectors into a wide vector.
  Value *WideVec = concatenateVectors(Builder, Vals);

  // Interleave the elements into the wide vector.
  const unsigned NumElts = VecTy->getElementCount().getFixedValue();
  return Builder.CreateShuffleVector(
      WideVec, createInterleaveMask(NumElts, Factor), Name);
}

// Try to vectorize the interleave group that \p Instr belongs to.
//
// E.g. Translate following interleaved load group (factor = 3):
//   for (i = 0; i < N; i+=3) {
//     R = Pic[i];             // Member of index 0
//     G = Pic[i+1];           // Member of index 1
//     B = Pic[i+2];           // Member of index 2
//     ... // do something to R, G, B
//   }
// To:
//   %wide.vec = load <12 x i32>                       ; Read 4 tuples of R,G,B
//   %R.vec = shuffle %wide.vec, poison, <0, 3, 6, 9>   ; R elements
//   %G.vec = shuffle %wide.vec, poison, <1, 4, 7, 10>  ; G elements
//   %B.vec = shuffle %wide.vec, poison, <2, 5, 8, 11>  ; B elements
//
// Or translate following interleaved store group (factor = 3):
//   for (i = 0; i < N; i+=3) {
//     ... do something to R, G, B
//     Pic[i]   = R;           // Member of index 0
//     Pic[i+1] = G;           // Member of index 1
//     Pic[i+2] = B;           // Member of index 2
//   }
// To:
//   %R_G.vec = shuffle %R.vec, %G.vec, <0, 1, 2, ..., 7>
//   %B_U.vec = shuffle %B.vec, poison, <0, 1, 2, 3, u, u, u, u>
//   %interleaved.vec = shuffle %R_G.vec, %B_U.vec,
//        <0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11>    ; Interleave R,G,B elements
//   store <12 x i32> %interleaved.vec              ; Write 4 tuples of R,G,B
void VPInterleaveRecipe::execute(VPTransformState &State) {
  assert(!State.Lane && "Interleave group being replicated.");
  const InterleaveGroup<Instruction> *Group = IG;
  Instruction *Instr = Group->getInsertPos();

  // Prepare for the vector type of the interleaved load/store.
  Type *ScalarTy = getLoadStoreType(Instr);
  unsigned InterleaveFactor = Group->getFactor();
  auto *VecTy = VectorType::get(ScalarTy, State.VF * InterleaveFactor);

  VPValue *BlockInMask = getMask();
  VPValue *Addr = getAddr();
  Value *ResAddr = State.get(Addr, VPLane(0));

  auto CreateGroupMask = [&BlockInMask, &State,
                          &InterleaveFactor](Value *MaskForGaps) -> Value * {
    if (State.VF.isScalable()) {
      assert(!MaskForGaps && "Interleaved groups with gaps are not supported.");
      assert(InterleaveFactor <= 8 &&
             "Unsupported deinterleave factor for scalable vectors");
      auto *ResBlockInMask = State.get(BlockInMask);
      SmallVector<Value *> Ops(InterleaveFactor, ResBlockInMask);
      return interleaveVectors(State.Builder, Ops, "interleaved.mask");
    }

    if (!BlockInMask)
      return MaskForGaps;

    Value *ResBlockInMask = State.get(BlockInMask);
    Value *ShuffledMask = State.Builder.CreateShuffleVector(
        ResBlockInMask,
        createReplicatedMask(InterleaveFactor, State.VF.getFixedValue()),
        "interleaved.mask");
    return MaskForGaps ? State.Builder.CreateBinOp(Instruction::And,
                                                   ShuffledMask, MaskForGaps)
                       : ShuffledMask;
  };

  const DataLayout &DL = Instr->getDataLayout();
  // Vectorize the interleaved load group.
  if (isa<LoadInst>(Instr)) {
    Value *MaskForGaps = nullptr;
    if (NeedsMaskForGaps) {
      MaskForGaps =
          createBitMaskForGaps(State.Builder, State.VF.getFixedValue(), *Group);
      assert(MaskForGaps && "Mask for Gaps is required but it is null");
    }

    Instruction *NewLoad;
    if (BlockInMask || MaskForGaps) {
      Value *GroupMask = CreateGroupMask(MaskForGaps);
      Value *PoisonVec = PoisonValue::get(VecTy);
      NewLoad = State.Builder.CreateMaskedLoad(VecTy, ResAddr,
                                               Group->getAlign(), GroupMask,
                                               PoisonVec, "wide.masked.vec");
    } else
      NewLoad = State.Builder.CreateAlignedLoad(VecTy, ResAddr,
                                                Group->getAlign(), "wide.vec");
    Group->addMetadata(NewLoad);

    ArrayRef<VPValue *> VPDefs = definedValues();
    if (VecTy->isScalableTy()) {
      // Scalable vectors cannot use arbitrary shufflevectors (only splats),
      // so must use intrinsics to deinterleave.
      assert(InterleaveFactor <= 8 &&
             "Unsupported deinterleave factor for scalable vectors");
      NewLoad = State.Builder.CreateIntrinsic(
          getDeinterleaveIntrinsicID(InterleaveFactor), NewLoad->getType(),
          NewLoad,
          /*FMFSource=*/nullptr, "strided.vec");
    }

    auto CreateStridedVector = [&InterleaveFactor, &State,
                                &NewLoad](unsigned Index) -> Value * {
      assert(Index < InterleaveFactor && "Illegal group index");
      if (State.VF.isScalable())
        return State.Builder.CreateExtractValue(NewLoad, Index);

      // For fixed length VF, use shuffle to extract the sub-vectors from the
      // wide load.
      auto StrideMask =
          createStrideMask(Index, InterleaveFactor, State.VF.getFixedValue());
      return State.Builder.CreateShuffleVector(NewLoad, StrideMask,
                                               "strided.vec");
    };

    for (unsigned I = 0, J = 0; I < InterleaveFactor; ++I) {
      Instruction *Member = Group->getMember(I);

      // Skip the gaps in the group.
      if (!Member)
        continue;

      Value *StridedVec = CreateStridedVector(I);

      // If this member has different type, cast the result type.
      if (Member->getType() != ScalarTy) {
        VectorType *OtherVTy = VectorType::get(Member->getType(), State.VF);
        StridedVec =
            createBitOrPointerCast(State.Builder, StridedVec, OtherVTy, DL);
      }

      if (Group->isReverse())
        StridedVec = State.Builder.CreateVectorReverse(StridedVec, "reverse");

      State.set(VPDefs[J], StridedVec);
      ++J;
    }
    return;
  }

  // The sub vector type for current instruction.
  auto *SubVT = VectorType::get(ScalarTy, State.VF);

  // Vectorize the interleaved store group.
  Value *MaskForGaps =
      createBitMaskForGaps(State.Builder, State.VF.getKnownMinValue(), *Group);
  assert((!MaskForGaps || !State.VF.isScalable()) &&
         "masking gaps for scalable vectors is not yet supported.");
  ArrayRef<VPValue *> StoredValues = getStoredValues();
  // Collect the stored vector from each member.
  SmallVector<Value *, 4> StoredVecs;
  unsigned StoredIdx = 0;
  for (unsigned i = 0; i < InterleaveFactor; i++) {
    assert((Group->getMember(i) || MaskForGaps) &&
           "Fail to get a member from an interleaved store group");
    Instruction *Member = Group->getMember(i);

    // Skip the gaps in the group.
    if (!Member) {
      Value *Undef = PoisonValue::get(SubVT);
      StoredVecs.push_back(Undef);
      continue;
    }

    Value *StoredVec = State.get(StoredValues[StoredIdx]);
    ++StoredIdx;

    if (Group->isReverse())
      StoredVec = State.Builder.CreateVectorReverse(StoredVec, "reverse");

    // If this member has different type, cast it to a unified type.

    if (StoredVec->getType() != SubVT)
      StoredVec = createBitOrPointerCast(State.Builder, StoredVec, SubVT, DL);

    StoredVecs.push_back(StoredVec);
  }

  // Interleave all the smaller vectors into one wider vector.
  Value *IVec = interleaveVectors(State.Builder, StoredVecs, "interleaved.vec");
  Instruction *NewStoreInstr;
  if (BlockInMask || MaskForGaps) {
    Value *GroupMask = CreateGroupMask(MaskForGaps);
    NewStoreInstr = State.Builder.CreateMaskedStore(
        IVec, ResAddr, Group->getAlign(), GroupMask);
  } else
    NewStoreInstr =
        State.Builder.CreateAlignedStore(IVec, ResAddr, Group->getAlign());

  Group->addMetadata(NewStoreInstr);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPInterleaveRecipe::print(raw_ostream &O, const Twine &Indent,
                               VPSlotTracker &SlotTracker) const {
  O << Indent << "INTERLEAVE-GROUP with factor " << IG->getFactor() << " at ";
  IG->getInsertPos()->printAsOperand(O, false);
  O << ", ";
  getAddr()->printAsOperand(O, SlotTracker);
  VPValue *Mask = getMask();
  if (Mask) {
    O << ", ";
    Mask->printAsOperand(O, SlotTracker);
  }

  unsigned OpIdx = 0;
  for (unsigned i = 0; i < IG->getFactor(); ++i) {
    if (!IG->getMember(i))
      continue;
    if (getNumStoreOperands() > 0) {
      O << "\n" << Indent << "  store ";
      getOperand(1 + OpIdx)->printAsOperand(O, SlotTracker);
      O << " to index " << i;
    } else {
      O << "\n" << Indent << "  ";
      getVPValue(OpIdx)->printAsOperand(O, SlotTracker);
      O << " = load from index " << i;
    }
    ++OpIdx;
  }
}
#endif

InstructionCost VPInterleaveRecipe::computeCost(ElementCount VF,
                                                VPCostContext &Ctx) const {
  Instruction *InsertPos = getInsertPos();
  // Find the VPValue index of the interleave group. We need to skip gaps.
  unsigned InsertPosIdx = 0;
  for (unsigned Idx = 0; IG->getFactor(); ++Idx)
    if (auto *Member = IG->getMember(Idx)) {
      if (Member == InsertPos)
        break;
      InsertPosIdx++;
    }
  Type *ValTy = Ctx.Types.inferScalarType(
      getNumDefinedValues() > 0 ? getVPValue(InsertPosIdx)
                                : getStoredValues()[InsertPosIdx]);
  auto *VectorTy = cast<VectorType>(toVectorTy(ValTy, VF));
  unsigned AS = getLoadStoreAddressSpace(InsertPos);

  unsigned InterleaveFactor = IG->getFactor();
  auto *WideVecTy = VectorType::get(ValTy, VF * InterleaveFactor);

  // Holds the indices of existing members in the interleaved group.
  SmallVector<unsigned, 4> Indices;
  for (unsigned IF = 0; IF < InterleaveFactor; IF++)
    if (IG->getMember(IF))
      Indices.push_back(IF);

  // Calculate the cost of the whole interleaved group.
  InstructionCost Cost = Ctx.TTI.getInterleavedMemoryOpCost(
      InsertPos->getOpcode(), WideVecTy, IG->getFactor(), Indices,
      IG->getAlign(), AS, Ctx.CostKind, getMask(), NeedsMaskForGaps);

  if (!IG->isReverse())
    return Cost;

  return Cost + IG->getNumMembers() *
                    Ctx.TTI.getShuffleCost(TargetTransformInfo::SK_Reverse,
                                           VectorTy, VectorTy, {}, Ctx.CostKind,
                                           0);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPCanonicalIVPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                   VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = CANONICAL-INDUCTION ";
  printOperands(O, SlotTracker);
}
#endif

bool VPWidenPointerInductionRecipe::onlyScalarsGenerated(bool IsScalable) {
  return IsScalarAfterVectorization &&
         (!IsScalable || vputils::onlyFirstLaneUsed(this));
}

void VPWidenPointerInductionRecipe::execute(VPTransformState &State) {
  assert(getInductionDescriptor().getKind() ==
             InductionDescriptor::IK_PtrInduction &&
         "Not a pointer induction according to InductionDescriptor!");
  assert(State.TypeAnalysis.inferScalarType(this)->isPointerTy() &&
         "Unexpected type.");
  assert(!onlyScalarsGenerated(State.VF.isScalable()) &&
         "Recipe should have been replaced");

  unsigned CurrentPart = getUnrollPart(*this);

  // Build a pointer phi
  Value *ScalarStartValue = getStartValue()->getLiveInIRValue();
  Type *ScStValueType = ScalarStartValue->getType();

  BasicBlock *VectorPH =
      State.CFG.VPBB2IRBB.at(getParent()->getCFGPredecessor(0));
  PHINode *NewPointerPhi = nullptr;
  if (CurrentPart == 0) {
    IRBuilder<>::InsertPointGuard Guard(State.Builder);
    if (State.Builder.GetInsertPoint() !=
        State.Builder.GetInsertBlock()->getFirstNonPHIIt())
      State.Builder.SetInsertPoint(
          State.Builder.GetInsertBlock()->getFirstNonPHIIt());
    NewPointerPhi = State.Builder.CreatePHI(ScStValueType, 2, "pointer.phi");
    NewPointerPhi->addIncoming(ScalarStartValue, VectorPH);
    NewPointerPhi->setDebugLoc(getDebugLoc());
  } else {
    // The recipe has been unrolled. In that case, fetch the single pointer phi
    // shared among all unrolled parts of the recipe.
    auto *GEP =
        cast<GetElementPtrInst>(State.get(getFirstUnrolledPartOperand()));
    NewPointerPhi = cast<PHINode>(GEP->getPointerOperand());
  }

  // A pointer induction, performed by using a gep
  BasicBlock::iterator InductionLoc = State.Builder.GetInsertPoint();
  Value *ScalarStepValue = State.get(getStepValue(), VPLane(0));
  Type *PhiType = State.TypeAnalysis.inferScalarType(getStepValue());
  Value *RuntimeVF = getRuntimeVF(State.Builder, PhiType, State.VF);
  // Add induction update using an incorrect block temporarily. The phi node
  // will be fixed after VPlan execution. Note that at this point the latch
  // block cannot be used, as it does not exist yet.
  // TODO: Model increment value in VPlan, by turning the recipe into a
  // multi-def and a subclass of VPHeaderPHIRecipe.
  if (CurrentPart == 0) {
    // The recipe represents the first part of the pointer induction. Create the
    // GEP to increment the phi across all unrolled parts.
    Value *NumUnrolledElems = State.get(getOperand(2), true);

    Value *InductionGEP = GetElementPtrInst::Create(
        State.Builder.getInt8Ty(), NewPointerPhi,
        State.Builder.CreateMul(
            ScalarStepValue,
            State.Builder.CreateTrunc(NumUnrolledElems, PhiType)),
        "ptr.ind", InductionLoc);

    NewPointerPhi->addIncoming(InductionGEP, VectorPH);
  }

  // Create actual address geps that use the pointer phi as base and a
  // vectorized version of the step value (<step*0, ..., step*N>) as offset.
  Type *VecPhiType = VectorType::get(PhiType, State.VF);
  Value *StartOffsetScalar = State.Builder.CreateMul(
      RuntimeVF, ConstantInt::get(PhiType, CurrentPart));
  Value *StartOffset =
      State.Builder.CreateVectorSplat(State.VF, StartOffsetScalar);
  // Create a vector of consecutive numbers from zero to VF.
  StartOffset = State.Builder.CreateAdd(
      StartOffset, State.Builder.CreateStepVector(VecPhiType));

  assert(ScalarStepValue == State.get(getOperand(1), VPLane(0)) &&
         "scalar step must be the same across all parts");
  Value *GEP = State.Builder.CreateGEP(
      State.Builder.getInt8Ty(), NewPointerPhi,
      State.Builder.CreateMul(StartOffset, State.Builder.CreateVectorSplat(
                                               State.VF, ScalarStepValue)),
      "vector.gep");
  State.set(this, GEP);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenPointerInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          VPSlotTracker &SlotTracker) const {
  assert((getNumOperands() == 3 || getNumOperands() == 5) &&
         "unexpected number of operands");
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = WIDEN-POINTER-INDUCTION ";
  getStartValue()->printAsOperand(O, SlotTracker);
  O << ", ";
  getStepValue()->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(2)->printAsOperand(O, SlotTracker);
  if (getNumOperands() == 5) {
    O << ", ";
    getOperand(3)->printAsOperand(O, SlotTracker);
    O << ", ";
    getOperand(4)->printAsOperand(O, SlotTracker);
  }
}
#endif

void VPExpandSCEVRecipe::execute(VPTransformState &State) {
  assert(!State.Lane && "cannot be used in per-lane");
  const DataLayout &DL = SE.getDataLayout();
  SCEVExpander Exp(SE, DL, "induction", /*PreserveLCSSA=*/true);
  Value *Res = Exp.expandCodeFor(Expr, Expr->getType(),
                                 &*State.Builder.GetInsertPoint());
  State.set(this, Res, VPLane(0));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPExpandSCEVRecipe::print(raw_ostream &O, const Twine &Indent,
                               VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = EXPAND SCEV " << *Expr;
}
#endif

void VPWidenCanonicalIVRecipe::execute(VPTransformState &State) {
  Value *CanonicalIV = State.get(getOperand(0), /*IsScalar*/ true);
  Type *STy = CanonicalIV->getType();
  IRBuilder<> Builder(State.CFG.PrevBB->getTerminator());
  ElementCount VF = State.VF;
  Value *VStart = VF.isScalar()
                      ? CanonicalIV
                      : Builder.CreateVectorSplat(VF, CanonicalIV, "broadcast");
  Value *VStep = createStepForVF(Builder, STy, VF, getUnrollPart(*this));
  if (VF.isVector()) {
    VStep = Builder.CreateVectorSplat(VF, VStep);
    VStep =
        Builder.CreateAdd(VStep, Builder.CreateStepVector(VStep->getType()));
  }
  Value *CanonicalVectorIV = Builder.CreateAdd(VStart, VStep, "vec.iv");
  State.set(this, CanonicalVectorIV);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenCanonicalIVRecipe::print(raw_ostream &O, const Twine &Indent,
                                     VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = WIDEN-CANONICAL-INDUCTION ";
  printOperands(O, SlotTracker);
}
#endif

void VPFirstOrderRecurrencePHIRecipe::execute(VPTransformState &State) {
  auto &Builder = State.Builder;
  // Create a vector from the initial value.
  auto *VectorInit = getStartValue()->getLiveInIRValue();

  Type *VecTy = State.VF.isScalar()
                    ? VectorInit->getType()
                    : VectorType::get(VectorInit->getType(), State.VF);

  BasicBlock *VectorPH =
      State.CFG.VPBB2IRBB.at(getParent()->getCFGPredecessor(0));
  if (State.VF.isVector()) {
    auto *IdxTy = Builder.getInt32Ty();
    auto *One = ConstantInt::get(IdxTy, 1);
    IRBuilder<>::InsertPointGuard Guard(Builder);
    Builder.SetInsertPoint(VectorPH->getTerminator());
    auto *RuntimeVF = getRuntimeVF(Builder, IdxTy, State.VF);
    auto *LastIdx = Builder.CreateSub(RuntimeVF, One);
    VectorInit = Builder.CreateInsertElement(
        PoisonValue::get(VecTy), VectorInit, LastIdx, "vector.recur.init");
  }

  // Create a phi node for the new recurrence.
  PHINode *Phi = PHINode::Create(VecTy, 2, "vector.recur");
  Phi->insertBefore(State.CFG.PrevBB->getFirstInsertionPt());
  Phi->addIncoming(VectorInit, VectorPH);
  State.set(this, Phi);
}

InstructionCost
VPFirstOrderRecurrencePHIRecipe::computeCost(ElementCount VF,
                                             VPCostContext &Ctx) const {
  if (VF.isScalar())
    return Ctx.TTI.getCFInstrCost(Instruction::PHI, Ctx.CostKind);

  return 0;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPFirstOrderRecurrencePHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                            VPSlotTracker &SlotTracker) const {
  O << Indent << "FIRST-ORDER-RECURRENCE-PHI ";
  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

void VPReductionPHIRecipe::execute(VPTransformState &State) {
  // Reductions do not have to start at zero. They can start with
  // any loop invariant values.
  VPValue *StartVPV = getStartValue();

  // In order to support recurrences we need to be able to vectorize Phi nodes.
  // Phi nodes have cycles, so we need to vectorize them in two stages. This is
  // stage #1: We create a new vector PHI node with no incoming edges. We'll use
  // this value when we vectorize all of the instructions that use the PHI.
  BasicBlock *VectorPH =
      State.CFG.VPBB2IRBB.at(getParent()->getCFGPredecessor(0));
  bool ScalarPHI = State.VF.isScalar() || IsInLoop;
  Value *StartV = State.get(StartVPV, ScalarPHI);
  Type *VecTy = StartV->getType();

  BasicBlock *HeaderBB = State.CFG.PrevBB;
  assert(State.CurrentParentLoop->getHeader() == HeaderBB &&
         "recipe must be in the vector loop header");
  auto *Phi = PHINode::Create(VecTy, 2, "vec.phi");
  Phi->insertBefore(HeaderBB->getFirstInsertionPt());
  State.set(this, Phi, IsInLoop);

  Phi->addIncoming(StartV, VectorPH);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPReductionPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                 VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-REDUCTION-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
  if (VFScaleFactor != 1)
    O << " (VF scaled by 1/" << VFScaleFactor << ")";
}
#endif

void VPWidenPHIRecipe::execute(VPTransformState &State) {
  Value *Op0 = State.get(getOperand(0));
  Type *VecTy = Op0->getType();
  Instruction *VecPhi = State.Builder.CreatePHI(VecTy, 2, Name);
  // Manually move it with the other PHIs in case PHI recipes above this one
  // also inserted non-phi instructions.
  // TODO: Remove once VPWidenPointerInductionRecipe is also expanded in
  // convertToConcreteRecipes.
  VecPhi->moveBefore(State.Builder.GetInsertBlock()->getFirstNonPHIIt());
  State.set(this, VecPhi);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                             VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printPhiOperands(O, SlotTracker);
}
#endif

// TODO: It would be good to use the existing VPWidenPHIRecipe instead and
// remove VPActiveLaneMaskPHIRecipe.
void VPActiveLaneMaskPHIRecipe::execute(VPTransformState &State) {
  BasicBlock *VectorPH =
      State.CFG.VPBB2IRBB.at(getParent()->getCFGPredecessor(0));
  Value *StartMask = State.get(getOperand(0));
  PHINode *Phi =
      State.Builder.CreatePHI(StartMask->getType(), 2, "active.lane.mask");
  Phi->addIncoming(StartMask, VectorPH);
  State.set(this, Phi);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPActiveLaneMaskPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                      VPSlotTracker &SlotTracker) const {
  O << Indent << "ACTIVE-LANE-MASK-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPEVLBasedIVPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent << "EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif
