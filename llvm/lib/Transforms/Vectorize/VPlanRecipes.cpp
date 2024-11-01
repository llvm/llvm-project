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

#include "VPlan.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <cassert>

using namespace llvm;

using VectorParts = SmallVector<Value *, 2>;

namespace llvm {
extern cl::opt<bool> EnableVPlanNativePath;
}

#define LV_NAME "loop-vectorize"
#define DEBUG_TYPE LV_NAME

bool VPRecipeBase::mayWriteToMemory() const {
  switch (getVPDefID()) {
  case VPWidenMemoryInstructionSC: {
    return cast<VPWidenMemoryInstructionRecipe>(this)->isStore();
  }
  case VPReplicateSC:
  case VPWidenCallSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayWriteToMemory();
  case VPBranchOnMaskSC:
  case VPScalarIVStepsSC:
  case VPPredInstPHISC:
    return false;
  case VPBlendSC:
  case VPReductionSC:
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
  case VPWidenMemoryInstructionSC: {
    return !cast<VPWidenMemoryInstructionRecipe>(this)->isStore();
  }
  case VPReplicateSC:
  case VPWidenCallSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayReadFromMemory();
  case VPBranchOnMaskSC:
  case VPScalarIVStepsSC:
  case VPPredInstPHISC:
    return false;
  case VPBlendSC:
  case VPReductionSC:
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
  case VPDerivedIVSC:
  case VPPredInstPHISC:
    return false;
  case VPInstructionSC:
    switch (cast<VPInstruction>(this)->getOpcode()) {
    case Instruction::ICmp:
    case VPInstruction::Not:
    case VPInstruction::CalculateTripCountMinusVF:
    case VPInstruction::CanonicalIVIncrement:
    case VPInstruction::CanonicalIVIncrementForPart:
      return false;
    default:
      return true;
    }
  case VPWidenCallSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayHaveSideEffects();
  case VPBlendSC:
  case VPReductionSC:
  case VPScalarIVStepsSC:
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
  case VPWidenMemoryInstructionSC:
    assert(cast<VPWidenMemoryInstructionRecipe>(this)
                   ->getIngredient()
                   .mayHaveSideEffects() == mayWriteToMemory() &&
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

void VPLiveOut::fixPhi(VPlan &Plan, VPTransformState &State) {
  auto Lane = VPLane::getLastLaneForVF(State.VF);
  VPValue *ExitValue = getOperand(0);
  if (vputils::isUniformAfterVectorization(ExitValue))
    Lane = VPLane::getFirstLane();
  VPBasicBlock *MiddleVPBB =
      cast<VPBasicBlock>(Plan.getVectorLoopRegion()->getSingleSuccessor());
  assert(MiddleVPBB->getNumSuccessors() == 0 &&
         "the middle block must not have any successors");
  BasicBlock *MiddleBB = State.CFG.VPBB2IRBB[MiddleVPBB];
  Phi->addIncoming(State.get(ExitValue, VPIteration(State.UF - 1, Lane)),
                   MiddleBB);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPLiveOut::print(raw_ostream &O, VPSlotTracker &SlotTracker) const {
  O << "Live-out ";
  getPhi()->printAsOperand(O);
  O << " = ";
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << "\n";
}
#endif

void VPRecipeBase::insertBefore(VPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  Parent = InsertPos->getParent();
  Parent->getRecipeList().insert(InsertPos->getIterator(), this);
}

void VPRecipeBase::insertBefore(VPBasicBlock &BB,
                                iplist<VPRecipeBase>::iterator I) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(I == BB.end() || I->getParent() == &BB);
  Parent = &BB;
  BB.getRecipeList().insert(I, this);
}

void VPRecipeBase::insertAfter(VPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  Parent = InsertPos->getParent();
  Parent->getRecipeList().insertAfter(InsertPos->getIterator(), this);
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

FastMathFlags VPRecipeWithIRFlags::getFastMathFlags() const {
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

VPInstruction::VPInstruction(unsigned Opcode, CmpInst::Predicate Pred,
                             VPValue *A, VPValue *B, DebugLoc DL,
                             const Twine &Name)
    : VPRecipeWithIRFlags(VPDef::VPInstructionSC, ArrayRef<VPValue *>({A, B}),
                          Pred, DL),
      VPValue(this), Opcode(Opcode), Name(Name.str()) {
  assert(Opcode == Instruction::ICmp &&
         "only ICmp predicates supported at the moment");
}

VPInstruction::VPInstruction(unsigned Opcode,
                             std::initializer_list<VPValue *> Operands,
                             FastMathFlags FMFs, DebugLoc DL, const Twine &Name)
    : VPRecipeWithIRFlags(VPDef::VPInstructionSC, Operands, FMFs, DL),
      VPValue(this), Opcode(Opcode), Name(Name.str()) {
  // Make sure the VPInstruction is a floating-point operation.
  assert(isFPMathOp() && "this op can't take fast-math flags");
}

Value *VPInstruction::generateInstruction(VPTransformState &State,
                                          unsigned Part) {
  IRBuilderBase &Builder = State.Builder;
  Builder.SetCurrentDebugLocation(getDebugLoc());

  if (Instruction::isBinaryOp(getOpcode())) {
    Value *A = State.get(getOperand(0), Part);
    Value *B = State.get(getOperand(1), Part);
    return Builder.CreateBinOp((Instruction::BinaryOps)getOpcode(), A, B, Name);
  }

  switch (getOpcode()) {
  case VPInstruction::Not: {
    Value *A = State.get(getOperand(0), Part);
    return Builder.CreateNot(A, Name);
  }
  case Instruction::ICmp: {
    Value *A = State.get(getOperand(0), Part);
    Value *B = State.get(getOperand(1), Part);
    return Builder.CreateCmp(getPredicate(), A, B, Name);
  }
  case Instruction::Select: {
    Value *Cond = State.get(getOperand(0), Part);
    Value *Op1 = State.get(getOperand(1), Part);
    Value *Op2 = State.get(getOperand(2), Part);
    return Builder.CreateSelect(Cond, Op1, Op2, Name);
  }
  case VPInstruction::ActiveLaneMask: {
    // Get first lane of vector induction variable.
    Value *VIVElem0 = State.get(getOperand(0), VPIteration(Part, 0));
    // Get the original loop tripcount.
    Value *ScalarTC = State.get(getOperand(1), VPIteration(Part, 0));

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

    // For the first part, use the recurrence phi (v1), otherwise v2.
    auto *V1 = State.get(getOperand(0), 0);
    Value *PartMinus1 = Part == 0 ? V1 : State.get(getOperand(1), Part - 1);
    if (!PartMinus1->getType()->isVectorTy())
      return PartMinus1;
    Value *V2 = State.get(getOperand(1), Part);
    return Builder.CreateVectorSplice(PartMinus1, V2, -1, Name);
  }
  case VPInstruction::CalculateTripCountMinusVF: {
    Value *ScalarTC = State.get(getOperand(0), {0, 0});
    Value *Step =
        createStepForVF(Builder, ScalarTC->getType(), State.VF, State.UF);
    Value *Sub = Builder.CreateSub(ScalarTC, Step);
    Value *Cmp = Builder.CreateICmp(CmpInst::Predicate::ICMP_UGT, ScalarTC, Step);
    Value *Zero = ConstantInt::get(ScalarTC->getType(), 0);
    return Builder.CreateSelect(Cmp, Sub, Zero);
  }
  case VPInstruction::CanonicalIVIncrement: {
    if (Part == 0) {
      auto *Phi = State.get(getOperand(0), 0);
      // The loop step is equal to the vectorization factor (num of SIMD
      // elements) times the unroll factor (num of SIMD instructions).
      Value *Step =
          createStepForVF(Builder, Phi->getType(), State.VF, State.UF);
      return Builder.CreateAdd(Phi, Step, Name, hasNoUnsignedWrap(),
                               hasNoSignedWrap());
    }
    return State.get(this, 0);
  }

  case VPInstruction::CanonicalIVIncrementForPart: {
    auto *IV = State.get(getOperand(0), VPIteration(0, 0));
    if (Part == 0)
      return IV;

    // The canonical IV is incremented by the vectorization factor (num of SIMD
    // elements) times the unroll part.
    Value *Step = createStepForVF(Builder, IV->getType(), State.VF, Part);
    return Builder.CreateAdd(IV, Step, Name, hasNoUnsignedWrap(),
                             hasNoSignedWrap());
  }
  case VPInstruction::BranchOnCond: {
    if (Part != 0)
      return nullptr;

    Value *Cond = State.get(getOperand(0), VPIteration(Part, 0));
    VPRegionBlock *ParentRegion = getParent()->getParent();
    VPBasicBlock *Header = ParentRegion->getEntryBasicBlock();

    // Replace the temporary unreachable terminator with a new conditional
    // branch, hooking it up to backward destination for exiting blocks now and
    // to forward destination(s) later when they are created.
    BranchInst *CondBr =
        Builder.CreateCondBr(Cond, Builder.GetInsertBlock(), nullptr);

    if (getParent()->isExiting())
      CondBr->setSuccessor(1, State.CFG.VPBB2IRBB[Header]);

    CondBr->setSuccessor(0, nullptr);
    Builder.GetInsertBlock()->getTerminator()->eraseFromParent();
    return CondBr;
  }
  case VPInstruction::BranchOnCount: {
    if (Part != 0)
      return nullptr;
    // First create the compare.
    Value *IV = State.get(getOperand(0), Part);
    Value *TC = State.get(getOperand(1), Part);
    Value *Cond = Builder.CreateICmpEQ(IV, TC);

    // Now create the branch.
    auto *Plan = getParent()->getPlan();
    VPRegionBlock *TopRegion = Plan->getVectorLoopRegion();
    VPBasicBlock *Header = TopRegion->getEntry()->getEntryBasicBlock();

    // Replace the temporary unreachable terminator with a new conditional
    // branch, hooking it up to backward destination (the header) now and to the
    // forward destination (the exit/middle block) later when it is created.
    // Note that CreateCondBr expects a valid BB as first argument, so we need
    // to set it to nullptr later.
    BranchInst *CondBr = Builder.CreateCondBr(Cond, Builder.GetInsertBlock(),
                                              State.CFG.VPBB2IRBB[Header]);
    CondBr->setSuccessor(0, nullptr);
    Builder.GetInsertBlock()->getTerminator()->eraseFromParent();
    return CondBr;
  }
  default:
    llvm_unreachable("Unsupported opcode for instruction");
  }
}

#if !defined(NDEBUG)
bool VPInstruction::isFPMathOp() const {
  // Inspired by FPMathOperator::classof. Notable differences are that we don't
  // support Call, PHI and Select opcodes here yet.
  return Opcode == Instruction::FAdd || Opcode == Instruction::FMul ||
         Opcode == Instruction::FNeg || Opcode == Instruction::FSub ||
         Opcode == Instruction::FDiv || Opcode == Instruction::FRem ||
         Opcode == Instruction::FCmp;
}
#endif

void VPInstruction::execute(VPTransformState &State) {
  assert(!State.Instance && "VPInstruction executing an Instance");
  IRBuilderBase::FastMathFlagGuard FMFGuard(State.Builder);
  assert(hasFastMathFlags() == isFPMathOp() &&
         "Recipe not a FPMathOp but has fast-math flags?");
  if (hasFastMathFlags())
    State.Builder.setFastMathFlags(getFastMathFlags());
  for (unsigned Part = 0; Part < State.UF; ++Part) {
    Value *GeneratedValue = generateInstruction(State, Part);
    if (!hasResult())
      continue;
    assert(GeneratedValue && "generateInstruction must produce a value");
    State.set(this, GeneratedValue, Part);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPInstruction::dump() const {
  VPSlotTracker SlotTracker(getParent()->getPlan());
  print(dbgs(), "", SlotTracker);
}

void VPInstruction::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";

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
  case VPInstruction::FirstOrderRecurrenceSplice:
    O << "first-order splice";
    break;
  case VPInstruction::CanonicalIVIncrement:
    O << "VF * UF +";
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

void VPWidenCallRecipe::execute(VPTransformState &State) {
  assert(State.VF.isVector() && "not widening");
  auto &CI = *cast<CallInst>(getUnderlyingInstr());
  assert(!isa<DbgInfoIntrinsic>(CI) &&
         "DbgInfoIntrinsic should have been dropped during VPlan construction");
  State.setDebugLocFrom(CI.getDebugLoc());

  for (unsigned Part = 0; Part < State.UF; ++Part) {
    SmallVector<Type *, 2> TysForDecl;
    // Add return type if intrinsic is overloaded on it.
    if (isVectorIntrinsicWithOverloadTypeAtArg(VectorIntrinsicID, -1)) {
      TysForDecl.push_back(
          VectorType::get(CI.getType()->getScalarType(), State.VF));
    }
    SmallVector<Value *, 4> Args;
    for (const auto &I : enumerate(operands())) {
      // Some intrinsics have a scalar argument - don't replace it with a
      // vector.
      Value *Arg;
      if (VectorIntrinsicID == Intrinsic::not_intrinsic ||
          !isVectorIntrinsicWithScalarOpAtArg(VectorIntrinsicID, I.index()))
        Arg = State.get(I.value(), Part);
      else
        Arg = State.get(I.value(), VPIteration(0, 0));
      if (isVectorIntrinsicWithOverloadTypeAtArg(VectorIntrinsicID, I.index()))
        TysForDecl.push_back(Arg->getType());
      Args.push_back(Arg);
    }

    Function *VectorF;
    if (VectorIntrinsicID != Intrinsic::not_intrinsic) {
      // Use vector version of the intrinsic.
      Module *M = State.Builder.GetInsertBlock()->getModule();
      VectorF = Intrinsic::getDeclaration(M, VectorIntrinsicID, TysForDecl);
      assert(VectorF && "Can't retrieve vector intrinsic.");
    } else {
#ifndef NDEBUG
      assert(Variant != nullptr && "Can't create vector function.");
#endif
      VectorF = Variant;
    }

    SmallVector<OperandBundleDef, 1> OpBundles;
    CI.getOperandBundlesAsDefs(OpBundles);
    CallInst *V = State.Builder.CreateCall(VectorF, Args, OpBundles);

    if (isa<FPMathOperator>(V))
      V->copyFastMathFlags(&CI);

    State.set(this, V, Part);
    State.addMetadata(V, &CI);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenCallRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-CALL ";

  auto *CI = cast<CallInst>(getUnderlyingInstr());
  if (CI->getType()->isVoidTy())
    O << "void ";
  else {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  O << "call @" << CI->getCalledFunction()->getName() << "(";
  printOperands(O, SlotTracker);
  O << ")";

  if (VectorIntrinsicID)
    O << " (using vector intrinsic)";
  else {
    O << " (using library function";
    if (Variant->hasName())
      O << ": " << Variant->getName();
    O << ")";
  }
}

void VPWidenSelectRecipe::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-SELECT ";
  printAsOperand(O, SlotTracker);
  O << " = select ";
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(1)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(2)->printAsOperand(O, SlotTracker);
  O << (isInvariantCond() ? " (condition is loop invariant)" : "");
}
#endif

void VPWidenSelectRecipe::execute(VPTransformState &State) {
  State.setDebugLocFrom(getDebugLoc());

  // The condition can be loop invariant but still defined inside the
  // loop. This means that we can't just use the original 'cond' value.
  // We have to take the 'vectorized' value and pick the first lane.
  // Instcombine will make this a no-op.
  auto *InvarCond =
      isInvariantCond() ? State.get(getCond(), VPIteration(0, 0)) : nullptr;

  for (unsigned Part = 0; Part < State.UF; ++Part) {
    Value *Cond = InvarCond ? InvarCond : State.get(getCond(), Part);
    Value *Op0 = State.get(getOperand(1), Part);
    Value *Op1 = State.get(getOperand(2), Part);
    Value *Sel = State.Builder.CreateSelect(Cond, Op0, Op1);
    State.set(this, Sel, Part);
    State.addMetadata(Sel, dyn_cast_or_null<Instruction>(getUnderlyingValue()));
  }
}

VPRecipeWithIRFlags::FastMathFlagsTy::FastMathFlagsTy(
    const FastMathFlags &FMF) {
  AllowReassoc = FMF.allowReassoc();
  NoNaNs = FMF.noNaNs();
  NoInfs = FMF.noInfs();
  NoSignedZeros = FMF.noSignedZeros();
  AllowReciprocal = FMF.allowReciprocal();
  AllowContract = FMF.allowContract();
  ApproxFunc = FMF.approxFunc();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPRecipeWithIRFlags::printFlags(raw_ostream &O) const {
  switch (OpType) {
  case OperationType::Cmp:
    O << " " << CmpInst::getPredicateName(getPredicate());
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
  case OperationType::FPMathOp:
    getFastMathFlags().print(O);
    break;
  case OperationType::GEPOp:
    if (GEPFlags.IsInBounds)
      O << " inbounds";
    break;
  case OperationType::Other:
    break;
  }
  if (getNumOperands() > 0)
    O << " ";
}
#endif

void VPWidenRecipe::execute(VPTransformState &State) {
  State.setDebugLocFrom(getDebugLoc());
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
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      SmallVector<Value *, 2> Ops;
      for (VPValue *VPOp : operands())
        Ops.push_back(State.get(VPOp, Part));

      Value *V = Builder.CreateNAryOp(Opcode, Ops);

      if (auto *VecOp = dyn_cast<Instruction>(V))
        setFlags(VecOp);

      // Use this vector value for all users of the original instruction.
      State.set(this, V, Part);
      State.addMetadata(V, dyn_cast_or_null<Instruction>(getUnderlyingValue()));
    }

    break;
  }
  case Instruction::Freeze: {
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Value *Op = State.get(getOperand(0), Part);

      Value *Freeze = Builder.CreateFreeze(Op);
      State.set(this, Freeze, Part);
    }
    break;
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    // Widen compares. Generate vector compares.
    bool FCmp = Opcode == Instruction::FCmp;
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Value *A = State.get(getOperand(0), Part);
      Value *B = State.get(getOperand(1), Part);
      Value *C = nullptr;
      if (FCmp) {
        // Propagate fast math flags.
        IRBuilder<>::FastMathFlagGuard FMFG(Builder);
        if (auto *I = dyn_cast_or_null<Instruction>(getUnderlyingValue()))
          Builder.setFastMathFlags(I->getFastMathFlags());
        C = Builder.CreateFCmp(getPredicate(), A, B);
      } else {
        C = Builder.CreateICmp(getPredicate(), A, B);
      }
      State.set(this, C, Part);
      State.addMetadata(C, dyn_cast_or_null<Instruction>(getUnderlyingValue()));
    }

    break;
  }
  default:
    // This instruction is not vectorized by simple widening.
    LLVM_DEBUG(dbgs() << "LV: Found an unhandled opcode : "
                      << Instruction::getOpcodeName(Opcode));
    llvm_unreachable("Unhandled instruction!");
  } // end of switch.
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
  State.setDebugLocFrom(getDebugLoc());
  auto &Builder = State.Builder;
  /// Vectorize casts.
  assert(State.VF.isVector() && "Not vectorizing?");
  Type *DestTy = VectorType::get(getResultType(), State.VF);

  for (unsigned Part = 0; Part < State.UF; ++Part) {
    Value *A = State.get(getOperand(0), Part);
    Value *Cast = Builder.CreateCast(Instruction::CastOps(Opcode), A, DestTy);
    State.set(this, Cast, Part);
    State.addMetadata(Cast, cast_or_null<Instruction>(getUnderlyingValue()));
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenCastRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-CAST ";
  printAsOperand(O, SlotTracker);
  O << " = " << Instruction::getOpcodeName(Opcode) << " ";
  printOperands(O, SlotTracker);
  O << " to " << *getResultType();
}
#endif

/// This function adds
/// (StartIdx * Step, (StartIdx + 1) * Step, (StartIdx + 2) * Step, ...)
/// to each vector element of Val. The sequence starts at StartIndex.
/// \p Opcode is relevant for FP induction variable.
static Value *getStepVector(Value *Val, Value *StartIdx, Value *Step,
                            Instruction::BinaryOps BinOp, ElementCount VF,
                            IRBuilderBase &Builder) {
  assert(VF.isVector() && "only vector VFs are supported");

  // Create and check the types.
  auto *ValVTy = cast<VectorType>(Val->getType());
  ElementCount VLen = ValVTy->getElementCount();

  Type *STy = Val->getType()->getScalarType();
  assert((STy->isIntegerTy() || STy->isFloatingPointTy()) &&
         "Induction Step must be an integer or FP");
  assert(Step->getType() == STy && "Step has wrong type");

  SmallVector<Constant *, 8> Indices;

  // Create a vector of consecutive numbers from zero to VF.
  VectorType *InitVecValVTy = ValVTy;
  if (STy->isFloatingPointTy()) {
    Type *InitVecValSTy =
        IntegerType::get(STy->getContext(), STy->getScalarSizeInBits());
    InitVecValVTy = VectorType::get(InitVecValSTy, VLen);
  }
  Value *InitVec = Builder.CreateStepVector(InitVecValVTy);

  // Splat the StartIdx
  Value *StartIdxSplat = Builder.CreateVectorSplat(VLen, StartIdx);

  if (STy->isIntegerTy()) {
    InitVec = Builder.CreateAdd(InitVec, StartIdxSplat);
    Step = Builder.CreateVectorSplat(VLen, Step);
    assert(Step->getType() == Val->getType() && "Invalid step vec");
    // FIXME: The newly created binary instructions should contain nsw/nuw
    // flags, which can be found from the original scalar operations.
    Step = Builder.CreateMul(InitVec, Step);
    return Builder.CreateAdd(Val, Step, "induction");
  }

  // Floating point induction.
  assert((BinOp == Instruction::FAdd || BinOp == Instruction::FSub) &&
         "Binary Opcode should be specified for FP induction");
  InitVec = Builder.CreateUIToFP(InitVec, ValVTy);
  InitVec = Builder.CreateFAdd(InitVec, StartIdxSplat);

  Step = Builder.CreateVectorSplat(VLen, Step);
  Value *MulOp = Builder.CreateFMul(InitVec, Step);
  return Builder.CreateBinOp(BinOp, Val, MulOp, "induction");
}

/// A helper function that returns an integer or floating-point constant with
/// value C.
static Constant *getSignedIntOrFpConstant(Type *Ty, int64_t C) {
  return Ty->isIntegerTy() ? ConstantInt::getSigned(Ty, C)
                           : ConstantFP::get(Ty, C);
}

static Value *getRuntimeVFAsFloat(IRBuilderBase &B, Type *FTy,
                                  ElementCount VF) {
  assert(FTy->isFloatingPointTy() && "Expected floating point type!");
  Type *IntTy = IntegerType::get(FTy->getContext(), FTy->getScalarSizeInBits());
  Value *RuntimeVF = getRuntimeVF(B, IntTy, VF);
  return B.CreateUIToFP(RuntimeVF, FTy);
}

void VPWidenIntOrFpInductionRecipe::execute(VPTransformState &State) {
  assert(!State.Instance && "Int or FP induction being replicated.");

  Value *Start = getStartValue()->getLiveInIRValue();
  const InductionDescriptor &ID = getInductionDescriptor();
  TruncInst *Trunc = getTruncInst();
  IRBuilderBase &Builder = State.Builder;
  assert(IV->getType() == ID.getStartValue()->getType() && "Types must match");
  assert(State.VF.isVector() && "must have vector VF");

  // The value from the original loop to which we are mapping the new induction
  // variable.
  Instruction *EntryVal = Trunc ? cast<Instruction>(Trunc) : IV;

  // Fast-math-flags propagate from the original induction instruction.
  IRBuilder<>::FastMathFlagGuard FMFG(Builder);
  if (ID.getInductionBinOp() && isa<FPMathOperator>(ID.getInductionBinOp()))
    Builder.setFastMathFlags(ID.getInductionBinOp()->getFastMathFlags());

  // Now do the actual transformations, and start with fetching the step value.
  Value *Step = State.get(getStepValue(), VPIteration(0, 0));

  assert((isa<PHINode>(EntryVal) || isa<TruncInst>(EntryVal)) &&
         "Expected either an induction phi-node or a truncate of it!");

  // Construct the initial value of the vector IV in the vector loop preheader
  auto CurrIP = Builder.saveIP();
  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  Builder.SetInsertPoint(VectorPH->getTerminator());
  if (isa<TruncInst>(EntryVal)) {
    assert(Start->getType()->isIntegerTy() &&
           "Truncation requires an integer type");
    auto *TruncType = cast<IntegerType>(EntryVal->getType());
    Step = Builder.CreateTrunc(Step, TruncType);
    Start = Builder.CreateCast(Instruction::Trunc, Start, TruncType);
  }

  Value *Zero = getSignedIntOrFpConstant(Start->getType(), 0);
  Value *SplatStart = Builder.CreateVectorSplat(State.VF, Start);
  Value *SteppedStart = getStepVector(
      SplatStart, Zero, Step, ID.getInductionOpcode(), State.VF, State.Builder);

  // We create vector phi nodes for both integer and floating-point induction
  // variables. Here, we determine the kind of arithmetic we will perform.
  Instruction::BinaryOps AddOp;
  Instruction::BinaryOps MulOp;
  if (Step->getType()->isIntegerTy()) {
    AddOp = Instruction::Add;
    MulOp = Instruction::Mul;
  } else {
    AddOp = ID.getInductionOpcode();
    MulOp = Instruction::FMul;
  }

  // Multiply the vectorization factor by the step using integer or
  // floating-point arithmetic as appropriate.
  Type *StepType = Step->getType();
  Value *RuntimeVF;
  if (Step->getType()->isFloatingPointTy())
    RuntimeVF = getRuntimeVFAsFloat(Builder, StepType, State.VF);
  else
    RuntimeVF = getRuntimeVF(Builder, StepType, State.VF);
  Value *Mul = Builder.CreateBinOp(MulOp, Step, RuntimeVF);

  // Create a vector splat to use in the induction update.
  //
  // FIXME: If the step is non-constant, we create the vector splat with
  //        IRBuilder. IRBuilder can constant-fold the multiply, but it doesn't
  //        handle a constant vector splat.
  Value *SplatVF = isa<Constant>(Mul)
                       ? ConstantVector::getSplat(State.VF, cast<Constant>(Mul))
                       : Builder.CreateVectorSplat(State.VF, Mul);
  Builder.restoreIP(CurrIP);

  // We may need to add the step a number of times, depending on the unroll
  // factor. The last of those goes into the PHI.
  PHINode *VecInd = PHINode::Create(SteppedStart->getType(), 2, "vec.ind");
  VecInd->insertBefore(State.CFG.PrevBB->getFirstInsertionPt());
  VecInd->setDebugLoc(EntryVal->getDebugLoc());
  Instruction *LastInduction = VecInd;
  for (unsigned Part = 0; Part < State.UF; ++Part) {
    State.set(this, LastInduction, Part);

    if (isa<TruncInst>(EntryVal))
      State.addMetadata(LastInduction, EntryVal);

    LastInduction = cast<Instruction>(
        Builder.CreateBinOp(AddOp, LastInduction, SplatVF, "step.add"));
    LastInduction->setDebugLoc(EntryVal->getDebugLoc());
  }

  LastInduction->setName("vec.ind.next");
  VecInd->addIncoming(SteppedStart, VectorPH);
  // Add induction update using an incorrect block temporarily. The phi node
  // will be fixed after VPlan execution. Note that at this point the latch
  // block cannot be used, as it does not exist yet.
  // TODO: Model increment value in VPlan, by turning the recipe into a
  // multi-def and a subclass of VPHeaderPHIRecipe.
  VecInd->addIncoming(LastInduction, VectorPH);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenIntOrFpInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-INDUCTION";
  if (getTruncInst()) {
    O << "\\l\"";
    O << " +\n" << Indent << "\"  " << VPlanIngredient(IV) << "\\l\"";
    O << " +\n" << Indent << "\"  ";
    getVPValue(0)->printAsOperand(O, SlotTracker);
  } else
    O << " " << VPlanIngredient(IV);

  O << ", ";
  getStepValue()->printAsOperand(O, SlotTracker);
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
  return StartC && StartC->isZero() && StepC && StepC->isOne();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPDerivedIVRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << Indent << "= DERIVED-IV ";
  getStartValue()->printAsOperand(O, SlotTracker);
  O << " + ";
  getCanonicalIV()->printAsOperand(O, SlotTracker);
  O << " * ";
  getStepValue()->printAsOperand(O, SlotTracker);

  if (TruncResultTy)
    O << " (truncated to " << *TruncResultTy << ")";
}
#endif

void VPScalarIVStepsRecipe::execute(VPTransformState &State) {
  // Fast-math-flags propagate from the original induction instruction.
  IRBuilder<>::FastMathFlagGuard FMFG(State.Builder);
  if (hasFastMathFlags())
    State.Builder.setFastMathFlags(getFastMathFlags());

  /// Compute scalar induction steps. \p ScalarIV is the scalar induction
  /// variable on which to base the steps, \p Step is the size of the step.

  Value *BaseIV = State.get(getOperand(0), VPIteration(0, 0));
  Value *Step = State.get(getStepValue(), VPIteration(0, 0));
  IRBuilderBase &Builder = State.Builder;

  // Ensure step has the same type as that of scalar IV.
  Type *BaseIVTy = BaseIV->getType()->getScalarType();
  if (BaseIVTy != Step->getType()) {
    // TODO: Also use VPDerivedIVRecipe when only the step needs truncating, to
    // avoid separate truncate here.
    assert(Step->getType()->isIntegerTy() &&
           "Truncation requires an integer step");
    Step = State.Builder.CreateTrunc(Step, BaseIVTy);
  }

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

  unsigned StartPart = 0;
  unsigned EndPart = State.UF;
  unsigned StartLane = 0;
  unsigned EndLane = FirstLaneOnly ? 1 : State.VF.getKnownMinValue();
  if (State.Instance) {
    StartPart = State.Instance->Part;
    EndPart = StartPart + 1;
    StartLane = State.Instance->Lane.getKnownLane();
    EndLane = StartLane + 1;
  }
  for (unsigned Part = StartPart; Part < EndPart; ++Part) {
    Value *StartIdx0 = createStepForVF(Builder, IntStepTy, State.VF, Part);

    if (!FirstLaneOnly && State.VF.isScalable()) {
      auto *SplatStartIdx = Builder.CreateVectorSplat(State.VF, StartIdx0);
      auto *InitVec = Builder.CreateAdd(SplatStartIdx, UnitStepVec);
      if (BaseIVTy->isFloatingPointTy())
        InitVec = Builder.CreateSIToFP(InitVec, VecIVTy);
      auto *Mul = Builder.CreateBinOp(MulOp, InitVec, SplatStep);
      auto *Add = Builder.CreateBinOp(AddOp, SplatIV, Mul);
      State.set(this, Add, Part);
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
      State.set(this, Add, VPIteration(Part, Lane));
    }
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPScalarIVStepsRecipe::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << Indent << "= SCALAR-STEPS ";
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
      Ops.push_back(State.get(getOperand(I), VPIteration(0, 0)));

    auto *NewGEP =
        State.Builder.CreateGEP(GEP->getSourceElementType(), Ops[0],
                                ArrayRef(Ops).drop_front(), "", isInBounds());
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Value *EntryPart = State.Builder.CreateVectorSplat(State.VF, NewGEP);
      State.set(this, EntryPart, Part);
      State.addMetadata(EntryPart, GEP);
    }
  } else {
    // If the GEP has at least one loop-varying operand, we are sure to
    // produce a vector of pointers. But if we are only unrolling, we want
    // to produce a scalar GEP for each unroll part. Thus, the GEP we
    // produce with the code below will be scalar (if VF == 1) or vector
    // (otherwise). Note that for the unroll-only case, we still maintain
    // values in the vector mapping with initVector, as we do for other
    // instructions.
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      // The pointer operand of the new GEP. If it's loop-invariant, we
      // won't broadcast it.
      auto *Ptr = isPointerLoopInvariant()
                      ? State.get(getOperand(0), VPIteration(0, 0))
                      : State.get(getOperand(0), Part);

      // Collect all the indices for the new GEP. If any index is
      // loop-invariant, we won't broadcast it.
      SmallVector<Value *, 4> Indices;
      for (unsigned I = 1, E = getNumOperands(); I < E; I++) {
        VPValue *Operand = getOperand(I);
        if (isIndexLoopInvariant(I - 1))
          Indices.push_back(State.get(Operand, VPIteration(0, 0)));
        else
          Indices.push_back(State.get(Operand, Part));
      }

      // Create the new GEP. Note that this GEP may be a scalar if VF == 1,
      // but it should be a vector, otherwise.
      auto *NewGEP = State.Builder.CreateGEP(GEP->getSourceElementType(), Ptr,
                                             Indices, "", isInBounds());
      assert((State.VF.isScalar() || NewGEP->getType()->isVectorTy()) &&
             "NewGEP is not a pointer vector");
      State.set(this, NewGEP, Part);
      State.addMetadata(NewGEP, GEP);
    }
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

void VPBlendRecipe::execute(VPTransformState &State) {
  State.setDebugLocFrom(getDebugLoc());
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
 VectorParts Entry(State.UF);
  for (unsigned In = 0; In < NumIncoming; ++In) {
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      // We might have single edge PHIs (blocks) - use an identity
      // 'select' for the first PHI operand.
      Value *In0 = State.get(getIncomingValue(In), Part);
      if (In == 0)
        Entry[Part] = In0; // Initialize with the first incoming value.
      else {
        // Select between the current value and the previous incoming edge
        // based on the incoming mask.
        Value *Cond = State.get(getMask(In), Part);
        Entry[Part] =
            State.Builder.CreateSelect(Cond, In0, Entry[Part], "predphi");
      }
    }
  }
  for (unsigned Part = 0; Part < State.UF; ++Part)
    State.set(this, Entry[Part], Part);
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
      O << "/";
      getMask(I)->printAsOperand(O, SlotTracker);
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
  if (isa<FPMathOperator>(getUnderlyingInstr()))
    O << getUnderlyingInstr()->getFastMathFlags();
  O << " reduce." << Instruction::getOpcodeName(RdxDesc.getOpcode()) << " (";
  getVecOp()->printAsOperand(O, SlotTracker);
  if (getCondOp()) {
    O << ", ";
    getCondOp()->printAsOperand(O, SlotTracker);
  }
  O << ")";
  if (RdxDesc.IntermediateStore)
    O << " (with final reduction value stored in invariant address sank "
         "outside of loop)";
}
#endif

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

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPReplicateRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << (IsUniform ? "CLONE " : "REPLICATE ");

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
  assert(State.Instance && "Branch on Mask works only on single instance.");

  unsigned Part = State.Instance->Part;
  unsigned Lane = State.Instance->Lane.getKnownLane();

  Value *ConditionBit = nullptr;
  VPValue *BlockInMask = getMask();
  if (BlockInMask) {
    ConditionBit = State.get(BlockInMask, Part);
    if (ConditionBit->getType()->isVectorTy())
      ConditionBit = State.Builder.CreateExtractElement(
          ConditionBit, State.Builder.getInt32(Lane));
  } else // Block in mask is all-one.
    ConditionBit = State.Builder.getTrue();

  // Replace the temporary unreachable terminator with a new conditional branch,
  // whose two destinations will be set later when they are created.
  auto *CurrentTerminator = State.CFG.PrevBB->getTerminator();
  assert(isa<UnreachableInst>(CurrentTerminator) &&
         "Expected to replace unreachable terminator with conditional branch.");
  auto *CondBr = BranchInst::Create(State.CFG.PrevBB, nullptr, ConditionBit);
  CondBr->setSuccessor(0, nullptr);
  ReplaceInstWithInst(CurrentTerminator, CondBr);
}

void VPPredInstPHIRecipe::execute(VPTransformState &State) {
  assert(State.Instance && "Predicated instruction PHI works per instance.");
  Instruction *ScalarPredInst =
      cast<Instruction>(State.get(getOperand(0), *State.Instance));
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
  unsigned Part = State.Instance->Part;
  if (State.hasVectorValue(getOperand(0), Part)) {
    Value *VectorValue = State.get(getOperand(0), Part);
    InsertElementInst *IEI = cast<InsertElementInst>(VectorValue);
    PHINode *VPhi = State.Builder.CreatePHI(IEI->getType(), 2);
    VPhi->addIncoming(IEI->getOperand(0), PredicatingBB); // Unmodified vector.
    VPhi->addIncoming(IEI, PredicatedBB); // New vector with inserted element.
    if (State.hasVectorValue(this, Part))
      State.reset(this, VPhi, Part);
    else
      State.set(this, VPhi, Part);
    // NOTE: Currently we need to update the value of the operand, so the next
    // predicated iteration inserts its generated value in the correct vector.
    State.reset(getOperand(0), VPhi, Part);
  } else {
    Type *PredInstType = getOperand(0)->getUnderlyingValue()->getType();
    PHINode *Phi = State.Builder.CreatePHI(PredInstType, 2);
    Phi->addIncoming(PoisonValue::get(ScalarPredInst->getType()),
                     PredicatingBB);
    Phi->addIncoming(ScalarPredInst, PredicatedBB);
    if (State.hasScalarValue(this, *State.Instance))
      State.reset(this, Phi, *State.Instance);
    else
      State.set(this, Phi, *State.Instance);
    // NOTE: Currently we need to update the value of the operand, so the next
    // predicated iteration inserts its generated value in the correct vector.
    State.reset(getOperand(0), Phi, *State.Instance);
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

void VPWidenMemoryInstructionRecipe::print(raw_ostream &O, const Twine &Indent,
                                           VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN ";

  if (!isStore()) {
    getVPSingleValue()->printAsOperand(O, SlotTracker);
    O << " = ";
  }
  O << Instruction::getOpcodeName(Ingredient.getOpcode()) << " ";

  printOperands(O, SlotTracker);
}
#endif

void VPCanonicalIVPHIRecipe::execute(VPTransformState &State) {
  Value *Start = getStartValue()->getLiveInIRValue();
  PHINode *EntryPart = PHINode::Create(Start->getType(), 2, "index");
  EntryPart->insertBefore(State.CFG.PrevBB->getFirstInsertionPt());

  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  EntryPart->addIncoming(Start, VectorPH);
  EntryPart->setDebugLoc(getDebugLoc());
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part)
    State.set(this, EntryPart, Part);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPCanonicalIVPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                   VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = CANONICAL-INDUCTION";
}
#endif

bool VPCanonicalIVPHIRecipe::isCanonical(
    InductionDescriptor::InductionKind Kind, VPValue *Start, VPValue *Step,
    Type *Ty) const {
  // The types must match and it must be an integer induction.
  if (Ty != getScalarType() || Kind != InductionDescriptor::IK_IntInduction)
    return false;
  // Start must match the start value of this canonical induction.
  if (Start != getStartValue())
    return false;

  // If the step is defined by a recipe, it is not a ConstantInt.
  if (Step->getDefiningRecipe())
    return false;

  ConstantInt *StepC = dyn_cast<ConstantInt>(Step->getLiveInIRValue());
  return StepC && StepC->isOne();
}

bool VPWidenPointerInductionRecipe::onlyScalarsGenerated(ElementCount VF) {
  return IsScalarAfterVectorization &&
         (!VF.isScalable() || vputils::onlyFirstLaneUsed(this));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenPointerInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = WIDEN-POINTER-INDUCTION ";
  getStartValue()->printAsOperand(O, SlotTracker);
  O << ", " << *IndDesc.getStep();
}
#endif

void VPExpandSCEVRecipe::execute(VPTransformState &State) {
  assert(!State.Instance && "cannot be used in per-lane");
  const DataLayout &DL = State.CFG.PrevBB->getModule()->getDataLayout();
  SCEVExpander Exp(SE, DL, "induction");

  Value *Res = Exp.expandCodeFor(Expr, Expr->getType(),
                                 &*State.Builder.GetInsertPoint());
  assert(!State.ExpandedSCEVs.contains(Expr) &&
         "Same SCEV expanded multiple times");
  State.ExpandedSCEVs[Expr] = Res;
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part)
    State.set(this, Res, {Part, 0});
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPExpandSCEVRecipe::print(raw_ostream &O, const Twine &Indent,
                               VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  getVPSingleValue()->printAsOperand(O, SlotTracker);
  O << " = EXPAND SCEV " << *Expr;
}
#endif

void VPWidenCanonicalIVRecipe::execute(VPTransformState &State) {
  Value *CanonicalIV = State.get(getOperand(0), 0);
  Type *STy = CanonicalIV->getType();
  IRBuilder<> Builder(State.CFG.PrevBB->getTerminator());
  ElementCount VF = State.VF;
  Value *VStart = VF.isScalar()
                      ? CanonicalIV
                      : Builder.CreateVectorSplat(VF, CanonicalIV, "broadcast");
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part) {
    Value *VStep = createStepForVF(Builder, STy, VF, Part);
    if (VF.isVector()) {
      VStep = Builder.CreateVectorSplat(VF, VStep);
      VStep =
          Builder.CreateAdd(VStep, Builder.CreateStepVector(VStep->getType()));
    }
    Value *CanonicalVectorIV = Builder.CreateAdd(VStart, VStep, "vec.iv");
    State.set(this, CanonicalVectorIV, Part);
  }
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

  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
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
  PHINode *EntryPart = PHINode::Create(VecTy, 2, "vector.recur");
  EntryPart->insertBefore(State.CFG.PrevBB->getFirstInsertionPt());
  EntryPart->addIncoming(VectorInit, VectorPH);
  State.set(this, EntryPart, 0);
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
  PHINode *PN = cast<PHINode>(getUnderlyingValue());
  auto &Builder = State.Builder;

  // In order to support recurrences we need to be able to vectorize Phi nodes.
  // Phi nodes have cycles, so we need to vectorize them in two stages. This is
  // stage #1: We create a new vector PHI node with no incoming edges. We'll use
  // this value when we vectorize all of the instructions that use the PHI.
  bool ScalarPHI = State.VF.isScalar() || IsInLoop;
  Type *VecTy =
      ScalarPHI ? PN->getType() : VectorType::get(PN->getType(), State.VF);

  BasicBlock *HeaderBB = State.CFG.PrevBB;
  assert(State.CurrentVectorLoop->getHeader() == HeaderBB &&
         "recipe must be in the vector loop header");
  unsigned LastPartForNewPhi = isOrdered() ? 1 : State.UF;
  for (unsigned Part = 0; Part < LastPartForNewPhi; ++Part) {
    Instruction *EntryPart = PHINode::Create(VecTy, 2, "vec.phi");
    EntryPart->insertBefore(HeaderBB->getFirstInsertionPt());
    State.set(this, EntryPart, Part);
  }

  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);

  // Reductions do not have to start at zero. They can start with
  // any loop invariant values.
  VPValue *StartVPV = getStartValue();
  Value *StartV = StartVPV->getLiveInIRValue();

  Value *Iden = nullptr;
  RecurKind RK = RdxDesc.getRecurrenceKind();
  if (RecurrenceDescriptor::isMinMaxRecurrenceKind(RK) ||
      RecurrenceDescriptor::isAnyOfRecurrenceKind(RK)) {
    // MinMax and AnyOf reductions have the start value as their identity.
    if (ScalarPHI) {
      Iden = StartV;
    } else {
      IRBuilderBase::InsertPointGuard IPBuilder(Builder);
      Builder.SetInsertPoint(VectorPH->getTerminator());
      StartV = Iden =
          Builder.CreateVectorSplat(State.VF, StartV, "minmax.ident");
    }
  } else {
    Iden = RdxDesc.getRecurrenceIdentity(RK, VecTy->getScalarType(),
                                         RdxDesc.getFastMathFlags());

    if (!ScalarPHI) {
      Iden = Builder.CreateVectorSplat(State.VF, Iden);
      IRBuilderBase::InsertPointGuard IPBuilder(Builder);
      Builder.SetInsertPoint(VectorPH->getTerminator());
      Constant *Zero = Builder.getInt32(0);
      StartV = Builder.CreateInsertElement(Iden, StartV, Zero);
    }
  }

  for (unsigned Part = 0; Part < LastPartForNewPhi; ++Part) {
    Value *EntryPart = State.get(this, Part);
    // Make sure to add the reduction start value only to the
    // first unroll part.
    Value *StartVal = (Part == 0) ? StartV : Iden;
    cast<PHINode>(EntryPart)->addIncoming(StartVal, VectorPH);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPReductionPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                 VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-REDUCTION-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

void VPWidenPHIRecipe::execute(VPTransformState &State) {
  assert(EnableVPlanNativePath &&
         "Non-native vplans are not expected to have VPWidenPHIRecipes.");

  // Currently we enter here in the VPlan-native path for non-induction
  // PHIs where all control flow is uniform. We simply widen these PHIs.
  // Create a vector phi with no operands - the vector phi operands will be
  // set at the end of vector code generation.
  VPBasicBlock *Parent = getParent();
  VPRegionBlock *LoopRegion = Parent->getEnclosingLoopRegion();
  unsigned StartIdx = 0;
  // For phis in header blocks of loop regions, use the index of the value
  // coming from the preheader.
  if (LoopRegion->getEntryBasicBlock() == Parent) {
    for (unsigned I = 0; I < getNumOperands(); ++I) {
      if (getIncomingBlock(I) ==
          LoopRegion->getSinglePredecessor()->getExitingBasicBlock())
        StartIdx = I;
    }
  }
  Value *Op0 = State.get(getOperand(StartIdx), 0);
  Type *VecTy = Op0->getType();
  Value *VecPhi = State.Builder.CreatePHI(VecTy, 2, "vec.phi");
  State.set(this, VecPhi, 0);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                             VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-PHI ";

  auto *OriginalPhi = cast<PHINode>(getUnderlyingValue());
  // Unless all incoming values are modeled in VPlan  print the original PHI
  // directly.
  // TODO: Remove once all VPWidenPHIRecipe instances keep all relevant incoming
  // values as VPValues.
  if (getNumOperands() != OriginalPhi->getNumOperands()) {
    O << VPlanIngredient(OriginalPhi);
    return;
  }

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

// TODO: It would be good to use the existing VPWidenPHIRecipe instead and
// remove VPActiveLaneMaskPHIRecipe.
void VPActiveLaneMaskPHIRecipe::execute(VPTransformState &State) {
  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part) {
    Value *StartMask = State.get(getOperand(0), Part);
    PHINode *EntryPart =
        State.Builder.CreatePHI(StartMask->getType(), 2, "active.lane.mask");
    EntryPart->addIncoming(StartMask, VectorPH);
    EntryPart->setDebugLoc(getDebugLoc());
    State.set(this, EntryPart, Part);
  }
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
