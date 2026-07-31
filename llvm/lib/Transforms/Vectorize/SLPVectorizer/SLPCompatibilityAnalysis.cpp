//===- SLPCompatibilityAnalysis.cpp - SLP same-opcode helpers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPCompatibilityAnalysis.h"
#include "SLPUtils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cassert>
#include <utility>

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm::slpvectorizer {

bool isValidForAlternation(unsigned Opcode) {
  return !Instruction::isIntDivRem(Opcode);
}

std::pair<Constant *, unsigned>
BinOpSameOpcodeHelper::isBinOpWithConstant(const Instruction *I) {
  [[maybe_unused]] unsigned Opcode = I->getOpcode();
  assert(binary_search(SupportedOp, Opcode) && "Unsupported opcode.");
  (void)SupportedOp;
  auto *BinOp = cast<BinaryOperator>(I);
  auto GetConstant = [](Value *V) -> Constant * {
    if (auto *CI = dyn_cast<ConstantInt>(V))
      return CI;
    return dyn_cast<ConstantFP>(V);
  };
  if (Constant *C = GetConstant(BinOp->getOperand(1)))
    return {C, 1};
  if (!isCommutative(I))
    return {nullptr, 0};
  if (Constant *C = GetConstant(BinOp->getOperand(0)))
    return {C, 0};
  return {nullptr, 0};
}

bool BinOpSameOpcodeHelper::InterchangeableInfo::trySet(
    MaskType OpcodeInMaskForm, MaskType InterchangeableMask) {
  if (Mask & InterchangeableMask) {
    SeenBefore |= OpcodeInMaskForm;
    Mask &= InterchangeableMask;
    return true;
  }
  return false;
}

unsigned BinOpSameOpcodeHelper::InterchangeableInfo::getOpcode() const {
  MaskType Candidate = Mask & SeenBefore;
  if (Candidate & MainOpBIT)
    return I->getOpcode();
  if (Candidate & ShlBIT)
    return Instruction::Shl;
  if (Candidate & AShrBIT)
    return Instruction::AShr;
  if (Candidate & MulBIT)
    return Instruction::Mul;
  if (Candidate & AddBIT)
    return Instruction::Add;
  if (Candidate & SubBIT)
    return Instruction::Sub;
  if (Candidate & FAddBIT)
    return Instruction::FAdd;
  if (Candidate & FSubBIT)
    return Instruction::FSub;
  if (Candidate & AndBIT)
    return Instruction::And;
  if (Candidate & OrBIT)
    return Instruction::Or;
  if (Candidate & XorBIT)
    return Instruction::Xor;
  llvm_unreachable("Cannot find interchangeable instruction.");
}

bool BinOpSameOpcodeHelper::InterchangeableInfo::hasCandidateOpcode(
    unsigned Opcode) const {
  MaskType Candidate = Mask & SeenBefore;
  switch (Opcode) {
  case Instruction::Shl:
    return Candidate & ShlBIT;
  case Instruction::AShr:
    return Candidate & AShrBIT;
  case Instruction::Mul:
    return Candidate & MulBIT;
  case Instruction::Add:
    return Candidate & AddBIT;
  case Instruction::Sub:
    return Candidate & SubBIT;
  case Instruction::And:
    return Candidate & AndBIT;
  case Instruction::Or:
    return Candidate & OrBIT;
  case Instruction::Xor:
    return Candidate & XorBIT;
  case Instruction::FAdd:
    return Candidate & FAddBIT;
  case Instruction::FSub:
    return Candidate & FSubBIT;
  case Instruction::LShr:
  case Instruction::FMul:
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::FDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::FRem:
    return false;
  default:
    break;
  }
  llvm_unreachable("Cannot find interchangeable instruction.");
}

SmallVector<Value *> BinOpSameOpcodeHelper::InterchangeableInfo::getOperand(
    const Instruction *To) const {
  unsigned ToOpcode = To->getOpcode();
  unsigned FromOpcode = I->getOpcode();
  if (FromOpcode == ToOpcode)
    return SmallVector<Value *>(I->operands());
  assert(binary_search(SupportedOp, ToOpcode) && "Unsupported opcode.");
  auto [C, Pos] = isBinOpWithConstant(I);
  Type *RHSType = I->getOperand(Pos)->getType();
  Constant *RHS;
  if (auto *CFP = dyn_cast<ConstantFP>(C)) {
    // fsub(x, c) == fadd(x, -c) for every FP constant c, since IEEE 754
    // defines subtraction as addition of the negated operand.
    assert(is_contained({Instruction::FAdd, Instruction::FSub}, ToOpcode) &&
           "Cannot convert the instruction.");
    RHS = ConstantFP::get(RHSType, -CFP->getValueAPF());
  } else {
    auto *CI = cast<ConstantInt>(C);
    const APInt &FromCIValue = CI->getValue();
    unsigned FromCIValueBitWidth = FromCIValue.getBitWidth();
    switch (FromOpcode) {
    case Instruction::Shl:
      if (ToOpcode == Instruction::Add && FromCIValue.isOne())
        return {I->getOperand(0), I->getOperand(0)};
      if (ToOpcode == Instruction::Mul) {
        RHS = ConstantInt::get(RHSType,
                               APInt::getOneBitSet(FromCIValueBitWidth,
                                                   FromCIValue.getZExtValue()));
      } else {
        assert(FromCIValue.isZero() && "Cannot convert the instruction.");
        RHS = ConstantExpr::getBinOpIdentity(ToOpcode, RHSType,
                                             /*AllowRHSConstant=*/true);
      }
      break;
    case Instruction::Mul:
      assert(FromCIValue.isPowerOf2() && "Cannot convert the instruction.");
      if (ToOpcode == Instruction::Shl) {
        RHS = ConstantInt::get(
            RHSType, APInt(FromCIValueBitWidth, FromCIValue.logBase2()));
      } else {
        assert(FromCIValue.isOne() && "Cannot convert the instruction.");
        RHS = ConstantExpr::getBinOpIdentity(ToOpcode, RHSType,
                                             /*AllowRHSConstant=*/true);
      }
      break;
    case Instruction::Add:
    case Instruction::Sub:
      if (FromCIValue.isZero()) {
        RHS = ConstantExpr::getBinOpIdentity(ToOpcode, RHSType,
                                             /*AllowRHSConstant=*/true);
      } else {
        assert(is_contained({Instruction::Add, Instruction::Sub}, ToOpcode) &&
               "Cannot convert the instruction.");
        APInt NegatedVal = APInt(FromCIValue);
        NegatedVal.negate();
        RHS = ConstantInt::get(RHSType, NegatedVal);
      }
      break;
    case Instruction::And:
      assert(FromCIValue.isAllOnes() && "Cannot convert the instruction.");
      RHS = ConstantExpr::getBinOpIdentity(ToOpcode, RHSType,
                                           /*AllowRHSConstant=*/true);
      break;
    default:
      assert(FromCIValue.isZero() && "Cannot convert the instruction.");
      RHS = ConstantExpr::getBinOpIdentity(ToOpcode, RHSType,
                                           /*AllowRHSConstant=*/true);
      break;
    }
  }
  Value *LHS = I->getOperand(1 - Pos);
  // If the target opcode is non-commutative (e.g., shl, sub),
  // force the variable to the left and the constant to the right.
  if (Pos == 1 || !Instruction::isCommutative(ToOpcode))
    return SmallVector<Value *>({LHS, RHS});

  return SmallVector<Value *>({RHS, LHS});
}

bool BinOpSameOpcodeHelper::isValidForAlternation(const Instruction *I) const {
  return slpvectorizer::isValidForAlternation(MainOp.I->getOpcode()) &&
         slpvectorizer::isValidForAlternation(I->getOpcode());
}

bool BinOpSameOpcodeHelper::initializeAltOp(const Instruction *I) {
  if (AltOp.I)
    return true;
  if (!isValidForAlternation(I))
    return false;
  AltOp.I = I;
  return true;
}

bool BinOpSameOpcodeHelper::add(const Instruction *I) {
  assert(isa<BinaryOperator>(I) &&
         "BinOpSameOpcodeHelper only accepts BinaryOperator.");
  unsigned Opcode = I->getOpcode();
  MaskType OpcodeInMaskForm;
  // Prefer Shl, AShr, Mul, Add, Sub, And, Or, Xor, FAdd and FSub over
  // MainOp.
  switch (Opcode) {
  case Instruction::Shl:
    OpcodeInMaskForm = ShlBIT;
    break;
  case Instruction::AShr:
    OpcodeInMaskForm = AShrBIT;
    break;
  case Instruction::Mul:
    OpcodeInMaskForm = MulBIT;
    break;
  case Instruction::Add:
    OpcodeInMaskForm = AddBIT;
    break;
  case Instruction::Sub:
    OpcodeInMaskForm = SubBIT;
    break;
  case Instruction::And:
    OpcodeInMaskForm = AndBIT;
    break;
  case Instruction::Or:
    OpcodeInMaskForm = OrBIT;
    break;
  case Instruction::Xor:
    OpcodeInMaskForm = XorBIT;
    break;
  case Instruction::FAdd:
    OpcodeInMaskForm = FAddBIT;
    break;
  case Instruction::FSub:
    OpcodeInMaskForm = FSubBIT;
    break;
  default:
    return MainOp.equal(Opcode) || (initializeAltOp(I) && AltOp.equal(Opcode));
  }
  MaskType InterchangeableMask = OpcodeInMaskForm;
  auto [C, Pos] = isBinOpWithConstant(I);
  if (auto *CI = dyn_cast_or_null<ConstantInt>(C)) {
    constexpr MaskType CanBeAll =
        XorBIT | OrBIT | AndBIT | SubBIT | AddBIT | MulBIT | AShrBIT | ShlBIT;
    const APInt &CIValue = CI->getValue();
    switch (Opcode) {
    case Instruction::Shl:
      if (CIValue.ult(CIValue.getBitWidth()))
        InterchangeableMask = CIValue.isZero() ? CanBeAll : MulBIT | ShlBIT;
      if (CIValue.isOne())
        InterchangeableMask |= AddBIT;
      break;
    case Instruction::Mul:
      if (CIValue.isOne()) {
        InterchangeableMask = CanBeAll;
        break;
      }
      if (CIValue.isPowerOf2())
        InterchangeableMask = MulBIT | ShlBIT;
      break;
    case Instruction::Add:
    case Instruction::Sub:
      InterchangeableMask = CIValue.isZero() ? CanBeAll : SubBIT | AddBIT;
      break;
    case Instruction::And:
      if (CIValue.isAllOnes())
        InterchangeableMask = CanBeAll;
      break;
    case Instruction::Xor:
      if (CIValue.isZero())
        InterchangeableMask = XorBIT | OrBIT | SubBIT | AddBIT;
      break;
    default:
      if (CIValue.isZero())
        InterchangeableMask = CanBeAll;
      break;
    }
  } else if (C && Pos == 1) {
    // FAdd/FSub with a constant RHS: negating the constant always
    // converts one into the other, so no value check is needed. A
    // constant LHS (Pos == 0, e.g. "0.0 - x") is excluded: unlike a
    // constant RHS, it cannot be moved to the other opcode without also
    // swapping the variable operand, which would misalign it against
    // lanes that keep their native opcode (their variable operand stays
    // on the other side).
    InterchangeableMask = FSubBIT | FAddBIT;
  }
  return MainOp.trySet(OpcodeInMaskForm, InterchangeableMask) ||
         (initializeAltOp(I) &&
          AltOp.trySet(OpcodeInMaskForm, InterchangeableMask));
}

bool InstructionsState::isSameOperation(const Instruction *I,
                                        const Instruction *Op) {
  if (I->getOpcode() != Op->getOpcode())
    return false;
  const auto *II = dyn_cast<IntrinsicInst>(I);
  const auto *IOp = dyn_cast<IntrinsicInst>(Op);
  if (II || IOp)
    return II && IOp &&
           isEquivalentIntrinsicID(II->getIntrinsicID(),
                                   IOp->getIntrinsicID()) !=
               Intrinsic::not_intrinsic;
  return true;
}

Instruction *InstructionsState::getMatchingMainOpOrAltOp(Instruction *I) const {
  assert(MainOp && "MainOp cannot be nullptr.");
  if (isSameOperation(I, MainOp))
    return MainOp;
  if (MainOp->getOpcode() == Instruction::Select &&
      I->getOpcode() == Instruction::ZExt && !isAltShuffle())
    return MainOp;
  // Prefer AltOp instead of interchangeable instruction of MainOp.
  assert(AltOp && "AltOp cannot be nullptr.");
  if (isSameOperation(I, AltOp))
    return AltOp;
  // BinOpSameOpcodeHelper handles only BinaryOperators; a call cannot match.
  if (!I->isBinaryOp() || !MainOp->isBinaryOp())
    return nullptr;
  BinOpSameOpcodeHelper Converter(MainOp);
  if (!Converter.add(I) || !Converter.add(MainOp))
    return nullptr;
  if (isAltShuffle() && !Converter.hasCandidateOpcode(MainOp->getOpcode())) {
    BinOpSameOpcodeHelper AltConverter(AltOp);
    if (AltConverter.add(I) && AltConverter.add(AltOp) &&
        AltConverter.hasCandidateOpcode(AltOp->getOpcode()))
      return AltOp;
  }
  if (Converter.hasAltOp() && !isAltShuffle())
    return nullptr;
  return Converter.hasAltOp() ? AltOp : MainOp;
}

bool InstructionsState::isMulDivLikeOp() const {
  constexpr std::array<unsigned, 8> MulDiv = {
      Instruction::Mul,  Instruction::FMul, Instruction::SDiv,
      Instruction::UDiv, Instruction::FDiv, Instruction::SRem,
      Instruction::URem, Instruction::FRem};
  return is_contained(MulDiv, getOpcode()) &&
         is_contained(MulDiv, getAltOpcode());
}

bool InstructionsState::isAddSubLikeOp() const {
  constexpr std::array<unsigned, 4> AddSub = {
      Instruction::Add, Instruction::Sub, Instruction::FAdd, Instruction::FSub};
  return is_contained(AddSub, getOpcode()) &&
         is_contained(AddSub, getAltOpcode());
}

bool InstructionsState::isCopyableElement(Value *V) const {
  assert(valid() && "InstructionsState is invalid.");
  if (!HasCopyables)
    return false;
  if (isAltShuffle() || getOpcode() == Instruction::GetElementPtr)
    return false;
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return !isa<PoisonValue>(V);
  if (I->getParent() != MainOp->getParent() &&
      (!isVectorLikeInstWithConstOps(I) ||
       !isVectorLikeInstWithConstOps(MainOp)))
    return true;
  if (isSameOperation(I, MainOp))
    return false;
  // BinOpSameOpcodeHelper handles only BinaryOperators; a call is copyable.
  if (!I->isBinaryOp() || !MainOp->isBinaryOp())
    return true;
  BinOpSameOpcodeHelper Converter(MainOp);
  return !Converter.add(I) || !Converter.add(MainOp) || Converter.hasAltOp() ||
         !Converter.hasCandidateOpcode(getOpcode());
}

bool isAbsorbableFMul(ArrayRef<Value *> VL, Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  return I && I->getOpcode() == Instruction::FMul && I->hasOneUse() &&
         none_of(I->operands(),
                 [&](Value *Op) { return is_contained(VL, Op); });
}

bool isAbsorbableCopyableFMul(const InstructionsState &S, Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  return I && S.isCopyableElement(I) && I->getOpcode() == Instruction::FMul &&
         I->hasOneUse();
}

bool InstructionsState::isExpandedBinOp(Value *V) const {
  assert(valid() && "InstructionsState is invalid.");
  if (isCopyableElement(V))
    return false;
  auto *ExpandingOp = dyn_cast<Instruction>(V);
  if (!ExpandingOp)
    return false;
  auto CheckForTransformedOpcode = [](const Instruction *RefOp,
                                      const Instruction *ExpandingOp) {
    switch (RefOp->getOpcode()) {
    case Instruction::Add:
      switch (ExpandingOp->getOpcode()) {
      case Instruction::Shl:
        return match(ExpandingOp, m_Shl(m_Value(), m_One()));
      default:
        break;
      }
      break;
    default:
      break;
    }
    return false;
  };
  // getMatchingMainOpOrAltOp() may legitimately return nullptr, e.g. for a
  // split node, whose Scalars combine two unrelated operations (main/alt
  // ops of the split state), so V is not required to match either of them.
  Instruction *MainOp = getMatchingMainOpOrAltOp(ExpandingOp);
  if (!MainOp)
    return false;
  return CheckForTransformedOpcode(MainOp, ExpandingOp);
}

bool InstructionsState::isExpandedOperand(Instruction *I, unsigned Idx) const {
  assert(isExpandedBinOp(I) && "Expected an expanded binop.");
  switch (I->getOpcode()) {
  case Instruction::Shl:
    assert(match(I, m_Shl(m_Value(), m_One())) && "Expected shl x, 1 only.");
    return Idx == 1;
  default:
    llvm_unreachable("Unexpected opcode for an expanded operand.");
  }
}

bool InstructionsState::isNonSchedulable(Value *V) const {
  assert(valid() && "InstructionsState is invalid.");
  auto *I = dyn_cast<Instruction>(V);
  if (!HasCopyables)
    return !I || isa<PHINode>(I) || isVectorLikeInstWithConstOps(I) ||
           doesNotNeedToBeScheduled(V);
  // MainOp for copyables always schedulable to correctly identify
  // non-schedulable copyables.
  if (getMainOp() == V)
    return false;
  if (isCopyableElement(V)) {
    auto IsNonSchedulableCopyableElement = [this](Value *V) {
      auto *I = dyn_cast<Instruction>(V);
      return !I || isa<PHINode>(I) || I->getParent() != MainOp->getParent() ||
             (doesNotNeedToBeScheduled(I) &&
              // If the copyable instructions comes after MainOp
              // (non-schedulable, but used in the block) - cannot vectorize
              // it, will possibly generate use before def.
              !MainOp->comesBefore(I));
    };

    return IsNonSchedulableCopyableElement(V);
  }
  return !I || isa<PHINode>(I) || isVectorLikeInstWithConstOps(I) ||
         doesNotNeedToBeScheduled(V);
}

} // namespace llvm::slpvectorizer
