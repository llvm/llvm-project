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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
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

bool isAbsorbableFMulOrFAdd(ArrayRef<Value *> VL, Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  return I &&
         (I->getOpcode() == Instruction::FMul ||
          I->getOpcode() == Instruction::FAdd) &&
         I->hasOneUse() && none_of(I->operands(), [&](Value *Op) {
           return is_contained(VL, Op);
         });
}

bool isAbsorbableCopyableFMulOrFAdd(const InstructionsState &S, Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  return I && S.isCopyableElement(I) &&
         (I->getOpcode() == Instruction::FMul ||
          I->getOpcode() == Instruction::FAdd) &&
         I->hasOneUse();
}

bool hasOnlyAbsorbableCopyableFMulOrFAdds(ArrayRef<Value *> VL) {
  bool HasFMulOrFAdd = false;
  for (Value *V : VL) {
    if (isa<PoisonValue>(V))
      continue;
    auto *I = dyn_cast<Instruction>(V);
    if (I && RecurrenceDescriptor::isFMulAddIntrinsic(I))
      continue;
    if (!isAbsorbableFMulOrFAdd(VL, V))
      return false;
    HasFMulOrFAdd = true;
  }
  return HasFMulOrFAdd;
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

/// Find an instruction with a specific opcode in VL.
/// \param VL Array of values to search through. Must contain only Instructions
///           and PoisonValues.
/// \param Opcode The instruction opcode to search for
/// \returns
/// - The first instruction found with matching opcode
/// - nullptr if no matching instruction is found
static Instruction *findInstructionWithOpcode(ArrayRef<Value *> VL,
                                              unsigned Opcode) {
  for (Value *V : VL) {
    if (isa<PoisonValue>(V))
      continue;
    assert(isa<Instruction>(V) && "Only accepts PoisonValue and Instruction.");
    auto *Inst = cast<Instruction>(V);
    if (Inst->getOpcode() == Opcode)
      return Inst;
  }
  return nullptr;
}

/// Checks if the provided operands of 2 cmp instructions are compatible, i.e.
/// compatible instructions or constants, or just some other regular values.
static bool areCompatibleCmpOps(Value *BaseOp0, Value *BaseOp1, Value *Op0,
                                Value *Op1, const TargetLibraryInfo &TLI) {
  return (isConstant(BaseOp0) && isConstant(Op0)) ||
         (isConstant(BaseOp1) && isConstant(Op1)) ||
         (!isa<Instruction>(BaseOp0) && !isa<Instruction>(Op0) &&
          !isa<Instruction>(BaseOp1) && !isa<Instruction>(Op1)) ||
         BaseOp0 == Op0 || BaseOp1 == Op1 ||
         getSameOpcode({BaseOp0, Op0}, TLI) ||
         getSameOpcode({BaseOp1, Op1}, TLI);
}

/// \returns true if a compare instruction \p CI has similar "look" and
/// same predicate as \p BaseCI, "as is" or with its operands and predicate
/// swapped, false otherwise.
static bool isCmpSameOrSwapped(const CmpInst *BaseCI, const CmpInst *CI,
                               const TargetLibraryInfo &TLI) {
  assert(BaseCI->getOperand(0)->getType() == CI->getOperand(0)->getType() &&
         "Assessing comparisons of different types?");
  CmpInst::Predicate BasePred = BaseCI->getPredicate();
  CmpInst::Predicate Pred = CI->getPredicate();
  CmpInst::Predicate SwappedPred = CmpInst::getSwappedPredicate(Pred);

  Value *BaseOp0 = BaseCI->getOperand(0);
  Value *BaseOp1 = BaseCI->getOperand(1);
  Value *Op0 = CI->getOperand(0);
  Value *Op1 = CI->getOperand(1);

  return (BasePred == Pred &&
          areCompatibleCmpOps(BaseOp0, BaseOp1, Op0, Op1, TLI)) ||
         (BasePred == SwappedPred &&
          areCompatibleCmpOps(BaseOp0, BaseOp1, Op1, Op0, TLI));
}

InstructionsState getSameOpcode(ArrayRef<Value *> VL,
                                const TargetLibraryInfo &TLI) {
  // Make sure these are all Instructions.
  if (!all_of(VL, IsaPred<Instruction, PoisonValue>))
    return InstructionsState::invalid();

  auto *It = find_if(VL, IsaPred<Instruction>);
  if (It == VL.end())
    return InstructionsState::invalid();

  Instruction *MainOp = cast<Instruction>(*It);
  unsigned InstCnt = std::count_if(It, VL.end(), IsaPred<Instruction>);
  if ((VL.size() > 2 && !isa<PHINode>(MainOp) && InstCnt < VL.size() / 2) ||
      (VL.size() == 2 && InstCnt < 2))
    return InstructionsState::invalid();

  bool IsCastOp = isa<CastInst>(MainOp);
  bool IsBinOp = isa<BinaryOperator>(MainOp);
  bool IsCmpOp = isa<CmpInst>(MainOp);
  CmpInst::Predicate BasePred = IsCmpOp ? cast<CmpInst>(MainOp)->getPredicate()
                                        : CmpInst::BAD_ICMP_PREDICATE;
  Instruction *AltOp = MainOp;
  unsigned Opcode = MainOp->getOpcode();
  unsigned AltOpcode = Opcode;

  BinOpSameOpcodeHelper BinOpHelper(MainOp);
  bool SwappedPredsCompatible = IsCmpOp && [&]() {
    SetVector<unsigned> UniquePreds, UniqueNonSwappedPreds;
    UniquePreds.insert(BasePred);
    UniqueNonSwappedPreds.insert(BasePred);
    for (Value *V : VL) {
      auto *I = dyn_cast<CmpInst>(V);
      if (!I)
        return false;
      CmpInst::Predicate CurrentPred = I->getPredicate();
      CmpInst::Predicate SwappedCurrentPred =
          CmpInst::getSwappedPredicate(CurrentPred);
      UniqueNonSwappedPreds.insert(CurrentPred);
      if (!UniquePreds.contains(CurrentPred) &&
          !UniquePreds.contains(SwappedCurrentPred))
        UniquePreds.insert(CurrentPred);
    }
    // Total number of predicates > 2, but if consider swapped predicates
    // compatible only 2, consider swappable predicates as compatible opcodes,
    // not alternate.
    return UniqueNonSwappedPreds.size() > 2 && UniquePreds.size() == 2;
  }();
  // Check for one alternate opcode from another BinaryOperator.
  // TODO - generalize to support all operators (types, calls etc.).
  Intrinsic::ID BaseID = 0;
  SmallVector<VFInfo, 4> BaseMappings;
  if (auto *CallBase = dyn_cast<CallInst>(MainOp)) {
    BaseID = getVectorIntrinsicIDForCall(CallBase, &TLI);
    BaseMappings = VFDatabase(*CallBase).getMappings(*CallBase);
    if (!isTriviallyVectorizable(BaseID) && BaseMappings.empty())
      return InstructionsState::invalid();
  }
  bool AnyPoison = InstCnt != VL.size();
  // Check MainOp too to be sure that it matches the requirements for the
  // instructions.
  for (Value *V : iterator_range(It, VL.end())) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      continue;

    // Cannot combine poison and divisions.
    // TODO: do some smart analysis of the CallInsts to exclude divide-like
    // intrinsics/functions only.
    if (AnyPoison && (I->isIntDivRem() || I->isFPDivRem() || isa<CallInst>(I)))
      return InstructionsState::invalid();
    unsigned InstOpcode = I->getOpcode();
    if (IsBinOp && isa<BinaryOperator>(I)) {
      if (BinOpHelper.add(I))
        continue;
    } else if (IsCastOp && isa<CastInst>(I)) {
      Value *Op0 = MainOp->getOperand(0);
      Type *Ty0 = Op0->getType();
      Value *Op1 = I->getOperand(0);
      Type *Ty1 = Op1->getType();
      if (Ty0 == Ty1) {
        if (InstOpcode == Opcode || InstOpcode == AltOpcode)
          continue;
        if (Opcode == AltOpcode) {
          assert(isValidForAlternation(Opcode) &&
                 isValidForAlternation(InstOpcode) &&
                 "Cast isn't safe for alternation, logic needs to be updated!");
          AltOpcode = InstOpcode;
          AltOp = I;
          continue;
        }
      }
    } else if (auto *Inst = dyn_cast<CmpInst>(I); Inst && IsCmpOp) {
      auto *BaseInst = cast<CmpInst>(MainOp);
      Type *Ty0 = BaseInst->getOperand(0)->getType();
      Type *Ty1 = Inst->getOperand(0)->getType();
      if (Ty0 == Ty1) {
        assert(InstOpcode == Opcode && "Expected same CmpInst opcode.");
        assert(InstOpcode == AltOpcode &&
               "Alternate instructions are only supported by BinaryOperator "
               "and CastInst.");
        // Check for compatible operands. If the corresponding operands are not
        // compatible - need to perform alternate vectorization.
        CmpInst::Predicate CurrentPred = Inst->getPredicate();
        CmpInst::Predicate SwappedCurrentPred =
            CmpInst::getSwappedPredicate(CurrentPred);

        if ((VL.size() == 2 || SwappedPredsCompatible) &&
            (BasePred == CurrentPred || BasePred == SwappedCurrentPred))
          continue;

        if (isCmpSameOrSwapped(BaseInst, Inst, TLI))
          continue;
        auto *AltInst = cast<CmpInst>(AltOp);
        if (MainOp != AltOp) {
          if (isCmpSameOrSwapped(AltInst, Inst, TLI))
            continue;
        } else if (BasePred != CurrentPred) {
          assert(
              isValidForAlternation(InstOpcode) &&
              "CmpInst isn't safe for alternation, logic needs to be updated!");
          AltOp = I;
          continue;
        }
        CmpInst::Predicate AltPred = AltInst->getPredicate();
        if (BasePred == CurrentPred || BasePred == SwappedCurrentPred ||
            AltPred == CurrentPred || AltPred == SwappedCurrentPred)
          continue;
      }
    } else if (InstOpcode == Opcode) {
      assert(InstOpcode == AltOpcode &&
             "Alternate instructions are only supported by BinaryOperator and "
             "CastInst.");
      if (auto *Gep = dyn_cast<GetElementPtrInst>(I)) {
        if (Gep->getNumOperands() != 2 ||
            Gep->getOperand(0)->getType() != MainOp->getOperand(0)->getType())
          return InstructionsState::invalid();
      } else if (auto *EI = dyn_cast<ExtractElementInst>(I)) {
        if (!isVectorLikeInstWithConstOps(EI))
          return InstructionsState::invalid();
      } else if (auto *LI = dyn_cast<LoadInst>(I)) {
        auto *BaseLI = cast<LoadInst>(MainOp);
        if (!LI->isSimple() || !BaseLI->isSimple())
          return InstructionsState::invalid();
      } else if (auto *Call = dyn_cast<CallInst>(I)) {
        auto *CallBase = cast<CallInst>(MainOp);
        Intrinsic::ID ID = getVectorIntrinsicIDForCall(Call, &TLI);
        Intrinsic::ID Equivalent = isEquivalentIntrinsicID(ID, BaseID);
        if (Call->getCalledFunction() != CallBase->getCalledFunction() &&
            isEquivalentIntrinsicID(Equivalent, Intrinsic::fmuladd) ==
                Intrinsic::not_intrinsic)
          return InstructionsState::invalid();
        if (Call->hasOperandBundles() &&
            (!CallBase->hasOperandBundles() ||
             !std::equal(Call->op_begin() + Call->getBundleOperandsStartIndex(),
                         Call->op_begin() + Call->getBundleOperandsEndIndex(),
                         CallBase->op_begin() +
                             CallBase->getBundleOperandsStartIndex())))
          return InstructionsState::invalid();
        if (ID != BaseID && Equivalent == Intrinsic::not_intrinsic)
          return InstructionsState::invalid();
        if (!ID) {
          SmallVector<VFInfo, 4> Mappings =
              VFDatabase(*Call).getMappings(*Call);
          if (Mappings.size() != BaseMappings.size() ||
              Mappings.front().ISA != BaseMappings.front().ISA ||
              Mappings.front().ScalarName != BaseMappings.front().ScalarName ||
              Mappings.front().VectorName != BaseMappings.front().VectorName ||
              Mappings.front().Shape.VF != BaseMappings.front().Shape.VF ||
              Mappings.front().Shape.Parameters !=
                  BaseMappings.front().Shape.Parameters)
            return InstructionsState::invalid();
        }
      }
      continue;
    }
    return InstructionsState::invalid();
  }

  if (IsBinOp) {
    if (!BinOpHelper.hasDefinedMainOpcode() ||
        !BinOpHelper.hasDefinedAltOpcode())
      return InstructionsState::invalid();
    MainOp = findInstructionWithOpcode(VL, BinOpHelper.getMainOpcode());
    assert(MainOp && "Cannot find MainOp with Opcode from BinOpHelper.");
    AltOp = findInstructionWithOpcode(VL, BinOpHelper.getAltOpcode());
    assert(AltOp && "Cannot find AltOp with Opcode from BinOpHelper.");
  } else if (auto *CB = dyn_cast<CallInst>(MainOp);
             CB &&
             getVectorIntrinsicIDForCall(CB, &TLI) == Intrinsic::fmuladd) {
    // fma and fmuladd share a single vector fma node; use the fma as the
    // representative so the fused form is not weakened to fmuladd.
    auto *It = find_if(VL, [&](Value *V) {
      auto *CI = dyn_cast<CallInst>(V);
      return CI && getVectorIntrinsicIDForCall(CI, &TLI) == Intrinsic::fma;
    });
    if (It != VL.end())
      MainOp = AltOp = cast<Instruction>(*It);
  }
  assert((MainOp == AltOp || !allSameOpcode(VL)) &&
         "Incorrect implementation of allSameOpcode.");
  InstructionsState S(MainOp, AltOp);
  assert(all_of(VL,
                [&](Value *V) {
                  return isa<PoisonValue>(V) ||
                         S.getMatchingMainOpOrAltOp(cast<Instruction>(V));
                }) &&
         "Invalid InstructionsState.");
  return S;
}

std::pair<Instruction *, SmallVector<Value *>>
convertTo(Instruction *I, const InstructionsState &S) {
  Instruction *SelectedOp = S.getMatchingMainOpOrAltOp(I);
  assert(SelectedOp && "Cannot convert the instruction.");
  if (I->isBinaryOp()) {
    BinOpSameOpcodeHelper Converter(I);
    return std::make_pair(SelectedOp, Converter.getOperand(SelectedOp));
  }
  // Use args() to skip the trailing callee operand in CallInst::operands().
  if (auto *CI = dyn_cast<CallInst>(I))
    return std::make_pair(SelectedOp, SmallVector<Value *>(CI->args()));
  return std::make_pair(SelectedOp, SmallVector<Value *>(I->operands()));
}

bool isAlternateInstruction(Instruction *I, Instruction *MainOp,
                            Instruction *AltOp, const TargetLibraryInfo &TLI) {
  if (auto *MainCI = dyn_cast<CmpInst>(MainOp)) {
    auto *AltCI = cast<CmpInst>(AltOp);
    CmpInst::Predicate MainP = MainCI->getPredicate();
    [[maybe_unused]] CmpInst::Predicate AltP = AltCI->getPredicate();
    assert(MainP != AltP && "Expected different main/alternate predicates.");
    auto *CI = cast<CmpInst>(I);
    if (isCmpSameOrSwapped(MainCI, CI, TLI))
      return false;
    if (isCmpSameOrSwapped(AltCI, CI, TLI))
      return true;
    CmpInst::Predicate P = CI->getPredicate();
    CmpInst::Predicate SwappedP = CmpInst::getSwappedPredicate(P);

    assert((MainP == P || AltP == P || MainP == SwappedP || AltP == SwappedP) &&
           "CmpInst expected to match either main or alternate predicate or "
           "their swap.");
    return MainP != P && MainP != SwappedP;
  }
  return InstructionsState(MainOp, AltOp).getMatchingMainOpOrAltOp(I) == AltOp;
}

SmallVector<SmallVector<Value *>> scanAltAssociativeOperands(
    const InstructionsState &S, const TargetLibraryInfo &TLI,
    ArrayRef<Value *> VL, ArrayRef<Value *> Op0, ArrayRef<Value *> Op1,
    SmallVectorImpl<Value *> &ReassocScalars, SmallBitVector &SubLanes) {
  assert(S.isAltShuffle() && "Expected an alternate node.");
  const unsigned NumLanes = VL.size();
  SmallVector<unsigned> LaneOpcodes =
      map_to_vector(seq<unsigned>(NumLanes), [&](unsigned Lane) {
        return isAlternateInstruction(cast<Instruction>(VL[Lane]),
                                      S.getMainOp(), S.getAltOp(), TLI)
                   ? S.getAltOpcode()
                   : S.getOpcode();
      });
  // A lane value peels only as a single-use chain link with the lane's own
  // opcode, keeping every combine level on the same main/alt pattern.
  auto GetChainLink = [&](unsigned Lane, Value *V) -> Instruction * {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !I->hasOneUse() || I->getOpcode() != LaneOpcodes[Lane] ||
        !isReassocChainLink(I))
      return nullptr;
    return I;
  };
  SmallVector<SmallVector<Value *>> Columns;
  Columns.emplace_back(Op0.begin(), Op0.end());
  Columns.emplace_back(Op1.begin(), Op1.end());
  // The chain link of a commutative lane may sit in the second column;
  // normalize so every lane's link leads.
  for (unsigned Lane : seq<unsigned>(NumLanes)) {
    if (GetChainLink(Lane, Columns[0][Lane]))
      continue;
    Instruction *Link = GetChainLink(Lane, Columns[1][Lane]);
    if (!Link || !Link->isCommutative())
      return {};
    std::swap(Columns[0][Lane], Columns[1][Lane]);
  }
  // Peel the leading column while every lane stays a matching chain link.
  while (all_of(seq<unsigned>(NumLanes), [&](unsigned Lane) {
    return GetChainLink(Lane, Columns[0][Lane]) != nullptr;
  })) {
    SmallVector<Value *> NewColumn(NumLanes);
    for (unsigned Lane : seq<unsigned>(NumLanes)) {
      Instruction *Link = GetChainLink(Lane, Columns[0][Lane]);
      ReassocScalars.push_back(Link);
      // The chain of a commutative lane may continue in the second operand;
      // keep the chain link as the running value.
      unsigned RunningOp = Link->isCommutative() &&
                                   !GetChainLink(Lane, Link->getOperand(0)) &&
                                   GetChainLink(Lane, Link->getOperand(1))
                               ? 1
                               : 0;
      NewColumn[Lane] = Link->getOperand(1 - RunningOp);
      Columns[0][Lane] = Link->getOperand(RunningOp);
    }
    Columns.insert(std::next(Columns.begin()), std::move(NewColumn));
  }
  assert(!ReassocScalars.empty() &&
         "Normalization guarantees at least one peeled level.");
  SubLanes.resize(NumLanes);
  for (unsigned Lane : seq<unsigned>(NumLanes))
    if (LaneOpcodes[Lane] == Instruction::Sub ||
        LaneOpcodes[Lane] == Instruction::FSub)
      SubLanes.set(Lane);
  return Columns;
}
} // namespace llvm::slpvectorizer
