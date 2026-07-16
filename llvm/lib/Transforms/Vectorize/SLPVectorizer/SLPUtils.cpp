//===- SLPUtils.cpp - SLP Vectorizer free utility helpers -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <type_traits>

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm::slpvectorizer {

bool isConstant(Value *V) {
  return isa<Constant>(V) && !isa<ConstantExpr, GlobalValue>(V);
}

bool isVectorLikeInstWithConstOps(Value *V) {
  if (!isa<InsertElementInst, InsertValueInst, ExtractElementInst>(V) &&
      !isa<ExtractValueInst, UndefValue>(V))
    return false;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || isa<ExtractValueInst>(I))
    return true;
  if (isa<ExtractElementInst>(I))
    return isa<FixedVectorType>(I->getOperand(0)->getType()) &&
           isConstant(I->getOperand(1));
  if (isa<InsertElementInst>(I))
    return isa<FixedVectorType>(I->getOperand(0)->getType()) &&
           isConstant(I->getOperand(2));
  assert(isa<InsertValueInst>(I) && "Expected InsertValueInst");
  return true;
}

unsigned getNumElements(Type *Ty) {
  assert(!isa<ScalableVectorType>(Ty) &&
         "ScalableVectorType is not supported.");
  if (isVectorizedTy(Ty))
    return getVectorizedTypeVF(Ty).getFixedValue();
  return 1;
}

unsigned getPartNumElems(unsigned Size, unsigned NumParts) {
  return std::min<unsigned>(Size, bit_ceil(divideCeil(Size, NumParts)));
}

unsigned getNumElems(unsigned Size, unsigned PartNumElems, unsigned Part) {
  return std::min<unsigned>(PartNumElems, Size - Part * PartNumElems);
}

#if !defined(NDEBUG)
std::string shortBundleName(ArrayRef<Value *> VL, int Idx) {
  std::string Result;
  raw_string_ostream OS(Result);
  if (Idx >= 0)
    OS << "Idx: " << Idx << ", ";
  OS << "n=" << VL.size() << " [" << *VL.front() << ", ..]";
  return Result;
}
#endif

bool allSameBlock(ArrayRef<Value *> VL) {
  auto *It = find_if(VL, IsaPred<Instruction>);
  if (It == VL.end())
    return false;
  Instruction *I0 = cast<Instruction>(*It);
  if (all_of(VL, isVectorLikeInstWithConstOps))
    return true;

  BasicBlock *BB = I0->getParent();
  for (Value *V : iterator_range(It, VL.end())) {
    if (isa<PoisonValue>(V))
      continue;
    auto *II = dyn_cast<Instruction>(V);
    if (!II)
      return false;

    if (BB != II->getParent())
      return false;
  }
  return true;
}

bool allConstant(ArrayRef<Value *> VL) {
  // Constant expressions and globals can't be vectorized like normal integer/FP
  // constants.
  return all_of(VL, isConstant);
}

bool isSplat(ArrayRef<Value *> VL) {
  Value *FirstNonUndef = nullptr;
  for (Value *V : VL) {
    if (isa<UndefValue>(V))
      continue;
    if (!FirstNonUndef) {
      FirstNonUndef = V;
      continue;
    }
    if (V != FirstNonUndef)
      return false;
  }
  return FirstNonUndef != nullptr;
}

bool isCommutative(const Instruction *I, const Value *ValWithUses,
                   bool IsCopyable) {
  if (auto *Cmp = dyn_cast<CmpInst>(I))
    return Cmp->isCommutative();
  if (auto *BO = dyn_cast<BinaryOperator>(I))
    return BO->isCommutative() ||
           (BO->getOpcode() == Instruction::Sub && ValWithUses->hasUseList() &&
            !ValWithUses->hasNUsesOrMore(UsesLimit) &&
            all_of(
                ValWithUses->uses(),
                [&](const Use &U) {
                  // Commutative, if icmp eq/ne sub, 0
                  CmpPredicate Pred;
                  if (match(U.getUser(),
                            m_ICmp(Pred, m_Specific(U.get()), m_Zero())) &&
                      (Pred == ICmpInst::ICMP_EQ || Pred == ICmpInst::ICMP_NE))
                    return true;
                  // Commutative, if abs(sub nsw, true) or abs(sub, false).
                  ConstantInt *Flag;
                  auto *I = dyn_cast<BinaryOperator>(U.get());
                  return match(U.getUser(),
                               m_Intrinsic<Intrinsic::abs>(
                                   m_Specific(U.get()), m_ConstantInt(Flag))) &&
                         ((!IsCopyable && I && !I->hasNoSignedWrap()) ||
                          Flag->isOne());
                })) ||
           (BO->getOpcode() == Instruction::FSub && ValWithUses->hasUseList() &&
            !ValWithUses->hasNUsesOrMore(UsesLimit) &&
            all_of(ValWithUses->uses(), [](const Use &U) {
              return match(U.getUser(),
                           m_Intrinsic<Intrinsic::fabs>(m_Specific(U.get())));
            }));
  return I->isCommutative();
}

bool isCommutative(const Instruction *I) { return isCommutative(I, I); }

bool isCommutableOperand(const Instruction *I, Value *ValWithUses, unsigned Op,
                         bool IsCopyable) {
  assert(isCommutative(I, ValWithUses, IsCopyable) &&
         "The instruction is not commutative.");
  if (isa<CmpInst>(I))
    return true;
  if (auto *BO = dyn_cast<BinaryOperator>(I)) {
    switch (BO->getOpcode()) {
    case Instruction::Sub:
    case Instruction::FSub:
      return true;
    default:
      break;
    }
  }
  return I->isCommutableOperand(Op);
}

unsigned getNumberOfPotentiallyCommutativeOps(Instruction *I) {
  if (isa<IntrinsicInst>(I) && isCommutative(I)) {
    // IntrinsicInst::isCommutative returns true if swapping the first "two"
    // arguments to the intrinsic produces the same result.
    constexpr unsigned IntrinsicNumOperands = 2;
    return IntrinsicNumOperands;
  }
  return I->getNumOperands();
}

std::optional<unsigned> getElementIndex(const Value *Inst, unsigned Offset) {
  if (auto Index = getInsertExtractIndex<InsertElementInst>(Inst, Offset))
    return Index;
  if (auto Index = getInsertExtractIndex<ExtractElementInst>(Inst, Offset))
    return Index;

  unsigned Index = Offset;

  const auto *IV = dyn_cast<InsertValueInst>(Inst);
  if (!IV)
    return std::nullopt;

  Type *CurrentType = IV->getType();
  for (unsigned I : IV->indices()) {
    if (const auto *ST = dyn_cast<StructType>(CurrentType)) {
      Index *= ST->getNumElements();
      CurrentType = ST->getElementType(I);
    } else if (const auto *AT = dyn_cast<ArrayType>(CurrentType)) {
      Index *= AT->getNumElements();
      CurrentType = AT->getElementType();
    } else {
      return std::nullopt;
    }
    Index += I;
  }
  return Index;
}

bool allSameOpcode(ArrayRef<Value *> VL) {
  auto *It = find_if(VL, IsaPred<Instruction>);
  if (It == VL.end())
    return true;
  Instruction *MainOp = cast<Instruction>(*It);
  unsigned Opcode = MainOp->getOpcode();
  bool IsCmpOp = isa<CmpInst>(MainOp);
  CmpInst::Predicate BasePred = IsCmpOp ? cast<CmpInst>(MainOp)->getPredicate()
                                        : CmpInst::BAD_ICMP_PREDICATE;
  return std::all_of(It, VL.end(), [&](Value *V) {
    if (auto *CI = dyn_cast<CmpInst>(V))
      return BasePred == CI->getPredicate();
    if (auto *I = dyn_cast<Instruction>(V))
      return I->getOpcode() == Opcode;
    return isa<PoisonValue>(V);
  });
}

std::optional<unsigned> getExtractIndex(const Instruction *E) {
  unsigned Opcode = E->getOpcode();
  assert((Opcode == Instruction::ExtractElement ||
          Opcode == Instruction::ExtractValue) &&
         "Expected extractelement or extractvalue instruction.");
  if (Opcode == Instruction::ExtractElement) {
    auto *CI = dyn_cast<ConstantInt>(E->getOperand(1));
    if (!CI)
      return std::nullopt;
    // Check if the index is out of bound. We can get the source vector from
    // operand 0.
    unsigned Idx = CI->getZExtValue();
    auto *EE = cast<ExtractElementInst>(E);
    const unsigned VF = getNumElements(EE->getVectorOperandType());
    if (Idx >= VF)
      return std::nullopt;
    return Idx;
  }
  auto *EI = cast<ExtractValueInst>(E);
  if (EI->getNumIndices() != 1)
    return std::nullopt;
  return *EI->idx_begin();
}

void inversePermutation(ArrayRef<unsigned> Indices,
                        SmallVectorImpl<int> &Mask) {
  Mask.clear();
  const unsigned E = Indices.size();
  Mask.resize(E, PoisonMaskElem);
  for (unsigned I = 0; I < E; ++I)
    Mask[Indices[I]] = I;
}

void reorderScalars(SmallVectorImpl<Value *> &Scalars, ArrayRef<int> Mask) {
  assert(!Mask.empty() && "Expected non-empty mask.");
  SmallVector<Value *> Prev(Scalars.size(),
                            PoisonValue::get(Scalars.front()->getType()));
  Prev.swap(Scalars);
  for (unsigned I = 0, E = Prev.size(); I < E; ++I)
    if (Mask[I] != PoisonMaskElem)
      Scalars[Mask[I]] = Prev[I];
}

bool allSameType(ArrayRef<Value *> VL) {
  Type *Ty = VL.consume_front()->getType();
  return all_of(VL, [&](Value *V) { return V->getType() == Ty; });
}

template <typename T>
std::optional<unsigned> getInsertExtractIndex(const Value *Inst,
                                              unsigned Offset) {
  static_assert(std::is_same_v<T, InsertElementInst> ||
                    std::is_same_v<T, ExtractElementInst>,
                "unsupported T");
  const auto *IE = dyn_cast<T>(Inst);
  if (!IE)
    return std::nullopt;
  // InsertElement: result is the vector, index is op 2.
  // ExtractElement: result is scalar, vector is op 0, index is op 1.
  constexpr bool IsInsert = std::is_same_v<T, InsertElementInst>;
  Type *VecTy = IsInsert ? IE->getType() : IE->getOperand(0)->getType();
  const auto *VT = dyn_cast<FixedVectorType>(VecTy);
  if (!VT)
    return std::nullopt;
  const auto *CI = dyn_cast<ConstantInt>(IE->getOperand(IsInsert ? 2 : 1));
  if (!CI)
    return std::nullopt;
  if (CI->getValue().uge(VT->getNumElements()))
    return std::nullopt;
  unsigned Index = Offset;
  Index *= VT->getNumElements();
  Index += CI->getZExtValue();
  return Index;
}

// Only these two specializations are used; instantiate them here so the
// definition can stay out of the header.
template std::optional<unsigned>
getInsertExtractIndex<InsertElementInst>(const Value *, unsigned);
template std::optional<unsigned>
getInsertExtractIndex<ExtractElementInst>(const Value *, unsigned);

bool areAllOperandsNonInsts(Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return true;
  return !mayHaveNonDefUseDependency(*I) &&
         all_of(I->operands(), [I](Value *V) {
           auto *IO = dyn_cast<Instruction>(V);
           if (!IO)
             return true;
           return isa<PHINode>(IO) || IO->getParent() != I->getParent();
         });
}

bool isUsedOutsideBlock(Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return true;
  // Limits the number of uses to save compile time.
  return !I->mayReadOrWriteMemory() && !I->hasNUsesOrMore(UsesLimit) &&
         all_of(I->users(), [I](User *U) {
           auto *IU = dyn_cast<Instruction>(U);
           if (!IU)
             return true;
           return IU->getParent() != I->getParent() || isa<PHINode>(IU);
         });
}

bool doesNotNeedToBeScheduled(Value *V) {
  return areAllOperandsNonInsts(V) && isUsedOutsideBlock(V);
}

bool doesNotNeedToSchedule(ArrayRef<Value *> VL) {
  return !VL.empty() &&
         (all_of(VL, isUsedOutsideBlock) || all_of(VL, areAllOperandsNonInsts));
}

} // namespace llvm::slpvectorizer
