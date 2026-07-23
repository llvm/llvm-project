//===- SLPUtils.cpp - SLP Vectorizer free utility helpers -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <type_traits>

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm::slpvectorizer {

bool isConstant(Value *V) {
  return isa<Constant>(V) && !isa<ConstantExpr, GlobalValue>(V);
}

bool isVectorLikeInstWithConstOps(Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  // Non-instructions are vector-like only if they are undef.
  if (!I)
    return isa<UndefValue>(V);
  switch (I->getOpcode()) {
  case Instruction::ExtractValue:
  case Instruction::InsertValue:
    return true;
  case Instruction::ExtractElement:
    return isa<FixedVectorType>(I->getOperand(0)->getType()) &&
           isConstant(I->getOperand(1));
  case Instruction::InsertElement:
    return isa<FixedVectorType>(I->getOperand(0)->getType()) &&
           isConstant(I->getOperand(2));
  default:
    return false;
  }
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

Intrinsic::ID isEquivalentIntrinsicID(Intrinsic::ID LHS, Intrinsic::ID RHS) {
  if (LHS == RHS)
    return RHS;
  if ((LHS == Intrinsic::fma || LHS == Intrinsic::fmuladd) &&
      (RHS == Intrinsic::fma || RHS == Intrinsic::fmuladd))
    return Intrinsic::fma;
  return Intrinsic::not_intrinsic;
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
  return all_of(make_range(It, VL.end()), [&](Value *V) {
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
  assert(!VL.empty() && "Expected non-empty list of values.");
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

void transformScalarShuffleIndiciesToVector(unsigned VecTyNumElements,
                                            SmallVectorImpl<int> &Mask) {
  // The ShuffleBuilder implementation use shufflevector to splat an "element".
  // But the element have different meaning for SLP (scalar) and REVEC
  // (vector). We need to expand Mask into masks which shufflevector can use
  // directly.
  SmallVector<int> NewMask(Mask.size() * VecTyNumElements);
  for (unsigned I : seq<unsigned>(Mask.size()))
    for (auto [J, MaskV] : enumerate(MutableArrayRef(NewMask).slice(
             I * VecTyNumElements, VecTyNumElements)))
      MaskV = Mask[I] == PoisonMaskElem ? PoisonMaskElem
                                        : Mask[I] * VecTyNumElements + J;
  Mask.swap(NewMask);
}

unsigned getShufflevectorNumGroups(ArrayRef<Value *> VL) {
  if (VL.empty())
    return 0;
  if (!all_of(VL, IsaPred<ShuffleVectorInst>))
    return 0;
  auto *SV = cast<ShuffleVectorInst>(VL.front());
  unsigned SVNumElements =
      cast<FixedVectorType>(SV->getOperand(0)->getType())->getNumElements();
  unsigned ShuffleMaskSize = SV->getShuffleMask().size();
  if (SVNumElements % ShuffleMaskSize != 0)
    return 0;
  unsigned GroupSize = SVNumElements / ShuffleMaskSize;
  if (GroupSize == 0 || (VL.size() % GroupSize) != 0)
    return 0;
  unsigned NumGroup = 0;
  for (size_t I = 0, E = VL.size(); I != E; I += GroupSize) {
    auto *SV = cast<ShuffleVectorInst>(VL[I]);
    Value *Src = SV->getOperand(0);
    ArrayRef<Value *> Group = VL.slice(I, GroupSize);
    SmallBitVector ExpectedIndex(GroupSize);
    if (!all_of(Group, [&](Value *V) {
          auto *SV = cast<ShuffleVectorInst>(V);
          // From the same source.
          if (SV->getOperand(0) != Src)
            return false;
          int Index;
          if (!SV->isExtractSubvectorMask(Index))
            return false;
          ExpectedIndex.set(Index / ShuffleMaskSize);
          return true;
        }))
      return 0;
    if (!ExpectedIndex.all())
      return 0;
    ++NumGroup;
  }
  assert(NumGroup == (VL.size() / GroupSize) && "Unexpected number of groups");
  return NumGroup;
}

SmallVector<int> calculateShufflevectorMask(ArrayRef<Value *> VL) {
  assert(getShufflevectorNumGroups(VL) && "Not supported shufflevector usage.");
  auto *SV = cast<ShuffleVectorInst>(VL.front());
  unsigned SVNumElements =
      cast<FixedVectorType>(SV->getOperand(0)->getType())->getNumElements();
  SmallVector<int> Mask;
  unsigned AccumulateLength = 0;
  for (Value *V : VL) {
    auto *SV = cast<ShuffleVectorInst>(V);
    for (int M : SV->getShuffleMask())
      Mask.push_back(M == PoisonMaskElem ? PoisonMaskElem
                                         : AccumulateLength + M);
    AccumulateLength += SVNumElements;
  }
  return Mask;
}

SmallBitVector buildUseMask(int VF, ArrayRef<int> Mask, UseMask MaskArg) {
  SmallBitVector UseMask(VF, true);
  for (auto [Idx, Value] : enumerate(Mask)) {
    if (Value == PoisonMaskElem) {
      if (MaskArg == UseMask::UndefsAsMask)
        UseMask.reset(Idx);
      continue;
    }
    if (MaskArg == UseMask::FirstArg && Value < VF)
      UseMask.reset(Value);
    else if (MaskArg == UseMask::SecondArg && Value >= VF)
      UseMask.reset(Value - VF);
  }
  return UseMask;
}

template <bool IsPoisonOnly>
SmallBitVector isUndefVector(const Value *V, const SmallBitVector &UseMask) {
  SmallBitVector Res(UseMask.empty() ? 1 : UseMask.size(), true);
  using T = std::conditional_t<IsPoisonOnly, PoisonValue, UndefValue>;
  if (isa<T>(V))
    return Res;
  auto *VecTy = dyn_cast<FixedVectorType>(V->getType());
  if (!VecTy)
    return Res.reset();
  auto *C = dyn_cast<Constant>(V);
  if (!C) {
    if (!UseMask.empty()) {
      const Value *Base = V;
      while (auto *II = dyn_cast<InsertElementInst>(Base)) {
        Base = II->getOperand(0);
        if (isa<T>(II->getOperand(1)))
          continue;
        std::optional<unsigned> Idx = getElementIndex(II);
        if (!Idx) {
          Res.reset();
          return Res;
        }
        if (*Idx < UseMask.size() && !UseMask.test(*Idx))
          Res.reset(*Idx);
      }
      // TODO: Add analysis for shuffles here too.
      if (V == Base) {
        Res.reset();
      } else {
        SmallBitVector SubMask(UseMask.size(), false);
        Res &= isUndefVector<IsPoisonOnly>(Base, SubMask);
      }
    } else {
      Res.reset();
    }
    return Res;
  }
  for (unsigned I = 0, E = VecTy->getNumElements(); I != E; ++I) {
    if (Constant *Elem = C->getAggregateElement(I))
      if (!isa<T>(Elem) &&
          (UseMask.empty() || (I < UseMask.size() && !UseMask.test(I))))
        Res.reset(I);
  }
  return Res;
}

template SmallBitVector isUndefVector<false>(const Value *,
                                             const SmallBitVector &);
template SmallBitVector isUndefVector<true>(const Value *,
                                            const SmallBitVector &);

bool doesInTreeUserNeedToExtract(Value *Scalar, Instruction *UserInst,
                                 TargetLibraryInfo *TLI,
                                 const TargetTransformInfo *TTI) {
  if (!UserInst)
    return false;
  unsigned Opcode = UserInst->getOpcode();
  switch (Opcode) {
  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(UserInst);
    return (LI->getPointerOperand() == Scalar);
  }
  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(UserInst);
    return (SI->getPointerOperand() == Scalar);
  }
  case Instruction::Call: {
    CallInst *CI = cast<CallInst>(UserInst);
    Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
    return any_of(enumerate(CI->args()), [&](auto &&Arg) {
      return isVectorIntrinsicWithScalarOpAtArg(ID, Arg.index(), TTI) &&
             Arg.value().get() == Scalar;
    });
  }
  default:
    return false;
  }
}

MemoryLocation getLocation(Instruction *I) {
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return MemoryLocation::get(SI);
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return MemoryLocation::get(LI);
  return MemoryLocation();
}

bool isSimple(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isSimple();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isSimple();
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I))
    return !MI->isVolatile();
  return true;
}

bool isSelectedBaseLoad(Type *ScalarTy, ArrayRef<Value *> PointerOps,
                        const DataLayout &DL, Value *&TrueBase,
                        Value *&FalseBase,
                        SmallVectorImpl<Value *> &Conditions) {
  TrueBase = nullptr;
  FalseBase = nullptr;
  uint64_t ScalarSize = DL.getTypeStoreSize(ScalarTy);
  Conditions.assign(PointerOps.size(), nullptr);
  for (auto [Idx, P] : enumerate(PointerOps)) {
    Value *Base = P;
    uint64_t Offset = 0;
    if (auto *GEP = dyn_cast<GetElementPtrInst>(P)) {
      APInt OffsetAP(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      if (!GEP->accumulateConstantOffset(DL, OffsetAP) || OffsetAP.isNegative())
        return false;
      Offset = OffsetAP.getZExtValue();
      Base = GEP->getPointerOperand();
    }
    auto *Sel = dyn_cast<SelectInst>(Base);
    if (!Sel)
      return false;
    Value *T = Sel->getTrueValue();
    Value *F = Sel->getFalseValue();
    if (!TrueBase) {
      if (T == F)
        return false;
      TrueBase = T;
      FalseBase = F;
    } else if (TrueBase != T || FalseBase != F) {
      return false;
    }
    // Lane Idx must be at exactly Base + Idx * sizeof(ScalarTy); codegen reads
    // contiguously from TrueBase/FalseBase starting at lane 0.
    if (Offset != static_cast<uint64_t>(Idx) * ScalarSize)
      return false;
    Conditions[Idx] = Sel->getCondition();
  }
  return TrueBase != nullptr;
}

void addMask(SmallVectorImpl<int> &Mask, ArrayRef<int> SubMask,
             bool ExtendingManyInputs) {
  if (SubMask.empty())
    return;
  assert(
      (!ExtendingManyInputs || SubMask.size() > Mask.size() ||
       // Check if input scalars were extended to match the size of other node.
       (SubMask.size() == Mask.size() && Mask.back() == PoisonMaskElem)) &&
      "SubMask with many inputs support must be larger than the mask.");
  if (Mask.empty()) {
    Mask.append(SubMask.begin(), SubMask.end());
    return;
  }
  SmallVector<int> NewMask(SubMask.size(), PoisonMaskElem);
  int TermValue = std::min(Mask.size(), SubMask.size());
  for (int I = 0, E = SubMask.size(); I < E; ++I) {
    if (SubMask[I] == PoisonMaskElem ||
        (!ExtendingManyInputs &&
         (SubMask[I] >= TermValue || Mask[SubMask[I]] >= TermValue)))
      continue;
    NewMask[I] = Mask[SubMask[I]];
  }
  Mask.swap(NewMask);
}

void fixupOrderingIndices(MutableArrayRef<unsigned> Order) {
  const size_t Sz = Order.size();
  SmallBitVector UnusedIndices(Sz, /*t=*/true);
  SmallBitVector MaskedIndices(Sz);
  for (unsigned I = 0; I < Sz; ++I) {
    if (Order[I] < Sz)
      UnusedIndices.reset(Order[I]);
    else
      MaskedIndices.set(I);
  }
  if (MaskedIndices.none())
    return;
  assert(UnusedIndices.count() == MaskedIndices.count() &&
         "Non-synced masked/available indices.");
  int Idx = UnusedIndices.find_first();
  int MIdx = MaskedIndices.find_first();
  while (MIdx >= 0) {
    assert(Idx >= 0 && "Indices must be synced.");
    Order[MIdx] = Idx;
    Idx = UnusedIndices.find_next(Idx);
    MIdx = MaskedIndices.find_next(MIdx);
  }
}

SmallBitVector getAltInstrMask(ArrayRef<Value *> VL, Type *ScalarTy,
                               unsigned Opcode0, unsigned Opcode1) {
  unsigned ScalarTyNumElements = getNumElements(ScalarTy);
  SmallBitVector OpcodeMask(VL.size() * ScalarTyNumElements, false);
  for (unsigned Lane : seq<unsigned>(VL.size())) {
    if (isa<PoisonValue>(VL[Lane]))
      continue;
    if (cast<Instruction>(VL[Lane])->getOpcode() == Opcode1)
      OpcodeMask.set(Lane * ScalarTyNumElements,
                     Lane * ScalarTyNumElements + ScalarTyNumElements);
  }
  return OpcodeMask;
}

SmallVector<Constant *> replicateMask(ArrayRef<Constant *> Val, unsigned VF) {
  assert(none_of(Val, [](Constant *C) { return C->getType()->isVectorTy(); }) &&
         "Expected scalar constants.");
  SmallVector<Constant *> NewVal(Val.size() * VF);
  for (auto [I, V] : enumerate(Val))
    std::fill_n(NewVal.begin() + I * VF, VF, V);
  return NewVal;
}

Intrinsic::ID getMaskedDivRemIntrinsic(unsigned Opcode) {
  switch (Opcode) {
  case Instruction::UDiv:
    return Intrinsic::masked_udiv;
  case Instruction::SDiv:
    return Intrinsic::masked_sdiv;
  case Instruction::URem:
    return Intrinsic::masked_urem;
  case Instruction::SRem:
    return Intrinsic::masked_srem;
  default:
    llvm_unreachable("Unexpected opcode");
  }
}

} // namespace llvm::slpvectorizer
