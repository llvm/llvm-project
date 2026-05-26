//===- InferAlignment.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infer alignment for load, stores and other memory operations based on
// trailing zero known bits information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/InferAlignment.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

static bool tryToImproveAlign(
    const DataLayout &DL, Instruction *I,
    function_ref<Align(Value *PtrOp, Align OldAlign, Align PrefAlign)> Fn) {

  if (auto *PtrOp = getLoadStorePointerOperand(I)) {
    Align OldAlign = getLoadStoreAlignment(I);
    Align PrefAlign = DL.getPrefTypeAlign(getLoadStoreType(I));

    Align NewAlign = Fn(PtrOp, OldAlign, PrefAlign);
    if (NewAlign > OldAlign) {
      setLoadStoreAlignment(I, NewAlign);
      return true;
    }
  }

  Value *PtrOp;
  const APInt *Const;
  if (match(I, m_And(m_PtrToIntOrAddr(m_Value(PtrOp)), m_APInt(Const)))) {
    Align ActualAlign = Fn(PtrOp, Align(1), Align(1));
    if (Const->ult(ActualAlign.value())) {
      I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
      return true;
    }
    if (Const->uge(
            APInt::getBitsSetFrom(Const->getBitWidth(), Log2(ActualAlign)))) {
      I->replaceAllUsesWith(I->getOperand(0));
      return true;
    }
  }
  if (match(I, m_Trunc(m_PtrToIntOrAddr(m_Value(PtrOp))))) {
    Align ActualAlign = Fn(PtrOp, Align(1), Align(1));
    if (Log2(ActualAlign) >= I->getType()->getScalarSizeInBits()) {
      I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
      return true;
    }
  }

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
  if (!II)
    return false;

  // TODO: Handle more memory intrinsics.
  switch (II->getIntrinsicID()) {
  case Intrinsic::masked_load:
  case Intrinsic::masked_store: {
    unsigned PtrOpIdx = II->getIntrinsicID() == Intrinsic::masked_load ? 0 : 1;
    Value *PtrOp = II->getArgOperand(PtrOpIdx);
    Type *Type = II->getIntrinsicID() == Intrinsic::masked_load
                     ? II->getType()
                     : II->getArgOperand(0)->getType();

    Align OldAlign = II->getParamAlign(PtrOpIdx).valueOrOne();
    Align PrefAlign = DL.getPrefTypeAlign(Type);
    Align NewAlign = Fn(PtrOp, OldAlign, PrefAlign);
    if (NewAlign <= OldAlign)
      return false;

    II->addParamAttr(PtrOpIdx,
                     Attribute::getWithAlignment(II->getContext(), NewAlign));
    return true;
  }
  default:
    return false;
  }
}

using ScopedHT =
    ScopedHashTable<Value *, Align, DenseMapInfo<Value *>, BumpPtrAllocator>;
struct AlignmentScope {
  // If BB is nullptr, the BB is processed.
  BasicBlock *BB;
  DomTreeNode::const_iterator Iter;
  DomTreeNode::const_iterator End;
  ScopedHT::ScopeTy Scope;

  AlignmentScope(DomTreeNode *N, ScopedHT &Table)
      : BB(N->getBlock()), Iter(N->begin()), End(N->end()), Scope(Table) {}
};

/// Return true if \p Ptr is (or strips to) a single-index GEP.
static bool hasSingleIndexGEP(Value *Ptr) {
  return isa<GEPOperator>(Ptr) && cast<GEPOperator>(Ptr)->getNumIndices() == 1;
}

/// Quickly compute alignment from common index patterns.
static std::optional<Align> getIndexAlignmentFromPattern(Value *Idx) {
  // Pattern: shl X, N -> alignment of 2^N
  const APInt *ShiftAmt;
  if (match(Idx, m_Shl(m_Value(), m_APInt(ShiftAmt)))) {
    uint64_t Shift = ShiftAmt->getZExtValue();
    if (Shift > 0 && Shift <= 63)
      return Align(1ull << Shift);
  }

  // Pattern: mul X, C where C is a power of 2 -> alignment of C
  const APInt *MulC;
  if (match(Idx, m_Mul(m_Value(), m_APInt(MulC))) ||
      match(Idx, m_Mul(m_APInt(MulC), m_Value()))) {
    uint64_t Val = MulC->getZExtValue();
    if (isPowerOf2_64(Val))
      return Align(Val);
  }

  // Pattern: add X, C -> GCD of X's alignment and C
  Value *AddOp;
  const APInt *AddC;
  if (match(Idx, m_Add(m_Value(AddOp), m_APInt(AddC))) ||
      match(Idx, m_Add(m_APInt(AddC), m_Value(AddOp)))) {
    if (auto XAlign = getIndexAlignmentFromPattern(AddOp))
      return commonAlignment(*XAlign, AddC->getZExtValue());
  }

  // Pattern: sub X, C -> GCD of X's alignment and C
  if (match(Idx, m_Sub(m_Value(AddOp), m_APInt(AddC)))) {
    if (auto XAlign = getIndexAlignmentFromPattern(AddOp))
      return commonAlignment(*XAlign, AddC->getZExtValue());
  }

  // Pattern: sext/zext - extensions preserve alignment
  Value *CastSrc;
  if (match(Idx, m_SExt(m_Value(CastSrc))) ||
      match(Idx, m_ZExt(m_Value(CastSrc))))
    return getIndexAlignmentFromPattern(CastSrc);

  return std::nullopt;
}

bool inferAlignment(Function &F, AssumptionCache &AC, DominatorTree &DT) {
  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;

  // Enforce preferred type alignment if possible. We do this as a separate
  // pass first, because it may improve the alignments we infer below.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Changed |= tryToImproveAlign(
          DL, &I, [&](Value *PtrOp, Align OldAlign, Align PrefAlign) {
            if (PrefAlign > OldAlign)
              return std::max(OldAlign,
                              tryEnforceAlignment(PtrOp, PrefAlign, DL));
            return OldAlign;
          });
    }
  }

  // Compute alignment from known bits.
  auto InferFromKnownBits = [&](Instruction &I, Value *PtrOp) {
    KnownBits Known = computeKnownBits(PtrOp, DL, &AC, &I, &DT);
    unsigned TrailZ =
        std::min(Known.countMinTrailingZeros(), +Value::MaxAlignmentExponent);
    return Align(1ull << std::min(Known.getBitWidth() - 1, TrailZ));
  };

  // Helper function to compute variable offset alignment and base pointer.
  // If ConstOffset > 0, the effective offset alignment is limited by the
  // constant offset.
  auto computeVariableOffsetAlignment =
      [&](Value *Ptr, uint64_t ConstOffset = 0) -> std::pair<Value *, Align> {
    Align VarOffsetAlign = Align(1);
    Value *VarBasePtr = Ptr;

    if (auto *GEP = dyn_cast<GEPOperator>(VarBasePtr)) {
      // We can only handle GEPs with a single index
      if (GEP->getNumIndices() == 1) {
        Value *Idx = GEP->idx_begin()->get();
        Align IndexAlign(1);

        if (auto PatternAlign = getIndexAlignmentFromPattern(Idx))
          IndexAlign = *PatternAlign;

        Type *EltTy = GEP->getSourceElementType();
        TypeSize EltSizeType = DL.getTypeAllocSize(EltTy);

        // If we encounter a scalable type, we can't compute alignment.
        if (!EltSizeType.isScalable()) {
          uint64_t EltSize = EltSizeType.getFixedValue();

          // Compute offset alignment: multiply index alignment by element size,
          // then take the greatest power of 2 that divides the product.
          uint64_t Product = IndexAlign.value() * EltSize;
          uint64_t ProductAlignValue =
              Product > 0 ? (Product & (~Product + 1)) : 1;
          VarOffsetAlign = Align(ProductAlignValue);
        }

        VarBasePtr = GEP->getPointerOperand();
      }
    }
    VarBasePtr = VarBasePtr->stripPointerCasts();

    // If we have a constant offset, the effective alignment is the GCD of both.
    if (ConstOffset > 0)
      VarOffsetAlign = commonAlignment(VarOffsetAlign, ConstOffset);

    return {VarBasePtr, VarOffsetAlign};
  };

  // Propagate alignment between loads and stores that originate from the
  // same base pointer.
  ScopedHT BestBasePointerAligns;

  auto updateBestBaseAlign = [&](Value *BasePtr, Align NewAlign) {
    if (Align Old = BestBasePointerAligns.lookup(BasePtr); Old != Align()) {
      if (NewAlign <= Old)
        return;
    }
    BestBasePointerAligns.insert(BasePtr, NewAlign);
  };

  // Compute final alignment from a base pointer and offset.
  auto computeFinalAlign = [&](Value *BasePtr, Align FallbackAlign,
                               bool UseConstOffset, uint64_t ConstOffset,
                               Align VarOffsetAlign) -> Align {
    Align StoredBaseAlign = BestBasePointerAligns.lookup(BasePtr);
    if (StoredBaseAlign == Align())
      StoredBaseAlign = Align(1);

    Align BaseAlign =
        StoredBaseAlign > Align(1) ? StoredBaseAlign : FallbackAlign;

    if (UseConstOffset)
      return commonAlignment(BaseAlign, ConstOffset);
    return commonAlignment(BaseAlign, VarOffsetAlign.value());
  };

  auto processAlignAssume = [&](Value *AAPtr, Align AssumedAlign) {
    APInt OffsetFromBase(DL.getIndexTypeSizeInBits(AAPtr->getType()), 0);
    Value *ConstBasePtr =
        AAPtr->stripAndAccumulateConstantOffsets(DL, OffsetFromBase, true);
    uint64_t ConstOffsetVal = OffsetFromBase.abs().getLimitedValue();

    Align ConstBaseAlign = computeFinalAlign(ConstBasePtr, AssumedAlign, true,
                                             ConstOffsetVal, Align(1));
    updateBestBaseAlign(ConstBasePtr, ConstBaseAlign);

    if (!hasSingleIndexGEP(ConstBasePtr))
      return;

    auto [VarBasePtr, VarOffsetAlign] =
        computeVariableOffsetAlignment(ConstBasePtr, ConstOffsetVal);
    Align VarBaseAlign =
        computeFinalAlign(VarBasePtr, AssumedAlign, false, 0, VarOffsetAlign);
    if (VarBasePtr != ConstBasePtr)
      updateBestBaseAlign(VarBasePtr, VarBaseAlign);
    else if (VarBaseAlign > ConstBaseAlign)
      updateBestBaseAlign(ConstBasePtr, VarBaseAlign);
  };

  auto InferFromBasePointer = [&](Value *PtrOp, Align LoadStoreAlign) {
    APInt OffsetFromBase(DL.getIndexTypeSizeInBits(PtrOp->getType()), 0);
    Value *ConstBasePtr =
        PtrOp->stripAndAccumulateConstantOffsets(DL, OffsetFromBase, true);
    uint64_t ConstOffsetVal = OffsetFromBase.abs().getLimitedValue();

    Align BasePointerAlign = commonAlignment(LoadStoreAlign, ConstOffsetVal);

    if (Align BestAlign = BestBasePointerAligns.lookup(ConstBasePtr);
        BestAlign != Align()) {
      if (BestAlign > BasePointerAlign) {
        Align BetterLoadStoreAlign = commonAlignment(BestAlign, ConstOffsetVal);
        if (!hasSingleIndexGEP(ConstBasePtr))
          return BetterLoadStoreAlign;
      } else if (BasePointerAlign > BestAlign) {
        BestBasePointerAligns.insert(ConstBasePtr, BasePointerAlign);
      }
    } else {
      BestBasePointerAligns.insert(ConstBasePtr, BasePointerAlign);
    }

    Align ConstFinalAlign = computeFinalAlign(ConstBasePtr, LoadStoreAlign,
                                              true, ConstOffsetVal, Align(1));

    if (!hasSingleIndexGEP(ConstBasePtr))
      return ConstFinalAlign;

    auto [VarBasePtr, VarOffsetAlign] =
        computeVariableOffsetAlignment(ConstBasePtr, ConstOffsetVal);
    Align VarFinalAlign =
        computeFinalAlign(VarBasePtr, LoadStoreAlign, false, 0, VarOffsetAlign);

    return std::max(ConstFinalAlign, VarFinalAlign);
  };

  // AlignmentScope is unmovable.
  std::list<AlignmentScope> Stack;
  Stack.emplace_back(DT.getRootNode(), BestBasePointerAligns);
  while (!Stack.empty()) {
    AlignmentScope &Top = Stack.back();
    if (Top.BB) {
      for (Instruction &I : *Top.BB) {
        if (auto *Assume = dyn_cast<AssumeInst>(&I)) {
          for (unsigned Idx = 0; Idx < Assume->getNumOperandBundles(); ++Idx) {
            OperandBundleUse OB = Assume->getOperandBundleAt(Idx);
            if (OB.getTagID() != LLVMContext::OB_align || OB.Inputs.size() < 2)
              continue;

            auto *AlignVal = dyn_cast<ConstantInt>(OB.Inputs[1].get());
            if (!AlignVal)
              continue;

            uint64_t AlignValue = AlignVal->getZExtValue();
            if (!isPowerOf2_64(AlignValue))
              continue;

            processAlignAssume(OB.Inputs[0].get(), Align(AlignValue));
          }
          continue;
        }

        Changed |= tryToImproveAlign(
            DL, &I, [&](Value *PtrOp, Align OldAlign, Align PrefAlign) {
              return std::max(InferFromKnownBits(I, PtrOp),
                              InferFromBasePointer(PtrOp, OldAlign));
            });
      }
      Top.BB = nullptr;
    }

    if (Top.Iter != Top.End)
      Stack.emplace_back(*Top.Iter++, BestBasePointerAligns);
    else
      Stack.pop_back();
  }

  return Changed;
}

PreservedAnalyses InferAlignmentPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  inferAlignment(F, AC, DT);
  // Changes to alignment shouldn't invalidated analyses.
  return PreservedAnalyses::all();
}
