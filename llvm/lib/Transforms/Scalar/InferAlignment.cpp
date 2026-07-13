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
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"

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

  // Propagate alignment between loads and stores that originate from the
  // same base pointer.
  ScopedHT BestBasePointerAligns;
  auto InferFromBasePointer = [&](Value *PtrOp, Align LoadStoreAlign) {
    APInt OffsetFromBase(DL.getIndexTypeSizeInBits(PtrOp->getType()), 0);
    PtrOp = PtrOp->stripAndAccumulateConstantOffsets(DL, OffsetFromBase, true);
    // Derive the base pointer alignment from the load/store alignment
    // and the offset from the base pointer.
    Align BasePointerAlign =
        commonAlignment(LoadStoreAlign, OffsetFromBase.getLimitedValue());

    if (auto BestAlign = BestBasePointerAligns.lookup(PtrOp);
        BestAlign != Align()) {
      // If the stored base pointer alignment is better than the
      // base pointer alignment we derived, we may be able to use it
      // to improve the load/store alignment. If not, store the
      // improved base pointer alignment for future iterations.
      if (BestAlign > BasePointerAlign) {
        Align BetterLoadStoreAlign =
            commonAlignment(BestAlign, OffsetFromBase.getLimitedValue());
        return BetterLoadStoreAlign;
      }
    }

    BestBasePointerAligns.insert(PtrOp, BasePointerAlign);
    return LoadStoreAlign;
  };

  // AlignmentScope is unmovable.
  std::list<AlignmentScope> Stack;
  Stack.emplace_back(DT.getRootNode(), BestBasePointerAligns);
  while (!Stack.empty()) {
    AlignmentScope &Top = Stack.back();
    if (Top.BB) {
      for (Instruction &I : *Top.BB) {
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
