//===- SLPMemoryUtils.cpp - SLP pointer/stride helpers --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPMemoryUtils.h"
#include "SLPCompatibilityAnalysis.h"
#include "SLPUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"

#include <algorithm>
#include <set>
#include <utility>

using namespace llvm;

namespace llvm::slpvectorizer {

bool arePointersCompatible(Value *Ptr1, Value *Ptr2,
                           const TargetLibraryInfo &TLI, unsigned MaxDepth,
                           bool CompareOpcodes) {
  if (getUnderlyingObject(Ptr1, MaxDepth) !=
      getUnderlyingObject(Ptr2, MaxDepth))
    return false;
  auto *GEP1 = dyn_cast<GetElementPtrInst>(Ptr1);
  auto *GEP2 = dyn_cast<GetElementPtrInst>(Ptr2);
  return (!GEP1 || GEP1->getNumOperands() == 2) &&
         (!GEP2 || GEP2->getNumOperands() == 2) &&
         (((!GEP1 || isConstant(GEP1->getOperand(1))) &&
           (!GEP2 || isConstant(GEP2->getOperand(1)))) ||
          !CompareOpcodes ||
          (GEP1 && GEP2 &&
           getSameOpcode({GEP1->getOperand(1), GEP2->getOperand(1)}, TLI)));
}

/// Calculates minimal alignment as a common alignment.
template <typename T> Align computeCommonAlignment(ArrayRef<Value *> VL) {
  Align CommonAlignment = cast<T>(VL.consume_front())->getAlign();
  for (Value *V : VL)
    CommonAlignment = std::min(CommonAlignment, cast<T>(V)->getAlign());
  return CommonAlignment;
}

template Align computeCommonAlignment<LoadInst>(ArrayRef<Value *>);
template Align computeCommonAlignment<StoreInst>(ArrayRef<Value *>);

const SCEV *calculateRtStride(ArrayRef<Value *> PointerOps, Type *ElemTy,
                              const DataLayout &DL, ScalarEvolution &SE,
                              SmallVectorImpl<unsigned> &SortedIndices) {
  SmallVector<const SCEV *> SCEVs;
  const SCEV *PtrSCEVLowest = nullptr;
  const SCEV *PtrSCEVHighest = nullptr;
  // Find lower/upper pointers from the PointerOps (i.e. with lowest and highest
  // addresses).
  for (Value *Ptr : PointerOps) {
    const SCEV *PtrSCEV = SE.getSCEV(Ptr);
    if (!PtrSCEV)
      return nullptr;
    SCEVs.push_back(PtrSCEV);
    if (!PtrSCEVLowest && !PtrSCEVHighest) {
      PtrSCEVLowest = PtrSCEVHighest = PtrSCEV;
      continue;
    }
    const SCEV *Diff = SE.getMinusSCEV(PtrSCEV, PtrSCEVLowest);
    if (isa<SCEVCouldNotCompute>(Diff))
      return nullptr;
    if (Diff->isNonConstantNegative()) {
      PtrSCEVLowest = PtrSCEV;
      continue;
    }
    const SCEV *Diff1 = SE.getMinusSCEV(PtrSCEVHighest, PtrSCEV);
    if (isa<SCEVCouldNotCompute>(Diff1))
      return nullptr;
    if (Diff1->isNonConstantNegative()) {
      PtrSCEVHighest = PtrSCEV;
      continue;
    }
  }
  // Dist = PtrSCEVHighest - PtrSCEVLowest;
  const SCEV *Dist = SE.getMinusSCEV(PtrSCEVHighest, PtrSCEVLowest);
  if (isa<SCEVCouldNotCompute>(Dist))
    return nullptr;
  int Size = DL.getTypeStoreSize(ElemTy);
  auto TryGetStride = [&](const SCEV *Dist,
                          const SCEV *Multiplier) -> const SCEV * {
    if (const auto *M = dyn_cast<SCEVMulExpr>(Dist)) {
      if (M->getOperand(0) == Multiplier)
        return M->getOperand(1);
      if (M->getOperand(1) == Multiplier)
        return M->getOperand(0);
      return nullptr;
    }
    if (Multiplier == Dist)
      return SE.getConstant(Dist->getType(), 1);
    return SE.getUDivExactExpr(Dist, Multiplier);
  };
  // Stride_in_elements = Dist / element_size * (num_elems - 1).
  const SCEV *Stride = nullptr;
  if (Size != 1 || SCEVs.size() > 1) {
    const SCEV *Sz = SE.getConstant(Dist->getType(), Size * (SCEVs.size() - 1));
    Stride = TryGetStride(Dist, Sz);
    if (!Stride)
      return nullptr;
  }
  if (!Stride || isa<SCEVConstant>(Stride))
    return nullptr;
  // Iterate through all pointers and check if all distances are
  // unique multiple of Stride.
  using DistOrdPair = std::pair<int64_t, int>;
  auto Compare = llvm::less_first();
  std::set<DistOrdPair, decltype(Compare)> Offsets(Compare);
  bool IsConsecutive = true;
  for (const auto [Idx, PtrSCEV] : enumerate(SCEVs)) {
    unsigned Dist = 0;
    if (PtrSCEV != PtrSCEVLowest) {
      const SCEV *Diff = SE.getMinusSCEV(PtrSCEV, PtrSCEVLowest);
      const SCEV *Coeff = TryGetStride(Diff, Stride);
      if (!Coeff)
        return nullptr;
      const auto *SC = dyn_cast<SCEVConstant>(Coeff);
      if (!SC || isa<SCEVCouldNotCompute>(SC))
        return nullptr;
      if (!SE.getMinusSCEV(PtrSCEV, SE.getAddExpr(PtrSCEVLowest,
                                                  SE.getMulExpr(Stride, SC)))
               ->isZero())
        return nullptr;
      Dist = SC->getAPInt().getZExtValue();
    }
    // If the strides are not the same or repeated, we can't vectorize.
    if ((Dist / Size) * Size != Dist || (Dist / Size) >= SCEVs.size())
      return nullptr;
    auto Res = Offsets.emplace(Dist, Idx);
    if (!Res.second)
      return nullptr;
    // Consecutive order if the inserted element is the last one.
    IsConsecutive = IsConsecutive && std::next(Res.first) == Offsets.end();
  }
  SortedIndices.clear();
  if (!IsConsecutive) {
    // Fill SortedIndices array only if it is non-consecutive.
    SortedIndices.resize(PointerOps.size());
    for (const auto [Idx, Pair] : enumerate(Offsets))
      SortedIndices[Idx] = Pair.second;
  }
  return Stride;
}

} // namespace llvm::slpvectorizer
