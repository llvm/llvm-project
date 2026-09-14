//===- SLPMemoryUtils.cpp - SLP pointer/stride helpers --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPMemoryUtils.h"
#include "SLPCompatibilityAnalysis.h"
#include "SLPCostAnalysis.h"
#include "SLPTypeUtils.h"
#include "SLPUtils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/InstructionCost.h"

#include <algorithm>
#include <limits>
#include <optional>
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

/// Builds compress-like mask for shuffles for the given \p PointerOps, ordered
/// with \p Order.
/// \return true if the mask represents strided access, false - otherwise.
static bool buildCompressMask(ArrayRef<Value *> PointerOps,
                              ArrayRef<unsigned> Order, Type *ScalarTy,
                              const DataLayout &DL, ScalarEvolution &SE,
                              SmallVectorImpl<int> &CompressMask) {
  const unsigned Sz = PointerOps.size();
  CompressMask.assign(Sz, PoisonMaskElem);
  // The first element always set.
  CompressMask[0] = 0;
  // Check if the mask represents strided access.
  std::optional<unsigned> Stride = 0;
  Value *Ptr0 = Order.empty() ? PointerOps.front() : PointerOps[Order.front()];
  for (unsigned I : seq<unsigned>(1, Sz)) {
    Value *Ptr = Order.empty() ? PointerOps[I] : PointerOps[Order[I]];
    std::optional<int64_t> OptPos =
        getPointersDiff(ScalarTy, Ptr0, ScalarTy, Ptr, DL, SE);
    if (!OptPos || OptPos > std::numeric_limits<unsigned>::max())
      return false;
    unsigned Pos = static_cast<unsigned>(*OptPos);
    CompressMask[I] = Pos;
    if (!Stride)
      continue;
    if (*Stride == 0) {
      *Stride = Pos;
      continue;
    }
    if (Pos != *Stride * I)
      Stride.reset();
  }
  return Stride.has_value();
}

/// Checks if the \p VL can be transformed to a (masked)load + compress or
/// (masked) interleaved load.
bool isMaskedLoadCompress(
    ArrayRef<Value *> VL, ArrayRef<Value *> PointerOps,
    ArrayRef<unsigned> Order, const TargetTransformInfo &TTI,
    const DataLayout &DL, ScalarEvolution &SE, AssumptionCache &AC,
    const DominatorTree &DT, const TargetLibraryInfo &TLI,
    const TargetTransformInfo::TargetCostKind CostKind,
    const function_ref<bool(Value *)> AreAllUsersVectorized, bool ReVec,
    bool &IsMasked, unsigned &InterleaveFactor,
    SmallVectorImpl<int> &CompressMask, VectorType *&LoadVecTy) {
  InterleaveFactor = 0;
  Type *ScalarTy = VL.front()->getType();
  const size_t Sz = VL.size();
  auto *VecTy = cast<VectorType>(getWidenedType(ScalarTy, Sz));
  SmallVector<int> Mask;
  if (!Order.empty())
    inversePermutation(Order, Mask);
  // Check external uses.
  for (const auto [I, V] : enumerate(VL)) {
    if (AreAllUsersVectorized(V))
      continue;
    InstructionCost ExtractCost =
        TTI.getVectorInstrCost(Instruction::ExtractElement, VecTy, CostKind,
                               Mask.empty() ? I : Mask[I]);
    InstructionCost ScalarCost =
        TTI.getInstructionCost(cast<Instruction>(V), CostKind);
    if (ExtractCost <= ScalarCost)
      return false;
  }
  Value *Ptr0;
  Value *PtrN;
  if (Order.empty()) {
    Ptr0 = PointerOps.front();
    PtrN = PointerOps.back();
  } else {
    Ptr0 = PointerOps[Order.front()];
    PtrN = PointerOps[Order.back()];
  }
  std::optional<int64_t> Diff =
      getPointersDiff(ScalarTy, Ptr0, ScalarTy, PtrN, DL, SE);
  if (!Diff)
    return false;
  const size_t MaxRegSize =
      TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
          .getFixedValue();
  // Check for very large distances between elements.
  if (*Diff / Sz >= MaxRegSize / 8)
    return false;
  LoadVecTy = cast<FixedVectorType>(getWidenedType(ScalarTy, *Diff + 1));
  auto *LI = cast<LoadInst>(Order.empty() ? VL.front() : VL[Order.front()]);
  Align CommonAlignment = LI->getAlign();
  SimplifyQuery SQ(
      DL, &TLI, &DT, &AC,
      cast<LoadInst>(Order.empty() ? VL.back() : VL[Order.back()]));
  IsMasked = !isSafeToLoadUnconditionally(Ptr0, LoadVecTy, CommonAlignment, SQ);
  if (IsMasked && !TTI.isLegalMaskedLoad(LoadVecTy, CommonAlignment,
                                         LI->getPointerAddressSpace()))
    return false;
  // TODO: perform the analysis of each scalar load for better
  // safe-load-unconditionally analysis.
  bool IsStrided =
      buildCompressMask(PointerOps, Order, ScalarTy, DL, SE, CompressMask);
  assert(CompressMask.size() >= 2 && "At least two elements are required");
  SmallVector<Value *> OrderedPointerOps(PointerOps);
  if (!Order.empty())
    reorderScalars(OrderedPointerOps, Mask);
  auto [ScalarGEPCost, VectorGEPCost] =
      getGEPCosts(TTI, OrderedPointerOps, OrderedPointerOps.front(),
                  Instruction::Load, CostKind, ScalarTy, LoadVecTy);
  // The cost of scalar loads.
  InstructionCost ScalarLoadsCost =
      accumulate(VL, InstructionCost(),
                 [&](InstructionCost C, Value *V) {
                   return C + TTI.getInstructionCost(cast<Instruction>(V),
                                                     CostKind);
                 }) +
      ScalarGEPCost;
  APInt DemandedElts = APInt::getAllOnes(Sz);
  InstructionCost GatherCost =
      getScalarizationOverhead(TTI, ReVec, ScalarTy, VecTy, DemandedElts,
                               /*Insert=*/true,
                               /*Extract=*/false, CostKind) +
      ScalarLoadsCost;
  InstructionCost LoadCost = 0;
  if (IsMasked) {
    LoadCost = TTI.getMemIntrinsicInstrCost(
        MemIntrinsicCostAttributes(Intrinsic::masked_load, LoadVecTy,
                                   CommonAlignment,
                                   LI->getPointerAddressSpace()),
        CostKind);
  } else {
    LoadCost =
        TTI.getMemoryOpCost(Instruction::Load, LoadVecTy, CommonAlignment,
                            LI->getPointerAddressSpace(), CostKind);
  }
  if (IsStrided && !IsMasked && Order.empty()) {
    // Check for potential segmented(interleaved) loads.
    VectorType *AlignedLoadVecTy = cast<VectorType>(getWidenedType(
        ScalarTy,
        getFullVectorNumberOfElements(TTI, ScalarTy, *Diff + 1, ReVec)));
    SimplifyQuery SQ(DL, &TLI, &DT, &AC, cast<LoadInst>(VL.back()));
    if (!isSafeToLoadUnconditionally(Ptr0, AlignedLoadVecTy, CommonAlignment,
                                     SQ))
      AlignedLoadVecTy = LoadVecTy;
    if (TTI.isLegalInterleavedAccessType(AlignedLoadVecTy, CompressMask[1],
                                         CommonAlignment,
                                         LI->getPointerAddressSpace())) {
      InstructionCost InterleavedCost =
          VectorGEPCost + TTI.getInterleavedMemoryOpCost(
                              Instruction::Load, AlignedLoadVecTy,
                              CompressMask[1], {}, CommonAlignment,
                              LI->getPointerAddressSpace(), CostKind, IsMasked);
      if (InterleavedCost < GatherCost) {
        InterleaveFactor = CompressMask[1];
        LoadVecTy = AlignedLoadVecTy;
        return true;
      }
    }
  }
  // Estimating the compression shuffle cost below can be extremely expensive
  // for a very wide LoadVecTy, which is split into a large number of vector
  // registers (see processShuffleMasks). The shuffle cost is always
  // non-negative, so if the load cost alone already reaches the gather cost the
  // masked-load-compress cannot be profitable. Bail out before the costly
  // shuffle cost estimation in that case.
  if (VectorGEPCost + LoadCost >= GatherCost)
    return false;
  InstructionCost CompressCost = getShuffleCost(
      TTI, TTI::SK_PermuteSingleSrc, LoadVecTy, CostKind, CompressMask);
  if (!Order.empty()) {
    SmallVector<int> NewMask(Sz, PoisonMaskElem);
    for (unsigned I : seq<unsigned>(Sz)) {
      NewMask[I] = CompressMask[Mask[I]];
    }
    CompressMask.swap(NewMask);
  }
  InstructionCost TotalVecCost = VectorGEPCost + LoadCost + CompressCost;
  return TotalVecCost < GatherCost;
}

/// Checks if the \p VL can be transformed to a (masked)load + compress or
/// (masked) interleaved load.
bool isMaskedLoadCompress(
    ArrayRef<Value *> VL, ArrayRef<Value *> PointerOps,
    ArrayRef<unsigned> Order, const TargetTransformInfo &TTI,
    const DataLayout &DL, ScalarEvolution &SE, AssumptionCache &AC,
    const DominatorTree &DT, const TargetLibraryInfo &TLI,
    const TargetTransformInfo::TargetCostKind CostKind,
    const function_ref<bool(Value *)> AreAllUsersVectorized, bool ReVec) {
  bool IsMasked;
  unsigned InterleaveFactor;
  SmallVector<int> CompressMask;
  VectorType *LoadVecTy;
  return isMaskedLoadCompress(VL, PointerOps, Order, TTI, DL, SE, AC, DT, TLI,
                              CostKind, AreAllUsersVectorized, ReVec, IsMasked,
                              InterleaveFactor, CompressMask, LoadVecTy);
}

/// Checks if the stores \p VL with pointers \p PointerOps can be lowered as a
/// single masked store. On success \p StoreVecTy is the widened store type and
/// \p ReuseShuffleIndices is the expand mask that places each stored value at
/// its element offset from the base (poison in the gaps).
bool isMaskedStoreCompress(ArrayRef<Value *> VL, ArrayRef<Value *> PointerOps,
                           ArrayRef<unsigned> Order,
                           const TargetTransformInfo &TTI, const DataLayout &DL,
                           ScalarEvolution &SE, Align CommonAlignment,
                           SmallVectorImpl<int> &ReuseShuffleIndices,
                           FixedVectorType *&StoreVecTy) {
  Type *ScalarTy = cast<StoreInst>(VL.front())->getValueOperand()->getType();
  const size_t Sz = VL.size();
  // Only simple scalar element types are supported.
  if (Sz < 2 || (!ScalarTy->isIntOrPtrTy() && !ScalarTy->isFloatingPointTy()))
    return false;
  Value *Ptr0 = Order.empty() ? PointerOps.front() : PointerOps[Order.front()];
  Value *PtrN = Order.empty() ? PointerOps.back() : PointerOps[Order.back()];
  std::optional<int64_t> Diff =
      getPointersDiff(ScalarTy, Ptr0, ScalarTy, PtrN, DL, SE);
  if (!Diff || *Diff <= 0)
    return false;
  // Avoid widened vectors with very large gaps between the stored elements.
  const unsigned MaxRegSize =
      TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
          .getFixedValue();
  const unsigned ScalarBits = DL.getTypeSizeInBits(ScalarTy).getFixedValue();
  if (ScalarBits == 0 ||
      static_cast<uint64_t>(*Diff) / Sz >= MaxRegSize / ScalarBits)
    return false;
  StoreVecTy = cast<FixedVectorType>(getWidenedType(ScalarTy, *Diff + 1));
  unsigned AS = cast<StoreInst>(VL.front())->getPointerAddressSpace();
  if (!TTI.isLegalMaskedStore(StoreVecTy, CommonAlignment, AS,
                              TTI::ConstantMask))
    return false;
  // Build the expand mask: store I (in address-sorted order) is placed at its
  // element offset from the base, other widened lanes are poison.
  ReuseShuffleIndices.assign(*Diff + 1, PoisonMaskElem);
  int64_t Prev = -1;
  for (unsigned I : seq<unsigned>(Sz)) {
    Value *Ptr = Order.empty() ? PointerOps[I] : PointerOps[Order[I]];
    std::optional<int64_t> Off =
        getPointersDiff(ScalarTy, Ptr0, ScalarTy, Ptr, DL, SE);
    if (!Off || *Off <= Prev || *Off > *Diff)
      return false;
    ReuseShuffleIndices[*Off] = static_cast<int>(I);
    Prev = *Off;
  }
  return true;
}

} // namespace llvm::slpvectorizer
