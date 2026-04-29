//===- LoopVectorizationPlanner.cpp - VF selection and planning -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements VFSelectionContext methods for loop vectorization
/// VF selection, independent of cost-modeling decisions.
///
//===----------------------------------------------------------------------===//

#include "LoopVectorizationPlanner.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"

using namespace llvm;

#define DEBUG_TYPE "loop-vectorize"

static cl::opt<bool> MaximizeBandwidth(
    "vectorizer-maximize-bandwidth", cl::init(false), cl::Hidden,
    cl::desc("Maximize bandwidth when selecting vectorization factor which "
             "will be determined by the smallest type in loop."));

static cl::opt<bool> UseWiderVFIfCallVariantsPresent(
    "vectorizer-maximize-bandwidth-for-vector-calls", cl::init(true),
    cl::Hidden,
    cl::desc("Try wider VFs if they enable the use of vector variants"));

static cl::opt<bool> ConsiderRegPressure(
    "vectorizer-consider-reg-pressure", cl::init(false), cl::Hidden,
    cl::desc("Discard VFs if their register pressure is too high."));

static cl::opt<bool> ForceTargetSupportsScalableVectors(
    "force-target-supports-scalable-vectors", cl::init(false), cl::Hidden,
    cl::desc(
        "Pretend that scalable vectors are supported, even if the target does "
        "not support them. This flag should only be used for testing."));

cl::opt<bool> llvm::PreferInLoopReductions(
    "prefer-inloop-reductions", cl::init(false), cl::Hidden,
    cl::desc("Prefer in-loop vector reductions, "
             "overriding the targets preference."));

/// Note: This currently only applies to `llvm.masked.load` and
/// `llvm.masked.store`. TODO: Extend this to cover other operations as needed.
static cl::opt<bool> ForceTargetSupportsMaskedMemoryOps(
    "force-target-supports-masked-memory-ops", cl::init(false), cl::Hidden,
    cl::desc("Assume the target supports masked memory operations (used for "
             "testing)."));

bool VFSelectionContext::isLegalMaskedStore(Type *DataType, Value *Ptr,
                                            Align Alignment,
                                            unsigned AddressSpace) const {
  return Legal->isConsecutivePtr(DataType, Ptr) &&
         (ForceTargetSupportsMaskedMemoryOps ||
          TTI.isLegalMaskedStore(DataType, Alignment, AddressSpace));
}

bool VFSelectionContext::isLegalMaskedLoad(Type *DataType, Value *Ptr,
                                           Align Alignment,
                                           unsigned AddressSpace) const {
  return Legal->isConsecutivePtr(DataType, Ptr) &&
         (ForceTargetSupportsMaskedMemoryOps ||
          TTI.isLegalMaskedLoad(DataType, Alignment, AddressSpace));
}

bool VFSelectionContext::isLegalGatherOrScatter(Value *V,
                                                ElementCount VF) const {
  bool LI = isa<LoadInst>(V);
  bool SI = isa<StoreInst>(V);
  if (!LI && !SI)
    return false;
  auto *Ty = getLoadStoreType(V);
  Align Align = getLoadStoreAlignment(V);
  if (VF.isVector())
    Ty = VectorType::get(Ty, VF);
  return (LI && TTI.isLegalMaskedGather(Ty, Align)) ||
         (SI && TTI.isLegalMaskedScatter(Ty, Align));
}

bool VFSelectionContext::supportsScalableVectors() const {
  return TTI.supportsScalableVectors() || ForceTargetSupportsScalableVectors;
}

bool VFSelectionContext::useMaxBandwidth(bool IsScalable) const {
  TargetTransformInfo::RegisterKind RegKind =
      IsScalable ? TargetTransformInfo::RGK_ScalableVector
                 : TargetTransformInfo::RGK_FixedWidthVector;
  return MaximizeBandwidth || (MaximizeBandwidth.getNumOccurrences() == 0 &&
                               (TTI.shouldMaximizeVectorBandwidth(RegKind) ||
                                (UseWiderVFIfCallVariantsPresent &&
                                 Legal->hasVectorCallVariants())));
}

bool VFSelectionContext::shouldConsiderRegPressureForVF(ElementCount VF) const {
  if (ConsiderRegPressure.getNumOccurrences())
    return ConsiderRegPressure;

  // TODO: We should eventually consider register pressure for all targets. The
  // TTI hook is temporary whilst target-specific issues are being fixed.
  if (TTI.shouldConsiderVectorizationRegPressure())
    return true;

  if (!useMaxBandwidth(VF.isScalable()))
    return false;
  // Only calculate register pressure for VFs enabled by MaxBandwidth.
  return ElementCount::isKnownGT(
      VF, VF.isScalable() ? MaxPermissibleVFWithoutMaxBW.ScalableVF
                          : MaxPermissibleVFWithoutMaxBW.FixedVF);
}

ElementCount VFSelectionContext::clampVFByMaxTripCount(
    ElementCount VF, unsigned MaxTripCount, unsigned UserIC,
    bool FoldTailByMasking, bool RequiresScalarEpilogue) const {
  unsigned EstimatedVF = VF.getKnownMinValue();
  if (VF.isScalable() && F.hasFnAttribute(Attribute::VScaleRange)) {
    auto Attr = F.getFnAttribute(Attribute::VScaleRange);
    auto Min = Attr.getVScaleRangeMin();
    EstimatedVF *= Min;
  }

  // When a scalar epilogue is required, at least one iteration of the scalar
  // loop has to execute. Adjust MaxTripCount accordingly to avoid picking a
  // max VF that results in a dead vector loop.
  if (MaxTripCount > 0 && RequiresScalarEpilogue)
    MaxTripCount -= 1;

  // When the user specifies an interleave count, we need to ensure that
  // VF * UserIC <= MaxTripCount to avoid a dead vector loop.
  unsigned IC = UserIC > 0 ? UserIC : 1;
  unsigned EstimatedVFTimesIC = EstimatedVF * IC;

  if (MaxTripCount && MaxTripCount <= EstimatedVFTimesIC &&
      (!FoldTailByMasking || isPowerOf2_32(MaxTripCount))) {
    // If upper bound loop trip count (TC) is known at compile time there is no
    // point in choosing VF greater than TC / IC (as done in the loop below).
    // Select maximum power of two which doesn't exceed TC / IC. If VF is
    // scalable, we only fall back on a fixed VF when the TC is less than or
    // equal to the known number of lanes.
    auto ClampedUpperTripCount = llvm::bit_floor(MaxTripCount / IC);
    if (ClampedUpperTripCount == 0)
      ClampedUpperTripCount = 1;
    LLVM_DEBUG(dbgs() << "LV: Clamping the MaxVF to maximum power of two not "
                         "exceeding the constant trip count"
                      << (UserIC > 0 ? " divided by UserIC" : "") << ": "
                      << ClampedUpperTripCount << "\n");
    return ElementCount::get(ClampedUpperTripCount,
                             FoldTailByMasking ? VF.isScalable() : false);
  }
  return VF;
}

ElementCount VFSelectionContext::getMaximizedVFForTarget(
    unsigned MaxTripCount, unsigned SmallestType, unsigned WidestType,
    ElementCount MaxSafeVF, unsigned UserIC, bool FoldTailByMasking,
    bool RequiresScalarEpilogue) {
  bool ComputeScalableMaxVF = MaxSafeVF.isScalable();
  const TypeSize WidestRegister = TTI.getRegisterBitWidth(
      ComputeScalableMaxVF ? TargetTransformInfo::RGK_ScalableVector
                           : TargetTransformInfo::RGK_FixedWidthVector);

  // Convenience function to return the minimum of two ElementCounts.
  auto MinVF = [](const ElementCount &LHS, const ElementCount &RHS) {
    assert((LHS.isScalable() == RHS.isScalable()) &&
           "Scalable flags must match");
    return ElementCount::isKnownLT(LHS, RHS) ? LHS : RHS;
  };

  // Ensure MaxVF is a power of 2; the dependence distance bound may not be.
  // Note that both WidestRegister and WidestType may not be a powers of 2.
  auto MaxVectorElementCount = ElementCount::get(
      llvm::bit_floor(WidestRegister.getKnownMinValue() / WidestType),
      ComputeScalableMaxVF);
  MaxVectorElementCount = MinVF(MaxVectorElementCount, MaxSafeVF);
  LLVM_DEBUG(dbgs() << "LV: The Widest register safe to use is: "
                    << (MaxVectorElementCount * WidestType) << " bits.\n");

  if (!MaxVectorElementCount) {
    LLVM_DEBUG(dbgs() << "LV: The target has no "
                      << (ComputeScalableMaxVF ? "scalable" : "fixed")
                      << " vector registers.\n");
    return ElementCount::getFixed(1);
  }

  ElementCount MaxVF =
      clampVFByMaxTripCount(MaxVectorElementCount, MaxTripCount, UserIC,
                            FoldTailByMasking, RequiresScalarEpilogue);
  // If the MaxVF was already clamped, there's no point in trying to pick a
  // larger one.
  if (MaxVF != MaxVectorElementCount)
    return MaxVF;

  if (MaxVF.isScalable())
    MaxPermissibleVFWithoutMaxBW.ScalableVF = MaxVF;
  else
    MaxPermissibleVFWithoutMaxBW.FixedVF = MaxVF;

  if (useMaxBandwidth(ComputeScalableMaxVF)) {
    auto MaxVectorElementCountMaxBW = ElementCount::get(
        llvm::bit_floor(WidestRegister.getKnownMinValue() / SmallestType),
        ComputeScalableMaxVF);
    MaxVF = MinVF(MaxVectorElementCountMaxBW, MaxSafeVF);

    if (ElementCount MinVF =
            TTI.getMinimumVF(SmallestType, ComputeScalableMaxVF)) {
      if (ElementCount::isKnownLT(MaxVF, MinVF)) {
        LLVM_DEBUG(dbgs() << "LV: Overriding calculated MaxVF(" << MaxVF
                          << ") with target's minimum: " << MinVF << '\n');
        MaxVF = MinVF;
      }
    }

    MaxVF = clampVFByMaxTripCount(MaxVF, MaxTripCount, UserIC,
                                  FoldTailByMasking, RequiresScalarEpilogue);
  }
  return MaxVF;
}

std::optional<unsigned> llvm::getMaxVScale(const Function &F,
                                           const TargetTransformInfo &TTI) {
  if (std::optional<unsigned> MaxVScale = TTI.getMaxVScale())
    return MaxVScale;

  if (F.hasFnAttribute(Attribute::VScaleRange))
    return F.getFnAttribute(Attribute::VScaleRange).getVScaleRangeMax();

  return std::nullopt;
}

bool VFSelectionContext::isScalableVectorizationAllowed() {
  if (IsScalableVectorizationAllowed)
    return *IsScalableVectorizationAllowed;

  IsScalableVectorizationAllowed = false;
  if (!supportsScalableVectors())
    return false;

  if (Hints->isScalableVectorizationDisabled()) {
    reportVectorizationInfo("Scalable vectorization is explicitly disabled",
                            "ScalableVectorizationDisabled", ORE, TheLoop);
    return false;
  }

  LLVM_DEBUG(dbgs() << "LV: Scalable vectorization is available\n");

  auto MaxScalableVF = ElementCount::getScalable(
      std::numeric_limits<ElementCount::ScalarTy>::max());

  // Test that the loop-vectorizer can legalize all operations for this MaxVF.
  // FIXME: While for scalable vectors this is currently sufficient, this should
  // be replaced by a more detailed mechanism that filters out specific VFs,
  // instead of invalidating vectorization for a whole set of VFs based on the
  // MaxVF.

  // Disable scalable vectorization if the loop contains unsupported reductions.
  if (!all_of(Legal->getReductionVars(), [&](const auto &Reduction) -> bool {
        return TTI.isLegalToVectorizeReduction(Reduction.second, MaxScalableVF);
      })) {
    reportVectorizationInfo(
        "Scalable vectorization not supported for the reduction "
        "operations found in this loop.",
        "ScalableVFUnfeasible", ORE, TheLoop);
    return false;
  }

  // Disable scalable vectorization if the loop contains any instructions
  // with element types not supported for scalable vectors.
  if (any_of(ElementTypesInLoop, [&](Type *Ty) {
        return !Ty->isVoidTy() && !TTI.isElementTypeLegalForScalableVector(Ty);
      })) {
    reportVectorizationInfo("Scalable vectorization is not supported "
                            "for all element types found in this loop.",
                            "ScalableVFUnfeasible", ORE, TheLoop);
    return false;
  }

  if (!Legal->isSafeForAnyVectorWidth() && !getMaxVScale(F, TTI)) {
    reportVectorizationInfo("The target does not provide maximum vscale value "
                            "for safe distance analysis.",
                            "ScalableVFUnfeasible", ORE, TheLoop);
    return false;
  }

  IsScalableVectorizationAllowed = true;
  return true;
}

ElementCount
VFSelectionContext::getMaxLegalScalableVF(unsigned MaxSafeElements) {
  if (!isScalableVectorizationAllowed())
    return ElementCount::getScalable(0);

  auto MaxScalableVF = ElementCount::getScalable(
      std::numeric_limits<ElementCount::ScalarTy>::max());
  if (Legal->isSafeForAnyVectorWidth())
    return MaxScalableVF;

  std::optional<unsigned> MaxVScale = getMaxVScale(F, TTI);
  // Limit MaxScalableVF by the maximum safe dependence distance.
  MaxScalableVF = ElementCount::getScalable(MaxSafeElements / *MaxVScale);

  if (!MaxScalableVF)
    reportVectorizationInfo(
        "Max legal vector width too small, scalable vectorization "
        "unfeasible.",
        "ScalableVFUnfeasible", ORE, TheLoop);

  return MaxScalableVF;
}

FixedScalableVFPair VFSelectionContext::computeFeasibleMaxVF(
    unsigned MaxTripCount, ElementCount UserVF, unsigned UserIC,
    bool FoldTailByMasking, bool RequiresScalarEpilogue) {
  auto [SmallestType, WidestType] = getSmallestAndWidestTypes();

  // Get the maximum safe dependence distance in bits computed by LAA.
  // It is computed by MaxVF * sizeOf(type) * 8, where type is taken from
  // the memory accesses that is most restrictive (involved in the smallest
  // dependence distance).
  unsigned MaxSafeElementsPowerOf2 =
      llvm::bit_floor(Legal->getMaxSafeVectorWidthInBits() / WidestType);
  if (!Legal->isSafeForAnyStoreLoadForwardDistances()) {
    unsigned SLDist = Legal->getMaxStoreLoadForwardSafeDistanceInBits();
    MaxSafeElementsPowerOf2 =
        std::min(MaxSafeElementsPowerOf2, SLDist / WidestType);
  }

  auto MaxSafeFixedVF = ElementCount::getFixed(MaxSafeElementsPowerOf2);
  auto MaxSafeScalableVF = getMaxLegalScalableVF(MaxSafeElementsPowerOf2);

  if (!Legal->isSafeForAnyVectorWidth())
    MaxSafeElements = MaxSafeElementsPowerOf2;

  LLVM_DEBUG(dbgs() << "LV: The max safe fixed VF is: " << MaxSafeFixedVF
                    << ".\n");
  LLVM_DEBUG(dbgs() << "LV: The max safe scalable VF is: " << MaxSafeScalableVF
                    << ".\n");

  // First analyze the UserVF, fall back if the UserVF should be ignored.
  if (UserVF) {
    auto MaxSafeUserVF =
        UserVF.isScalable() ? MaxSafeScalableVF : MaxSafeFixedVF;

    if (ElementCount::isKnownLE(UserVF, MaxSafeUserVF)) {
      // If `VF=vscale x N` is safe, then so is `VF=N`
      if (UserVF.isScalable())
        return FixedScalableVFPair(
            ElementCount::getFixed(UserVF.getKnownMinValue()), UserVF);

      return UserVF;
    }

    assert(ElementCount::isKnownGT(UserVF, MaxSafeUserVF));

    // Only clamp if the UserVF is not scalable. If the UserVF is scalable, it
    // is better to ignore the hint and let the compiler choose a suitable VF.
    if (!UserVF.isScalable()) {
      LLVM_DEBUG(dbgs() << "LV: User VF=" << UserVF
                        << " is unsafe, clamping to max safe VF="
                        << MaxSafeFixedVF << ".\n");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                          TheLoop->getStartLoc(),
                                          TheLoop->getHeader())
               << "User-specified vectorization factor "
               << ore::NV("UserVectorizationFactor", UserVF)
               << " is unsafe, clamping to maximum safe vectorization factor "
               << ore::NV("VectorizationFactor", MaxSafeFixedVF);
      });
      return MaxSafeFixedVF;
    }

    if (!supportsScalableVectors()) {
      LLVM_DEBUG(dbgs() << "LV: User VF=" << UserVF
                        << " is ignored because scalable vectors are not "
                           "available.\n");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                          TheLoop->getStartLoc(),
                                          TheLoop->getHeader())
               << "User-specified vectorization factor "
               << ore::NV("UserVectorizationFactor", UserVF)
               << " is ignored because the target does not support scalable "
                  "vectors. The compiler will pick a more suitable value.";
      });
    } else {
      LLVM_DEBUG(dbgs() << "LV: User VF=" << UserVF
                        << " is unsafe. Ignoring scalable UserVF.\n");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                          TheLoop->getStartLoc(),
                                          TheLoop->getHeader())
               << "User-specified vectorization factor "
               << ore::NV("UserVectorizationFactor", UserVF)
               << " is unsafe. Ignoring the hint to let the compiler pick a "
                  "more suitable value.";
      });
    }
  }

  LLVM_DEBUG(dbgs() << "LV: The Smallest and Widest types: " << SmallestType
                    << " / " << WidestType << " bits.\n");

  FixedScalableVFPair Result(ElementCount::getFixed(1),
                             ElementCount::getScalable(0));
  if (auto MaxVF = getMaximizedVFForTarget(
          MaxTripCount, SmallestType, WidestType, MaxSafeFixedVF, UserIC,
          FoldTailByMasking, RequiresScalarEpilogue))
    Result.FixedVF = MaxVF;

  if (auto MaxVF = getMaximizedVFForTarget(
          MaxTripCount, SmallestType, WidestType, MaxSafeScalableVF, UserIC,
          FoldTailByMasking, RequiresScalarEpilogue))
    if (MaxVF.isScalable()) {
      Result.ScalableVF = MaxVF;
      LLVM_DEBUG(dbgs() << "LV: Found feasible scalable VF = " << MaxVF
                        << "\n");
    }

  return Result;
}

std::pair<unsigned, unsigned>
VFSelectionContext::getSmallestAndWidestTypes() const {
  unsigned MinWidth = -1U;
  unsigned MaxWidth = 8;
  const DataLayout &DL = F.getDataLayout();
  // For in-loop reductions, no element types are added to ElementTypesInLoop
  // if there are no loads/stores in the loop. In this case, check through the
  // reduction variables to determine the maximum width.
  if (ElementTypesInLoop.empty() && !Legal->getReductionVars().empty()) {
    for (const auto &[_, RdxDesc] : Legal->getReductionVars()) {
      // When finding the min width used by the recurrence we need to account
      // for casts on the input operands of the recurrence.
      MinWidth = std::min(
          MinWidth,
          std::min(RdxDesc.getMinWidthCastToRecurrenceTypeInBits(),
                   RdxDesc.getRecurrenceType()->getScalarSizeInBits()));
      MaxWidth = std::max(MaxWidth,
                          RdxDesc.getRecurrenceType()->getScalarSizeInBits());
    }
  } else {
    for (Type *T : ElementTypesInLoop) {
      MinWidth = std::min<unsigned>(
          MinWidth, DL.getTypeSizeInBits(T->getScalarType()).getFixedValue());
      MaxWidth = std::max<unsigned>(
          MaxWidth, DL.getTypeSizeInBits(T->getScalarType()).getFixedValue());
    }
  }
  return {MinWidth, MaxWidth};
}

void VFSelectionContext::collectElementTypesForWidening(
    const SmallPtrSetImpl<const Value *> *ValuesToIgnore) {
  ElementTypesInLoop.clear();
  // For each block.
  for (BasicBlock *BB : TheLoop->blocks()) {
    // For each instruction in the loop.
    for (Instruction &I : *BB) {
      Type *T = I.getType();

      // Skip ignored values.
      if (ValuesToIgnore && ValuesToIgnore->contains(&I))
        continue;

      // Only examine Loads, Stores and PHINodes.
      if (!isa<LoadInst, StoreInst, PHINode>(I))
        continue;

      // Examine PHI nodes that are reduction variables. Update the type to
      // account for the recurrence type.
      if (auto *PN = dyn_cast<PHINode>(&I)) {
        if (!Legal->isReductionVariable(PN))
          continue;
        const RecurrenceDescriptor &RdxDesc =
            Legal->getRecurrenceDescriptor(PN);
        if (PreferInLoopReductions || useOrderedReductions(RdxDesc) ||
            TTI.preferInLoopReduction(RdxDesc.getRecurrenceKind(),
                                      RdxDesc.getRecurrenceType()))
          continue;
        T = RdxDesc.getRecurrenceType();
      }

      // Examine the stored values.
      if (auto *ST = dyn_cast<StoreInst>(&I))
        T = ST->getValueOperand()->getType();

      assert(T->isSized() &&
             "Expected the load/store/recurrence type to be sized");

      ElementTypesInLoop.insert(T);
    }
  }
}

void VFSelectionContext::initializeVScaleForTuning() {
  if (!supportsScalableVectors())
    return;

  if (F.hasFnAttribute(Attribute::VScaleRange)) {
    auto Attr = F.getFnAttribute(Attribute::VScaleRange);
    auto Min = Attr.getVScaleRangeMin();
    auto Max = Attr.getVScaleRangeMax();
    if (Max && Min == Max) {
      VScaleForTuning = Max;
      return;
    }
  }

  VScaleForTuning = TTI.getVScaleForTuning();
}

bool VFSelectionContext::useOrderedReductions(
    const RecurrenceDescriptor &RdxDesc) const {
  return !Hints->allowReordering() && RdxDesc.isOrdered();
}

bool VFSelectionContext::runtimeChecksRequired() {
  LLVM_DEBUG(dbgs() << "LV: Performing code size checks.\n");

  Loop *L = const_cast<Loop *>(TheLoop);
  if (Legal->getRuntimePointerChecking()->Need) {
    reportVectorizationFailure(
        "Runtime ptr check is required with -Os/-Oz",
        "runtime pointer checks needed. Enable vectorization of this "
        "loop with '#pragma clang loop vectorize(enable)' when "
        "compiling with -Os/-Oz",
        "CantVersionLoopWithOptForSize", ORE, L);
    return true;
  }

  if (!PSE.getPredicate().isAlwaysTrue()) {
    reportVectorizationFailure(
        "Runtime SCEV check is required with -Os/-Oz",
        "runtime SCEV checks needed. Enable vectorization of this "
        "loop with '#pragma clang loop vectorize(enable)' when "
        "compiling with -Os/-Oz",
        "CantVersionLoopWithOptForSize", ORE, L);
    return true;
  }

  // FIXME: Avoid specializing for stride==1 instead of bailing out.
  if (!Legal->getLAI()->getSymbolicStrides().empty()) {
    reportVectorizationFailure(
        "Runtime stride check for small trip count",
        "runtime stride == 1 checks needed. Enable vectorization of "
        "this loop without such check by compiling with -Os/-Oz",
        "CantVersionLoopWithOptForSize", ORE, L);
    return true;
  }

  return false;
}

void VFSelectionContext::computeMinimalBitwidths() {
  MinBWs = computeMinimumValueSizes(TheLoop->getBlocks(), *DB, &TTI);
}

void VFSelectionContext::collectInLoopReductions() {
  // Avoid duplicating work finding in-loop reductions.
  if (!InLoopReductions.empty())
    return;

  for (const auto &Reduction : Legal->getReductionVars()) {
    PHINode *Phi = Reduction.first;
    const RecurrenceDescriptor &RdxDesc = Reduction.second;

    // Multi-use reductions (e.g., used in FindLastIV patterns) are handled
    // separately and should not be considered for in-loop reductions.
    if (RdxDesc.hasUsesOutsideReductionChain())
      continue;

    // We don't collect reductions that are type promoted (yet).
    if (RdxDesc.getRecurrenceType() != Phi->getType())
      continue;

    // In-loop AnyOf and FindIV reductions are not yet supported.
    RecurKind Kind = RdxDesc.getRecurrenceKind();
    if (RecurrenceDescriptor::isAnyOfRecurrenceKind(Kind) ||
        RecurrenceDescriptor::isFindIVRecurrenceKind(Kind) ||
        RecurrenceDescriptor::isFindLastRecurrenceKind(Kind))
      continue;

    // If the target would prefer this reduction to happen "in-loop", then we
    // want to record it as such.
    if (!PreferInLoopReductions && !useOrderedReductions(RdxDesc) &&
        !TTI.preferInLoopReduction(Kind, Phi->getType()))
      continue;

    // Check that we can correctly put the reductions into the loop, by
    // finding the chain of operations that leads from the phi to the loop
    // exit value.
    SmallVector<Instruction *, 4> ReductionOperations =
        RdxDesc.getReductionOpChain(Phi, const_cast<Loop *>(TheLoop));
    bool InLoop = !ReductionOperations.empty();

    if (InLoop) {
      InLoopReductions.insert(Phi);
      // Add the elements to InLoopReductionImmediateChains for cost modelling.
      Instruction *LastChain = Phi;
      for (auto *I : ReductionOperations) {
        InLoopReductionImmediateChains[I] = LastChain;
        LastChain = I;
      }
    }
    LLVM_DEBUG(dbgs() << "LV: Using " << (InLoop ? "inloop" : "out of loop")
                      << " reduction for phi: " << *Phi << "\n");
  }
}
