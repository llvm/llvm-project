//===- ProfileSummaryInfo.cpp - Global profile summary information --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that provides access to the global profile summary
// information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

// Knobs for profile summary based thresholds.
extern cl::opt<int> ProfileSummaryCutoffHot;
extern cl::opt<int> ProfileSummaryCutoffCold;
extern cl::opt<unsigned> ProfileSummaryHugeWorkingSetSizeThreshold;
extern cl::opt<unsigned> ProfileSummaryLargeWorkingSetSizeThreshold;
extern cl::opt<int> ProfileSummaryHotCount;
extern cl::opt<int> ProfileSummaryColdCount;

static cl::opt<bool> PartialProfile(
    "partial-profile", cl::Hidden, cl::init(false),
    cl::desc("Specify the current profile is used as a partial profile."));

cl::opt<bool> ScalePartialSampleProfileWorkingSetSize(
    "scale-partial-sample-profile-working-set-size", cl::Hidden, cl::init(true),
    cl::desc(
        "If true, scale the working set size of the partial sample profile "
        "by the partial profile ratio to reflect the size of the program "
        "being compiled."));

static cl::opt<double> PartialSampleProfileWorkingSetSizeScaleFactor(
    "partial-sample-profile-working-set-size-scale-factor", cl::Hidden,
    cl::init(0.008),
    cl::desc("The scale factor used to scale the working set size of the "
             "partial sample profile along with the partial profile ratio. "
             "This includes the factor of the profile counter per block "
             "and the factor to scale the working set size to use the same "
             "shared thresholds as PGO."));

// The profile summary metadata may be attached either by the frontend or by
// any backend passes (IR level instrumentation, for example). This method
// checks if the Summary is null and if so checks if the summary metadata is now
// available in the module and parses it to get the Summary object.
void ProfileSummaryInfo::refresh() {
  if (hasProfileSummary())
    return;
  // First try to get context sensitive ProfileSummary.
  auto *SummaryMD = M->getProfileSummary(/* IsCS */ true);
  if (SummaryMD)
    Summary.reset(ProfileSummary::getFromMD(SummaryMD));

  if (!hasProfileSummary()) {
    // This will actually return PSK_Instr or PSK_Sample summary.
    SummaryMD = M->getProfileSummary(/* IsCS */ false);
    if (SummaryMD)
      Summary.reset(ProfileSummary::getFromMD(SummaryMD));
  }
  if (!hasProfileSummary())
    return;
  computeThresholds();
}

Optional<uint64_t> ProfileSummaryInfo::getProfileCount(
    const CallBase &Call, BlockFrequencyInfo *BFI, bool AllowSynthetic) const {
  assert((isa<CallInst>(Call) || isa<InvokeInst>(Call)) &&
         "We can only get profile count for call/invoke instruction.");
  if (hasSampleProfile()) {
    // In sample PGO mode, check if there is a profile metadata on the
    // instruction. If it is present, determine hotness solely based on that,
    // since the sampled entry count may not be accurate. If there is no
    // annotated on the instruction, return None.
    uint64_t TotalCount;
    if (Call.extractProfTotalWeight(TotalCount))
      return TotalCount;
    return None;
  }
  if (BFI)
    return BFI->getBlockProfileCount(Call.getParent(), AllowSynthetic);
  return None;
}

/// Returns true if the function's entry is hot. If it returns false, it
/// either means it is not hot or it is unknown whether it is hot or not (for
/// example, no profile data is available).
bool ProfileSummaryInfo::isFunctionEntryHot(const Function *F) const {
  if (!F || !hasProfileSummary())
    return false;
  auto FunctionCount = F->getEntryCount();
  // FIXME: The heuristic used below for determining hotness is based on
  // preliminary SPEC tuning for inliner. This will eventually be a
  // convenience method that calls isHotCount.
  return FunctionCount && isHotCount(FunctionCount->getCount());
}

/// Returns true if the function contains hot code. This can include a hot
/// function entry count, hot basic block, or (in the case of Sample PGO)
/// hot total call edge count.
/// If it returns false, it either means it is not hot or it is unknown
/// (for example, no profile data is available).
bool ProfileSummaryInfo::isFunctionHotInCallGraph(
    const Function *F, BlockFrequencyInfo &BFI) const {
  if (!F || !hasProfileSummary())
    return false;
  if (auto FunctionCount = F->getEntryCount())
    if (isHotCount(FunctionCount->getCount()))
      return true;

  if (hasSampleProfile()) {
    uint64_t TotalCallCount = 0;
    for (const auto &BB : *F)
      for (const auto &I : BB)
        if (isa<CallInst>(I) || isa<InvokeInst>(I))
          if (auto CallCount = getProfileCount(cast<CallBase>(I), nullptr))
            TotalCallCount += *CallCount;
    if (isHotCount(TotalCallCount))
      return true;
  }
  for (const auto &BB : *F)
    if (isHotBlock(&BB, &BFI))
      return true;
  return false;
}

/// Returns true if the function only contains cold code. This means that
/// the function entry and blocks are all cold, and (in the case of Sample PGO)
/// the total call edge count is cold.
/// If it returns false, it either means it is not cold or it is unknown
/// (for example, no profile data is available).
bool ProfileSummaryInfo::isFunctionColdInCallGraph(
    const Function *F, BlockFrequencyInfo &BFI) const {
  if (!F || !hasProfileSummary())
    return false;
  if (auto FunctionCount = F->getEntryCount())
    if (!isColdCount(FunctionCount->getCount()))
      return false;

  if (hasSampleProfile()) {
    uint64_t TotalCallCount = 0;
    for (const auto &BB : *F)
      for (const auto &I : BB)
        if (isa<CallInst>(I) || isa<InvokeInst>(I))
          if (auto CallCount = getProfileCount(cast<CallBase>(I), nullptr))
            TotalCallCount += *CallCount;
    if (!isColdCount(TotalCallCount))
      return false;
  }
  for (const auto &BB : *F)
    if (!isColdBlock(&BB, &BFI))
      return false;
  return true;
}

bool ProfileSummaryInfo::isFunctionHotnessUnknown(const Function &F) const {
  assert(hasPartialSampleProfile() && "Expect partial sample profile");
  return !F.getEntryCount();
}

template <bool isHot>
bool ProfileSummaryInfo::isFunctionHotOrColdInCallGraphNthPercentile(
    int PercentileCutoff, const Function *F, BlockFrequencyInfo &BFI) const {
  if (!F || !hasProfileSummary())
    return false;
  if (auto FunctionCount = F->getEntryCount()) {
    if (isHot &&
        isHotCountNthPercentile(PercentileCutoff, FunctionCount->getCount()))
      return true;
    if (!isHot &&
        !isColdCountNthPercentile(PercentileCutoff, FunctionCount->getCount()))
      return false;
  }
  if (hasSampleProfile()) {
    uint64_t TotalCallCount = 0;
    for (const auto &BB : *F)
      for (const auto &I : BB)
        if (isa<CallInst>(I) || isa<InvokeInst>(I))
          if (auto CallCount = getProfileCount(cast<CallBase>(I), nullptr))
            TotalCallCount += *CallCount;
    if (isHot && isHotCountNthPercentile(PercentileCutoff, TotalCallCount))
      return true;
    if (!isHot && !isColdCountNthPercentile(PercentileCutoff, TotalCallCount))
      return false;
  }
  for (const auto &BB : *F) {
    if (isHot && isHotBlockNthPercentile(PercentileCutoff, &BB, &BFI))
      return true;
    if (!isHot && !isColdBlockNthPercentile(PercentileCutoff, &BB, &BFI))
      return false;
  }
  return !isHot;
}

// Like isFunctionHotInCallGraph but for a given cutoff.
bool ProfileSummaryInfo::isFunctionHotInCallGraphNthPercentile(
    int PercentileCutoff, const Function *F, BlockFrequencyInfo &BFI) const {
  return isFunctionHotOrColdInCallGraphNthPercentile<true>(
      PercentileCutoff, F, BFI);
}

bool ProfileSummaryInfo::isFunctionColdInCallGraphNthPercentile(
    int PercentileCutoff, const Function *F, BlockFrequencyInfo &BFI) const {
  return isFunctionHotOrColdInCallGraphNthPercentile<false>(
      PercentileCutoff, F, BFI);
}

/// Returns true if the function's entry is a cold. If it returns false, it
/// either means it is not cold or it is unknown whether it is cold or not (for
/// example, no profile data is available).
bool ProfileSummaryInfo::isFunctionEntryCold(const Function *F) const {
  if (!F)
    return false;
  if (F->hasFnAttribute(Attribute::Cold))
    return true;
  if (!hasProfileSummary())
    return false;
  auto FunctionCount = F->getEntryCount();
  // FIXME: The heuristic used below for determining coldness is based on
  // preliminary SPEC tuning for inliner. This will eventually be a
  // convenience method that calls isHotCount.
  return FunctionCount && isColdCount(FunctionCount->getCount());
}

/// Compute the hot and cold thresholds.
void ProfileSummaryInfo::computeThresholds() {
  auto &DetailedSummary = Summary->getDetailedSummary();
  auto &HotEntry = ProfileSummaryBuilder::getEntryForPercentile(
      DetailedSummary, ProfileSummaryCutoffHot);
  HotCountThreshold =
      ProfileSummaryBuilder::getHotCountThreshold(DetailedSummary);
  ColdCountThreshold =
      ProfileSummaryBuilder::getColdCountThreshold(DetailedSummary);
  assert(ColdCountThreshold <= HotCountThreshold &&
         "Cold count threshold cannot exceed hot count threshold!");
  if (!hasPartialSampleProfile() || !ScalePartialSampleProfileWorkingSetSize) {
    HasHugeWorkingSetSize =
        HotEntry.NumCounts > ProfileSummaryHugeWorkingSetSizeThreshold;
    HasLargeWorkingSetSize =
        HotEntry.NumCounts > ProfileSummaryLargeWorkingSetSizeThreshold;
  } else {
    // Scale the working set size of the partial sample profile to reflect the
    // size of the program being compiled.
    double PartialProfileRatio = Summary->getPartialProfileRatio();
    uint64_t ScaledHotEntryNumCounts =
        static_cast<uint64_t>(HotEntry.NumCounts * PartialProfileRatio *
                              PartialSampleProfileWorkingSetSizeScaleFactor);
    HasHugeWorkingSetSize =
        ScaledHotEntryNumCounts > ProfileSummaryHugeWorkingSetSizeThreshold;
    HasLargeWorkingSetSize =
        ScaledHotEntryNumCounts > ProfileSummaryLargeWorkingSetSizeThreshold;
  }
}

Optional<uint64_t>
ProfileSummaryInfo::computeThreshold(int PercentileCutoff) const {
  if (!hasProfileSummary())
    return None;
  auto iter = ThresholdCache.find(PercentileCutoff);
  if (iter != ThresholdCache.end()) {
    return iter->second;
  }
  auto &DetailedSummary = Summary->getDetailedSummary();
  auto &Entry = ProfileSummaryBuilder::getEntryForPercentile(DetailedSummary,
                                                             PercentileCutoff);
  uint64_t CountThreshold = Entry.MinCount;
  ThresholdCache[PercentileCutoff] = CountThreshold;
  return CountThreshold;
}

bool ProfileSummaryInfo::hasHugeWorkingSetSize() const {
  return HasHugeWorkingSetSize && HasHugeWorkingSetSize.getValue();
}

bool ProfileSummaryInfo::hasLargeWorkingSetSize() const {
  return HasLargeWorkingSetSize && HasLargeWorkingSetSize.getValue();
}

bool ProfileSummaryInfo::isHotCount(uint64_t C) const {
  return HotCountThreshold && C >= HotCountThreshold.getValue();
}

bool ProfileSummaryInfo::isColdCount(uint64_t C) const {
  return ColdCountThreshold && C <= ColdCountThreshold.getValue();
}

template <bool isHot>
bool ProfileSummaryInfo::isHotOrColdCountNthPercentile(int PercentileCutoff,
                                                       uint64_t C) const {
  auto CountThreshold = computeThreshold(PercentileCutoff);
  if (isHot)
    return CountThreshold && C >= CountThreshold.getValue();
  else
    return CountThreshold && C <= CountThreshold.getValue();
}

bool ProfileSummaryInfo::isHotCountNthPercentile(int PercentileCutoff,
                                                 uint64_t C) const {
  return isHotOrColdCountNthPercentile<true>(PercentileCutoff, C);
}

bool ProfileSummaryInfo::isColdCountNthPercentile(int PercentileCutoff,
                                                  uint64_t C) const {
  return isHotOrColdCountNthPercentile<false>(PercentileCutoff, C);
}

uint64_t ProfileSummaryInfo::getOrCompHotCountThreshold() const {
  return HotCountThreshold.value_or(UINT64_MAX);
}

uint64_t ProfileSummaryInfo::getOrCompColdCountThreshold() const {
  return ColdCountThreshold.value_or(0);
}

bool ProfileSummaryInfo::isHotBlock(const BasicBlock *BB,
                                    BlockFrequencyInfo *BFI) const {
  auto Count = BFI->getBlockProfileCount(BB);
  return Count && isHotCount(*Count);
}

bool ProfileSummaryInfo::isColdBlock(const BasicBlock *BB,
                                     BlockFrequencyInfo *BFI) const {
  auto Count = BFI->getBlockProfileCount(BB);
  return Count && isColdCount(*Count);
}

template <bool isHot>
bool ProfileSummaryInfo::isHotOrColdBlockNthPercentile(
    int PercentileCutoff, const BasicBlock *BB, BlockFrequencyInfo *BFI) const {
  auto Count = BFI->getBlockProfileCount(BB);
  if (isHot)
    return Count && isHotCountNthPercentile(PercentileCutoff, *Count);
  else
    return Count && isColdCountNthPercentile(PercentileCutoff, *Count);
}

bool ProfileSummaryInfo::isHotBlockNthPercentile(
    int PercentileCutoff, const BasicBlock *BB, BlockFrequencyInfo *BFI) const {
  return isHotOrColdBlockNthPercentile<true>(PercentileCutoff, BB, BFI);
}

bool ProfileSummaryInfo::isColdBlockNthPercentile(
    int PercentileCutoff, const BasicBlock *BB, BlockFrequencyInfo *BFI) const {
  return isHotOrColdBlockNthPercentile<false>(PercentileCutoff, BB, BFI);
}

bool ProfileSummaryInfo::isHotCallSite(const CallBase &CB,
                                       BlockFrequencyInfo *BFI) const {
  auto C = getProfileCount(CB, BFI);
  return C && isHotCount(*C);
}

bool ProfileSummaryInfo::isColdCallSite(const CallBase &CB,
                                        BlockFrequencyInfo *BFI) const {
  auto C = getProfileCount(CB, BFI);
  if (C)
    return isColdCount(*C);

  // In SamplePGO, if the caller has been sampled, and there is no profile
  // annotated on the callsite, we consider the callsite as cold.
  return hasSampleProfile() && CB.getCaller()->hasProfileData();
}

bool ProfileSummaryInfo::hasPartialSampleProfile() const {
  return hasProfileSummary() &&
         Summary->getKind() == ProfileSummary::PSK_Sample &&
         (PartialProfile || Summary->isPartialProfile());
}

INITIALIZE_PASS(ProfileSummaryInfoWrapperPass, "profile-summary-info",
                "Profile summary info", false, true)

ProfileSummaryInfoWrapperPass::ProfileSummaryInfoWrapperPass()
    : ImmutablePass(ID) {
  initializeProfileSummaryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

bool ProfileSummaryInfoWrapperPass::doInitialization(Module &M) {
  PSI.reset(new ProfileSummaryInfo(M));
  return false;
}

bool ProfileSummaryInfoWrapperPass::doFinalization(Module &M) {
  PSI.reset();
  return false;
}

AnalysisKey ProfileSummaryAnalysis::Key;
ProfileSummaryInfo ProfileSummaryAnalysis::run(Module &M,
                                               ModuleAnalysisManager &) {
  return ProfileSummaryInfo(M);
}

PreservedAnalyses ProfileSummaryPrinterPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  ProfileSummaryInfo &PSI = AM.getResult<ProfileSummaryAnalysis>(M);

  OS << "Functions in " << M.getName() << " with hot/cold annotations: \n";
  for (auto &F : M) {
    OS << F.getName();
    if (PSI.isFunctionEntryHot(&F))
      OS << " :hot entry ";
    else if (PSI.isFunctionEntryCold(&F))
      OS << " :cold entry ";
    OS << "\n";
  }
  return PreservedAnalyses::all();
}

char ProfileSummaryInfoWrapperPass::ID = 0;
