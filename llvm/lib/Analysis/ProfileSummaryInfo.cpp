//===- ProfileSummaryInfo.cpp - Global profile summary information --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that provides access to the global profile summary
// information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ProfileSummary.h"
using namespace llvm;

// The following two parameters determine the threshold for a count to be
// considered hot/cold. These two parameters are percentile values (multiplied
// by 10000). If the counts are sorted in descending order, the minimum count to
// reach ProfileSummaryCutoffHot gives the threshold to determine a hot count.
// Similarly, the minimum count to reach ProfileSummaryCutoffCold gives the
// threshold for determining cold count (everything <= this threshold is
// considered cold).

static cl::opt<int> ProfileSummaryCutoffHot(
    "profile-summary-cutoff-hot", cl::Hidden, cl::init(999000), cl::ZeroOrMore,
    cl::desc("A count is hot if it exceeds the minimum count to"
             " reach this percentile of total counts."));

static cl::opt<int> ProfileSummaryCutoffCold(
    "profile-summary-cutoff-cold", cl::Hidden, cl::init(999999), cl::ZeroOrMore,
    cl::desc("A count is cold if it is below the minimum count"
             " to reach this percentile of total counts."));

// Find the minimum count to reach a desired percentile of counts.
static uint64_t getMinCountForPercentile(SummaryEntryVector &DS,
                                         uint64_t Percentile) {
  auto Compare = [](const ProfileSummaryEntry &Entry, uint64_t Percentile) {
    return Entry.Cutoff < Percentile;
  };
  auto It = std::lower_bound(DS.begin(), DS.end(), Percentile, Compare);
  // The required percentile has to be <= one of the percentiles in the
  // detailed summary.
  if (It == DS.end())
    report_fatal_error("Desired percentile exceeds the maximum cutoff");
  return It->MinCount;
}

// The profile summary metadata may be attached either by the frontend or by
// any backend passes (IR level instrumentation, for example). This method
// checks if the Summary is null and if so checks if the summary metadata is now
// available in the module and parses it to get the Summary object.
void ProfileSummaryInfo::computeSummary() {
  if (Summary)
    return;
  auto *SummaryMD = M.getProfileSummary();
  if (!SummaryMD)
    return;
  Summary.reset(ProfileSummary::getFromMD(SummaryMD));
}

// Returns true if the function is a hot function. If it returns false, it
// either means it is not hot or it is unknown whether F is hot or not (for
// example, no profile data is available).
bool ProfileSummaryInfo::isHotFunction(const Function *F) {
  computeSummary();
  if (!F || !Summary)
    return false;
  auto FunctionCount = F->getEntryCount();
  // FIXME: The heuristic used below for determining hotness is based on
  // preliminary SPEC tuning for inliner. This will eventually be a
  // convenience method that calls isHotCount.
  return (FunctionCount &&
          FunctionCount.getValue() >=
              (uint64_t)(0.3 * (double)Summary->getMaxFunctionCount()));
}

// Returns true if the function is a cold function. If it returns false, it
// either means it is not cold or it is unknown whether F is cold or not (for
// example, no profile data is available).
bool ProfileSummaryInfo::isColdFunction(const Function *F) {
  computeSummary();
  if (!F)
    return false;
  if (F->hasFnAttribute(Attribute::Cold)) {
    return true;
  }
  if (!Summary)
    return false;
  auto FunctionCount = F->getEntryCount();
  // FIXME: The heuristic used below for determining coldness is based on
  // preliminary SPEC tuning for inliner. This will eventually be a
  // convenience method that calls isHotCount.
  return (FunctionCount &&
          FunctionCount.getValue() <=
              (uint64_t)(0.01 * (double)Summary->getMaxFunctionCount()));
}

// Compute the hot and cold thresholds.
void ProfileSummaryInfo::computeThresholds() {
  if (!Summary)
    computeSummary();
  if (!Summary)
    return;
  auto &DetailedSummary = Summary->getDetailedSummary();
  HotCountThreshold =
      getMinCountForPercentile(DetailedSummary, ProfileSummaryCutoffHot);
  ColdCountThreshold =
      getMinCountForPercentile(DetailedSummary, ProfileSummaryCutoffCold);
}

bool ProfileSummaryInfo::isHotCount(uint64_t C) {
  if (!HotCountThreshold)
    computeThresholds();
  return HotCountThreshold && C >= HotCountThreshold.getValue();
}

bool ProfileSummaryInfo::isColdCount(uint64_t C) {
  if (!ColdCountThreshold)
    computeThresholds();
  return ColdCountThreshold && C <= ColdCountThreshold.getValue();
}

ProfileSummaryInfo *ProfileSummaryInfoWrapperPass::getPSI(Module &M) {
  if (!PSI)
    PSI.reset(new ProfileSummaryInfo(M));
  return PSI.get();
}

INITIALIZE_PASS(ProfileSummaryInfoWrapperPass, "profile-summary-info",
                "Profile summary info", false, true)

ProfileSummaryInfoWrapperPass::ProfileSummaryInfoWrapperPass()
    : ImmutablePass(ID) {
  initializeProfileSummaryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

char ProfileSummaryAnalysis::PassID;
ProfileSummaryInfo ProfileSummaryAnalysis::run(Module &M,
                                               ModuleAnalysisManager &) {
  return ProfileSummaryInfo(M);
}

// FIXME: This only tests isHotFunction and isColdFunction and not the
// isHotCount and isColdCount calls.
PreservedAnalyses ProfileSummaryPrinterPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  ProfileSummaryInfo &PSI = AM.getResult<ProfileSummaryAnalysis>(M);

  OS << "Functions in " << M.getName() << " with hot/cold annotations: \n";
  for (auto &F : M) {
    OS << F.getName();
    if (PSI.isHotFunction(&F))
      OS << " :hot ";
    else if (PSI.isColdFunction(&F))
      OS << " :cold ";
    OS << "\n";
  }
  return PreservedAnalyses::all();
}

char ProfileSummaryInfoWrapperPass::ID = 0;
