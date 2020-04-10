//===- llvm/Analysis/ProfileSummaryInfo.h - profile summary ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that provides access to profile summary
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PROFILE_SUMMARY_INFO_H
#define LLVM_ANALYSIS_PROFILE_SUMMARY_INFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/Pass.h"
#include <memory>

namespace llvm {
class BasicBlock;
class BlockFrequencyInfo;
class CallSite;
class Instruction;
class ProfileSummary;
/// Analysis providing profile information.
///
/// This is an immutable analysis pass that provides ability to query global
/// (program-level) profile information. The main APIs are isHotCount and
/// isColdCount that tells whether a given profile count is considered hot/cold
/// based on the profile summary. This also provides convenience methods to
/// check whether a function is hot or cold.

// FIXME: Provide convenience methods to determine hotness/coldness of other IR
// units. This would require making this depend on BFI.
class ProfileSummaryInfo {
private:
  Module &M;
  std::unique_ptr<ProfileSummary> Summary;
  bool computeSummary();
  void computeThresholds();
  // Count thresholds to answer isHotCount and isColdCount queries.
  Optional<uint64_t> HotCountThreshold, ColdCountThreshold;
  // True if the working set size of the code is considered huge,
  // because the number of profile counts required to reach the hot
  // percentile is above a huge threshold.
  Optional<bool> HasHugeWorkingSetSize;
  // True if the working set size of the code is considered large,
  // because the number of profile counts required to reach the hot
  // percentile is above a large threshold.
  Optional<bool> HasLargeWorkingSetSize;
  // Compute the threshold for a given cutoff.
  Optional<uint64_t> computeThreshold(int PercentileCutoff);
  // The map that caches the threshold values. The keys are the percentile
  // cutoff values and the values are the corresponding threshold values.
  DenseMap<int, uint64_t> ThresholdCache;

public:
  ProfileSummaryInfo(Module &M) : M(M) {}
  ProfileSummaryInfo(ProfileSummaryInfo &&Arg)
      : M(Arg.M), Summary(std::move(Arg.Summary)) {}

  /// Returns true if profile summary is available.
  bool hasProfileSummary() { return computeSummary(); }

  /// Returns true if module \c M has sample profile.
  bool hasSampleProfile() {
    return hasProfileSummary() &&
           Summary->getKind() == ProfileSummary::PSK_Sample;
  }

  /// Returns true if module \c M has instrumentation profile.
  bool hasInstrumentationProfile() {
    return hasProfileSummary() &&
           Summary->getKind() == ProfileSummary::PSK_Instr;
  }

  /// Returns true if module \c M has context sensitive instrumentation profile.
  bool hasCSInstrumentationProfile() {
    return hasProfileSummary() &&
           Summary->getKind() == ProfileSummary::PSK_CSInstr;
  }

  /// Handle the invalidation of this information.
  ///
  /// When used as a result of \c ProfileSummaryAnalysis this method will be
  /// called when the module this was computed for changes. Since profile
  /// summary is immutable after it is annotated on the module, we return false
  /// here.
  bool invalidate(Module &, const PreservedAnalyses &,
                  ModuleAnalysisManager::Invalidator &) {
    return false;
  }

  /// Returns the profile count for \p CallInst.
  Optional<uint64_t> getProfileCount(const Instruction *CallInst,
                                     BlockFrequencyInfo *BFI,
                                     bool AllowSynthetic = false);
  /// Returns true if the working set size of the code is considered huge.
  bool hasHugeWorkingSetSize();
  /// Returns true if the working set size of the code is considered large.
  bool hasLargeWorkingSetSize();
  /// Returns true if \p F has hot function entry.
  bool isFunctionEntryHot(const Function *F);
  /// Returns true if \p F contains hot code.
  bool isFunctionHotInCallGraph(const Function *F, BlockFrequencyInfo &BFI);
  /// Returns true if \p F has cold function entry.
  bool isFunctionEntryCold(const Function *F);
  /// Returns true if \p F contains only cold code.
  bool isFunctionColdInCallGraph(const Function *F, BlockFrequencyInfo &BFI);
  /// Returns true if \p F contains hot code with regard to a given hot
  /// percentile cutoff value.
  bool isFunctionHotInCallGraphNthPercentile(int PercentileCutoff,
                                             const Function *F,
                                             BlockFrequencyInfo &BFI);
  /// Returns true if \p F contains cold code with regard to a given cold
  /// percentile cutoff value.
  bool isFunctionColdInCallGraphNthPercentile(int PercentileCutoff,
                                              const Function *F,
                                              BlockFrequencyInfo &BFI);
  /// Returns true if count \p C is considered hot.
  bool isHotCount(uint64_t C);
  /// Returns true if count \p C is considered cold.
  bool isColdCount(uint64_t C);
  /// Returns true if count \p C is considered hot with regard to a given
  /// hot percentile cutoff value.
  bool isHotCountNthPercentile(int PercentileCutoff, uint64_t C);
  /// Returns true if count \p C is considered cold with regard to a given
  /// cold percentile cutoff value.
  bool isColdCountNthPercentile(int PercentileCutoff, uint64_t C);
  /// Returns true if BasicBlock \p BB is considered hot.
  bool isHotBlock(const BasicBlock *BB, BlockFrequencyInfo *BFI);
  /// Returns true if BasicBlock \p BB is considered cold.
  bool isColdBlock(const BasicBlock *BB, BlockFrequencyInfo *BFI);
  /// Returns true if BasicBlock \p BB is considered hot with regard to a given
  /// hot percentile cutoff value.
  bool isHotBlockNthPercentile(int PercentileCutoff,
                               const BasicBlock *BB, BlockFrequencyInfo *BFI);
  /// Returns true if BasicBlock \p BB is considered cold with regard to a given
  /// cold percentile cutoff value.
  bool isColdBlockNthPercentile(int PercentileCutoff,
                                const BasicBlock *BB, BlockFrequencyInfo *BFI);
  /// Returns true if CallSite \p CS is considered hot.
  bool isHotCallSite(const CallSite &CS, BlockFrequencyInfo *BFI);
  /// Returns true if Callsite \p CS is considered cold.
  bool isColdCallSite(const CallSite &CS, BlockFrequencyInfo *BFI);
  /// Returns HotCountThreshold if set. Recompute HotCountThreshold
  /// if not set.
  uint64_t getOrCompHotCountThreshold();
  /// Returns ColdCountThreshold if set. Recompute HotCountThreshold
  /// if not set.
  uint64_t getOrCompColdCountThreshold();
  /// Returns HotCountThreshold if set.
  uint64_t getHotCountThreshold() {
    return HotCountThreshold ? HotCountThreshold.getValue() : 0;
  }
  /// Returns ColdCountThreshold if set.
  uint64_t getColdCountThreshold() {
    return ColdCountThreshold ? ColdCountThreshold.getValue() : 0;
  }

 private:
  template<bool isHot>
  bool isFunctionHotOrColdInCallGraphNthPercentile(int PercentileCutoff,
                                                   const Function *F,
                                                   BlockFrequencyInfo &BFI);
  template<bool isHot>
  bool isHotOrColdCountNthPercentile(int PercentileCutoff, uint64_t C);
  template<bool isHot>
  bool isHotOrColdBlockNthPercentile(int PercentileCutoff, const BasicBlock *BB,
                                     BlockFrequencyInfo *BFI);
};

/// An analysis pass based on legacy pass manager to deliver ProfileSummaryInfo.
class ProfileSummaryInfoWrapperPass : public ImmutablePass {
  std::unique_ptr<ProfileSummaryInfo> PSI;

public:
  static char ID;
  ProfileSummaryInfoWrapperPass();

  ProfileSummaryInfo &getPSI() { return *PSI; }
  const ProfileSummaryInfo &getPSI() const { return *PSI; }

  bool doInitialization(Module &M) override;
  bool doFinalization(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

/// An analysis pass based on the new PM to deliver ProfileSummaryInfo.
class ProfileSummaryAnalysis
    : public AnalysisInfoMixin<ProfileSummaryAnalysis> {
public:
  typedef ProfileSummaryInfo Result;

  Result run(Module &M, ModuleAnalysisManager &);

private:
  friend AnalysisInfoMixin<ProfileSummaryAnalysis>;
  static AnalysisKey Key;
};

/// Printer pass that uses \c ProfileSummaryAnalysis.
class ProfileSummaryPrinterPass
    : public PassInfoMixin<ProfileSummaryPrinterPass> {
  raw_ostream &OS;

public:
  explicit ProfileSummaryPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif
