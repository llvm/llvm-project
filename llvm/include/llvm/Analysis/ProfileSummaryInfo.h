//===- llvm/Analysis/ProfileSummaryInfo.h - profile summary ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that provides access to profile summary
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PROFILE_SUMMARY_INFO_H
#define LLVM_ANALYSIS_PROFILE_SUMMARY_INFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include <memory>

namespace llvm {
class BasicBlock;
class BlockFrequencyInfo;
class CallSite;
class ProfileSummary;
/// \brief Analysis providing profile information.
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

public:
  ProfileSummaryInfo(Module &M) : M(M) {}
  ProfileSummaryInfo(ProfileSummaryInfo &&Arg)
      : M(Arg.M), Summary(std::move(Arg.Summary)) {}
  /// Returns the profile count for \p CallInst.
  static Optional<uint64_t> getProfileCount(const Instruction *CallInst,
                                            BlockFrequencyInfo *BFI);
  /// \brief Returns true if \p F has hot function entry.
  bool isFunctionEntryHot(const Function *F);
  /// Returns true if \p F has hot function entry or hot call edge.
  bool isFunctionHotInCallGraph(const Function *F);
  /// \brief Returns true if \p F has cold function entry.
  bool isFunctionEntryCold(const Function *F);
  /// Returns true if \p F has cold function entry or cold call edge.
  bool isFunctionColdInCallGraph(const Function *F);
  /// \brief Returns true if \p F is a hot function.
  bool isHotCount(uint64_t C);
  /// \brief Returns true if count \p C is considered cold.
  bool isColdCount(uint64_t C);
  /// \brief Returns true if BasicBlock \p B is considered hot.
  bool isHotBB(const BasicBlock *B, BlockFrequencyInfo *BFI);
  /// \brief Returns true if BasicBlock \p B is considered cold.
  bool isColdBB(const BasicBlock *B, BlockFrequencyInfo *BFI);
  /// \brief Returns true if CallSite \p CS is considered hot.
  bool isHotCallSite(const CallSite &CS, BlockFrequencyInfo *BFI);
  /// \brief Returns true if Callsite \p CS is considered cold.
  bool isColdCallSite(const CallSite &CS, BlockFrequencyInfo *BFI);
};

/// An analysis pass based on legacy pass manager to deliver ProfileSummaryInfo.
class ProfileSummaryInfoWrapperPass : public ImmutablePass {
  std::unique_ptr<ProfileSummaryInfo> PSI;

public:
  static char ID;
  ProfileSummaryInfoWrapperPass();

  ProfileSummaryInfo *getPSI() {
    return &*PSI;
  }

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

/// \brief Printer pass that uses \c ProfileSummaryAnalysis.
class ProfileSummaryPrinterPass
    : public PassInfoMixin<ProfileSummaryPrinterPass> {
  raw_ostream &OS;

public:
  explicit ProfileSummaryPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif
