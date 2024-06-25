//===- Transforms/IPO/SampleProfileMatcher.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides the interface for SampleProfileMatcher.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_SAMPLEPROFILEMATCHER_H
#define LLVM_TRANSFORMS_IPO_SAMPLEPROFILEMATCHER_H

#include "llvm/ADT/StringSet.h"
#include "llvm/Transforms/Utils/SampleProfileLoaderBaseImpl.h"

namespace llvm {

using AnchorList = std::vector<std::pair<LineLocation, FunctionId>>;
using AnchorMap = std::map<LineLocation, FunctionId>;

// Sample profile matching - fuzzy match.
class SampleProfileMatcher {
  Module &M;
  SampleProfileReader &Reader;
  const PseudoProbeManager *ProbeManager;
  const ThinOrFullLTOPhase LTOPhase;
  SampleProfileMap FlattenedProfiles;
  // For each function, the matcher generates a map, of which each entry is a
  // mapping from the source location of current build to the source location
  // in the profile.
  StringMap<LocToLocMap> FuncMappings;

  // Match state for an anchor/callsite.
  enum class MatchState {
    Unknown = 0,
    // Initial match between input profile and current IR.
    InitialMatch = 1,
    // Initial mismatch between input profile and current IR.
    InitialMismatch = 2,
    // InitialMatch stays matched after fuzzy profile matching.
    UnchangedMatch = 3,
    // InitialMismatch stays mismatched after fuzzy profile matching.
    UnchangedMismatch = 4,
    // InitialMismatch is recovered after fuzzy profile matching.
    RecoveredMismatch = 5,
    // InitialMatch is removed and becomes mismatched after fuzzy profile
    // matching.
    RemovedMatch = 6,
  };

  // For each function, store every callsite and its matching state into this
  // map, of which each entry is a pair of callsite location and MatchState.
  // This is used for profile staleness computation and report.
  StringMap<std::unordered_map<LineLocation, MatchState, LineLocationHash>>
      FuncCallsiteMatchStates;

  // Profile mismatch statstics:
  uint64_t TotalProfiledFunc = 0;
  // Num of checksum-mismatched function.
  uint64_t NumStaleProfileFunc = 0;
  uint64_t TotalProfiledCallsites = 0;
  uint64_t NumMismatchedCallsites = 0;
  uint64_t NumRecoveredCallsites = 0;
  // Total samples for all profiled functions.
  uint64_t TotalFunctionSamples = 0;
  // Total samples for all checksum-mismatched functions.
  uint64_t MismatchedFunctionSamples = 0;
  uint64_t MismatchedCallsiteSamples = 0;
  uint64_t RecoveredCallsiteSamples = 0;

  // A dummy name for unknown indirect callee, used to differentiate from a
  // non-call instruction that also has an empty callee name.
  static constexpr const char *UnknownIndirectCallee =
      "unknown.indirect.callee";

public:
  SampleProfileMatcher(Module &M, SampleProfileReader &Reader,
                       const PseudoProbeManager *ProbeManager,
                       ThinOrFullLTOPhase LTOPhase)
      : M(M), Reader(Reader), ProbeManager(ProbeManager), LTOPhase(LTOPhase){};
  void runOnModule();
  void clearMatchingData() {
    // Do not clear FuncMappings, it stores IRLoc to ProfLoc remappings which
    // will be used for sample loader.
    FuncCallsiteMatchStates.clear();
  }

private:
  FunctionSamples *getFlattenedSamplesFor(const Function &F) {
    StringRef CanonFName = FunctionSamples::getCanonicalFnName(F);
    auto It = FlattenedProfiles.find(FunctionId(CanonFName));
    if (It != FlattenedProfiles.end())
      return &It->second;
    return nullptr;
  }
  void runOnFunction(Function &F);
  void findIRAnchors(const Function &F, AnchorMap &IRAnchors);
  void findProfileAnchors(const FunctionSamples &FS, AnchorMap &ProfileAnchors);
  // Record the callsite match states for profile staleness report, the result
  // is saved in FuncCallsiteMatchStates.
  void recordCallsiteMatchStates(const Function &F, const AnchorMap &IRAnchors,
                                 const AnchorMap &ProfileAnchors,
                                 const LocToLocMap *IRToProfileLocationMap);

  bool isMismatchState(const enum MatchState &State) {
    return State == MatchState::InitialMismatch ||
           State == MatchState::UnchangedMismatch ||
           State == MatchState::RemovedMatch;
  };

  bool isInitialState(const enum MatchState &State) {
    return State == MatchState::InitialMatch ||
           State == MatchState::InitialMismatch;
  };

  bool isFinalState(const enum MatchState &State) {
    return State == MatchState::UnchangedMatch ||
           State == MatchState::UnchangedMismatch ||
           State == MatchState::RecoveredMismatch ||
           State == MatchState::RemovedMatch;
  };

  // Count the samples of checksum mismatched function for the top-level
  // function and all inlinees.
  void countMismatchedFuncSamples(const FunctionSamples &FS, bool IsTopLevel);
  // Count the number of mismatched or recovered callsites.
  void countMismatchCallsites(const FunctionSamples &FS);
  // Count the samples of mismatched or recovered callsites for top-level
  // function and all inlinees.
  void countMismatchedCallsiteSamples(const FunctionSamples &FS);
  void computeAndReportProfileStaleness();

  LocToLocMap &getIRToProfileLocationMap(const Function &F) {
    auto Ret = FuncMappings.try_emplace(
        FunctionSamples::getCanonicalFnName(F.getName()), LocToLocMap());
    return Ret.first->second;
  }
  void distributeIRToProfileLocationMap();
  void distributeIRToProfileLocationMap(FunctionSamples &FS);
  // This function implements the Myers diff algorithm used for stale profile
  // matching. The algorithm provides a simple and efficient way to find the
  // Longest Common Subsequence(LCS) or the Shortest Edit Script(SES) of two
  // sequences. For more details, refer to the paper 'An O(ND) Difference
  // Algorithm and Its Variations' by Eugene W. Myers.
  // In the scenario of profile fuzzy matching, the two sequences are the IR
  // callsite anchors and profile callsite anchors. The subsequence equivalent
  // parts from the resulting SES are used to remap the IR locations to the
  // profile locations. As the number of function callsite is usually not big,
  // we currently just implements the basic greedy version(page 6 of the paper).
  LocToLocMap
  longestCommonSequence(const AnchorList &IRCallsiteAnchors,
                        const AnchorList &ProfileCallsiteAnchors) const;
  void matchNonCallsiteLocs(const LocToLocMap &AnchorMatchings,
                            const AnchorMap &IRAnchors,
                            LocToLocMap &IRToProfileLocationMap);
  void runStaleProfileMatching(const Function &F, const AnchorMap &IRAnchors,
                               const AnchorMap &ProfileAnchors,
                               LocToLocMap &IRToProfileLocationMap);
  void reportOrPersistProfileStats();
};
} // end namespace llvm
#endif // LLVM_TRANSFORMS_IPO_SAMPLEPROFILEMATCHER_H
