//===- SampleProfileMatcher.cpp - Sampling-based Stale Profile Matcher ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SampleProfileMatcher used for stale
// profile matching.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/SampleProfileMatcher.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"

using namespace llvm;
using namespace sampleprof;

#define DEBUG_TYPE "sample-profile-matcher"

extern cl::opt<bool> SalvageStaleProfile;
extern cl::opt<bool> PersistProfileStaleness;
extern cl::opt<bool> ReportProfileStaleness;

void SampleProfileMatcher::findIRAnchors(
    const Function &F, std::map<LineLocation, StringRef> &IRAnchors) {
  // For inlined code, recover the original callsite and callee by finding the
  // top-level inline frame. e.g. For frame stack "main:1 @ foo:2 @ bar:3", the
  // top-level frame is "main:1", the callsite is "1" and the callee is "foo".
  auto FindTopLevelInlinedCallsite = [](const DILocation *DIL) {
    assert((DIL && DIL->getInlinedAt()) && "No inlined callsite");
    const DILocation *PrevDIL = nullptr;
    do {
      PrevDIL = DIL;
      DIL = DIL->getInlinedAt();
    } while (DIL->getInlinedAt());

    LineLocation Callsite = FunctionSamples::getCallSiteIdentifier(DIL);
    StringRef CalleeName = PrevDIL->getSubprogramLinkageName();
    return std::make_pair(Callsite, CalleeName);
  };

  auto GetCanonicalCalleeName = [](const CallBase *CB) {
    StringRef CalleeName = UnknownIndirectCallee;
    if (Function *Callee = CB->getCalledFunction())
      CalleeName = FunctionSamples::getCanonicalFnName(Callee->getName());
    return CalleeName;
  };

  // Extract profile matching anchors in the IR.
  for (auto &BB : F) {
    for (auto &I : BB) {
      DILocation *DIL = I.getDebugLoc();
      if (!DIL)
        continue;

      if (FunctionSamples::ProfileIsProbeBased) {
        if (auto Probe = extractProbe(I)) {
          // Flatten inlined IR for the matching.
          if (DIL->getInlinedAt()) {
            IRAnchors.emplace(FindTopLevelInlinedCallsite(DIL));
          } else {
            // Use empty StringRef for basic block probe.
            StringRef CalleeName;
            if (const auto *CB = dyn_cast<CallBase>(&I)) {
              // Skip the probe inst whose callee name is "llvm.pseudoprobe".
              if (!isa<IntrinsicInst>(&I))
                CalleeName = GetCanonicalCalleeName(CB);
            }
            IRAnchors.emplace(LineLocation(Probe->Id, 0), CalleeName);
          }
        }
      } else {
        // TODO: For line-number based profile(AutoFDO), currently only support
        // find callsite anchors. In future, we need to parse all the non-call
        // instructions to extract the line locations for profile matching.
        if (!isa<CallBase>(&I) || isa<IntrinsicInst>(&I))
          continue;

        if (DIL->getInlinedAt()) {
          IRAnchors.emplace(FindTopLevelInlinedCallsite(DIL));
        } else {
          LineLocation Callsite = FunctionSamples::getCallSiteIdentifier(DIL);
          StringRef CalleeName = GetCanonicalCalleeName(dyn_cast<CallBase>(&I));
          IRAnchors.emplace(Callsite, CalleeName);
        }
      }
    }
  }
}

void SampleProfileMatcher::findProfileAnchors(
    const FunctionSamples &FS,
    std::map<LineLocation, std::unordered_set<FunctionId>> &ProfileAnchors) {
  auto isInvalidLineOffset = [](uint32_t LineOffset) {
    return LineOffset & 0x8000;
  };

  for (const auto &I : FS.getBodySamples()) {
    const LineLocation &Loc = I.first;
    if (isInvalidLineOffset(Loc.LineOffset))
      continue;
    for (const auto &I : I.second.getCallTargets()) {
      auto Ret =
          ProfileAnchors.try_emplace(Loc, std::unordered_set<FunctionId>());
      Ret.first->second.insert(I.first);
    }
  }

  for (const auto &I : FS.getCallsiteSamples()) {
    const LineLocation &Loc = I.first;
    if (isInvalidLineOffset(Loc.LineOffset))
      continue;
    const auto &CalleeMap = I.second;
    for (const auto &I : CalleeMap) {
      auto Ret =
          ProfileAnchors.try_emplace(Loc, std::unordered_set<FunctionId>());
      Ret.first->second.insert(I.first);
    }
  }
}

// Call target name anchor based profile fuzzy matching.
// Input:
// For IR locations, the anchor is the callee name of direct callsite; For
// profile locations, it's the call target name for BodySamples or inlinee's
// profile name for CallsiteSamples.
// Matching heuristic:
// First match all the anchors in lexical order, then split the non-anchor
// locations between the two anchors evenly, first half are matched based on the
// start anchor, second half are matched based on the end anchor.
// For example, given:
// IR locations:      [1, 2(foo), 3, 5, 6(bar), 7]
// Profile locations: [1, 2, 3(foo), 4, 7, 8(bar), 9]
// The matching gives:
//   [1,    2(foo), 3,  5,  6(bar), 7]
//    |     |       |   |     |     |
//   [1, 2, 3(foo), 4,  7,  8(bar), 9]
// The output mapping: [2->3, 3->4, 5->7, 6->8, 7->9].
void SampleProfileMatcher::runStaleProfileMatching(
    const Function &F, const std::map<LineLocation, StringRef> &IRAnchors,
    const std::map<LineLocation, std::unordered_set<FunctionId>>
        &ProfileAnchors,
    LocToLocMap &IRToProfileLocationMap) {
  LLVM_DEBUG(dbgs() << "Run stale profile matching for " << F.getName()
                    << "\n");
  assert(IRToProfileLocationMap.empty() &&
         "Run stale profile matching only once per function");

  std::unordered_map<FunctionId, std::set<LineLocation>> CalleeToCallsitesMap;
  for (const auto &I : ProfileAnchors) {
    const auto &Loc = I.first;
    const auto &Callees = I.second;
    // Filter out possible indirect calls, use direct callee name as anchor.
    if (Callees.size() == 1) {
      FunctionId CalleeName = *Callees.begin();
      const auto &Candidates = CalleeToCallsitesMap.try_emplace(
          CalleeName, std::set<LineLocation>());
      Candidates.first->second.insert(Loc);
    }
  }

  auto InsertMatching = [&](const LineLocation &From, const LineLocation &To) {
    // Skip the unchanged location mapping to save memory.
    if (From != To)
      IRToProfileLocationMap.insert({From, To});
  };

  // Use function's beginning location as the initial anchor.
  int32_t LocationDelta = 0;
  SmallVector<LineLocation> LastMatchedNonAnchors;

  for (const auto &IR : IRAnchors) {
    const auto &Loc = IR.first;
    auto CalleeName = IR.second;
    bool IsMatchedAnchor = false;
    // Match the anchor location in lexical order.
    if (!CalleeName.empty()) {
      auto CandidateAnchors =
          CalleeToCallsitesMap.find(getRepInFormat(CalleeName));
      if (CandidateAnchors != CalleeToCallsitesMap.end() &&
          !CandidateAnchors->second.empty()) {
        auto CI = CandidateAnchors->second.begin();
        const auto Candidate = *CI;
        CandidateAnchors->second.erase(CI);
        InsertMatching(Loc, Candidate);
        LLVM_DEBUG(dbgs() << "Callsite with callee:" << CalleeName
                          << " is matched from " << Loc << " to " << Candidate
                          << "\n");
        LocationDelta = Candidate.LineOffset - Loc.LineOffset;

        // Match backwards for non-anchor locations.
        // The locations in LastMatchedNonAnchors have been matched forwards
        // based on the previous anchor, spilt it evenly and overwrite the
        // second half based on the current anchor.
        for (size_t I = (LastMatchedNonAnchors.size() + 1) / 2;
             I < LastMatchedNonAnchors.size(); I++) {
          const auto &L = LastMatchedNonAnchors[I];
          uint32_t CandidateLineOffset = L.LineOffset + LocationDelta;
          LineLocation Candidate(CandidateLineOffset, L.Discriminator);
          InsertMatching(L, Candidate);
          LLVM_DEBUG(dbgs() << "Location is rematched backwards from " << L
                            << " to " << Candidate << "\n");
        }

        IsMatchedAnchor = true;
        LastMatchedNonAnchors.clear();
      }
    }

    // Match forwards for non-anchor locations.
    if (!IsMatchedAnchor) {
      uint32_t CandidateLineOffset = Loc.LineOffset + LocationDelta;
      LineLocation Candidate(CandidateLineOffset, Loc.Discriminator);
      InsertMatching(Loc, Candidate);
      LLVM_DEBUG(dbgs() << "Location is matched from " << Loc << " to "
                        << Candidate << "\n");
      LastMatchedNonAnchors.emplace_back(Loc);
    }
  }
}

void SampleProfileMatcher::runOnFunction(Function &F) {
  // We need to use flattened function samples for matching.
  // Unlike IR, which includes all callsites from the source code, the callsites
  // in profile only show up when they are hit by samples, i,e. the profile
  // callsites in one context may differ from those in another context. To get
  // the maximum number of callsites, we merge the function profiles from all
  // contexts, aka, the flattened profile to find profile anchors.
  const auto *FSFlattened = getFlattenedSamplesFor(F);
  if (!FSFlattened)
    return;

  // Anchors for IR. It's a map from IR location to callee name, callee name is
  // empty for non-call instruction and use a dummy name(UnknownIndirectCallee)
  // for unknown indrect callee name.
  std::map<LineLocation, StringRef> IRAnchors;
  findIRAnchors(F, IRAnchors);
  // Anchors for profile. It's a map from callsite location to a set of callee
  // name.
  std::map<LineLocation, std::unordered_set<FunctionId>> ProfileAnchors;
  findProfileAnchors(*FSFlattened, ProfileAnchors);

  // Compute the callsite match states for profile staleness report.
  if (ReportProfileStaleness || PersistProfileStaleness)
    recordCallsiteMatchStates(F, IRAnchors, ProfileAnchors, nullptr);

  // For probe-based profiles, run matching only when the current profile is not
  // valid.
  if (SalvageStaleProfile && (!FunctionSamples::ProfileIsProbeBased ||
                              !ProbeManager->profileIsValid(F, *FSFlattened))) {
    // For imported functions, the checksum metadata(pseudo_probe_desc) are
    // dropped, so we leverage function attribute(profile-checksum-mismatch) to
    // transfer the info: add the attribute during pre-link phase and check it
    // during post-link phase(see "profileIsValid").
    if (FunctionSamples::ProfileIsProbeBased &&
        LTOPhase == ThinOrFullLTOPhase::ThinLTOPreLink)
      F.addFnAttr("profile-checksum-mismatch");

    // The matching result will be saved to IRToProfileLocationMap, create a
    // new map for each function.
    auto &IRToProfileLocationMap = getIRToProfileLocationMap(F);
    runStaleProfileMatching(F, IRAnchors, ProfileAnchors,
                            IRToProfileLocationMap);
    // Find and update callsite match states after matching.
    if (ReportProfileStaleness || PersistProfileStaleness)
      recordCallsiteMatchStates(F, IRAnchors, ProfileAnchors,
                                &IRToProfileLocationMap);
  }
}

void SampleProfileMatcher::recordCallsiteMatchStates(
    const Function &F, const std::map<LineLocation, StringRef> &IRAnchors,
    const std::map<LineLocation, std::unordered_set<FunctionId>>
        &ProfileAnchors,
    const LocToLocMap *IRToProfileLocationMap) {
  bool IsPostMatch = IRToProfileLocationMap != nullptr;
  auto &CallsiteMatchStates =
      FuncCallsiteMatchStates[FunctionSamples::getCanonicalFnName(F.getName())];

  auto MapIRLocToProfileLoc = [&](const LineLocation &IRLoc) {
    // IRToProfileLocationMap is null in pre-match phrase.
    if (!IRToProfileLocationMap)
      return IRLoc;
    const auto &ProfileLoc = IRToProfileLocationMap->find(IRLoc);
    if (ProfileLoc != IRToProfileLocationMap->end())
      return ProfileLoc->second;
    else
      return IRLoc;
  };

  for (const auto &I : IRAnchors) {
    // After fuzzy profile matching, use the matching result to remap the
    // current IR callsite.
    const auto &ProfileLoc = MapIRLocToProfileLoc(I.first);
    const auto &IRCalleeName = I.second;
    const auto &It = ProfileAnchors.find(ProfileLoc);
    if (It == ProfileAnchors.end())
      continue;
    const auto &Callees = It->second;

    bool IsCallsiteMatched = false;
    // Since indirect call does not have CalleeName, check conservatively if
    // callsite in the profile is a callsite location. This is to reduce num of
    // false positive since otherwise all the indirect call samples will be
    // reported as mismatching.
    if (IRCalleeName == SampleProfileMatcher::UnknownIndirectCallee)
      IsCallsiteMatched = true;
    else if (Callees.size() == 1 && Callees.count(getRepInFormat(IRCalleeName)))
      IsCallsiteMatched = true;

    if (IsCallsiteMatched) {
      auto It = CallsiteMatchStates.find(ProfileLoc);
      if (It == CallsiteMatchStates.end())
        CallsiteMatchStates.emplace(ProfileLoc, MatchState::InitialMatch);
      else if (IsPostMatch) {
        if (It->second == MatchState::InitialMatch)
          It->second = MatchState::UnchangedMatch;
        else if (It->second == MatchState::InitialMismatch)
          It->second = MatchState::RecoveredMismatch;
      }
    }
  }

  // Check if there are any callsites in the profile that does not match to any
  // IR callsites.
  for (const auto &I : ProfileAnchors) {
    const auto &Loc = I.first;
    [[maybe_unused]] const auto &Callees = I.second;
    assert(!Callees.empty() && "Callees should not be empty");
    auto It = CallsiteMatchStates.find(Loc);
    if (It == CallsiteMatchStates.end())
      CallsiteMatchStates.emplace(Loc, MatchState::InitialMismatch);
    else if (IsPostMatch) {
      // Update the state if it's not matched(UnchangedMatch or
      // RecoveredMismatch).
      if (It->second == MatchState::InitialMismatch)
        It->second = MatchState::UnchangedMismatch;
      else if (It->second == MatchState::InitialMatch)
        It->second = MatchState::RemovedMatch;
    }
  }
}

void SampleProfileMatcher::countMismatchedFuncSamples(const FunctionSamples &FS,
                                                      bool IsTopLevel) {
  const auto *FuncDesc = ProbeManager->getDesc(FS.getGUID());
  // Skip the function that is external or renamed.
  if (!FuncDesc)
    return;

  if (ProbeManager->profileIsHashMismatched(*FuncDesc, FS)) {
    if (IsTopLevel)
      NumStaleProfileFunc++;
    // Given currently all probe ids are after block probe ids, once the
    // checksum is mismatched, it's likely all the callites are mismatched and
    // dropped. We conservatively count all the samples as mismatched and stop
    // counting the inlinees' profiles.
    MismatchedFunctionSamples += FS.getTotalSamples();
    return;
  }

  // Even the current-level function checksum is matched, it's possible that the
  // nested inlinees' checksums are mismatched that affect the inlinee's sample
  // loading, we need to go deeper to check the inlinees' function samples.
  // Similarly, count all the samples as mismatched if the inlinee's checksum is
  // mismatched using this recursive function.
  for (const auto &I : FS.getCallsiteSamples())
    for (const auto &CS : I.second)
      countMismatchedFuncSamples(CS.second, false);
}

void SampleProfileMatcher::countMismatchedCallsiteSamples(
    const FunctionSamples &FS) {
  auto It = FuncCallsiteMatchStates.find(FS.getFuncName());
  // Skip it if no mismatched callsite or this is an external function.
  if (It == FuncCallsiteMatchStates.end() || It->second.empty())
    return;
  const auto &CallsiteMatchStates = It->second;

  auto findMatchState = [&](const LineLocation &Loc) {
    auto It = CallsiteMatchStates.find(Loc);
    if (It == CallsiteMatchStates.end())
      return MatchState::Unknown;
    return It->second;
  };

  auto AttributeMismatchedSamples = [&](const enum MatchState &State,
                                        uint64_t Samples) {
    if (isMismatchState(State))
      MismatchedCallsiteSamples += Samples;
    else if (State == MatchState::RecoveredMismatch)
      RecoveredCallsiteSamples += Samples;
  };

  // The non-inlined callsites are saved in the body samples of function
  // profile, go through it to count the non-inlined callsite samples.
  for (const auto &I : FS.getBodySamples())
    AttributeMismatchedSamples(findMatchState(I.first), I.second.getSamples());

  // Count the inlined callsite samples.
  for (const auto &I : FS.getCallsiteSamples()) {
    auto State = findMatchState(I.first);
    uint64_t CallsiteSamples = 0;
    for (const auto &CS : I.second)
      CallsiteSamples += CS.second.getTotalSamples();
    AttributeMismatchedSamples(State, CallsiteSamples);

    if (isMismatchState(State))
      continue;

    // When the current level of inlined call site matches the profiled call
    // site, we need to go deeper along the inline tree to count mismatches from
    // lower level inlinees.
    for (const auto &CS : I.second)
      countMismatchedCallsiteSamples(CS.second);
  }
}

void SampleProfileMatcher::countMismatchCallsites(const FunctionSamples &FS) {
  auto It = FuncCallsiteMatchStates.find(FS.getFuncName());
  // Skip it if no mismatched callsite or this is an external function.
  if (It == FuncCallsiteMatchStates.end() || It->second.empty())
    return;
  const auto &MatchStates = It->second;
  [[maybe_unused]] bool OnInitialState =
      isInitialState(MatchStates.begin()->second);
  for (const auto &I : MatchStates) {
    TotalProfiledCallsites++;
    assert(
        (OnInitialState ? isInitialState(I.second) : isFinalState(I.second)) &&
        "Profile matching state is inconsistent");

    if (isMismatchState(I.second))
      NumMismatchedCallsites++;
    else if (I.second == MatchState::RecoveredMismatch)
      NumRecoveredCallsites++;
  }
}

void SampleProfileMatcher::computeAndReportProfileStaleness() {
  if (!ReportProfileStaleness && !PersistProfileStaleness)
    return;

  // Count profile mismatches for profile staleness report.
  for (const auto &F : M) {
    if (skipProfileForFunction(F))
      continue;
    // As the stats will be merged by linker, skip reporting the metrics for
    // imported functions to avoid repeated counting.
    if (GlobalValue::isAvailableExternallyLinkage(F.getLinkage()))
      continue;
    const auto *FS = Reader.getSamplesFor(F);
    if (!FS)
      continue;
    TotalProfiledFunc++;
    TotalFunctionSamples += FS->getTotalSamples();

    // Checksum mismatch is only used in pseudo-probe mode.
    if (FunctionSamples::ProfileIsProbeBased)
      countMismatchedFuncSamples(*FS, true);

    // Count mismatches and samples for calliste.
    countMismatchCallsites(*FS);
    countMismatchedCallsiteSamples(*FS);
  }

  if (ReportProfileStaleness) {
    if (FunctionSamples::ProfileIsProbeBased) {
      errs() << "(" << NumStaleProfileFunc << "/" << TotalProfiledFunc
             << ") of functions' profile are invalid and ("
             << MismatchedFunctionSamples << "/" << TotalFunctionSamples
             << ") of samples are discarded due to function hash mismatch.\n";
    }
    errs() << "(" << (NumMismatchedCallsites + NumRecoveredCallsites) << "/"
           << TotalProfiledCallsites
           << ") of callsites' profile are invalid and ("
           << (MismatchedCallsiteSamples + RecoveredCallsiteSamples) << "/"
           << TotalFunctionSamples
           << ") of samples are discarded due to callsite location mismatch.\n";
    errs() << "(" << NumRecoveredCallsites << "/"
           << (NumRecoveredCallsites + NumMismatchedCallsites)
           << ") of callsites and (" << RecoveredCallsiteSamples << "/"
           << (RecoveredCallsiteSamples + MismatchedCallsiteSamples)
           << ") of samples are recovered by stale profile matching.\n";
  }

  if (PersistProfileStaleness) {
    LLVMContext &Ctx = M.getContext();
    MDBuilder MDB(Ctx);

    SmallVector<std::pair<StringRef, uint64_t>> ProfStatsVec;
    if (FunctionSamples::ProfileIsProbeBased) {
      ProfStatsVec.emplace_back("NumStaleProfileFunc", NumStaleProfileFunc);
      ProfStatsVec.emplace_back("TotalProfiledFunc", TotalProfiledFunc);
      ProfStatsVec.emplace_back("MismatchedFunctionSamples",
                                MismatchedFunctionSamples);
      ProfStatsVec.emplace_back("TotalFunctionSamples", TotalFunctionSamples);
    }

    ProfStatsVec.emplace_back("NumMismatchedCallsites", NumMismatchedCallsites);
    ProfStatsVec.emplace_back("NumRecoveredCallsites", NumRecoveredCallsites);
    ProfStatsVec.emplace_back("TotalProfiledCallsites", TotalProfiledCallsites);
    ProfStatsVec.emplace_back("MismatchedCallsiteSamples",
                              MismatchedCallsiteSamples);
    ProfStatsVec.emplace_back("RecoveredCallsiteSamples",
                              RecoveredCallsiteSamples);

    auto *MD = MDB.createLLVMStats(ProfStatsVec);
    auto *NMD = M.getOrInsertNamedMetadata("llvm.stats");
    NMD->addOperand(MD);
  }
}

void SampleProfileMatcher::runOnModule() {
  ProfileConverter::flattenProfile(Reader.getProfiles(), FlattenedProfiles,
                                   FunctionSamples::ProfileIsCS);
  for (auto &F : M) {
    if (skipProfileForFunction(F))
      continue;
    runOnFunction(F);
  }
  if (SalvageStaleProfile)
    distributeIRToProfileLocationMap();

  computeAndReportProfileStaleness();
}

void SampleProfileMatcher::distributeIRToProfileLocationMap(
    FunctionSamples &FS) {
  const auto ProfileMappings = FuncMappings.find(FS.getFuncName());
  if (ProfileMappings != FuncMappings.end()) {
    FS.setIRToProfileLocationMap(&(ProfileMappings->second));
  }

  for (auto &Callees :
       const_cast<CallsiteSampleMap &>(FS.getCallsiteSamples())) {
    for (auto &FS : Callees.second) {
      distributeIRToProfileLocationMap(FS.second);
    }
  }
}

// Use a central place to distribute the matching results. Outlined and inlined
// profile with the function name will be set to the same pointer.
void SampleProfileMatcher::distributeIRToProfileLocationMap() {
  for (auto &I : Reader.getProfiles()) {
    distributeIRToProfileLocationMap(I.second);
  }
}
