//===-- CSPreInliner.cpp - Profile guided preinliner -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CSPreInliner.h"
#include "ProfiledBinary.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Transforms/IPO/SampleProfile.h"
#include <cstdint>
#include <map>

#define DEBUG_TYPE "cs-preinliner"

using namespace llvm;
using namespace sampleprof;

STATISTIC(PreInlNumCSInlined,
          "Number of functions inlined with context sensitive profile");
STATISTIC(PreInlNumCSNotInlined,
          "Number of functions not inlined with context sensitive profile");
STATISTIC(PreInlNumCSICPRejected,
          "Number of indirect call target contexts not inlined because the "
          "sample profile loader would not promote them");
STATISTIC(PreInlNumCSInlinedHitMinLimit,
          "Number of functions with FDO inline stopped due to min size limit");
STATISTIC(PreInlNumCSInlinedHitMaxLimit,
          "Number of functions with FDO inline stopped due to max size limit");
STATISTIC(
    PreInlNumCSInlinedHitGrowthLimit,
    "Number of functions with FDO inline stopped due to growth size limit");

// The switches specify inline thresholds used in SampleProfileLoader inlining.
// TODO: the actual threshold to be tuned here because the size here is based
// on machine code not LLVM IR.
namespace llvm {
cl::opt<bool> EnableCSPreInliner(
    "csspgo-preinliner", cl::Hidden, cl::init(true),
    cl::desc("Run a global pre-inliner to merge context profile based on "
             "estimated global top-down inline decisions"));

cl::opt<bool> UseContextCostForPreInliner(
    "use-context-cost-for-preinliner", cl::Hidden, cl::init(true),
    cl::desc("Use context-sensitive byte size cost for preinliner decisions"));
} // namespace llvm

static cl::opt<bool> SamplePreInlineReplay(
    "csspgo-replay-preinline", cl::Hidden, cl::init(false),
    cl::desc(
        "Replay previous inlining and adjust context profile accordingly"));

static cl::opt<bool> CSPreinlHonorICPLimits(
    "csspgo-preinliner-honor-icp-limits", cl::Hidden, cl::init(true),
    cl::desc("Do not inline indirect call target contexts that the sample "
             "profile loader will not promote (see "
             "-sample-profile-icp-relative-hotness)."));

static cl::opt<int> CSPreinlMultiplierForPrevInl(
    "csspgo-preinliner-multiplier-for-previous-inlining", cl::Hidden,
    cl::init(100),
    cl::desc(
        "Multiplier to bump up callsite threshold for previous inlining."));

CSPreInliner::CSPreInliner(SampleContextTracker &Tracker,
                           ProfiledBinary &Binary, ProfileSummary *Summary)
    : UseContextCost(UseContextCostForPreInliner),
      // TODO: Pass in a guid-to-name map in order for
      // ContextTracker.getFuncNameFor to work, if `Profiles` can have md5 codes
      // as their profile context.
      ContextTracker(Tracker), Binary(Binary), Summary(Summary) {
  // Set default preinliner hot/cold call site threshold tuned with CSSPGO.
  // for good performance with reasonable profile size.
  if (!SampleHotCallSiteThreshold.getNumOccurrences())
    SampleHotCallSiteThreshold = 1500;
  if (!SampleColdCallSiteThreshold.getNumOccurrences())
    SampleColdCallSiteThreshold = 0;
  if (!ProfileInlineLimitMax.getNumOccurrences())
    ProfileInlineLimitMax = 50000;

  ProfileICPRelativeHotness =
      getSampleProfileICPRelativeHotness(Binary.getLibraryOptions());
  ProfileICPRelativeHotnessSkip =
      getSampleProfileICPRelativeHotnessSkip(Binary.getLibraryOptions());
}

std::vector<FunctionId> CSPreInliner::buildTopDownOrder() {
  std::vector<FunctionId> Order;
  // Trim cold edges to get a more stable call graph. This allows for a more
  // stable top-down order which in turns helps the stablity of the generated
  // profile from run to run.
  uint64_t ColdCountThreshold = ProfileSummaryBuilder::getColdCountThreshold(
      (Summary->getDetailedSummary()));
  ProfiledCallGraph ProfiledCG(ContextTracker, ColdCountThreshold);

  // Now that we have a profiled call graph, construct top-down order
  // by building up SCC and reversing SCC order.
  scc_iterator<ProfiledCallGraph *> I = scc_begin(&ProfiledCG);
  while (!I.isAtEnd()) {
    auto Range = *I;
    if (SortProfiledSCC) {
      // Sort nodes in one SCC based on callsite hotness.
      scc_member_iterator<ProfiledCallGraph *> SI(*I);
      Range = *SI;
    }
    for (auto *Node : Range) {
      if (Node != ProfiledCG.getEntryNode())
        Order.push_back(Node->Name);
    }
    ++I;
  }
  std::reverse(Order.begin(), Order.end());

  return Order;
}

// SampleProfileLoader inlines an indirect call target only after promoting it.
// inlineHotFunctionsWithPriority walks the nested callee profiles at the call
// site in descending head-sample order and promotes a target only if its count
// is hot and, once ProfileICPRelativeHotnessSkip targets were promoted, at
// least ProfileICPRelativeHotness percent of the site total; it stops at the
// first target that fails. A target rejected by these checks should not be
// predicted inline. Its samples would be stranded at the call site, and the
// loader does not merge them back into the callee, so the out-of-line callee
// would be compiled with an almost empty profile.
//
// Predicting a target inline changes how the loader counts it. When we nest
// its profile under the caller, convertCSProfiles removes its call-target map
// entry. The loader then uses the nested profile's head count instead. These
// two counts can differ, changing the call site's total and each target's
// percentage of that total.
// Try targets in the loader's order. For each one, replace its map count with
// its head count, then check that it and all previously accepted targets still
// pass the hotness checks. Otherwise, reject it and the remaining targets.
DenseSet<const ContextTrieNode *>
CSPreInliner::getICPRejectedCallees(ContextTrieNode *CallerNode,
                                    const FunctionSamples *CallerSamples) {
  DenseSet<const ContextTrieNode *> Rejected;
  // Only a pseudo probe identifies a single call instruction. With line-based
  // profiles, distinct direct calls on one source line share a call site, so
  // several targets there do not imply an indirect call.
  if (!CSPreinlHonorICPLimits || SamplePreInlineReplay ||
      !FunctionSamples::ProfileIsProbeBased)
    return Rejected;

  struct Target {
    ContextTrieNode *Node;
    uint64_t HeadCount;
    uint64_t CallTargetCount = 0;
  };
  std::map<LineLocation, SmallVector<Target, 4>> CallSites;
  for (auto &Child : CallerNode->getAllChildContext()) {
    ContextTrieNode *CalleeNode = &Child.second;
    if (FunctionSamples *CalleeSamples = CalleeNode->getFunctionSamples())
      CallSites[CalleeNode->getCallSiteLoc()].push_back(
          {CalleeNode, CalleeSamples->getHeadSamplesEstimate()});
  }

  uint64_t HotCountThreshold = ProfileSummaryBuilder::getHotCountThreshold(
      Summary->getDetailedSummary());
  for (auto &Entry : CallSites) {
    const LineLocation &CallSite = Entry.first;
    auto &Callees = Entry.second;
    // Site total with every target outlined, i.e. the raw call target map.
    uint64_t SiteTotal = 0;
    size_t NumTargets = Callees.size();
    if (auto CallTargets = CallerSamples->findCallTargetMapAt(CallSite)) {
      NumTargets = CallTargets->size();
      for (const auto &[Name, Count] : *CallTargets)
        SiteTotal += Count;
      for (Target &T : Callees) {
        auto It = CallTargets->find(T.Node->getFuncName());
        if (It == CallTargets->end()) {
          ++NumTargets;
          continue;
        }
        T.CallTargetCount = It->second;
      }
    }
    // Only indirect calls are subject to ICP. If the binary has a probe for
    // this call, use its type: it comes from the source call, so it stays
    // indirect after ICP. Otherwise (e.g. the call was inlined everywhere),
    // treat a site with a single target as a direct call.
    std::optional<bool> IsIndirect = Binary.isIndirectCallProbe(
        CallerNode->getFuncName().getHashCode(), CallSite.LineOffset);
    if (IsIndirect ? !*IsIndirect : NumTargets < 2)
      continue;

    llvm::sort(Callees, [](const Target &L, const Target &R) {
      if (L.HeadCount != R.HeadCount)
        return L.HeadCount > R.HeadCount;
      return L.Node->getFunctionSamples()->getGUID() <
             R.Node->getFunctionSamples()->getGUID();
    });
    auto CanPromoteAll = [&](size_t NumCandidates, uint64_t NewTotal) {
      for (size_t I = 0; I < NumCandidates; ++I) {
        uint64_t HeadCount = Callees[I].HeadCount;
        if (HeadCount < HotCountThreshold)
          return false;
        // The first few targets are exempt only from the percentage check.
        if (I < ProfileICPRelativeHotnessSkip)
          continue;
        if (HeadCount * 100 < NewTotal * ProfileICPRelativeHotness)
          return false;
      }
      return true;
    };
    LLVM_DEBUG(dbgs() << "  Indirect call site " << CallSite.LineOffset << "."
                      << CallSite.Discriminator << ": " << NumTargets
                      << " targets, total count " << SiteTotal << "\n");
    size_t NumAccepted = 0;
    for (; NumAccepted < Callees.size(); ++NumAccepted) {
      const Target &T = Callees[NumAccepted];
      uint64_t NewTotal = SiteTotal - T.CallTargetCount + T.HeadCount;
      // The new total must keep this target and the accepted prefix eligible.
      if (!CanPromoteAll(NumAccepted + 1, NewTotal))
        break;
      SiteTotal = NewTotal;
    }
    for (size_t I = NumAccepted; I < Callees.size(); ++I) {
      const Target &T = Callees[I];
      Rejected.insert(T.Node);
      LLVM_DEBUG(dbgs() << "    Not promotable by ICP: "
                        << ContextTracker.getContextString(T.Node) << " (head "
                        << T.HeadCount << ", site total " << SiteTotal
                        << ")\n");
    }
  }
  return Rejected;
}

bool CSPreInliner::getInlineCandidates(ProfiledCandidateQueue &CQueue,
                                       const FunctionSamples *CallerSamples) {
  assert(CallerSamples && "Expect non-null caller samples");

  // Ideally we want to consider everything a function calls, but as far as
  // context profile is concerned, only those frames that are children of
  // current one in the trie is relavent. So we walk the trie instead of call
  // targets from function profile.
  ContextTrieNode *CallerNode =
      ContextTracker.getContextNodeForProfile(CallerSamples);
  DenseSet<const ContextTrieNode *> ICPRejected =
      getICPRejectedCallees(CallerNode, CallerSamples);

  bool HasNewCandidate = false;
  for (auto &Child : CallerNode->getAllChildContext()) {
    ContextTrieNode *CalleeNode = &Child.second;
    FunctionSamples *CalleeSamples = CalleeNode->getFunctionSamples();
    if (!CalleeSamples)
      continue;

    // Call site count is more reliable, so we look up the corresponding call
    // target profile in caller's context profile to retrieve call site count.
    uint64_t CalleeEntryCount = CalleeSamples->getHeadSamplesEstimate();
    uint64_t CallsiteCount = 0;
    LineLocation Callsite = CalleeNode->getCallSiteLoc();
    if (auto CallTargets = CallerSamples->findCallTargetMapAt(Callsite)) {
      auto It = CallTargets->find(CalleeSamples->getFunction());
      if (It != CallTargets->end())
        CallsiteCount = It->second;
    }

    // TODO: call site and callee entry count should be mostly consistent, add
    // check for that.
    HasNewCandidate = true;
    uint32_t CalleeSize = getFuncSize(CalleeNode);
    ProfiledInlineCandidate Candidate(
        CalleeSamples, std::max(CallsiteCount, CalleeEntryCount), CalleeSize);
    Candidate.ICPRejected = ICPRejected.contains(CalleeNode);
    CQueue.push(Candidate);
  }

  return HasNewCandidate;
}

uint32_t CSPreInliner::getFuncSize(const ContextTrieNode *ContextNode) {
  if (UseContextCost)
    return Binary.getFuncSizeForContext(ContextNode);

  return ContextNode->getFunctionSamples()->getBodySamples().size();
}

bool CSPreInliner::shouldInline(ProfiledInlineCandidate &Candidate) {
  bool WasInlined =
      Candidate.CalleeSamples->getContext().hasAttribute(ContextWasInlined);
  // If replay inline is requested, simply follow the inline decision of the
  // profiled binary.
  if (SamplePreInlineReplay)
    return WasInlined;

  if (Candidate.ICPRejected) {
    ++PreInlNumCSICPRejected;
    return false;
  }

  unsigned int SampleThreshold = SampleColdCallSiteThreshold;
  uint64_t ColdCountThreshold = ProfileSummaryBuilder::getColdCountThreshold(
      (Summary->getDetailedSummary()));

  if (Candidate.CallsiteCount <= ColdCountThreshold)
    SampleThreshold = SampleColdCallSiteThreshold;
  else {
    // Linearly adjust threshold based on normalized hotness, i.e, a value in
    // [0,1]. Use 10% cutoff instead of the max count as the normalization
    // upperbound for stability.
    double NormalizationUpperBound =
        ProfileSummaryBuilder::getEntryForPercentile(
            Summary->getDetailedSummary(), 100000 /* 10% */)
            .MinCount;
    double NormalizationLowerBound = ColdCountThreshold;
    double NormalizedHotness =
        (Candidate.CallsiteCount - NormalizationLowerBound) /
        (NormalizationUpperBound - NormalizationLowerBound);
    if (NormalizedHotness > 1.0)
      NormalizedHotness = 1.0;
    // Add 1 to ensure hot callsites get a non-zero threshold, which could
    // happen when SampleColdCallSiteThreshold is 0. This is when we do not
    // want any inlining for cold callsites.
    SampleThreshold = SampleHotCallSiteThreshold * NormalizedHotness * 100 +
                      SampleColdCallSiteThreshold + 1;
    // Bump up the threshold to favor previous compiler inline decision. The
    // compiler has more insight and knowledge about functions based on their IR
    // and attribures and should be able to make a more reasonable inline
    // decision.
    if (WasInlined)
      SampleThreshold *= CSPreinlMultiplierForPrevInl;
  }

  return (Candidate.SizeCost < SampleThreshold);
}

void CSPreInliner::processFunction(const FunctionId Name) {
  FunctionSamples *FSamples = ContextTracker.getBaseSamplesFor(Name);
  if (!FSamples)
    return;

  unsigned FuncSize =
      getFuncSize(ContextTracker.getContextNodeForProfile(FSamples));
  unsigned FuncFinalSize = FuncSize;
  unsigned SizeLimit = FuncSize * ProfileInlineGrowthLimit;
  SizeLimit = std::min(SizeLimit, (unsigned)ProfileInlineLimitMax);
  SizeLimit = std::max(SizeLimit, (unsigned)ProfileInlineLimitMin);

  LLVM_DEBUG(dbgs() << "Process " << Name
                    << " for context-sensitive pre-inlining (pre-inline size: "
                    << FuncSize << ", size limit: " << SizeLimit << ")\n");

  ProfiledCandidateQueue CQueue;
  getInlineCandidates(CQueue, FSamples);

  while (!CQueue.empty() && FuncFinalSize < SizeLimit) {
    ProfiledInlineCandidate Candidate = CQueue.top();
    CQueue.pop();
    bool ShouldInline = false;
    if ((ShouldInline = shouldInline(Candidate))) {
      // We mark context as inlined as the corresponding context profile
      // won't be merged into that function's base profile.
      ++PreInlNumCSInlined;
      ContextTracker.markContextSamplesInlined(Candidate.CalleeSamples);
      Candidate.CalleeSamples->getContext().setAttribute(
          ContextShouldBeInlined);
      FuncFinalSize += Candidate.SizeCost;
      getInlineCandidates(CQueue, Candidate.CalleeSamples);
    } else {
      ++PreInlNumCSNotInlined;
    }
    LLVM_DEBUG(
        dbgs() << (ShouldInline ? "  Inlined" : "  Outlined")
               << " context profile for: "
               << ContextTracker.getContextString(*Candidate.CalleeSamples)
               << " (callee size: " << Candidate.SizeCost
               << ", call count:" << Candidate.CallsiteCount << ")\n");
  }

  if (!CQueue.empty()) {
    if (SizeLimit == (unsigned)ProfileInlineLimitMax)
      ++PreInlNumCSInlinedHitMaxLimit;
    else if (SizeLimit == (unsigned)ProfileInlineLimitMin)
      ++PreInlNumCSInlinedHitMinLimit;
    else
      ++PreInlNumCSInlinedHitGrowthLimit;
  }

  LLVM_DEBUG({
    if (!CQueue.empty())
      dbgs() << "  Inline candidates ignored due to size limit (inliner "
                "original size: "
             << FuncSize << ", inliner final size: " << FuncFinalSize
             << ", size limit: " << SizeLimit << ")\n";

    while (!CQueue.empty()) {
      ProfiledInlineCandidate Candidate = CQueue.top();
      CQueue.pop();
      bool WasInlined =
          Candidate.CalleeSamples->getContext().hasAttribute(ContextWasInlined);
      dbgs() << "    "
             << ContextTracker.getContextString(*Candidate.CalleeSamples)
             << " (candidate size:" << Candidate.SizeCost
             << ", call count: " << Candidate.CallsiteCount << ", previously "
             << (WasInlined ? "inlined)\n" : "not inlined)\n");
    }
  });
}

void CSPreInliner::run() {
#ifndef NDEBUG
  auto printProfileNames = [](SampleContextTracker &ContextTracker,
                              bool IsInput) {
    uint32_t Size = 0;
    for (auto *Node : ContextTracker) {
      FunctionSamples *FSamples = Node->getFunctionSamples();
      if (FSamples) {
        Size++;
        dbgs() << "  [" << ContextTracker.getContextString(Node) << "] "
               << FSamples->getTotalSamples() << ":"
               << FSamples->getHeadSamples() << "\n";
      }
    }
    dbgs() << (IsInput ? "Input" : "Output") << " context-sensitive profiles ("
           << Size << " total):\n";
  };
#endif

  LLVM_DEBUG(printProfileNames(ContextTracker, true));

  // Execute global pre-inliner to estimate a global top-down inline
  // decision and merge profiles accordingly. This helps with profile
  // merge for ThinLTO otherwise we won't be able to merge profiles back
  // to base profile across module/thin-backend boundaries.
  // It also helps better compress context profile to control profile
  // size, as we now only need context profile for functions going to
  // be inlined.
  for (FunctionId FuncName : buildTopDownOrder()) {
    processFunction(FuncName);
  }

  // Not inlined context profiles are merged into its base, so we can
  // trim out such profiles from the output.
  for (auto *Node : ContextTracker) {
    FunctionSamples *FProfile = Node->getFunctionSamples();
    if (FProfile &&
        (Node->getParentContext() != &ContextTracker.getRootContext() &&
         !FProfile->getContext().hasState(InlinedContext))) {
      Node->setFunctionSamples(nullptr);
    }
  }
  FunctionSamples::ProfileIsPreInlined = true;

  LLVM_DEBUG(printProfileNames(ContextTracker, false));
}
