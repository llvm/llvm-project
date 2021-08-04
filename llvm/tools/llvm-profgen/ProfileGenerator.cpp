//===-- ProfileGenerator.cpp - Profile Generator  ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProfileGenerator.h"
#include "llvm/ProfileData/ProfileCommon.h"

static cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                           cl::Required,
                                           cl::desc("Output profile file"));
static cl::alias OutputA("o", cl::desc("Alias for --output"),
                         cl::aliasopt(OutputFilename));

static cl::opt<SampleProfileFormat> OutputFormat(
    "format", cl::desc("Format of output profile"), cl::init(SPF_Ext_Binary),
    cl::values(
        clEnumValN(SPF_Binary, "binary", "Binary encoding (default)"),
        clEnumValN(SPF_Compact_Binary, "compbinary", "Compact binary encoding"),
        clEnumValN(SPF_Ext_Binary, "extbinary", "Extensible binary encoding"),
        clEnumValN(SPF_Text, "text", "Text encoding"),
        clEnumValN(SPF_GCC, "gcc",
                   "GCC encoding (only meaningful for -sample)")));

static cl::opt<int32_t, true> RecursionCompression(
    "compress-recursion",
    cl::desc("Compressing recursion by deduplicating adjacent frame "
             "sequences up to the specified size. -1 means no size limit."),
    cl::Hidden,
    cl::location(llvm::sampleprof::CSProfileGenerator::MaxCompressionSize));

static cl::opt<bool> CSProfMergeColdContext(
    "csprof-merge-cold-context", cl::init(true), cl::ZeroOrMore,
    cl::desc("If the total count of context profile is smaller than "
             "the threshold, it will be merged into context-less base "
             "profile."));

static cl::opt<bool> CSProfTrimColdContext(
    "csprof-trim-cold-context", cl::init(true), cl::ZeroOrMore,
    cl::desc("If the total count of the profile after all merge is done "
             "is still smaller than threshold, it will be trimmed."));

static cl::opt<uint32_t> CSProfColdContextFrameDepth(
    "csprof-frame-depth-for-cold-context", cl::init(1), cl::ZeroOrMore,
    cl::desc("Keep the last K frames while merging cold profile. 1 means the "
             "context-less base profile"));

static cl::opt<bool> EnableCSPreInliner(
    "csspgo-preinliner", cl::Hidden, cl::init(false),
    cl::desc("Run a global pre-inliner to merge context profile based on "
             "estimated global top-down inline decisions"));

extern cl::opt<int> ProfileSummaryCutoffCold;

using namespace llvm;
using namespace sampleprof;

namespace llvm {
namespace sampleprof {

// Initialize the MaxCompressionSize to -1 which means no size limit
int32_t CSProfileGenerator::MaxCompressionSize = -1;

static bool
usePseudoProbes(const BinarySampleCounterMap &BinarySampleCounters) {
  return BinarySampleCounters.size() &&
         BinarySampleCounters.begin()->first->usePseudoProbes();
}

std::unique_ptr<ProfileGenerator>
ProfileGenerator::create(const BinarySampleCounterMap &BinarySampleCounters,
                         enum PerfScriptType SampleType) {
  std::unique_ptr<ProfileGenerator> ProfileGenerator;
  if (SampleType == PERF_LBR_STACK) {
    if (usePseudoProbes(BinarySampleCounters)) {
      ProfileGenerator.reset(
          new PseudoProbeCSProfileGenerator(BinarySampleCounters));
    } else {
      ProfileGenerator.reset(new CSProfileGenerator(BinarySampleCounters));
    }
  } else {
    // TODO:
    llvm_unreachable("Unsupported perfscript!");
  }

  return ProfileGenerator;
}

void ProfileGenerator::write(std::unique_ptr<SampleProfileWriter> Writer,
                             StringMap<FunctionSamples> &ProfileMap) {
  if (std::error_code EC = Writer->write(ProfileMap))
    exitWithError(std::move(EC));
}

void ProfileGenerator::write() {
  auto WriterOrErr = SampleProfileWriter::create(OutputFilename, OutputFormat);
  if (std::error_code EC = WriterOrErr.getError())
    exitWithError(EC, OutputFilename);
  write(std::move(WriterOrErr.get()), ProfileMap);
}

void ProfileGenerator::findDisjointRanges(RangeSample &DisjointRanges,
                                          const RangeSample &Ranges) {

  /*
  Regions may overlap with each other. Using the boundary info, find all
  disjoint ranges and their sample count. BoundaryPoint contains the count
  multiple samples begin/end at this points.

  |<--100-->|           Sample1
  |<------200------>|   Sample2
  A         B       C

  In the example above,
  Sample1 begins at A, ends at B, its value is 100.
  Sample2 beings at A, ends at C, its value is 200.
  For A, BeginCount is the sum of sample begins at A, which is 300 and no
  samples ends at A, so EndCount is 0.
  Then boundary points A, B, and C with begin/end counts are:
  A: (300, 0)
  B: (0, 100)
  C: (0, 200)
  */
  struct BoundaryPoint {
    // Sum of sample counts beginning at this point
    uint64_t BeginCount;
    // Sum of sample counts ending at this point
    uint64_t EndCount;

    BoundaryPoint() : BeginCount(0), EndCount(0){};

    void addBeginCount(uint64_t Count) { BeginCount += Count; }

    void addEndCount(uint64_t Count) { EndCount += Count; }
  };

  /*
  For the above example. With boundary points, follwing logic finds two
  disjoint region of

  [A,B]:   300
  [B+1,C]: 200

  If there is a boundary point that both begin and end, the point itself
  becomes a separate disjoint region. For example, if we have original
  ranges of

  |<--- 100 --->|
                |<--- 200 --->|
  A             B             C

  there are three boundary points with their begin/end counts of

  A: (100, 0)
  B: (200, 100)
  C: (0, 200)

  the disjoint ranges would be

  [A, B-1]: 100
  [B, B]:   300
  [B+1, C]: 200.
  */
  std::map<uint64_t, BoundaryPoint> Boundaries;

  for (auto Item : Ranges) {
    uint64_t Begin = Item.first.first;
    uint64_t End = Item.first.second;
    uint64_t Count = Item.second;
    if (Boundaries.find(Begin) == Boundaries.end())
      Boundaries[Begin] = BoundaryPoint();
    Boundaries[Begin].addBeginCount(Count);

    if (Boundaries.find(End) == Boundaries.end())
      Boundaries[End] = BoundaryPoint();
    Boundaries[End].addEndCount(Count);
  }

  uint64_t BeginAddress = UINT64_MAX;
  int Count = 0;
  for (auto Item : Boundaries) {
    uint64_t Address = Item.first;
    BoundaryPoint &Point = Item.second;
    if (Point.BeginCount) {
      if (BeginAddress != UINT64_MAX)
        DisjointRanges[{BeginAddress, Address - 1}] = Count;
      Count += Point.BeginCount;
      BeginAddress = Address;
    }
    if (Point.EndCount) {
      assert((BeginAddress != UINT64_MAX) &&
             "First boundary point cannot be 'end' point");
      DisjointRanges[{BeginAddress, Address}] = Count;
      Count -= Point.EndCount;
      BeginAddress = Address + 1;
    }
  }
}

FunctionSamples &
CSProfileGenerator::getFunctionProfileForContext(StringRef ContextStr,
                                                 bool WasLeafInlined) {
  auto Ret = ProfileMap.try_emplace(ContextStr, FunctionSamples());
  if (Ret.second) {
    // Make a copy of the underlying context string in string table
    // before StringRef wrapper is used for context.
    auto It = ContextStrings.insert(ContextStr.str());
    SampleContext FContext(*It.first, RawContext);
    if (WasLeafInlined)
      FContext.setAttribute(ContextWasInlined);
    FunctionSamples &FProfile = Ret.first->second;
    FProfile.setContext(FContext);
    FProfile.setName(FContext.getNameWithoutContext());
  }
  return Ret.first->second;
}

void CSProfileGenerator::generateProfile() {
  FunctionSamples::ProfileIsCS = true;
  for (const auto &BI : BinarySampleCounters) {
    ProfiledBinary *Binary = BI.first;
    for (const auto &CI : BI.second) {
      const StringBasedCtxKey *CtxKey =
          dyn_cast<StringBasedCtxKey>(CI.first.getPtr());
      StringRef ContextId(CtxKey->Context);
      // Get or create function profile for the range
      FunctionSamples &FunctionProfile =
          getFunctionProfileForContext(ContextId, CtxKey->WasLeafInlined);

      // Fill in function body samples
      populateFunctionBodySamples(FunctionProfile, CI.second.RangeCounter,
                                  Binary);
      // Fill in boundary sample counts as well as call site samples for calls
      populateFunctionBoundarySamples(ContextId, FunctionProfile,
                                      CI.second.BranchCounter, Binary);
    }
  }
  // Fill in call site value sample for inlined calls and also use context to
  // infer missing samples. Since we don't have call count for inlined
  // functions, we estimate it from inlinee's profile using the entry of the
  // body sample.
  populateInferredFunctionSamples();

  postProcessProfiles();
}

void CSProfileGenerator::updateBodySamplesforFunctionProfile(
    FunctionSamples &FunctionProfile, const FrameLocation &LeafLoc,
    uint64_t Count) {
  // Filter out invalid negative(int type) lineOffset
  if (LeafLoc.second.LineOffset & 0x80000000)
    return;
  // Use the maximum count of samples with same line location
  ErrorOr<uint64_t> R = FunctionProfile.findSamplesAt(
      LeafLoc.second.LineOffset, LeafLoc.second.Discriminator);
  uint64_t PreviousCount = R ? R.get() : 0;
  if (PreviousCount < Count) {
    FunctionProfile.addBodySamples(LeafLoc.second.LineOffset,
                                   LeafLoc.second.Discriminator,
                                   Count - PreviousCount);
  }
}

void CSProfileGenerator::populateFunctionBodySamples(
    FunctionSamples &FunctionProfile, const RangeSample &RangeCounter,
    ProfiledBinary *Binary) {
  // Compute disjoint ranges first, so we can use MAX
  // for calculating count for each location.
  RangeSample Ranges;
  findDisjointRanges(Ranges, RangeCounter);
  for (auto Range : Ranges) {
    uint64_t RangeBegin = Binary->offsetToVirtualAddr(Range.first.first);
    uint64_t RangeEnd = Binary->offsetToVirtualAddr(Range.first.second);
    uint64_t Count = Range.second;
    // Disjoint ranges have introduce zero-filled gap that
    // doesn't belong to current context, filter them out.
    if (Count == 0)
      continue;

    InstructionPointer IP(Binary, RangeBegin, true);

    // Disjoint ranges may have range in the middle of two instr,
    // e.g. If Instr1 at Addr1, and Instr2 at Addr2, disjoint range
    // can be Addr1+1 to Addr2-1. We should ignore such range.
    if (IP.Address > RangeEnd)
      continue;

    while (IP.Address <= RangeEnd) {
      uint64_t Offset = Binary->virtualAddrToOffset(IP.Address);
      auto LeafLoc = Binary->getInlineLeafFrameLoc(Offset);
      if (LeafLoc.hasValue()) {
        // Recording body sample for this specific context
        updateBodySamplesforFunctionProfile(FunctionProfile, *LeafLoc, Count);
      }
      // Accumulate total sample count even it's a line with invalid debug info
      FunctionProfile.addTotalSamples(Count);
      // Move to next IP within the range
      IP.advance();
    }
  }
}

void CSProfileGenerator::populateFunctionBoundarySamples(
    StringRef ContextId, FunctionSamples &FunctionProfile,
    const BranchSample &BranchCounters, ProfiledBinary *Binary) {

  for (auto Entry : BranchCounters) {
    uint64_t SourceOffset = Entry.first.first;
    uint64_t TargetOffset = Entry.first.second;
    uint64_t Count = Entry.second;
    // Get the callee name by branch target if it's a call branch
    StringRef CalleeName = FunctionSamples::getCanonicalFnName(
        Binary->getFuncFromStartOffset(TargetOffset));
    if (CalleeName.size() == 0)
      continue;

    // Record called target sample and its count
    auto LeafLoc = Binary->getInlineLeafFrameLoc(SourceOffset);
    if (!LeafLoc.hasValue())
      continue;
    FunctionProfile.addCalledTargetSamples(LeafLoc->second.LineOffset,
                                           LeafLoc->second.Discriminator,
                                           CalleeName, Count);

    // Record head sample for called target(callee)
    std::ostringstream OCalleeCtxStr;
    if (ContextId.find(" @ ") != StringRef::npos) {
      OCalleeCtxStr << ContextId.rsplit(" @ ").first.str();
      OCalleeCtxStr << " @ ";
    }
    OCalleeCtxStr << getCallSite(*LeafLoc) << " @ " << CalleeName.str();

    FunctionSamples &CalleeProfile =
        getFunctionProfileForContext(OCalleeCtxStr.str());
    assert(Count != 0 && "Unexpected zero weight branch");
    CalleeProfile.addHeadSamples(Count);
  }
}

static FrameLocation getCallerContext(StringRef CalleeContext,
                                      StringRef &CallerNameWithContext) {
  StringRef CallerContext = CalleeContext.rsplit(" @ ").first;
  CallerNameWithContext = CallerContext.rsplit(':').first;
  auto ContextSplit = CallerContext.rsplit(" @ ");
  StringRef CallerFrameStr = ContextSplit.second.size() == 0
                                 ? ContextSplit.first
                                 : ContextSplit.second;
  FrameLocation LeafFrameLoc = {"", {0, 0}};
  StringRef Funcname;
  SampleContext::decodeContextString(CallerFrameStr, Funcname,
                                     LeafFrameLoc.second);
  LeafFrameLoc.first = Funcname.str();
  return LeafFrameLoc;
}

void CSProfileGenerator::populateInferredFunctionSamples() {
  for (const auto &Item : ProfileMap) {
    const StringRef CalleeContext = Item.first();
    const FunctionSamples &CalleeProfile = Item.second;

    // If we already have head sample counts, we must have value profile
    // for call sites added already. Skip to avoid double counting.
    if (CalleeProfile.getHeadSamples())
      continue;
    // If we don't have context, nothing to do for caller's call site.
    // This could happen for entry point function.
    if (CalleeContext.find(" @ ") == StringRef::npos)
      continue;

    // Infer Caller's frame loc and context ID through string splitting
    StringRef CallerContextId;
    FrameLocation &&CallerLeafFrameLoc =
        getCallerContext(CalleeContext, CallerContextId);

    // It's possible that we haven't seen any sample directly in the caller,
    // in which case CallerProfile will not exist. But we can't modify
    // ProfileMap while iterating it.
    // TODO: created function profile for those callers too
    if (ProfileMap.find(CallerContextId) == ProfileMap.end())
      continue;
    FunctionSamples &CallerProfile = ProfileMap[CallerContextId];

    // Since we don't have call count for inlined functions, we
    // estimate it from inlinee's profile using entry body sample.
    uint64_t EstimatedCallCount = CalleeProfile.getEntrySamples();
    // If we don't have samples with location, use 1 to indicate live.
    if (!EstimatedCallCount && !CalleeProfile.getBodySamples().size())
      EstimatedCallCount = 1;
    CallerProfile.addCalledTargetSamples(
        CallerLeafFrameLoc.second.LineOffset,
        CallerLeafFrameLoc.second.Discriminator,
        CalleeProfile.getContext().getNameWithoutContext(), EstimatedCallCount);
    CallerProfile.addBodySamples(CallerLeafFrameLoc.second.LineOffset,
                                 CallerLeafFrameLoc.second.Discriminator,
                                 EstimatedCallCount);
    CallerProfile.addTotalSamples(EstimatedCallCount);
  }
}

void CSProfileGenerator::postProcessProfiles() {
  // Compute hot/cold threshold based on profile. This will be used for cold
  // context profile merging/trimming.
  computeSummaryAndThreshold();

  // Run global pre-inliner to adjust/merge context profile based on estimated
  // inline decisions.
  if (EnableCSPreInliner)
    CSPreInliner(ProfileMap, HotCountThreshold, ColdCountThreshold).run();

  // Trim and merge cold context profile using cold threshold above;
  SampleContextTrimmer(ProfileMap)
      .trimAndMergeColdContextProfiles(
          ColdCountThreshold, CSProfTrimColdContext, CSProfMergeColdContext,
          CSProfColdContextFrameDepth);
}

void CSProfileGenerator::computeSummaryAndThreshold() {
  // Update the default value of cold cutoff for llvm-profgen.
  // Do it here because we don't want to change the global default,
  // which would lead CS profile size too large.
  if (!ProfileSummaryCutoffCold.getNumOccurrences())
    ProfileSummaryCutoffCold = 999000;

  SampleProfileSummaryBuilder Builder(ProfileSummaryBuilder::DefaultCutoffs);
  auto Summary = Builder.computeSummaryForProfiles(ProfileMap);
  HotCountThreshold = ProfileSummaryBuilder::getHotCountThreshold(
      (Summary->getDetailedSummary()));
  ColdCountThreshold = ProfileSummaryBuilder::getColdCountThreshold(
      (Summary->getDetailedSummary()));
}

void CSProfileGenerator::write(std::unique_ptr<SampleProfileWriter> Writer,
                               StringMap<FunctionSamples> &ProfileMap) {
  if (std::error_code EC = Writer->write(ProfileMap))
    exitWithError(std::move(EC));
}

// Helper function to extract context prefix string stack
// Extract context stack for reusing, leaf context stack will
// be added compressed while looking up function profile
static void
extractPrefixContextStack(SmallVectorImpl<std::string> &ContextStrStack,
    const SmallVectorImpl<const MCDecodedPseudoProbe *> &Probes,
    ProfiledBinary *Binary) {
  for (const auto *P : Probes) {
    Binary->getInlineContextForProbe(P, ContextStrStack, true);
  }
}

void PseudoProbeCSProfileGenerator::generateProfile() {
  // Enable pseudo probe functionalities in SampleProf
  FunctionSamples::ProfileIsProbeBased = true;
  FunctionSamples::ProfileIsCS = true;
  for (const auto &BI : BinarySampleCounters) {
    ProfiledBinary *Binary = BI.first;
    for (const auto &CI : BI.second) {
      const ProbeBasedCtxKey *CtxKey =
          dyn_cast<ProbeBasedCtxKey>(CI.first.getPtr());
      SmallVector<std::string, 16> ContextStrStack;
      extractPrefixContextStack(ContextStrStack, CtxKey->Probes, Binary);
      // Fill in function body samples from probes, also infer caller's samples
      // from callee's probe
      populateBodySamplesWithProbes(CI.second.RangeCounter, ContextStrStack,
                                    Binary);
      // Fill in boundary samples for a call probe
      populateBoundarySamplesWithProbes(CI.second.BranchCounter,
                                        ContextStrStack, Binary);
    }
  }

  postProcessProfiles();
}

void PseudoProbeCSProfileGenerator::extractProbesFromRange(
    const RangeSample &RangeCounter, ProbeCounterMap &ProbeCounter,
    ProfiledBinary *Binary) {
  RangeSample Ranges;
  findDisjointRanges(Ranges, RangeCounter);
  for (const auto &Range : Ranges) {
    uint64_t RangeBegin = Binary->offsetToVirtualAddr(Range.first.first);
    uint64_t RangeEnd = Binary->offsetToVirtualAddr(Range.first.second);
    uint64_t Count = Range.second;
    // Disjoint ranges have introduce zero-filled gap that
    // doesn't belong to current context, filter them out.
    if (Count == 0)
      continue;

    InstructionPointer IP(Binary, RangeBegin, true);

    // Disjoint ranges may have range in the middle of two instr,
    // e.g. If Instr1 at Addr1, and Instr2 at Addr2, disjoint range
    // can be Addr1+1 to Addr2-1. We should ignore such range.
    if (IP.Address > RangeEnd)
      continue;

    while (IP.Address <= RangeEnd) {
      const AddressProbesMap &Address2ProbesMap =
          Binary->getAddress2ProbesMap();
      auto It = Address2ProbesMap.find(IP.Address);
      if (It != Address2ProbesMap.end()) {
        for (const auto &Probe : It->second) {
          if (!Probe.isBlock())
            continue;
          ProbeCounter[&Probe] += Count;
        }
      }

      IP.advance();
    }
  }
}

void PseudoProbeCSProfileGenerator::populateBodySamplesWithProbes(
    const RangeSample &RangeCounter,
    SmallVectorImpl<std::string> &ContextStrStack, ProfiledBinary *Binary) {
  ProbeCounterMap ProbeCounter;
  // Extract the top frame probes by looking up each address among the range in
  // the Address2ProbeMap
  extractProbesFromRange(RangeCounter, ProbeCounter, Binary);
  std::unordered_map<MCDecodedPseudoProbeInlineTree *, FunctionSamples *>
      FrameSamples;
  for (auto PI : ProbeCounter) {
    const MCDecodedPseudoProbe *Probe = PI.first;
    uint64_t Count = PI.second;
    FunctionSamples &FunctionProfile =
        getFunctionProfileForLeafProbe(ContextStrStack, Probe, Binary);
    // Record the current frame and FunctionProfile whenever samples are
    // collected for non-danglie probes. This is for reporting all of the
    // zero count probes of the frame later.
    FrameSamples[Probe->getInlineTreeNode()] = &FunctionProfile;
    FunctionProfile.addBodySamplesForProbe(Probe->getIndex(), Count);
    FunctionProfile.addTotalSamples(Count);
    if (Probe->isEntry()) {
      FunctionProfile.addHeadSamples(Count);
      // Look up for the caller's function profile
      const auto *InlinerDesc = Binary->getInlinerDescForProbe(Probe);
      if (InlinerDesc != nullptr) {
        // Since the context id will be compressed, we have to use callee's
        // context id to infer caller's context id to ensure they share the
        // same context prefix.
        StringRef CalleeContextId =
            FunctionProfile.getContext().getNameWithContext();
        StringRef CallerContextId;
        FrameLocation &&CallerLeafFrameLoc =
            getCallerContext(CalleeContextId, CallerContextId);
        uint64_t CallerIndex = CallerLeafFrameLoc.second.LineOffset;
        assert(CallerIndex &&
               "Inferred caller's location index shouldn't be zero!");
        FunctionSamples &CallerProfile =
            getFunctionProfileForContext(CallerContextId);
        CallerProfile.setFunctionHash(InlinerDesc->FuncHash);
        CallerProfile.addBodySamples(CallerIndex, 0, Count);
        CallerProfile.addTotalSamples(Count);
        CallerProfile.addCalledTargetSamples(
            CallerIndex, 0,
            FunctionProfile.getContext().getNameWithoutContext(), Count);
      }
    }

    // Assign zero count for remaining probes without sample hits to
    // differentiate from probes optimized away, of which the counts are unknown
    // and will be inferred by the compiler.
    for (auto &I : FrameSamples) {
      auto *FunctionProfile = I.second;
      for (auto *Probe : I.first->getProbes()) {
        FunctionProfile->addBodySamplesForProbe(Probe->getIndex(), 0);
      }
    }
  }
}

void PseudoProbeCSProfileGenerator::populateBoundarySamplesWithProbes(
    const BranchSample &BranchCounter,
    SmallVectorImpl<std::string> &ContextStrStack, ProfiledBinary *Binary) {
  for (auto BI : BranchCounter) {
    uint64_t SourceOffset = BI.first.first;
    uint64_t TargetOffset = BI.first.second;
    uint64_t Count = BI.second;
    uint64_t SourceAddress = Binary->offsetToVirtualAddr(SourceOffset);
    const MCDecodedPseudoProbe *CallProbe =
        Binary->getCallProbeForAddr(SourceAddress);
    if (CallProbe == nullptr)
      continue;
    FunctionSamples &FunctionProfile =
        getFunctionProfileForLeafProbe(ContextStrStack, CallProbe, Binary);
    FunctionProfile.addBodySamples(CallProbe->getIndex(), 0, Count);
    FunctionProfile.addTotalSamples(Count);
    StringRef CalleeName = FunctionSamples::getCanonicalFnName(
        Binary->getFuncFromStartOffset(TargetOffset));
    if (CalleeName.size() == 0)
      continue;
    FunctionProfile.addCalledTargetSamples(CallProbe->getIndex(), 0, CalleeName,
                                           Count);
  }
}

FunctionSamples &PseudoProbeCSProfileGenerator::getFunctionProfileForLeafProbe(
    SmallVectorImpl<std::string> &ContextStrStack,
    const MCPseudoProbeFuncDesc *LeafFuncDesc, bool WasLeafInlined) {
  assert(ContextStrStack.size() && "Profile context must have the leaf frame");
  // Compress the context string except for the leaf frame
  std::string LeafFrame = ContextStrStack.back();
  ContextStrStack.pop_back();
  CSProfileGenerator::compressRecursionContext(ContextStrStack);

  std::ostringstream OContextStr;
  for (uint32_t I = 0; I < ContextStrStack.size(); I++) {
    if (OContextStr.str().size())
      OContextStr << " @ ";
    OContextStr << ContextStrStack[I];
  }
  // For leaf inlined context with the top frame, we should strip off the top
  // frame's probe id, like:
  // Inlined stack: [foo:1, bar:2], the ContextId will be "foo:1 @ bar"
  if (OContextStr.str().size())
    OContextStr << " @ ";
  OContextStr << StringRef(LeafFrame).split(":").first.str();

  FunctionSamples &FunctionProile =
      getFunctionProfileForContext(OContextStr.str(), WasLeafInlined);
  FunctionProile.setFunctionHash(LeafFuncDesc->FuncHash);
  return FunctionProile;
}

FunctionSamples &PseudoProbeCSProfileGenerator::getFunctionProfileForLeafProbe(
    SmallVectorImpl<std::string> &ContextStrStack,
    const MCDecodedPseudoProbe *LeafProbe, ProfiledBinary *Binary) {

  // Explicitly copy the context for appending the leaf context
  SmallVector<std::string, 16> ContextStrStackCopy(ContextStrStack.begin(),
                                                   ContextStrStack.end());
  Binary->getInlineContextForProbe(LeafProbe, ContextStrStackCopy, true);
  const auto *FuncDesc = Binary->getFuncDescForGUID(LeafProbe->getGuid());
  bool WasLeafInlined = LeafProbe->getInlineTreeNode()->hasInlineSite();
  return getFunctionProfileForLeafProbe(ContextStrStackCopy, FuncDesc,
                                        WasLeafInlined);
}

} // end namespace sampleprof
} // end namespace llvm
