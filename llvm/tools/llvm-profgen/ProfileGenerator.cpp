//===-- ProfileGenerator.cpp - Profile Generator  ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProfileGenerator.h"
#include "ProfiledBinary.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include <unordered_set>

cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
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

cl::opt<bool> UseMD5(
    "use-md5", cl::init(false), cl::Hidden,
    cl::desc("Use md5 to represent function names in the output profile (only "
             "meaningful for -extbinary)"));

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
    "csprof-trim-cold-context", cl::init(false), cl::ZeroOrMore,
    cl::desc("If the total count of the profile after all merge is done "
             "is still smaller than threshold, it will be trimmed."));

static cl::opt<uint32_t> CSProfMaxColdContextDepth(
    "csprof-max-cold-context-depth", cl::init(1), cl::ZeroOrMore,
    cl::desc("Keep the last K contexts while merging cold profile. 1 means the "
             "context-less base profile"));

static cl::opt<int, true> CSProfMaxContextDepth(
    "csprof-max-context-depth", cl::ZeroOrMore,
    cl::desc("Keep the last K contexts while merging profile. -1 means no "
             "depth limit."),
    cl::location(llvm::sampleprof::CSProfileGenerator::MaxContextDepth));

extern cl::opt<int> ProfileSummaryCutoffCold;

using namespace llvm;
using namespace sampleprof;

namespace llvm {
namespace sampleprof {

// Initialize the MaxCompressionSize to -1 which means no size limit
int32_t CSProfileGenerator::MaxCompressionSize = -1;

int CSProfileGenerator::MaxContextDepth = -1;

std::unique_ptr<ProfileGeneratorBase>
ProfileGeneratorBase::create(ProfiledBinary *Binary,
                             const ContextSampleCounterMap &SampleCounters,
                             enum PerfScriptType SampleType) {
  std::unique_ptr<ProfileGeneratorBase> Generator;
  if (SampleType == PERF_LBR) {
    // TODO: Support probe based profile generation
    Generator.reset(new ProfileGenerator(Binary, SampleCounters));
  } else if (SampleType == PERF_LBR_STACK) {
    Generator.reset(new CSProfileGenerator(Binary, SampleCounters));
  } else {
    llvm_unreachable("Unsupported perfscript!");
  }

  return Generator;
}

void ProfileGeneratorBase::write(std::unique_ptr<SampleProfileWriter> Writer,
                                 SampleProfileMap &ProfileMap) {
  if (std::error_code EC = Writer->write(ProfileMap))
    exitWithError(std::move(EC));
}

void ProfileGeneratorBase::write() {
  auto WriterOrErr = SampleProfileWriter::create(OutputFilename, OutputFormat);
  if (std::error_code EC = WriterOrErr.getError())
    exitWithError(EC, OutputFilename);

  if (UseMD5) {
    if (OutputFormat != SPF_Ext_Binary)
      WithColor::warning() << "-use-md5 is ignored. Specify "
                              "--format=extbinary to enable it\n";
    else
      WriterOrErr.get()->setUseMD5();
  }

  write(std::move(WriterOrErr.get()), ProfileMap);
}

void ProfileGeneratorBase::findDisjointRanges(RangeSample &DisjointRanges,
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
    assert(Begin <= End && "Invalid instruction range");
    uint64_t Count = Item.second;
    if (Boundaries.find(Begin) == Boundaries.end())
      Boundaries[Begin] = BoundaryPoint();
    Boundaries[Begin].addBeginCount(Count);

    if (Boundaries.find(End) == Boundaries.end())
      Boundaries[End] = BoundaryPoint();
    Boundaries[End].addEndCount(Count);
  }

  uint64_t BeginAddress = UINT64_MAX;
  uint64_t Count = 0;
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
      assert(Count >= Point.EndCount && "Mismatched live ranges");
      Count -= Point.EndCount;
      BeginAddress = Address + 1;
    }
  }
}

void ProfileGeneratorBase::updateBodySamplesforFunctionProfile(
    FunctionSamples &FunctionProfile, const SampleContextFrame &LeafLoc,
    uint64_t Count) {
  // Filter out invalid negative(int type) lineOffset
  if (LeafLoc.Callsite.LineOffset & 0x80000000)
    return;
  // Use the maximum count of samples with same line location
  ErrorOr<uint64_t> R = FunctionProfile.findSamplesAt(
      LeafLoc.Callsite.LineOffset, LeafLoc.Callsite.Discriminator);
  uint64_t PreviousCount = R ? R.get() : 0;
  if (PreviousCount < Count) {
    FunctionProfile.addBodySamples(LeafLoc.Callsite.LineOffset,
                                   LeafLoc.Callsite.Discriminator,
                                   Count - PreviousCount);
  }
}

FunctionSamples &
ProfileGenerator::getTopLevelFunctionProfile(StringRef FuncName) {
  SampleContext Context(FuncName);
  auto Ret = ProfileMap.emplace(Context, FunctionSamples());
  if (Ret.second) {
    FunctionSamples &FProfile = Ret.first->second;
    FProfile.setContext(Context);
  }
  return Ret.first->second;
}

void ProfileGenerator::generateProfile() {
  if (Binary->usePseudoProbes()) {
    // TODO: Support probe based profile generation
  } else {
    generateLineNumBasedProfile();
  }
}

void ProfileGenerator::generateLineNumBasedProfile() {
  assert(SampleCounters.size() == 1 &&
         "Must have one entry for profile generation.");
  const SampleCounter &SC = SampleCounters.begin()->second;
  // Fill in function body samples
  populateBodySamplesForAllFunctions(SC.RangeCounter);
  // Fill in boundary sample counts as well as call site samples for calls
  populateBoundarySamplesForAllFunctions(SC.BranchCounter);
}

FunctionSamples &ProfileGenerator::getLeafProfileAndAddTotalSamples(
    const SampleContextFrameVector &FrameVec, uint64_t Count) {
  // Get top level profile
  FunctionSamples *FunctionProfile =
      &getTopLevelFunctionProfile(FrameVec[0].CallerName);
  FunctionProfile->addTotalSamples(Count);

  for (size_t I = 1; I < FrameVec.size(); I++) {
    FunctionSamplesMap &SamplesMap =
        FunctionProfile->functionSamplesAt(FrameVec[I - 1].Callsite);
    auto Ret =
        SamplesMap.emplace(FrameVec[I].CallerName.str(), FunctionSamples());
    if (Ret.second) {
      SampleContext Context(FrameVec[I].CallerName);
      Ret.first->second.setContext(Context);
    }
    FunctionProfile = &Ret.first->second;
    FunctionProfile->addTotalSamples(Count);
  }

  return *FunctionProfile;
}

void ProfileGenerator::populateBodySamplesForAllFunctions(
    const RangeSample &RangeCounter) {
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
    while (IP.Address <= RangeEnd) {
      uint64_t Offset = Binary->virtualAddrToOffset(IP.Address);
      const SampleContextFrameVector &FrameVec =
          Binary->getFrameLocationStack(Offset);
      if (!FrameVec.empty()) {
        FunctionSamples &FunctionProfile =
            getLeafProfileAndAddTotalSamples(FrameVec, Count);
        updateBodySamplesforFunctionProfile(FunctionProfile, FrameVec.back(),
                                            Count);
      }
      // Move to next IP within the range.
      IP.advance();
    }
  }
}

void ProfileGenerator::populateBoundarySamplesForAllFunctions(
    const BranchSample &BranchCounters) {
  for (auto Entry : BranchCounters) {
    uint64_t SourceOffset = Entry.first.first;
    uint64_t TargetOffset = Entry.first.second;
    uint64_t Count = Entry.second;
    assert(Count != 0 && "Unexpected zero weight branch");

    // Get the callee name by branch target if it's a call branch.
    StringRef CalleeName = FunctionSamples::getCanonicalFnName(
        Binary->getFuncFromStartOffset(TargetOffset));
    if (CalleeName.size() == 0)
      continue;
    // Record called target sample and its count.
    const SampleContextFrameVector &FrameVec =
        Binary->getFrameLocationStack(SourceOffset);
    if (!FrameVec.empty()) {
      FunctionSamples &FunctionProfile =
          getLeafProfileAndAddTotalSamples(FrameVec, Count);
      FunctionProfile.addCalledTargetSamples(
          FrameVec.back().Callsite.LineOffset,
          FrameVec.back().Callsite.Discriminator, CalleeName, Count);
    }
    // Add head samples for callee.
    FunctionSamples &CalleeProfile = getTopLevelFunctionProfile(CalleeName);
    CalleeProfile.addHeadSamples(Count);
  }
}

FunctionSamples &CSProfileGenerator::getFunctionProfileForContext(
    const SampleContextFrameVector &Context, bool WasLeafInlined) {
  auto I = ProfileMap.find(SampleContext(Context));
  if (I == ProfileMap.end()) {
    // Save the new context for future references.
    SampleContextFrames NewContext = *Contexts.insert(Context).first;
    SampleContext FContext(NewContext, RawContext);
    auto Ret = ProfileMap.emplace(FContext, FunctionSamples());
    if (WasLeafInlined)
      FContext.setAttribute(ContextWasInlined);
    FunctionSamples &FProfile = Ret.first->second;
    FProfile.setContext(FContext);
    return Ret.first->second;
  }
  return I->second;
}

void CSProfileGenerator::generateProfile() {
  FunctionSamples::ProfileIsCS = true;
  if (Binary->usePseudoProbes()) {
    // Enable pseudo probe functionalities in SampleProf
    FunctionSamples::ProfileIsProbeBased = true;
    generateProbeBasedProfile();
  } else {
    generateLineNumBasedProfile();
  }
  postProcessProfiles();
}

void CSProfileGenerator::generateLineNumBasedProfile() {
  for (const auto &CI : SampleCounters) {
    const StringBasedCtxKey *CtxKey =
        dyn_cast<StringBasedCtxKey>(CI.first.getPtr());
    // Get or create function profile for the range
    FunctionSamples &FunctionProfile =
        getFunctionProfileForContext(CtxKey->Context, CtxKey->WasLeafInlined);

    // Fill in function body samples
    populateBodySamplesForFunction(FunctionProfile, CI.second.RangeCounter);
    // Fill in boundary sample counts as well as call site samples for calls
    populateBoundarySamplesForFunction(CtxKey->Context, FunctionProfile,
                                       CI.second.BranchCounter);
  }
  // Fill in call site value sample for inlined calls and also use context to
  // infer missing samples. Since we don't have call count for inlined
  // functions, we estimate it from inlinee's profile using the entry of the
  // body sample.
  populateInferredFunctionSamples();
}

void CSProfileGenerator::populateBodySamplesForFunction(
    FunctionSamples &FunctionProfile, const RangeSample &RangeCounter) {
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

void CSProfileGenerator::populateBoundarySamplesForFunction(
    SampleContextFrames ContextId, FunctionSamples &FunctionProfile,
    const BranchSample &BranchCounters) {

  for (auto Entry : BranchCounters) {
    uint64_t SourceOffset = Entry.first.first;
    uint64_t TargetOffset = Entry.first.second;
    uint64_t Count = Entry.second;
    assert(Count != 0 && "Unexpected zero weight branch");

    // Get the callee name by branch target if it's a call branch
    StringRef CalleeName = FunctionSamples::getCanonicalFnName(
        Binary->getFuncFromStartOffset(TargetOffset));
    if (CalleeName.size() == 0)
      continue;

    // Record called target sample and its count
    auto LeafLoc = Binary->getInlineLeafFrameLoc(SourceOffset);
    if (!LeafLoc.hasValue())
      continue;
    FunctionProfile.addCalledTargetSamples(LeafLoc->Callsite.LineOffset,
                                           LeafLoc->Callsite.Discriminator,
                                           CalleeName, Count);

    // Record head sample for called target(callee)
    SampleContextFrameVector CalleeCtx(ContextId.begin(), ContextId.end());
    assert(CalleeCtx.back().CallerName == LeafLoc->CallerName &&
           "Leaf function name doesn't match");
    CalleeCtx.back() = *LeafLoc;
    CalleeCtx.emplace_back(CalleeName, LineLocation(0, 0));
    FunctionSamples &CalleeProfile = getFunctionProfileForContext(CalleeCtx);
    CalleeProfile.addHeadSamples(Count);
  }
}

static SampleContextFrame
getCallerContext(SampleContextFrames CalleeContext,
                 SampleContextFrameVector &CallerContext) {
  assert(CalleeContext.size() > 1 && "Unexpected empty context");
  CalleeContext = CalleeContext.drop_back();
  CallerContext.assign(CalleeContext.begin(), CalleeContext.end());
  SampleContextFrame CallerFrame = CallerContext.back();
  CallerContext.back().Callsite = LineLocation(0, 0);
  return CallerFrame;
}

void CSProfileGenerator::populateInferredFunctionSamples() {
  for (const auto &Item : ProfileMap) {
    const auto &CalleeContext = Item.first;
    const FunctionSamples &CalleeProfile = Item.second;

    // If we already have head sample counts, we must have value profile
    // for call sites added already. Skip to avoid double counting.
    if (CalleeProfile.getHeadSamples())
      continue;
    // If we don't have context, nothing to do for caller's call site.
    // This could happen for entry point function.
    if (CalleeContext.isBaseContext())
      continue;

    // Infer Caller's frame loc and context ID through string splitting
    SampleContextFrameVector CallerContextId;
    SampleContextFrame &&CallerLeafFrameLoc =
        getCallerContext(CalleeContext.getContextFrames(), CallerContextId);
    SampleContextFrames CallerContext(CallerContextId);

    // It's possible that we haven't seen any sample directly in the caller,
    // in which case CallerProfile will not exist. But we can't modify
    // ProfileMap while iterating it.
    // TODO: created function profile for those callers too
    if (ProfileMap.find(CallerContext) == ProfileMap.end())
      continue;
    FunctionSamples &CallerProfile = ProfileMap[CallerContext];

    // Since we don't have call count for inlined functions, we
    // estimate it from inlinee's profile using entry body sample.
    uint64_t EstimatedCallCount = CalleeProfile.getEntrySamples();
    // If we don't have samples with location, use 1 to indicate live.
    if (!EstimatedCallCount && !CalleeProfile.getBodySamples().size())
      EstimatedCallCount = 1;
    CallerProfile.addCalledTargetSamples(
        CallerLeafFrameLoc.Callsite.LineOffset,
        CallerLeafFrameLoc.Callsite.Discriminator,
        CalleeProfile.getContext().getName(), EstimatedCallCount);
    CallerProfile.addBodySamples(CallerLeafFrameLoc.Callsite.LineOffset,
                                 CallerLeafFrameLoc.Callsite.Discriminator,
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
  if (EnableCSPreInliner) {
    CSPreInliner(ProfileMap, *Binary, HotCountThreshold, ColdCountThreshold)
        .run();
  }

  // Trim and merge cold context profile using cold threshold above. By default,
  // we skip such merging and trimming when preinliner is on.
  if (!EnableCSPreInliner || CSProfTrimColdContext.getNumOccurrences() ||
      CSProfMergeColdContext.getNumOccurrences()) {
    SampleContextTrimmer(ProfileMap)
        .trimAndMergeColdContextProfiles(
            HotCountThreshold, CSProfTrimColdContext, CSProfMergeColdContext,
            CSProfMaxColdContextDepth);
  }
}

void CSProfileGenerator::computeSummaryAndThreshold() {
  SampleProfileSummaryBuilder Builder(ProfileSummaryBuilder::DefaultCutoffs);
  auto Summary = Builder.computeSummaryForProfiles(ProfileMap);
  HotCountThreshold = ProfileSummaryBuilder::getHotCountThreshold(
      (Summary->getDetailedSummary()));
  ColdCountThreshold = ProfileSummaryBuilder::getColdCountThreshold(
      (Summary->getDetailedSummary()));
}

// Helper function to extract context prefix string stack
// Extract context stack for reusing, leaf context stack will
// be added compressed while looking up function profile
static void extractPrefixContextStack(
    SampleContextFrameVector &ContextStack,
    const SmallVectorImpl<const MCDecodedPseudoProbe *> &Probes,
    ProfiledBinary *Binary) {
  for (const auto *P : Probes) {
    Binary->getInlineContextForProbe(P, ContextStack, true);
  }
}

void CSProfileGenerator::generateProbeBasedProfile() {
  for (const auto &CI : SampleCounters) {
    const ProbeBasedCtxKey *CtxKey =
        dyn_cast<ProbeBasedCtxKey>(CI.first.getPtr());
    SampleContextFrameVector ContextStack;
    extractPrefixContextStack(ContextStack, CtxKey->Probes, Binary);
    // Fill in function body samples from probes, also infer caller's samples
    // from callee's probe
    populateBodySamplesWithProbes(CI.second.RangeCounter, ContextStack);
    // Fill in boundary samples for a call probe
    populateBoundarySamplesWithProbes(CI.second.BranchCounter, ContextStack);
  }
}

void CSProfileGenerator::extractProbesFromRange(const RangeSample &RangeCounter,
                                                ProbeCounterMap &ProbeCounter) {
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

void CSProfileGenerator::populateBodySamplesWithProbes(
    const RangeSample &RangeCounter, SampleContextFrames ContextStack) {
  ProbeCounterMap ProbeCounter;
  // Extract the top frame probes by looking up each address among the range in
  // the Address2ProbeMap
  extractProbesFromRange(RangeCounter, ProbeCounter);
  std::unordered_map<MCDecodedPseudoProbeInlineTree *,
                     std::unordered_set<FunctionSamples *>>
      FrameSamples;
  for (auto PI : ProbeCounter) {
    const MCDecodedPseudoProbe *Probe = PI.first;
    uint64_t Count = PI.second;
    FunctionSamples &FunctionProfile =
        getFunctionProfileForLeafProbe(ContextStack, Probe);
    // Record the current frame and FunctionProfile whenever samples are
    // collected for non-danglie probes. This is for reporting all of the
    // zero count probes of the frame later.
    FrameSamples[Probe->getInlineTreeNode()].insert(&FunctionProfile);
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
        SampleContextFrames CalleeContextId =
            FunctionProfile.getContext().getContextFrames();
        SampleContextFrameVector CallerContextId;
        SampleContextFrame &&CallerLeafFrameLoc =
            getCallerContext(CalleeContextId, CallerContextId);
        uint64_t CallerIndex = CallerLeafFrameLoc.Callsite.LineOffset;
        assert(CallerIndex &&
               "Inferred caller's location index shouldn't be zero!");
        FunctionSamples &CallerProfile =
            getFunctionProfileForContext(CallerContextId);
        CallerProfile.setFunctionHash(InlinerDesc->FuncHash);
        CallerProfile.addBodySamples(CallerIndex, 0, Count);
        CallerProfile.addTotalSamples(Count);
        CallerProfile.addCalledTargetSamples(
            CallerIndex, 0, FunctionProfile.getContext().getName(), Count);
      }
    }
  }

  // Assign zero count for remaining probes without sample hits to
  // differentiate from probes optimized away, of which the counts are unknown
  // and will be inferred by the compiler.
  for (auto &I : FrameSamples) {
    for (auto *FunctionProfile : I.second) {
      for (auto *Probe : I.first->getProbes()) {
        FunctionProfile->addBodySamplesForProbe(Probe->getIndex(), 0);
      }
    }
  }
}

void CSProfileGenerator::populateBoundarySamplesWithProbes(
    const BranchSample &BranchCounter, SampleContextFrames ContextStack) {
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
        getFunctionProfileForLeafProbe(ContextStack, CallProbe);
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

FunctionSamples &CSProfileGenerator::getFunctionProfileForLeafProbe(
    SampleContextFrames ContextStack, const MCDecodedPseudoProbe *LeafProbe) {

  // Explicitly copy the context for appending the leaf context
  SampleContextFrameVector NewContextStack(ContextStack.begin(),
                                           ContextStack.end());
  Binary->getInlineContextForProbe(LeafProbe, NewContextStack, true);
  // For leaf inlined context with the top frame, we should strip off the top
  // frame's probe id, like:
  // Inlined stack: [foo:1, bar:2], the ContextId will be "foo:1 @ bar"
  auto LeafFrame = NewContextStack.back();
  LeafFrame.Callsite = LineLocation(0, 0);
  NewContextStack.pop_back();
  // Compress the context string except for the leaf frame
  CSProfileGenerator::compressRecursionContext(NewContextStack);
  CSProfileGenerator::trimContext(NewContextStack);
  NewContextStack.push_back(LeafFrame);

  const auto *FuncDesc = Binary->getFuncDescForGUID(LeafProbe->getGuid());
  bool WasLeafInlined = LeafProbe->getInlineTreeNode()->hasInlineSite();
  FunctionSamples &FunctionProile =
      getFunctionProfileForContext(NewContextStack, WasLeafInlined);
  FunctionProile.setFunctionHash(FuncDesc->FuncHash);
  return FunctionProile;
}

} // end namespace sampleprof
} // end namespace llvm
