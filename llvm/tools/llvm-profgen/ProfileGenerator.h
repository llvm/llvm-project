//===-- ProfileGenerator.h - Profile Generator -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROGEN_PROFILEGENERATOR_H
#define LLVM_TOOLS_LLVM_PROGEN_PROFILEGENERATOR_H
#include "ErrorHandling.h"
#include "PerfReader.h"
#include "ProfiledBinary.h"
#include "llvm/ProfileData/SampleProfWriter.h"

using namespace llvm;
using namespace sampleprof;

namespace llvm {
namespace sampleprof {

class ProfileGenerator {

public:
  ProfileGenerator(){};
  virtual ~ProfileGenerator() = default;
  static std::unique_ptr<ProfileGenerator>
  create(const BinarySampleCounterMap &BinarySampleCounters,
         enum PerfScriptType SampleType);
  virtual void generateProfile() = 0;
  // Use SampleProfileWriter to serialize profile map
  virtual void write(std::unique_ptr<SampleProfileWriter> Writer,
                     StringMap<FunctionSamples> &ProfileMap);
  void write();

protected:
  /*
  For each region boundary point, mark if it is begin or end (or both) of
  the region. Boundary points are inclusive. Log the sample count as well
  so we can use it when we compute the sample count of each disjoint region
  later. Note that there might be multiple ranges with different sample
  count that share same begin/end point. We need to accumulate the sample
  count for the boundary point for such case, because for the example
  below,

  |<--100-->|
  |<------200------>|
  A         B       C

  sample count for disjoint region [A,B] would be 300.
  */
  void findDisjointRanges(RangeSample &DisjointRanges,
                          const RangeSample &Ranges);

  // Used by SampleProfileWriter
  StringMap<FunctionSamples> ProfileMap;
};

class CSProfileGenerator : public ProfileGenerator {
protected:
  const BinarySampleCounterMap &BinarySampleCounters;

public:
  CSProfileGenerator(const BinarySampleCounterMap &Counters)
      : BinarySampleCounters(Counters){};

public:
  void generateProfile() override {
    for (const auto &BI : BinarySampleCounters) {
      ProfiledBinary *Binary = BI.first;
      for (const auto &CI : BI.second) {
        const StringBasedCtxKey *CtxKey =
            dyn_cast<StringBasedCtxKey>(CI.first.getPtr());
        StringRef ContextId(CtxKey->Context);
        // Get or create function profile for the range
        FunctionSamples &FunctionProfile =
            getFunctionProfileForContext(ContextId);

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
  }

  // Remove adjacent repeated context sequences up to a given sequence length,
  // -1 means no size limit. Note that repeated sequences are identified based
  // on the exact call site, this is finer granularity than function recursion.
  template <typename T>
  static void compressRecursionContext(SmallVectorImpl<T> &Context,
                                       int32_t CSize = MaxCompressionSize) {
    uint32_t I = 1;
    uint32_t HS = static_cast<uint32_t>(Context.size() / 2);
    uint32_t MaxDedupSize =
        CSize == -1 ? HS : std::min(static_cast<uint32_t>(CSize), HS);
    auto BeginIter = Context.begin();
    // Use an in-place algorithm to save memory copy
    // End indicates the end location of current iteration's data
    uint32_t End = 0;
    // Deduplicate from length 1 to the max possible size of a repeated
    // sequence.
    while (I <= MaxDedupSize) {
      // This is a linear algorithm that deduplicates adjacent repeated
      // sequences of size I. The deduplication detection runs on a sliding
      // window whose size is 2*I and it keeps sliding the window to deduplicate
      // the data inside. Once duplication is detected, deduplicate it by
      // skipping the right half part of the window, otherwise just copy back
      // the new one by appending them at the back of End pointer(for the next
      // iteration).
      //
      // For example:
      // Input: [a1, a2, b1, b2]
      // (Added index to distinguish the same char, the origin is [a, a, b,
      // b], the size of the dedup window is 2(I = 1) at the beginning)
      //
      // 1) The initial status is a dummy window[null, a1], then just copy the
      // right half of the window(End = 0), then slide the window.
      // Result: [a1], a2, b1, b2 (End points to the element right before ],
      // after ] is the data of the previous iteration)
      //
      // 2) Next window is [a1, a2]. Since a1 == a2, then skip the right half of
      // the window i.e the duplication happen. Only slide the window.
      // Result: [a1], a2, b1, b2
      //
      // 3) Next window is [a2, b1], copy the right half of the window(b1 is
      // new) to the End and slide the window.
      // Result: [a1, b1], b1, b2
      //
      // 4) Next window is [b1, b2], same to 2), skip b2.
      // Result: [a1, b1], b1, b2
      // After resize, it will be [a, b]

      // Use pointers like below to do comparison inside the window
      //    [a         b         c        a       b        c]
      //     |         |         |                |        |
      // LeftBoundary Left     Right           Left+I    Right+I
      // A duplication found if Left < LeftBoundry.

      int32_t Right = I - 1;
      End = I;
      int32_t LeftBoundary = 0;
      while (Right + I < Context.size()) {
        // To avoids scanning a part of a sequence repeatedly, it finds out
        // the common suffix of two hald in the window. The common suffix will
        // serve as the common prefix of next possible pair of duplicate
        // sequences. The non-common part will be ignored and never scanned
        // again.

        // For example.
        // Input: [a, b1], c1, b2, c2
        // I = 2
        //
        // 1) For the window [a, b1, c1, b2], non-common-suffix for the right
        // part is 'c1', copy it and only slide the window 1 step.
        // Result: [a, b1, c1], b2, c2
        //
        // 2) Next window is [b1, c1, b2, c2], so duplication happen.
        // Result after resize: [a, b, c]

        int32_t Left = Right;
        while (Left >= LeftBoundary && Context[Left] == Context[Left + I]) {
          // Find the longest suffix inside the window. When stops, Left points
          // at the diverging point in the current sequence.
          Left--;
        }

        bool DuplicationFound = (Left < LeftBoundary);
        // Don't need to recheck the data before Right
        LeftBoundary = Right + 1;
        if (DuplicationFound) {
          // Duplication found, skip right half of the window.
          Right += I;
        } else {
          // Copy the non-common-suffix part of the adjacent sequence.
          std::copy(BeginIter + Right + 1, BeginIter + Left + I + 1,
                    BeginIter + End);
          End += Left + I - Right;
          // Only slide the window by the size of non-common-suffix
          Right = Left + I;
        }
      }
      // Don't forget the remaining part that's not scanned.
      std::copy(BeginIter + Right + 1, Context.end(), BeginIter + End);
      End += Context.size() - Right - 1;
      I++;
      Context.resize(End);
      MaxDedupSize = std::min(static_cast<uint32_t>(End / 2), MaxDedupSize);
    }
  }

protected:
  // Lookup or create FunctionSamples for the context
  FunctionSamples &getFunctionProfileForContext(StringRef ContextId);
  // Merge cold context profile whose total sample is below threshold
  // into base profile.
  void mergeAndTrimColdProfile(StringMap<FunctionSamples> &ProfileMap);
  void write(std::unique_ptr<SampleProfileWriter> Writer,
             StringMap<FunctionSamples> &ProfileMap) override;

private:
  // Helper function for updating body sample for a leaf location in
  // FunctionProfile
  void updateBodySamplesforFunctionProfile(FunctionSamples &FunctionProfile,
                                           const FrameLocation &LeafLoc,
                                           uint64_t Count);
  void populateFunctionBodySamples(FunctionSamples &FunctionProfile,
                                   const RangeSample &RangeCounters,
                                   ProfiledBinary *Binary);
  void populateFunctionBoundarySamples(StringRef ContextId,
                                       FunctionSamples &FunctionProfile,
                                       const BranchSample &BranchCounters,
                                       ProfiledBinary *Binary);
  void populateInferredFunctionSamples();

public:
  // Deduplicate adjacent repeated context sequences up to a given sequence
  // length. -1 means no size limit.
  static int32_t MaxCompressionSize;
};

using ProbeCounterMap = std::unordered_map<const PseudoProbe *, uint64_t>;

class PseudoProbeCSProfileGenerator : public CSProfileGenerator {

public:
  PseudoProbeCSProfileGenerator(const BinarySampleCounterMap &Counters)
      : CSProfileGenerator(Counters) {}
  void generateProfile() override;

private:
  // Go through each address from range to extract the top frame probe by
  // looking up in the Address2ProbeMap
  void extractProbesFromRange(const RangeSample &RangeCounter,
                              ProbeCounterMap &ProbeCounter,
                              ProfiledBinary *Binary);
  // Fill in function body samples from probes
  void
  populateBodySamplesWithProbes(const RangeSample &RangeCounter,
                                SmallVectorImpl<std::string> &ContextStrStack,
                                ProfiledBinary *Binary);
  // Fill in boundary samples for a call probe
  void populateBoundarySamplesWithProbes(
      const BranchSample &BranchCounter,
      SmallVectorImpl<std::string> &ContextStrStack, ProfiledBinary *Binary);
  // Helper function to get FunctionSamples for the leaf inlined context
  FunctionSamples &
  getFunctionProfileForLeafProbe(SmallVectorImpl<std::string> &ContextStrStack,
                                 const PseudoProbeFuncDesc *LeafFuncDesc);
  // Helper function to get FunctionSamples for the leaf probe
  FunctionSamples &
  getFunctionProfileForLeafProbe(SmallVectorImpl<std::string> &ContextStrStack,
                                 const PseudoProbe *LeafProbe,
                                 ProfiledBinary *Binary);
};

} // end namespace sampleprof
} // end namespace llvm

#endif
