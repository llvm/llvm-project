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
  create(const BinarySampleCounterMap &SampleCounters,
         enum PerfScriptType SampleType);
  virtual void generateProfile() = 0;

  // Use SampleProfileWriter to serialize profile map
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
  const BinarySampleCounterMap &BinarySampleCounters;

public:
  CSProfileGenerator(const BinarySampleCounterMap &Counters)
      : BinarySampleCounters(Counters){};

public:
  void generateProfile() override {
    // Fill in function body samples
    populateFunctionBodySamples();

    // Fill in boundary sample counts as well as call site samples for calls
    populateFunctionBoundarySamples();

    // Fill in call site value sample for inlined calls and also use context to
    // infer missing samples. Since we don't have call count for inlined
    // functions, we estimate it from inlinee's profile using the entry of the
    // body sample.
    populateInferredFunctionSamples();
  }

private:
  // Helper function for updating body sample for a leaf location in
  // FunctionProfile
  void updateBodySamplesforFunctionProfile(FunctionSamples &FunctionProfile,
                                           const FrameLocation &LeafLoc,
                                           uint64_t Count);
  // Lookup or create FunctionSamples for the context
  FunctionSamples &getFunctionProfileForContext(StringRef ContextId);
  void populateFunctionBodySamples();
  void populateFunctionBoundarySamples();
  void populateInferredFunctionSamples();
};

} // end namespace sampleprof
} // end namespace llvm

#endif
