//===- SimplifyCFGOptions.h - Control structure for SimplifyCFG -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A set of parameters used to control the transforms in the SimplifyCFG pass.
// Options may change depending on the position in the optimization pipeline.
// For example, canonical form that includes switches and branches may later be
// replaced by lookup tables and selects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SIMPLIFYCFGOPTIONS_H
#define LLVM_TRANSFORMS_SCALAR_SIMPLIFYCFGOPTIONS_H

namespace llvm {

class AssumptionCache;

struct SimplifyCFGOptions {
  int BonusInstThreshold;
  bool ForwardSwitchCondToPhi;
  bool ConvertSwitchToLookupTable;
  bool NeedCanonicalLoop;
  bool SinkCommonInsts;
  bool SimplifyCondBranch;
  bool FoldTwoEntryPHINode;

  AssumptionCache *AC;

  SimplifyCFGOptions(unsigned BonusThreshold = 1,
                     bool ForwardSwitchCond = false,
                     bool SwitchToLookup = false, bool CanonicalLoops = true,
                     bool SinkCommon = false,
                     AssumptionCache *AssumpCache = nullptr,
                     bool SimplifyCondBranch = true,
                     bool FoldTwoEntryPHINode = true)
      : BonusInstThreshold(BonusThreshold),
        ForwardSwitchCondToPhi(ForwardSwitchCond),
        ConvertSwitchToLookupTable(SwitchToLookup),
        NeedCanonicalLoop(CanonicalLoops), SinkCommonInsts(SinkCommon),
        SimplifyCondBranch(SimplifyCondBranch),
        FoldTwoEntryPHINode(FoldTwoEntryPHINode), AC(AssumpCache) {}

  // Support 'builder' pattern to set members by name at construction time.
  SimplifyCFGOptions &bonusInstThreshold(int I) {
    BonusInstThreshold = I;
    return *this;
  }
  SimplifyCFGOptions &forwardSwitchCondToPhi(bool B) {
    ForwardSwitchCondToPhi = B;
    return *this;
  }
  SimplifyCFGOptions &convertSwitchToLookupTable(bool B) {
    ConvertSwitchToLookupTable = B;
    return *this;
  }
  SimplifyCFGOptions &needCanonicalLoops(bool B) {
    NeedCanonicalLoop = B;
    return *this;
  }
  SimplifyCFGOptions &sinkCommonInsts(bool B) {
    SinkCommonInsts = B;
    return *this;
  }
  SimplifyCFGOptions &setAssumptionCache(AssumptionCache *Cache) {
    AC = Cache;
    return *this;
  }
  SimplifyCFGOptions &setSimplifyCondBranch(bool B) {
    SimplifyCondBranch = B;
    return *this;
  }

  SimplifyCFGOptions &setFoldTwoEntryPHINode(bool B) {
    FoldTwoEntryPHINode = B;
    return *this;
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SIMPLIFYCFGOPTIONS_H
