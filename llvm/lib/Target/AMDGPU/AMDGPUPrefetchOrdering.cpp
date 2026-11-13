//===--- AMDGPUPrefetchOrdering.cpp - Order prefetches and loads ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// A DAG mutation that keeps VMEM and SMEM prefetches in program order with
/// respect to loads: each prefetch stays after the loads before it and ahead
/// of the loads after it. Prefetches are otherwise not ordered against memory
/// accesses, so stores can move across them. Prefetches are chained in program
/// order, each load is ordered after the closest preceding prefetch, and each
/// prefetch after the loads since the previous one.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUPrefetchOrdering.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

using namespace llvm;

namespace {

class PrefetchOrdering : public ScheduleDAGMutation {
public:
  void apply(ScheduleDAGInstrs *DAG) override;
};

// Returns true if MI, or any instruction in the bundle MI heads, satisfies
// Pred.
template <typename PredT> bool anyInBundle(const MachineInstr &MI, PredT Pred) {
  if (!MI.isBundle())
    return Pred(MI);
  for (auto I = std::next(MI.getIterator()), E = getBundleEnd(MI.getIterator());
       I != E; ++I)
    if (Pred(*I))
      return true;
  return false;
}

bool isNonPrefetchLoad(const MachineInstr &MI) {
  return MI.mayLoad() && !SIInstrInfo::isPrefetch(MI);
}

void PrefetchOrdering::apply(ScheduleDAGInstrs *DAG) {
  // Prefetches that every later load has to follow. Normally this is only the
  // last prefetch, which itself follows the earlier ones.
  SmallVector<SUnit *, 4> Pending;
  // Loads since the last prefetch, which the next prefetch has to follow.
  SmallVector<SUnit *, 8> Loads;
  for (SUnit &SU : DAG->SUnits) {
    const MachineInstr &MI = *SU.getInstr();
    bool IsPrefetch = anyInBundle(MI, SIInstrInfo::isPrefetch);
    if (anyInBundle(MI, isNonPrefetchLoad)) {
      for (SUnit *Pf : Pending)
        DAG->addEdge(&SU, SDep(Pf, SDep::Artificial));
      // A bundle that also holds a prefetch must not be ordered after itself.
      if (!IsPrefetch)
        Loads.push_back(&SU);
    }

    if (!IsPrefetch)
      continue;

    for (SUnit *L : Loads)
      DAG->addEdge(&SU, SDep(L, SDep::Artificial));
    Loads.clear();

    // A pending prefetch that cannot be ordered before SU without creating a
    // cycle stays pending, so that it is still ordered before later loads.
    erase_if(Pending, [&](SUnit *Pf) {
      return DAG->addEdge(&SU, SDep(Pf, SDep::Artificial));
    });
    Pending.push_back(&SU);
  }
}

} // end namespace

std::unique_ptr<ScheduleDAGMutation>
llvm::createAMDGPUPrefetchOrderingDAGMutation() {
  return std::make_unique<PrefetchOrdering>();
}
