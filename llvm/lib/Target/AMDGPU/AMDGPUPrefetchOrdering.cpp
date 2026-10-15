//===--- AMDGPUPrefetchOrdering.cpp - Order prefetches before loads -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// A DAG mutation that keeps each VMEM and SMEM prefetch ahead of the loads
/// that follow it in program order. Prefetches are otherwise not ordered
/// against memory accesses, so earlier accesses and later stores can move
/// across them. Prefetches are chained in program order and each load is
/// ordered after the closest preceding prefetch, so every prefetch precedes
/// every later load.
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
  for (SUnit &SU : DAG->SUnits) {
    const MachineInstr &MI = *SU.getInstr();
    if (!Pending.empty() && anyInBundle(MI, isNonPrefetchLoad))
      for (SUnit *Pf : Pending)
        DAG->addEdge(&SU, SDep(Pf, SDep::Artificial));

    if (!anyInBundle(MI, SIInstrInfo::isPrefetch))
      continue;

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
