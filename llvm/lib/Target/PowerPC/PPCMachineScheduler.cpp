//===- PPCMachineScheduler.cpp - MI Scheduler for PowerPC -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCMachineScheduler.h"
#include "MCTargetDesc/PPCMCTargetDesc.h"

using namespace llvm;

static cl::opt<bool> 
DisableAddiLoadHeuristic("disable-ppc-sched-addi-load",
                         cl::desc("Disable scheduling addi instruction before" 
                                  "load for ppc"), cl::Hidden);
static cl::opt<bool>
    EnableAddiHeuristic("ppc-postra-bias-addi",
                        cl::desc("Enable scheduling addi instruction as early"
                                 "as possible post ra"),
                        cl::Hidden, cl::init(true));

static bool isADDIInstr(const GenericScheduler::SchedCandidate &Cand) {
  return Cand.SU->getInstr()->getOpcode() == PPC::ADDI ||
         Cand.SU->getInstr()->getOpcode() == PPC::ADDI8;
}

bool PPCPreRASchedStrategy::biasAddiLoadCandidate(SchedCandidate &Cand,
                                                  SchedCandidate &TryCand,
                                                  SchedBoundary &Zone) const {
  if (DisableAddiLoadHeuristic)
    return false;

  SchedCandidate &FirstCand = Zone.isTop() ? TryCand : Cand;
  SchedCandidate &SecondCand = Zone.isTop() ? Cand : TryCand;
  if (isADDIInstr(FirstCand) && SecondCand.SU->getInstr()->mayLoad()) {
    TryCand.Reason = Stall;
    return true;
  }
  if (FirstCand.SU->getInstr()->mayLoad() && isADDIInstr(SecondCand)) {
    TryCand.Reason = NoCand;
    return true;
  }

  return false;
}

void PPCPreRASchedStrategy::tryCandidate(SchedCandidate &Cand,
                                         SchedCandidate &TryCand,
                                         SchedBoundary *Zone) const {
  GenericScheduler::tryCandidate(Cand, TryCand, Zone);

  if (!Cand.isValid() || !Zone)
    return;

  // Add powerpc specific heuristic only when TryCand isn't selected or
  // selected as node order.
  if (TryCand.Reason != NodeOrder && TryCand.Reason != NoCand)
    return;

  // There are some benefits to schedule the ADDI before the load to hide the
  // latency, as RA may create a true dependency between the load and addi.
  if (biasAddiLoadCandidate(Cand, TryCand, *Zone))
    return;
}

bool PPCPostRASchedStrategy::biasAddiCandidate(SchedCandidate &Cand,
                                               SchedCandidate &TryCand) const {
  if (!EnableAddiHeuristic)
    return false;

  if (isADDIInstr(TryCand) && !isADDIInstr(Cand)) {
    TryCand.Reason = Stall;
    return true;
  }
  return false;
}

void PPCPostRASchedStrategy::tryCandidate(SchedCandidate &Cand,
                                          SchedCandidate &TryCand) {
  PostGenericScheduler::tryCandidate(Cand, TryCand);

  if (!Cand.isValid())
    return;

  // Add powerpc post ra specific heuristic only when TryCand isn't selected or
  // selected as node order.
  if (TryCand.Reason != NodeOrder && TryCand.Reason != NoCand)
    return;

  // There are some benefits to schedule the ADDI as early as possible post ra
  // to avoid stalled by vector instructions which take up all the hw units.
  // And ADDI is usually used to post inc the loop indvar, which matters the
  // performance.
  if (biasAddiCandidate(Cand, TryCand))
    return;
}

void PPCPostRASchedStrategy::enterMBB(MachineBasicBlock *MBB) {
  // Custom PPC PostRA specific behavior here.
  PostGenericScheduler::enterMBB(MBB);
}

void PPCPostRASchedStrategy::leaveMBB() {
  // Custom PPC PostRA specific behavior here.
  PostGenericScheduler::leaveMBB();
}

void PPCPostRASchedStrategy::initialize(ScheduleDAGMI *Dag) {
  // Custom PPC PostRA specific initialization here.
  PostGenericScheduler::initialize(Dag);
}

SUnit *PPCPostRASchedStrategy::pickNode(bool &IsTopNode) {
  // Custom PPC PostRA specific scheduling here.
  return PostGenericScheduler::pickNode(IsTopNode);
}

