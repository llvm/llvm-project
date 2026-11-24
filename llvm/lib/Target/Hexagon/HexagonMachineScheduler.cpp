//===- HexagonMachineScheduler.cpp - MI Scheduler for Hexagon -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MachineScheduler schedules machine instructions after phi elimination. It
// preserves LiveIntervals so it can be invoked before register allocation.
//
//===----------------------------------------------------------------------===//

#include "HexagonMachineScheduler.h"
#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/VLIWMachineScheduler.h"

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

/// Return true if there is a dependence between SUd and SUu.
bool HexagonVLIWResourceModel::hasDependence(const SUnit *SUd,
                                             const SUnit *SUu) {
  const auto *QII = static_cast<const HexagonInstrInfo *>(TII);

  // Enable .cur formation.
  if (QII->mayBeCurLoad(*SUd->getInstr()))
    return false;

  if (QII->canExecuteInBundle(*SUd->getInstr(), *SUu->getInstr()))
    return false;

  return VLIWResourceModel::hasDependence(SUd, SUu);
}

VLIWResourceModel *HexagonConvergingVLIWScheduler::createVLIWResourceModel(
    const TargetSubtargetInfo &STI, const TargetSchedModel *SchedModel) const {
  return new HexagonVLIWResourceModel(STI, SchedModel);
}

int HexagonConvergingVLIWScheduler::SchedulingCost(ReadyQueue &Q, SUnit *SU,
                                                   SchedCandidate &Candidate,
                                                   RegPressureDelta &Delta,
                                                   bool verbose) {
  int ResCount =
      ConvergingVLIWScheduler::SchedulingCost(Q, SU, Candidate, Delta, verbose);

  if (!SU || SU->isScheduled || !SU->isInstr())
    return ResCount;

  auto &QST = DAG->MF.getSubtarget<HexagonSubtarget>();
  auto &QII = *QST.getInstrInfo();
  MachineInstr *MI = SU->getInstr();
  if (QII.mayBeCurLoad(*MI)) {
    if (Q.getID() == TopQID &&
        Top.ResourceModel->isResourceAvailable(SU, true)) {
      ResCount += PriorityTwo;
      LLVM_DEBUG(if (verbose) dbgs() << "C|");
    } else if (Q.getID() == BotQID &&
               Bot.ResourceModel->isResourceAvailable(SU, false)) {
      ResCount += PriorityTwo;
      LLVM_DEBUG(if (verbose) dbgs() << "C|");
    }
    return ResCount;
  }

  // Keep HLX load uses and store definitions close to their pseudos so the
  // packetizer can place each pair in the same packet.
  if (Q.getID() == TopQID) {
    if (QII.isNewStorePseudo(*MI)) {
      ResCount += PriorityOne;
    } else if (!QII.isTmpLoadPseudo(*MI)) {
      for (const SDep &PredDep : SU->Preds) {
        SUnit *Pred = PredDep.getSUnit();
        if (PredDep.getKind() == SDep::Data && Pred->isInstr() &&
            QII.isTmpLoadPseudo(*Pred->getInstr()) && Pred->NumSuccs < 3)
          ResCount += PriorityOne;
      }
    }
  } else if (Q.getID() == BotQID) {
    if (QII.isTmpLoadPseudo(*MI) && SU->NumSuccs < 3) {
      ResCount += PriorityOne;
    } else if (!QII.isNewStorePseudo(*MI)) {
      for (const SDep &SuccDep : SU->Succs) {
        SUnit *Succ = SuccDep.getSUnit();
        if (SuccDep.getKind() == SDep::Data && Succ->isInstr() &&
            QII.isNewStorePseudo(*Succ->getInstr()))
          ResCount += PriorityOne;
      }
    }
  }

  return ResCount;
}
