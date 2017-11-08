//===- MacroFusion.cpp - Macro Fusion -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the implementation of the DAG scheduling mutation
/// to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "machine-scheduler"

STATISTIC(NumFused, "Number of instr pairs fused");

using namespace llvm;

static cl::opt<bool> EnableMacroFusion("misched-fusion", cl::Hidden,
  cl::desc("Enable scheduling for macro fusion."), cl::init(true));

static void fuseInstructionPair(ScheduleDAGMI &DAG, SUnit &FirstSU,
                                SUnit &SecondSU) {
  // Create a single weak edge between the adjacent instrs. The only effect is
  // to cause bottom-up scheduling to heavily prioritize the clustered instrs.
  DAG.addEdge(&SecondSU, SDep(&FirstSU, SDep::Cluster));

  // Adjust the latency between the anchor instr and its
  // predecessors.
  for (SDep &IDep : SecondSU.Preds)
    if (IDep.getSUnit() == &FirstSU)
      IDep.setLatency(0);

  // Adjust the latency between the dependent instr and its
  // predecessors.
  for (SDep &IDep : FirstSU.Succs)
    if (IDep.getSUnit() == &SecondSU)
      IDep.setLatency(0);

  DEBUG(dbgs() << DAG.MF.getName() << "(): Macro fuse ";
        FirstSU.print(dbgs(), &DAG); dbgs() << " - ";
        SecondSU.print(dbgs(), &DAG); dbgs() << " /  ";
        dbgs() << DAG.TII->getName(FirstSU.getInstr()->getOpcode()) << " - " <<
                  DAG.TII->getName(SecondSU.getInstr()->getOpcode()) << '\n'; );

  if (&SecondSU != &DAG.ExitSU)
    // Make instructions dependent on FirstSU also dependent on SecondSU to
    // prevent them from being scheduled between FirstSU and and SecondSU.
    for (const SDep &SI : FirstSU.Succs) {
      if (SI.getSUnit() == &SecondSU)
        continue;
      DEBUG(dbgs() << "  Copy Succ ";
            SI.getSUnit()->print(dbgs(), &DAG); dbgs() << '\n';);
      DAG.addEdge(SI.getSUnit(), SDep(&SecondSU, SDep::Artificial));
    }

  ++NumFused;
}

namespace {

/// \brief Post-process the DAG to create cluster edges between instrs that may
/// be fused by the processor into a single operation.
class MacroFusion : public ScheduleDAGMutation {
  ShouldSchedulePredTy shouldScheduleAdjacent;
  bool FuseBlock;
  bool scheduleAdjacentImpl(ScheduleDAGMI &DAG, SUnit &AnchorSU);

public:
  MacroFusion(ShouldSchedulePredTy shouldScheduleAdjacent, bool FuseBlock)
    : shouldScheduleAdjacent(shouldScheduleAdjacent), FuseBlock(FuseBlock) {}

  void apply(ScheduleDAGInstrs *DAGInstrs) override;
};

} // end anonymous namespace

void MacroFusion::apply(ScheduleDAGInstrs *DAGInstrs) {
  ScheduleDAGMI *DAG = static_cast<ScheduleDAGMI*>(DAGInstrs);

  if (FuseBlock)
    // For each of the SUnits in the scheduling block, try to fuse the instr in
    // it with one in its predecessors.
    for (SUnit &ISU : DAG->SUnits)
        scheduleAdjacentImpl(*DAG, ISU);

  if (DAG->ExitSU.getInstr())
    // Try to fuse the instr in the ExitSU with one in its predecessors.
    scheduleAdjacentImpl(*DAG, DAG->ExitSU);
}

/// \brief Implement the fusion of instr pairs in the scheduling DAG,
/// anchored at the instr in AnchorSU..
bool MacroFusion::scheduleAdjacentImpl(ScheduleDAGMI &DAG, SUnit &AnchorSU) {
  const MachineInstr &AnchorMI = *AnchorSU.getInstr();
  const TargetInstrInfo &TII = *DAG.TII;
  const TargetSubtargetInfo &ST = DAG.MF.getSubtarget();

  // Check if the anchor instr may be fused.
  if (!shouldScheduleAdjacent(TII, ST, nullptr, AnchorMI))
    return false;

  // Explorer for fusion candidates among the dependencies of the anchor instr.
  for (SDep &Dep : AnchorSU.Preds) {
    // Ignore dependencies that don't enforce ordering.
    if (Dep.getKind() == SDep::Anti || Dep.getKind() == SDep::Output ||
        Dep.isWeak())
      continue;

    SUnit &DepSU = *Dep.getSUnit();
    if (DepSU.isBoundaryNode())
      continue;

    const MachineInstr *DepMI = DepSU.getInstr();
    if (!shouldScheduleAdjacent(TII, ST, DepMI, AnchorMI))
      continue;

    fuseInstructionPair(DAG, DepSU, AnchorSU);
    return true;
  }

  return false;
}

std::unique_ptr<ScheduleDAGMutation>
llvm::createMacroFusionDAGMutation(
     ShouldSchedulePredTy shouldScheduleAdjacent) {
  if(EnableMacroFusion)
    return llvm::make_unique<MacroFusion>(shouldScheduleAdjacent, true);
  return nullptr;
}

std::unique_ptr<ScheduleDAGMutation>
llvm::createBranchMacroFusionDAGMutation(
     ShouldSchedulePredTy shouldScheduleAdjacent) {
  if(EnableMacroFusion)
    return llvm::make_unique<MacroFusion>(shouldScheduleAdjacent, false);
  return nullptr;
}
