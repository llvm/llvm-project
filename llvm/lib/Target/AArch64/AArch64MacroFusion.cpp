//===- AArch64MacroFusion.cpp - AArch64 Macro Fusion ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file This file contains the AArch64 implementation of the DAG scheduling mutation
// to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "AArch64MacroFusion.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetInstrInfo.h"

#define DEBUG_TYPE "misched"

STATISTIC(NumFused, "Number of instr pairs fused");

using namespace llvm;

static cl::opt<bool> EnableMacroFusion("aarch64-misched-fusion", cl::Hidden,
  cl::desc("Enable scheduling for macro fusion."), cl::init(true));

namespace {

/// \brief Verify that the instr pair, FirstMI and SecondMI, should be fused
/// together.  Given an anchor instr, when the other instr is unspecified, then
/// check if the anchor instr may be part of a fused pair at all.
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr *SecondMI) {
  assert((FirstMI || SecondMI) && "At least one instr must be specified");

  const AArch64InstrInfo &II = static_cast<const AArch64InstrInfo&>(TII);
  const AArch64Subtarget &ST = static_cast<const AArch64Subtarget&>(TSI);

  // Assume wildcards for unspecified instrs.
  unsigned FirstOpcode =
    FirstMI ? FirstMI->getOpcode()
	    : static_cast<unsigned>(AArch64::INSTRUCTION_LIST_END);
  unsigned SecondOpcode =
    SecondMI ? SecondMI->getOpcode()
             : static_cast<unsigned>(AArch64::INSTRUCTION_LIST_END);

  if (ST.hasArithmeticBccFusion())
    // Fuse CMN, CMP, TST followed by Bcc.
    if (SecondOpcode == AArch64::Bcc)
      switch (FirstOpcode) {
      default:
        return false;
      case AArch64::ADDSWri:
      case AArch64::ADDSWrr:
      case AArch64::ADDSXri:
      case AArch64::ADDSXrr:
      case AArch64::ANDSWri:
      case AArch64::ANDSWrr:
      case AArch64::ANDSXri:
      case AArch64::ANDSXrr:
      case AArch64::SUBSWri:
      case AArch64::SUBSWrr:
      case AArch64::SUBSXri:
      case AArch64::SUBSXrr:
      case AArch64::BICSWrr:
      case AArch64::BICSXrr:
        return true;
      case AArch64::ADDSWrs:
      case AArch64::ADDSXrs:
      case AArch64::ANDSWrs:
      case AArch64::ANDSXrs:
      case AArch64::SUBSWrs:
      case AArch64::SUBSXrs:
      case AArch64::BICSWrs:
      case AArch64::BICSXrs:
        // Shift value can be 0 making these behave like the "rr" variant...
        return !II.hasShiftedReg(*FirstMI);
      case AArch64::INSTRUCTION_LIST_END:
        return true;
      }

  if (ST.hasArithmeticCbzFusion())
    // Fuse ALU operations followed by CBZ/CBNZ.
    if (SecondOpcode == AArch64::CBNZW || SecondOpcode == AArch64::CBNZX ||
        SecondOpcode == AArch64::CBZW || SecondOpcode == AArch64::CBZX)
      switch (FirstOpcode) {
      default:
        return false;
      case AArch64::ADDWri:
      case AArch64::ADDWrr:
      case AArch64::ADDXri:
      case AArch64::ADDXrr:
      case AArch64::ANDWri:
      case AArch64::ANDWrr:
      case AArch64::ANDXri:
      case AArch64::ANDXrr:
      case AArch64::EORWri:
      case AArch64::EORWrr:
      case AArch64::EORXri:
      case AArch64::EORXrr:
      case AArch64::ORRWri:
      case AArch64::ORRWrr:
      case AArch64::ORRXri:
      case AArch64::ORRXrr:
      case AArch64::SUBWri:
      case AArch64::SUBWrr:
      case AArch64::SUBXri:
      case AArch64::SUBXrr:
        return true;
      case AArch64::ADDWrs:
      case AArch64::ADDXrs:
      case AArch64::ANDWrs:
      case AArch64::ANDXrs:
      case AArch64::SUBWrs:
      case AArch64::SUBXrs:
      case AArch64::BICWrs:
      case AArch64::BICXrs:
        // Shift value can be 0 making these behave like the "rr" variant...
        return !II.hasShiftedReg(*FirstMI);
      case AArch64::INSTRUCTION_LIST_END:
        return true;
      }

  if (ST.hasFuseAES())
    // Fuse AES crypto operations.
    switch(FirstOpcode) {
    // AES encode.
    case AArch64::AESErr:
      return SecondOpcode == AArch64::AESMCrr ||
             SecondOpcode == AArch64::INSTRUCTION_LIST_END;
    // AES decode.
    case AArch64::AESDrr:
      return SecondOpcode == AArch64::AESIMCrr ||
             SecondOpcode == AArch64::INSTRUCTION_LIST_END;
    }

  if (ST.hasFuseLiterals())
    // Fuse literal generation operations.
    switch (FirstOpcode) {
    // PC relative address.
    case AArch64::ADRP:
      return SecondOpcode == AArch64::ADDXri ||
             SecondOpcode == AArch64::INSTRUCTION_LIST_END;
    // 32 bit immediate.
    case AArch64::MOVZWi:
      return (SecondOpcode == AArch64::MOVKWi &&
              SecondMI->getOperand(3).getImm() == 16) ||
             SecondOpcode == AArch64::INSTRUCTION_LIST_END;
    // Lower half of 64 bit immediate.
    case AArch64::MOVZXi:
      return (SecondOpcode == AArch64::MOVKXi &&
              SecondMI->getOperand(3).getImm() == 16) ||
             SecondOpcode == AArch64::INSTRUCTION_LIST_END;
    // Upper half of 64 bit immediate.
    case AArch64::MOVKXi:
      return FirstMI->getOperand(3).getImm() == 32 &&
             ((SecondOpcode == AArch64::MOVKXi &&
               SecondMI->getOperand(3).getImm() == 48) ||
              SecondOpcode == AArch64::INSTRUCTION_LIST_END);
    }

  return false;
}

/// \brief Implement the fusion of instr pairs in the scheduling DAG,
/// anchored at the instr in AnchorSU..
static bool scheduleAdjacentImpl(ScheduleDAGMI *DAG, SUnit &AnchorSU) {
  const MachineInstr *AnchorMI = AnchorSU.getInstr();
  if (!AnchorMI || AnchorMI->isPseudo() || AnchorMI->isTransient())
    return false;

  // If the anchor instr is the ExitSU, then consider its predecessors;
  // otherwise, its successors.
  bool Preds = (&AnchorSU == &DAG->ExitSU);
  SmallVectorImpl<SDep> &AnchorDeps = Preds ? AnchorSU.Preds : AnchorSU.Succs;

  const MachineInstr *FirstMI = Preds ? nullptr : AnchorMI;
  const MachineInstr *SecondMI = Preds ? AnchorMI : nullptr;

  // Check if the anchor instr may be fused.
  if (!shouldScheduleAdjacent(*DAG->TII, DAG->MF.getSubtarget(),
                              FirstMI, SecondMI))
    return false;

  // Explorer for fusion candidates among the dependencies of the anchor instr.
  for (SDep &Dep : AnchorDeps) {
    // Ignore dependencies that don't enforce ordering.
    if (Dep.isWeak())
      continue;

    SUnit &DepSU = *Dep.getSUnit();
    // Ignore the ExitSU if the dependents are successors.
    if (!Preds && &DepSU == &DAG->ExitSU)
      continue;

    const MachineInstr *DepMI = DepSU.getInstr();
    if (!DepMI || DepMI->isPseudo() || DepMI->isTransient())
      continue;

    FirstMI = Preds ? DepMI : AnchorMI;
    SecondMI = Preds ? AnchorMI : DepMI;
    if (!shouldScheduleAdjacent(*DAG->TII, DAG->MF.getSubtarget(),
                                FirstMI, SecondMI))
      continue;

    // Create a single weak edge between the adjacent instrs. The only effect is
    // to cause bottom-up scheduling to heavily prioritize the clustered instrs.
    SUnit &FirstSU = Preds ? DepSU : AnchorSU;
    SUnit &SecondSU = Preds ? AnchorSU : DepSU;
    DAG->addEdge(&SecondSU, SDep(&FirstSU, SDep::Cluster));

    // Adjust the latency between the anchor instr and its
    // predecessors/successors.
    for (SDep &IDep : AnchorDeps)
      if (IDep.getSUnit() == &DepSU)
        IDep.setLatency(0);

    // Adjust the latency between the dependent instr and its
    // successors/predecessors.
    for (SDep &IDep : Preds ? DepSU.Succs : DepSU.Preds)
      if (IDep.getSUnit() == &AnchorSU)
        IDep.setLatency(0);

    DEBUG(dbgs() << DAG->MF.getName() << "(): Macro fuse ";
          FirstSU.print(dbgs(), DAG); dbgs() << " - ";
          SecondSU.print(dbgs(), DAG); dbgs() << " /  ";
          dbgs() << DAG->TII->getName(FirstMI->getOpcode()) << " - " <<
                    DAG->TII->getName(SecondMI->getOpcode()) << '\n'; );

    if (&SecondSU != &DAG->ExitSU)
      // Make instructions dependent on FirstSU also dependent on SecondSU to
      // prevent them from being scheduled between FirstSU and and SecondSU.
      for (SUnit::const_succ_iterator
             SI = FirstSU.Succs.begin(), SE = FirstSU.Succs.end();
           SI != SE; ++SI) {
        if (!SI->getSUnit() || SI->getSUnit() == &SecondSU)
          continue;
        DEBUG(dbgs() << "  Copy Succ ";
              SI->getSUnit()->print(dbgs(), DAG); dbgs() << '\n';);
        DAG->addEdge(SI->getSUnit(), SDep(&SecondSU, SDep::Artificial));
      }

    ++NumFused;
    return true;
  }

  return false;
}

/// \brief Post-process the DAG to create cluster edges between instrs that may
/// be fused by the processor into a single operation.
class AArch64MacroFusion : public ScheduleDAGMutation {
public:
  AArch64MacroFusion() {}

  void apply(ScheduleDAGInstrs *DAGInstrs) override;
};

void AArch64MacroFusion::apply(ScheduleDAGInstrs *DAGInstrs) {
  ScheduleDAGMI *DAG = static_cast<ScheduleDAGMI*>(DAGInstrs);

  // For each of the SUnits in the scheduling block, try to fuse the instr in it
  // with one in its successors.
  for (SUnit &ISU : DAG->SUnits)
    scheduleAdjacentImpl(DAG, ISU);

  // Try to fuse the instr in the ExitSU with one in its predecessors.
  scheduleAdjacentImpl(DAG, DAG->ExitSU);
}

} // end namespace


namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createAArch64MacroFusionDAGMutation () {
  return EnableMacroFusion ? make_unique<AArch64MacroFusion>() : nullptr;
}

} // end namespace llvm
