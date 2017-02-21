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

/// \brief Verify that the instruction pair, First and Second,
/// should be scheduled back to back.  Given an anchor instruction, if the other
/// instruction is unspecified, then verify that the anchor instruction may be
/// part of a pair at all.
static bool shouldScheduleAdjacent(const AArch64InstrInfo &TII,
                                   const AArch64Subtarget &ST,
                                   const MachineInstr *First,
                                   const MachineInstr *Second) {
  assert((First || Second) && "At least one instr must be specified");
  unsigned FirstOpcode =
    First ? First->getOpcode()
	  : static_cast<unsigned>(AArch64::INSTRUCTION_LIST_END);
  unsigned SecondOpcode =
    Second ? Second->getOpcode()
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
        return !TII.hasShiftedReg(*First);
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
        return !TII.hasShiftedReg(*First);
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
              Second->getOperand(3).getImm() == 16) ||
             SecondOpcode == AArch64::INSTRUCTION_LIST_END;
    // Lower half of 64 bit immediate.
    case AArch64::MOVZXi:
      return (SecondOpcode == AArch64::MOVKXi &&
              Second->getOperand(3).getImm() == 16) ||
             SecondOpcode == AArch64::INSTRUCTION_LIST_END;
    // Upper half of 64 bit immediate.
    case AArch64::MOVKXi:
      return First->getOperand(3).getImm() == 32 &&
             ((SecondOpcode == AArch64::MOVKXi &&
               Second->getOperand(3).getImm() == 48) ||
              SecondOpcode == AArch64::INSTRUCTION_LIST_END);
    }

  return false;
}

/// \brief Implement the fusion of instruction pairs in the scheduling
/// DAG, anchored at the instruction in ASU. Preds
/// indicates if its dependencies in \param APreds are predecessors instead of
/// successors.
static bool scheduleAdjacentImpl(ScheduleDAGMI *DAG, SUnit *ASU,
                                 SmallVectorImpl<SDep> &APreds, bool Preds) {
  const AArch64InstrInfo *TII = static_cast<const AArch64InstrInfo *>(DAG->TII);
  const AArch64Subtarget &ST = DAG->MF.getSubtarget<AArch64Subtarget>();

  const MachineInstr *AMI = ASU->getInstr();
  if (!AMI || AMI->isPseudo() || AMI->isTransient() ||
      (Preds && !shouldScheduleAdjacent(*TII, ST, nullptr, AMI)) ||
      (!Preds && !shouldScheduleAdjacent(*TII, ST, AMI, nullptr)))
    return false;

  for (SDep &BDep : APreds) {
    if (BDep.isWeak())
      continue;

    SUnit *BSU = BDep.getSUnit();
    const MachineInstr *BMI = BSU->getInstr();
    if (!BMI || BMI->isPseudo() || BMI->isTransient() ||
        (Preds && !shouldScheduleAdjacent(*TII, ST, BMI, AMI)) ||
        (!Preds && !shouldScheduleAdjacent(*TII, ST, AMI, BMI)))
      continue;

    // Create a single weak edge between the adjacent instrs. The only
    // effect is to cause bottom-up scheduling to heavily prioritize the
    // clustered instrs.
    if (Preds)
      DAG->addEdge(ASU, SDep(BSU, SDep::Cluster));
    else
      DAG->addEdge(BSU, SDep(ASU, SDep::Cluster));

    // Adjust the latency between the 1st instr and its predecessors/successors.
    for (SDep &Dep : APreds)
      if (Dep.getSUnit() == BSU)
        Dep.setLatency(0);

    // Adjust the latency between the 2nd instr and its successors/predecessors.
    auto &BSuccs = Preds ? BSU->Succs : BSU->Preds;
    for (SDep &Dep : BSuccs)
      if (Dep.getSUnit() == ASU)
        Dep.setLatency(0);

    ++NumFused;
    DEBUG(dbgs() << "Macro fuse ";
          Preds ? BSU->print(dbgs(), DAG) : ASU->print(dbgs(), DAG);
          dbgs() << " - ";
          Preds ? ASU->print(dbgs(), DAG) : BSU->print(dbgs(), DAG);
          dbgs() << '\n');

    return true;
  }

  return false;
}

/// \brief Post-process the DAG to create cluster edges between instructions
/// that may be fused by the processor into a single operation.
class AArch64MacroFusion : public ScheduleDAGMutation {
public:
  AArch64MacroFusion() {}

  void apply(ScheduleDAGInstrs *DAGInstrs) override;
};

void AArch64MacroFusion::apply(ScheduleDAGInstrs *DAGInstrs) {
  ScheduleDAGMI *DAG = static_cast<ScheduleDAGMI*>(DAGInstrs);

  // For each of the SUnits in the scheduling block, try to fuse the instruction
  // in it with one in its successors.
  for (SUnit &ASU : DAG->SUnits)
    scheduleAdjacentImpl(DAG, &ASU, ASU.Succs, false);

  // Try to fuse the instruction in the ExitSU with one in its predecessors.
  scheduleAdjacentImpl(DAG, &DAG->ExitSU, DAG->ExitSU.Preds, true);
}

} // end namespace


namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createAArch64MacroFusionDAGMutation () {
  return EnableMacroFusion ? make_unique<AArch64MacroFusion>() : nullptr;
}

} // end namespace llvm
