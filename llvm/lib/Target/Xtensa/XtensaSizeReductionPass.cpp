//===- XtensaSizeReductionPass.cpp - Xtensa Size Reduction ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Xtensa.h"
#include "XtensaInstrInfo.h"
#include "XtensaSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen//MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "xtensa-size-reduce-pass"

STATISTIC(NumReduced, "Number of 24-bit instructions reduced to 16-bit ones");

class XtensaSizeReduce : public MachineFunctionPass {
public:
  static char ID;
  XtensaSizeReduce() : MachineFunctionPass(ID) {}

  const XtensaSubtarget *Subtarget;
  static const XtensaInstrInfo *XtensaII;

  bool runOnMachineFunction(MachineFunction &MF) override;

  llvm::StringRef getPassName() const override {
    return "Xtensa instruction size reduction pass";
  }

private:
  /// Reduces width of instructions in the specified basic block.
  bool ReduceMBB(MachineBasicBlock &MBB);

  /// Attempts to reduce MI, returns true on success.
  bool ReduceMI(const MachineBasicBlock::instr_iterator &MII);
};

char XtensaSizeReduce::ID = 0;
const XtensaInstrInfo *XtensaSizeReduce::XtensaII;

bool XtensaSizeReduce::ReduceMI(const MachineBasicBlock::instr_iterator &MII) {
  MachineInstr *MI = &*MII;
  MachineBasicBlock &MBB = *MI->getParent();
  unsigned Opcode = MI->getOpcode();

  switch (Opcode) {
  case Xtensa::L32I: {
    MachineOperand Op0 = MI->getOperand(0);
    MachineOperand Op1 = MI->getOperand(1);
    MachineOperand Op2 = MI->getOperand(2);

    int64_t Imm = Op2.getImm();
    if (Imm >= 0 && Imm <= 60) {
      // Replace L32I to L32I.N
      DebugLoc dl = MI->getDebugLoc();
      const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::L32I_N);
      MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, NewMCID);
      MIB.add(Op0);
      MIB.add(Op1);
      MIB.add(Op2);
      // Transfer MI flags.
      MIB.setMIFlags(MI->getFlags());
      LLVM_DEBUG(dbgs() << "       to 16-bit: " << *MIB);
      NumReduced++;
      MBB.erase_instr(MI);
      return true;
    }
  } break;

  case Xtensa::S32I: {
    MachineOperand Op0 = MI->getOperand(0);
    MachineOperand Op1 = MI->getOperand(1);
    MachineOperand Op2 = MI->getOperand(2);

    int64_t Imm = Op2.getImm();
    if (Imm >= 0 && Imm <= 60) {
      // Replace S32I to S32I.N
      DebugLoc dl = MI->getDebugLoc();
      const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::S32I_N);
      MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, NewMCID);
      MIB.add(Op0);
      MIB.add(Op1);
      MIB.add(Op2);
      // Transfer MI flags.
      MIB.setMIFlags(MI->getFlags());
      LLVM_DEBUG(dbgs() << "       to 16-bit: " << *MIB);
      NumReduced++;
      MBB.erase_instr(MI);
      return true;
    }

  } break;

  case Xtensa::MOVI: {
    MachineOperand Op0 = MI->getOperand(0);
    MachineOperand Op1 = MI->getOperand(1);

    int64_t Imm = Op1.getImm();
    if (Imm >= -32 && Imm <= 95) {
      // Replace MOVI to MOVI.N
      DebugLoc dl = MI->getDebugLoc();
      const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::MOVI_N);
      MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, NewMCID);
      MIB.add(Op0);
      MIB.add(Op1);
      // Transfer MI flags.
      MIB.setMIFlags(MI->getFlags());
      LLVM_DEBUG(dbgs() << "       to 16-bit: " << *MIB);
      NumReduced++;
      MBB.erase_instr(MI);
      return true;
    }

  } break;

  case Xtensa::ADD: {
    MachineOperand Op0 = MI->getOperand(0);
    MachineOperand Op1 = MI->getOperand(1);
    MachineOperand Op2 = MI->getOperand(2);

    // Replace ADD to ADD.N
    DebugLoc dl = MI->getDebugLoc();
    const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::ADD_N);
    MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, NewMCID);
    MIB.add(Op0);
    MIB.add(Op1);
    MIB.add(Op2);
    // Transfer MI flags.
    MIB.setMIFlags(MI->getFlags());
    LLVM_DEBUG(dbgs() << "       to 16-bit: " << *MIB);
    NumReduced++;
    MBB.erase_instr(MI);
    return true;

  } break;

  case Xtensa::ADDI: {
    MachineOperand Op0 = MI->getOperand(0);
    MachineOperand Op1 = MI->getOperand(1);
    MachineOperand Op2 = MI->getOperand(2);

    int64_t Imm = Op2.getImm();
    if ((Imm >= 1 && Imm <= 15) || (Imm == -1)) {
      // Replace ADDI to ADDI.N
      DebugLoc dl = MI->getDebugLoc();
      const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::ADDI_N);
      MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, NewMCID);
      MIB.add(Op0);
      MIB.add(Op1);
      MIB.add(Op2);
      // Transfer MI flags.
      MIB.setMIFlags(MI->getFlags());
      LLVM_DEBUG(dbgs() << "       to 16-bit: " << *MIB);
      NumReduced++;
      MBB.erase_instr(MI);
      return true;
    }
  } break;

  case Xtensa::RET: {
    // Replace RET to RET.N
    DebugLoc dl = MI->getDebugLoc();
    const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::RET_N);
    MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, NewMCID);
    // Transfer MI flags.
    MIB.setMIFlags(MI->getFlags());
    LLVM_DEBUG(dbgs() << "       to 16-bit: " << *MIB);
    NumReduced++;
    MBB.erase_instr(MI);
    return true;
  } break;

  case Xtensa::RETW: {
    // Replace RETW to RETW.N
    DebugLoc dl = MI->getDebugLoc();
    const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::RETW_N);
    MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, NewMCID);
    // Transfer MI flags.
    MIB.setMIFlags(MI->getFlags());
    LLVM_DEBUG(dbgs() << "       to 16-bit: " << *MIB);
    NumReduced++;
    MBB.erase_instr(MI);
    return true;
  } break;

  default:
    break;
  }

  return false;
}

bool XtensaSizeReduce::ReduceMBB(MachineBasicBlock &MBB) {
  bool Modified = false;
  MachineBasicBlock::instr_iterator MII = MBB.instr_begin(),
                                    E = MBB.instr_end();
  MachineBasicBlock::instr_iterator NextMII;

  // Iterate through the instructions in the basic block
  for (; MII != E; MII = NextMII) {
    NextMII = std::next(MII);
    MachineInstr *MI = &*MII;

    // Don't reduce bundled instructions or pseudo operations
    if (MI->isBundle() || MI->isTransient())
      continue;

    // Try to reduce 24-bit instruction into 16-bit instruction
    Modified |= ReduceMI(MII);
  }

  return Modified;
}

bool XtensaSizeReduce::runOnMachineFunction(MachineFunction &MF) {

  Subtarget = &static_cast<const XtensaSubtarget &>(MF.getSubtarget());
  XtensaII = static_cast<const XtensaInstrInfo *>(Subtarget->getInstrInfo());
  bool Modified = false;

  if (!Subtarget->hasDensity())
    return Modified;

  MachineFunction::iterator I = MF.begin(), E = MF.end();

  for (; I != E; ++I)
    Modified |= ReduceMBB(*I);
  return Modified;
}

FunctionPass *llvm::createXtensaSizeReductionPass() {
  return new XtensaSizeReduce();
}
