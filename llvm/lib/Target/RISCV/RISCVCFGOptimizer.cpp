//===- RISCVCFGOptimizer.cpp - CFG optimizations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <iostream>

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"

using namespace llvm;

#define DEBUG_TYPE "riscv_cfg"

namespace llvm {

FunctionPass *createRISCVCFGOptimizer();
void initializeRISCVCFGOptimizerPass(PassRegistry&);

} // end namespace llvm

namespace {

class RISCVCFGOptimizer : public MachineFunctionPass {
private:
  bool isOnFallThroughPath(MachineBasicBlock *MBB);

public:
  const RISCVInstrInfo *TII;
  static char ID;

  RISCVCFGOptimizer() : MachineFunctionPass(ID) {
    std::cout << "RISCVCFGOptimizer in" << std::endl;
    initializeRISCVCFGOptimizerPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "RISCV CFG Optimizer"; }
  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }
};

} // end anonymous namespace

char RISCVCFGOptimizer::ID = 0;

static bool IsConditionalBranch(int Opc) {
  switch (Opc) {
    case RISCV::BEQ:
    case RISCV::BNE:
    case RISCV::BLT:
    case RISCV::BGE:
    case RISCV::BLTU:
    case RISCV::BGEU:
      return true;
  }
  return false;
}

static bool IsUnconditionalJump(int Opc) {
  switch (Opc) {
    case RISCV::JAL:
    case RISCV::C_J:
    case RISCV::PseudoBR:
      return true;
  }

  return false;
}

int GetNotBranchOpecode(int Opc) {
  int NotOpc;
  switch (Opc) {
  case RISCV::BEQ:
    NotOpc = RISCV::BNE;
    break;
  case RISCV::BNE:
    NotOpc = RISCV::BEQ;
    break;
  case RISCV::BLT:
    NotOpc = RISCV::BGE;
    break;
  case RISCV::BGE:
    NotOpc = RISCV::BLT;
    break;
  case RISCV::BLTU:
    NotOpc = RISCV::BGEU;
    break;
  case RISCV::BGEU:
    NotOpc = RISCV::BLTU;
    break;
  default:
    NotOpc = RISCV::BNE;
  }
  return NotOpc;
}

bool RISCVCFGOptimizer::runOnMachineFunction(MachineFunction &MF) {
  bool ModCode = false;

  TII = static_cast<const RISCVInstrInfo *>(MF.getSubtarget().getInstrInfo());

  if (skipFunction(MF.getFunction())) {
    return ModCode;
  }

  MachineBasicBlock* BranchMBB = NULL;
  MachineBasicBlock* BranchTargetMBB = NULL;
  Register BranchReg;

  // Loop over all of the basic blocks.
  for (MachineFunction::iterator MBBb = MF.begin(), MBBe = MF.end(); MBBb != MBBe; ++MBBb) {
    MachineBasicBlock *MBB = &*MBBb;

    // Traverse the basic block.
    MachineBasicBlock::iterator MII = MBB->getFirstTerminator();
    if (MII != MBB->end()) {

      MachineInstr &MI = *MII;
      int Opc = MI.getOpcode();

      if (IsConditionalBranch(Opc)) {
          BranchTargetMBB = MI.getOperand(2).getMBB();
          BranchReg = MI.getOperand(0).getReg();
          BranchMBB = MBB;
          continue;
      }

      if (IsUnconditionalJump(Opc)) {

        if(BranchTargetMBB == NULL) {
          continue;
        }

        if(MBB->isLayoutSuccessor(BranchTargetMBB)) {
          // Copy condition variable
          MachineBasicBlock::iterator BranchMII = BranchMBB->getFirstTerminator();
          MachineInstr &BranchMI = *BranchMII;
          DebugLoc MvDL = BranchMI.getDebugLoc();

          MI.addOperand(MachineOperand::CreateReg(RISCV::X17, false, true));
          Register MvReg = MI.getOperand(1).getReg();

          BuildMI(*BranchMBB, BranchMI, MvDL, TII->get(RISCV::ADDI), MvReg)
                .addDef(BranchReg)
                .addImm(0);

          // Get the jump destination MBB
          MachineBasicBlock* JumpTargetMBB = MI.getOperand(0).getMBB();

          // Delete jump instruction
          MBB->erase(MI);

          // Insert branch instruction
          MachineInstr &TargetMI = BranchTargetMBB->front();
          DebugLoc BrDL = TargetMI.getDebugLoc();
          int NotOpcode = GetNotBranchOpecode(TargetMI.getOpcode());
          BuildMI(*BranchTargetMBB, TargetMI, BrDL, TII->get(NotOpcode))
                .addDef(MvReg)
                .addDef(RISCV::X0)
                .addMBB(JumpTargetMBB);

          BranchTargetMBB = NULL;
          BranchMBB = NULL;
          ModCode = true;
        }
      }
    }
  }

  return ModCode;
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

INITIALIZE_PASS(RISCVCFGOptimizer, "riscv-cfg", "RISCV CFG Optimizer", false, false)

FunctionPass *llvm::createRISCVCFGOptimizer() {
  return new RISCVCFGOptimizer();
}
