//===-- RISCVCountLRSC.cpp - Count LR/SC instruction pairs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that counts the # of the LR/SC pairs. This pass
// should run just before code generation.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"

using namespace llvm;

#define RISCV_COUNT_LR_SC_NAME "RISC-V count LR/SC instruction pairs"
#define DEBUG_TYPE "riscvcntlrsc"

namespace {

class RISCVCountLRSC : public MachineFunctionPass {
  public:
    const RISCVSubtarget *STI;
    const RISCVInstrInfo *TII;

    static char ID;

    RISCVCountLRSC() : MachineFunctionPass(ID) {}
    ~RISCVCountLRSC();

    bool runOnMachineFunction(MachineFunction &MF) override;

    StringRef getPassName() const override { return RISCV_COUNT_LR_SC_NAME; }

    void print(raw_ostream &OS) const;

 private:

    unsigned countLRSC(MachineBasicBlock &MBB);
    unsigned totalCount = 0; 
};

} // end anonymous namespace

char RISCVCountLRSC::ID = 0;
INITIALIZE_PASS(RISCVCountLRSC, "riscv-count-lr-sc",
                RISCV_COUNT_LR_SC_NAME, false, false)

RISCVCountLRSC::~RISCVCountLRSC() {
  print(dbgs());
}

FunctionPass *llvm::createRISCVCountLRSCPass() {
  return new RISCVCountLRSC();
}

bool RISCVCountLRSC::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();
  TII = STI->getInstrInfo();

  // traverse through the machine basic blocks and 
  // get the running count of detected LR/SC pairs
  for (auto &MBB : MF) {
    totalCount += countLRSC(MBB);
  }
  return true;
}

unsigned RISCVCountLRSC::countLRSC(MachineBasicBlock &MBB) {

  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineBasicBlock::iterator E = MBB.end();

  unsigned bbCnt = 0;
  while (MBBI != E) {
    if (MBBI->getOpcode() == RISCV::LR_W || 
        MBBI->getOpcode() == RISCV::LR_D ||
        MBBI->getOpcode() == RISCV::LR_D_AQ ||
        MBBI->getOpcode() == RISCV::LR_W_AQ ||
        MBBI->getOpcode() == RISCV::LR_D_RL ||
        MBBI->getOpcode() == RISCV::LR_W_RL ||
        MBBI->getOpcode() == RISCV::LR_D_AQRL || 
        MBBI->getOpcode() == RISCV::LR_W_AQRL) {
      bbCnt ++;
    } else if (MBBI->getOpcode() == RISCV::SC_W  || 
               MBBI->getOpcode() == RISCV::SC_D  ||
               MBBI->getOpcode() == RISCV::SC_D_AQ ||
               MBBI->getOpcode() == RISCV::SC_W_AQ ||
               MBBI->getOpcode() == RISCV::SC_D_RL ||
               MBBI->getOpcode() == RISCV::SC_W_AQ ||
               MBBI->getOpcode() == RISCV::SC_D_AQRL ||
               MBBI->getOpcode() == RISCV::SC_W_AQRL) {
      bbCnt++;
    }
    MBBI++;
  }

  return bbCnt;
}

void RISCVCountLRSC::print(raw_ostream &OS) const {
  OS << "Number of LR/SC instruction pairs: " << " " << totalCount << "\n";
}


