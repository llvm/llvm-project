//===- RISCVTagged.cpp - Replace instructions with their tagged version --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass rewrites Arithmetic instructions to their tagged version
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
#define DEBUG_TYPE "riscv-tagged-collapse"
#define RISCV_TAGGED_NAME "RISC-V Tagged definitions"

STATISTIC(NumDeadDefsReplaced, "Number of dead definitions replaced");

namespace {
class RISCVTagged : public MachineFunctionPass {
public:
  static char ID;

  RISCVTagged() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return RISCV_TAGGED_NAME; }
};
} // end anonymous namespace

char RISCVTagged::ID = 0;
INITIALIZE_PASS(RISCVTagged, DEBUG_TYPE,
                RISCV_TAGGED_NAME, false, false)

FunctionPass *llvm::createRISCVTaggedPass() {
  return new RISCVTagged();
}

bool RISCVTagged::runOnMachineFunction(MachineFunction &MF) {
    bool MadeChange = false;
    //Gets the list of targetInstructionInfos
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    //For every MachineFunction there is a list of MachineBasicBlocks which has a list of MachineInstr. We loop through that
    for (MachineBasicBlock &MBB : MF) {
        for (MachineInstr &MI : MBB) {
            switch (MI.getOpcode())
            {
            case RISCV::SLLI:
                MI.setDesc(TII->get(RISCV::SLI));
                MadeChange = true;
                break;
            case RISCV::SRLI:
            case RISCV::SRAI:
                MI.setDesc(TII->get(RISCV::SRI));
                MadeChange = true;
                break;
            default:
                break;
            }
        }
    }

    #ifndef NDEBUG
        // Safety check: ensure no pseudos remain (helps catch missed cases).
        for (MachineBasicBlock &MBB : MF) {
        for (MachineInstr &MI : MBB) {
            unsigned Opc = MI.getOpcode();
            if (Opc == RISCV::SLLI || Opc == RISCV::SRLI || Opc == RISCV::SRAI) {
            report_fatal_error("Tagged collapse: found unlowered SLLI/SRLI/SRAI");
            }
        }
        }
    #endif
    return MadeChange;
}
