//===-- RISCVInsertNTLHInsts.cpp - Insert NTLH extension instrution -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts non-temporal hint
// instructions where needed.
//
// It checks the MachineMemOperand of all MachineInstr.
// If the instruction has a MachineMemOperand and isNontemporal is true,
// then ntlh instruction is inserted before it.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define RISCV_INSERT_NTLH_INSTS_NAME "RISC-V insert NTLH instruction pass"

namespace {

class RISCVInsertNTLHInsts : public MachineFunctionPass {
public:
  const RISCVInstrInfo *TII;
  static char ID;

  RISCVInsertNTLHInsts() : MachineFunctionPass(ID) {
    initializeRISCVInsertNTLHInstsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return RISCV_INSERT_NTLH_INSTS_NAME;
  }
};

} // end of anonymous namespace

char RISCVInsertNTLHInsts::ID = 0;

bool RISCVInsertNTLHInsts::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  TII = ST.getInstrInfo();

  if (!ST.hasStdExtZihintntl())
    return false;

  bool Changed = false;
  for (auto &MBB : MF) {
    for (auto &MBBI : MBB) {
      if (MBBI.memoperands_empty())
        continue;
      MachineMemOperand *MMO = *(MBBI.memoperands_begin());
      if (MMO->isNonTemporal()) {
        DebugLoc DL = MBBI.getDebugLoc();
        if (ST.hasStdExtCOrZca() && ST.enableRVCHintInstrs())
          BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoCNTLALL));
        else
          BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoNTLALL));
        Changed = true;
      }
    }
  }

  return Changed;
}

INITIALIZE_PASS(RISCVInsertNTLHInsts, "riscv-insert-ntlh-insts",
                RISCV_INSERT_NTLH_INSTS_NAME, false, false)

namespace llvm {

FunctionPass *createRISCVInsertNTLHInstsPass() {
  return new RISCVInsertNTLHInsts();
}

} // end of namespace llvm
