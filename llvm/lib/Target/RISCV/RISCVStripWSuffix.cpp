//===-------------- RISCVStripWSuffix.cpp - -w Suffix Removal -------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass removes the -w suffix from each addiw and slliw instructions
// whenever all users are dependent only on the lower word of the result of the
// instruction. We do this only for addiw and slliw because the -w forms are
// less compressible.
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"

using namespace llvm;

static cl::opt<bool> DisableStripWSuffix("riscv-disable-strip-w-suffix",
                                         cl::desc("Disable strip W suffix"),
                                         cl::init(false), cl::Hidden);

namespace {

class RISCVStripWSuffix : public MachineFunctionPass {
public:
  static char ID;

  RISCVStripWSuffix() : MachineFunctionPass(ID) {
    initializeRISCVStripWSuffixPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "RISC-V Strip W Suffix"; }
};

} // end anonymous namespace

char RISCVStripWSuffix::ID = 0;
INITIALIZE_PASS(RISCVStripWSuffix, "riscv-strip-w-suffix",
                "RISC-V Strip W Suffix", false, false)

FunctionPass *llvm::createRISCVStripWSuffixPass() {
  return new RISCVStripWSuffix();
}

bool RISCVStripWSuffix::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || DisableStripWSuffix)
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo &TII = *ST.getInstrInfo();

  if (!ST.is64Bit())
    return false;

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), IE = MBB.end(); I != IE; ++I) {
      MachineInstr &MI = *I;

      unsigned Opc;
      switch (MI.getOpcode()) {
      default:
        continue;
      case RISCV::ADDW:  Opc = RISCV::ADD;  break;
      case RISCV::MULW:  Opc = RISCV::MUL;  break;
      case RISCV::SLLIW: Opc = RISCV::SLLI; break;
      }

      if (TII.hasAllWUsers(MI, MRI)) {
        MI.setDesc(TII.get(Opc));
        MadeChange = true;
      }
    }
  }

  return MadeChange;
}
