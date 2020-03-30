//===-- SIPostRABundler.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass creates bundles of memory instructions to protect adjacent loads
/// and stores from beeing rescheduled apart from each other post-RA.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-post-ra-bundler"

namespace {

class SIPostRABundler : public MachineFunctionPass {
public:
  static char ID;

public:
  SIPostRABundler() : MachineFunctionPass(ID) {
    initializeSIPostRABundlerPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI post-RA bundler";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const SIRegisterInfo *TRI;

  SmallSet<Register, 16> Defs;

  bool isDependentLoad(const MachineInstr &MI) const;

};

} // End anonymous namespace.

INITIALIZE_PASS(SIPostRABundler, DEBUG_TYPE, "SI post-RA bundler", false, false)

char SIPostRABundler::ID = 0;

char &llvm::SIPostRABundlerID = SIPostRABundler::ID;

FunctionPass *llvm::createSIPostRABundlerPass() {
  return new SIPostRABundler();
}

bool SIPostRABundler::isDependentLoad(const MachineInstr &MI) const {
  if (!MI.mayLoad())
    return false;

  for (const MachineOperand &Op : MI.explicit_operands()) {
    if (!Op.isReg())
      continue;
    Register Reg = Op.getReg();
    for (Register Def : Defs)
      if (TRI->regsOverlap(Reg, Def))
        return true;
  }

  return false;
}

bool SIPostRABundler::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  bool Changed = false;
  const unsigned MemFlags = SIInstrFlags::MTBUF | SIInstrFlags::MUBUF |
                            SIInstrFlags::SMRD | SIInstrFlags::DS |
                            SIInstrFlags::FLAT | SIInstrFlags::MIMG;

  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::instr_iterator Next;
    MachineBasicBlock::instr_iterator B = MBB.instr_begin();
    MachineBasicBlock::instr_iterator E = MBB.instr_end();
    for (auto I = B; I != E; I = Next) {
      Next = std::next(I);

      if (I->isBundled() || !I->mayLoadOrStore() ||
          B->mayLoad() != I->mayLoad() || B->mayStore() != I->mayStore() ||
          (B->getDesc().TSFlags & MemFlags) !=
          (I->getDesc().TSFlags & MemFlags) ||
          isDependentLoad(*I)) {

        if (B != I) {
          if (std::next(B) != I) {
            finalizeBundle(MBB, B, I);
            Changed = true;
          }
          Next = I;
        }

        B = Next;
        Defs.clear();
        continue;
      }

      if (I->getNumExplicitDefs() == 0)
        continue;

      Defs.insert(I->defs().begin()->getReg());
    }

    if (B != E && std::next(B) != E) {
      finalizeBundle(MBB, B, E);
      Changed = true;
    }

    Defs.clear();
  }

  return Changed;
}
