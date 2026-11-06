//===-- LVLGen.cpp - LVL instruction generator ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VE.h"
#include "VEInstrInfo.h"
#include "VERegisterInfo.h"
#include "VESubtarget.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "lvl-gen"

namespace {
struct VELVLGenLegacy : public MachineFunctionPass {
  static char ID;
  VELVLGenLegacy() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
char VELVLGenLegacy::ID = 0;

} // end of anonymous namespace

FunctionPass *llvm::createVELVLGenLegacyPass() { return new VELVLGenLegacy; }

static int getVLIndex(unsigned Opcode, const VEInstrInfo *TII) {
  const MCInstrDesc &MCID = TII->get(Opcode);

  // If an instruction has VLIndex information, return it.
  if (HAS_VLINDEX(MCID.TSFlags))
    return GET_VLINDEX(MCID.TSFlags);

  return -1;
}

// returns a register holding a vector length. NoRegister is returned when
// this MI does not have a vector length.
static unsigned getVL(const MachineInstr &MI, const VEInstrInfo *TII) {
  int Index = getVLIndex(MI.getOpcode(), TII);
  if (Index >= 0)
    return MI.getOperand(Index).getReg();

  return VE::NoRegister;
}

static bool runLVLGenOnMachineBasicBlock(MachineBasicBlock &MBB,
                                         const VEInstrInfo *TII,
                                         const VERegisterInfo *TRI) {
#define RegName(no)                                                            \
  (MBB.getParent()->getSubtarget<VESubtarget>().getRegisterInfo()->getName(no))

  bool Changed = false;
  bool HasRegForVL = false;
  unsigned RegForVL;

  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end();) {
    MachineBasicBlock::iterator MI = I;

    // Check whether MI uses a vector length operand.  If so, we prepare for VL
    // register.  We would like to reuse VL register as much as possible.  We
    // also would like to keep the number of LEA instructions as fewer as
    // possible.  Therefore, we use a regular scalar register to hold immediate
    // values to load VL register.  And try to reuse identical scalar registers
    // to avoid new LVLr instructions as much as possible.
    unsigned Reg = getVL(*MI, TII);
    if (Reg != VE::NoRegister) {
      LLVM_DEBUG(dbgs() << "Vector instruction found: ");
      LLVM_DEBUG(MI->dump());
      LLVM_DEBUG(dbgs() << "Vector length is " << RegName(Reg) << ". ");
      LLVM_DEBUG(dbgs() << "Current VL is "
                        << (HasRegForVL ? RegName(RegForVL) : "unknown")
                        << ". ");

      if (!HasRegForVL || RegForVL != Reg) {
        // Use VL, but a different value in a different scalar register.
        // So, generate new LVL instruction just before the current instruction.
        LLVM_DEBUG(dbgs() << "Generate a LVL instruction to load "
                          << RegName(Reg) << ".\n");
        BuildMI(MBB, I, MI->getDebugLoc(), TII->get(VE::LVLr)).addReg(Reg);
        HasRegForVL = true;
        RegForVL = Reg;
        Changed = true;
      } else {
        LLVM_DEBUG(dbgs() << "Reuse current VL.\n");
      }
    }
    // Check the update of a given scalar register holding an immediate value
    // for VL register.  Also, a call doesn't preserve VL register.
    if (HasRegForVL) {
      if (MI->definesRegister(RegForVL, TRI) ||
          MI->modifiesRegister(RegForVL, TRI) ||
          MI->killsRegister(RegForVL, TRI) || MI->isCall()) {
        // The latest VL is needed to be updated, so disable HasRegForVL.
        LLVM_DEBUG(dbgs() << RegName(RegForVL) << " is needed to be updated: ");
        LLVM_DEBUG(MI->dump());
        HasRegForVL = false;
      }
    }

    ++I;
  }
  return Changed;
}

static bool runLVLGenOnMachineFunction(MachineFunction &F) {
  LLVM_DEBUG(dbgs() << "********** Begin LVLGen **********\n");
  LLVM_DEBUG(dbgs() << "********** Function: " << F.getName() << '\n');
  LLVM_DEBUG(F.dump());

  bool Changed = false;

  const VESubtarget &Subtarget = F.getSubtarget<VESubtarget>();
  const VEInstrInfo *TII = Subtarget.getInstrInfo();
  const VERegisterInfo *TRI = Subtarget.getRegisterInfo();

  for (MachineBasicBlock &MBB : F)
    Changed |= runLVLGenOnMachineBasicBlock(MBB, TII, TRI);

  if (Changed) {
    LLVM_DEBUG(dbgs() << "\n");
    LLVM_DEBUG(F.dump());
  }
  LLVM_DEBUG(dbgs() << "********** End LVLGen **********\n");
  return Changed;
}

bool VELVLGenLegacy::runOnMachineFunction(MachineFunction &MF) {
  return runLVLGenOnMachineFunction(MF);
}

PreservedAnalyses VELVLGenPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {
  return runLVLGenOnMachineFunction(MF)
             ? getMachineFunctionPassPreservedAnalyses()
                   .preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}
