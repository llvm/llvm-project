//===-- PISAReplaceIntrinsics.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass runs just after IRTranslator. It will replace some of intrinsics
// with an equivalent GMIR opcode, so that subsequent optimizations can be made.
//
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "PISAInstrInfo.h"
#include "PISATargetMachine.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "pisa-replace-intrinsics"
#define DEBUG_NAME "PISA replace intrinsics"

using namespace llvm;

namespace {

class PISAReplaceIntrinsics : public MachineFunctionPass {

public:
  static char ID;
  PISAReplaceIntrinsics() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return DEBUG_NAME; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const PISAInstrInfo *TII = nullptr;
};

} // namespace

char PISAReplaceIntrinsics::ID = 0;
INITIALIZE_PASS(PISAReplaceIntrinsics, DEBUG_TYPE, DEBUG_NAME, false, false)

bool PISAReplaceIntrinsics::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget<PISASubtarget>().getInstrInfo();
  bool Changed = false;

  SmallVector<MachineInstr *, 8> Delete;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() == TargetOpcode::G_INTRINSIC) {
        auto ID = cast<GIntrinsic>(MI).getIntrinsicID();
        switch (ID) {
        case Intrinsic::pisa_sbfe:
        case Intrinsic::pisa_ubfe: {
          // Res = pisa_[su]bfe (Base,Width,Offset)
          // Res = G_[SU]BFX (Base,LSB,Width)
          auto Opcode = (ID == Intrinsic::pisa_sbfe) ? TargetOpcode::G_SBFX
                                                     : TargetOpcode::G_UBFX;
          DebugLoc DL = MI.getDebugLoc();
          auto &Dst = MI.getOperand(0);
          BuildMI(MBB, &MI, DL, TII->get(Opcode))
              .addDef(Dst.getReg()) // Operand(0) is actual intrinsic
              .add(MI.getOperand(2))
              .add(MI.getOperand(4))
              .add(MI.getOperand(3));
          Delete.push_back(&MI);
          Changed = true;
        } break;
        default:
          break;
        }
      }
    }
  }
  for (auto *MI : Delete) {
    MI->eraseFromParent();
  }

  return Changed;
}

FunctionPass *llvm::createPISAReplaceIntrinsicsPass() {
  return new PISAReplaceIntrinsics();
}
