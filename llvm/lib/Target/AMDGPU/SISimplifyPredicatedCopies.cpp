//===-- SISimplifyPredicatedCopies.cpp - Simplify Copies after regalloc
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Simplify the predicated COPY (PRED_COPY) instructions for various register
/// classes. AMDGPU vector register copies have a predicated dependency with
/// EXEC register and should be marked exec as an implicit operand post-RA. The
/// scalar registers don't carry any such dependency and hence the regular COPY
/// opcode can be used. AMDGPU by default uses PRED_COPY opcode right from the
/// instruction selection and this pass would simplify the COPY opcode and the
/// implicit operand field as mentioned above.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-simplify-predicated-copies"

namespace {

class SISimplifyPredicatedCopies : public MachineFunctionPass {
public:
  static char ID;

  SISimplifyPredicatedCopies() : MachineFunctionPass(ID) {
    initializeSISimplifyPredicatedCopiesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Simplify Predicated Copies";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const SIRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;
  SIMachineFunctionInfo *MFI;
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SISimplifyPredicatedCopies, DEBUG_TYPE,
                      "SI Simplify Predicated Copies", false, false)
INITIALIZE_PASS_END(SISimplifyPredicatedCopies, DEBUG_TYPE,
                    "SI Simplify Predicated Copies", false, false)

char SISimplifyPredicatedCopies::ID = 0;

char &llvm::SISimplifyPredicatedCopiesID = SISimplifyPredicatedCopies::ID;

bool SISimplifyPredicatedCopies::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();

  MFI = MF.getInfo<SIMachineFunctionInfo>();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      switch (Opcode) {
      case AMDGPU::COPY:
      case AMDGPU::PRED_COPY:
        if (!TII->isVGPRCopy(MI) &&
            TRI->isSGPRReg(*MRI, MI.getOperand(1).getReg())) {
          // For PRED_COPY with SGPR regclass, change the opcode back to the
          // regular COPY.
          if (Opcode == AMDGPU::PRED_COPY) {
            LLVM_DEBUG(dbgs() << MI << " to use COPY opcode");
            MI.setDesc(TII->get(AMDGPU::COPY));
            Changed = true;
          }
        } else {
          // For vector registers, add implicit exec use.
          if (!MI.readsRegister(AMDGPU::EXEC, TRI)) {
            MI.addOperand(MF,
                          MachineOperand::CreateReg(AMDGPU::EXEC, false, true));
            LLVM_DEBUG(dbgs() << "Add exec use to " << MI);
            Changed = true;
          }
        }
        break;
      default:
        break;
      }
    }
  }

  return Changed;
}
