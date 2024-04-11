//===- ARMRandezvousShadowStack.cpp - ARM Randezvous Shadow Stack ---------===//
//
// Copyright (c) 2021-2022, University of Rochester
//
// Part of the Randezvous Project, under the Apache License v2.0 with
// LLVM Exceptions.  See LICENSE.txt in the llvm directory for license
// information.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a pass that instruments ARM machine
// code to save/load the return address to/from a randomized compact shadow
// stack.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm_shadow_randomization"

#include "ARMTrampoline.h"
#include "ARMRandezvousCLR.h"
#include "ARMRandezvousOptions.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

STATISTIC(NumPrologues, "Number of prologues transformed to use shadow stack");
STATISTIC(NumEpilogues, "Number of epilogues transformed to use shadow stack");
STATISTIC(NumNullified, "Number of return addresses nullified");

char ARMTrampoline::ID = 0;

ARMTrampoline::ARMTrampoline() : ModulePass(ID) {}

StringRef ARMTrampoline::getPassName() const {
  return "ARM Trampoline Pass";
}

void ARMTrampoline::getAnalysisUsage(AnalysisUsage &AU) const {
  // We need this to access MachineFunctions
  AU.addRequired<MachineModuleInfoWrapperPass>();

  AU.setPreservesCFG();
  ModulePass::getAnalysisUsage(AU);
}

bool ARMTrampoline::insertNop(MachineInstr &MI) {
  MachineFunction &MF = *MI.getMF();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const DebugLoc &DL = MI.getDebugLoc();

  std::vector<MachineInstr *> NewInsts;
  for(int i=0;i<4;i++){
      NewInsts.push_back(
    BuildMI(MF,DL,TII->get(ARM::tHINT)).addImm(0).addImm(ARMCC::AL).addReg(0));//Nop
  }

  // 4. Now insert these new instructions into the basic block
  insertInstsBefore(MI, NewInsts);

  ++NumPrologues;
  return true;
}

bool ARMTrampoline::BlxTrampoline(MachineInstr &MI, MachineOperand &MO) {
    // before update:
    // blx ri

    // after update:
    // mov r8,ri
    // blx r12

    MachineFunction &MF = *MI.getMF();
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    const DebugLoc &DL = MI.getDebugLoc();

    std::vector<MachineInstr *> NewInsts;
    NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::tMOVr),ARM::R8)
                        .addReg(MI.getOperand(2).getReg())
                        .add(predOps(ARMCC::AL)));
    
    NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::tBLXr))
              .add(predOps(ARMCC::AL))
              .addReg(ARM::R11));
    insertInstsBefore(MI, NewInsts);

    removeInst(MI);

    return true;
}


//
// Method: runOnModule()
//
// Description:
//   This method is called when the PassManager wants this pass to transform
//   the specified Module.  This method
//
//   * creates a global variable as the shadow stack,
//
//   * creates a function that initializes the reserved registers for the
//     shadow stack, and
//
//   * transforms the Module to utilize the shadow stack for saving/restoring
//     return addresses and/or to nullify a saved return address on returns.
//
// Input:
//   M - A reference to the Module to transform.
//
// Output:
//   M - The transformed Module.
//
// Return value:
//   true  - The Module was transformed.
//   false - The Module was not transformed.
//

bool ARMTrampoline::runOnModule(Module &M) {
  if (!EnableTrampoline) {
    return false;
  }

  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();

  // Instrument pushes and pops in each function
  bool changed = false;

  for (Function &F : M) {
      MachineFunction *MF = MMI.getMachineFunction(F);
      if (MF == nullptr) {
        continue;
      }

      std::vector<std::pair<MachineInstr *, MachineOperand *>> BLs;
      std::vector<std::pair<MachineInstr *, MachineOperand *>> BLXs;

      for (MachineBasicBlock &MBB : *MF) {
        for (MachineInstr &MI : MBB) {
          switch (MI.getOpcode()) {
          case ARM::BL:
          case ARM::BL_pred:
          case ARM::BMOVPCB_CALL:
          case ARM::BL_PUSHLR:
          case ARM::tBL:
          case ARM::tBL_PUSHLR:
          case ARM::tTAILJMPr:
          case ARM::BMOVPCRX_CALL:
            BLs.push_back(std::make_pair(&MI, nullptr));
            break;
          case ARM::BLX:
          case ARM::BLX_noip:
          case ARM::BLX_pred:
          case ARM::BLX_pred_noip:
          case ARM::BX_CALL:
          case ARM::tBLXr:
          case ARM::tBLXr_noip:
          case ARM::tBLXNSr:
          case ARM::tBLXNS_CALL:
          case ARM::tBX_CALL:
          case ARM::tBLXi:
          case ARM::BLXi:
            for (MachineOperand &MO : MI.explicit_operands()) {
                if (MO.isReg()) {
                  if (MO.getReg() != ARM::LR) {
                    BLXs.push_back(std::make_pair(&MI, &MO));
                    break;
                  }
                }
            }
            break;
          default:
            break;
          }
        }
      }

      for (auto &MIMO : BLs) {
        changed |= insertNop(*MIMO.first);
      }

      for (auto &MIMO : BLXs) {
        changed |= BlxTrampoline(*MIMO.first,*MIMO.second);
      }

  }

  return changed;
}

ModulePass *llvm::createARMTrampoline(void) { return new ARMTrampoline(); }