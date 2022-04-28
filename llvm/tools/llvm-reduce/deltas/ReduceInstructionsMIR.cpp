//===- ReduceInstructionsMIR.cpp - Specialized Delta Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce uninteresting MachineInstr from the MachineFunction.
//
//===----------------------------------------------------------------------===//

#include "ReduceInstructionsMIR.h"

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

static Register getPrevDefOfRCInMBB(MachineBasicBlock &MBB,
                                    MachineBasicBlock::reverse_iterator &RI,
                                    const RegClassOrRegBank &RC, LLT Ty,
                                    SetVector<MachineInstr *> &ExcludeMIs) {
  auto MRI = &MBB.getParent()->getRegInfo();
  for (MachineBasicBlock::reverse_instr_iterator E = MBB.instr_rend(); RI != E;
       ++RI) {
    auto &MI = *RI;
    // All Def operands explicit and implicit.
    for (auto &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      auto Reg = MO.getReg();
      if (Register::isPhysicalRegister(Reg))
        continue;

      if (MRI->getRegClassOrRegBank(Reg) == RC && MRI->getType(Reg) == Ty &&
          !ExcludeMIs.count(MO.getParent()))
        return Reg;
    }
  }
  return 0;
}

static void extractInstrFromFunction(Oracle &O, MachineFunction &MF) {
  MachineDominatorTree MDT;
  MDT.runOnMachineFunction(MF);

  auto MRI = &MF.getRegInfo();
  SetVector<MachineInstr *> ToDelete;

  MachineInstr *TopMI = nullptr;

  // Mark MIs for deletion according to some criteria.
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.isTerminator())
        continue;
      if (MBB.isEntryBlock() && !TopMI) {
        TopMI = &MI;
        continue;
      }
      if (!O.shouldKeep())
        ToDelete.insert(&MI);
    }
  }

  // For each MI to be deleted update users of regs defined by that MI to use
  // some other dominating definition (that is not to be deleted).
  for (auto *MI : ToDelete) {
    for (auto &MO : MI->operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      auto Reg = MO.getReg();
      if (Register::isPhysicalRegister(Reg))
        continue;
      auto UI = MRI->use_begin(Reg);
      auto UE = MRI->use_end();

      const auto &RegRC = MRI->getRegClassOrRegBank(Reg);
      LLT RegTy = MRI->getType(Reg);

      Register NewReg = 0;
      // If this is not a physical register and there are some uses.
      if (UI != UE) {
        MachineBasicBlock::reverse_iterator RI(*MI);
        MachineBasicBlock *BB = MI->getParent();
        ++RI;
        while (NewReg == 0 && BB) {
          NewReg = getPrevDefOfRCInMBB(*BB, RI, RegRC, RegTy, ToDelete);
          // Prepare for idom(BB).
          if (auto *IDM = MDT.getNode(BB)->getIDom()) {
            BB = IDM->getBlock();
            RI = BB->rbegin();
          } else {
            BB = nullptr;
          }
        }
      }

      // If no dominating definition was found then add an implicit one to the
      // first instruction in the entry block.

      // FIXME: This should really insert IMPLICIT_DEF or G_IMPLICIT_DEF. We
      // need to refine the reduction quality metric from number of serialized
      // bytes to continue progressing if we're going to introduce new
      // instructions.
      if (!NewReg && TopMI) {
        NewReg = MRI->cloneVirtualRegister(Reg);
        TopMI->addOperand(MachineOperand::CreateReg(
            NewReg, true /*IsDef*/, true /*IsImp*/, false /*IsKill*/,
            MO.isDead(), MO.isUndef(), MO.isEarlyClobber(), MO.getSubReg(),
            /*IsDebug*/ false, MO.isInternalRead()));
      }

      // Update all uses.
      while (UI != UE) {
        auto &UMO = *UI++;
        UMO.setReg(NewReg);
      }
    }
  }

  // Finally delete the MIs.
  for (auto *MI : ToDelete)
    MI->eraseFromParent();
}

static void extractInstrFromModule(Oracle &O, ReducerWorkItem &WorkItem) {
  for (const Function &F : WorkItem.getModule()) {
    if (MachineFunction *MF = WorkItem.MMI->getMachineFunction(F))
      extractInstrFromFunction(O, *MF);
  }
}

void llvm::reduceInstructionsMIRDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Instructions...\n";
  runDeltaPass(Test, extractInstrFromModule);
}
