//===- DeadMachineInstructionElim.cpp - Remove dead machine instructions --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an extremely simple MachineInstr-level dead-code-elimination pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN DeadMachineInstructionElim : 
        public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &MF);
    
    const TargetRegisterInfo *TRI;
    const MachineRegisterInfo *MRI;
    const TargetInstrInfo *TII;
    BitVector LivePhysRegs;

  public:
    static char ID; // Pass identification, replacement for typeid
    DeadMachineInstructionElim() : MachineFunctionPass(&ID) {}

  private:
    bool isDead(MachineInstr *MI) const;
  };
}
char DeadMachineInstructionElim::ID = 0;

static RegisterPass<DeadMachineInstructionElim>
Y("dead-mi-elimination",
  "Remove dead machine instructions");

FunctionPass *llvm::createDeadMachineInstructionElimPass() {
  return new DeadMachineInstructionElim();
}

bool DeadMachineInstructionElim::isDead(MachineInstr *MI) const {
  // Don't delete instructions with side effects.
  bool SawStore = false;
  if (!MI->isSafeToMove(TII, SawStore))
    return false;

  // Examine each operand.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef()) {
      unsigned Reg = MO.getReg();
      if (TargetRegisterInfo::isPhysicalRegister(Reg) ?
          LivePhysRegs[Reg] : !MRI->use_empty(Reg)) {
        // This def has a use. Don't delete the instruction!
        return false;
      }
    }
  }

  // If there are no defs with uses, the instruction is dead.
  return true;
}

bool DeadMachineInstructionElim::runOnMachineFunction(MachineFunction &MF) {
  bool AnyChanges = false;
  MRI = &MF.getRegInfo();
  TRI = MF.getTarget().getRegisterInfo();
  TII = MF.getTarget().getInstrInfo();

  // Compute a bitvector to represent all non-allocatable physregs.
  BitVector NonAllocatableRegs = TRI->getAllocatableSet(MF);
  NonAllocatableRegs.flip();

  // Loop over all instructions in all blocks, from bottom to top, so that it's
  // more likely that chains of dependent but ultimately dead instructions will
  // be cleaned up.
  for (MachineFunction::reverse_iterator I = MF.rbegin(), E = MF.rend();
       I != E; ++I) {
    MachineBasicBlock *MBB = &*I;

    // Start out assuming that all non-allocatable registers are live
    // out of this block.
    LivePhysRegs = NonAllocatableRegs;

    // Also add any explicit live-out physregs for this block.
    if (!MBB->empty() && MBB->back().getDesc().isReturn())
      for (MachineRegisterInfo::liveout_iterator LOI = MRI->liveout_begin(),
           LOE = MRI->liveout_end(); LOI != LOE; ++LOI) {
        unsigned Reg = *LOI;
        if (TargetRegisterInfo::isPhysicalRegister(Reg))
          LivePhysRegs.set(Reg);
      }

    // Now scan the instructions and delete dead ones, tracking physreg
    // liveness as we go.
    for (MachineBasicBlock::reverse_iterator MII = MBB->rbegin(),
         MIE = MBB->rend(); MII != MIE; ) {
      MachineInstr *MI = &*MII;

      // If the instruction is dead, delete it!
      if (isDead(MI)) {
        DOUT << "DeadMachineInstructionElim: DELETING: " << *MI;
        AnyChanges = true;
        MI->eraseFromParent();
        MIE = MBB->rend();
        // MII is now pointing to the next instruction to process,
        // so don't increment it.
        continue;
      }

      // Record the physreg defs.
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        if (MO.isReg() && MO.isDef()) {
          unsigned Reg = MO.getReg();
          if (Reg != 0 && TargetRegisterInfo::isPhysicalRegister(Reg)) {
            LivePhysRegs.reset(Reg);
            // Check the subreg set, not the alias set, because a def
            // of a super-register may still be partially live after
            // this def.
            for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
                 *SubRegs; ++SubRegs)
              LivePhysRegs.reset(*SubRegs);
          }
        }
      }
      // Record the physreg uses, after the defs, in case a physreg is
      // both defined and used in the same instruction.
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        if (MO.isReg() && MO.isUse()) {
          unsigned Reg = MO.getReg();
          if (Reg != 0 && TargetRegisterInfo::isPhysicalRegister(Reg)) {
            LivePhysRegs.set(Reg);
            for (const unsigned *AliasSet = TRI->getAliasSet(Reg);
                 *AliasSet; ++AliasSet)
              LivePhysRegs.set(*AliasSet);
          }
        }
      }

      // We didn't delete the current instruction, so increment MII to
      // the next one.
      ++MII;
    }
  }

  LivePhysRegs.clear();
  return AnyChanges;
}
