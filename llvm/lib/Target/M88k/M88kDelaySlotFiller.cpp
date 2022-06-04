//===-- M88kDelaySlotFiller.cpp - Delay Slot Filler for M88k --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple pass to fill delay slots with useful instructions.
//
//===----------------------------------------------------------------------===//

#include "M88kInstrInfo.h"
#include "M88kTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "m88k-delay-slot-filler"

using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");

namespace {
class M88kDelaySlotFiller : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineBasicBlock::instr_iterator LastFiller;

public:
  static char ID;

  M88kDelaySlotFiller();

  MachineFunctionProperties getRequiredProperties() const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool runOnMachineBasicBlock(MachineBasicBlock &MBB);

  bool findDelayInstr(MachineBasicBlock &MBB,
                      MachineBasicBlock::instr_iterator Slot,
                      MachineBasicBlock::instr_iterator &Filler);

  void insertDefsUses(MachineBasicBlock::instr_iterator MI,
                      SmallSet<unsigned, 32> &RegDefs,
                      SmallSet<unsigned, 32> &RegUses);

  bool isRegInSet(SmallSet<unsigned, 32> &RegSet, unsigned Reg);

  bool delayHasHazard(MachineBasicBlock::instr_iterator MI, bool &SawLoad,
                      bool &SawStore, SmallSet<unsigned, 32> &RegDefs,
                      SmallSet<unsigned, 32> &RegUses);
};
} // end anonymous namespace

M88kDelaySlotFiller::M88kDelaySlotFiller() : MachineFunctionPass(ID) {
  initializeM88kDelaySlotFillerPass(*PassRegistry::getPassRegistry());
}

MachineFunctionProperties M88kDelaySlotFiller::getRequiredProperties() const {
  return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::NoVRegs);
}

bool M88kDelaySlotFiller::runOnMachineFunction(MachineFunction &MF) {
  const M88kSubtarget &Subtarget = MF.getSubtarget<M88kSubtarget>();
  TII = Subtarget.getInstrInfo();
  TRI = Subtarget.getRegisterInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= runOnMachineBasicBlock(MBB);

  // This pass invalidates liveness information when it reorders instructions to
  // fill delay slot. Without this, -verify-machineinstrs will fail.
  if (Changed)
    MF.getRegInfo().invalidateLiveness();

  return Changed;
}

// Fill in delay slots for the given basic block.
bool M88kDelaySlotFiller::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  LastFiller = MBB.instr_end();

  unsigned Opc;
  for (MachineBasicBlock::instr_iterator I = MBB.instr_begin();
       I != MBB.instr_end(); ++I) {
    if ((I->getDesc().isBranch() || I->getDesc().isCall() ||
         I->getDesc().isReturn()) &&
        (Opc = M88k::getOpcodeWithDelaySlot(I->getOpcode())) != -1) {
      MachineBasicBlock::instr_iterator InstrWithSlot = I;
      MachineBasicBlock::instr_iterator Filler = I;

      // Try to find a suitable filler instruction for the delay slot.
      if (!findDelayInstr(MBB, I, Filler))
        continue;

      // Replace the opcode.
      I->setDesc(TII->get(Opc));

      // Move the filler instruction into the delay slot position.
      MBB.splice(std::next(I), &MBB, Filler);

      // Update statistic count and record the change.
      ++FilledSlots;
      Changed = true;

      // Record the filler instruction that filled the delay slot.
      // The instruction after it will be visited in the next iteration.
      LastFiller = ++I;

      // Bundle the delay slot filler to InstrWithSlot so that the machine
      // verifier doesn't expect this instruction to be a terminator.
      InstrWithSlot->bundleWithSucc();
    }
  }
  return Changed;
}

bool M88kDelaySlotFiller::findDelayInstr(
    MachineBasicBlock &MBB, MachineBasicBlock::instr_iterator Slot,
    MachineBasicBlock::instr_iterator &Filler) {
  SmallSet<unsigned, 32> RegDefs;
  SmallSet<unsigned, 32> RegUses;

  insertDefsUses(Slot, RegDefs, RegUses);

  bool SawLoad = false;
  bool SawStore = false;

  for (MachineBasicBlock::reverse_instr_iterator I = ++Slot.getReverse();
       I != MBB.instr_rend(); ++I) {
    // Skip debug value.
    if (I->isDebugInstr())
      continue;

    // Convert to forward iterator.
    MachineBasicBlock::instr_iterator FI = I.getReverse();
    if (/*I->hasUnmodeledSideEffects() ||*/ I->isInlineAsm() || I->isLabel() ||
        FI == LastFiller || I->isPseudo())
      break;

    if (delayHasHazard(FI, SawLoad, SawStore, RegDefs, RegUses)) {
      insertDefsUses(FI, RegDefs, RegUses);
      continue;
    }
    Filler = FI;
    return true;
  }
  return false;
}

bool M88kDelaySlotFiller::delayHasHazard(MachineBasicBlock::instr_iterator MI,
                                         bool &SawLoad, bool &SawStore,
                                         SmallSet<unsigned, 32> &RegDefs,
                                         SmallSet<unsigned, 32> &RegUses) {
  if (MI->isImplicitDef() || MI->isKill())
    return true;

  // Loads or stores cannot be moved past a store to the delay slot
  // and stores cannot be moved past a load.
  if (MI->mayLoad()) {
    if (SawStore)
      return true;
    SawLoad = true;
  }

  if (MI->mayStore()) {
    if (SawStore)
      return true;
    SawStore = true;
    if (SawLoad)
      return true;
  }

  assert((!MI->isCall() && !MI->isReturn()) &&
         "Cannot put calls or returns in delay slot.");

  for (const MachineOperand &MO : MI->operands()) {
    Register Reg;

    if (!MO.isReg() || !(Reg = MO.getReg()))
      continue;

    if (MO.isDef()) {
      // Check whether Reg is defined or used before delay slot.
      if (isRegInSet(RegDefs, Reg) || isRegInSet(RegUses, Reg))
        return true;
    }
    if (MO.isUse()) {
      // Check whether Reg is defined before delay slot.
      if (isRegInSet(RegDefs, Reg))
        return true;
    }
  }
  return false;
}

// Insert Defs and Uses of MI into the sets RegDefs and RegUses.
void M88kDelaySlotFiller::insertDefsUses(MachineBasicBlock::instr_iterator MI,
                                         SmallSet<unsigned, 32> &RegDefs,
                                         SmallSet<unsigned, 32> &RegUses) {
  // Only examine the explicit and non-variadic operands.
  for (unsigned I = 0, E = MI->getDesc().getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    Register Reg;

    if (!MO.isReg() || !(Reg = MO.getReg()))
      continue;

    if (MO.isDef())
      RegDefs.insert(Reg);
    else if (MO.isUse())
      RegUses.insert(Reg);
  }
}

// Returns true if the Reg or its alias is in the RegSet.
bool M88kDelaySlotFiller::isRegInSet(SmallSet<unsigned, 32> &RegSet,
                                     unsigned Reg) {
  // Check Reg and all aliased Registers.
  for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
    if (RegSet.count(*AI))
      return true;
  return false;
}

char M88kDelaySlotFiller::ID = 0;
INITIALIZE_PASS(M88kDelaySlotFiller, DEBUG_TYPE, "Fill M88k delay slots", false,
                false)

namespace llvm {
FunctionPass *createM88kDelaySlotFiller() { return new M88kDelaySlotFiller(); }
} // end namespace llvm
