//===-- DelaySlotFiller.cpp - SPARC delay slot filler ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a simple local pass that attempts to fill delay slots with useful
// instructions. If no instructions can be moved into the delay slot, then a
// NOP is placed.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "delay-slot-filler"
#include "Sparc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");

static cl::opt<bool> DisableDelaySlotFiller(
  "disable-sparc-delay-filler",
  cl::init(false),
  cl::desc("Disable the Sparc delay slot filler."),
  cl::Hidden);

namespace {
  struct Filler : public MachineFunctionPass {
    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
    Filler(TargetMachine &tm) 
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "SPARC Delay Slot Filler";
    }

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F) {
      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

    bool isDelayFiller(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator candidate);

    void insertCallUses(MachineBasicBlock::iterator MI,
                        SmallSet<unsigned, 32>& RegUses);

    void insertDefsUses(MachineBasicBlock::iterator MI,
                        SmallSet<unsigned, 32>& RegDefs,
                        SmallSet<unsigned, 32>& RegUses);

    bool IsRegInSet(SmallSet<unsigned, 32>& RegSet,
                    unsigned Reg);

    bool delayHasHazard(MachineBasicBlock::iterator candidate,
                        bool &sawLoad, bool &sawStore,
                        SmallSet<unsigned, 32> &RegDefs,
                        SmallSet<unsigned, 32> &RegUses);

    MachineBasicBlock::iterator
    findDelayInstr(MachineBasicBlock &MBB, MachineBasicBlock::iterator slot);

    bool needsUnimp(MachineBasicBlock::iterator I, unsigned &StructSize);

  };
  char Filler::ID = 0;
} // end of anonymous namespace

/// createSparcDelaySlotFillerPass - Returns a pass that fills in delay
/// slots in Sparc MachineFunctions
///
FunctionPass *llvm::createSparcDelaySlotFillerPass(TargetMachine &tm) {
  return new Filler(tm);
}


/// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
/// We assume there is only one delay slot per delayed instruction.
///
bool Filler::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I)
    if (I->hasDelaySlot()) {
      MachineBasicBlock::iterator D = MBB.end();
      MachineBasicBlock::iterator J = I;

      if (!DisableDelaySlotFiller)
        D = findDelayInstr(MBB, I);

      ++FilledSlots;
      Changed = true;

      if (D == MBB.end())
        BuildMI(MBB, ++J, I->getDebugLoc(), TII->get(SP::NOP));
      else
        MBB.splice(++J, &MBB, D);
      unsigned structSize = 0;
      if (needsUnimp(I, structSize)) {
        MachineBasicBlock::iterator J = I;
        ++J; //skip the delay filler.
        BuildMI(MBB, ++J, I->getDebugLoc(),
                TII->get(SP::UNIMP)).addImm(structSize);
      }
    }
  return Changed;
}

MachineBasicBlock::iterator
Filler::findDelayInstr(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator slot)
{
  SmallSet<unsigned, 32> RegDefs;
  SmallSet<unsigned, 32> RegUses;
  bool sawLoad = false;
  bool sawStore = false;

  MachineBasicBlock::iterator I = slot;

  if (slot->getOpcode() == SP::RET)
    return MBB.end();

  if (slot->getOpcode() == SP::RETL) {
    --I;
    if (I->getOpcode() != SP::RESTORErr)
      return MBB.end();
    //change retl to ret
    slot->setDesc(TII->get(SP::RET));
    return I;
  }

  //Call's delay filler can def some of call's uses.
  if (slot->isCall())
    insertCallUses(slot, RegUses);
  else
    insertDefsUses(slot, RegDefs, RegUses);

  bool done = false;

  while (!done) {
    done = (I == MBB.begin());

    if (!done)
      --I;

    // skip debug value
    if (I->isDebugValue())
      continue;


    if (I->hasUnmodeledSideEffects()
        || I->isInlineAsm()
        || I->isLabel()
        || I->hasDelaySlot()
        || isDelayFiller(MBB, I))
      break;

    if (delayHasHazard(I, sawLoad, sawStore, RegDefs, RegUses)) {
      insertDefsUses(I, RegDefs, RegUses);
      continue;
    }

    return I;
  }
  return MBB.end();
}

bool Filler::delayHasHazard(MachineBasicBlock::iterator candidate,
                            bool &sawLoad,
                            bool &sawStore,
                            SmallSet<unsigned, 32> &RegDefs,
                            SmallSet<unsigned, 32> &RegUses)
{

  if (candidate->isImplicitDef() || candidate->isKill())
    return true;

  if (candidate->mayLoad()) {
    sawLoad = true;
    if (sawStore)
      return true;
  }

  if (candidate->mayStore()) {
    if (sawStore)
      return true;
    sawStore = true;
    if (sawLoad)
      return true;
  }

  for (unsigned i = 0, e = candidate->getNumOperands(); i!= e; ++i) {
    const MachineOperand &MO = candidate->getOperand(i);
    if (!MO.isReg())
      continue; // skip

    unsigned Reg = MO.getReg();

    if (MO.isDef()) {
      //check whether Reg is defined or used before delay slot.
      if (IsRegInSet(RegDefs, Reg) || IsRegInSet(RegUses, Reg))
        return true;
    }
    if (MO.isUse()) {
      //check whether Reg is defined before delay slot.
      if (IsRegInSet(RegDefs, Reg))
        return true;
    }
  }
  return false;
}


void Filler::insertCallUses(MachineBasicBlock::iterator MI,
                            SmallSet<unsigned, 32>& RegUses)
{

  switch(MI->getOpcode()) {
  default: llvm_unreachable("Unknown opcode.");
  case SP::CALL: break;
  case SP::JMPLrr:
  case SP::JMPLri:
    assert(MI->getNumOperands() >= 2);
    const MachineOperand &Reg = MI->getOperand(0);
    assert(Reg.isReg() && "JMPL first operand is not a register.");
    assert(Reg.isUse() && "JMPL first operand is not a use.");
    RegUses.insert(Reg.getReg());

    const MachineOperand &RegOrImm = MI->getOperand(1);
    if (RegOrImm.isImm())
        break;
    assert(RegOrImm.isReg() && "JMPLrr second operand is not a register.");
    assert(RegOrImm.isUse() && "JMPLrr second operand is not a use.");
    RegUses.insert(RegOrImm.getReg());
    break;
  }
}

//Insert Defs and Uses of MI into the sets RegDefs and RegUses.
void Filler::insertDefsUses(MachineBasicBlock::iterator MI,
                            SmallSet<unsigned, 32>& RegDefs,
                            SmallSet<unsigned, 32>& RegUses)
{
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;

    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;
    if (MO.isDef())
      RegDefs.insert(Reg);
    if (MO.isUse())
      RegUses.insert(Reg);

  }
}

//returns true if the Reg or its alias is in the RegSet.
bool Filler::IsRegInSet(SmallSet<unsigned, 32>& RegSet, unsigned Reg)
{
  // Check Reg and all aliased Registers.
  for (MCRegAliasIterator AI(Reg, TM.getRegisterInfo(), true);
       AI.isValid(); ++AI)
    if (RegSet.count(*AI))
      return true;
  return false;
}

// return true if the candidate is a delay filler.
bool Filler::isDelayFiller(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator candidate)
{
  if (candidate == MBB.begin())
    return false;
  if (candidate->getOpcode() == SP::UNIMP)
    return true;
  --candidate;
  return candidate->hasDelaySlot();
}

bool Filler::needsUnimp(MachineBasicBlock::iterator I, unsigned &StructSize)
{
  if (!I->isCall())
    return false;

  unsigned structSizeOpNum = 0;
  switch (I->getOpcode()) {
  default: llvm_unreachable("Unknown call opcode.");
  case SP::CALL: structSizeOpNum = 1; break;
  case SP::JMPLrr:
  case SP::JMPLri: structSizeOpNum = 2; break;
  }

  const MachineOperand &MO = I->getOperand(structSizeOpNum);
  if (!MO.isImm())
    return false;
  StructSize = MO.getImm();
  return true;
}
