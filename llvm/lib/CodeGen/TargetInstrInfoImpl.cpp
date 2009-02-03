//===-- TargetInstrInfoImpl.cpp - Target Instruction Information ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfoImpl class, it just provides default
// implementations of various methods.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
using namespace llvm;

// commuteInstruction - The default implementation of this method just exchanges
// operand 1 and 2.
MachineInstr *TargetInstrInfoImpl::commuteInstruction(MachineInstr *MI,
                                                      bool NewMI) const {
  assert(MI->getOperand(1).isReg() && MI->getOperand(2).isReg() &&
         "This only knows how to commute register operands so far");
  unsigned Reg1 = MI->getOperand(1).getReg();
  unsigned Reg2 = MI->getOperand(2).getReg();
  bool Reg1IsKill = MI->getOperand(1).isKill();
  bool Reg2IsKill = MI->getOperand(2).isKill();
  bool ChangeReg0 = false;
  if (MI->getOperand(0).getReg() == Reg1) {
    // Must be two address instruction!
    assert(MI->getDesc().getOperandConstraint(0, TOI::TIED_TO) &&
           "Expecting a two-address instruction!");
    Reg2IsKill = false;
    ChangeReg0 = true;
  }

  if (NewMI) {
    // Create a new instruction.
    unsigned Reg0 = ChangeReg0 ? Reg2 : MI->getOperand(0).getReg();
    bool Reg0IsDead = MI->getOperand(0).isDead();
    MachineFunction &MF = *MI->getParent()->getParent();
    return BuildMI(MF, MI->getDebugLoc(), MI->getDesc())
      .addReg(Reg0, true, false, false, Reg0IsDead)
      .addReg(Reg2, false, false, Reg2IsKill)
      .addReg(Reg1, false, false, Reg1IsKill);
  }

  if (ChangeReg0)
    MI->getOperand(0).setReg(Reg2);
  MI->getOperand(2).setReg(Reg1);
  MI->getOperand(1).setReg(Reg2);
  MI->getOperand(2).setIsKill(Reg1IsKill);
  MI->getOperand(1).setIsKill(Reg2IsKill);
  return MI;
}

/// CommuteChangesDestination - Return true if commuting the specified
/// instruction will also changes the destination operand. Also return the
/// current operand index of the would be new destination register by
/// reference. This can happen when the commutable instruction is also a
/// two-address instruction.
bool TargetInstrInfoImpl::CommuteChangesDestination(MachineInstr *MI,
                                                    unsigned &OpIdx) const{
  assert(MI->getOperand(1).isReg() && MI->getOperand(2).isReg() &&
         "This only knows how to commute register operands so far");
  if (MI->getOperand(0).getReg() == MI->getOperand(1).getReg()) {
    // Must be two address instruction!
    assert(MI->getDesc().getOperandConstraint(0, TOI::TIED_TO) &&
           "Expecting a two-address instruction!");
    OpIdx = 2;
    return true;
  }
  return false;
}


bool TargetInstrInfoImpl::PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const {
  bool MadeChange = false;
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.isPredicable())
    return false;
  
  for (unsigned j = 0, i = 0, e = MI->getNumOperands(); i != e; ++i) {
    if (TID.OpInfo[i].isPredicate()) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg()) {
        MO.setReg(Pred[j].getReg());
        MadeChange = true;
      } else if (MO.isImm()) {
        MO.setImm(Pred[j].getImm());
        MadeChange = true;
      } else if (MO.isMBB()) {
        MO.setMBB(Pred[j].getMBB());
        MadeChange = true;
      }
      ++j;
    }
  }
  return MadeChange;
}

void TargetInstrInfoImpl::reMaterialize(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned DestReg,
                                        const MachineInstr *Orig) const {
  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MI->getOperand(0).setReg(DestReg);
  MBB.insert(I, MI);
}

unsigned
TargetInstrInfoImpl::GetFunctionSizeInBytes(const MachineFunction &MF) const {
  unsigned FnSize = 0;
  for (MachineFunction::const_iterator MBBI = MF.begin(), E = MF.end();
       MBBI != E; ++MBBI) {
    const MachineBasicBlock &MBB = *MBBI;
    for (MachineBasicBlock::const_iterator I = MBB.begin(),E = MBB.end();
         I != E; ++I)
      FnSize += GetInstSizeInBytes(I);
  }
  return FnSize;
}

/// foldMemoryOperand - Attempt to fold a load or store of the specified stack
/// slot into the specified machine instruction for the specified operand(s).
/// If this is possible, a new instruction is returned with the specified
/// operand folded, otherwise NULL is returned. The client is responsible for
/// removing the old instruction and adding the new one in the instruction
/// stream.
MachineInstr*
TargetInstrInfo::foldMemoryOperand(MachineFunction &MF,
                                   MachineInstr* MI,
                                   const SmallVectorImpl<unsigned> &Ops,
                                   int FrameIndex) const {
  unsigned Flags = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    if (MI->getOperand(Ops[i]).isDef())
      Flags |= MachineMemOperand::MOStore;
    else
      Flags |= MachineMemOperand::MOLoad;

  // Ask the target to do the actual folding.
  MachineInstr *NewMI = foldMemoryOperandImpl(MF, MI, Ops, FrameIndex);
  if (!NewMI) return 0;

  assert((!(Flags & MachineMemOperand::MOStore) ||
          NewMI->getDesc().mayStore()) &&
         "Folded a def to a non-store!");
  assert((!(Flags & MachineMemOperand::MOLoad) ||
          NewMI->getDesc().mayLoad()) &&
         "Folded a use to a non-load!");
  const MachineFrameInfo &MFI = *MF.getFrameInfo();
  assert(MFI.getObjectOffset(FrameIndex) != -1);
  MachineMemOperand MMO(PseudoSourceValue::getFixedStack(FrameIndex),
                        Flags,
                        MFI.getObjectOffset(FrameIndex),
                        MFI.getObjectSize(FrameIndex),
                        MFI.getObjectAlignment(FrameIndex));
  NewMI->addMemOperand(MF, MMO);

  return NewMI;
}

/// foldMemoryOperand - Same as the previous version except it allows folding
/// of any load and store from / to any address, not just from a specific
/// stack slot.
MachineInstr*
TargetInstrInfo::foldMemoryOperand(MachineFunction &MF,
                                   MachineInstr* MI,
                                   const SmallVectorImpl<unsigned> &Ops,
                                   MachineInstr* LoadMI) const {
  assert(LoadMI->getDesc().canFoldAsLoad() && "LoadMI isn't foldable!");
#ifndef NDEBUG
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    assert(MI->getOperand(Ops[i]).isUse() && "Folding load into def!");
#endif

  // Ask the target to do the actual folding.
  MachineInstr *NewMI = foldMemoryOperandImpl(MF, MI, Ops, LoadMI);
  if (!NewMI) return 0;

  // Copy the memoperands from the load to the folded instruction.
  for (std::list<MachineMemOperand>::iterator I = LoadMI->memoperands_begin(),
       E = LoadMI->memoperands_end(); I != E; ++I)
    NewMI->addMemOperand(MF, *I);

  return NewMI;
}
