//===- ARMRandezvousInstrumentor.cpp - A helper class for instrumentation -===//
//
// Copyright (c) 2021-2022, University of Rochester
//
// Part of the Randezvous Project, under the Apache License v2.0 with
// LLVM Exceptions.  See LICENSE.txt in the llvm directory for license
// information.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a class that can help passes of
// its subclass easily instrument ARM machine IR without concerns of breaking
// IT blocks.
//
//===----------------------------------------------------------------------===//

#include "ARMRandezvousInstrumentor.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

//
// Method: getITBlockSize()
//
// Description:
//   This method computes how many predicated instructions an IT instruction
//   covers.
//
// Input:
//   IT - A reference to an IT instruction.
//
// Return value:
//   The number of predicated instructions IT covers.
//
unsigned
ARMRandezvousInstrumentor::getITBlockSize(const MachineInstr & IT) {
  assert(IT.getOpcode() == ARM::t2IT && "Not an IT instruction!");

  unsigned Mask = IT.getOperand(1).getImm() & 0xf;
  assert(Mask != 0 && "Invalid IT mask!");

  if (Mask & 0x1) {
    return 4;
  } else if (Mask & 0x2) {
    return 3;
  } else if (Mask & 0x4) {
    return 2;
  } else {
    return 1;
  }
}

//
// Method: findIT()
//
// Description:
//   This method finds the IT instruction that forms an IT block containing a
//   given instruction MI.  It also computes the distance (from 0 to 4, 0 means
//   MI itself is IT) between the IT and MI.  If there is no such IT, a null
//   pointer is returned.
//
// Input:
//   MI - A reference to an instruction from which to find IT.
//
// Output:
//   distance - A reference to an unsigned to store the distance.
//
// Return value:
//   A pointer to IT if found, nullptr otherwise.
//
MachineInstr *
ARMRandezvousInstrumentor::findIT(MachineInstr & MI, unsigned & distance) {
  MachineInstr * Prev = &MI;
  unsigned dist = 0;
  while (Prev != nullptr && dist < 5 && Prev->getOpcode() != ARM::t2IT) {
    // Only count non-meta instructions
    if (!Prev->isMetaInstruction()) {
      ++dist;
    }
    Prev = Prev->getPrevNode();
  }
  if (Prev != nullptr && dist < 5 && Prev->getOpcode() == ARM::t2IT) {
    if (getITBlockSize(*Prev) >= dist) {
      distance = dist;
      return Prev;
    }
  }
  return nullptr;
}

//
// Method: findIT()
//
// Description:
//   This method finds the IT instruction that forms an IT block containing a
//   given instruction MI.  It also computes the distance (from 0 to 4, 0 means
//   MI itself is IT) between the IT and MI.  If there is no such IT, a null
//   pointer is returned.
//
// Input:
//   MI - A const reference to an instruction from which to find IT.
//
// Output:
//   distance - A reference to an unsigned to store the distance.
//
// Return value:
//   A const pointer to IT if found, nullptr otherwise.
//
const MachineInstr *
ARMRandezvousInstrumentor::findIT(const MachineInstr & MI, unsigned & distance) {
  return findIT(const_cast<MachineInstr &>(MI), distance);
}

//
// Method: insertInstBefore()
//
// Description:
//   This method inserts an instruction Inst before a given instruction MI.  If
//   MI is a predicated instruction within an IT block, then Inst will have the
//   same predicate as MI and also end up in an IT block.
//
// Inputs:
//   MI   - A reference to an instruction before which to insert Inst.
//   Inst - A pointer to an instruction to insert.
//
void
ARMRandezvousInstrumentor::insertInstBefore(MachineInstr & MI,
                                            MachineInstr * Inst) {
  insertInstsBefore(MI, { Inst });
}

//
// Method: insertInstAfter()
//
// Description:
//   This method inserts an instruction Inst after a given instruction MI.  If
//   MI is a predicated instruction within an IT block, then Inst will have the
//   same predicate as MI and also end up in an IT block.
//
// Inputs:
//   MI   - A reference to an instruction after which to insert Inst.
//   Inst - A pointer to an instruction to insert.
//
void
ARMRandezvousInstrumentor::insertInstAfter(MachineInstr & MI,
                                           MachineInstr * Inst) {
  insertInstsAfter(MI, { Inst });
}

//
// Method: insertInstsBefore()
//
// Description:
//   This method inserts a group of instructions contained in an array before a
//   given instruction MI.  If MI is a predicated instruction within an IT
//   block, then the new instructions will have the same predicate as MI and
//   also end up in one or more IT blocks.
//
// Inputs:
//   MI    - A reference to an instruction before which to insert instructions.
//   Insts - A reference to an array containing the instructions.
//
void
ARMRandezvousInstrumentor::insertInstsBefore(MachineInstr & MI,
                                             ArrayRef<MachineInstr *> Insts) {
  assert(!MI.isMetaInstruction() && "Cannot instrument meta instruction!");

  MachineFunction & MF = *MI.getMF();
  MachineBasicBlock & MBB = *MI.getParent();
  const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();

  unsigned distance;
  MachineInstr * IT = findIT(MI, distance);

  // Do insert new instructions before MI
  for (MachineInstr * Inst : Insts) {
    MBB.insert(MI, Inst);
  }

  // If MI is inside an IT block, we should make sure to cover all new
  // instructions with IT(s)
  if (IT != nullptr && distance != 0) {
    unsigned ITBlockSize = getITBlockSize(*IT);
    unsigned Mask = IT->getOperand(1).getImm() & 0xf;
    ARMCC::CondCodes firstCond = (ARMCC::CondCodes)IT->getOperand(0).getImm();
    std::deque<bool> DQMask = decodeITMask(Mask);
    bool sameAsFirstCond = DQMask[distance - 1];

    // Find the range of instructions that are supposed to be in IT block(s)
    MachineBasicBlock::iterator firstMI(IT->getNextNode()); // Inclusive
    MachineBasicBlock::iterator lastMI(MI);                 // Non-inclusive
    for (unsigned i = distance; i <= ITBlockSize; ) {
      ++lastMI;
      // Skip meta instructions if we have not reached the end
      if (i == ITBlockSize || !lastMI->isMetaInstruction()) {
        ++i;
      }
    }

    // Track new non-meta instructions in DQMask
    auto it = DQMask.begin();
    for (unsigned i = 0; i < distance - 1; ++i) {
      it++;
    }
    size_t NumRealInsts = Insts.size();
    for (MachineInstr * Inst : Insts) {
      if (Inst->isMetaInstruction()) {
        --NumRealInsts;
      }
    }
    DQMask.insert(it, NumRealInsts, sameAsFirstCond);

    // Insert ITs to cover instructions in [firstMI, lastMI)
    for (MachineBasicBlock::iterator i(firstMI); i != lastMI; ) {
      std::deque<bool> NewDQMask;
      MachineBasicBlock::iterator j(i);
      for (unsigned k = 0; k < 4 && j != lastMI; ++j) {
        if (j->isMetaInstruction()) {
          continue;
        }
        NewDQMask.push_back(DQMask.front());
        DQMask.pop_front();
        ++k;
      }
      bool flip = false;
      if (!NewDQMask[0]) {
        for (unsigned k = 0; k < NewDQMask.size(); ++k) {
          NewDQMask[k] = !NewDQMask[k];
        }
        flip = true;
      }
      BuildMI(MBB, i, IT->getDebugLoc(), TII->get(ARM::t2IT))
      .addImm(flip ? ARMCC::getOppositeCondition(firstCond) : firstCond)
      .addImm(encodeITMask(NewDQMask));
      i = j; // Update i here
    }

    // Remove the original IT
    IT->eraseFromParent();
  }
}

//
// Method: insertInstsAfter()
//
// Description:
//   This method inserts a group of instructions contained in an array after a
//   given instruction MI.  If MI is a predicated instruction within an IT
//   block, then the new instructions will have the same predicate as MI and
//   also end up in one or more IT blocks.
//
// Inputs:
//   MI    - A reference to an instruction after which to insert instructions.
//   Insts - A reference to an array containing the instructions.
//
void
ARMRandezvousInstrumentor::insertInstsAfter(MachineInstr & MI,
                                            ArrayRef<MachineInstr *> Insts) {
  assert(!MI.isMetaInstruction() && "Cannot instrument meta instruction!");

  MachineFunction & MF = *MI.getMF();
  MachineBasicBlock & MBB = *MI.getParent();
  const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();
  MachineBasicBlock::iterator NextMI(MI); ++NextMI;

  unsigned distance;
  MachineInstr * IT = findIT(MI, distance);

  // Do insert new instructions after MI
  for (MachineInstr * Inst : Insts) {
    MBB.insert(NextMI, Inst);
  }

  // If MI is inside an IT block, we should make sure to cover all new
  // instructions with IT(s)
  if (IT != nullptr && distance != 0) {
    unsigned ITBlockSize = getITBlockSize(*IT);
    unsigned Mask = IT->getOperand(1).getImm() & 0xf;
    ARMCC::CondCodes firstCond = (ARMCC::CondCodes)IT->getOperand(0).getImm();
    std::deque<bool> DQMask = decodeITMask(Mask);
    bool sameAsFirstCond = DQMask[distance - 1];

    // Find the range of instructions that are supposed to be in IT block(s)
    MachineBasicBlock::iterator firstMI(IT->getNextNode()); // Inclusive
    MachineBasicBlock::iterator lastMI(Insts.back());       // Non-inclusive
    for (unsigned i = distance; i <= ITBlockSize; ) {
      ++lastMI;
      // Skip meta instructions if we have not reached the end
      if (i == ITBlockSize || !lastMI->isMetaInstruction()) {
        ++i;
      }
    }

    // Track new non-meta instructions in DQMask
    auto it = DQMask.begin();
    for (unsigned i = 0; i <= distance - 1; ++i) {
      it++;
    }
    size_t NumRealInsts = Insts.size();
    for (MachineInstr * Inst : Insts) {
      if (Inst->isMetaInstruction()) {
        --NumRealInsts;
      }
    }
    DQMask.insert(it, NumRealInsts, sameAsFirstCond);

    // Insert ITs to cover instructions in [firstMI, lastMI)
    for (MachineBasicBlock::iterator i(firstMI); i != lastMI; ) {
      std::deque<bool> NewDQMask;
      MachineBasicBlock::iterator j(i);
      for (unsigned k = 0; k < 4 && j != lastMI; ++j) {
        if (j->isMetaInstruction()) {
          continue;
        }
        NewDQMask.push_back(DQMask.front());
        DQMask.pop_front();
        ++k;
      }
      bool flip = false;
      if (!NewDQMask[0]) {
        for (unsigned k = 0; k < NewDQMask.size(); ++k) {
          NewDQMask[k] = !NewDQMask[k];
        }
        flip = true;
      }
      BuildMI(MBB, i, IT->getDebugLoc(), TII->get(ARM::t2IT))
      .addImm(flip ? ARMCC::getOppositeCondition(firstCond) : firstCond)
      .addImm(encodeITMask(NewDQMask));
      i = j; // Update i here
    }

    // Remove the original IT
    IT->eraseFromParent();
  }
}

//
// Method: removeInst()
//
// Description:
//   This method removes a given instruction MI from machine IR.  If MI is a
//   predicated instruction within an IT block, then its corresponding IT
//   instruction will be updated or removed as well.  Note that MI cannot be an
//   IT instruction itself.
//
// Input:
//   MI - A reference to the instruction to remove.
//
void
ARMRandezvousInstrumentor::removeInst(MachineInstr & MI) {
  assert(!MI.isMetaInstruction() && "Cannot instrument meta instruction!");

  unsigned distance;
  MachineInstr * IT = findIT(MI, distance);

  // If MI was inside an IT block, we should make sure to update/remove the IT
  // instruction
  if (IT != nullptr) {
    assert(distance != 0 && "Cannot remove an IT instruction directly!");

    unsigned Mask = IT->getOperand(1).getImm() & 0xf;
    ARMCC::CondCodes firstCond = (ARMCC::CondCodes)IT->getOperand(0).getImm();
    std::deque<bool> DQMask = decodeITMask(Mask);

    // Remove MI's entry from DQMask
    auto it = DQMask.begin();
    for (unsigned i = 0; i < distance - 1; ++i) {
      it++;
    }
    DQMask.erase(it);

    // Remove IT as well if MI was the only instruction in the IT block
    if (DQMask.empty()) {
      IT->eraseFromParent();
    } else {
      // If MI was the first instruction in the IT block, removing MI might
      // change the first condition, in which case we need to flip it
      if (!DQMask[0]) {
        for (unsigned i = 0; i < DQMask.size(); ++i) {
          DQMask[i] = !DQMask[i];
        }
        IT->getOperand(0).setImm(ARMCC::getOppositeCondition(firstCond));
      }
      // Update the IT mask
      IT->getOperand(1).setImm(encodeITMask(DQMask));
    }
  }

  // Now do remove MI
  MI.eraseFromParent();
}

//
// Method: splitBasicBlockBefore()
//
// Description:
//   This method splits a basic block into two basic blocks at the point right
//   before a given instruction MI.  If MI is the first instruction in the
//   basic block, an empty basic block will be created.  If MI is a predicated
//   instruction within an IT block, then a new IT instruction will be inserted
//   at the beginning of the new basic block to cover MI and any succeeding
//   instructions that were in the original IT block.  In that case, the old IT
//   instruction will also be updated accordingly.
//
// Input:
//   MI - A reference to an instruction before which to split the basic block.
//
// Return value:
//   A pointer to the new basic block that contains MI.
//
MachineBasicBlock *
ARMRandezvousInstrumentor::splitBasicBlockBefore(MachineInstr & MI) {
  assert(!MI.isMetaInstruction() && "Cannot instrument meta instruction!");

  unsigned distance;
  MachineInstr * IT = findIT(MI, distance);

  MachineFunction & MF = *MI.getMF();
  MachineBasicBlock & MBB = *MI.getParent();
  const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();

  // Create a new basic block
  MachineBasicBlock & NewMBB = *MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MF.push_back(&NewMBB);
  NewMBB.moveAfter(&MBB);

  // Move instructions from the old basic block to the new basic block
  NewMBB.splice(NewMBB.end(), &MBB, &MI, MBB.end());

  // Move successors from the old basic block to the new basic block
  NewMBB.transferSuccessors(&MBB);
  MBB.addSuccessor(&NewMBB);

  // If MI was inside an IT block (but not the IT instruction itself), we
  // should make sure to update/remove the IT instruction and insert a new IT
  // in the new basic block
  if (IT != nullptr && distance > 0) {
    // Shortcut for just moving the IT over
    if (distance == 1) {
      NewMBB.splice(&MI, &MBB, IT);
    } else {
      unsigned Mask = IT->getOperand(1).getImm() & 0xf;
      ARMCC::CondCodes firstCond = (ARMCC::CondCodes)IT->getOperand(0).getImm();
      std::deque<bool> DQMask = decodeITMask(Mask);

      // Create a new DQMask and adjust both
      auto it = DQMask.begin();
      for (unsigned i = 0; i < distance - 1; ++i) {
        it++;
      }
      std::deque<bool> NewDQMask(it, DQMask.end());
      DQMask.erase(it, DQMask.end());

      // Update the old IT mask
      IT->getOperand(1).setImm(encodeITMask(DQMask));

      // Create a new IT
      bool flip = false;
      if (!NewDQMask[0]) {
        for (unsigned i = 0; i < NewDQMask.size(); ++i) {
          NewDQMask[i] = !NewDQMask[i];
        }
        flip = true;
      }
      BuildMI(NewMBB, MI, IT->getDebugLoc(), TII->get(ARM::t2IT))
      .addImm(flip ? ARMCC::getOppositeCondition(firstCond) : firstCond)
      .addImm(encodeITMask(NewDQMask));
    }
  }

  return &NewMBB;
}

//
// Method: splitBasicBlockAfter()
//
// Description:
//   This method splits a basic block into two basic blocks at the point right
//   after a given instruction MI.  If MI is the last instruction in the basic
//   block, an empty basic block will be created.  If MI is a predicated
//   instruction within an IT block and is not the last one, then a new IT
//   instruction will be inserted at the beginning of the new basic block to
//   cover MI's succeeding instructions that were in the original IT block.  In
//   that case, the old IT instruction will also be updated accordingly.
//
// Input:
//   MI - A reference to an instruction after which to split the basic block.
//
// Return value:
//   A pointer to the new basic block that contains MI's next instruction if
//   the split was done, nullptr otherwise.
//
MachineBasicBlock *
ARMRandezvousInstrumentor::splitBasicBlockAfter(MachineInstr & MI) {
  assert(!MI.isMetaInstruction() && "Cannot instrument meta instruction!");

  unsigned distance;
  MachineInstr * IT = findIT(MI, distance);

  MachineFunction & MF = *MI.getMF();
  MachineBasicBlock & MBB = *MI.getParent();
  const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();

  // Create a new basic block
  MachineBasicBlock & NewMBB = *MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MF.push_back(&NewMBB);
  NewMBB.moveAfter(&MBB);

  // Move instructions from the old basic block to the new basic block
  MachineBasicBlock::iterator next = ++MI.getIterator();
  NewMBB.splice(NewMBB.end(), &MBB, next, MBB.end());

  // Move successors from the old basic block to the new basic block
  NewMBB.transferSuccessors(&MBB);
  MBB.addSuccessor(&NewMBB);

  // If MI was inside an IT block and is not the last one, we should make sure
  // to update/remove the IT instruction and insert a new IT in the new basic
  // block
  if (IT != nullptr && distance < 4) {
    // Shortcut for just moving the IT over
    if (distance == 0) {
      NewMBB.splice(NewMBB.begin(), &MBB, IT);
    } else {
      unsigned Mask = IT->getOperand(1).getImm() & 0xf;
      ARMCC::CondCodes firstCond = (ARMCC::CondCodes)IT->getOperand(0).getImm();
      std::deque<bool> DQMask = decodeITMask(Mask);

      // Create a new DQMask and adjust both
      auto it = DQMask.begin();
      for (unsigned i = 0; i < distance; ++i) {
        it++;
      }
      std::deque<bool> NewDQMask(it, DQMask.end());
      DQMask.erase(it, DQMask.end());

      // Update the old IT mask
      IT->getOperand(1).setImm(encodeITMask(DQMask));

      // Create a new IT
      bool flip = false;
      if (!NewDQMask[0]) {
        for (unsigned i = 0; i < NewDQMask.size(); ++i) {
          NewDQMask[i] = !NewDQMask[i];
        }
        flip = true;
      }
      BuildMI(NewMBB, NewMBB.begin(), IT->getDebugLoc(), TII->get(ARM::t2IT))
      .addImm(flip ? ARMCC::getOppositeCondition(firstCond) : firstCond)
      .addImm(encodeITMask(NewDQMask));
    }
  }

  return &NewMBB;
}

//
// Method: decodeITMask()
//
// Description:
//   This method decodes an IT mask in LLVM's representation and puts a list of
//   boolean values in a deque to return.  The boolean values represent whether
//   their corresponding instructions in an IT block have the same predicate as
//   the first one (which indicates that the first boolean value is always
//   true).
//
// Input:
//   Mask - The IT mask in LLVM's representation (immediate value of the second
//          operand of a t2IT instruction).
//
// Return value:
//   A deque of boolean values (see the above description).
//
std::deque<bool>
ARMRandezvousInstrumentor::decodeITMask(unsigned Mask) {
  Mask &= 0xf;
  assert(Mask != 0 && "Invalid IT mask!");

  std::deque<bool> DQMask { true };
  unsigned size = 4;
  for (unsigned i = 0x1; i < 0x10; i <<= 1) {
    if (Mask & i) {
      break;
    }
    --size;
  }
  for (unsigned i = 3; i > 4 - size; --i) {
    DQMask.push_back((Mask & (1 << i)) == 0);
  }

  return DQMask;
}

//
// Method: encodeITMask()
//
// Description:
//   This method takes an IT mask in the form of a list of boolean values and
//   encodes it into LLVM's representation.  The boolean values represent
//   whether their corresponding instructions in an IT block have the same
//   predicate as the first one (which requires that the first boolean value
//   be always true).
//
// Input:
//   DQMask - An IT mask in the form of a list of boolean values.
//
// Return value:
//   The IT mask in LLVM's representation (immediate value of the second
//   operand of a t2IT instruction).
//
unsigned
ARMRandezvousInstrumentor::encodeITMask(std::deque<bool> DQMask) {
  assert(!DQMask.empty() && "Invalid deque representation of an IT mask!");
  assert(DQMask.size() <= 4 && "Invalid deque representation of an IT mask!");
  assert(DQMask[0] && "Invalid deque representation of an IT mask!");

  unsigned Mask = 0;
  for (unsigned i = 1; i < DQMask.size(); ++i) {
    Mask |= DQMask[i] ? 0 : 1;
    Mask <<= 1;
  }
  Mask |= 1;
  Mask <<= (4 - DQMask.size());

  return Mask;
}

//
// Method: findFreeRegistersBefore()
//
// Description:
//   This method computes the liveness of ARM core registers before a given
//   instruction MI and returns a list of free core registers that can be
//   used for instrumentation purposes.
//
// Inputs:
//   MI    - A reference to the instruction before which to find free
//           registers.
//   Thumb - Whether we are looking for Thumb registers (low registers, i,e,,
//           R0 -- R7) or ARM registers (both low and high registers, i.e.,
//           R0 -- R12 and LR).
//
// Return value:
//   A vector of free registers (might be empty, if none is found).
//
std::vector<Register>
ARMRandezvousInstrumentor::findFreeRegistersBefore(const MachineInstr & MI,
                                                   bool Thumb) {
  assert(!MI.isMetaInstruction() && "Cannot instrument meta instruction!");

  unsigned distance;
  const MachineInstr * IT = findIT(MI, distance);

  Register PredReg;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);

  const MachineFunction & MF = *MI.getMF();
  const MachineBasicBlock & MBB = *MI.getParent();
  const MachineRegisterInfo & MRI = MF.getRegInfo();
  const TargetRegisterInfo * TRI = MF.getSubtarget().getRegisterInfo();
  LivePhysRegs UsedRegs(*TRI);

  // First add live-out registers of MBB; these registers are considered live
  // at the end of MBB
  UsedRegs.addLiveOuts(MBB);

  // Then move backward step by step to compute live registers before MI
  MachineBasicBlock::const_iterator MBBI(MI);
  MachineBasicBlock::const_iterator I = MBB.end();
  while (I != MBBI) {
    unsigned distance2;
    const MachineInstr * IT2 = findIT(*--I, distance2);
    Register PredReg2;
    ARMCC::CondCodes Pred2 = getInstrPredicate(*I, PredReg2);

    if (IT2 != nullptr && IT == IT2) {
      // Skip instructions in the same IT block but with a different predicate
      if (Pred != Pred2) {
        continue;
      }

      // A return in the same IT block with the same predicate can reset live
      // registers to the callee-saved registers
      if (I->isReturn()) {
        UsedRegs.init(*TRI);
        for (auto CSR = MRI.getCalleeSavedRegs(); CSR && *CSR; ++CSR) {
          UsedRegs.addReg(*CSR);
        }
      }
    }

    UsedRegs.stepBackward(*I);
  }

  // Now add registers that are neither reserved nor live to a free list
  const auto LoGPRs = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3, ARM::R4, ARM::R5, ARM::R6, ARM::R7,
  };
  const auto HiGPRs = {
    ARM::R8, ARM::R9, ARM::R10, ARM::R11, ARM::R12, ARM::LR,
  };
  std::vector<Register> FreeRegs;
  for (Register Reg : LoGPRs) {
    if (!MRI.isReserved(Reg) && !UsedRegs.contains(Reg)) {
      FreeRegs.push_back(Reg);
    }
  }
  if (!Thumb) {
    for (Register Reg : HiGPRs) {
      if (!MRI.isReserved(Reg) && !UsedRegs.contains(Reg)) {
        FreeRegs.push_back(Reg);
      }
    }
  }

  return FreeRegs;
}

//
// Method: findFreeRegistersAfter()
//
// Description:
//   This method computes the liveness of ARM core registers after a given
//   instruction MI and returns a list of free core registers that can be
//   used for instrumentation purposes.
//
// Inputs:
//   MI    - A reference to the instruction after which to find free
//           registers.
//   Thumb - Whether we are looking for Thumb registers (low registers, i,e,,
//           R0 -- R7) or ARM registers (both low and high registers, i.e.,
//           R0 -- R12 and LR).
//
// Return value:
//   A vector of free registers (might be empty, if none is found).
//
std::vector<Register>
ARMRandezvousInstrumentor::findFreeRegistersAfter(const MachineInstr & MI,
                                                  bool Thumb) {
  assert(!MI.isMetaInstruction() && "Cannot instrument meta instruction!");

  unsigned distance;
  const MachineInstr * IT = findIT(MI, distance);

  Register PredReg;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);

  const MachineFunction & MF = *MI.getMF();
  const MachineBasicBlock & MBB = *MI.getParent();
  const MachineRegisterInfo & MRI = MF.getRegInfo();
  const TargetRegisterInfo * TRI = MF.getSubtarget().getRegisterInfo();
  LivePhysRegs UsedRegs(*TRI);

  // First add live-out registers of MBB; these registers are considered live
  // at the end of MBB
  UsedRegs.addLiveOuts(MBB);

  // If there is a return, add registers used by the return as well; here the
  // rationale is that, if MI is the return, MI will not be stepped over and
  // therefore the (potentially live) registers used in MI would not be counted
  MachineBasicBlock::const_iterator Terminator = MBB.getLastNonDebugInstr();
  if (Terminator != MBB.end() && Terminator->isReturn()) {
    UsedRegs.addUses(*Terminator);
  }

  // Then move backward step by step to compute live registers after MI
  MachineBasicBlock::const_iterator MBBI(MI);
  MachineBasicBlock::const_iterator I = MBB.end();
  while (I != MBBI) {
    unsigned distance2;
    const MachineInstr * IT2 = findIT(*--I, distance2);
    Register PredReg2;
    ARMCC::CondCodes Pred2 = getInstrPredicate(*I, PredReg2);

    if (IT2 != nullptr && IT == IT2) {
      // Skip instructions in the same IT block but with a different predicate
      if (Pred != Pred2) {
        continue;
      }

      // A return in the same IT block with the same predicate can reset live
      // registers to the callee-saved registers
      if (I->isReturn()) {
        UsedRegs.init(*TRI);
        for (auto CSR = MRI.getCalleeSavedRegs(); CSR && *CSR; ++CSR) {
          UsedRegs.addReg(*CSR);
        }

        // Add registers used by the return; if MI is the return, MI will not
        // be stepped over and therefore the (potentially live) registers used
        // in MI would not be counted
        UsedRegs.addUses(*I);
      }
    }

    if (I != MBBI) {
      UsedRegs.stepBackward(*I);
    }
  }

  // Now add registers that are neither reserved nor live to a free list
  const auto LoGPRs = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3, ARM::R4, ARM::R5, ARM::R6, ARM::R7,
  };
  const auto HiGPRs = {
    ARM::R8, ARM::R9, ARM::R10, ARM::R11, ARM::R12, ARM::LR,
  };
  std::vector<Register> FreeRegs;
  for (Register Reg : LoGPRs) {
    if (!MRI.isReserved(Reg) && !UsedRegs.contains(Reg)) {
      FreeRegs.push_back(Reg);
    }
  }
  if (!Thumb) {
    for (Register Reg : HiGPRs) {
      if (!MRI.isReserved(Reg) && !UsedRegs.contains(Reg)) {
        FreeRegs.push_back(Reg);
      }
    }
  }

  return FreeRegs;
}