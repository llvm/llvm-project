//===-- SuperHBasicBlockInfo.cpp - Basic Block Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuperHBasicBlockInfo.h"

#define DEBUG_TYPE "sh-bb-utils"

using namespace llvm;

namespace llvm {

void SuperHBasicBlockUtils::computeBlockSize(MachineBasicBlock *MBB) {
  LLVM_DEBUG(dbgs() << "computeBlockSize: " << MBB->getName() << "\n");
  BasicBlockInfo &BBI = BBInfo[MBB->getNumber()];
  BBI.Size = 0;
  BBI.Unalign = 0;
  BBI.PostAlign = Align(1);

  for (MachineInstr &I : *MBB) {
    BBI.Size += TII->getInstSizeInBytes(I);
    // For inline asm, getInstSizeInBytes returns a conservative estimate.
    // The actual size may be smaller, but still a multiple of the instr size.
    if (I.isInlineAsm())
      BBI.Unalign = 2;
  }
}

/// getOffsetOf - Return the current offset of the specified machine instruction
/// from the start of the function.  This offset changes as stuff is moved
/// around inside the function.
unsigned SuperHBasicBlockUtils::getOffsetOf(MachineInstr *MI) const {
  const MachineBasicBlock *MBB = MI->getParent();

  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = BBInfo[MBB->getNumber()].Offset;

  // Sum instructions before MI in MBB.
  for (MachineBasicBlock::const_iterator I = MBB->begin(); &*I != MI; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    Offset += TII->getInstSizeInBytes(*I);
  }
  return Offset;
}

/// isBBInRange - Returns true if the distance between specific MI and
/// specific BB can fit in MI's displacement field.
bool SuperHBasicBlockUtils::isBBInRange(MachineInstr *MI,
                                     MachineBasicBlock *DestBB,
                                     unsigned MaxDisp) const {
  unsigned PCAdj      = 4;
  unsigned BrOffset   = getOffsetOf(MI) + PCAdj;
  unsigned DestOffset = BBInfo[DestBB->getNumber()].Offset;

  LLVM_DEBUG(dbgs() << "Branch of destination " << printMBBReference(*DestBB)
                    << " from " << printMBBReference(*MI->getParent())
                    << " max delta=" << MaxDisp << " from " << getOffsetOf(MI)
                    << " to " << DestOffset << " offset "
                    << int(DestOffset - BrOffset) << "\t" << *MI);

  if (BrOffset <= DestOffset) {
    // Branch before the Dest.
    if (DestOffset-BrOffset <= MaxDisp)
      return true;
  } else {
    if (BrOffset-DestOffset <= MaxDisp)
      return true;
  }
  return false;
}

void SuperHBasicBlockUtils::adjustBBOffsetsAfter(MachineBasicBlock *BB) {
  assert(BB->getParent() == &MF &&
         "Basic block is not a child of the current function.\n");

  unsigned BBNum = BB->getNumber();
  LLVM_DEBUG(dbgs() << "Adjust block:\n"
             << " - name: " << BB->getName() << "\n"
             << " - number: " << BB->getNumber() << "\n"
             << " - function: " << MF.getName() << "\n"
             << "   - blocks: " << MF.getNumBlockIDs() << "\n");

  for(unsigned i = BBNum + 1, e = MF.getNumBlockIDs(); i < e; ++i) {
    // Get the offset and known bits at the end of the layout predecessor.
    // Include the alignment of the current block.
    const Align Align = MF.getBlockNumbered(i)->getAlignment();
    const unsigned Offset = BBInfo[i - 1].postOffset(Align);
    const unsigned KnownBits = BBInfo[i - 1].postKnownBits(Align);

    // This is where block i begins.  Stop if the offset is already correct,
    // and we have updated 2 blocks.  This is the maximum number of blocks
    // changed before calling this function.
    if (i > BBNum + 2 &&
        BBInfo[i].Offset == Offset &&
        BBInfo[i].KnownBits == KnownBits)
      break;

    BBInfo[i].Offset = Offset;
    BBInfo[i].KnownBits = KnownBits;
  }
}


} // namespace llvm