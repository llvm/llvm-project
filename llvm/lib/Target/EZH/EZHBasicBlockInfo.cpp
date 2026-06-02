//===--- EZHBasicBlockInfo.cpp - Utilities for block sizes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZHBasicBlockInfo.h"
#include "EZH.h"
#include "EZHInstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ezh-bb-utils"

using namespace llvm;

void llvm::EZHBasicBlockUtils::computeBlockSize(MachineBasicBlock *MBB) {
  EZHBasicBlockInfo &BBI = BBInfo[MBB->getNumber()];
  BBI.Size = 0;
  BBI.PostAlign = Align(1);

  for (MachineInstr &I : *MBB) {
    BBI.Size += TII->getInstSizeInBytes(I);
  }
}

unsigned llvm::EZHBasicBlockUtils::getOffsetOf(MachineInstr *MI) const {
  const MachineBasicBlock *MBB = MI->getParent();
  unsigned Offset = BBInfo[MBB->getNumber()].Offset;

  bool Found = false;
  for (const MachineInstr &I : *MBB) {
    if (&I == MI) {
      Found = true;
      break;
    }
    Offset += TII->getInstSizeInBytes(I);
  }
  assert(Found && "Didn't find MI in its own basic block?");
  return Offset;
}

bool llvm::EZHBasicBlockUtils::isBBInRange(MachineInstr *MI,
                                           MachineBasicBlock *DestBB,
                                           unsigned MaxDisp) const {
  unsigned PCAdj = 4;
  unsigned BrOffset = getOffsetOf(MI) + PCAdj;
  unsigned DestOffset = BBInfo[DestBB->getNumber()].Offset;

  LLVM_DEBUG(dbgs() << "Branch of destination " << printMBBReference(*DestBB)
                    << " from " << printMBBReference(*MI->getParent())
                    << " max delta=" << MaxDisp << " from " << getOffsetOf(MI)
                    << " to " << DestOffset << " offset "
                    << int(DestOffset - BrOffset) << "\t" << *MI);

  if (BrOffset <= DestOffset)
    return (DestOffset - BrOffset) <= MaxDisp;
  return (BrOffset - DestOffset) <= MaxDisp;
}

void llvm::EZHBasicBlockUtils::adjustBBOffsetsAfter(MachineBasicBlock *BB) {
  assert(BB->getParent() == &MF &&
         "Basic block is not a child of the current function.\n");

  unsigned BBNum = BB->getNumber();
  LLVM_DEBUG(dbgs() << "Adjust block:\n"
                    << " - name: " << BB->getName() << "\n"
                    << " - number: " << BB->getNumber() << "\n"
                    << " - function: " << MF.getName() << "\n"
                    << "   - blocks: " << MF.getNumBlockIDs() << "\n");

  for (unsigned i = BBNum + 1, e = MF.getNumBlockIDs(); i < e; ++i) {
    const Align Align = MF.getBlockNumbered(i)->getAlignment();
    const unsigned Offset = BBInfo[i - 1].postOffset(Align);

    if (i > BBNum + 2 && BBInfo[i].Offset == Offset)
      break;

    BBInfo[i].Offset = Offset;
  }
}
