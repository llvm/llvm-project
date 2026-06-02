//===-- EZHBasicBlockInfo.h - Basic Block Information -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_EZH_EZHBASICBLOCKINFO_H
#define LLVM_LIB_TARGET_EZH_EZHBASICBLOCKINFO_H

#include "EZHInstrInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

namespace llvm {

struct EZHBasicBlockInfo;
using EZHBBInfoVector = SmallVectorImpl<EZHBasicBlockInfo>;

/// Basic Block Information for EZH target, tracking offset, size, and
/// alignment.
struct EZHBasicBlockInfo {
  unsigned Offset = 0;
  unsigned Size = 0;
  Align PostAlign;

  EZHBasicBlockInfo() = default;

  unsigned postOffset(Align Alignment = Align(1)) const {
    unsigned PO = Offset + Size;
    const Align PA = std::max(PostAlign, Alignment);
    return alignTo(PO, PA);
  }

  unsigned postKnownBits(Align Alignment = Align(1)) const {
    return Log2(std::max(PostAlign, Alignment));
  }
};

/// Utilities for computing and adjusting basic block offsets and sizes.
class EZHBasicBlockUtils {
private:
  MachineFunction &MF;
  const EZHInstrInfo *TII = nullptr;
  SmallVector<EZHBasicBlockInfo, 8> BBInfo;

public:
  EZHBasicBlockUtils(MachineFunction &MF) : MF(MF) {
    TII = static_cast<const EZHInstrInfo *>(MF.getSubtarget().getInstrInfo());
  }

  void computeAllBlockSizes() {
    BBInfo.resize(MF.getNumBlockIDs());
    for (MachineBasicBlock &MBB : MF)
      computeBlockSize(&MBB);
  }

  void computeBlockSize(MachineBasicBlock *MBB);

  unsigned getOffsetOf(MachineInstr *MI) const;

  unsigned getOffsetOf(MachineBasicBlock *MBB) const {
    return BBInfo[MBB->getNumber()].Offset;
  }

  void adjustBBOffsetsAfter(MachineBasicBlock *MBB);

  void adjustBBSize(MachineBasicBlock *MBB, int Size) {
    BBInfo[MBB->getNumber()].Size += Size;
  }

  bool isBBInRange(MachineInstr *MI, MachineBasicBlock *DestBB,
                   unsigned MaxDisp) const;

  void insert(unsigned BBNum, EZHBasicBlockInfo BBI) {
    BBInfo.insert(BBInfo.begin() + BBNum, BBI);
  }

  void clear() { BBInfo.clear(); }

  EZHBBInfoVector &getBBInfo() { return BBInfo; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_EZH_EZHBASICBLOCKINFO_H
