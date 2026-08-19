//===-- HexagonGlobalRegion.h - VLIW global scheduling infrastructure -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic infrastructure for global scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_GLOBAL_REGION_H
#define HEXAGON_GLOBAL_REGION_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

#include <map>
#include <memory>
#include <vector>

namespace llvm {
/// Class to track incremental liveness update.
class LivenessInfo {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  BitVector LiveIns;
  BitVector LiveOuts;

public:
  LivenessInfo(const TargetInstrInfo *TII, const TargetRegisterInfo *TRI,
               MachineBasicBlock *MBB);
  ~LivenessInfo() {}
  void parseOperands(MachineInstr *MI, BitVector &Gen, BitVector &Kill,
                     BitVector &Use);
  void parseOperandsWithReset(MachineInstr *MI, BitVector &Gen, BitVector &Kill,
                              BitVector &Use);
  void setUsed(BitVector &Set, unsigned Reg);
  // Update Liveness for BB.
  void UpdateLiveness(MachineBasicBlock *MBB);
  void dump();
};

/// Generic sequence of BBs. A trace or SB.
/// Maintains its own liveness info.
class BasicBlockRegion {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  // Sequence of BBs in a larger block.
  std::vector<MachineBasicBlock *> Elements;
  std::map<MachineBasicBlock *, std::unique_ptr<LivenessInfo>> LiveInfo;
  llvm::DenseMap<MachineBasicBlock *, unsigned> ElementIndex;

public:
  BasicBlockRegion(const TargetInstrInfo *TII, const TargetRegisterInfo *TRI,
                   MachineBasicBlock *MBB);
  ~BasicBlockRegion();

  void addBBtoRegion(MachineBasicBlock *MBB);

  MachineBasicBlock *getEntryBB() { return Elements.front(); }

  MachineBasicBlock *findMBB(MachineBasicBlock *MBB) {
    return ElementIndex.find(MBB) != ElementIndex.end() ? MBB : nullptr;
  }

  void RemoveBBFromRegion(MachineBasicBlock *MBB) {
    auto It = ElementIndex.find(MBB);
    if (It == ElementIndex.end())
      return;
    unsigned Index = It->second;
    Elements.erase(Elements.begin() + Index);
    ElementIndex.erase(It);
    LiveInfo.erase(MBB);
    for (unsigned I = Index, E = static_cast<unsigned>(Elements.size()); I != E;
         ++I)
      ElementIndex[Elements[I]] = I;
  }

  MachineBasicBlock *findNextMBB(MachineBasicBlock *MBB) {
    auto It = ElementIndex.find(MBB);
    if (It == ElementIndex.end())
      return nullptr;
    unsigned Next = It->second + 1;
    if (Next >= Elements.size())
      return nullptr;
    return Elements[Next];
  }

  unsigned size() { return static_cast<unsigned>(Elements.size()); }

  std::vector<MachineBasicBlock *>::iterator getRootMBB() {
    return Elements.begin();
  }

  std::vector<MachineBasicBlock *>::iterator getLastMBB() {
    return Elements.end();
  }

  LivenessInfo *getLivenessInfoForBB(MachineBasicBlock *MBB);
};
} // namespace llvm

#endif
