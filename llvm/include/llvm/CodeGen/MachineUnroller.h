//===- MachineUnroller.h - Machine loop unrolling utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines loop unrolling utilities used at the machine instruction
// (MI) level.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEUNROLLER_H
#define LLVM_CODEGEN_MACHINEUNROLLER_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

namespace llvm {

// This is a utility for unrolling loops at MI level.
// It only unroll loops with the run-time trip count and
// with a single basic block.
//
// After unrolling, the loop structure will be the following:
//
// Original LoopPreheader
// Unrolled LoopPreheader
//   Unrolled Loop
// Unrolled LoopExit
// Remainder LoopPreheader
//   Remainder Loop
// Remainder LoopExit
// Original LoopExit

struct MachineUnrollerContext {
  MachineFunction *MF = nullptr;
  MachineLoopInfo *MLI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineUnrollerContext() {}
  MachineUnrollerContext(MachineFunction *mf, MachineLoopInfo *mli,
                         const TargetInstrInfo *tii)
      : MF(mf), MLI(mli), TII(tii) {}
};

class MachineUnroller {
protected:
  MachineFunction *MF = nullptr;
  MachineLoopInfo *MLI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineLoop *L;
  MachineBasicBlock *OrigHeader;
  MachineBasicBlock *OrigPreheader;
  MachineBasicBlock *ULPreheader;
  MachineBasicBlock *ULHeader;
  MachineBasicBlock *ULExit;
  MachineBasicBlock *RLPreheader;
  MachineBasicBlock *RLHeader;
  MachineBasicBlock *RLExit;
  MachineBasicBlock *OrigLoopExit;
  unsigned UnrollFactor;
  unsigned LC;
  SmallVector<MachineBasicBlock *, 4> LoopBBs;
  SmallVector<unsigned, 4> ExitBBLiveIns;

  typedef SmallDenseMap<MachineBasicBlock *, DenseMap<unsigned, unsigned>, 4>
      ValueMapTy;
  ValueMapTy VRMap;
  DenseMap<unsigned, unsigned> ULPhiVRMap;
  void createUnrolledLoopStruct();
  void updateInstruction(MachineInstr *NewMI, bool FirstIter,
                         ValueMapTy &OldVRMap);
  void generateUnrolledLoop();
  unsigned getMappedRegORCreate(unsigned Reg, MachineBasicBlock *BB);
  void generateNewPhis(MachineBasicBlock *BB, MachineBasicBlock *BB1,
                       MachineBasicBlock *BB2);
  void generatePhisForRLExit();
  void generatePhisForULExit();
  void getExitBBLiveIns();
  void addBBIntoVRMap(MachineBasicBlock *BB);
  void fixBranchesAndLoopCount(unsigned ULCount, unsigned RLCount);
  unsigned getLatestInstance(unsigned reg, MachineBasicBlock *BB,
                             ValueMapTy &VRMap);
  void init(MachineLoop *loop, unsigned unrollFactor);
  bool canUnroll();
  void preprocessPhiNodes(MachineBasicBlock &B);

public:
  MachineUnroller(MachineUnrollerContext *C)
      : MF(C->MF), MLI(C->MLI), TII(C->TII) {
    MRI = &MF->getRegInfo();
  }

  virtual ~MachineUnroller() = default;

  bool unroll(MachineLoop *loop, unsigned unrollFactor);

  virtual unsigned getLoopCount(MachineBasicBlock &LoopBB) const = 0;

  /// Add instruction to compute trip count for the unrolled loop.
  virtual unsigned addUnrolledLoopCountMI(MachineBasicBlock &MBB, unsigned LC,
                                          unsigned UnrollFactor) const = 0;

  /// Add instruction to compute remainder trip count for the unrolled loop.
  virtual unsigned addRemLoopCountMI(MachineBasicBlock &MBB, unsigned LC,
                                     unsigned UnrollFactor) const = 0;

  virtual void changeLoopCount(MachineBasicBlock &BB,
                               MachineBasicBlock &Preheader,
                               MachineBasicBlock &Header,
                               MachineBasicBlock &LoopBB, unsigned LC,
                               SmallVectorImpl<MachineOperand> &Cond) const = 0;

  bool computeDelta(MachineInstr &MI, unsigned &Delta) const;
  void updateMemOperands(MachineInstr *NewMI, MachineInstr *OldMI,
                         unsigned iter) const;
  virtual void optimize(MachineBasicBlock &BB) const {};
};
} // namespace llvm
#endif
