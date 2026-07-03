//===- MemSafetyAnalysis.cpp - Memory access safety for a loop ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Originally developed by Advanced Micro Devices, Inc. (2015).
//
// MemSafetyAnalysis identifies memory accesses that are guaranteed to execute
// on every dynamic iteration of the loop body.
//
// Guaranteed-access classification:
//   Case#1: P is accessed in a top-level block (one that dominates the latch)
//     loop {
//       if (C1)  *P;
//       else { if (C2) *P; else ... }
//       *P;                  // <-- top-level
//     }
//
//   Case#2: P is accessed on every successor path of a top-level subtree
//     loop {
//       if (C1) *P;
//       else { if (C2) *P; else *P; }   // both branches touch *P
//     }
//
//   Case#3: P is not accessed on some path -- not guaranteed
//     loop {
//       if (C1) *P;
//       else { if (C2) *P; else { if (C3) *P; else /* no access */ } }
//     }
//
// See MemSafetyAnalysis.h for the public API.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemSafetyAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

using namespace llvm;

#define DEBUG_TYPE "mem-access-safety"

//===----------------------------------------------------------------------===//
// BlockMemInfo
//===----------------------------------------------------------------------===//

void BlockMemInfo::addMemoryAccess(const SCEV *S, MemProperty AccessTy) {
  BlockMemAccess[S] = AccessTy;
}

void BlockMemInfo::copyBlockMemInfo(BlockMemInfo *BMI) {
  for (auto &Itr : BMI->BlockMemAccess)
    BlockMemAccess[Itr.first] = Itr.second;
}

//===----------------------------------------------------------------------===//
// MemSafetyAnalysis
//===----------------------------------------------------------------------===//

MemSafetyAnalysis::MemSafetyAnalysis(Loop *L, LoopInfo *LI, ScalarEvolution *SE,
                                     DominatorTree *DT,
                                     const TargetLibraryInfo *TLI,
                                     const TargetTransformInfo *TTI)
    : L(L), LI(LI), SE(SE), DT(DT), TLI(TLI), TTI(TTI) {
  if (isLegalLoopStructure())
    analyzeMemoryAccessesInLoop();
}

MemSafetyAnalysis::~MemSafetyAnalysis() {
  // BlockMemAccessMap is cleared as part of analyzeMemoryAccessesInLoop's
  // tail; no-op here if it already ran.
  clearLocalMemory();
}

bool MemSafetyAnalysis::isLegalLoopStructure() {
  if (!L->getLoopLatch())
    return false;
  if (!L->getSubLoops().empty())
    return false;
  if (L->getNumBackEdges() != 1)
    return false;
  if (!L->getExitingBlock())
    return false;
  if (L->getExitingBlock() != L->getLoopLatch())
    return false;
  return true;
}

void MemSafetyAnalysis::clearLocalMemory() {
  for (auto &Itr : BlockMemAccessMap)
    delete Itr.second;
  BlockMemAccessMap.clear();
}

void MemSafetyAnalysis::addGuaranteedMemoryAccess(const SCEV *S) {
  SafeMemoryAccesses[S] = MemProperty::Safe;
}

bool MemSafetyAnalysis::isGuaranteedMemoryAccess(const SCEV *S) const {
  auto Itr = SafeMemoryAccesses.find(S);
  if (Itr == SafeMemoryAccesses.end() || Itr->second != MemProperty::Safe)
    return false;
  return true;
}

bool MemSafetyAnalysis::blockGuaranteedToExecute(BasicBlock *BB) {
  return DT->dominates(BB, L->getLoopLatch());
}

BlockMemInfo *MemSafetyAnalysis::getBlockMemInfo(BasicBlock *BB) {
  auto Itr = BlockMemAccessMap.find(BB);
  if (Itr != BlockMemAccessMap.end())
    return Itr->second;
  auto *BMI = new BlockMemInfo(BB, blockGuaranteedToExecute(BB));
  BlockMemAccessMap[BB] = BMI;
  return BMI;
}

bool MemSafetyAnalysis::processBlock(BasicBlock *BB) {
  Instruction *Term = BB->getTerminator();
  BlockMemInfo *BMI = getBlockMemInfo(BB);

  if (auto *CB = dyn_cast<CondBrInst>(Term)) {
    BlockMemInfo *Succ0BMI = getBlockMemInfo(CB->getSuccessor(0));
    BlockMemInfo *Succ1BMI = getBlockMemInfo(CB->getSuccessor(1));
    // Case#2: an access registered as LocalSafe on both successors gets
    // promoted to Safe on the current block, and to a loop-level guaranteed
    // access if the current block is itself guaranteed.
    for (auto &Itr0 : Succ0BMI->BlockMemAccess) {
      for (auto &Itr1 : Succ1BMI->BlockMemAccess) {
        if (Itr0.first != Itr1.first)
          continue;
        if (Itr0.second != MemProperty::LocalSafe ||
            Itr1.second != MemProperty::LocalSafe)
          continue;
        if (BMI->isBlockGuaranteedToExecute()) {
          BMI->addMemoryAccess(Itr0.first, MemProperty::Safe);
          addGuaranteedMemoryAccess(Itr0.first);
        } else {
          BMI->addMemoryAccess(Itr0.first, MemProperty::LocalSafe);
        }
      }
    }
  } else if (auto *UB = dyn_cast<UncondBrInst>(Term)) {
    BlockMemInfo *Succ0BMI = getBlockMemInfo(UB->getSuccessor(0));
    BMI->copyBlockMemInfo(Succ0BMI);
  } else {
    // Not a branch terminator (return, unreachable, switch, indirectbr,
    // resume, ...) -- bail out and let the analysis stay invalid.
    return false;
  }

  // Now register the block's own accesses.
  for (auto &I : *BB) {
    Value *Ptr = nullptr;
    if (auto *Ld = dyn_cast<LoadInst>(&I)) {
      if (!Ld->isSimple())
        continue;
      Ptr = Ld->getPointerOperand();
    } else if (auto *St = dyn_cast<StoreInst>(&I)) {
      if (!St->isSimple())
        continue;
      Ptr = St->getPointerOperand();
    } else {
      continue;
    }

    const SCEV *PtrSCEV = SE->getSCEV(Ptr);
    if (isGuaranteedMemoryAccess(PtrSCEV)) {
      BMI->addMemoryAccess(PtrSCEV, MemProperty::Safe);
      continue;
    }
    if (BMI->isBlockGuaranteedToExecute()) {
      addGuaranteedMemoryAccess(PtrSCEV);
      BMI->addMemoryAccess(PtrSCEV, MemProperty::Safe);
      continue;
    }
    BMI->addMemoryAccess(PtrSCEV, MemProperty::LocalSafe);
  }
  return true;
}

bool MemSafetyAnalysis::analyzeMemoryAccessesInLoop() {
  LoopBlocksRPO RPOT(L);
  RPOT.perform(LI);
  // Bottom-up traversal: process a block after its successors so that
  // Case#2 propagation has the successor info ready.
  std::stack<BasicBlock *> WorkList;
  for (BasicBlock *BB : RPOT)
    WorkList.push(BB);
  while (!WorkList.empty()) {
    if (!processBlock(WorkList.top())) {
      clearLocalMemory();
      return false;
    }
    WorkList.pop();
  }
  IsAnalysisValid = true;
  clearLocalMemory();
  return true;
}

void MemSafetyAnalysis::printAnalysis() const {
  LLVM_DEBUG(dbgs() << "MemSafetyAnalysis\n");
  for (auto *Block : L->getBlocks()) {
    for (auto &Inst : *Block) {
      Value *Ptr = nullptr;
      if (auto *Ld = dyn_cast<LoadInst>(&Inst)) {
        if (!Ld->isSimple())
          continue;
        Ptr = Ld->getPointerOperand();
      } else if (auto *St = dyn_cast<StoreInst>(&Inst)) {
        if (!St->isSimple())
          continue;
        Ptr = St->getPointerOperand();
      } else {
        continue;
      }
      const SCEV *PtrSCEV = SE->getSCEV(Ptr);
      if (isGuaranteedMemoryAccess(PtrSCEV)) {
        LLVM_DEBUG(dbgs() << "   [Guaranteed]: " << Inst
                          << "  SCEV: " << *PtrSCEV << "\n");
      }
    }
  }
}
