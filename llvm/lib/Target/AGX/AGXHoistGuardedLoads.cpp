//===-- AGXHoistGuardedLoads.cpp - Loads skipped for one index, hoisted -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// All-pairs loops skip their own element:
//
//   p = pos[i];                      // before the loop
//   for (j = ...; ...; j += n) {     // unrolled: j, j + n, j + 2n, ...
//     if (j == i) continue;
//     q = pos[j];  ...
//   }
//
// Each load waits in its own block, so the loads of the unrolled copies run
// one after the other (Geekbench's particle kernel: ~30% of its time). A load
// in a block that runs exactly when j != i, in a loop that runs that test every
// iteration, can run at the top of every iteration: when j == i its address
// is the one loaded before the loop, so it is as safe, and the loop writes no
// memory, so its value is the same. The address computation is copied into
// the loop header and the load moved there, where the loads of all copies
// issue together.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-hoist-guarded-loads"

static cl::opt<bool> AGXHoistGuardedLoadsOpt(
    "agx-hoist-skipped-loads", cl::init(true), cl::Hidden,
    cl::desc("Hoist loads skipped only for an index whose element was loaded "
             "before the loop into the loop header"));

namespace {

class AGXHoistGuardedLoads : public FunctionPass {
public:
  static char ID;
  AGXHoistGuardedLoads() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Hoist Guarded Loads"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesCFG();
  }

private:
  bool runOnLoop(Loop *L, DominatorTree &DT);
};

} // namespace

char AGXHoistGuardedLoads::ID = 0;

INITIALIZE_PASS_BEGIN(AGXHoistGuardedLoads, DEBUG_TYPE,
                      "AGX Hoist Guarded Loads", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(AGXHoistGuardedLoads, DEBUG_TYPE, "AGX Hoist Guarded Loads",
                    false, false)

FunctionPass *llvm::createAGXHoistGuardedLoadsPass() {
  return new AGXHoistGuardedLoads();
}

/// Whether A computes what B computes once X in A is replaced by C: the
/// address of the element j, A, against the one of i, B.
static bool sameWithXAsC(const Value *A, const Value *B, const Value *X,
                         const Value *C, unsigned Depth = 0) {
  if (A == X)
    return B == C;
  if (A == B)
    return true;
  if (Depth > 8)
    return false;
  auto *IA = dyn_cast<Instruction>(A), *IB = dyn_cast<Instruction>(B);
  if (!IA || !IB || IA->getOpcode() != IB->getOpcode() ||
      IA->getType() != IB->getType() ||
      IA->getNumOperands() != IB->getNumOperands() || isa<PHINode>(IA) ||
      IA->mayReadOrWriteMemory())
    return false;
  if (auto *GA = dyn_cast<GetElementPtrInst>(IA))
    if (GA->getSourceElementType() !=
        cast<GetElementPtrInst>(IB)->getSourceElementType())
      return false;
  if (auto *CA = dyn_cast<CmpInst>(IA))
    if (CA->getPredicate() != cast<CmpInst>(IB)->getPredicate())
      return false;
  for (unsigned I = 0; I < IA->getNumOperands(); ++I)
    if (!sameWithXAsC(IA->getOperand(I), IB->getOperand(I), X, C, Depth + 1))
      return false;
  return true;
}

/// The instructions of V's expression inside L (outside the header), in
/// definition order; false if one cannot be computed earlier (not
/// speculatable, a phi, reads memory).
static bool collectExpression(Value *V, Loop *L, SmallVectorImpl<Instruction *> &Out,
                              SmallPtrSetImpl<Instruction *> &Seen,
                              unsigned Depth = 0) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I || !L->contains(I) || I->getParent() == L->getHeader() ||
      Seen.contains(I))
    return true;
  if (Depth > 12 || isa<PHINode>(I) || I->mayReadOrWriteMemory() ||
      !isSafeToSpeculativelyExecute(I))
    return false;
  for (Value *Op : I->operands())
    if (!collectExpression(Op, L, Out, Seen, Depth + 1))
      return false;
  Seen.insert(I);
  Out.push_back(I);
  return true;
}

bool AGXHoistGuardedLoads::runOnLoop(Loop *L, DominatorTree &DT) {
  BasicBlock *Header = L->getHeader(), *Latch = L->getLoopLatch();
  // every block between the header and the latch runs each iteration, or is
  // skipped by a branch inside the loop: the latch is the only exit
  if (!Latch || L->getExitingBlock() != Latch)
    return false;
  // the loop writes no memory and has no other effects
  for (BasicBlock *BB : L->blocks())
    for (Instruction &I : *BB)
      if (I.mayWriteToMemory() || I.mayHaveSideEffects())
        return false;

  // the loads before the loop (they dominate its header)
  SmallVector<LoadInst *, 16> Before;
  for (BasicBlock &BB : *Header->getParent())
    if (!L->contains(&BB) && DT.dominates(&BB, Header))
      for (Instruction &I : BB)
        if (auto *LI = dyn_cast<LoadInst>(&I))
          if (LI->isSimple())
            Before.push_back(LI);
  if (Before.empty())
    return false;

  bool Changed = false;
  // copies of address instructions in the header, shared by the loads (the
  // load/store vectorizer then sees one base for x and y)
  ValueToValueMapTy Map;
  for (BasicBlock *P : L->blocks()) {
    if (!DT.dominates(P, Latch))
      continue;
    Value *X, *C;
    CmpPredicate Pred;
    BasicBlock *IfTrue, *IfFalse;
    if (!match(P->getTerminator(),
               m_Br(m_ICmp(Pred, m_Value(X), m_Value(C)), IfTrue, IfFalse)) ||
        !L->isLoopInvariant(C))
      continue;
    // B runs exactly when X != C
    BasicBlock *B = Pred == ICmpInst::ICMP_EQ   ? IfFalse
                    : Pred == ICmpInst::ICMP_NE ? IfTrue
                                                : nullptr;
    if (!B || !L->contains(B) || B->getSinglePredecessor() != P ||
        B == Header)
      continue;
    SmallVector<LoadInst *, 4> Loads;
    for (Instruction &I : *B)
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (LI->isSimple() &&
            LI->getPointerAddressSpace() == AGX::DEVICE_ADDRESS)
          Loads.push_back(LI);
    for (LoadInst *LI : Loads) {
      Value *Ptr = LI->getPointerOperand();
      const DataLayout &DL = LI->getDataLayout();
      uint64_t Size = DL.getTypeStoreSize(LI->getType());
      // the element of C was loaded before the loop, at least as wide
      bool Safe = any_of(Before, [&](LoadInst *B0) {
        return DL.getTypeStoreSize(B0->getType()) >= Size &&
               B0->getAlign() >= LI->getAlign() &&
               sameWithXAsC(Ptr, B0->getPointerOperand(), X, C);
      });
      if (!Safe)
        continue;
      SmallVector<Instruction *, 8> Expr;
      SmallPtrSet<Instruction *, 8> Seen;
      if (!collectExpression(Ptr, L, Expr, Seen))
        continue;
      // the address and the load at the end of the header
      Instruction *At = Header->getTerminator();
      for (Instruction *I : Expr) {
        if (Map.count(I))
          continue;
        Instruction *Copy = I->clone();
        Copy->insertBefore(At->getIterator());
        Copy->setName(I->getName());
        for (Use &U : Copy->operands())
          if (Value *M = Map.lookup(U.get()))
            U.set(M);
        Map[I] = Copy;
      }
      LI->moveBefore(At->getIterator());
      if (Value *M = Map.lookup(Ptr))
        LI->setOperand(LI->getPointerOperandIndex(), M);
      Changed = true;
    }
  }
  return Changed;
}

bool AGXHoistGuardedLoads::runOnFunction(Function &F) {
  if (!AGXHoistGuardedLoadsOpt)
    return false;
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder())
    if (L->isInnermost())
      Changed |= runOnLoop(L, DT);
  return Changed;
}
