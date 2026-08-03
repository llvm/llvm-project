//===-- PISALayout.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define GET_PISAAtomicSubOpcode_DECL
#include "PISAGenSearchableTables.inc"

#include "PISA.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/PISAAddrSpace.h"

#define DEBUG_TYPE "pisa-layout"
#define DEBUG_NAME "PISA layout"

using namespace llvm;

namespace {
/// This is a temporary solution to the issues with divergent barriers.
/// Ultimately it's going to be replaced with MachineBlockPlacement, which
/// currently is causing issues in the PISA backend and cannot be
/// enabled yet.
///
/// @brief sort basic blocks into topological order
/// Arbitrary reverse postorder is not sufficient.
/// Whenever it is possible, we want to layout blocks in such way
/// that the we can recognize the control-flow structures.
class PISALayout : public FunctionPass {
public:
  static char ID;
  PISALayout() : FunctionPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// @brief Provides name of pass
  virtual StringRef getPassName() const override { return DEBUG_NAME; }

  virtual bool runOnFunction(Function &F) override;

private:
  BasicBlock *getLastReturnBlock(Function &F);
  void layoutBlocks(Function &F, LoopInfo &LI);
  void layoutBlocks(Function &F);
  BasicBlock *selectSucc(BasicBlock *CurrBlk, bool SelectNoInstBlk,
                         const LoopInfo &LI,
                         const SmallSet<BasicBlock *, 8> &VisitSet);
  bool isAtomicWrite(Instruction *Inst, bool OnlyLocalMem);
  bool isAtomicRead(Instruction *Inst, bool OnlyLocalMem);
  Value *getMemoryOperand(Instruction *Inst, bool OnlyLocalMem);
  bool isReturnBlock(BasicBlock *BB);
  bool tryMovingWrite(Instruction *Write, Loop *Loop, LoopInfo &LI);
  void moveAtomicWrites2Loop(Function &F, LoopInfo &LI, bool OnlyLocalMem);

  PostDominatorTree *PDT = nullptr;
  DominatorTree *DT = nullptr;
};
} // namespace

char PISALayout::ID = 0;
INITIALIZE_PASS_BEGIN(PISALayout, DEBUG_TYPE, DEBUG_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(PISALayout, DEBUG_TYPE, DEBUG_NAME, false, false)

constexpr unsigned BreakBlockSizeLimit = 3;

static void pushSucc(BasicBlock *BB, std::function<bool(BasicBlock *)> Cond,
                     SmallVector<BasicBlock *> &VisitVec,
                     SmallSet<BasicBlock *, 8> &VisitSet) {
  for (succ_iterator IT = succ_begin(BB), End = succ_end(BB); IT != End; ++IT) {
    BasicBlock *Succ = *IT;
    if (!VisitSet.count(Succ) && Cond(Succ)) {
      VisitVec.push_back(Succ);
      VisitSet.insert(Succ);
      break;
    }
  }
}

inline static auto sizeWithoutDebug(const BasicBlock *BB) { return BB->size(); }

void PISALayout::getAnalysisUsage(AnalysisUsage &AU) const {
  // Doesn't change the IR at all, it just move the blocks so no changes in the
  // IR
  AU.setPreservesAll();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<PostDominatorTreeWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
}

bool PISALayout::runOnFunction(Function &Func) {
  PDT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  if (LI.empty())
    layoutBlocks(Func);
  else
    layoutBlocks(Func, LI);

  return true;
}

// check if the instruction is atomic write (xchg or cmpxchng)
bool PISALayout::isAtomicWrite(Instruction *Inst, bool OnlyLocalMem) {
  auto MemCond = [Inst, OnlyLocalMem]() {
    Value *Ptr = Inst->getOperand(0);
    bool IsLocalMem = Ptr->getType()->getPointerAddressSpace() ==
                      static_cast<unsigned>(PISAAS::AddressSpace::SHARED);
    return (!OnlyLocalMem || IsLocalMem);
  };

  if (isa<AtomicCmpXchgInst>(Inst) && MemCond())
    return true;

  if (auto *AtomicRMW = dyn_cast<AtomicRMWInst>(Inst))
    return AtomicRMW->getOperation() == AtomicRMWInst::Xchg && MemCond();

  if (auto *CI = dyn_cast<CallInst>(Inst)) {
    Function *F = CI->getCalledFunction();
    if (!F)
      return false;

    Intrinsic::ID IntrID = F->getIntrinsicID();
    if (IntrID == Intrinsic::pisa_cas_fatom && MemCond())
      return true;
  }

  return false;
}

// check if the instruction is atomic read (ATOMIC_OR with src == 0)
bool PISALayout::isAtomicRead(Instruction *Inst, bool OnlyLocalMem) {
  auto LocalMemCond = [Inst, OnlyLocalMem]() {
    return !OnlyLocalMem ||
           Inst->getOperand(0)->getType()->getPointerAddressSpace() ==
               static_cast<unsigned>(PISAAS::AddressSpace::SHARED);
  };

  auto *AtomicRMW = dyn_cast<AtomicRMWInst>(Inst);
  if (AtomicRMW && AtomicRMW->getOperation() == AtomicRMWInst::Or &&
      LocalMemCond()) {
    ConstantInt *Src = dyn_cast<ConstantInt>(AtomicRMW->getValOperand());
    return Src && Src->getZExtValue() == 0;
  }

  return false;
}

// get memory operand for atomic read or write
Value *PISALayout::getMemoryOperand(Instruction *Inst, bool OnlyLocalMem) {
  if (!isAtomicRead(Inst, OnlyLocalMem) && !isAtomicWrite(Inst, OnlyLocalMem))
    return nullptr;

  Value *DstAddr = Inst->getOperand(0);
  if (auto *PTI = dyn_cast<PtrToIntInst>(DstAddr))
    return PTI->getPointerOperand();

  return DstAddr;
}

bool PISALayout::isReturnBlock(BasicBlock *BB) {
  return isa<ReturnInst>(BB->getTerminator());
}

// Try moving atomic write (or its loop) into the given destination loop
// If there are no direct predecessor in the needed loop,
// Try to move it together with a chain of predecessors. New BB is added in
// chain if it is either single predecessor or it is a previous node in current
// layout.
//
bool PISALayout::tryMovingWrite(Instruction *Write, Loop *LP, LoopInfo &LI) {
  SmallVector<BasicBlock *> BlocksToMove;

  if (Loop *WritingLoop = LI.getLoopFor(Write->getParent())) {
    auto Blocks = WritingLoop->getBlocks();
    for (auto &BB : Blocks) {
      if (isReturnBlock(BB))
        return false;
      BlocksToMove.push_back(BB);
    }
  } else {
    if (!isReturnBlock(Write->getParent()))
      BlocksToMove.push_back(Write->getParent());
    else
      return false;
  }

  // Loop exits when:
  // - all BasicBlocks with the Write or its loop has been moved
  // - processed BasicBlock contains a return or has more than one predecessor
  while (true) {
    BasicBlock *Blk = BlocksToMove.back();

    // If one (and only one) of the predecessors is in the needed loop, move
    // blocks after it
    BasicBlock *InsertPoint = nullptr;
    int PredsInLoop = 0;
    for (pred_iterator PredIter = pred_begin(Blk), PredEnd = pred_end(Blk);
         PredIter != PredEnd; ++PredIter) {
      BasicBlock *Pred = *PredIter;
      if (LP->contains(Pred)) {
        PredsInLoop++;
        InsertPoint = Pred;
      }
    }
    if (PredsInLoop == 1) {
      for (auto *BB : BlocksToMove)
        BB->moveAfter(InsertPoint);
      return true;
    }
    if (PredsInLoop > 1)
      return false;

    // Add prev node if it is the predecessor of the block
    bool PredPushed = false;
    for (pred_iterator PredIter = pred_begin(Blk), PredEnd = pred_end(Blk);
         PredIter != PredEnd; ++PredIter) {
      BasicBlock *Pred = *PredIter;

      if ((Pred == Blk->getPrevNode()) && !isReturnBlock(Pred)) {
        BlocksToMove.push_back(Pred);
        PredPushed = true;
        break;
      }
    }

    if (PredPushed)
      continue;

    // Add predecessor if it is single
    BasicBlock *Pred = Blk->getSinglePredecessor();
    if (Pred && !isReturnBlock(Pred)) {
      BlocksToMove.push_back(Pred);
      PredPushed = true;
    } else
      // Don't move the blocks and return
      return false;
  }
}

// Place basic blocks with atomic write (or the whole loop with the
// atomic write) into the other loop if there is an atomic read
// from the same memory, which dominates the write.
//
// It benefits cases like:
//
// Loop:
//    Load A
//    if (!pred(Load A))
//    {
//        break;
//    }
//    if (success(do_work())
//    {
//        Store A;
//        break;
//    }
// Br Loop
//
// If the Store is placed after the back edge of the loop
// there will be goto instruction disabling channels based on some
// "success(do_work())" condition placed before the back edge in SIMD control
// flow, and the store will be delayed until the whole loop is finished. It
// makes "if (!pred(Load A))" checking useless and doesn't allow to perform
// early break based on the condition.
//
void PISALayout::moveAtomicWrites2Loop(Function &F, LoopInfo &LI,
                                       bool OnlyLocalMem) {
  SmallVector<Instruction *> Writes;
  SmallVector<Instruction *> Reads;
  for (auto &I : instructions(F)) {
    if (isAtomicWrite(&I, OnlyLocalMem))
      Writes.push_back(&I);
    else if (isAtomicRead(&I, OnlyLocalMem))
      Reads.push_back(&I);
  }

  // write: LoopWhereToMove mapping
  MapVector<Instruction *, Loop *> WritesToMove;

  for (auto *Read : Reads)
    for (auto *Write : Writes)
      if (getMemoryOperand(Read, OnlyLocalMem) ==
          getMemoryOperand(Write, OnlyLocalMem)) {
        Loop *ReadLoop = LI.getLoopFor(Read->getParent());
        Loop *WriteLoop = LI.getLoopFor(Write->getParent());
        if (ReadLoop && (ReadLoop != WriteLoop) &&
            ((DT->dominates(Read, Write))))
          WritesToMove[Write] = LI.getLoopFor(Read->getParent());
      }

  for (const auto &Pair : WritesToMove) {
    Instruction *Write = Pair.first;
    Loop *Loop = Pair.second;

    tryMovingWrite(Write, Loop, LI);
  }
}

static bool hasThreadGroupBarrierInBlock(BasicBlock *BB) {
  Module *M = BB->getParent()->getParent();
  for (Function &F : *M) {
    if (!F.isDeclaration() || !F.isIntrinsic())
      continue;

    Intrinsic::ID IntrID = F.getIntrinsicID();
    if (IntrID == Intrinsic::pisa_workgroup_barrier)
      for (auto *U : F.users()) {
        auto *Inst = dyn_cast<Instruction>(U);
        if (Inst && Inst->getParent() == BB)
          return true;
      }
  }
  return false;
}

BasicBlock *PISALayout::getLastReturnBlock(Function &F) {
  // If Func has any return BB, return the last return BB (may have multiple);
  // otherwise, return the last BB that has no succ;
  //     or nullptr if every BB has Succ (infinite looping)
  BasicBlock *NoRetAndNoSucc = nullptr; // for func that never returns
  for (auto RI = std::make_reverse_iterator(F.end()),
            RE = std::make_reverse_iterator(F.begin());
       RI != RE; ++RI) {
    BasicBlock *BB = &*RI;
    if (succ_begin(BB) == succ_end(BB)) {
      if (isa_and_nonnull<ReturnInst>(BB->getTerminator()))
        return BB;
      if (!NoRetAndNoSucc)
        NoRetAndNoSucc = BB;
    }
  }
  // Function does not have a return block
  return NoRetAndNoSucc;
}

//
// selectSucc: select a succ with condition SelectNoInstBlk and return it.
//
// This is used to select one if there are two Successors with condition
// SelectNoInstBlk, rather than take the first one in the succ list.
//
// Condition SelectNoInstBlk: If SelectNoInstBlk is true, select an empty
// block, if it is false, select non-empty block.
//
BasicBlock *PISALayout::selectSucc(BasicBlock *CurrBlk, bool SelectNoInstBlk,
                                   const LoopInfo &LI,
                                   const SmallSet<BasicBlock *, 8> &VisitSet) {
  SmallVector<BasicBlock *, 4> Succs;
  for (succ_iterator SI = succ_begin(CurrBlk), SE = succ_end(CurrBlk); SI != SE;
       ++SI) {
    BasicBlock *Succ = *SI;
    auto Size = sizeWithoutDebug(Succ);
    if (VisitSet.count(Succ) == 0 &&
        ((SelectNoInstBlk && Size <= 1) || (!SelectNoInstBlk && Size > 1)))
      Succs.push_back(Succ);
  }

  // Right now, only handle the case of two empty blocks.
  // If it has no two empty blocks, just take the first
  // one and return it.
  if (Succs.size() != 2 || !SelectNoInstBlk)
    return Succs.empty() ? nullptr : Succs[0];

  // For two empty blocks, the case we want to handle
  // is the following:
  //
  //     (B0 = CurrBlk)
  //   B0 : if (c) goto THEN  (else goto ELSE)
  //   ELSE : goto B2
  //   B1 : ....
  //   B2 : ....
  //    ......
  //   Bn :
  //      (ELSE, B1, B2, ..., Bn) has END as single exit
  //   THEN: goto END:
  //   END :
  //       PHI...
  //
  // where ELSE and THEN are empty BBs, and END has phi in it.
  // In this case, THEN and ELSE might have phi moves as the result
  // DeSSA when emitting visa. For example, suppose  d0 = s0 will
  // be emitted in THEN.  If s0 is dead after THEN, it would be good
  // to lay out THEN right after B0 as the live-range of s0 will not
  // be overlapped with ones in ELSE. (If s0 is live out of THEN,
  // moving THEN right after B0 or right before END does not matter
  // as far as liveness is concerned.).  To lay out THEN first, this
  // function will select ELSE to return (as the algo does layout
  // backward).
  //
  // For simplicity, assume those BBs are not inside loops. It could
  // be applied to Loop later when appropriate testing is done.
  BasicBlock *Suc0 = Succs[0], *Suc1 = Succs[1];
  BasicBlock *SS0 = Suc0->getSingleSuccessor();

  if (SS0 && (SS0 != Suc1) && isa<PHINode>(&*SS0->begin()) &&
      !LI.getLoopFor(Suc0) && PDT->dominates(SS0, Suc1))
    return Suc1;

  return Suc0;
}

void PISALayout::layoutBlocks(Function &F, LoopInfo &LI) {
  SmallVector<BasicBlock *> VisitVec;
  SmallSet<BasicBlock *, 8> VisitSet;
  // Insertion Position per loop header
  MapVector<BasicBlock *, BasicBlock *> InsPos;

  BasicBlock *Entry = &(F.getEntryBlock());
  VisitVec.push_back(Entry);
  VisitSet.insert(Entry);
  InsPos[Entry] = Entry;

  // Push a return block to make sure the last BB is the return block.
  if (BasicBlock *LastReturnBlock = getLastReturnBlock(F)) {
    if (LastReturnBlock != Entry) {
      VisitVec.push_back(LastReturnBlock);
      VisitSet.insert(LastReturnBlock);
    }
  }

  while (!VisitVec.empty()) {
    BasicBlock *BB = VisitVec.back();
    Loop *CurLoop = LI.getLoopFor(BB);
    if (CurLoop) {
      auto *HD = CurLoop->getHeader();
      if (BB == HD && InsPos.find(HD) == InsPos.end())
        InsPos[BB] = BB;
    }

    // push: time for DFS visit
    auto PushHasInstCond = [](BasicBlock *Succ) -> bool {
      return sizeWithoutDebug(Succ) > 1;
    };
    pushSucc(BB, PushHasInstCond, VisitVec, VisitSet);
    if (BB != VisitVec.back())
      continue;
    // push: time for DFS visit
    if (BasicBlock *ABlk = selectSucc(BB, true, LI, VisitSet)) {
      VisitVec.push_back(ABlk);
      VisitSet.insert(ABlk);
      continue;
    }

    // pop: time to move the block to the right location
    if (BB == VisitVec.back()) {
      VisitVec.pop_back();
      if (CurLoop) {
        auto *HD = CurLoop->getHeader();
        if (BB != HD) {
          // move the block to the beginning of the loop
          auto *Insp = InsPos[HD];
          assert(Insp);
          if (BB != Insp) {
            BB->moveBefore(Insp);
            InsPos[HD] = BB;
          }
        } else {
          // move the entire loop to the beginning of
          // the parent loop
          auto *LoopStart = InsPos[HD];
          assert(LoopStart);
          auto *PaLoop = CurLoop->getParentLoop();
          auto *PaHd = PaLoop ? PaLoop->getHeader() : Entry;
          auto *Insp = InsPos[PaHd];
          if (LoopStart == HD)
            // single-block loop
            HD->moveBefore(Insp);
          else {
            // loop-header is not moved yet, so should be at the end
            // use splice
            F.splice(Insp->getIterator(), &F, LoopStart->getIterator(),
                     HD->getIterator());
            HD->moveBefore(LoopStart);
          }
          InsPos[PaHd] = HD;
        }
      } else {
        auto *Insp = InsPos[Entry];
        if (BB != Insp) {
          BB->moveBefore(Insp);
          InsPos[Entry] = BB;
        }
      }
    }
  }

  moveAtomicWrites2Loop(F, LI, false);

  // if function has a single exit, then the last block must be an exit
  // fix the loop-exit pattern, put break-blocks into the loop
  for (BasicBlock &BB : F) {
    Loop *CurLoop = LI.getLoopFor(&BB);
    bool AllPredLoopExit = true;
    unsigned NumPreds = 0;
    SmallPtrSet<BasicBlock *, 4> PredSet;
    for (pred_iterator PredIter = pred_begin(&BB), PredEnd = pred_end(&BB);
         PredIter != PredEnd; ++PredIter) {
      BasicBlock *Pred = *PredIter;
      NumPreds++;
      Loop *PredLoop = LI.getLoopFor(Pred);
      if (CurLoop == PredLoop) {
        BasicBlock *PredPred = Pred->getSinglePredecessor();
        if (PredPred) {
          Loop *PredPredLoop = LI.getLoopFor(PredPred);
          if (PredPredLoop != CurLoop &&
              (!CurLoop || CurLoop->contains(PredPredLoop))) {
            // Debug instructions should not be counted into considered size
            if (sizeWithoutDebug(Pred) <= BreakBlockSizeLimit &&
                !hasThreadGroupBarrierInBlock(Pred))
              PredSet.insert(Pred);
            else
              AllPredLoopExit = false;
            break;
          }
        }
      } else if (!CurLoop || CurLoop->contains(PredLoop))
        continue;
      else {
        AllPredLoopExit = false;
        break;
      }
    }
    if (AllPredLoopExit && NumPreds > 1) {
      for (BasicBlock *Pred : PredSet) {
        BasicBlock *PredPred = Pred->getSinglePredecessor();
        Pred->moveAfter(PredPred);
      }
    }
  }
}

void PISALayout::layoutBlocks(Function &F) {
  SmallVector<BasicBlock *> VisitVec;
  SmallSet<BasicBlock *, 8> VisitSet;
  // Reorder basic block to allow more fall-through
  BasicBlock *Entry = &(F.getEntryBlock());
  VisitVec.push_back(Entry);

  // Push a return block to make sure the last BB is the return block.
  if (BasicBlock *LastReturnBlock = getLastReturnBlock(F))
    if (LastReturnBlock != Entry) {
      VisitVec.push_back(LastReturnBlock);
      VisitSet.insert(LastReturnBlock);
    }

  while (!VisitVec.empty()) {
    BasicBlock *BB = VisitVec.back();
    // push in the empty successor
    auto PushNoInstCond = [](BasicBlock *Succ) -> bool {
      return sizeWithoutDebug(Succ) <= 1;
    };
    pushSucc(BB, PushNoInstCond, VisitVec, VisitSet);
    if (BB != VisitVec.back())
      continue;
    // push in all the same-loop successors
    auto PushAnyCond = [](BasicBlock *Succ) -> bool { return true; };
    pushSucc(BB, PushAnyCond, VisitVec, VisitSet);
    //  pop
    if (BB == VisitVec.back()) {
      VisitVec.pop_back();
      if (BB != Entry) {
        BB->moveBefore(Entry);
        Entry = BB;
      }
    }
  }
}

FunctionPass *llvm::createPISALayoutPass() { return new PISALayout(); }
