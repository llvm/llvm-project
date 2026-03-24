//===-- CFG.cpp - BasicBlock analysis --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This family of functions performs analyses on basic blocks, and instructions
// contained within basic blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// The max number of basic blocks explored during reachability analysis between
// two basic blocks. This is kept reasonably small to limit compile time when
// repeatedly used by clients of this analysis (such as captureTracking).
static cl::opt<unsigned> DefaultMaxBBsToExplore(
    "dom-tree-reachability-max-bbs-to-explore", cl::Hidden,
    cl::desc("Max number of BBs to explore for reachability analysis"),
    cl::init(32));

/// FindFunctionBackedges - Analyze the specified function to find all of the
/// loop backedges in the function and return them.  This is a relatively cheap
/// (compared to computing dominators and loop info) analysis.
///
/// The output is added to Result, as pairs of <from,to> edge info.
void llvm::FindFunctionBackedges(const Function &F,
     SmallVectorImpl<std::pair<const BasicBlock*,const BasicBlock*> > &Result) {
  const BasicBlock *BB = &F.getEntryBlock();

  // In the DFS traversal, we maintain three states: unvisited, visited in the
  // past, and visited and currently in the DFS stack. If we have an edge to a
  // block in the stack, we have found a backedge.
  enum VisitState : uint8_t { Unvisited = 0, Visited = 1, InStack = 2 };
  SmallVector<VisitState> BlockState(F.getMaxBlockNumber(), Unvisited);
  struct StackEntry {
    const BasicBlock *BB;
    const_succ_iterator SuccIt;
    const_succ_iterator SuccEnd;

    StackEntry(const BasicBlock *BB)
        : BB(BB), SuccIt(nullptr), SuccEnd(nullptr) {
      auto Succs = successors(BB);
      SuccIt = Succs.begin();
      SuccEnd = Succs.end();
    }
  };
  SmallVector<StackEntry, 8> VisitStack;

  BlockState[BB->getNumber()] = InStack;
  VisitStack.emplace_back(BB);
  do {
    StackEntry &Top = VisitStack.back();
    bool FoundNew = false;
    while (Top.SuccIt != Top.SuccEnd) {
      BB = *Top.SuccIt++;
      if (BlockState[BB->getNumber()] == Unvisited) {
        // Unvisited successor => go down one level.
        BlockState[BB->getNumber()] = InStack;
        VisitStack.emplace_back(BB);
        FoundNew = true;
        break;
      }
      // Successor in VisitStack => backedge.
      if (BlockState[BB->getNumber()] == InStack)
        Result.emplace_back(Top.BB, BB);
    }

    // Go up one level.
    if (!FoundNew) {
      BlockState[Top.BB->getNumber()] = Visited;
      VisitStack.pop_back();
    }
  } while (!VisitStack.empty());
}

/// GetSuccessorNumber - Search for the specified successor of basic block BB
/// and return its position in the terminator instruction's list of
/// successors.  It is an error to call this with a block that is not a
/// successor.
unsigned llvm::GetSuccessorNumber(const BasicBlock *BB,
    const BasicBlock *Succ) {
  const Instruction *Term = BB->getTerminator();
#ifndef NDEBUG
  unsigned e = Term->getNumSuccessors();
#endif
  for (unsigned i = 0; ; ++i) {
    assert(i != e && "Didn't find edge?");
    if (Term->getSuccessor(i) == Succ)
      return i;
  }
}

/// isCriticalEdge - Return true if the specified edge is a critical edge.
/// Critical edges are edges from a block with multiple successors to a block
/// with multiple predecessors.
bool llvm::isCriticalEdge(const Instruction *TI, unsigned SuccNum,
                          bool AllowIdenticalEdges) {
  assert(SuccNum < TI->getNumSuccessors() && "Illegal edge specification!");
  return isCriticalEdge(TI, TI->getSuccessor(SuccNum), AllowIdenticalEdges);
}

bool llvm::isCriticalEdge(const Instruction *TI, const BasicBlock *Dest,
                          bool AllowIdenticalEdges) {
  assert(TI->isTerminator() && "Must be a terminator to have successors!");
  if (TI->getNumSuccessors() == 1) return false;

  assert(is_contained(predecessors(Dest), TI->getParent()) &&
         "No edge between TI's block and Dest.");

  const_pred_iterator I = pred_begin(Dest), E = pred_end(Dest);

  // If there is more than one predecessor, this is a critical edge...
  assert(I != E && "No preds, but we have an edge to the block?");
  const BasicBlock *FirstPred = *I;
  ++I;        // Skip one edge due to the incoming arc from TI.
  if (!AllowIdenticalEdges)
    return I != E;

  // If AllowIdenticalEdges is true, then we allow this edge to be considered
  // non-critical iff all preds come from TI's block.
  for (; I != E; ++I)
    if (*I != FirstPred)
      return true;
  return false;
}

// LoopInfo contains a mapping from basic block to the innermost loop. Find
// the outermost loop in the loop nest that contains BB.
static const Loop *getOutermostLoop(const LoopInfo *LI, const BasicBlock *BB) {
  const Loop *L = LI->getLoopFor(BB);
  return L ? L->getOutermostLoop() : nullptr;
}

template <class StopSetT>
static bool isReachableImpl(SmallVectorImpl<BasicBlock *> &Worklist,
                            const StopSetT &StopSet,
                            const SmallPtrSetImpl<BasicBlock *> *ExclusionSet,
                            const DominatorTree *DT, const LoopInfo *LI,
                            const CycleInfo *CI) {
  // If both LI and CI are passed, use CI, which gives us more information.
  if (CI)
    LI = nullptr;

  // When a stop block is unreachable, it's dominated from everywhere,
  // regardless of whether there's a path between the two blocks.
  if (DT) {
    for (auto *BB : StopSet) {
      if (!DT->isReachableFromEntry(BB)) {
        DT = nullptr;
        break;
      }
    }
  }

  // We can't skip directly from a block that dominates the stop block if the
  // exclusion block is potentially in between.
  if (ExclusionSet && !ExclusionSet->empty())
    DT = nullptr;

  // Normally any block in a loop is reachable from any other block in a loop,
  // however excluded blocks might partition the body of a loop to make that
  // untrue.
  SmallPtrSet<const Loop *, 8> LoopsWithHoles;
  if (LI && ExclusionSet) {
    for (auto *BB : *ExclusionSet) {
      if (const Loop *L = getOutermostLoop(LI, BB))
        LoopsWithHoles.insert(L);
    }
  }

  SmallPtrSet<const Cycle *, 8> CyclesWithHoles;
  if (CI && ExclusionSet) {
    for (auto *BB : *ExclusionSet) {
      if (const Cycle *C = CI->getTopLevelParentCycle(BB))
        CyclesWithHoles.insert(C);
    }
  }

  SmallPtrSet<const Loop *, 2> StopLoops;
  if (LI) {
    for (auto *StopSetBB : StopSet) {
      if (const Loop *L = getOutermostLoop(LI, StopSetBB))
        StopLoops.insert(L);
    }
  }

  SmallPtrSet<const Cycle *, 2> StopCycles;
  if (CI) {
    for (auto *StopSetBB : StopSet) {
      if (const Cycle *C = CI->getTopLevelParentCycle(StopSetBB))
        StopCycles.insert(C);
    }
  }

  unsigned Limit = DefaultMaxBBsToExplore;
  SmallPtrSet<const BasicBlock*, 32> Visited;
  do {
    BasicBlock *BB = Worklist.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;
    if (StopSet.contains(BB))
      return true;
    if (ExclusionSet && ExclusionSet->count(BB))
      continue;
    if (DT) {
      if (llvm::any_of(StopSet, [&](const BasicBlock *StopBB) {
            return DT->dominates(BB, StopBB);
          }))
        return true;
    }

    const Loop *OuterL = nullptr;
    if (LI) {
      OuterL = getOutermostLoop(LI, BB);
      // If we're in a loop with a hole, not all blocks in the loop are
      // reachable from all other blocks. That implies we can't simply jump to
      // the loop's exit blocks, as that exit might need to pass through an
      // excluded block. Clear Outer so we process BB's successors.
      if (LoopsWithHoles.count(OuterL))
        OuterL = nullptr;
      else if (StopLoops.contains(OuterL))
        return true;
    }

    const Cycle *OuterC = nullptr;
    if (CI) {
      OuterC = CI->getTopLevelParentCycle(BB);
      if (CyclesWithHoles.count(OuterC))
        OuterC = nullptr;
      else if (StopCycles.contains(OuterC))
        return true;
    }

    if (!--Limit) {
      // We haven't been able to prove it one way or the other. Conservatively
      // answer true -- that there is potentially a path.
      return true;
    }

    if (OuterL) {
      // All blocks in a single loop are reachable from all other blocks. From
      // any of these blocks, we can skip directly to the exits of the loop,
      // ignoring any other blocks inside the loop body.
      OuterL->getExitBlocks(Worklist);
    } else if (OuterC) {
      OuterC->getExitBlocks(Worklist);
    } else {
      Worklist.append(succ_begin(BB), succ_end(BB));
    }
  } while (!Worklist.empty());

  // We have exhausted all possible paths and are certain that 'To' can not be
  // reached from 'From'.
  return false;
}

template <class T> class SingleEntrySet {
public:
  using const_iterator = const T *;

  SingleEntrySet(T Elem) : Elem(Elem) {}

  bool contains(T Other) const { return Elem == Other; }

  const_iterator begin() const { return &Elem; }
  const_iterator end() const { return &Elem + 1; }

private:
  T Elem;
};

bool llvm::isPotentiallyReachableFromMany(
    SmallVectorImpl<BasicBlock *> &Worklist, const BasicBlock *StopBB,
    const SmallPtrSetImpl<BasicBlock *> *ExclusionSet, const DominatorTree *DT,
    const LoopInfo *LI, const CycleInfo *CI) {
  return isReachableImpl<SingleEntrySet<const BasicBlock *>>(
      Worklist, SingleEntrySet<const BasicBlock *>(StopBB), ExclusionSet, DT,
      LI, CI);
}

bool llvm::isManyPotentiallyReachableFromMany(
    SmallVectorImpl<BasicBlock *> &Worklist,
    const SmallPtrSetImpl<const BasicBlock *> &StopSet,
    const SmallPtrSetImpl<BasicBlock *> *ExclusionSet, const DominatorTree *DT,
    const LoopInfo *LI, const CycleInfo *CI) {
  return isReachableImpl<SmallPtrSetImpl<const BasicBlock *>>(
      Worklist, StopSet, ExclusionSet, DT, LI, CI);
}

bool llvm::isPotentiallyReachable(
    const BasicBlock *A, const BasicBlock *B,
    const SmallPtrSetImpl<BasicBlock *> *ExclusionSet, const DominatorTree *DT,
    const LoopInfo *LI, const CycleInfo *CI) {
  assert(A->getParent() == B->getParent() &&
         "This analysis is function-local!");

  if (DT) {
    if (DT->isReachableFromEntry(A) && !DT->isReachableFromEntry(B))
      return false;
    if (!ExclusionSet || ExclusionSet->empty()) {
      if (A->isEntryBlock() && DT->isReachableFromEntry(B))
        return true;
      if (B->isEntryBlock() && DT->isReachableFromEntry(A))
        return false;
    }
  }

  SmallVector<BasicBlock*, 32> Worklist;
  Worklist.push_back(const_cast<BasicBlock*>(A));

  return isPotentiallyReachableFromMany(Worklist, B, ExclusionSet, DT, LI, CI);
}

bool llvm::isPotentiallyReachable(
    const Instruction *A, const Instruction *B,
    const SmallPtrSetImpl<BasicBlock *> *ExclusionSet, const DominatorTree *DT,
    const LoopInfo *LI, const CycleInfo *CI) {
  assert(A->getParent()->getParent() == B->getParent()->getParent() &&
         "This analysis is function-local!");

  if (A->getParent() == B->getParent()) {
    // The same block case is special because it's the only time we're looking
    // within a single block to see which instruction comes first. Once we
    // start looking at multiple blocks, the first instruction of the block is
    // reachable, so we only need to determine reachability between whole
    // blocks.
    BasicBlock *BB = const_cast<BasicBlock *>(A->getParent());

    // If A comes before B, then B is definitively reachable from A.
    if (A == B || A->comesBefore(B))
      return true;

    // If the block is in a cycle (and there are no excluded blocks), then we
    // can reach any instruction in the block from any other instruction in the
    // block by going around a backedge.
    if (!ExclusionSet || ExclusionSet->empty()) {
      // If cycle info is available, we can know for sure whether or not a
      // block is part of a cycle.
      if (CI)
        return CI->getCycle(BB) != nullptr;

      // If only loop info is available, even if the block is not part of a
      // natural loop, it may still be part of an irreducible cycle.
      if (LI && LI->getLoopFor(BB) != nullptr)
        return true;
    }

    // Can't be in a loop if it's the entry block -- the entry block may not
    // have predecessors.
    if (BB->isEntryBlock())
      return false;

    // Otherwise, continue doing the normal per-BB CFG walk.
    SmallVector<BasicBlock*, 32> Worklist;
    Worklist.append(succ_begin(BB), succ_end(BB));
    if (Worklist.empty()) {
      // We've proven that there's no path!
      return false;
    }

    return isPotentiallyReachableFromMany(Worklist, B->getParent(),
                                          ExclusionSet, DT, LI, CI);
  }

  return isPotentiallyReachable(A->getParent(), B->getParent(), ExclusionSet,
                                DT, LI, CI);
}

static bool instructionDoesNotReturn(const Instruction &I) {
  if (auto *CB = dyn_cast<CallBase>(&I))
    return CB->hasFnAttr(Attribute::NoReturn);
  return false;
}

// A basic block can only return if it terminates with a ReturnInst and does not
// contain calls to noreturn functions.
static bool basicBlockCanReturn(const BasicBlock &BB) {
  if (!isa<ReturnInst>(BB.getTerminator()))
    return false;
  return none_of(BB, instructionDoesNotReturn);
}

// FIXME: this doesn't handle recursion.
bool llvm::canReturn(const Function &F) {
  SmallVector<const BasicBlock *, 16> Worklist;
  SmallPtrSet<const BasicBlock *, 16> Visited;

  Visited.insert(&F.front());
  Worklist.push_back(&F.front());

  do {
    const BasicBlock *BB = Worklist.pop_back_val();
    if (basicBlockCanReturn(*BB))
      return true;
    for (const BasicBlock *Succ : successors(BB))
      if (Visited.insert(Succ).second)
        Worklist.push_back(Succ);
  } while (!Worklist.empty());

  return false;
}

bool llvm::isPresplitCoroSuspendExitEdge(const BasicBlock &Src,
                                         const BasicBlock &Dest) {
  assert(Src.getParent() == Dest.getParent());
  if (!Src.getParent()->isPresplitCoroutine())
    return false;
  if (auto *SW = dyn_cast<SwitchInst>(Src.getTerminator()))
    if (auto *Intr = dyn_cast<IntrinsicInst>(SW->getCondition()))
      return Intr->getIntrinsicID() == Intrinsic::coro_suspend &&
             SW->getDefaultDest() == &Dest;
  return false;
}
