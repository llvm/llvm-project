//===- FixIrreducible.cpp - Convert irreducible control-flow into loops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// To convert an irreducible cycle C to a natural loop L:
//
// 1. Add a new node N to C.
// 2. Redirect all external incoming edges through N.
// 3. Redirect all edges incident on header H through N.
//
// This is sufficient to ensure that:
//
// a. Every closed path in C also exists in L, with the modification that any
//    path passing through H now passes through N before reaching H.
// b. Every external path incident on any entry of C is now incident on N and
//    then redirected to the entry.
//
// Thus, L is a strongly connected component dominated by N, and hence L is a
// natural loop with header N.
//
// INPUT CFG: The blocks H and B form an irreducible loop with two headers.
//
//                        Entry
//                       /     \
//                      v       v
//                      H ----> B
//                      ^      /|
//                       `----' |
//                              v
//                             Exit
//
// OUTPUT CFG:
//
//                        Entry
//                          |
//                          v
//                          N <---.
//                         / \     \
//                        /   \     |
//                       v     v    /
//                       H --> B --'
//                             |
//                             v
//                            Exit
//
//
// The actual transformation is handled by function CreateControlFlowHub, which
// takes a set of incoming blocks (the predecessors) and outgoing blocks (the
// entries). The function also moves every PHINode in an outgoing block to the
// hub. Since the hub dominates all the outgoing blocks, each such PHINode
// continues to dominate its uses. Since every entry the cycle has at least two
// predecessors, every value used in the entry (or later) but defined in a
// predecessor (or earlier) is represented by a PHINode in a entry. Hence the
// above handling of PHINodes is sufficient and no further processing is
// required to restore SSA.
//
// Limitation: The pass cannot handle switch statements and indirect
//             branches. Both must be lowered to plain branches first.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/FixIrreducible.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "fix-irreducible"

using namespace llvm;

namespace {
struct FixIrreducible : public FunctionPass {
  static char ID;
  FixIrreducible() : FunctionPass(ID) {
    initializeFixIrreduciblePass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<CycleInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<CycleInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};
} // namespace

char FixIrreducible::ID = 0;

FunctionPass *llvm::createFixIrreduciblePass() { return new FixIrreducible(); }

INITIALIZE_PASS_BEGIN(FixIrreducible, "fix-irreducible",
                      "Convert irreducible control-flow into natural loops",
                      false /* Only looks at CFG */, false /* Analysis Pass */)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(FixIrreducible, "fix-irreducible",
                    "Convert irreducible control-flow into natural loops",
                    false /* Only looks at CFG */, false /* Analysis Pass */)

// When a new loop is created, existing children of the parent loop may now be
// fully inside the new loop. Reconnect these as children of the new loop.
static void reconnectChildLoops(LoopInfo &LI, Loop *ParentLoop, Loop *NewLoop,
                                BasicBlock *OldHeader) {
  auto &CandidateLoops = ParentLoop ? ParentLoop->getSubLoopsVector()
                                    : LI.getTopLevelLoopsVector();
  // Any candidate is a child iff its header is owned by the new loop. Move all
  // the children to a new vector.
  auto FirstChild = std::partition(
      CandidateLoops.begin(), CandidateLoops.end(),
      [&](Loop *L) { return !NewLoop->contains(L->getHeader()); });
  SmallVector<Loop *, 8> ChildLoops(FirstChild, CandidateLoops.end());
  CandidateLoops.erase(FirstChild, CandidateLoops.end());

  for (Loop *Child : ChildLoops) {
    LLVM_DEBUG(dbgs() << "child loop: " << Child->getHeader()->getName()
                      << "\n");
    // A child loop whose header was the old cycle header gets destroyed since
    // its backedges are removed.
    if (Child->getHeader() == OldHeader) {
      for (auto *BB : Child->blocks()) {
        if (LI.getLoopFor(BB) != Child)
          continue;
        LI.changeLoopFor(BB, NewLoop);
        LLVM_DEBUG(dbgs() << "moved block from child: " << BB->getName()
                          << "\n");
      }
      std::vector<Loop *> GrandChildLoops;
      std::swap(GrandChildLoops, Child->getSubLoopsVector());
      for (auto *GrandChildLoop : GrandChildLoops) {
        GrandChildLoop->setParentLoop(nullptr);
        NewLoop->addChildLoop(GrandChildLoop);
      }
      LI.destroy(Child);
      LLVM_DEBUG(dbgs() << "subsumed child loop (common header)\n");
      continue;
    }

    Child->setParentLoop(nullptr);
    NewLoop->addChildLoop(Child);
    LLVM_DEBUG(dbgs() << "added child loop to new loop\n");
  }
}

// Given a set of blocks and headers in an irreducible SCC, convert it into a
// natural loop. Also insert this new loop at its appropriate place in the
// hierarchy of loops.
static bool fixIrreducible(Cycle &C, CycleInfo &CI, DominatorTree &DT,
                           LoopInfo *LI) {
  if (C.isReducible())
    return false;

  SetVector<BasicBlock *> Predecessors;

  // Redirect internal edges incident on the header.
  BasicBlock *OldHeader = C.getHeader();
  for (BasicBlock *P : predecessors(OldHeader)) {
    if (C.contains(P))
      Predecessors.insert(P);
  }

  // Redirect external incoming edges. This includes the edges on the header.
  for (BasicBlock *E : C.entries()) {
    for (BasicBlock *P : predecessors(E)) {
      if (!C.contains(P))
        Predecessors.insert(P);
    }
  }

  LLVM_DEBUG(
      dbgs() << "Found predecessors:";
      for (auto P : Predecessors) {
        dbgs() << " " << P->getName();
      }
      dbgs() << "\n");

  // Redirect all the backedges through a "hub" consisting of a series
  // of guard blocks that manage the flow of control from the
  // predecessors to the headers.
  SmallVector<BasicBlock *> GuardBlocks;

  // Minor optimization: The cycle entries are discovered in an order that is
  // the opposite of the order in which these blocks appear as branch targets.
  // This results in a lot of condition inversions in the control flow out of
  // the new ControlFlowHub, which can be mitigated if the orders match. So we
  // reverse the entries when adding them to the hub.
  SetVector<BasicBlock *> Entries;
  Entries.insert(C.entry_rbegin(), C.entry_rend());

  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
  CreateControlFlowHub(&DTU, GuardBlocks, Predecessors, Entries, "irr");
#if defined(EXPENSIVE_CHECKS)
  assert(DT.verify(DominatorTree::VerificationLevel::Full));
#else
  assert(DT.verify(DominatorTree::VerificationLevel::Fast));
#endif

  for (auto *G : GuardBlocks) {
    LLVM_DEBUG(dbgs() << "added guard block: " << G->getName() << "\n");
    CI.addBlockToCycle(G, &C);
  }
  C.setSingleEntry(GuardBlocks[0]);

  if (!LI)
    return true;

  Loop *ParentLoop = LI->getLoopFor(OldHeader);
  // Create a new loop from the now-transformed cycle
  auto *NewLoop = LI->AllocateLoop();
  if (ParentLoop) {
    ParentLoop->addChildLoop(NewLoop);
  } else {
    LI->addTopLevelLoop(NewLoop);
  }

  // Add the guard blocks to the new loop. The first guard block is
  // the head of all the backedges, and it is the first to be inserted
  // in the loop. This ensures that it is recognized as the
  // header. Since the new loop is already in LoopInfo, the new blocks
  // are also propagated up the chain of parent loops.
  for (auto *G : GuardBlocks) {
    LLVM_DEBUG(dbgs() << "added guard block: " << G->getName() << "\n");
    NewLoop->addBasicBlockToLoop(G, *LI);
  }

  for (auto *BB : C.blocks()) {
    NewLoop->addBlockEntry(BB);
    if (LI->getLoopFor(BB) == ParentLoop) {
      LLVM_DEBUG(dbgs() << "moved block from parent: " << BB->getName()
                        << "\n");
      LI->changeLoopFor(BB, NewLoop);
    } else {
      LLVM_DEBUG(dbgs() << "added block from child: " << BB->getName() << "\n");
    }
  }
  LLVM_DEBUG(dbgs() << "header for new loop: "
                    << NewLoop->getHeader()->getName() << "\n");

  reconnectChildLoops(*LI, ParentLoop, NewLoop, OldHeader);

  NewLoop->verifyLoop();
  if (ParentLoop) {
    ParentLoop->verifyLoop();
  }
#if defined(EXPENSIVE_CHECKS)
  LI->verify(DT);
#endif // EXPENSIVE_CHECKS

  return true;
}

static bool FixIrreducibleImpl(Function &F, CycleInfo &CI, DominatorTree &DT,
                               LoopInfo *LI) {
  LLVM_DEBUG(dbgs() << "===== Fix irreducible control-flow in function: "
                    << F.getName() << "\n");

  assert(hasOnlySimpleTerminator(F) && "Unsupported block terminator.");

  bool Changed = false;
  SmallVector<Cycle *> Worklist{CI.toplevel_cycles()};

  while (!Worklist.empty()) {
    Cycle *C = Worklist.pop_back_val();
    Changed |= fixIrreducible(*C, CI, DT, LI);
    append_range(Worklist, C->children());
  }

  return Changed;
}

bool FixIrreducible::runOnFunction(Function &F) {
  auto *LIWP = getAnalysisIfAvailable<LoopInfoWrapperPass>();
  LoopInfo *LI = LIWP ? &LIWP->getLoopInfo() : nullptr;
  auto &CI = getAnalysis<CycleInfoWrapperPass>().getResult();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  return FixIrreducibleImpl(F, CI, DT, LI);
}

PreservedAnalyses FixIrreduciblePass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  auto *LI = AM.getCachedResult<LoopAnalysis>(F);
  auto &CI = AM.getResult<CycleAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

  if (!FixIrreducibleImpl(F, CI, DT, LI))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  PA.preserve<CycleAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
