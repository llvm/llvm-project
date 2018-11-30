//===--------- LoopSimplifyCFG.cpp - Loop CFG Simplification Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Loop SimplifyCFG Pass. This pass is responsible for
// basic loop CFG cleanup, primarily to assist other loop passes. If you
// encounter a noncanonical CFG construct that causes another loop pass to
// perform suboptimally, this is the place to fix it up.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DomTreeUpdater.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
using namespace llvm;

#define DEBUG_TYPE "loop-simplifycfg"

static cl::opt<bool> EnableTermFolding("enable-loop-simplifycfg-term-folding",
                                       cl::init(true));

STATISTIC(NumTerminatorsFolded,
          "Number of terminators folded to unconditional branches");

/// If \p BB is a switch or a conditional branch, but only one of its successors
/// can be reached from this block in runtime, return this successor. Otherwise,
/// return nullptr.
static BasicBlock *getOnlyLiveSuccessor(BasicBlock *BB) {
  Instruction *TI = BB->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isUnconditional())
      return nullptr;
    if (BI->getSuccessor(0) == BI->getSuccessor(1))
      return BI->getSuccessor(0);
    ConstantInt *Cond = dyn_cast<ConstantInt>(BI->getCondition());
    if (!Cond)
      return nullptr;
    return Cond->isZero() ? BI->getSuccessor(1) : BI->getSuccessor(0);
  }

  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    auto *CI = dyn_cast<ConstantInt>(SI->getCondition());
    if (!CI)
      return nullptr;
    for (auto Case : SI->cases())
      if (Case.getCaseValue() == CI)
        return Case.getCaseSuccessor();
    return SI->getDefaultDest();
  }

  return nullptr;
}

/// Helper class that can turn branches and switches with constant conditions
/// into unconditional branches.
class ConstantTerminatorFoldingImpl {
private:
  Loop &L;
  LoopInfo &LI;
  DominatorTree &DT;
  MemorySSAUpdater *MSSAU;

  // Whether or not the current loop will still exist after terminator constant
  // folding will be done. In theory, there are two ways how it can happen:
  // 1. Loop's latch(es) become unreachable from loop header;
  // 2. Loop's header becomes unreachable from method entry.
  // In practice, the second situation is impossible because we only modify the
  // current loop and its preheader and do not affect preheader's reachibility
  // from any other block. So this variable set to true means that loop's latch
  // has become unreachable from loop header.
  bool DeleteCurrentLoop = false;

  // The blocks of the original loop that will still be reachable from entry
  // after the constant folding.
  SmallPtrSet<BasicBlock *, 8> LiveLoopBlocks;
  // The blocks of the original loop that will become unreachable from entry
  // after the constant folding.
  SmallPtrSet<BasicBlock *, 8> DeadLoopBlocks;
  // The exits of the original loop that will still be reachable from entry
  // after the constant folding.
  SmallPtrSet<BasicBlock *, 8> LiveExitBlocks;
  // The exits of the original loop that will become unreachable from entry
  // after the constant folding.
  SmallVector<BasicBlock *, 8> DeadExitBlocks;
  // The blocks that will still be a part of the current loop after folding.
  SmallPtrSet<BasicBlock *, 8> BlocksInLoopAfterFolding;
  // The blocks that have terminators with constant condition that can be
  // folded. Note: fold candidates should be in L but not in any of its
  // subloops to avoid complex LI updates.
  SmallVector<BasicBlock *, 8> FoldCandidates;

  void dump() const {
    dbgs() << "Constant terminator folding for loop " << L << "\n";
    dbgs() << "After terminator constant-folding, the loop will";
    if (!DeleteCurrentLoop)
      dbgs() << " not";
    dbgs() << " be destroyed\n";
    auto PrintOutVector = [&](const char *Message,
                           const SmallVectorImpl<BasicBlock *> &S) {
      dbgs() << Message << "\n";
      for (const BasicBlock *BB : S)
        dbgs() << "\t" << BB->getName() << "\n";
    };
    auto PrintOutSet = [&](const char *Message,
                           const SmallPtrSetImpl<BasicBlock *> &S) {
      dbgs() << Message << "\n";
      for (const BasicBlock *BB : S)
        dbgs() << "\t" << BB->getName() << "\n";
    };
    PrintOutVector("Blocks in which we can constant-fold terminator:",
                   FoldCandidates);
    PrintOutSet("Live blocks from the original loop:", LiveLoopBlocks);
    PrintOutSet("Dead blocks from the original loop:", DeadLoopBlocks);
    PrintOutSet("Live exit blocks:", LiveExitBlocks);
    PrintOutVector("Dead exit blocks:", DeadExitBlocks);
    if (!DeleteCurrentLoop)
      PrintOutSet("The following blocks will still be part of the loop:",
                  BlocksInLoopAfterFolding);
  }

  /// Fill all information about status of blocks and exits of the current loop
  /// if constant folding of all branches will be done.
  void analyze() {
    LoopBlocksDFS DFS(&L);
    DFS.perform(&LI);
    assert(DFS.isComplete() && "DFS is expected to be finished");

    // Collect live and dead loop blocks and exits.
    LiveLoopBlocks.insert(L.getHeader());
    for (auto I = DFS.beginRPO(), E = DFS.endRPO(); I != E; ++I) {
      BasicBlock *BB = *I;

      // If a loop block wasn't marked as live so far, then it's dead.
      if (!LiveLoopBlocks.count(BB)) {
        DeadLoopBlocks.insert(BB);
        continue;
      }

      BasicBlock *TheOnlySucc = getOnlyLiveSuccessor(BB);

      // If a block has only one live successor, it's a candidate on constant
      // folding. Only handle blocks from current loop: branches in child loops
      // are skipped because if they can be folded, they should be folded during
      // the processing of child loops.
      if (TheOnlySucc && LI.getLoopFor(BB) == &L)
        FoldCandidates.push_back(BB);

      // Handle successors.
      for (BasicBlock *Succ : successors(BB))
        if (!TheOnlySucc || TheOnlySucc == Succ) {
          if (L.contains(Succ))
            LiveLoopBlocks.insert(Succ);
          else
            LiveExitBlocks.insert(Succ);
        }
    }

    // Sanity check: amount of dead and live loop blocks should match the total
    // number of blocks in loop.
    assert(L.getNumBlocks() == LiveLoopBlocks.size() + DeadLoopBlocks.size() &&
           "Malformed block sets?");

    // Now, all exit blocks that are not marked as live are dead.
    SmallVector<BasicBlock *, 8> ExitBlocks;
    L.getExitBlocks(ExitBlocks);
    for (auto *ExitBlock : ExitBlocks)
      if (!LiveExitBlocks.count(ExitBlock))
        DeadExitBlocks.push_back(ExitBlock);

    // Whether or not the edge From->To will still be present in graph after the
    // folding.
    auto IsEdgeLive = [&](BasicBlock *From, BasicBlock *To) {
      if (!LiveLoopBlocks.count(From))
        return false;
      BasicBlock *TheOnlySucc = getOnlyLiveSuccessor(From);
      return !TheOnlySucc || TheOnlySucc == To;
    };

    // The loop will not be destroyed if its latch is live.
    DeleteCurrentLoop = !IsEdgeLive(L.getLoopLatch(), L.getHeader());

    // If we are going to delete the current loop completely, no extra analysis
    // is needed.
    if (DeleteCurrentLoop)
      return;

    // Otherwise, we should check which blocks will still be a part of the
    // current loop after the transform.
    BlocksInLoopAfterFolding.insert(L.getLoopLatch());
    // If the loop is live, then we should compute what blocks are still in
    // loop after all branch folding has been done. A block is in loop if
    // it has a live edge to another block that is in the loop; by definition,
    // latch is in the loop.
    auto BlockIsInLoop = [&](BasicBlock *BB) {
      return any_of(successors(BB), [&](BasicBlock *Succ) {
        return BlocksInLoopAfterFolding.count(Succ) && IsEdgeLive(BB, Succ);
      });
    };
    for (auto I = DFS.beginPostorder(), E = DFS.endPostorder(); I != E; ++I) {
      BasicBlock *BB = *I;
      if (BlockIsInLoop(BB))
        BlocksInLoopAfterFolding.insert(BB);
    }

    // Sanity check: header must be in loop.
    assert(BlocksInLoopAfterFolding.count(L.getHeader()) &&
           "Header not in loop?");
    assert(BlocksInLoopAfterFolding.size() <= LiveLoopBlocks.size() &&
           "All blocks that stay in loop should be live!");
  }

  /// Constant-fold terminators of blocks acculumated in FoldCandidates into the
  /// unconditional branches.
  void foldTerminators() {
    DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

    for (BasicBlock *BB : FoldCandidates) {
      assert(LI.getLoopFor(BB) == &L && "Should be a loop block!");
      BasicBlock *TheOnlySucc = getOnlyLiveSuccessor(BB);
      assert(TheOnlySucc && "Should have one live successor!");

      LLVM_DEBUG(dbgs() << "Replacing terminator of " << BB->getName()
                        << " with an unconditional branch to the block "
                        << TheOnlySucc->getName() << "\n");

      SmallPtrSet<BasicBlock *, 2> DeadSuccessors;
      // Remove all BB's successors except for the live one.
      unsigned TheOnlySuccDuplicates = 0;
      for (auto *Succ : successors(BB))
        if (Succ != TheOnlySucc) {
          DeadSuccessors.insert(Succ);
          // If our successor lies in a different loop, we don't want to remove
          // the one-input Phi because it is a LCSSA Phi.
          bool PreserveLCSSAPhi = !L.contains(Succ);
          Succ->removePredecessor(BB, PreserveLCSSAPhi);
          if (MSSAU)
            MSSAU->removeEdge(BB, Succ);
        } else
          ++TheOnlySuccDuplicates;

      assert(TheOnlySuccDuplicates > 0 && "Should be!");
      // If TheOnlySucc was BB's successor more than once, after transform it
      // will be its successor only once. Remove redundant inputs from
      // TheOnlySucc's Phis.
      bool PreserveLCSSAPhi = !L.contains(TheOnlySucc);
      for (unsigned Dup = 1; Dup < TheOnlySuccDuplicates; ++Dup)
        TheOnlySucc->removePredecessor(BB, PreserveLCSSAPhi);
      if (MSSAU && TheOnlySuccDuplicates > 1)
        MSSAU->removeDuplicatePhiEdgesBetween(BB, TheOnlySucc);

      IRBuilder<> Builder(BB->getContext());
      Instruction *Term = BB->getTerminator();
      Builder.SetInsertPoint(Term);
      Builder.CreateBr(TheOnlySucc);
      Term->eraseFromParent();

      for (auto *DeadSucc : DeadSuccessors)
        DTU.deleteEdge(BB, DeadSucc);

      ++NumTerminatorsFolded;
    }
  }

public:
  ConstantTerminatorFoldingImpl(Loop &L, LoopInfo &LI, DominatorTree &DT,
                                MemorySSAUpdater *MSSAU)
      : L(L), LI(LI), DT(DT), MSSAU(MSSAU) {}
  bool run() {
    assert(L.getLoopLatch() && "Should be single latch!");

    // Collect all available information about status of blocks after constant
    // folding.
    analyze();

    LLVM_DEBUG(dbgs() << "In function " << L.getHeader()->getParent()->getName()
                      << ": ");

    // Nothing to constant-fold.
    if (FoldCandidates.empty()) {
      LLVM_DEBUG(
          dbgs() << "No constant terminator folding candidates found in loop "
                 << L.getHeader()->getName() << "\n");
      return false;
    }

    // TODO: Support deletion of the current loop.
    if (DeleteCurrentLoop) {
      LLVM_DEBUG(
          dbgs()
          << "Give up constant terminator folding in loop "
          << L.getHeader()->getName()
          << ": we don't currently support deletion of the current loop.\n");
      return false;
    }

    // TODO: Support deletion of dead loop blocks.
    if (!DeadLoopBlocks.empty()) {
      LLVM_DEBUG(dbgs() << "Give up constant terminator folding in loop "
                        << L.getHeader()->getName()
                        << ": we don't currently"
                           " support deletion of dead in-loop blocks.\n");
      return false;
    }

    // TODO: Support dead loop exits.
    if (!DeadExitBlocks.empty()) {
      LLVM_DEBUG(dbgs() << "Give up constant terminator folding in loop "
                        << L.getHeader()->getName()
                        << ": we don't currently support dead loop exits.\n");
      return false;
    }

    // TODO: Support blocks that are not dead, but also not in loop after the
    // folding.
    if (BlocksInLoopAfterFolding.size() != L.getNumBlocks()) {
      LLVM_DEBUG(
          dbgs() << "Give up constant terminator folding in loop "
                 << L.getHeader()->getName()
                 << ": we don't currently"
                    " support blocks that are not dead, but will stop "
                    "being a part of the loop after constant-folding.\n");
      return false;
    }

    // Dump analysis results.
    LLVM_DEBUG(dump());

    LLVM_DEBUG(dbgs() << "Constant-folding " << FoldCandidates.size()
                      << " terminators in loop " << L.getHeader()->getName()
                      << "\n");

    // Make the actual transforms.
    foldTerminators();

#ifndef NDEBUG
    // Make sure that we have preserved all data structures after the transform.
    DT.verify();
    assert(DT.isReachableFromEntry(L.getHeader()));
    LI.verify(DT);
#endif

    return true;
  }
};

/// Turn branches and switches with known constant conditions into unconditional
/// branches.
static bool constantFoldTerminators(Loop &L, DominatorTree &DT, LoopInfo &LI,
                                    MemorySSAUpdater *MSSAU) {
  if (!EnableTermFolding)
    return false;

  // To keep things simple, only process loops with single latch. We
  // canonicalize most loops to this form. We can support multi-latch if needed.
  if (!L.getLoopLatch())
    return false;

  ConstantTerminatorFoldingImpl BranchFolder(L, LI, DT, MSSAU);
  return BranchFolder.run();
}

static bool mergeBlocksIntoPredecessors(Loop &L, DominatorTree &DT,
                                        LoopInfo &LI, MemorySSAUpdater *MSSAU) {
  bool Changed = false;
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
  // Copy blocks into a temporary array to avoid iterator invalidation issues
  // as we remove them.
  SmallVector<WeakTrackingVH, 16> Blocks(L.blocks());

  for (auto &Block : Blocks) {
    // Attempt to merge blocks in the trivial case. Don't modify blocks which
    // belong to other loops.
    BasicBlock *Succ = cast_or_null<BasicBlock>(Block);
    if (!Succ)
      continue;

    BasicBlock *Pred = Succ->getSinglePredecessor();
    if (!Pred || !Pred->getSingleSuccessor() || LI.getLoopFor(Pred) != &L)
      continue;

    // Merge Succ into Pred and delete it.
    MergeBlockIntoPredecessor(Succ, &DTU, &LI, MSSAU);

    Changed = true;
  }

  return Changed;
}

static bool simplifyLoopCFG(Loop &L, DominatorTree &DT, LoopInfo &LI,
                            ScalarEvolution &SE, MemorySSAUpdater *MSSAU) {
  bool Changed = false;

  // Constant-fold terminators with known constant conditions.
  Changed |= constantFoldTerminators(L, DT, LI, MSSAU);

  // Eliminate unconditional branches by merging blocks into their predecessors.
  Changed |= mergeBlocksIntoPredecessors(L, DT, LI, MSSAU);

  if (Changed)
    SE.forgetTopmostLoop(&L);

  return Changed;
}

PreservedAnalyses LoopSimplifyCFGPass::run(Loop &L, LoopAnalysisManager &AM,
                                           LoopStandardAnalysisResults &AR,
                                           LPMUpdater &) {
  Optional<MemorySSAUpdater> MSSAU;
  if (EnableMSSALoopDependency && AR.MSSA)
    MSSAU = MemorySSAUpdater(AR.MSSA);
  if (!simplifyLoopCFG(L, AR.DT, AR.LI, AR.SE,
                       MSSAU.hasValue() ? MSSAU.getPointer() : nullptr))
    return PreservedAnalyses::all();

  return getLoopPassPreservedAnalyses();
}

namespace {
class LoopSimplifyCFGLegacyPass : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid
  LoopSimplifyCFGLegacyPass() : LoopPass(ID) {
    initializeLoopSimplifyCFGLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnLoop(Loop *L, LPPassManager &) override {
    if (skipLoop(L))
      return false;

    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    Optional<MemorySSAUpdater> MSSAU;
    if (EnableMSSALoopDependency) {
      MemorySSA *MSSA = &getAnalysis<MemorySSAWrapperPass>().getMSSA();
      MSSAU = MemorySSAUpdater(MSSA);
      if (VerifyMemorySSA)
        MSSA->verifyMemorySSA();
    }
    return simplifyLoopCFG(*L, DT, LI, SE,
                           MSSAU.hasValue() ? MSSAU.getPointer() : nullptr);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    if (EnableMSSALoopDependency) {
      AU.addRequired<MemorySSAWrapperPass>();
      AU.addPreserved<MemorySSAWrapperPass>();
    }
    AU.addPreserved<DependenceAnalysisWrapperPass>();
    getLoopAnalysisUsage(AU);
  }
};
}

char LoopSimplifyCFGLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LoopSimplifyCFGLegacyPass, "loop-simplifycfg",
                      "Simplify loop CFG", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_END(LoopSimplifyCFGLegacyPass, "loop-simplifycfg",
                    "Simplify loop CFG", false, false)

Pass *llvm::createLoopSimplifyCFGPass() {
  return new LoopSimplifyCFGLegacyPass();
}
