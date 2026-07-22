//===- DeadBranchElimination.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates conditional branches that are unreachable but that
// cannot be proven unreachable directly, because the branch body modifies the
// very values its condition depends on. The motivating example:
//
//   int a = 0, b = 0, limit = 100;
//   while (a < limit) {
//     if (b == limit)   // Unreachable: a == b always, and a < limit here.
//       limit += 1;     // ...but it modifies limit (circular dependency).
//     a++; b++;
//   }
//
// Proving the branch dead requires knowing limit is loop-invariant, which
// requires knowing the branch is dead.
//
// The algorithm is an optimistic fixed point over the two bodies (true side,
// false side) of every conditional branch. Each body starts as Unknown
// ("assumed dead") and can only be promoted to ProvenReachable:
//
//   1. Clone the function. In the clone, replace every Unknown body with
//      'unreachable' (redirect the branch edge). Nested branches inside a
//      removed body disappear with it. PHI nodes at merge points drop the
//      incoming values of removed edges, so the bodies' effects vanish.
//   2. Run ScalarEvolution on the cleaned clone.
//   3. For each Unknown body, check whether the clone proves its branch edge
//      is never taken (context-sensitive SCEV proof, or the branch sits in
//      unreachable code). If the edge cannot be proven dead, mark it
//      ProvenReachable; its body is restored in the next iteration.
//   4. Repeat until no status changes. Statuses move in one direction only,
//      so this terminates.
//
// At convergence the remaining Unknown set is self-consistent: assuming
// those bodies never run, the analysis proves they indeed never run
// (consider the first time one would run -- the clone models the program
// state exactly up to that point and proves the edge is not taken). Those
// branches are then folded in the original function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/DeadBranchElimination.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "dead-branch-elim"

STATISTIC(NumBranchesFolded, "Number of provably dead branch edges folded");

static cl::opt<unsigned> MaxRefineIterations(
    "dbe-max-iterations", cl::init(8), cl::Hidden,
    cl::desc("Maximum fixed-point iterations per function before giving up"));

namespace {

enum class Status { Unknown, ProvenReachable };

/// One side (body) of a conditional branch.
struct BranchBody {
  BasicBlock *BranchBB; ///< Block whose terminator is the conditional branch.
  unsigned SuccIdx;     ///< Which side (0 = true body, 1 = false body).
  Status St = Status::Unknown;
};

/// Cloning and re-running ScalarEvolution is only worthwhile for the
/// pattern this pass targets: a branch inside a loop whose condition SCEV
/// can reason about. Everything else is seeded ProvenReachable so that
/// functions without such branches are never cloned at all. Straight-line
/// provably-dead branches are left to SCCP/SimplifyCFG.
SmallVector<BranchBody> collectBranchBodies(Function &F, LoopInfo &LI) {
  SmallVector<BranchBody> Bodies;
  for (BasicBlock &BB : F) {
    auto *BI = dyn_cast<CondBrInst>(BB.getTerminator());
    if (!BI)
      continue;
    if (BI->getSuccessor(0) == BI->getSuccessor(1))
      continue;
    Value *Cond = BI->getCondition();
    Status St = Status::Unknown;
    if (!LI.getLoopFor(&BB) || !isa<ICmpInst>(Cond))
      St = Status::ProvenReachable;
    Bodies.push_back({&BB, 0, St});
    Bodies.push_back({&BB, 1, St});
  }
  return Bodies;
}

/// Cheap prescan deciding whether collectBranchBodies can find any Unknown
/// candidate, before paying for a DominatorTree and LoopInfo.
bool hasCandidateShapedBranch(Function &F) {
  for (BasicBlock &BB : F)
    if (auto *BI = dyn_cast<CondBrInst>(BB.getTerminator()))
      if (BI->getSuccessor(0) != BI->getSuccessor(1) &&
          isa<ICmpInst>(BI->getCondition()))
        return true;
  return false;
}

/// Does the clone prove that this branch edge is never taken? The bodies of
/// all Unknown branches have already been replaced with 'unreachable', so
/// the condition is evaluated on the cleaned-up code.
bool isEdgeProvenDead(ScalarEvolution &SE, CondBrInst *BI, unsigned SuccIdx) {
  Value *Cond = BI->getCondition();
  if (auto *CI = dyn_cast<ConstantInt>(Cond))
    return CI->isOne() ? SuccIdx == 1 : SuccIdx == 0;
  if (auto *Cmp = dyn_cast<ICmpInst>(Cond)) {
    const SCEV *L = SE.getSCEV(Cmp->getOperand(0));
    const SCEV *R = SE.getSCEV(Cmp->getOperand(1));
    // The true side is dead when the condition is provably always false, the
    // false side when it is provably always true. The proof may use
    // conditions of dominating branches (e.g. the loop guard in an unrotated
    // while loop), so anchor the query at the branch itself.
    CmpPredicate P =
        SuccIdx == 0 ? Cmp->getInverseCmpPredicate() : Cmp->getCmpPredicate();
    return SE.isKnownPredicateAt(P, L, R, BI);
  }
  return false;
}

/// The analyses are computed on short-lived clone functions, so use a
/// private, uninstrumented analysis manager rather than the surrounding
/// pipeline's one: the clones must not show up in pass-manager debug logs,
/// and their cached results must die with them.
FunctionAnalysisManager makePrivateFAM() {
  FunctionAnalysisManager FAM;
  FAM.registerPass([] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([] { return TargetLibraryAnalysis(); });
  FAM.registerPass([] { return TargetIRAnalysis(); });
  FAM.registerPass([] { return AssumptionAnalysis(); });
  FAM.registerPass([] { return DominatorTreeAnalysis(); });
  FAM.registerPass([] { return LoopAnalysis(); });
  FAM.registerPass([] { return ScalarEvolutionAnalysis(); });
  return FAM;
}

/// One fixed-point iteration: rebuild the clone with all Unknown bodies
/// replaced by 'unreachable', re-run the analysis, and promote every body
/// whose edge cannot be proven dead. Returns true if any status changed.
bool refineOnce(Function &F, MutableArrayRef<BranchBody> Bodies) {
  ValueToValueMapTy VMap;
  Function *Clone = CloneFunction(&F, VMap);
  LLVMContext &Ctx = F.getContext();

  // Replace every Unknown body with 'unreachable': redirect the branch edge
  // to a trap block. The condition itself is never touched.
  BasicBlock *TrapBB = nullptr;
  for (BranchBody &B : Bodies) {
    if (B.St != Status::Unknown)
      continue;
    auto *BB = cast<BasicBlock>(VMap[B.BranchBB]);
    auto *BI = cast<CondBrInst>(BB->getTerminator());
    if (!TrapBB) {
      TrapBB = BasicBlock::Create(Ctx, "dbe.unreachable", Clone);
      new UnreachableInst(Ctx, TrapBB);
    }
    // Drop the PHI entries while the edge still exists, then redirect it.
    BasicBlock *Succ = BI->getSuccessor(B.SuccIdx);
    Succ->removePredecessor(BB);
    BI->setSuccessor(B.SuccIdx, TrapBB);
  }

  // A branch nested inside a removed body is itself unreachable in the
  // clone; it stays Unknown and is removed together with its parent.
  SmallPtrSet<BasicBlock *, 32> Reachable;
  for (BasicBlock *BB : depth_first(&Clone->getEntryBlock()))
    Reachable.insert(BB);
  SmallVector<std::pair<BranchBody *, BasicBlock *>> ToCheck;
  for (BranchBody &B : Bodies)
    if (B.St == Status::Unknown) {
      auto *BB = cast<BasicBlock>(VMap[B.BranchBB]);
      if (Reachable.contains(BB))
        ToCheck.push_back({&B, BB});
    }

  // Delete the unreachable blocks so PHI nodes in live blocks drop their
  // dead incoming values. DeleteDeadBlocks (unlike removeUnreachableBlocks)
  // never rewrites live terminators, so the branches under test survive.
  SmallVector<BasicBlock *> DeadBlocks;
  for (BasicBlock &BB : *Clone)
    if (!Reachable.contains(&BB))
      DeadBlocks.push_back(&BB);
  DeleteDeadBlocks(DeadBlocks);

  bool Changed = false;
  {
    FunctionAnalysisManager FAM = makePrivateFAM();
    auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(*Clone);
    for (auto &[B, BB] : ToCheck) {
      auto *BI = dyn_cast<CondBrInst>(BB->getTerminator());
      if (!BI) {
        // Something rewrote the branch under test; assume reachable.
        B->St = Status::ProvenReachable;
        Changed = true;
        continue;
      }
      if (!isEdgeProvenDead(SE, BI, B->SuccIdx)) {
        B->St = Status::ProvenReachable;
        Changed = true;
        LLVM_DEBUG(dbgs() << "DBE: promote " << B->BranchBB->getName() << "/"
                          << B->SuccIdx << "\n");
      } else
        LLVM_DEBUG(dbgs() << "DBE: still-dead " << B->BranchBB->getName() << "/"
                          << B->SuccIdx << "\n");
    }
  }

  Clone->eraseFromParent();
  return Changed;
}

/// Redirect each dead branch to its other side, then delete whatever became
/// unreachable. Returns true if anything changed.
bool foldDeadBranches(Function &F, ArrayRef<BranchBody> Dead,
                      OptimizationRemarkEmitter &ORE) {
  bool Changed = false;
  for (const BranchBody &B : Dead) {
    auto *BI = dyn_cast<CondBrInst>(B.BranchBB->getTerminator());
    if (!BI)
      continue; // Already folded together with a parent body.
    LLVM_DEBUG(dbgs() << "DBE: folding dead edge " << B.BranchBB->getName()
                      << " -> " << BI->getSuccessor(B.SuccIdx)->getName()
                      << " in " << F.getName() << "\n");
    ORE.emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "DeadBranchFolded", BI)
             << "removed branch to provably unreachable code";
    });
    ++NumBranchesFolded;
    Value *Cond = BI->getCondition();
    BI->setCondition(ConstantInt::getBool(F.getContext(), B.SuccIdx == 1));
    ConstantFoldTerminator(B.BranchBB);
    RecursivelyDeleteTriviallyDeadInstructions(Cond);
    Changed = true;
  }
  if (Changed)
    removeUnreachableBlocks(F);
  return Changed;
}

bool runOnFunction(Function &F) {
  if (!hasCandidateShapedBranch(F))
    return false;

  DominatorTree DT(F);
  LoopInfo LI(DT);
  SmallVector<BranchBody> Bodies = collectBranchBodies(F, LI);
  if (Bodies.empty())
    return false;

  auto HasUnknown = [&] {
    return any_of(Bodies,
                  [](const BranchBody &B) { return B.St == Status::Unknown; });
  };
  unsigned Iterations = 0;
  bool StatusChanged = true;
  while (HasUnknown() && StatusChanged) {
    // The fold below is only sound at a verified fixed point: an iteration
    // that promoted bodies invalidates the proofs of the remaining Unknown
    // set. If the cap cuts the loop short, give up on this function.
    if (++Iterations > MaxRefineIterations)
      return false;
    StatusChanged = refineOnce(F, Bodies);
  }

  SmallVector<BranchBody> Dead;
  for (BranchBody &B : Bodies)
    if (B.St == Status::Unknown)
      Dead.push_back(B);
  if (Dead.empty())
    return false;

  OptimizationRemarkEmitter ORE(&F);
  return foldDeadBranches(F, Dead, ORE);
}

} // namespace

PreservedAnalyses DeadBranchEliminationPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  bool Changed = false;
  // This is a module pass (it temporarily creates function clones), but it
  // transforms functions independently.
  for (Function &F : M) {
    if (F.isDeclaration() || F.isPresplitCoroutine() || F.hasOptNone())
      continue;
    Changed |= runOnFunction(F);
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
