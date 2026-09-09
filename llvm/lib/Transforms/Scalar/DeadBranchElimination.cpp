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
//   1. Materialize the assumption set in place (no cloning): PHI slots fed
//      by assumed-dead edges or assumed-unreachable regions are temporarily
//      overwritten with the surviving values (see AssumedDeadEdges), so the
//      assumed-dead bodies' effects vanish from the analysis.
//   2. Run ScalarEvolution on the function in this state.
//   3. For each Unknown body, check whether the analysis proves its branch
//      edge is never taken (context-sensitive SCEV proof, or the branch
//      sits in an assumed-dead region). If the edge cannot be proven dead,
//      mark it ProvenReachable; its body is restored for the next
//      iteration.
//   4. Undo the PHI rewrites exactly and repeat until no status changes.
//      Statuses move in one direction only, so this terminates.
//
// At convergence the remaining Unknown set is self-consistent: assuming
// those bodies never run, the analysis proves they indeed never run
// (consider the first time one would run -- up to that point the rewritten
// values match the real execution, and the analysis proves the edge is not
// taken). Those branches are then folded.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/DeadBranchElimination.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

/// The fixed point is only worthwhile for the pattern this pass targets: a
/// branch inside a loop whose condition SCEV can reason about. Everything
/// else is seeded ProvenReachable so that functions without such branches
/// are never analyzed at all. Straight-line provably-dead branches are left
/// to SCCP/SimplifyCFG.
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

/// Is this branch edge provably never taken? The assumption set is already
/// applied (see AssumedDeadEdges), so the condition is evaluated as if all
/// still-Unknown bodies were dead.
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

/// The analyses run on the function while the assumption set is applied, so
/// use a private, uninstrumented analysis manager rather than the
/// surrounding pipeline's one: results computed in the assumed state must
/// never leak into the pipeline's cache, and the extra runs must not show
/// up in pass-manager debug logs.
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

/// Materializes "assume these edges are never taken" directly on the
/// function, without cloning it. The CFG is never touched -- the whole
/// effect of a dead body is expressed at the value level:
///
///  - Virtual reachability: a DFS from the entry that skips assumed-dead
///    edges determines which blocks the assumptions keep alive.
///
///  - PHI slot rewriting: in every live block, a PHI slot whose edge is
///    assumed dead (or whose predecessor is virtually unreachable) is
///    overwritten with a surviving value, and live slots are replaced by
///    their resolved value (see resolve()). This is what makes the assumed
///    dead bodies' side effects vanish: the merge PHI of the motivating
///    example turns into phi [100, ...], [100, ...] and ScalarEvolution
///    sees the loop limit as the constant it really is.
///
/// Every write is journaled and undone in reverse in the destructor, so the
/// function is restored exactly (including PHI operand order). While the
/// assumptions are applied the function must only be inspected by analyses,
/// never verified or transformed.
class AssumedDeadEdges {
public:
  AssumedDeadEdges(Function &F, ArrayRef<BranchBody> Bodies) {
    for (const BranchBody &B : Bodies)
      if (B.St == Status::Unknown)
        DeadEdges.insert({B.BranchBB, B.SuccIdx});

    // Virtual reachability: DFS that does not follow assumed-dead edges.
    SmallVector<BasicBlock *> Worklist{&F.getEntryBlock()};
    Reachable.insert(&F.getEntryBlock());
    while (!Worklist.empty()) {
      BasicBlock *BB = Worklist.pop_back_val();
      Instruction *T = BB->getTerminator();
      for (unsigned I = 0, E = T->getNumSuccessors(); I != E; ++I) {
        if (DeadEdges.contains({BB, I}))
          continue;
        if (Reachable.insert(T->getSuccessor(I)).second)
          Worklist.push_back(T->getSuccessor(I));
      }
    }

    // Rewrite PHI slots in live blocks. Compute all new values first so the
    // resolver only ever sees original operands.
    SmallVector<std::tuple<PHINode *, unsigned, Value *>> Rewrites;
    for (BasicBlock &BB : F) {
      if (!Reachable.contains(&BB))
        continue;
      for (PHINode &PN : BB.phis()) {
        Value *DeadFill = nullptr;
        for (unsigned I = 0, E = PN.getNumIncomingValues(); I != E; ++I)
          if (isLiveSlot(PN, I)) {
            DeadFill = resolveTopLevel(PN.getIncomingValue(I));
            break;
          }
        assert(DeadFill && "live block with no live PHI slot");
        for (unsigned I = 0, E = PN.getNumIncomingValues(); I != E; ++I) {
          Value *NewV = isLiveSlot(PN, I)
                            ? resolveTopLevel(PN.getIncomingValue(I))
                            : DeadFill;
          if (NewV != PN.getIncomingValue(I))
            Rewrites.push_back({&PN, I, NewV});
        }
      }
    }
    for (auto &[PN, Idx, NewV] : Rewrites) {
      Journal.push_back({PN, Idx, PN->getIncomingValue(Idx)});
      PN->setIncomingValue(Idx, NewV);
    }
  }

  ~AssumedDeadEdges() {
    for (auto &[PN, Idx, OldV] : reverse(Journal))
      PN->setIncomingValue(Idx, OldV);
  }

  bool isReachable(BasicBlock *BB) const { return Reachable.contains(BB); }

private:
  /// A PHI slot is live when its predecessor is virtually reachable and the
  /// edge it flows along is not assumed dead.
  bool isLiveSlot(const PHINode &PN, unsigned Idx) const {
    BasicBlock *Pred = PN.getIncomingBlock(Idx);
    if (!Reachable.contains(Pred))
      return false;
    Instruction *T = Pred->getTerminator();
    for (unsigned I = 0, E = T->getNumSuccessors(); I != E; ++I)
      if (T->getSuccessor(I) == PN.getParent() && DeadEdges.contains({Pred, I}))
        return false;
    return true;
  }

  /// Resolve a value under the assumptions: look through PHI chains whose
  /// live inputs all agree. A reference back into the PHI strongly-connected
  /// component under resolution contributes no value (the generalization of
  /// InstSimplify's "ignore self references" rule): a PHI-SCC holds a single
  /// value X iff all inputs entering the SCC are X. Only top-level results
  /// are memoized; intermediate results computed with an incomplete view of
  /// the SCC would not be valid on their own.
  Value *resolveTopLevel(Value *V) {
    Value *R = resolveImpl(V);
    if (auto *PN = dyn_cast<PHINode>(V))
      Memo[PN] = R;
    return R;
  }

  Value *resolveImpl(Value *V) {
    auto *PN = dyn_cast<PHINode>(V);
    if (!PN || !Reachable.contains(PN->getParent()))
      return V;
    if (auto It = Memo.find(PN); It != Memo.end())
      return It->second;
    if (!Visiting.insert(PN).second)
      return nullptr; // Cycle back into the SCC: contributes nothing.
    Value *Common = nullptr;
    bool Multiple = false;
    for (unsigned I = 0, E = PN->getNumIncomingValues(); I != E; ++I) {
      if (!isLiveSlot(*PN, I))
        continue;
      Value *R = resolveImpl(PN->getIncomingValue(I));
      if (!R)
        continue;
      if (!Common)
        Common = R;
      else if (Common != R) {
        Multiple = true;
        break;
      }
    }
    Visiting.erase(PN);
    return (Multiple || !Common) ? PN : Common;
  }

  DenseSet<std::pair<BasicBlock *, unsigned>> DeadEdges;
  SmallPtrSet<BasicBlock *, 32> Reachable;
  DenseMap<PHINode *, Value *> Memo;
  SmallPtrSet<PHINode *, 8> Visiting;
  SmallVector<std::tuple<PHINode *, unsigned, Value *>> Journal;
};

/// One fixed-point iteration: apply the current assumption set in place,
/// re-run the analysis, and promote every body whose edge cannot be proven
/// dead. Returns true if any status changed.
bool refineOnce(Function &F, MutableArrayRef<BranchBody> Bodies) {
  AssumedDeadEdges Assumed(F, Bodies);

  bool Changed = false;
  {
    FunctionAnalysisManager FAM = makePrivateFAM();
    auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
    for (BranchBody &B : Bodies) {
      // A branch nested inside an assumed-dead region stays Unknown; it is
      // removed together with its parent.
      if (B.St != Status::Unknown || !Assumed.isReachable(B.BranchBB))
        continue;
      auto *BI = cast<CondBrInst>(B.BranchBB->getTerminator());
      if (!isEdgeProvenDead(SE, BI, B.SuccIdx)) {
        B.St = Status::ProvenReachable;
        Changed = true;
        LLVM_DEBUG(dbgs() << "DBE: promote " << B.BranchBB->getName() << "/"
                          << B.SuccIdx << "\n");
      } else
        LLVM_DEBUG(dbgs() << "DBE: still-dead " << B.BranchBB->getName() << "/"
                          << B.SuccIdx << "\n");
    }
  }

  // The journal in Assumed restores the exact original IR on destruction.
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
  for (Function &F : M) {
    if (F.isDeclaration() || F.isPresplitCoroutine() || F.hasOptNone())
      continue;
    Changed |= runOnFunction(F);
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
