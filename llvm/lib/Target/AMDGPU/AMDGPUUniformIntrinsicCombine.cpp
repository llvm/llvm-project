//===-- AMDGPUUniformIntrinsicCombine.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass simplifies certain intrinsic calls when the arguments are uniform.
/// It's true that this pass has transforms that can lead to a situation where
/// some instruction whose operand was previously recognized as statically
/// uniform is later on no longer recognized as statically uniform. However, the
/// semantics of how programs execute don't (and must not, for this precise
/// reason) care about static uniformity, they only ever care about dynamic
/// uniformity. And every instruction that's downstream and cares about dynamic
/// uniformity must be convergent (and isel will introduce v_readfirstlane for
/// them if their operands can't be proven statically uniform).
///
/// The pass additionally performs a convergence-aware CSE of
/// llvm.amdgcn.ballot. Because ballot is convergent, the generic CSE passes
/// refuse to merge two calls that live in different basic blocks: the result
/// implicitly depends on the set of currently active lanes (exec). Here we can
/// do better, because uniformity analysis lets us prove that exec is unchanged
/// between two identical calls, in which case the later one is redundant.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "amdgpu-uniform-intrinsic-combine"

using namespace llvm;
using namespace llvm::AMDGPU;
using namespace llvm::PatternMatch;

/// Wrapper for querying uniformity info that first checks locally tracked
/// instructions.
static bool
isDivergentUseWithNew(const Use &U, const UniformityInfo &UI,
                      const ValueMap<const Value *, bool> &Tracker) {
  Value *V = U.get();
  if (auto It = Tracker.find(V); It != Tracker.end())
    return !It->second; // divergent if marked false
  return UI.isDivergentAtUse(U);
}

/// Optimizes uniform intrinsics calls if their operand can be proven uniform.
static bool optimizeUniformIntrinsic(IntrinsicInst &II,
                                     const UniformityInfo &UI,
                                     ValueMap<const Value *, bool> &Tracker) {
  llvm::Intrinsic::ID IID = II.getIntrinsicID();
  /// We deliberately do not simplify readfirstlane with a uniform argument, so
  /// that frontends can use it to force a copy to SGPR and thereby prevent the
  /// backend from generating unwanted waterfall loops.
  switch (IID) {
  case Intrinsic::amdgcn_permlane64:
  case Intrinsic::amdgcn_readlane: {
    Value *Src = II.getArgOperand(0);
    if (isDivergentUseWithNew(II.getOperandUse(0), UI, Tracker))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing " << II << " with " << *Src << '\n');
    II.replaceAllUsesWith(Src);
    II.eraseFromParent();
    return true;
  }
  case Intrinsic::amdgcn_ballot: {
    Value *Src = II.getArgOperand(0);
    if (isDivergentUseWithNew(II.getOperandUse(0), UI, Tracker))
      return false;
    LLVM_DEBUG(dbgs() << "Found uniform ballot intrinsic: " << II << '\n');

    bool Changed = false;
    for (User *U : make_early_inc_range(II.users())) {
      if (auto *ICmp = dyn_cast<ICmpInst>(U)) {
        Value *Op0 = ICmp->getOperand(0);
        Value *Op1 = ICmp->getOperand(1);
        ICmpInst::Predicate Pred = ICmp->getPredicate();
        Value *OtherOp = Op0 == &II ? Op1 : Op0;

        if (Pred == ICmpInst::ICMP_EQ && match(OtherOp, m_Zero())) {
          // Case: (icmp eq %ballot, 0) -> xor %ballot_arg, 1
          Instruction *NotOp =
              BinaryOperator::CreateNot(Src, "", ICmp->getIterator());
          Tracker[NotOp] = true; // NOT preserves uniformity
          LLVM_DEBUG(dbgs() << "Replacing ICMP_EQ: " << *NotOp << '\n');
          ICmp->replaceAllUsesWith(NotOp);
          Changed = true;
        } else if (Pred == ICmpInst::ICMP_NE && match(OtherOp, m_Zero())) {
          // Case: (icmp ne %ballot, 0) -> %ballot_arg
          LLVM_DEBUG(dbgs() << "Replacing ICMP_NE with ballot argument: "
                            << *Src << '\n');
          ICmp->replaceAllUsesWith(Src);
          Changed = true;
        }
      }
    }
    // Erase the intrinsic if it has no remaining uses.
    if (II.use_empty())
      II.eraseFromParent();
    return Changed;
  }
  case Intrinsic::amdgcn_wave_shuffle: {
    Use &Val = II.getOperandUse(0);
    Use &Idx = II.getOperandUse(1);

    // Like with readlane, if Value is uniform then just propagate it
    if (!isDivergentUseWithNew(Val, UI, Tracker)) {
      II.replaceAllUsesWith(Val);
      II.eraseFromParent();
      return true;
    }

    // Otherwise, when Index is uniform, this is just a readlane operation
    if (isDivergentUseWithNew(Idx, UI, Tracker))
      return false;

    // The readlane intrinsic we want to call has the exact same function
    // signature, so we can quickly modify the instruction in-place
    Module *Mod = II.getModule();
    II.setCalledFunction(Intrinsic::getOrInsertDeclaration(
        Mod, Intrinsic::amdgcn_readlane, II.getType()));
    return true;
  }
  default:
    return false;
  }
  return false;
}

/// Maximum number of basic blocks inspected while proving that exec is
/// invariant between two ballots. Keeps the walk below linear-per-pair in
/// pathological CFGs.
static constexpr unsigned MaxExecInvarianceBlocks = 100;

/// Returns true if \p I may change exec, i.e. the set of lanes that are active
/// when the following instructions execute.
static bool isExecModifyingInst(const Instruction &I) {
  const auto *CB = dyn_cast<CallBase>(&I);
  if (!CB)
    return false;

  switch (CB->getIntrinsicID()) {
  case Intrinsic::amdgcn_kill:
  case Intrinsic::amdgcn_wqm_demote:
  case Intrinsic::amdgcn_init_exec:
  case Intrinsic::amdgcn_init_exec_from_input:
  case Intrinsic::amdgcn_init_whole_wave:
  // The WQM/WWM family makes the exec state at a given point depend on
  // WQM/Exact decisions that SIWholeQuadMode only makes much later, so treat
  // any occurrence of it as opaque.
  case Intrinsic::amdgcn_wqm:
  case Intrinsic::amdgcn_softwqm:
  case Intrinsic::amdgcn_strict_wqm:
  case Intrinsic::amdgcn_wwm:
  case Intrinsic::amdgcn_strict_wwm:
  case Intrinsic::amdgcn_set_inactive:
  case Intrinsic::amdgcn_set_inactive_chain_arg:
    return true;
  case Intrinsic::not_intrinsic:
    // A callee may itself execute llvm.amdgcn.kill.
    return true;
  default:
    return false;
  }
}

/// Returns true if exec is provably the same at \p A and at \p B, given that
/// \p A dominates \p B. That holds when no path from \p A to \p B crosses a
/// divergent terminator or an instruction that writes exec.
static bool isExecInvariantBetween(const Instruction *A, const Instruction *B,
                                   const UniformityInfo &UI) {
  const BasicBlock *ABB = A->getParent();
  const BasicBlock *BBB = B->getParent();

  auto HasExecModifier = [](BasicBlock::const_iterator Begin,
                            BasicBlock::const_iterator End) {
    return any_of(make_range(Begin, End), isExecModifyingInst);
  };

  // Straight-line case: only the instructions in between can matter. Note that
  // if this block is part of a cycle then A re-executes before B does, so the
  // two still pair up within an iteration.
  if (ABB == BBB)
    return !HasExecModifier(std::next(A->getIterator()), B->getIterator());

  // Everything in BBB ahead of B runs between A and B.
  if (HasExecModifier(BBB->begin(), B->getIterator()))
    return false;

  // Collect every block on some path ABB ->* BBB that does not re-enter ABB.
  // Since A dominates B, every such path starts at ABB, and every block found
  // this way is dominated by ABB. Stopping the walk at ABB is correct: if a
  // path did revisit ABB, then a later dynamic instance of A would be the one
  // reaching B, and the path from that instance does not revisit ABB.
  SmallPtrSet<const BasicBlock *, 8> Region;
  SmallVector<const BasicBlock *, 8> Worklist;
  Region.insert(ABB);
  for (const BasicBlock *Pred : predecessors(BBB)) {
    if (Region.insert(Pred).second)
      Worklist.push_back(Pred);
  }

  while (!Worklist.empty()) {
    const BasicBlock *Cur = Worklist.pop_back_val();
    if (Cur == ABB)
      continue;
    for (const BasicBlock *Pred : predecessors(Cur))
      if (Region.insert(Pred).second)
        Worklist.push_back(Pred);
  }

  if (Region.size() > MaxExecInvarianceBlocks)
    return false;

  // Note that BBB lands in the region only if it can reach itself without
  // passing through ABB, i.e. B sits in a cycle that A is outside of. In that
  // case B re-executes and its whole block, terminator included, is checked
  // below; otherwise BBB is absent and its terminator, which runs after B, is
  // correctly left out.
  for (const BasicBlock *Blk : Region) {
    if (!UI.isUniformTerminator(Blk->getTerminator()))
      return false;

    // Instructions ahead of A never run between the last A and B.
    BasicBlock::const_iterator Begin =
        Blk == ABB ? std::next(A->getIterator()) : Blk->begin();
    if (HasExecModifier(Begin, Blk->end()))
      return false;
  }
  return true;
}

/// Removes ballot calls that are made redundant by an earlier identical call
/// which is guaranteed to have executed with the same exec mask.
static bool combineRedundantBallots(Function &F, UniformityInfo &UI,
                                    const DominatorTree &DT) {
  // Bucket the ballots by (result type, condition); only calls landing in the
  // same bucket can possibly be identical. Visiting the dominator tree in
  // pre-order means a dominating call always precedes the calls it dominates.
  SmallMapVector<std::pair<Type *, Value *>, SmallVector<CallInst *, 4>, 4>
      Buckets;
  for (const DomTreeNode *N : depth_first(DT.getRootNode())) {
    for (Instruction &I : *N->getBlock()) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (CI && CI->getIntrinsicID() == Intrinsic::amdgcn_ballot)
        Buckets[{CI->getType(), CI->getArgOperand(0)}].push_back(CI);
    }
  }

  bool Changed = false;
  for (auto &[Key, Ballots] : Buckets) {
    for (unsigned I = 1, E = Ballots.size(); I < E; ++I) {
      CallInst *B = Ballots[I];
      for (unsigned J = 0; J < I; ++J) {
        CallInst *A = Ballots[J];
        // Null entries are calls that have already been erased.
        if (!A || !A->isIdenticalToWhenDefined(B) || !DT.dominates(A, B) ||
            !isExecInvariantBetween(A, B, UI))
          continue;

        LLVM_DEBUG(dbgs() << "Replacing redundant ballot " << *B << " with "
                          << *A << '\n');
        UI.forgetValue(B);
        B->replaceAllUsesWith(A);
        B->eraseFromParent();
        Ballots[I] = nullptr;
        Changed = true;
        break;
      }
    }
  }
  return Changed;
}

/// Iterates over intrinsic calls in the Function to optimize.
static bool runUniformIntrinsicCombine(Function &F, UniformityInfo &UI,
                                       const DominatorTree &DT) {
  bool IsChanged = false;
  ValueMap<const Value *, bool> Tracker;

  for (Instruction &I : make_early_inc_range(instructions(F))) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    IsChanged |= optimizeUniformIntrinsic(*II, UI, Tracker);
  }

  IsChanged |= combineRedundantBallots(F, UI, DT);

  return IsChanged;
}

PreservedAnalyses
AMDGPUUniformIntrinsicCombinePass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  auto &UI = AM.getResult<UniformityInfoAnalysis>(F);
  const auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  if (!runUniformIntrinsicCombine(F, UI, DT))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<UniformityInfoAnalysis>();
  return PA;
}

namespace {
class AMDGPUUniformIntrinsicCombineLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPUUniformIntrinsicCombineLegacy() : FunctionPass(ID) {}

private:
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetPassConfig>();
  }
};
} // namespace

char AMDGPUUniformIntrinsicCombineLegacy::ID = 0;
char &llvm::AMDGPUUniformIntrinsicCombineLegacyPassID =
    AMDGPUUniformIntrinsicCombineLegacy::ID;

bool AMDGPUUniformIntrinsicCombineLegacy::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  UniformityInfo &UI =
      getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
  const DominatorTree &DT =
      getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  return runUniformIntrinsicCombine(F, UI, DT);
}

INITIALIZE_PASS_BEGIN(AMDGPUUniformIntrinsicCombineLegacy, DEBUG_TYPE,
                      "AMDGPU Uniform Intrinsic Combine", false, false)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUUniformIntrinsicCombineLegacy, DEBUG_TYPE,
                    "AMDGPU Uniform Intrinsic Combine", false, false)

FunctionPass *llvm::createAMDGPUUniformIntrinsicCombineLegacyPass() {
  return new AMDGPUUniformIntrinsicCombineLegacy();
}
