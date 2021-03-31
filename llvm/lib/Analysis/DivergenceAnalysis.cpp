//===---- DivergenceAnalysis.cpp --- Divergence Analysis Implementation ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a general divergence analysis for loop vectorization
// and GPU programs. It determines which branches and values in a loop or GPU
// program are divergent. It can help branch optimizations such as jump
// threading and loop unswitching to make better decisions.
//
// GPU programs typically use the SIMD execution model, where multiple threads
// in the same execution group have to execute in lock-step. Therefore, if the
// code contains divergent branches (i.e., threads in a group do not agree on
// which path of the branch to take), the group of threads has to execute all
// the paths from that branch with different subsets of threads enabled until
// they re-converge.
//
// Due to this execution model, some optimizations such as jump
// threading and loop unswitching can interfere with thread re-convergence.
// Therefore, an analysis that computes which branches in a GPU program are
// divergent can help the compiler to selectively run these optimizations.
//
// This implementation is derived from the Vectorization Analysis of the
// Region Vectorizer (RV). That implementation in turn is based on the approach
// described in
//
//   Improving Performance of OpenCL on CPUs
//   Ralf Karrenberg and Sebastian Hack
//   CC '12
//
// This implementation is generic in the sense that it does
// not itself identify original sources of divergence.
// Instead specialized adapter classes, (LoopDivergenceAnalysis) for loops and
// (DivergenceAnalysis) for functions, identify the sources of divergence
// (e.g., special variables that hold the thread ID or the iteration variable).
//
// The generic implementation propagates divergence to variables that are data
// or sync dependent on a source of divergence.
//
// While data dependency is a well-known concept, the notion of sync dependency
// is worth more explanation. Sync dependence characterizes the control flow
// aspect of the propagation of branch divergence. For example,
//
//   %cond = icmp slt i32 %tid, 10
//   br i1 %cond, label %then, label %else
// then:
//   br label %merge
// else:
//   br label %merge
// merge:
//   %a = phi i32 [ 0, %then ], [ 1, %else ]
//
// Suppose %tid holds the thread ID. Although %a is not data dependent on %tid
// because %tid is not on its use-def chains, %a is sync dependent on %tid
// because the branch "br i1 %cond" depends on %tid and affects which value %a
// is assigned to.
//
// The sync dependence detection (which branch induces divergence in which join
// points) is implemented in the SyncDependenceAnalysis.
//
// The current implementation has the following limitations:
// 1. intra-procedural. It conservatively considers the arguments of a
//    non-kernel-entry function and the return value of a function call as
//    divergent.
// 2. memory as black box. It conservatively considers values loaded from
//    generic or local address as divergent. This can be improved by leveraging
//    pointer analysis and/or by modelling non-escaping memory objects in SSA
//    as done in RV.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "divergence"

DivergenceAnalysisImpl::DivergenceAnalysisImpl(
    const Function &F, const Loop *RegionLoop, const DominatorTree &DT,
    const LoopInfo &LI, SyncDependenceAnalysis &SDA, bool IsLCSSAForm)
    : F(F), RegionLoop(RegionLoop), DT(DT), LI(LI), SDA(SDA),
      IsLCSSAForm(IsLCSSAForm) {}

bool DivergenceAnalysisImpl::markDivergent(const Value &DivVal) {
  if (isAlwaysUniform(DivVal))
    return false;
  assert(isa<Instruction>(DivVal) || isa<Argument>(DivVal));
  assert(!isAlwaysUniform(DivVal) && "cannot be a divergent");
  return DivergentValues.insert(&DivVal).second;
}

void DivergenceAnalysisImpl::addUniformOverride(const Value &UniVal) {
  UniformOverrides.insert(&UniVal);
}

bool DivergenceAnalysisImpl::isTemporalDivergent(
    const BasicBlock &ObservingBlock, const Value &Val) const {
  const auto *Inst = dyn_cast<const Instruction>(&Val);
  if (!Inst)
    return false;
  // check whether any divergent loop carrying Val terminates before control
  // proceeds to ObservingBlock
  for (const auto *Loop = LI.getLoopFor(Inst->getParent());
       Loop != RegionLoop && !Loop->contains(&ObservingBlock);
       Loop = Loop->getParentLoop()) {
    if (DivergentLoops.contains(Loop))
      return true;
  }

  return false;
}

bool DivergenceAnalysisImpl::inRegion(const Instruction &I) const {
  return I.getParent() && inRegion(*I.getParent());
}

bool DivergenceAnalysisImpl::inRegion(const BasicBlock &BB) const {
  return (!RegionLoop && BB.getParent() == &F) || RegionLoop->contains(&BB);
}

void DivergenceAnalysisImpl::pushUsers(const Value &V) {
  const auto *I = dyn_cast<const Instruction>(&V);

  if (I && I->isTerminator()) {
    analyzeControlDivergence(*I);
    return;
  }

  for (const auto *User : V.users()) {
    const auto *UserInst = dyn_cast<const Instruction>(User);
    if (!UserInst)
      continue;

    // only compute divergent inside loop
    if (!inRegion(*UserInst))
      continue;

    // All users of divergent values are immediate divergent
    if (markDivergent(*UserInst))
      Worklist.push_back(UserInst);
  }
}

static const Instruction *getIfCarriedInstruction(const Use &U,
                                                  const Loop &DivLoop) {
  const auto *I = dyn_cast<const Instruction>(&U);
  if (!I)
    return nullptr;
  if (!DivLoop.contains(I))
    return nullptr;
  return I;
}

void DivergenceAnalysisImpl::analyzeTemporalDivergence(
    const Instruction &I, const Loop &OuterDivLoop) {
  if (isAlwaysUniform(I))
    return;
  if (isDivergent(I))
    return;

  LLVM_DEBUG(dbgs() << "Analyze temporal divergence: " << I.getName() << "\n");
  assert((isa<PHINode>(I) || !IsLCSSAForm) &&
         "In LCSSA form all users of loop-exiting defs are Phi nodes.");
  for (const Use &Op : I.operands()) {
    const auto *OpInst = getIfCarriedInstruction(Op, OuterDivLoop);
    if (!OpInst)
      continue;
    if (markDivergent(I))
      pushUsers(I);
    return;
  }
}

// marks all users of loop-carried values of the loop headed by LoopHeader as
// divergent
void DivergenceAnalysisImpl::analyzeLoopExitDivergence(
    const BasicBlock &DivExit, const Loop &OuterDivLoop) {
  // All users are in immediate exit blocks
  if (IsLCSSAForm) {
    for (const auto &Phi : DivExit.phis()) {
      analyzeTemporalDivergence(Phi, OuterDivLoop);
    }
    return;
  }

  // For non-LCSSA we have to follow all live out edges wherever they may lead.
  const BasicBlock &LoopHeader = *OuterDivLoop.getHeader();
  SmallVector<const BasicBlock *, 8> TaintStack;
  TaintStack.push_back(&DivExit);

  // Otherwise potential users of loop-carried values could be anywhere in the
  // dominance region of DivLoop (including its fringes for phi nodes)
  DenseSet<const BasicBlock *> Visited;
  Visited.insert(&DivExit);

  do {
    auto *UserBlock = TaintStack.pop_back_val();

    // don't spread divergence beyond the region
    if (!inRegion(*UserBlock))
      continue;

    assert(!OuterDivLoop.contains(UserBlock) &&
           "irreducible control flow detected");

    // phi nodes at the fringes of the dominance region
    if (!DT.dominates(&LoopHeader, UserBlock)) {
      // all PHI nodes of UserBlock become divergent
      for (auto &Phi : UserBlock->phis()) {
        analyzeTemporalDivergence(Phi, OuterDivLoop);
      }
      continue;
    }

    // Taint outside users of values carried by OuterDivLoop.
    for (auto &I : *UserBlock) {
      analyzeTemporalDivergence(I, OuterDivLoop);
    }

    // visit all blocks in the dominance region
    for (auto *SuccBlock : successors(UserBlock)) {
      if (!Visited.insert(SuccBlock).second) {
        continue;
      }
      TaintStack.push_back(SuccBlock);
    }
  } while (!TaintStack.empty());
}

void DivergenceAnalysisImpl::propagateLoopExitDivergence(
    const BasicBlock &DivExit, const Loop &InnerDivLoop) {
  LLVM_DEBUG(dbgs() << "\tpropLoopExitDiv " << DivExit.getName() << "\n");

  // Find outer-most loop that does not contain \p DivExit
  const Loop *DivLoop = &InnerDivLoop;
  const Loop *OuterDivLoop = DivLoop;
  const Loop *ExitLevelLoop = LI.getLoopFor(&DivExit);
  const unsigned LoopExitDepth =
      ExitLevelLoop ? ExitLevelLoop->getLoopDepth() : 0;
  while (DivLoop && DivLoop->getLoopDepth() > LoopExitDepth) {
    DivergentLoops.insert(DivLoop); // all crossed loops are divergent
    OuterDivLoop = DivLoop;
    DivLoop = DivLoop->getParentLoop();
  }
  LLVM_DEBUG(dbgs() << "\tOuter-most left loop: " << OuterDivLoop->getName()
                    << "\n");

  analyzeLoopExitDivergence(DivExit, *OuterDivLoop);
}

// this is a divergent join point - mark all phi nodes as divergent and push
// them onto the stack.
void DivergenceAnalysisImpl::taintAndPushPhiNodes(const BasicBlock &JoinBlock) {
  LLVM_DEBUG(dbgs() << "taintAndPushPhiNodes in " << JoinBlock.getName()
                    << "\n");

  // ignore divergence outside the region
  if (!inRegion(JoinBlock)) {
    return;
  }

  // push non-divergent phi nodes in JoinBlock to the worklist
  for (const auto &Phi : JoinBlock.phis()) {
    if (isDivergent(Phi))
      continue;
    // FIXME Theoretically ,the 'undef' value could be replaced by any other
    // value causing spurious divergence.
    if (Phi.hasConstantOrUndefValue())
      continue;
    if (markDivergent(Phi))
      Worklist.push_back(&Phi);
  }
}

void DivergenceAnalysisImpl::analyzeControlDivergence(const Instruction &Term) {
  LLVM_DEBUG(dbgs() << "analyzeControlDiv " << Term.getParent()->getName()
                    << "\n");

  // Don't propagate divergence from unreachable blocks.
  if (!DT.isReachableFromEntry(Term.getParent()))
    return;

  const auto *BranchLoop = LI.getLoopFor(Term.getParent());

  const auto &DivDesc = SDA.getJoinBlocks(Term);

  // Iterate over all blocks now reachable by a disjoint path join
  for (const auto *JoinBlock : DivDesc.JoinDivBlocks) {
    taintAndPushPhiNodes(*JoinBlock);
  }

  assert(DivDesc.LoopDivBlocks.empty() || BranchLoop);
  for (const auto *DivExitBlock : DivDesc.LoopDivBlocks) {
    propagateLoopExitDivergence(*DivExitBlock, *BranchLoop);
  }
}

void DivergenceAnalysisImpl::compute() {
  // Initialize worklist.
  auto DivValuesCopy = DivergentValues;
  for (const auto *DivVal : DivValuesCopy) {
    assert(isDivergent(*DivVal) && "Worklist invariant violated!");
    pushUsers(*DivVal);
  }

  // All values on the Worklist are divergent.
  // Their users may not have been updated yed.
  while (!Worklist.empty()) {
    const Instruction &I = *Worklist.back();
    Worklist.pop_back();

    // propagate value divergence to users
    assert(isDivergent(I) && "Worklist invariant violated!");
    pushUsers(I);
  }
}

bool DivergenceAnalysisImpl::isAlwaysUniform(const Value &V) const {
  return UniformOverrides.contains(&V);
}

bool DivergenceAnalysisImpl::isDivergent(const Value &V) const {
  return DivergentValues.contains(&V);
}

bool DivergenceAnalysisImpl::isDivergentUse(const Use &U) const {
  Value &V = *U.get();
  Instruction &I = *cast<Instruction>(U.getUser());
  return isDivergent(V) || isTemporalDivergent(*I.getParent(), V);
}

DivergenceInfo::DivergenceInfo(Function &F, const DominatorTree &DT,
                               const PostDominatorTree &PDT, const LoopInfo &LI,
                               const TargetTransformInfo &TTI,
                               bool KnownReducible)
    : F(F), ContainsIrreducible(false) {
  if (!KnownReducible) {
    using RPOTraversal = ReversePostOrderTraversal<const Function *>;
    RPOTraversal FuncRPOT(&F);
    if (containsIrreducibleCFG<const BasicBlock *, const RPOTraversal,
                               const LoopInfo>(FuncRPOT, LI)) {
      ContainsIrreducible = true;
      return;
    }
  }
  SDA = std::make_unique<SyncDependenceAnalysis>(DT, PDT, LI);
  DA = std::make_unique<DivergenceAnalysisImpl>(F, nullptr, DT, LI, *SDA,
                                                /* LCSSA */ false);
  for (auto &I : instructions(F)) {
    if (TTI.isSourceOfDivergence(&I)) {
      DA->markDivergent(I);
    } else if (TTI.isAlwaysUniform(&I)) {
      DA->addUniformOverride(I);
    }
  }
  for (auto &Arg : F.args()) {
    if (TTI.isSourceOfDivergence(&Arg)) {
      DA->markDivergent(Arg);
    }
  }

  DA->compute();
}

AnalysisKey DivergenceAnalysis::Key;

DivergenceAnalysis::Result
DivergenceAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);

  return DivergenceInfo(F, DT, PDT, LI, TTI, /* KnownReducible = */ false);
}

PreservedAnalyses
DivergenceAnalysisPrinterPass::run(Function &F, FunctionAnalysisManager &FAM) {
  auto &DI = FAM.getResult<DivergenceAnalysis>(F);
  OS << "'Divergence Analysis' for function '" << F.getName() << "':\n";
  if (DI.hasDivergence()) {
    for (auto &Arg : F.args()) {
      OS << (DI.isDivergent(Arg) ? "DIVERGENT: " : "           ");
      OS << Arg << "\n";
    }
    for (const BasicBlock &BB : F) {
      OS << "\n           " << BB.getName() << ":\n";
      for (auto &I : BB.instructionsWithoutDebug()) {
        OS << (DI.isDivergent(I) ? "DIVERGENT:     " : "               ");
        OS << I << "\n";
      }
    }
  }
  return PreservedAnalyses::all();
}
