//===- SCCP.cpp - Sparse Conditional Constant Propagation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements sparse conditional constant propagation and merging:
//
// Specifically, this:
//   * Assumes values are constant unless proven otherwise
//   * Assumes BasicBlocks are dead unless proven otherwise
//   * Proves values to be constant, and replaces them with constants
//   * Proves conditional branches to be unconditional
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"
#include <cassert>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "sccp"

STATISTIC(NumInstRemoved, "Number of instructions removed");
STATISTIC(NumDeadBlocks , "Number of basic blocks unreachable");
STATISTIC(NumInstReplaced,
          "Number of instructions replaced with (simpler) instruction");

// runSCCP() - Run the Sparse Conditional Constant Propagation algorithm,
// and return true if the function was modified.
static bool runSCCP(Function &F, const DataLayout &DL,
                    const TargetLibraryInfo *TLI, DomTreeUpdater &DTU) {
  LLVM_DEBUG(dbgs() << "SCCP on function '" << F.getName() << "'\n");
  SCCPSolver Solver(
      DL, [TLI](Function &F) -> const TargetLibraryInfo & { return *TLI; },
      F.getContext());

  // Mark the first block of the function as being executable.
  Solver.markBlockExecutable(&F.front());

  // Mark all arguments to the function as being overdefined.
  for (Argument &AI : F.args())
    Solver.markOverdefined(&AI);

  // Solve for constants.
  bool ResolvedUndefs = true;
  while (ResolvedUndefs) {
    Solver.solve();
    LLVM_DEBUG(dbgs() << "RESOLVING UNDEFs\n");
    ResolvedUndefs = Solver.resolvedUndefsIn(F);
  }

  bool MadeChanges = false;

  // If we decided that there are basic blocks that are dead in this function,
  // delete their contents now.  Note that we cannot actually delete the blocks,
  // as we cannot modify the CFG of the function.

  SmallPtrSet<Value *, 32> InsertedValues;
  SmallVector<BasicBlock *, 8> BlocksToErase;
  for (BasicBlock &BB : F) {
    if (!Solver.isBlockExecutable(&BB)) {
      LLVM_DEBUG(dbgs() << "  BasicBlock Dead:" << BB);
      ++NumDeadBlocks;
      BlocksToErase.push_back(&BB);
      MadeChanges = true;
      continue;
    }

    MadeChanges |= Solver.simplifyInstsInBlock(BB, InsertedValues,
                                               NumInstRemoved, NumInstReplaced);
  }

  // Remove unreachable blocks and non-feasible edges.
  for (BasicBlock *DeadBB : BlocksToErase)
    NumInstRemoved += changeToUnreachable(DeadBB->getFirstNonPHI(),
                                          /*PreserveLCSSA=*/false, &DTU);

  BasicBlock *NewUnreachableBB = nullptr;
  for (BasicBlock &BB : F)
    MadeChanges |= Solver.removeNonFeasibleEdges(&BB, DTU, NewUnreachableBB);

  for (BasicBlock *DeadBB : BlocksToErase)
    if (!DeadBB->hasAddressTaken())
      DTU.deleteBB(DeadBB);

  return MadeChanges;
}

PreservedAnalyses SCCPPass::run(Function &F, FunctionAnalysisManager &AM) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto *DT = AM.getCachedResult<DominatorTreeAnalysis>(F);
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  if (!runSCCP(F, DL, &TLI, DTU))
    return PreservedAnalyses::all();

  auto PA = PreservedAnalyses();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

namespace {

//===--------------------------------------------------------------------===//
//
/// SCCP Class - This class uses the SCCPSolver to implement a per-function
/// Sparse Conditional Constant Propagator.
///
class SCCPLegacyPass : public FunctionPass {
public:
  // Pass identification, replacement for typeid
  static char ID;

  SCCPLegacyPass() : FunctionPass(ID) {
    initializeSCCPLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  // runOnFunction - Run the Sparse Conditional Constant Propagation
  // algorithm, and return true if the function was modified.
  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    const DataLayout &DL = F.getParent()->getDataLayout();
    const TargetLibraryInfo *TLI =
        &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
    DomTreeUpdater DTU(DTWP ? &DTWP->getDomTree() : nullptr,
                       DomTreeUpdater::UpdateStrategy::Lazy);
    return runSCCP(F, DL, TLI, DTU);
  }
};

} // end anonymous namespace

char SCCPLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(SCCPLegacyPass, "sccp",
                      "Sparse Conditional Constant Propagation", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(SCCPLegacyPass, "sccp",
                    "Sparse Conditional Constant Propagation", false, false)

// createSCCPPass - This is the public interface to this file.
FunctionPass *llvm::createSCCPPass() { return new SCCPLegacyPass(); }

