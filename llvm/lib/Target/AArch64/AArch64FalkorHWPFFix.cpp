//===-- AArch64FalkorHWPFFix.cpp - Avoid HW prefetcher pitfalls on Falkor--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file For Falkor, we want to avoid HW prefetcher instruction tag collisions
/// that may inhibit the HW prefetching.  This is done in two steps.  Before
/// ISel, we mark strided loads (i.e. those that will likely benefit from
/// prefetching) with metadata.  Then, after opcodes have been finalized, we
/// insert MOVs and re-write loads to prevent unintnentional tag collisions.
// ===---------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "falkor-hwpf-fix"

STATISTIC(NumStridedLoadsMarked, "Number of strided loads marked");

namespace {

class FalkorMarkStridedAccesses {
public:
  FalkorMarkStridedAccesses(LoopInfo &LI, ScalarEvolution &SE)
      : LI(LI), SE(SE) {}

  bool run();

private:
  bool runOnLoop(Loop &L);

  LoopInfo &LI;
  ScalarEvolution &SE;
};

class FalkorMarkStridedAccessesLegacy : public FunctionPass {
public:
  static char ID; // Pass ID, replacement for typeid
  FalkorMarkStridedAccessesLegacy() : FunctionPass(ID) {
    initializeFalkorMarkStridedAccessesLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    // FIXME: For some reason, preserving SE here breaks LSR (even if
    // this pass changes nothing).
    // AU.addPreserved<ScalarEvolutionWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};
} // namespace

char FalkorMarkStridedAccessesLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(FalkorMarkStridedAccessesLegacy, DEBUG_TYPE,
                      "Falkor HW Prefetch Fix", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(FalkorMarkStridedAccessesLegacy, DEBUG_TYPE,
                    "Falkor HW Prefetch Fix", false, false)

FunctionPass *llvm::createFalkorMarkStridedAccessesPass() {
  return new FalkorMarkStridedAccessesLegacy();
}

bool FalkorMarkStridedAccessesLegacy::runOnFunction(Function &F) {
  TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const AArch64Subtarget *ST =
      TPC.getTM<AArch64TargetMachine>().getSubtargetImpl(F);
  if (ST->getProcFamily() != AArch64Subtarget::Falkor)
    return false;

  if (skipFunction(F))
    return false;

  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  FalkorMarkStridedAccesses LDP(LI, SE);
  return LDP.run();
}

bool FalkorMarkStridedAccesses::run() {
  bool MadeChange = false;

  for (Loop *L : LI)
    for (auto LIt = df_begin(L), LE = df_end(L); LIt != LE; ++LIt)
      MadeChange |= runOnLoop(**LIt);

  return MadeChange;
}

bool FalkorMarkStridedAccesses::runOnLoop(Loop &L) {
  // Only mark strided loads in the inner-most loop
  if (!L.empty())
    return false;

  bool MadeChange = false;

  for (BasicBlock *BB : L.blocks()) {
    for (Instruction &I : *BB) {
      LoadInst *LoadI = dyn_cast<LoadInst>(&I);
      if (!LoadI)
        continue;

      Value *PtrValue = LoadI->getPointerOperand();
      if (L.isLoopInvariant(PtrValue))
        continue;

      const SCEV *LSCEV = SE.getSCEV(PtrValue);
      const SCEVAddRecExpr *LSCEVAddRec = dyn_cast<SCEVAddRecExpr>(LSCEV);
      if (!LSCEVAddRec || !LSCEVAddRec->isAffine())
        continue;

      LoadI->setMetadata(FALKOR_STRIDED_ACCESS_MD,
                         MDNode::get(LoadI->getContext(), {}));
      ++NumStridedLoadsMarked;
      DEBUG(dbgs() << "Load: " << I << " marked as strided\n");
      MadeChange = true;
    }
  }

  return MadeChange;
}
