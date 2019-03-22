//===- DataRaceFreeAliasAnalysis.cpp - DRF-based Alias Analysis -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DataRaceFreeAliasAnalysis pass, which implements alias
// analysis based on the assumption that a Tapir program is data-race free.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataRaceFreeAliasAnalysis.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

// A handy option for disabling scoped no-alias functionality.
static cl::opt<bool> EnableDRFAA("enable-drf-aa-results",
                                 cl::init(false), cl::Hidden);

bool DRFAAResult::invalidate(Function &Fn, const PreservedAnalyses &PA,
                             FunctionAnalysisManager::Invalidator &Inv) {
  // We don't care if this analysis itself is preserved, it has no state. But we
  // need to check that the analyses it depends on have been.
  if (Inv.invalidate<TaskAnalysis>(Fn, PA))
    return true;

  // Otherwise this analysis result remains valid.
  return false;
}

AliasResult DRFAAResult::alias(const MemoryLocation &LocA,
                               const MemoryLocation &LocB) {
  if (!EnableDRFAA)
    return AAResultBase::alias(LocA, LocB);

  dbgs() << "DRFAA:\n\tLocA.Ptr = " << *LocA.Ptr << "\n\tLocB.Ptr = " << *LocB.Ptr << "\n";
  if (const Instruction *AddrA = dyn_cast<Instruction>(LocA.Ptr))
    if (const Instruction *AddrB = dyn_cast<Instruction>(LocB.Ptr))
      if (TI.mayHappenInParallel(AddrA->getParent(), AddrB->getParent()))
        return NoAlias;
  return AAResultBase::alias(LocA, LocB);
}

ModRefInfo DRFAAResult::getModRefInfo(ImmutableCallSite CS,
                                      const MemoryLocation &Loc) {
  if (!EnableDRFAA)
    return AAResultBase::getModRefInfo(CS, Loc);

  dbgs() << "DRFAA:getModRefInfo(CS, Loc)\n";

  if (const Instruction *Addr = dyn_cast<Instruction>(Loc.Ptr))
    if (TI.mayHappenInParallel(CS.getParent(), Addr->getParent()))
      return ModRefInfo::NoModRef;

  return AAResultBase::getModRefInfo(CS, Loc);
}

ModRefInfo DRFAAResult::getModRefInfo(ImmutableCallSite CS1,
                                      ImmutableCallSite CS2) {
  if (!EnableDRFAA)
    return AAResultBase::getModRefInfo(CS1, CS2);

  dbgs() << "DRFAA:getModRefInfo(CS1, CS2)\n";

  if (TI.mayHappenInParallel(CS1.getParent(), CS2.getParent()))
    return ModRefInfo::NoModRef;

  return AAResultBase::getModRefInfo(CS1, CS2);
}

AnalysisKey DRFAA::Key;

DRFAAResult DRFAA::run(Function &F, FunctionAnalysisManager &AM) {
  return DRFAAResult(AM.getResult<TaskAnalysis>(F));
}

char DRFAAWrapperPass::ID = 0;
INITIALIZE_PASS_BEGIN(DRFAAWrapperPass, "drf-aa",
                      "DRF-based Alias Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_END(DRFAAWrapperPass, "drf-aa",
                    "DRF-based Alias Analysis", false, true)

FunctionPass *llvm::createDRFAAWrapperPass() {
  return new DRFAAWrapperPass();
}

DRFAAWrapperPass::DRFAAWrapperPass() : FunctionPass(ID) {
  initializeDRFAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

bool DRFAAWrapperPass::runOnFunction(Function &F) {
  Result.reset(
      new DRFAAResult(getAnalysis<TaskInfoWrapperPass>().getTaskInfo()));
  return false;
}

void DRFAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TaskInfoWrapperPass>();
}
