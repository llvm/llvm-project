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
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "drf-aa-result"

// A handy option for disabling this flavor of DRFAA functionality.
static cl::opt<bool> EnableDRFAA("enable-drf-aa-result",
                                 cl::init(true), cl::Hidden);

bool DRFAAResult::invalidate(Function &Fn, const PreservedAnalyses &PA,
                             FunctionAnalysisManager::Invalidator &Inv) {
  // We don't care if this analysis itself is preserved, it has no state. But we
  // need to check that the analyses it depends on have been.
  if (Inv.invalidate<TaskAnalysis>(Fn, PA))
    return true;

  // Otherwise this analysis result remains valid.
  return false;
}

#ifndef NDEBUG
static const Function *getParent(const Value *V) {
  if (const Instruction *inst = dyn_cast<Instruction>(V)) {
    if (!inst->getParent())
      return nullptr;
    return inst->getParent()->getParent();
  }

  if (const Argument *arg = dyn_cast<Argument>(V))
    return arg->getParent();

  return nullptr;
}

static bool notDifferentParent(const Value *O1, const Value *O2) {

  const Function *F1 = getParent(O1);
  const Function *F2 = getParent(O2);

  return !F1 || !F2 || F1 == F2;
}
#endif

AliasResult DRFAAResult::alias(const MemoryLocation &LocA,
                               const MemoryLocation &LocB) {
  if (!EnableDRFAA)
    return AAResultBase::alias(LocA, LocB);

  LLVM_DEBUG(dbgs() << "DRFAA:\n\tLocA.Ptr = " << *LocA.Ptr
             << "\n\tLocB.Ptr = " << *LocB.Ptr << "\n");
  assert(notDifferentParent(LocA.Ptr, LocB.Ptr) &&
         "DRFAliasAnalysis doesn't support interprocedural queries.");

  if (const Instruction *AddrA = dyn_cast<Instruction>(LocA.Ptr))
    if (const Instruction *AddrB = dyn_cast<Instruction>(LocB.Ptr))
      if (TI.mayHappenInParallel(AddrA->getParent(), AddrB->getParent()))
        return NoAlias;
  return AAResultBase::alias(LocA, LocB);
}

ModRefInfo DRFAAResult::getModRefInfo(const CallBase *Call,
                                      const MemoryLocation &Loc) {
  if (!EnableDRFAA)
    return AAResultBase::getModRefInfo(Call, Loc);

  LLVM_DEBUG(dbgs() << "DRFAA:getModRefInfo(Call, Loc)\n");
  assert(notDifferentParent(Call, Loc.Ptr) &&
         "DRFAliasAnalysis doesn't support interprocedural queries.");

  if (const Instruction *Addr = dyn_cast<Instruction>(Loc.Ptr))
    if (TI.mayHappenInParallel(Call->getParent(), Addr->getParent()))
      return ModRefInfo::NoModRef;

  return AAResultBase::getModRefInfo(Call, Loc);
}

ModRefInfo DRFAAResult::getModRefInfo(const CallBase *Call1,
                                      const CallBase *Call2) {
  if (!EnableDRFAA)
    return AAResultBase::getModRefInfo(Call1, Call2);

  LLVM_DEBUG(dbgs() << "DRFAA:getModRefInfo(Call1, Call2)\n");

  if (TI.mayHappenInParallel(Call1->getParent(), Call2->getParent()))
    return ModRefInfo::NoModRef;

  return AAResultBase::getModRefInfo(Call1, Call2);
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
