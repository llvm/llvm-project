//===- DataRaceFreeAliasAnalysis.cpp - DRF-based Alias Analysis -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "drf-aa-result"

cl::opt<bool> llvm::EnableDRFAA(
    "enable-drf-aa", cl::init(false), cl::Hidden,
    cl::desc("Enable AA based on the data-race-free assumption "
             "(default = off)"));

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
                               const MemoryLocation &LocB, AAQueryInfo &AAQI) {
  if (!EnableDRFAA)
    return AAResultBase::alias(LocA, LocB, AAQI);

  LLVM_DEBUG(dbgs() << "DRFAA:\n\tLocA.Ptr = " << *LocA.Ptr
             << "\n\tLocB.Ptr = " << *LocB.Ptr << "\n");
  assert(notDifferentParent(LocA.Ptr, LocB.Ptr) &&
         "DRFAliasAnalysis doesn't support interprocedural queries.");

  if (const Instruction *AddrA = dyn_cast<Instruction>(LocA.Ptr))
    if (const Instruction *AddrB = dyn_cast<Instruction>(LocB.Ptr))
      if (TI.mayHappenInParallel(AddrA->getParent(), AddrB->getParent()))
        return NoAlias;
  return AAResultBase::alias(LocA, LocB, AAQI);
}

ModRefInfo DRFAAResult::getModRefInfo(const CallBase *Call,
                                      const MemoryLocation &Loc,
                                      AAQueryInfo &AAQI) {
  if (!EnableDRFAA)
    return AAResultBase::getModRefInfo(Call, Loc, AAQI);

  LLVM_DEBUG(dbgs() << "DRFAA:getModRefInfo(Call, Loc)\n");
  assert(notDifferentParent(Call, Loc.Ptr) &&
         "DRFAliasAnalysis doesn't support interprocedural queries.");

  if (const Instruction *Addr = dyn_cast<Instruction>(Loc.Ptr))
    if (TI.mayHappenInParallel(Call->getParent(), Addr->getParent()))
      return ModRefInfo::NoModRef;

  return AAResultBase::getModRefInfo(Call, Loc, AAQI);
}

ModRefInfo DRFAAResult::getModRefInfo(const CallBase *Call1,
                                      const CallBase *Call2,
                                      AAQueryInfo &AAQI) {
  if (!EnableDRFAA)
    return AAResultBase::getModRefInfo(Call1, Call2, AAQI);

  LLVM_DEBUG(dbgs() << "DRFAA:getModRefInfo(Call1, Call2)\n");

  if (TI.mayHappenInParallel(Call1->getParent(), Call2->getParent()))
    return ModRefInfo::NoModRef;

  return AAResultBase::getModRefInfo(Call1, Call2, AAQI);
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
