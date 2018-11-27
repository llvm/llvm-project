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
#include "llvm/IR/Instruction.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

ModRefInfo DRFAAResult::getModRefInfo(ImmutableCallSite CS1,
                                      ImmutableCallSite CS2) {
  if (TI.isSerial())
    return AAResultBase::getModRefInfo(CS1, CS2);

  if (TI.getTaskFor(CS1.getParent()) == TI.getTaskFor(CS2.getParent()))
    return AAResultBase::getModRefInfo(CS1, CS2);

  const Task *Enclosing = TI.getEnclosingTask(CS1.getParent(), CS2.getParent());
  const Task *CS1Task = Enclosing, *CS2Task = Enclosing;
  const Spindle *CS1Spindle = TI.getSpindleFor(CS1.getParent());
  const Spindle *CS2Spindle = TI.getSpindleFor(CS2.getParent());
  for (const Task *SubT : Enclosing->subtasks()) {
    if (SubT->encloses(CS1.getParent()))
      CS1Task = SubT;
    if (SubT->encloses(CS2.getParent()))
      CS2Task = SubT;
  }
  if (CS1Task != Enclosing)
    CS1Spindle = TI.getSpindleFor(CS1Task->getDetach()->getParent());
  if (CS2Task != Enclosing)
    CS2Spindle = TI.getSpindleFor(CS2Task->getDetach()->getParent());

  if (MPTasks.TaskList[CS1Spindle].count(CS2Task) ||
      MPTasks.TaskList[CS2Spindle].count(CS1Task))
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
