//===--------------------- Pipeline.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements an ordered container of stages that simulate the
/// pipeline of a hardware backend.
///
//===----------------------------------------------------------------------===//

#include "Pipeline.h"
#include "HWEventListener.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/Debug.h"

namespace mca {

#define DEBUG_TYPE "llvm-mca"

using namespace llvm;

void Pipeline::addEventListener(HWEventListener *Listener) {
  if (Listener)
    Listeners.insert(Listener);
  for (auto &S : Stages)
    S->addListener(Listener);
}

bool Pipeline::hasWorkToProcess() {
  return llvm::any_of(Stages, [](const std::unique_ptr<Stage> &S) {
    return S->hasWorkToComplete();
  });
}

// This routine returns early if any stage returns 'false' after execute() is
// called on it.
Stage::Status Pipeline::executeStages(InstRef &IR) {
  for (const std::unique_ptr<Stage> &S : Stages) {
    Stage::Status StatusOrErr = S->execute(IR);
    if (!StatusOrErr)
      return StatusOrErr.takeError();
    else if (StatusOrErr.get() == Stage::Stop)
      return Stage::Stop;
  }
  return Stage::Continue;
}

void Pipeline::preExecuteStages() {
  for (const std::unique_ptr<Stage> &S : Stages)
    S->preExecute();
}

void Pipeline::postExecuteStages() {
  for (const std::unique_ptr<Stage> &S : Stages)
    S->postExecute();
}

llvm::Error Pipeline::run() {
  assert(!Stages.empty() && "Unexpected empty pipeline found!");

  while (hasWorkToProcess()) {
    notifyCycleBegin();
    if (llvm::Error Err = runCycle())
      return Err;
    notifyCycleEnd();
    ++Cycles;
  }
  return llvm::ErrorSuccess();
}

llvm::Error Pipeline::runCycle() {
  // Update stages before we start processing new instructions.
  llvm::Error Err = llvm::ErrorSuccess();
  for (auto I = Stages.begin(), E = Stages.end(); I != E && !Err; ++I) {
    const std::unique_ptr<Stage> &S = *I;
    Err = S->cycleStart();
  }

  if (Err)
    return Err;

  // Now fetch and execute new instructions.
  InstRef IR;
  while (true) {
    preExecuteStages();
    Stage::Status Val = executeStages(IR);
    if (!Val)
      return Val.takeError();
    if (Val.get() == Stage::Stop)
      break;
    postExecuteStages();
  }

  // Update stages in preparation for a new cycle.
  for (auto I = Stages.begin(), E = Stages.end(); I != E && !Err; ++I) {
    const std::unique_ptr<Stage> &S = *I;
    Err = S->cycleEnd();
  }
  return Err;
}

void Pipeline::notifyCycleBegin() {
  LLVM_DEBUG(dbgs() << "[E] Cycle begin: " << Cycles << '\n');
  for (HWEventListener *Listener : Listeners)
    Listener->onCycleBegin();
}

void Pipeline::notifyCycleEnd() {
  LLVM_DEBUG(dbgs() << "[E] Cycle end: " << Cycles << "\n\n");
  for (HWEventListener *Listener : Listeners)
    Listener->onCycleEnd();
}
} // namespace mca.
