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
}

bool Pipeline::hasWorkToProcess() {
  const auto It = llvm::find_if(Stages, [](const std::unique_ptr<Stage> &S) {
    return S->hasWorkToComplete();
  });
  return It != Stages.end();
}

// This routine returns early if any stage returns 'false' after execute() is
// called on it.
bool Pipeline::executeStages(InstRef &IR) {
  for (const std::unique_ptr<Stage> &S : Stages)
    if (!S->execute(IR))
      return false;
  return true;
}

void Pipeline::postExecuteStages(const InstRef &IR) {
  for (const std::unique_ptr<Stage> &S : Stages)
    S->postExecute(IR);
}

void Pipeline::run() {
  while (hasWorkToProcess())
    runCycle(Cycles++);
}

void Pipeline::runCycle(unsigned Cycle) {
  notifyCycleBegin(Cycle);

  // Update the stages before we do any processing for this cycle.
  InstRef IR;
  for (auto &S : Stages)
    S->preExecute(IR);

  // Continue executing this cycle until any stage claims it cannot make
  // progress.
  while (executeStages(IR))
    postExecuteStages(IR);

  notifyCycleEnd(Cycle);
}

void Pipeline::notifyCycleBegin(unsigned Cycle) {
  LLVM_DEBUG(dbgs() << "[E] Cycle begin: " << Cycle << '\n');
  for (HWEventListener *Listener : Listeners)
    Listener->onCycleBegin();
}

void Pipeline::notifyInstructionEvent(const HWInstructionEvent &Event) {
  for (HWEventListener *Listener : Listeners)
    Listener->onInstructionEvent(Event);
}

void Pipeline::notifyStallEvent(const HWStallEvent &Event) {
  for (HWEventListener *Listener : Listeners)
    Listener->onStallEvent(Event);
}

void Pipeline::notifyResourceAvailable(const ResourceRef &RR) {
  LLVM_DEBUG(dbgs() << "[E] Resource Available: [" << RR.first << '.'
                    << RR.second << "]\n");
  for (HWEventListener *Listener : Listeners)
    Listener->onResourceAvailable(RR);
}

void Pipeline::notifyReservedBuffers(ArrayRef<unsigned> Buffers) {
  for (HWEventListener *Listener : Listeners)
    Listener->onReservedBuffers(Buffers);
}

void Pipeline::notifyReleasedBuffers(ArrayRef<unsigned> Buffers) {
  for (HWEventListener *Listener : Listeners)
    Listener->onReleasedBuffers(Buffers);
}

void Pipeline::notifyCycleEnd(unsigned Cycle) {
  LLVM_DEBUG(dbgs() << "[E] Cycle end: " << Cycle << "\n\n");
  for (HWEventListener *Listener : Listeners)
    Listener->onCycleEnd();
}
} // namespace mca.
