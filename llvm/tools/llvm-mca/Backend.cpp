//===--------------------- Backend.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Implementation of class Backend which emulates an hardware OoO backend.
///
//===----------------------------------------------------------------------===//

#include "Backend.h"
#include "FetchStage.h"
#include "HWEventListener.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/Debug.h"

namespace mca {

#define DEBUG_TYPE "llvm-mca"

using namespace llvm;

void Backend::addEventListener(HWEventListener *Listener) {
  if (Listener)
    Listeners.insert(Listener);
}

void Backend::run() {
  while (Fetch->isReady() || !Dispatch->isReady())
    runCycle(Cycles++);
}

void Backend::runCycle(unsigned Cycle) {
  notifyCycleBegin(Cycle);

  // Update the stages before we do any processing for this cycle.
  InstRef IR;
  Retire->preExecute(IR);
  Dispatch->preExecute(IR);
  Execute->preExecute(IR);

  // Fetch instructions and dispatch them to the hardware.
  while (Fetch->execute(IR)) {
    if (!Dispatch->execute(IR))
      break;
    Execute->execute(IR);
    Fetch->postExecute(IR);
  }

  notifyCycleEnd(Cycle);
}

void Backend::notifyCycleBegin(unsigned Cycle) {
  LLVM_DEBUG(dbgs() << "[E] Cycle begin: " << Cycle << '\n');
  for (HWEventListener *Listener : Listeners)
    Listener->onCycleBegin();
}

void Backend::notifyInstructionEvent(const HWInstructionEvent &Event) {
  for (HWEventListener *Listener : Listeners)
    Listener->onInstructionEvent(Event);
}

void Backend::notifyStallEvent(const HWStallEvent &Event) {
  for (HWEventListener *Listener : Listeners)
    Listener->onStallEvent(Event);
}

void Backend::notifyResourceAvailable(const ResourceRef &RR) {
  LLVM_DEBUG(dbgs() << "[E] Resource Available: [" << RR.first << '.'
                    << RR.second << "]\n");
  for (HWEventListener *Listener : Listeners)
    Listener->onResourceAvailable(RR);
}

void Backend::notifyReservedBuffers(ArrayRef<unsigned> Buffers) {
  for (HWEventListener *Listener : Listeners)
    Listener->onReservedBuffers(Buffers);
}

void Backend::notifyReleasedBuffers(ArrayRef<unsigned> Buffers) {
  for (HWEventListener *Listener : Listeners)
    Listener->onReleasedBuffers(Buffers);
}

void Backend::notifyCycleEnd(unsigned Cycle) {
  LLVM_DEBUG(dbgs() << "[E] Cycle end: " << Cycle << "\n\n");
  for (HWEventListener *Listener : Listeners)
    Listener->onCycleEnd();
}
} // namespace mca.
