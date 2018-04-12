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

void Backend::runCycle(unsigned Cycle) {
  notifyCycleBegin(Cycle);

  while (SM.hasNext()) {
    InstRef IR = SM.peekNext();
    std::unique_ptr<Instruction> NewIS =
        IB.createInstruction(IR.first, *IR.second);
    const InstrDesc &Desc = NewIS->getDesc();
    if (!DU->isAvailable(Desc.NumMicroOps) ||
        !DU->canDispatch(IR.first, *NewIS))
      break;

    Instruction *IS = NewIS.get();
    Instructions[IR.first] = std::move(NewIS);
    DU->dispatch(IR.first, IS, STI);
    SM.updateNext();
  }

  notifyCycleEnd(Cycle);
}

void Backend::notifyCycleBegin(unsigned Cycle) {
  DEBUG(dbgs() << "[E] Cycle begin: " << Cycle << '\n');
  for (HWEventListener *Listener : Listeners)
    Listener->onCycleBegin();

  DU->cycleEvent();
  HWS->cycleEvent();
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
  DEBUG(dbgs() << "[E] Resource Available: [" << RR.first << '.' << RR.second
               << "]\n");
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
  DEBUG(dbgs() << "[E] Cycle end: " << Cycle << "\n\n");
  for (HWEventListener *Listener : Listeners)
    Listener->onCycleEnd();
}
} // namespace mca.
