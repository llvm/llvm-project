//===--------------------- DispatchStatistics.cpp ---------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the DispatchStatistics interface.
///
//===----------------------------------------------------------------------===//

#include "DispatchStatistics.h"
#include "llvm/Support/Format.h"

using namespace llvm;

namespace mca {

void DispatchStatistics::onStallEvent(const HWStallEvent &Event) {
  if (Event.Type < HWStallEvent::LastGenericEvent)
    HWStalls[Event.Type]++;
}

void DispatchStatistics::onInstructionEvent(const HWInstructionEvent &Event) {
  if (Event.Type == HWInstructionEvent::Dispatched)
    ++NumDispatched;
}

void DispatchStatistics::printDispatchHistogram(llvm::raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nDispatch Logic - "
             << "number of cycles where we saw N instructions dispatched:\n";
  TempStream << "[# dispatched], [# cycles]\n";
  for (const std::pair<unsigned, unsigned> &Entry : DispatchGroupSizePerCycle) {
    TempStream << " " << Entry.first << ",              " << Entry.second
               << "  ("
               << format("%.1f", ((double)Entry.second / NumCycles) * 100.0)
               << "%)\n";
  }

  TempStream.flush();
  OS << Buffer;
}

void DispatchStatistics::printDispatchStalls(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nDynamic Dispatch Stall Cycles:\n";
  TempStream << "RAT     - Register unavailable:                      "
             << HWStalls[HWStallEvent::RegisterFileStall];
  TempStream << "\nRCU     - Retire tokens unavailable:                 "
             << HWStalls[HWStallEvent::RetireControlUnitStall];
  TempStream << "\nSCHEDQ  - Scheduler full:                            "
             << HWStalls[HWStallEvent::SchedulerQueueFull];
  TempStream << "\nLQ      - Load queue full:                           "
             << HWStalls[HWStallEvent::LoadQueueFull];
  TempStream << "\nSQ      - Store queue full:                          "
             << HWStalls[HWStallEvent::StoreQueueFull];
  TempStream << "\nGROUP   - Static restrictions on the dispatch group: "
             << HWStalls[HWStallEvent::DispatchGroupStall];
  TempStream << '\n';
  TempStream.flush();
  OS << Buffer;
}

} // namespace mca
