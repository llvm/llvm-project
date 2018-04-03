//===--------------------- BackendStatistics.cpp ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Functionalities used by the BackendPrinter to print out histograms
/// related to number of {dispatch/issue/retire} per number of cycles.
///
//===----------------------------------------------------------------------===//

#include "BackendStatistics.h"
#include "llvm/Support/Format.h"

using namespace llvm;

namespace mca {

void BackendStatistics::onInstructionEvent(const HWInstructionEvent &Event) {
  switch (Event.Type) {
  default:
    break;
  case HWInstructionEvent::Retired: {
    ++NumRetired;
    break;
  }
  case HWInstructionEvent::Issued:
    ++NumIssued;
    break;
  case HWInstructionEvent::Dispatched: {
    ++NumDispatched;
  }
  }
}

void BackendStatistics::onReservedBuffers(ArrayRef<unsigned> Buffers) {
  for (const unsigned Buffer : Buffers) {
    if (BufferedResources.find(Buffer) != BufferedResources.end()) {
      BufferUsage &BU = BufferedResources[Buffer];
      BU.SlotsInUse++;
      BU.MaxUsedSlots = std::max(BU.MaxUsedSlots, BU.SlotsInUse);
      continue;
    }

    BufferedResources.insert(
        std::pair<unsigned, BufferUsage>(Buffer, {1U, 1U}));
  }
}

void BackendStatistics::onReleasedBuffers(ArrayRef<unsigned> Buffers) {
  for (const unsigned Buffer : Buffers) {
    assert(BufferedResources.find(Buffer) != BufferedResources.end() &&
           "Buffered resource not in map?");
    BufferUsage &BU = BufferedResources[Buffer];
    BU.SlotsInUse--;
  }
}

void BackendStatistics::printRetireUnitStatistics(llvm::raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nRetire Control Unit - "
             << "number of cycles where we saw N instructions retired:\n";
  TempStream << "[# retired], [# cycles]\n";

  for (const std::pair<unsigned, unsigned> &Entry : RetiredPerCycle) {
    TempStream << " " << Entry.first;
    if (Entry.first < 10)
      TempStream << ",           ";
    else
      TempStream << ",          ";
    TempStream << Entry.second << "  ("
               << format("%.1f", ((double)Entry.second / NumCycles) * 100.0)
               << "%)\n";
  }

  TempStream.flush();
  OS << Buffer;
}

void BackendStatistics::printDispatchUnitStatistics(
    llvm::raw_ostream &OS) const {
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

void BackendStatistics::printSchedulerStatistics(llvm::raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nSchedulers - number of cycles where we saw N instructions "
                "issued:\n";
  TempStream << "[# issued], [# cycles]\n";
  for (const std::pair<unsigned, unsigned> &Entry : IssuedPerCycle) {
    TempStream << " " << Entry.first << ",          " << Entry.second << "  ("
               << format("%.1f", ((double)Entry.second / NumCycles) * 100)
               << "%)\n";
  }

  TempStream.flush();
  OS << Buffer;
}

void BackendStatistics::printDispatchStalls(raw_ostream &OS) const {
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

void BackendStatistics::printSchedulerUsage(raw_ostream &OS,
                                            const MCSchedModel &SM) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nScheduler's queue usage:\n";
  // Early exit if no buffered resources were consumed.
  if (BufferedResources.empty()) {
    TempStream << "No scheduler resources used.\n";
    TempStream.flush();
    OS << Buffer;
    return;
  }

  for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    if (ProcResource.BufferSize <= 0)
      continue;

    const BufferUsage &BU = BufferedResources.lookup(I);
    TempStream << ProcResource.Name << ",  " << BU.MaxUsedSlots << '/'
               << ProcResource.BufferSize << '\n';
  }

  TempStream.flush();
  OS << Buffer;
}
} // namespace mca
