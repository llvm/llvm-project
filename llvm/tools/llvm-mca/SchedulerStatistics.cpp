//===--------------------- SchedulerStatistics.cpp --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the SchedulerStatistics interface.
///
//===----------------------------------------------------------------------===//

#include "SchedulerStatistics.h"
#include "llvm/Support/Format.h"

using namespace llvm;

namespace mca {

void SchedulerStatistics::onEvent(const HWInstructionEvent &Event) {
  if (Event.Type == HWInstructionEvent::Issued)
    ++NumIssued;
}

void SchedulerStatistics::onReservedBuffers(ArrayRef<unsigned> Buffers) {
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

void SchedulerStatistics::onReleasedBuffers(ArrayRef<unsigned> Buffers) {
  for (const unsigned Buffer : Buffers) {
    assert(BufferedResources.find(Buffer) != BufferedResources.end() &&
           "Buffered resource not in map?");
    BufferUsage &BU = BufferedResources[Buffer];
    BU.SlotsInUse--;
  }
}

void SchedulerStatistics::printSchedulerStatistics(
    llvm::raw_ostream &OS) const {
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

void SchedulerStatistics::printSchedulerUsage(raw_ostream &OS) const {
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

    const auto It = BufferedResources.find(I);
    unsigned MaxUsedSlots =
        It == BufferedResources.end() ? 0 : It->second.MaxUsedSlots;
    TempStream << ProcResource.Name << ",  " << MaxUsedSlots << '/'
               << ProcResource.BufferSize << '\n';
  }

  TempStream.flush();
  OS << Buffer;
}
} // namespace mca
