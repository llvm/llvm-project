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

#include "Views/SchedulerStatistics.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

namespace mca {

void SchedulerStatistics::onEvent(const HWInstructionEvent &Event) {
  if (Event.Type == HWInstructionEvent::Issued)
    ++NumIssued;
}

void SchedulerStatistics::onReservedBuffers(const InstRef & /* unused */,
                                            ArrayRef<unsigned> Buffers) {
  for (const unsigned Buffer : Buffers) {
    BufferUsage &BU = Usage[Buffer];
    BU.SlotsInUse++;
    BU.MaxUsedSlots = std::max(BU.MaxUsedSlots, BU.SlotsInUse);
  }
}

void SchedulerStatistics::onReleasedBuffers(const InstRef & /* unused */,
                                            ArrayRef<unsigned> Buffers) {
  for (const unsigned Buffer : Buffers)
    Usage[Buffer].SlotsInUse--;
}

void SchedulerStatistics::updateHistograms() {
  for (BufferUsage &BU : Usage)
    BU.CumulativeNumUsedSlots += BU.SlotsInUse;
  IssuedPerCycle[NumIssued]++;
  NumIssued = 0;
}

void SchedulerStatistics::printSchedulerStats(raw_ostream &OS) const {
  OS << "\n\nSchedulers - "
     << "number of cycles where we saw N instructions issued:\n";
  OS << "[# issued], [# cycles]\n";

  const auto It =
      std::max_element(IssuedPerCycle.begin(), IssuedPerCycle.end());
  unsigned Index = std::distance(IssuedPerCycle.begin(), It);

  bool HasColors = OS.has_colors();
  for (unsigned I = 0, E = IssuedPerCycle.size(); I < E; ++I) {
    unsigned IPC = IssuedPerCycle[I];
    if (!IPC)
      continue;

    if (I == Index && HasColors)
      OS.changeColor(raw_ostream::SAVEDCOLOR, true, false);

    OS << " " << I << ",          " << IPC << "  ("
       << format("%.1f", ((double)IPC / NumCycles) * 100) << "%)\n";
    if (HasColors)
      OS.resetColor();
  }
}

void SchedulerStatistics::printSchedulerUsage(raw_ostream &OS) const {
  assert(NumCycles && "Unexpected number of cycles!");

  OS << "\nScheduler's queue usage:\n";
  if (all_of(Usage, [](const BufferUsage &BU) { return !BU.MaxUsedSlots; })) {
    OS << "No scheduler resources used.\n";
    return;
  }

  OS << "[1] Resource name.\n"
     << "[2] Average number of used buffer entries.\n"
     << "[3] Maximum number of used buffer entries.\n"
     << "[4] Total number of buffer entries.\n\n"
     << " [1]            [2]        [3]        [4]\n";

  formatted_raw_ostream FOS(OS);
  bool HasColors = FOS.has_colors();
  for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    if (ProcResource.BufferSize <= 0)
      continue;

    const BufferUsage &BU = Usage[I];
    double AvgUsage = (double)BU.CumulativeNumUsedSlots / NumCycles;
    double AlmostFullThreshold = (double)(ProcResource.BufferSize * 4) / 5;
    unsigned NormalizedAvg = floor((AvgUsage * 10) + 0.5) / 10;
    unsigned NormalizedThreshold = floor((AlmostFullThreshold * 10) + 0.5) / 10;

    FOS << ProcResource.Name;
    FOS.PadToColumn(17);
    if (HasColors && NormalizedAvg >= NormalizedThreshold)
      FOS.changeColor(raw_ostream::YELLOW, true, false);
    FOS << NormalizedAvg;
    if (HasColors)
      FOS.resetColor();
    FOS.PadToColumn(28);
    if (HasColors &&
        BU.MaxUsedSlots == static_cast<unsigned>(ProcResource.BufferSize))
      FOS.changeColor(raw_ostream::RED, true, false);
    FOS << BU.MaxUsedSlots;
    if (HasColors)
      FOS.resetColor();
    FOS.PadToColumn(39);
    FOS << ProcResource.BufferSize << '\n';
  }

  FOS.flush();
}

void SchedulerStatistics::printView(llvm::raw_ostream &OS) const {
  printSchedulerStats(OS);
  printSchedulerUsage(OS);
}

} // namespace mca
