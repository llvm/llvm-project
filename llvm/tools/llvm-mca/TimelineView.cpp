//===--------------------- TimelineView.cpp ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \brief
///
/// This file implements the TimelineView interface.
///
//===----------------------------------------------------------------------===//

#include "TimelineView.h"

using namespace llvm;

namespace mca {

void TimelineView::initialize(unsigned MaxIterations) {
  unsigned NumInstructions =
      AsmSequence.getNumIterations() * AsmSequence.size();
  if (!MaxIterations)
    MaxIterations = DEFAULT_ITERATIONS;
  unsigned NumEntries =
      std::min(NumInstructions, MaxIterations * AsmSequence.size());
  Timeline.resize(NumEntries);
  TimelineViewEntry NullTVEntry = {0, 0, 0, 0, 0};
  std::fill(Timeline.begin(), Timeline.end(), NullTVEntry);

  WaitTime.resize(AsmSequence.size());
  WaitTimeEntry NullWTEntry = {0, 0, 0, 0};
  std::fill(WaitTime.begin(), WaitTime.end(), NullWTEntry);
}

void TimelineView::onInstructionEvent(const HWInstructionEvent &Event) {
  if (CurrentCycle >= MaxCycle || Event.Index >= Timeline.size())
    return;
  switch (Event.Type) {
  case HWInstructionEvent::Retired: {
    TimelineViewEntry &TVEntry = Timeline[Event.Index];
    TVEntry.CycleRetired = CurrentCycle;

    // Update the WaitTime entry which corresponds to this Index.
    WaitTimeEntry &WTEntry = WaitTime[Event.Index % AsmSequence.size()];
    WTEntry.Executions++;
    WTEntry.CyclesSpentInSchedulerQueue +=
        TVEntry.CycleIssued - TVEntry.CycleDispatched;
    assert(TVEntry.CycleDispatched <= TVEntry.CycleReady);
    WTEntry.CyclesSpentInSQWhileReady +=
        TVEntry.CycleIssued - TVEntry.CycleReady;
    WTEntry.CyclesSpentAfterWBAndBeforeRetire +=
        (TVEntry.CycleRetired - 1) - TVEntry.CycleExecuted;
    break;
  }
  case HWInstructionEvent::Ready:
    Timeline[Event.Index].CycleReady = CurrentCycle;
    break;
  case HWInstructionEvent::Issued:
    Timeline[Event.Index].CycleIssued = CurrentCycle;
    break;
  case HWInstructionEvent::Executed:
    Timeline[Event.Index].CycleExecuted = CurrentCycle;
    break;
  case HWInstructionEvent::Dispatched:
    Timeline[Event.Index].CycleDispatched = CurrentCycle;
    break;
  default:
    return;
  }
  LastCycle = std::max(LastCycle, CurrentCycle);
}

static void printAverageTime(raw_string_ostream &OS, double AverageTime) {
  // Round to the nearest tenth.
  OS << format("%.1f", floor((AverageTime * 10) + 0.5)/10);
  if (AverageTime < 10.0)
    OS << "    ";
  else if (AverageTime < 100.0)
    OS << "   ";
  else
    OS << "  ";
}

void TimelineView::printWaitTimeEntry(raw_string_ostream &OS,
                                      const WaitTimeEntry &Entry,
                                      unsigned SourceIndex) const {
  OS << SourceIndex << '.';
  if (SourceIndex < 10)
    OS << "    ";
  else if (SourceIndex < 100)
    OS << "   ";
  else if (SourceIndex < 1000)
    OS << "  ";
  else
    OS << ' ';

  if (Entry.Executions == 0) {
    OS << " -      -      -      -   ";
  } else {
    double AverageTime1, AverageTime2, AverageTime3;
    unsigned Executions = Entry.Executions;
    AverageTime1 = (double)Entry.CyclesSpentInSchedulerQueue / Executions;
    AverageTime2 = (double)Entry.CyclesSpentInSQWhileReady / Executions;
    AverageTime3 = (double)Entry.CyclesSpentAfterWBAndBeforeRetire / Executions;
    if (Executions < 10)
      OS << ' ' << Executions << "     ";
    else if (Executions < 100)
      OS << ' ' << Executions << "    ";
    else
      OS << Executions << "   ";

    printAverageTime(OS, AverageTime1);
    printAverageTime(OS, AverageTime2);
    printAverageTime(OS, AverageTime3);
  }
}

void TimelineView::printAverageWaitTimes(raw_ostream &OS) const {
  if (WaitTime.empty())
    return;

  std::string Buffer;
  raw_string_ostream TempStream(Buffer);

  TempStream
      << "\n\nAverage Wait times (based on the timeline view):\n"
      << "[0]: Executions\n"
      << "[1]: Average time spent waiting in a scheduler's queue\n"
      << "[2]: Average time spent waiting in a scheduler's queue while ready\n"
      << "[3]: Average time elapsed from WB until retire stage\n\n";
  TempStream << "      [0]    [1]    [2]    [3]\n";

  for (unsigned I = 0, E = WaitTime.size(); I < E; ++I) {
    printWaitTimeEntry(TempStream, WaitTime[I], I);
    // Append the instruction info at the end of the line.
    const MCInst &Inst = AsmSequence.getMCInstFromIndex(I);
    MCIP.printInst(&Inst, TempStream, "", STI);
    TempStream << '\n';
    TempStream.flush();
    OS << Buffer;
    Buffer = "";
  }
}

void TimelineView::printTimelineViewEntry(raw_string_ostream &OS,
                                          const TimelineViewEntry &Entry,
                                          unsigned Iteration,
                                          unsigned SourceIndex) const {
  if (Iteration == 0 && SourceIndex == 0)
    OS << '\n';
  OS << '[' << Iteration << ',' << SourceIndex << "]\t";
  for (unsigned I = 0, E = Entry.CycleDispatched; I < E; ++I)
    OS << ((I % 5 == 0) ? '.' : ' ');
  OS << TimelineView::DisplayChar::Dispatched;
  if (Entry.CycleDispatched != Entry.CycleExecuted) {
    // Zero latency instructions have the same value for CycleDispatched,
    // CycleIssued and CycleExecuted.
    for (unsigned I = Entry.CycleDispatched + 1, E = Entry.CycleIssued; I < E;
         ++I)
      OS << TimelineView::DisplayChar::Waiting;
    if (Entry.CycleIssued == Entry.CycleExecuted)
      OS << TimelineView::DisplayChar::DisplayChar::Executed;
    else {
      if (Entry.CycleDispatched != Entry.CycleIssued)
        OS << TimelineView::DisplayChar::Executing;
      for (unsigned I = Entry.CycleIssued + 1, E = Entry.CycleExecuted; I < E;
           ++I)
        OS << TimelineView::DisplayChar::Executing;
      OS << TimelineView::DisplayChar::Executed;
    }
  }

  for (unsigned I = Entry.CycleExecuted + 1, E = Entry.CycleRetired; I < E; ++I)
    OS << TimelineView::DisplayChar::RetireLag;
  OS << TimelineView::DisplayChar::Retired;

  // Skip other columns.
  for (unsigned I = Entry.CycleRetired + 1, E = LastCycle; I <= E; ++I)
    OS << ((I % 5 == 0 || I == LastCycle) ? '.' : ' ');
}

static void printTimelineHeader(raw_string_ostream &OS, unsigned Cycles) {
  OS << "\n\nTimeline view:\n";
  OS << "     \t";
  for (unsigned I = 0; I <= Cycles; ++I) {
    if (((I / 10) & 1) == 0)
      OS << ' ';
    else
      OS << I % 10;
  }

  OS << "\nIndex\t";
  for (unsigned I = 0; I <= Cycles; ++I) {
    if (((I / 10) & 1) == 0)
      OS << I % 10;
    else
      OS << ' ';
  }
  OS << '\n';
}

void TimelineView::printTimeline(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);

  printTimelineHeader(TempStream, LastCycle);
  TempStream.flush();
  OS << Buffer;

  for (unsigned I = 0, E = Timeline.size(); I < E; ++I) {
    Buffer = "";
    const TimelineViewEntry &Entry = Timeline[I];
    if (Entry.CycleRetired == 0)
      return;

    unsigned Iteration = I / AsmSequence.size();
    unsigned SourceIndex = I % AsmSequence.size();
    printTimelineViewEntry(TempStream, Entry, Iteration, SourceIndex);
    // Append the instruction info at the end of the line.
    const MCInst &Inst = AsmSequence.getMCInstFromIndex(I);
    MCIP.printInst(&Inst, TempStream, "", STI);
    TempStream << '\n';
    TempStream.flush();
    OS << Buffer;
  }
}
} // namespace mca
