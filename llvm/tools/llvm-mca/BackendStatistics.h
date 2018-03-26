//===--------------------- BackendStatistics.h ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements a printer class for printing generic Backend
/// statistics related to the dispatch logic, scheduler and retire unit.
///
/// Example:
/// ========
///
/// Dynamic Dispatch Stall Cycles:
/// RAT     - Register unavailable:                      0
/// RCU     - Retire tokens unavailable:                 0
/// SCHEDQ  - Scheduler full:                            42
/// LQ      - Load queue full:                           0
/// SQ      - Store queue full:                          0
/// GROUP   - Static restrictions on the dispatch group: 0
///
///
/// Register Alias Table:
/// Total number of mappings created: 210
/// Max number of mappings used:      35
///
///
/// Dispatch Logic - number of cycles where we saw N instructions dispatched:
/// [# dispatched], [# cycles]
///  0,              15  (11.5%)
///  5,              4  (3.1%)
///
/// Schedulers - number of cycles where we saw N instructions issued:
/// [# issued], [# cycles]
///  0,          7  (5.4%)
///  1,          4  (3.1%)
///  2,          8  (6.2%)
///
/// Retire Control Unit - number of cycles where we saw N instructions retired:
/// [# retired], [# cycles]
///  0,           9  (6.9%)
///  1,           6  (4.6%)
///  2,           1  (0.8%)
///  4,           3  (2.3%)
///
///
/// Scheduler's queue usage:
/// JALU01,  0/20
/// JFPU01,  18/18
/// JLSAGU,  0/12
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_BACKENDSTATISTICS_H
#define LLVM_TOOLS_LLVM_MCA_BACKENDSTATISTICS_H

#include "Backend.h"
#include "View.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace mca {

class BackendStatistics : public View {
  const llvm::MCSubtargetInfo &STI;

  using Histogram = std::map<unsigned, unsigned>;
  Histogram DispatchGroupSizePerCycle;
  Histogram RetiredPerCycle;
  Histogram IssuedPerCycle;

  unsigned NumDispatched;
  unsigned NumIssued;
  unsigned NumRetired;
  unsigned NumCycles;

  // Counts dispatch stall events caused by unavailability of resources.  There
  // is one counter for every generic stall kind (see class HWStallEvent).
  llvm::SmallVector<unsigned, 8> HWStalls;

  // Tracks the usage of a scheduler's queue.
  struct BufferUsage {
    unsigned SlotsInUse;
    unsigned MaxUsedSlots;
  };

  // There is a map entry for each buffered resource in the scheduling model.
  // Every time a buffer is consumed/freed, this view updates the corresponding
  // entry.
  llvm::DenseMap<unsigned, BufferUsage> BufferedResources;

  void updateHistograms() {
    DispatchGroupSizePerCycle[NumDispatched]++;
    IssuedPerCycle[NumIssued]++;
    RetiredPerCycle[NumRetired]++;
    NumDispatched = 0;
    NumIssued = 0;
    NumRetired = 0;
  }

  // Used to track the number of physical registers used in a register file.
  struct RegisterFileUsage {
    unsigned TotalMappings;
    unsigned MaxUsedMappings;
    unsigned CurrentlyUsedMappings;
  };

  // There is one entry for each register file implemented by the processor.
  llvm::SmallVector<RegisterFileUsage, 4> RegisterFiles;

  void printRetireUnitStatistics(llvm::raw_ostream &OS) const;
  void printDispatchUnitStatistics(llvm::raw_ostream &OS) const;
  void printSchedulerStatistics(llvm::raw_ostream &OS) const;

  void printDispatchStalls(llvm::raw_ostream &OS) const;
  void printRATStatistics(llvm::raw_ostream &OS) const;
  void printRCUStatistics(llvm::raw_ostream &OS, const Histogram &Histogram,
                          unsigned Cycles) const;
  void printDispatchUnitUsage(llvm::raw_ostream &OS, const Histogram &Stats,
                              unsigned Cycles) const;
  void printIssuePerCycle(const Histogram &IssuePerCycle,
                          unsigned TotalCycles) const;
  void printSchedulerUsage(llvm::raw_ostream &OS,
                           const llvm::MCSchedModel &SM) const;

public:
  BackendStatistics(const llvm::MCSubtargetInfo &sti)
      : STI(sti), NumDispatched(0), NumIssued(0), NumRetired(0), NumCycles(0),
        HWStalls(HWStallEvent::LastGenericEvent),
        // TODO: The view currently assumes a single register file. This will
        // change in future.
        RegisterFiles(1) {}

  void onInstructionEvent(const HWInstructionEvent &Event) override;

  void onCycleBegin(unsigned Cycle) override { NumCycles++; }

  void onCycleEnd(unsigned Cycle) override { updateHistograms(); }

  void onStallEvent(const HWStallEvent &Event) override {
    if (Event.Type < HWStallEvent::LastGenericEvent)
      HWStalls[Event.Type]++;
  }

  // Increases the number of used scheduler queue slots of every buffered
  // resource in the Buffers set.
  void onReservedBuffers(llvm::ArrayRef<unsigned> Buffers) override;

  // Decreases by one the number of used scheduler queue slots of every
  // buffered resource in the Buffers set.
  void onReleasedBuffers(llvm::ArrayRef<unsigned> Buffers) override;

  void printView(llvm::raw_ostream &OS) const override {
    printDispatchStalls(OS);
    printRATStatistics(OS);
    printDispatchUnitStatistics(OS);
    printSchedulerStatistics(OS);
    printRetireUnitStatistics(OS);
    printSchedulerUsage(OS, STI.getSchedModel());
  }
};
} // namespace mca

#endif
