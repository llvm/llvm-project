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
/// statistics related to the scheduler and retire unit.
///
/// Example:
/// ========
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
/// Scheduler's queue usage:
/// JALU01,  0/20
/// JFPU01,  18/18
/// JLSAGU,  0/12
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_BACKENDSTATISTICS_H
#define LLVM_TOOLS_LLVM_MCA_BACKENDSTATISTICS_H

#include "View.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace mca {

class BackendStatistics : public View {
  const llvm::MCSubtargetInfo &STI;

  using Histogram = llvm::DenseMap<unsigned, unsigned>;
  Histogram RetiredPerCycle;
  Histogram IssuedPerCycle;

  unsigned NumIssued;
  unsigned NumRetired;
  unsigned NumCycles;

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
    IssuedPerCycle[NumIssued]++;
    RetiredPerCycle[NumRetired]++;
    NumIssued = 0;
    NumRetired = 0;
  }

  void printRetireUnitStatistics(llvm::raw_ostream &OS) const;
  void printSchedulerStatistics(llvm::raw_ostream &OS) const;

  void printRCUStatistics(llvm::raw_ostream &OS, const Histogram &Histogram,
                          unsigned Cycles) const;
  void printIssuePerCycle(const Histogram &IssuePerCycle,
                          unsigned TotalCycles) const;
  void printSchedulerUsage(llvm::raw_ostream &OS,
                           const llvm::MCSchedModel &SM) const;

public:
  BackendStatistics(const llvm::MCSubtargetInfo &sti)
      : STI(sti), NumIssued(0), NumRetired(0), NumCycles(0) { }

  void onInstructionEvent(const HWInstructionEvent &Event) override;

  void onCycleBegin(unsigned Cycle) override { NumCycles++; }

  void onCycleEnd(unsigned Cycle) override { updateHistograms(); }

  // Increases the number of used scheduler queue slots of every buffered
  // resource in the Buffers set.
  void onReservedBuffers(llvm::ArrayRef<unsigned> Buffers) override;

  // Decreases by one the number of used scheduler queue slots of every
  // buffered resource in the Buffers set.
  void onReleasedBuffers(llvm::ArrayRef<unsigned> Buffers) override;

  void printView(llvm::raw_ostream &OS) const override {
    printSchedulerStatistics(OS);
    printRetireUnitStatistics(OS);
    printSchedulerUsage(OS, STI.getSchedModel());
  }
};
} // namespace mca

#endif
