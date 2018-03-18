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
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace mca {

class BackendStatistics : public View {
  // TODO: remove the dependency from Backend.
  const Backend &B;
  const llvm::MCSubtargetInfo &STI;

  using Histogram = std::map<unsigned, unsigned>;
  Histogram DispatchGroupSizePerCycle;
  Histogram RetiredPerCycle;
  Histogram IssuedPerCycle;

  unsigned NumDispatched;
  unsigned NumIssued;
  unsigned NumRetired;
  unsigned NumCycles;

  void updateHistograms() {
    DispatchGroupSizePerCycle[NumDispatched]++;
    IssuedPerCycle[NumIssued]++;
    RetiredPerCycle[NumRetired]++;
    NumDispatched = 0;
    NumIssued = 0;
    NumRetired = 0;
  }

  void printRetireUnitStatistics(llvm::raw_ostream &OS) const;
  void printDispatchUnitStatistics(llvm::raw_ostream &OS) const;
  void printSchedulerStatistics(llvm::raw_ostream &OS) const;

  void printDispatchStalls(llvm::raw_ostream &OS, unsigned RATStalls,
                           unsigned RCUStalls, unsigned SQStalls,
                           unsigned LDQStalls, unsigned STQStalls,
                           unsigned DGStalls) const;
  void printRATStatistics(llvm::raw_ostream &OS, unsigned Mappings,
                          unsigned MaxUsedMappings) const;
  void printRCUStatistics(llvm::raw_ostream &OS, const Histogram &Histogram,
                          unsigned Cycles) const;
  void printDispatchUnitUsage(llvm::raw_ostream &OS, const Histogram &Stats,
                              unsigned Cycles) const;
  void printIssuePerCycle(const Histogram &IssuePerCycle,
                          unsigned TotalCycles) const;
  void printSchedulerUsage(llvm::raw_ostream &OS, const llvm::MCSchedModel &SM,
                           const llvm::ArrayRef<BufferUsageEntry> &Usage) const;

public:
  BackendStatistics(const Backend &backend, const llvm::MCSubtargetInfo &sti)
      : B(backend), STI(sti), NumDispatched(0), NumIssued(0), NumRetired(0),
        NumCycles(0) {}

  void onInstructionEvent(const HWInstructionEvent &Event) override;

  void onCycleBegin(unsigned Cycle) override { NumCycles++; }

  void onCycleEnd(unsigned Cycle) override { updateHistograms(); }

  void printView(llvm::raw_ostream &OS) const override {
    printDispatchStalls(OS, B.getNumRATStalls(), B.getNumRCUStalls(),
                        B.getNumSQStalls(), B.getNumLDQStalls(),
                        B.getNumSTQStalls(), B.getNumDispatchGroupStalls());
    printRATStatistics(OS, B.getTotalRegisterMappingsCreated(),
                       B.getMaxUsedRegisterMappings());
    printDispatchUnitStatistics(OS);
    printSchedulerStatistics(OS);
    printRetireUnitStatistics(OS);

    std::vector<BufferUsageEntry> Usage;
    B.getBuffersUsage(Usage);
    printSchedulerUsage(OS, STI.getSchedModel(), Usage);
  }
};

} // namespace mca

#endif
