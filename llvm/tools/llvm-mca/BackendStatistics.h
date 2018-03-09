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
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_BACKENDSTATISTICS_H
#define LLVM_TOOLS_LLVM_MCA_BACKENDSTATISTICS_H

#include "HWEventListener.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace mca {

class BackendStatistics : public HWEventListener {
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

public:
  BackendStatistics() : NumDispatched(0), NumIssued(0), NumRetired(0) {}

  void onInstructionDispatched(unsigned Index) override { NumDispatched++; }
  void
  onInstructionIssued(unsigned Index,
                      const llvm::ArrayRef<std::pair<ResourceRef, unsigned>>
                          & /* unused */) override {
    NumIssued++;
  }
  void onInstructionRetired(unsigned Index) override { NumRetired++; }

  void onCycleBegin(unsigned Cycle) override { NumCycles++; }

  void onCycleEnd(unsigned Cycle) override { updateHistograms(); }

  void printHistograms(llvm::raw_ostream &OS) {
    printDispatchUnitStatistics(OS);
    printSchedulerStatistics(OS);
    printRetireUnitStatistics(OS);
  }
};

} // namespace mca

#endif
