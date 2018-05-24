//===--------------------- SummaryView.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the summary view.
///
/// The goal of the summary view is to give a very quick overview of the
/// performance throughput. Below is an example of summary view:
///
///
/// Iterations:        300
/// Instructions:      900
/// Total Cycles:      610
/// Dispatch Width:    2
/// IPC:               1.48
/// Block RThroughput: 2.0
///
/// The summary view collects a few performance numbers. The two main
/// performance indicators are 'Total Cycles' and IPC (Instructions Per Cycle).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_SUMMARYVIEW_H
#define LLVM_TOOLS_LLVM_MCA_SUMMARYVIEW_H

#include "SourceMgr.h"
#include "View.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Support/raw_ostream.h"

namespace mca {

/// A view that collects and prints a few performance numbers.
class SummaryView : public View {
  const llvm::MCSchedModel &SM;
  const SourceMgr &Source;
  const unsigned DispatchWidth;
  unsigned TotalCycles;
  // The total number of micro opcodes contributed by a block of instructions.
  unsigned NumMicroOps;
  // For each processor resource, this map stores the cumulative number of
  // resource cycles consumed by a block of instructions. The resource mask ID
  // is used as the key value to access elements of this map.
  llvm::DenseMap<uint64_t, unsigned> ProcResourceUsage;

  // Compute the reciprocal throughput for the analyzed code block.
  // The reciprocal block throughput is computed as the MAX between:
  //   - NumMicroOps / DispatchWidth
  //   - Total Resource Cycles / #Units   (for every resource consumed).
  double getBlockRThroughput() const;

public:
  SummaryView(const llvm::MCSchedModel &Model, const SourceMgr &S,
              unsigned Width)
      : SM(Model), Source(S), DispatchWidth(Width), TotalCycles(0),
        NumMicroOps(0) {}

  void onCycleEnd() override { ++TotalCycles; }

  void onInstructionEvent(const HWInstructionEvent &Event) override;

  void printView(llvm::raw_ostream &OS) const override;
};
} // namespace mca

#endif
