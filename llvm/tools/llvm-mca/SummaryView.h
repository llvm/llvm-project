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
/// Iterations:     300
/// Instructions:   900
/// Total Cycles:   610
/// Dispatch Width: 2
/// IPC:            1.48
///
///
/// The summary view collects a few performance numbers. The two main
/// performance indicators are 'Total Cycles' and IPC (Instructions Per Cycle).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_SUMMARYVIEW_H
#define LLVM_TOOLS_LLVM_MCA_SUMMARYVIEW_H

#include "SourceMgr.h"
#include "View.h"
#include "llvm/Support/raw_ostream.h"

namespace mca {

/// \brief A view that collects and prints a few performance numbers.
class SummaryView : public View {
  const SourceMgr &Source;
  const unsigned DispatchWidth;
  unsigned TotalCycles;

public:
  SummaryView(const SourceMgr &S, unsigned Width)
      : Source(S), DispatchWidth(Width), TotalCycles(0) {}

  void onCycleEnd(unsigned /* unused */) override { ++TotalCycles; }

  void printView(llvm::raw_ostream &OS) const override;
};
} // namespace mca

#endif
