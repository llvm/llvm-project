//===--------------------- RetireControlUnitStatistics.h ------------------*-
//C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines class RetireControlUnitStatistics: a view that knows how
/// to print general statistics related to the retire control unit.
///
/// Example:
/// ========
///
/// Retire Control Unit - number of cycles where we saw N instructions retired:
/// [# retired], [# cycles]
///  0,           9  (6.9%)
///  1,           6  (4.6%)
///  2,           1  (0.8%)
///  4,           3  (2.3%)
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_RETIRECONTROLUNITSTATISTICS_H
#define LLVM_TOOLS_LLVM_MCA_RETIRECONTROLUNITSTATISTICS_H

#include "View.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace mca {

class RetireControlUnitStatistics : public View {
  using Histogram = llvm::DenseMap<unsigned, unsigned>;
  Histogram RetiredPerCycle;

  unsigned NumRetired;
  unsigned NumCycles;

  void updateHistograms() {
    RetiredPerCycle[NumRetired]++;
    NumRetired = 0;
  }

public:
  RetireControlUnitStatistics() : NumRetired(0), NumCycles(0) {}

  void onInstructionEvent(const HWInstructionEvent &Event) override;

  void onCycleBegin() override { NumCycles++; }

  void onCycleEnd() override { updateHistograms(); }

  void printView(llvm::raw_ostream &OS) const override;
};
} // namespace mca

#endif
