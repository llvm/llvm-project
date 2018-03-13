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
/// Instruction Info:
/// [1]: #uOps
/// [2]: Latency
/// [3]: RThroughput
/// [4]: MayLoad
/// [5]: MayStore
/// [6]: HasSideEffects
///
/// [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
///  1      2     1.00                    	vmulps	%xmm0, %xmm1, %xmm2
///  1      3     1.00                    	vhaddps	%xmm2, %xmm2, %xmm3
///  1      3     1.00                    	vhaddps	%xmm3, %xmm3, %xmm4
///
/// The summary view is structured in two sections.
///
/// The first section collects a a few performance numbers. The two main
/// performance indicators are 'Total Cycles' and IPC (Instructions Per Cycle).
///
/// The second section shows the latency and reciprocal throughput of every
/// instruction in the sequence. This section also reports extra informaton
/// related to the number of micro opcodes, and opcode properties (i.e.
/// 'MayLoad', 'MayStore', 'HasSideEffects)
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_SUMMARYVIEW_H
#define LLVM_TOOLS_LLVM_MCA_SUMMARYVIEW_H

#include "SourceMgr.h"
#include "View.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "llvm-mca"

namespace mca {

/// \brief A printer class that knows how to collects statistics on the
/// code analyzed by the llvm-mca tool.
///
/// This class knows how to print out the analysis information collected
/// during the execution of the code. Internally, it delegates to other
/// classes the task of printing out timeline information as well as
/// resource pressure.
class SummaryView : public View {
  const llvm::MCSubtargetInfo &STI;
  const llvm::MCInstrInfo &MCII;
  const SourceMgr &Source;
  llvm::MCInstPrinter &MCIP;

  const unsigned DispatchWidth;
  unsigned TotalCycles;

  void printSummary(llvm::raw_ostream &OS) const;
  void printInstructionInfo(llvm::raw_ostream &OS) const;

public:
  SummaryView(const llvm::MCSubtargetInfo &sti, const llvm::MCInstrInfo &mcii,
              const SourceMgr &S, llvm::MCInstPrinter &IP, unsigned Width)
      : STI(sti), MCII(mcii), Source(S), MCIP(IP), DispatchWidth(Width),
        TotalCycles(0) {}

  void onCycleEnd(unsigned /* unused */) override { ++TotalCycles; }

  void printView(llvm::raw_ostream &OS) const override {
    printSummary(OS);
    printInstructionInfo(OS);
  }
};

} // namespace mca

#endif
