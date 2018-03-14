//===--------------------- SummaryView.cpp -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the functionalities used by the SummaryView to print
/// the report information.
///
//===----------------------------------------------------------------------===//

#include "SummaryView.h"
#include "llvm/CodeGen/TargetSchedule.h"

namespace mca {

using namespace llvm;

void SummaryView::printSummary(raw_ostream &OS) const {
  unsigned Iterations = Source.getNumIterations();
  unsigned Instructions = Source.size();
  unsigned TotalInstructions = Instructions * Iterations;
  double IPC = (double)TotalInstructions / TotalCycles;

  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "Iterations:     " << Iterations;
  TempStream << "\nInstructions:   " << TotalInstructions;
  TempStream << "\nTotal Cycles:   " << TotalCycles;
  TempStream << "\nDispatch Width: " << DispatchWidth;
  TempStream << "\nIPC:            " << format("%.2f", IPC) << '\n';
  TempStream.flush();
  OS << Buffer;
}

void SummaryView::printInstructionInfo(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  const MCSchedModel &SM = STI.getSchedModel();
  unsigned Instructions = Source.size();

  TempStream << "\n\nInstruction Info:\n";
  TempStream << "[1]: #uOps\n[2]: Latency\n[3]: RThroughput\n"
             << "[4]: MayLoad\n[5]: MayStore\n[6]: HasSideEffects\n\n";

  TempStream << "[1]    [2]    [3]    [4]    [5]    [6]\tInstructions:\n";
  for (unsigned I = 0, E = Instructions; I < E; ++I) {
    const MCInst &Inst = Source.getMCInstFromIndex(I);
    const MCInstrDesc &MCDesc = MCII.get(Inst.getOpcode());
    const MCSchedClassDesc &SCDesc =
        *SM.getSchedClassDesc(MCDesc.getSchedClass());

    unsigned NumMicroOpcodes = SCDesc.NumMicroOps;
    unsigned Latency = MCSchedModel::computeInstrLatency(STI, SCDesc);
    Optional<double> RThroughput =
        MCSchedModel::getReciprocalThroughput(STI, SCDesc);

    TempStream << ' ' << NumMicroOpcodes << "    ";
    if (NumMicroOpcodes < 10)
      TempStream << "  ";
    else if (NumMicroOpcodes < 100)
      TempStream << ' ';
    TempStream << Latency << "   ";
    if (Latency < 10)
      TempStream << "  ";
    else if (Latency < 100)
      TempStream << ' ';

    if (RThroughput.hasValue()) {
      double RT = RThroughput.getValue();
      TempStream << format("%.2f", RT) << ' ';
      if (RT < 10.0)
        TempStream << "  ";
      else if (RT < 100.0)
        TempStream << ' ';
    } else {
      TempStream << " -     ";
    }
    TempStream << (MCDesc.mayLoad() ? " *     " : "       ");
    TempStream << (MCDesc.mayStore() ? " *     " : "       ");
    TempStream << (MCDesc.hasUnmodeledSideEffects() ? " * " : "   ");
    MCIP.printInst(&Inst, TempStream, "", STI);
    TempStream << '\n';
  }

  TempStream.flush();
  OS << Buffer;
}

} // namespace mca.
