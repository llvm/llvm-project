//===--------------------- BackendPrinter.cpp -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the BackendPrinter interface.
///
//===----------------------------------------------------------------------===//

#include "BackendPrinter.h"
#include "View.h"
#include "llvm/CodeGen/TargetSchedule.h"

namespace mca {

using namespace llvm;

void BackendPrinter::printGeneralStatistics(raw_ostream &OS,
                                            unsigned Iterations,
                                            unsigned Cycles,
                                            unsigned Instructions,
                                            unsigned DispatchWidth) const {
  unsigned TotalInstructions = Instructions * Iterations;
  double IPC = (double)TotalInstructions / Cycles;

  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "Iterations:     " << Iterations;
  TempStream << "\nInstructions:   " << TotalInstructions;
  TempStream << "\nTotal Cycles:   " << Cycles;
  TempStream << "\nDispatch Width: " << DispatchWidth;
  TempStream << "\nIPC:            " << format("%.2f", IPC) << '\n';
  TempStream.flush();
  OS << Buffer;
}

void BackendPrinter::printInstructionInfo(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);

  TempStream << "\n\nInstruction Info:\n";
  TempStream << "[1]: #uOps\n[2]: Latency\n[3]: RThroughput\n"
             << "[4]: MayLoad\n[5]: MayStore\n[6]: HasSideEffects\n\n";

  TempStream << "[1]    [2]    [3]    [4]    [5]    [6]\tInstructions:\n";
  for (unsigned I = 0, E = B.getNumInstructions(); I < E; ++I) {
    const MCInst &Inst = B.getMCInstFromIndex(I);
    const InstrDesc &ID = B.getInstrDesc(Inst);
    unsigned NumMicroOpcodes = ID.NumMicroOps;
    unsigned Latency = ID.MaxLatency;
    double RThroughput = B.getRThroughput(ID);
    TempStream << ' ' << NumMicroOpcodes << "    ";
    if (NumMicroOpcodes < 10)
      TempStream << "  ";
    else if (NumMicroOpcodes < 100)
      TempStream << ' ';
    TempStream << Latency << "   ";
    if (Latency < 10.0)
      TempStream << "  ";
    else if (Latency < 100.0)
      TempStream << ' ';
    if (RThroughput) {
      TempStream << format("%.2f", RThroughput) << ' ';
      if (RThroughput < 10.0)
        TempStream << "  ";
      else if (RThroughput < 100.0)
        TempStream << ' ';
    } else {
      TempStream << " -     ";
    }
    TempStream << (ID.MayLoad ? " *     " : "       ");
    TempStream << (ID.MayStore ? " *     " : "       ");
    TempStream << (ID.HasSideEffects ? " * " : "   ");
    MCIP.printInst(&Inst, TempStream, "", B.getSTI());
    TempStream << '\n';
  }

  TempStream.flush();
  OS << Buffer;
}

void BackendPrinter::printReport(llvm::raw_ostream &OS) const {
  unsigned Cycles = B.getNumCycles();
  printGeneralStatistics(OS, B.getNumIterations(), Cycles, B.getNumInstructions(),
                         B.getDispatchWidth());
  printInstructionInfo(OS);

  for (const auto &V : Views)
    V->printView(OS);
}

} // namespace mca.
