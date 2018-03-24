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
#include "llvm/Support/Format.h"

namespace mca {

#define DEBUG_TYPE "llvm-mca"

using namespace llvm;

void SummaryView::printView(raw_ostream &OS) const {
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
} // namespace mca.
