//===--------------------- RetireControlUnitStatistics.cpp ---------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the RetireControlUnitStatistics interface.
///
//===----------------------------------------------------------------------===//

#include "RetireControlUnitStatistics.h"
#include "llvm/Support/Format.h"

using namespace llvm;

namespace mca {

void RetireControlUnitStatistics::onInstructionEvent(
    const HWInstructionEvent &Event) {
  if (Event.Type == HWInstructionEvent::Retired)
    ++NumRetired;
}

void RetireControlUnitStatistics::printView(llvm::raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  TempStream << "\n\nRetire Control Unit - "
             << "number of cycles where we saw N instructions retired:\n";
  TempStream << "[# retired], [# cycles]\n";

  for (const std::pair<unsigned, unsigned> &Entry : RetiredPerCycle) {
    TempStream << " " << Entry.first;
    if (Entry.first < 10)
      TempStream << ",           ";
    else
      TempStream << ",          ";
    TempStream << Entry.second << "  ("
               << format("%.1f", ((double)Entry.second / NumCycles) * 100.0)
               << "%)\n";
  }

  TempStream.flush();
  OS << Buffer;
}

} // namespace mca
