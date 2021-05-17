//===----------------------- View.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the member functions of the class InstructionView.
///
//===----------------------------------------------------------------------===//

#include <sstream>
#include "Views/InstructionView.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {
namespace mca {

StringRef InstructionView::printInstructionString(const llvm::MCInst &MCI) const {
  InstructionString = "";
  MCIP.printInst(&MCI, 0, "", STI, InstrStream);
  InstrStream.flush();
  // Remove any tabs or spaces at the beginning of the instruction.
  return StringRef(InstructionString).ltrim();
}

json::Value InstructionView::toJSON() const {
  json::Object JO;
  json::Array SourceInfo;
  for (const auto &MCI : getSource()) {
    StringRef Instruction = printInstructionString(MCI);
    SourceInfo.push_back(Instruction.str());
  }
  JO.try_emplace("Instructions", std::move(SourceInfo));

  json::Array Resources;
  const MCSchedModel &SM = STI.getSchedModel();
  for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    unsigned NumUnits = ProcResource.NumUnits;
    // Skip groups and invalid resources with zero units.
    if (ProcResource.SubUnitsIdxBegin || !NumUnits)
      continue;
    for (unsigned J = 0; J < NumUnits; ++J) {
      std::stringstream ResNameStream;
      ResNameStream << ProcResource.Name;
      if (NumUnits > 1)
        ResNameStream << "." << J;
      Resources.push_back(ResNameStream.str());
    }
  }
  JO.try_emplace("Resources", json::Object({{"CPUName", MCPU}, {"Resources", std::move(Resources)}}));

  return JO;
}
} // namespace mca
} // namespace llvm
