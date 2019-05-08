//===-------------------------- CodeRegion.cpp -----------------*- C++ -* -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods from the CodeRegions interface.
///
//===----------------------------------------------------------------------===//

#include "CodeRegion.h"

namespace llvm {
namespace mca {

CodeRegions::CodeRegions(llvm::SourceMgr &S) : SM(S) {
  // Create a default region for the input code sequence.
  Regions.emplace_back(make_unique<CodeRegion>("Default", SMLoc()));
}

bool CodeRegion::isLocInRange(SMLoc Loc) const {
  if (RangeEnd.isValid() && Loc.getPointer() > RangeEnd.getPointer())
    return false;
  if (RangeStart.isValid() && Loc.getPointer() < RangeStart.getPointer())
    return false;
  return true;
}

void CodeRegions::beginRegion(StringRef Description, SMLoc Loc) {
  assert(!Regions.empty() && "Missing Default region");
  const CodeRegion &CurrentRegion = *Regions.back();
  if (CurrentRegion.startLoc().isValid() && !CurrentRegion.endLoc().isValid()) {
    SM.PrintMessage(Loc, SourceMgr::DK_Warning,
                    "Ignoring invalid region start");
    return;
  }

  // Remove the default region if there are user defined regions.
  if (!CurrentRegion.startLoc().isValid())
    Regions.erase(Regions.begin());
  Regions.emplace_back(make_unique<CodeRegion>(Description, Loc));
}

void CodeRegions::endRegion(SMLoc Loc) {
  assert(!Regions.empty() && "Missing Default region");
  CodeRegion &CurrentRegion = *Regions.back();
  if (CurrentRegion.endLoc().isValid()) {
    SM.PrintMessage(Loc, SourceMgr::DK_Warning,
                    "Ignoring invalid region end");
    return;
  }

  CurrentRegion.setEndLocation(Loc);
}

void CodeRegions::addInstruction(const MCInst &Instruction) {
  const SMLoc &Loc = Instruction.getLoc();
  const auto It =
      std::find_if(Regions.rbegin(), Regions.rend(),
                   [Loc](const UniqueCodeRegion &Region) {
                     return Region->isLocInRange(Loc);
                   });
  if (It != Regions.rend())
    (*It)->addInstruction(Instruction);
}

} // namespace mca
} // namespace llvm
