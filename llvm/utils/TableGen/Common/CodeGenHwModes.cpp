//===--- CodeGenHwModes.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Classes to parse and store HW mode information for instruction selection
//===----------------------------------------------------------------------===//

#include "CodeGenHwModes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

StringRef CodeGenHwModes::DefaultModeName = "DefaultMode";

HwMode::HwMode(const Record *R) {
  Name = R->getName();
  Features = std::string(R->getValueAsString("Features"));

  SmallString<128> PredicateCheck;
  raw_svector_ostream OS(PredicateCheck);
  ListSeparator LS(" && ");
  for (const Record *Pred : R->getValueAsListOfDefs("Predicates")) {
    StringRef CondString = Pred->getValueAsString("CondString");
    if (CondString.empty())
      continue;
    OS << LS << '(' << CondString << ')';
  }

  Predicates = std::string(PredicateCheck);
}

LLVM_DUMP_METHOD
void HwMode::dump() const { dbgs() << Name << ": " << Features << '\n'; }

HwModeSelect::HwModeSelect(const Record *R, CodeGenHwModes &CGH) {
  std::vector<const Record *> Modes = R->getValueAsListOfDefs("Modes");
  std::vector<const Record *> Objects = R->getValueAsListOfDefs("Objects");
  for (auto [Mode, Object] : zip_equal(Modes, Objects)) {
    unsigned ModeId = CGH.getHwModeId(Mode);
    Items.emplace_back(ModeId, Object);
  }
}

LLVM_DUMP_METHOD
void HwModeSelect::dump() const {
  dbgs() << '{';
  for (const PairType &P : Items)
    dbgs() << " (" << P.first << ',' << P.second->getName() << ')';
  dbgs() << " }\n";
}

CodeGenHwModes::CodeGenHwModes(const RecordKeeper &RK) : Records(RK) {
  for (const Record *R : Records.getAllDerivedDefinitions("HwMode")) {
    // The default mode needs a definition in the .td sources for TableGen
    // to accept references to it. We need to ignore the definition here.
    if (R->getName() == DefaultModeName)
      continue;
    Modes.emplace_back(R);
    ModeIds.try_emplace(R, Modes.size());
  }

  assert(Modes.size() <= 32 && "number of HwModes exceeds maximum of 32");

  for (const Record *R : Records.getAllDerivedDefinitions("HwModeSelect")) {
    auto P = ModeSelects.emplace(R, HwModeSelect(R, *this));
    assert(P.second);
    (void)P;
  }
}

unsigned CodeGenHwModes::getHwModeId(const Record *R) const {
  if (R->getName() == DefaultModeName)
    return DefaultMode;
  auto F = ModeIds.find(R);
  assert(F != ModeIds.end() && "Unknown mode name");
  return F->second;
}

const HwModeSelect &CodeGenHwModes::getHwModeSelect(const Record *R) const {
  auto F = ModeSelects.find(R);
  assert(F != ModeSelects.end() && "Record is not a \"mode select\"");
  return F->second;
}

LLVM_DUMP_METHOD
void CodeGenHwModes::dump() const {
  dbgs() << "Modes: {\n";
  for (const HwMode &M : Modes) {
    dbgs() << "  ";
    M.dump();
  }
  dbgs() << "}\n";

  dbgs() << "ModeIds: {\n";
  for (const auto &P : ModeIds)
    dbgs() << "  " << P.first->getName() << " -> " << P.second << '\n';
  dbgs() << "}\n";

  dbgs() << "ModeSelects: {\n";
  for (const auto &P : ModeSelects) {
    dbgs() << "  " << P.first->getName() << " -> ";
    P.second.dump();
  }
  dbgs() << "}\n";
}
