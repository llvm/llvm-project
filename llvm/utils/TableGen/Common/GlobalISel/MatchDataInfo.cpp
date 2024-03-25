//===- MatchDataInfo.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "MatchDataInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace gi {

StringMap<std::vector<std::string>> AllMatchDataVars;

StringRef MatchDataInfo::getVariableName() const {
  assert(hasVariableName());
  return VarName;
}

void MatchDataInfo::print(raw_ostream &OS) const {
  OS << "(MatchDataInfo pattern_symbol:" << PatternSymbol << " type:'" << Type
     << "' var_name:" << (VarName.empty() ? "<unassigned>" : VarName) << ")";
}

void MatchDataInfo::dump() const { print(dbgs()); }

void AssignMatchDataVariables(MutableArrayRef<MatchDataInfo> Infos) {
  static unsigned NextVarID = 0;

  StringMap<unsigned> SeenTypes;
  for (auto &Info : Infos) {
    unsigned &NumSeen = SeenTypes[Info.getType()];
    auto &ExistingVars = AllMatchDataVars[Info.getType()];

    if (NumSeen == ExistingVars.size())
      ExistingVars.push_back("MDInfo" + std::to_string(NextVarID++));

    Info.setVariableName(ExistingVars[NumSeen++]);
  }
}

} // namespace gi
} // namespace llvm
