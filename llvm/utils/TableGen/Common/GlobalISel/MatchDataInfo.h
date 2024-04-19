//===- MatchDataInfo.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Contains utilities related to handling "match data" for GlobalISel
///  Combiners. Match data allows for setting some arbitrary data in the "match"
///  phase and pass it down to the "apply" phase.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_MIRPATTERNS_MATCHDATAINFO_H
#define LLVM_UTILS_MIRPATTERNS_MATCHDATAINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace llvm {

class raw_ostream;

namespace gi {

/// Represents MatchData defined by the match stage and required by the apply
/// stage.
///
/// This allows the plumbing of arbitrary data from C++ predicates between the
/// stages.
///
/// When this class is initially created, it only has a pattern symbol and a
/// type. When all of the MatchDatas declarations of a given pattern have been
/// parsed, `AssignVariables` must be called to assign storage variable names to
/// each MatchDataInfo.
class MatchDataInfo {
  StringRef PatternSymbol;
  StringRef Type;
  std::string VarName;

public:
  static constexpr StringLiteral StructTypeName = "MatchInfosTy";
  static constexpr StringLiteral StructName = "MatchInfos";

  MatchDataInfo(StringRef PatternSymbol, StringRef Type)
      : PatternSymbol(PatternSymbol), Type(Type.trim()) {}

  StringRef getPatternSymbol() const { return PatternSymbol; };
  StringRef getType() const { return Type; };

  bool hasVariableName() const { return !VarName.empty(); }
  void setVariableName(StringRef Name) { VarName = Name; }
  StringRef getVariableName() const;

  std::string getQualifiedVariableName() const {
    return StructName.str() + "." + getVariableName().str();
  }

  void print(raw_ostream &OS) const;
  void dump() const;
};

/// Pool of type -> variables used to emit MatchData variables declarations.
///
/// e.g. if the map contains "int64_t" -> ["MD0", "MD1"], then two variable
/// declarations must be emitted: `int64_t MD0` and `int64_t MD1`.
///
/// This has a static lifetime and will outlive all the `MatchDataInfo` objects
/// by design. It needs a static lifetime so the backends can emit variable
/// declarations after processing all the inputs.
extern StringMap<std::vector<std::string>> AllMatchDataVars;

/// Assign variable names to all MatchDatas used by a pattern. This must be
/// called after all MatchData decls have been parsed for a given processing
/// unit (e.g. a combine rule)
///
/// Requires an array of MatchDataInfo so we can handle cases where a pattern
/// uses multiple instances of the same MatchData type.
///
/// Writes to \ref AllMatchDataVars.
void AssignMatchDataVariables(MutableArrayRef<MatchDataInfo> Infos);

} // namespace gi
} // end namespace llvm

#endif // ifndef LLVM_UTILS_MIRPATTERNS_MATCHDATAINFO_H
