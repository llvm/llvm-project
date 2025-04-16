//===- StringToOffsetTable.h - Emit a big concatenated string ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_STRINGTOOFFSETTABLE_H
#define LLVM_TABLEGEN_STRINGTOOFFSETTABLE_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include <optional>

namespace llvm {

/// StringToOffsetTable - This class uniques a bunch of nul-terminated strings
/// and keeps track of their offset in a massive contiguous string allocation.
/// It can then output this string blob and use indexes into the string to
/// reference each piece.
class StringToOffsetTable {
  StringMap<unsigned> StringOffset;
  std::string AggregateString;

public:
  StringToOffsetTable() {
    // Ensure we always put the empty string at offset zero. That lets empty
    // initialization also be zero initialization for offsets into the table.
    GetOrAddStringOffset("");
  }

  bool empty() const { return StringOffset.empty(); }
  size_t size() const { return AggregateString.size(); }

  unsigned GetOrAddStringOffset(StringRef Str, bool appendZero = true);

  // Returns the offset of `Str` in the table if its preset, else return
  // std::nullopt.
  std::optional<unsigned> GetStringOffset(StringRef Str) const {
    auto II = StringOffset.find(Str);
    if (II == StringOffset.end())
      return std::nullopt;
    return II->second;
  }

  // Emit a string table definition with the provided name and indent.
  //
  // When possible, this uses string-literal concatenation to emit the string
  // contents in a readable and searchable way. However, for (very) large string
  // tables MSVC cannot reliably use string literals and so there we use a large
  // character array. We still use a line oriented emission and add comments to
  // provide searchability even in this case.
  //
  // The string table, and its input string contents, are always emitted as both
  // `static` and `constexpr`. Both `Name` and (`Name` + "Storage") must be
  // valid identifiers to declare.
  void EmitStringTableDef(raw_ostream &OS, const Twine &Name,
                          const Twine &Indent = "") const;

  // Emit the string as one single string.
  void EmitString(raw_ostream &O) const;
};

} // end namespace llvm

#endif
