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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
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
  bool empty() const { return StringOffset.empty(); }
  size_t size() const { return AggregateString.size(); }

  unsigned GetOrAddStringOffset(StringRef Str, bool appendZero = true) {
    auto [II, Inserted] = StringOffset.insert({Str, size()});
    if (Inserted) {
      // Add the string to the aggregate if this is the first time found.
      AggregateString.append(Str.begin(), Str.end());
      if (appendZero)
        AggregateString += '\0';
    }

    return II->second;
  }

  // Returns the offset of `Str` in the table if its preset, else return
  // std::nullopt.
  std::optional<unsigned> GetStringOffset(StringRef Str) const {
    auto II = StringOffset.find(Str);
    if (II == StringOffset.end())
      return std::nullopt;
    return II->second;
  }

  // Emit the string using string literal concatenation, for better readability
  // and searchability.
  void EmitStringLiteralDef(raw_ostream &OS, const Twine &Decl,
                            const Twine &Indent = "  ") const {
    OS << formatv(R"(
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
{0}{1} = )",
                  Indent, Decl);

    for (StringRef Str : split(AggregateString, '\0')) {
      OS << "\n" << Indent << "  \"";
      OS.write_escaped(Str);
      OS << "\\0\"";
    }
    OS << R"(;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
)";
  }

  // Emit the string as one single string.
  void EmitString(raw_ostream &O) const {
    // Escape the string.
    SmallString<256> EscapedStr;
    raw_svector_ostream(EscapedStr).write_escaped(AggregateString);

    O << "    \"";
    unsigned CharsPrinted = 0;
    for (unsigned i = 0, e = EscapedStr.size(); i != e; ++i) {
      if (CharsPrinted > 70) {
        O << "\"\n    \"";
        CharsPrinted = 0;
      }
      O << EscapedStr[i];
      ++CharsPrinted;

      // Print escape sequences all together.
      if (EscapedStr[i] != '\\')
        continue;

      assert(i + 1 < EscapedStr.size() && "Incomplete escape sequence!");
      if (isDigit(EscapedStr[i + 1])) {
        assert(isDigit(EscapedStr[i + 2]) && isDigit(EscapedStr[i + 3]) &&
               "Expected 3 digit octal escape!");
        O << EscapedStr[++i];
        O << EscapedStr[++i];
        O << EscapedStr[++i];
        CharsPrinted += 3;
      } else {
        O << EscapedStr[++i];
        ++CharsPrinted;
      }
    }
    O << "\"";
  }

  /// Emit the string using character literals. MSVC has a limitation that
  /// string literals cannot be longer than 64K.
  void EmitCharArray(raw_ostream &O) {
    assert(AggregateString.find(')') == std::string::npos &&
           "can't emit raw string with closing parens");
    int Count = 0;
    O << ' ';
    for (char C : AggregateString) {
      O << " \'";
      O.write_escaped(StringRef(&C, 1));
      O << "\',";
      Count++;
      if (Count > 14) {
        O << "\n ";
        Count = 0;
      }
    }
    O << '\n';
  }
};

} // end namespace llvm

#endif
