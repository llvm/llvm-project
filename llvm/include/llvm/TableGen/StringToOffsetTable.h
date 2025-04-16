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
  StringToOffsetTable() {
    // Ensure we always put the empty string at offset zero. That lets empty
    // initialization also be zero initialization for offsets into the table.
    GetOrAddStringOffset("");
  }

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
                          const Twine &Indent = "") const {
    OS << formatv(R"(
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
{0}static constexpr char {1}Storage[] = )",
                  Indent, Name);

    // MSVC silently miscompiles string literals longer than 64k in some
    // circumstances. When the string table is longer, emit it as an array of
    // character literals.
    bool UseChars = AggregateString.size() > (64 * 1024);
    OS << (UseChars ? "{\n" : "\n");

    llvm::ListSeparator LineSep(UseChars ? ",\n" : "\n");
    llvm::SmallVector<StringRef> Strings(split(AggregateString, '\0'));
    // We should always have an empty string at the start, and because these are
    // null terminators rather than separators, we'll have one at the end as
    // well. Skip the end one.
    assert(Strings.front().empty() && "Expected empty initial string!");
    assert(Strings.back().empty() &&
           "Expected empty string at the end due to terminators!");
    Strings.pop_back();
    for (StringRef Str : Strings) {
      OS << LineSep << Indent << "  ";
      // If we can, just emit this as a string literal to be concatenated.
      if (!UseChars) {
        OS << "\"";
        OS.write_escaped(Str);
        OS << "\\0\"";
        continue;
      }

      llvm::ListSeparator CharSep(", ");
      for (char C : Str) {
        OS << CharSep << "'";
        OS.write_escaped(StringRef(&C, 1));
        OS << "'";
      }
      OS << CharSep << "'\\0'";
    }
    OS << LineSep << Indent << (UseChars ? "};" : "  ;");

    OS << formatv(R"(
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

{0}static constexpr llvm::StringTable {1} =
{0}    {1}Storage;
)",
                  Indent, Name);
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
};

} // end namespace llvm

#endif
