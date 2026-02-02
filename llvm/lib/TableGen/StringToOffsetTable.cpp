//===- StringToOffsetTable.cpp - Emit a big concatenated string -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"

using namespace llvm;

unsigned StringToOffsetTable::GetOrAddStringOffset(StringRef Str) {
  auto [II, Inserted] = StringOffset.insert({Str, size()});
  if (Inserted) {
    // Add the string to the aggregate if this is the first time found.
    AggregateString.append(Str.begin(), Str.end());
    if (AppendZero)
      AggregateString += '\0';
  }

  return II->second;
}

void StringToOffsetTable::EmitStringTableDef(raw_ostream &OS,
                                             const Twine &Name) const {
  // This generates a `llvm::StringTable` which expects that entries are null
  // terminated. So fail with an error if `AppendZero` is false.
  if (!AppendZero)
    PrintFatalError("llvm::StringTable requires null terminated strings");

  OS << formatv(R"(
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
{} constexpr char {}{}Storage[] =)",
                ClassPrefix.empty() ? "static" : "",
                UsePrefixForStorageMember ? ClassPrefix : "", Name);

  // MSVC silently miscompiles string literals longer than 64k in some
  // circumstances. The build system sets EmitLongStrLiterals to false when it
  // detects that it is targetting MSVC. When that option is false and the
  // string table is longer than 64k, emit it as an array of character
  // literals.
  bool UseChars = !EmitLongStrLiterals && AggregateString.size() > (64 * 1024);
  OS << (UseChars ? "{\n" : "\n");

  ListSeparator LineSep(UseChars ? ",\n" : "\n");
  SmallVector<StringRef> Strings(split(AggregateString, '\0'));
  // We should always have an empty string at the start, and because these are
  // null terminators rather than separators, we'll have one at the end as
  // well. Skip the end one.
  assert(Strings.front().empty() && "Expected empty initial string!");
  assert(Strings.back().empty() &&
         "Expected empty string at the end due to terminators!");
  Strings.pop_back();
  for (StringRef Str : Strings) {
    OS << LineSep << "  ";
    // If we can, just emit this as a string literal to be concatenated.
    if (!UseChars) {
      OS << "\"";
      OS.write_escaped(Str);
      OS << "\\0\"";
      continue;
    }

    ListSeparator CharSep(", ");
    for (char C : Str) {
      OS << CharSep << "'";
      OS.write_escaped(StringRef(&C, 1));
      OS << "'";
    }
    OS << CharSep << "'\\0'";
  }
  OS << LineSep << (UseChars ? "};" : "  ;");

  OS << formatv(R"(
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

{1} llvm::StringTable
{2}{0} = {0}Storage;
)",
                Name, ClassPrefix.empty() ? "static constexpr" : "const",
                ClassPrefix);
}

void StringToOffsetTable::EmitString(raw_ostream &O) const {
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
