//===- llvm/Support/Unicode.h - Unicode character properties  -*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines functions that allow querying certain properties of Unicode
// characters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_UNICODE_H
#define LLVM_SUPPORT_UNICODE_H

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Compiler.h"
#include <optional>
#include <string>

namespace llvm {
class StringRef;

namespace sys {
namespace unicode {

enum class CharCategory : unsigned {
  Lu, // Letter, Uppercase
  Ll, // Letter, Lowercase
  Lt, // Letter, Titlecase
  Lm, // Letter, Modifier
  Lo, // Letter, Other
  Mn, // Mark, Non-Spacing
  Mc, // Mark, Spacing Combining
  Me, // Mark, Enclosing
  Nd, // Number, Decimal Digit
  Nl, // Number, Letter
  No, // Number, Other
  Pc, // Punctuation, Connector
  Pd, // Punctuation, Dash
  Ps, // Punctuation, Open
  Pe, // Punctuation, Close
  Pi, // Punctuation, Initial Quote
  Pf, // Punctuation, Final Quote
  Po, // Punctuation, Other
  Sm, // Symbol, Math
  Sc, // Symbol, Currency
  Sk, // Symbol, Modifier
  So, // Symbol, Other
  Zs, // Separator, Space
  Zl, // Separator, Line
  Zp, // Separator, Paragraph
  Co, // Other, Private Use
  Cf, // Other, Format
  Cc, // Other, Control
  Cs, // Other, Surrogate
  Cn, // Other, Not Assigned
};

enum class EastAsianWidth : unsigned {
  N,  // Neutral
  Na, // East Asian Narrow
  H,  // East Asian Halfwidth
  A,  // East Asian Ambiguous
  F,  // East Asian Fullwidth
  W,  // East Asian Wide
};

struct CharProperties {
  CharCategory Gc : 5;
  EastAsianWidth Ea : 3;
  unsigned XIDStart : 1;
  unsigned XIDContinue : 1;
  unsigned MathStart : 1;
  unsigned MathContinue : 1;

  bool isCharacter() const { return Gc <= CharCategory::Cc; }

  bool isCombining() const {
    return Gc == CharCategory::Me || Gc == CharCategory::Mn;
  }

  bool isControl() const { return Gc == CharCategory::Cc; }

  bool isDoubleWidth() const { return Ea >= EastAsianWidth::F; }

  bool isFormatting() const { return Gc == CharCategory::Cf; }

  bool isPrintable(int UCS) const {
    return Gc <= CharCategory::Zs || UCS == 0x00AD;
  }
};

enum ColumnWidthErrors {
  ErrorInvalidUTF8 = -2,
  ErrorNonPrintableCharacter = -1
};

/// Return the character properties of the given Unicode code point. If UCS is
/// out of range, the character properties will match that of any in-range
/// non-character code point.
LLVM_ABI LLVM_READONLY const CharProperties &charProperties(int UCS);

/// Unicode code points that are considered characters. This function
/// returns true for any code point defined by Unicode as a character, false for
/// any code point defined as a non-character, as reserved, as a surrogate code
/// point, or not defined at all.
inline bool isCharacter(int UCS) { return charProperties(UCS).isCharacter(); }

/// Determines if a character is likely to be displayed correctly on the
/// terminal. Exact implementation would have to depend on the specific
/// terminal, so we define the semantic that should be suitable for generic case
/// of a terminal capable to output Unicode characters.
///
/// Printable codepoints are those in the categories L, M, N, P, S and Zs. In
/// addition, U+00AD SOFT HYPHEN is also considered printable, as it's actually
/// displayed on most terminals.
///
/// \return true if the character is considered printable.
inline bool isPrintable(int UCS) {
  return charProperties(UCS).isPrintable(UCS);
}

// Control codepoints are codepoints in the Cc category.
inline bool isControl(int UCS) { return charProperties(UCS).isControl(); }

// Formatting codepoints are codepoints in the Cf category.
inline bool isFormatting(int UCS) { return charProperties(UCS).isFormatting(); }

/// Gets the number of positions the UTF8-encoded \p Text is likely to occupy
/// when output on a terminal ("character width"). This depends on the
/// implementation of the terminal, and there's no standard definition of
/// character width.
///
/// The implementation defines it in a way that is expected to be compatible
/// with a generic Unicode-capable terminal.
///
/// \return Character width:
///   * ErrorNonPrintableCharacter (-1) if \p Text contains non-printable
///     characters (as identified by isPrintable);
///   * 0 for each non-spacing and enclosing combining mark;
///   * 2 for each CJK character excluding halfwidth forms;
///   * 1 for each of the remaining characters.
LLVM_ABI int columnWidthUTF8(StringRef Text);

/// Fold input unicode character according the Simple unicode case folding
/// rules.
LLVM_ABI int foldCharSimple(int C);

/// Maps the name or the alias of a Unicode character to its associated
/// codepoints.
/// The names and aliases are derived from UnicodeData.txt and NameAliases.txt
/// For compatibility with the semantics of named character escape sequences in
/// C++, this mapping does an exact match sensitive to casing and spacing.
/// \return The codepoint of the corresponding character, if any.
LLVM_ABI std::optional<char32_t> nameToCodepointStrict(StringRef Name);

struct LooseMatchingResult {
  char32_t CodePoint;
  SmallString<64> Name;
};

LLVM_ABI std::optional<LooseMatchingResult>
nameToCodepointLooseMatching(StringRef Name);

struct MatchForCodepointName {
  std::string Name;
  uint32_t Distance = 0;
  char32_t Value = 0;
};

LLVM_ABI SmallVector<MatchForCodepointName>
nearestMatchesForCodepointName(StringRef Pattern, std::size_t MaxMatchesCount);

} // namespace unicode
} // namespace sys
} // namespace llvm

#endif
