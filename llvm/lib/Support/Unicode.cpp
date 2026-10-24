//===- llvm/Support/Unicode.cpp - Unicode character properties  -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions that allow querying certain properties of
// Unicode characters.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Unicode.h"
#include "UnicodeCharSetsGenerated.h"
#include "llvm/Support/ConvertUTF.h"

namespace llvm {
namespace sys {
namespace unicode {

const CharProperties &charProperties(int UCS) {
  // We know 0x10FFFF is a non-character and treat any out-of-range code point
  // the same.
  UCS = std::min<unsigned>(UCS, 0x10FFFF);

  namespace T = CharPropertyTables;
  auto A = T::Lookup1[UCS >> 12];
  auto B = T::Lookup2[A][(UCS >> 8) & 0xf];
  auto C = T::Lookup3[B][(UCS >> 4) & 0xf];
  auto D = T::Lookup4[C][(UCS >> 2) & 0x3];
  auto E = T::Lookup5[D][UCS & 0x3];
  return T::Lookup6[E];
}

/// Gets the number of positions a character is likely to occupy when output
/// on a terminal ("character width"). This depends on the implementation of the
/// terminal, and there's no standard definition of character width.
/// The implementation defines it in a way that is expected to be compatible
/// with a generic Unicode-capable terminal.
/// \return Character width:
///   * ErrorNonPrintableCharacter (-1) for non-printable characters (as
///     identified by isPrintable);
///   * 0 for non-spacing and enclosing combining marks;
///   * 2 for CJK characters excluding halfwidth forms;
///   * 1 for all remaining characters.
static inline int charWidth(int UCS) {
  if (!isPrintable(UCS))
    return ErrorNonPrintableCharacter;

  const CharProperties &P = charProperties(UCS);
  if (P.isCombining())
    return 0;

  if (P.isDoubleWidth())
    return 2;

  return 1;
}

static bool isprintableascii(char c) { return c > 31 && c < 127; }

int columnWidthUTF8(StringRef Text) {
  unsigned ColumnWidth = 0;
  unsigned Length;
  for (size_t i = 0, e = Text.size(); i < e; i += Length) {
    Length = getNumBytesForUTF8(Text[i]);

    // fast path for ASCII characters
    if (Length == 1) {
      if (!isprintableascii(Text[i]))
        return ErrorNonPrintableCharacter;
      ColumnWidth += 1;
      continue;
    }

    if (Length <= 0 || i + Length > Text.size())
      return ErrorInvalidUTF8;
    UTF32 buf[1];
    const UTF8 *Start = reinterpret_cast<const UTF8 *>(Text.data() + i);
    UTF32 *Target = &buf[0];
    if (conversionOK != ConvertUTF8toUTF32(&Start, Start + Length, &Target,
                                           Target + 1, strictConversion))
      return ErrorInvalidUTF8;
    int Width = charWidth(buf[0]);
    if (Width < 0)
      return ErrorNonPrintableCharacter;
    ColumnWidth += Width;
  }
  return ColumnWidth;
}

} // namespace unicode
} // namespace sys
} // namespace llvm
