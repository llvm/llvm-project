//===- Format.cpp - Efficient printf-style formatting for streams -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the non-template part of Format.h, which is used to
// provide a type-safe-ish interface to printf-style formatting.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Format.h"

namespace {
/// Enum representation of a printf-style length specifier.
enum ArgLength : char {
  /// Corresponds to 'hh' length specifier.
  AL_ShortShort,
  /// Corresponds to 'h' length specifier.
  AL_Short,
  /// Corresponds to default length specifier.
  AL_Default,
  /// Corresponds to 'l' length specifier.
  AL_Long,
  /// Corresponds to 'll' length specifier.
  AL_LongLong,
  /// Corresponds to 'j' length specifier.
  AL_IntMax,
  /// Corresponds to 'z' length specifier.
  AL_Size,
  /// Corresponds to 't' length specifier.
  AL_Ptrdiff,
  /// Corresponds to 'L' length specifier.
  AL_LongDouble,
  /// First invalid value of \p ArgLength.
  AL_End,
};

/// Enum representation of a printf-style specifier.
enum SpecifierChar : char {
  /// Corresponds to any of 'd', 'i', 'u', 'o', 'x' or 'X' specifiers.
  SC_Int,
  /// Corresponds to any of 'f', 'F', 'e', 'E', 'g', 'G', 'a' or 'A' specifiers.
  SC_Float,
  /// Corresponds to 'c' specifier.
  SC_Char,
  /// Corresponds to 's' specifier.
  SC_String,
  /// Corresponds to 'p' specifier.
  SC_VoidPointer,
  /// Corresponds to 'n' specifier.
  SC_Count,
  /// First invalid value of \p SpecifierChar.
  SC_End,
};

constexpr uint64_t specifierBit(char C) { return 1 << (C - 0x40); }

template <size_t N>
constexpr /* consteval */ uint64_t specifierMask(const char (&Specifiers)[N]) {
  uint64_t Mask = 0;
  for (const char *I = std::begin(Specifiers); I != std::end(Specifiers); ++I) {
    if (*I == 0)
      break;
    Mask |= specifierBit(*I);
  }
  return Mask;
}

constexpr auto ST_Unknown = llvm::PrintfStyleFormatReader::ST_Unknown;
constexpr auto ST_WideChar = llvm::PrintfStyleFormatReader::ST_WideChar;
constexpr auto ST_Int = llvm::PrintfStyleFormatReader::ST_Int;
constexpr auto ST_Long = llvm::PrintfStyleFormatReader::ST_Long;
constexpr auto ST_LongLong = llvm::PrintfStyleFormatReader::ST_LongLong;
constexpr auto ST_IntMax = llvm::PrintfStyleFormatReader::ST_IntMax;
constexpr auto ST_Size = llvm::PrintfStyleFormatReader::ST_Size;
constexpr auto ST_Ptrdiff = llvm::PrintfStyleFormatReader::ST_Ptrdiff;
constexpr auto ST_Double = llvm::PrintfStyleFormatReader::ST_Double;
constexpr auto ST_LongDouble = llvm::PrintfStyleFormatReader::ST_LongDouble;
constexpr auto ST_CString = llvm::PrintfStyleFormatReader::ST_CString;
constexpr auto ST_WideCString = llvm::PrintfStyleFormatReader::ST_WideCString;
constexpr auto ST_VoidPointer = llvm::PrintfStyleFormatReader::ST_VoidPointer;
constexpr auto ST_Count_Char = llvm::PrintfStyleFormatReader::ST_Count_Char;
constexpr auto ST_Count_Short = llvm::PrintfStyleFormatReader::ST_Count_Short;
constexpr auto ST_Count_Int = llvm::PrintfStyleFormatReader::ST_Count_Int;
constexpr auto ST_Count_Long = llvm::PrintfStyleFormatReader::ST_Count_Long;
constexpr auto ST_Count_LongLong =
    llvm::PrintfStyleFormatReader::ST_Count_LongLong;
constexpr auto ST_Count_IntMax = llvm::PrintfStyleFormatReader::ST_Count_IntMax;
constexpr auto ST_Count_Size = llvm::PrintfStyleFormatReader::ST_Count_Size;
constexpr auto ST_Count_Ptrdiff =
    llvm::PrintfStyleFormatReader::ST_Count_Ptrdiff;

llvm::PrintfStyleFormatReader::SpecifierType SpecifierTable[SC_End][AL_End] = {
    {
        // SC_Int
        ST_Int,
        ST_Int,
        ST_Int,
        ST_Long,
        ST_LongLong,
        ST_IntMax,
        ST_Size,
        ST_Ptrdiff,
        ST_Unknown,
    },
    {
        // SC_Float
        ST_Unknown,
        ST_Unknown,
        ST_Double,
        ST_Double,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
        ST_LongDouble,
    },
    {
        // SC_Char
        ST_Unknown,
        ST_Unknown,
        ST_Int,
        ST_WideChar,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
    },
    {
        // SC_String
        ST_Unknown,
        ST_Unknown,
        ST_CString,
        ST_WideCString,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
    },
    {
        // SC_VoidPointer
        ST_Unknown,
        ST_Unknown,
        ST_VoidPointer,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
        ST_Unknown,
    },
    {
        // SC_Count
        ST_Count_Char,
        ST_Count_Short,
        ST_Count_Int,
        ST_Count_Long,
        ST_Count_LongLong,
        ST_Count_IntMax,
        ST_Count_Size,
        ST_Count_Ptrdiff,
        ST_Unknown,
    },
};
} // namespace

namespace llvm {

void PrintfStyleFormatReader::refillSpecifierQueue() {
  if (auto PercentPtr = strchr(Fmt, '%')) {
    Fmt = PercentPtr;
  } else {
    SpecifierQueue.push_back(ST_EndOfFormatString);
    return;
  }

  if (*++Fmt == '%') {
    // %% case: skip and try again
    ++Fmt;
    refillSpecifierQueue();
    return;
  }

  // Push ST_Unknown to SpecifierQueue. If we bail out early, this is what
  // the caller gets. Fill in real specifiers to Specifiers: if we
  // successfully get to the end, then swap Specifiers with SpecifierQueue.
  SpecifierQueue.push_back(ST_Unknown);
  llvm::SmallVector<SpecifierType, 3> Specifiers;

  // Bitfield keeping track of which specifier characters are allowed given
  // flags and precision settings. Each bit tells whether ascii character
  // 0x40 + <bit index> is allowed as a specifier. '%', which has an ASCII value
  // less than 0x40 and does not allow any customization, is handled by a check
  // above. The starting value contains all standard specifiers.
  uint64_t ValidSpecifiers = specifierMask("diuoxXfFeEgGaAcspn");

  // update specifier mask based on flags
  bool ReadAllFlags = false;
  while (!ReadAllFlags) {
    switch (*Fmt) {
    case '+':
    case '-':
    case ' ':
      // valid for all specifiers
      ++Fmt;
      break;
    case '#':
      ValidSpecifiers &= specifierMask("xXaAeEfFgG");
      ++Fmt;
      break;
    case '0':
      ValidSpecifiers &= specifierMask("diouxXaAeEfFgG");
      ++Fmt;
      break;
    default:
      ReadAllFlags = true;
      break;
    }
  }

  // skip width
  if (*Fmt == '*') {
    Specifiers.push_back(ST_Int);
    ++Fmt;
  } else
    while (*Fmt >= '0' && *Fmt <= '9')
      ++Fmt;

  // test precision
  if (*Fmt == '.') {
    ValidSpecifiers &= specifierMask("diouxXaAeEfFgGs");
    ++Fmt;
    if (*Fmt == '*') {
      Specifiers.push_back(ST_Int);
      ++Fmt;
    } else
      while (*Fmt >= '0' && *Fmt <= '9')
        ++Fmt;
  }

  // parse length
  bool FoundLength = false;
  ArgLength AL = AL_Default;
  while (!FoundLength) {
    ArgLength NewAL;
    switch (*Fmt) {
    case 'h':
      NewAL = AL_Short;
      break;
    case 'l':
      NewAL = AL_Long;
      break;
    case 'j':
      NewAL = AL_IntMax;
      break;
    case 'z':
      NewAL = AL_Size;
      break;
    case 't':
      NewAL = AL_Ptrdiff;
      break;
    case 'L':
      NewAL = AL_LongDouble;
      break;
    default:
      FoundLength = true;
      continue;
    }

    if (NewAL == AL_Long && AL == AL_Long)
      AL = AL_LongLong;
    else if (NewAL == AL_Short && AL == AL_Short)
      AL = AL_ShortShort;
    else if (AL == AL_Default)
      AL = NewAL;
    else
      return;
    ++Fmt;
  }

  // parse specifier; verify that the character is a valid specifier given
  // restrictions imposed by by the use of flags and precision values
  char Next = *Fmt;
  ++Fmt;
  if (Next < 0x40 || (specifierBit(Next) & ValidSpecifiers) == 0)
    return;

  SpecifierChar SC;
  switch (Next) {
  case 'd':
  case 'i':
  case 'u':
  case 'o':
  case 'x':
  case 'X':
    SC = SC_Int;
    break;

  case 'a':
  case 'A':
  case 'e':
  case 'E':
  case 'f':
  case 'F':
  case 'g':
  case 'G':
    SC = SC_Float;
    break;

  case 'c':
    SC = SC_Char;
    break;

  case 's':
    SC = SC_String;
    break;

  case 'p':
    SC = SC_VoidPointer;
    break;

  case 'n':
    SC = SC_Count;
    break;

  default:
    return;
  }

  auto Spec = SpecifierTable[SC][AL];
  if (Spec == ST_Unknown)
    return;

  Specifiers.push_back(Spec);
  std::reverse(Specifiers.begin(), Specifiers.end());
  std::swap(Specifiers, SpecifierQueue);
}

const char *PrintfStyleFormatReader::ensureCompatible(const char *Expected,
                                                      const char *Fmt) {
  PrintfStyleFormatReader EFR(Expected);
  PrintfStyleFormatReader FFR(Fmt);
  SpecifierType EST;
  do {
    EST = EFR.nextSpecifier();
    if (EST != FFR.nextSpecifier())
      return Expected;
  } while (EST);
  return Fmt;
}

} // namespace llvm
