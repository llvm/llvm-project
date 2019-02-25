//===-- MIUtilString.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Third party headers:
#include <cinttypes>
#include <cstdarg>
#include <string>
#include <vector>

// In-house headers:
#include "MIDataTypes.h"

//++
//============================================================================
// Details: MI common code utility class. Used to help handle text.
//          Derived from std::string
//--
class CMIUtilString : public std::string {
  // Typedefs:
public:
  typedef std::vector<CMIUtilString> VecString_t;

  // Static method:
public:
  static CMIUtilString Format(const char *vFormating, ...);
  static CMIUtilString FormatBinary(const MIuint64 vnDecimal);
  static CMIUtilString FormatValist(const CMIUtilString &vrFormating,
                                    va_list vArgs);
  static bool IsAllValidAlphaAndNumeric(const char *vpText);
  static const char *WithNullAsEmpty(const char *vpText) {
    return vpText ? vpText : "";
  }
  static bool Compare(const CMIUtilString &vrLhs, const CMIUtilString &vrRhs);
  static CMIUtilString ConvertToPrintableASCII(const char vChar,
                                               bool bEscapeQuotes = false);
  static CMIUtilString ConvertToPrintableASCII(const char16_t vChar16,
                                               bool bEscapeQuotes = false);
  static CMIUtilString ConvertToPrintableASCII(const char32_t vChar32,
                                               bool bEscapeQuotes = false);

  // Methods:
public:
  /* ctor */ CMIUtilString();
  /* ctor */ CMIUtilString(const char *vpData);
  /* ctor */ CMIUtilString(const std::string &vrStr);
  //
  bool ExtractNumber(MIint64 &vwrNumber) const;
  CMIUtilString FindAndReplace(const CMIUtilString &vFind,
                               const CMIUtilString &vReplaceWith) const;
  bool IsNumber() const;
  bool IsHexadecimalNumber() const;
  bool IsQuoted() const;
  CMIUtilString RemoveRepeatedCharacters(const char vChar);
  size_t Split(const CMIUtilString &vDelimiter, VecString_t &vwVecSplits) const;
  size_t SplitConsiderQuotes(const CMIUtilString &vDelimiter,
                             VecString_t &vwVecSplits) const;
  size_t SplitLines(VecString_t &vwVecSplits) const;
  CMIUtilString StripCREndOfLine() const;
  CMIUtilString StripCRAll() const;
  CMIUtilString Trim() const;
  CMIUtilString Trim(const char vChar) const;
  size_t FindFirst(const CMIUtilString &vrPattern, size_t vnPos = 0) const;
  size_t FindFirst(const CMIUtilString &vrPattern, bool vbSkipQuotedText,
                   bool &vrwbNotFoundClosedQuote, size_t vnPos = 0) const;
  size_t FindFirstNot(const CMIUtilString &vrPattern, size_t vnPos = 0) const;
  CMIUtilString Escape(bool vbEscapeQuotes = false) const;
  CMIUtilString AddSlashes() const;
  CMIUtilString StripSlashes() const;
  //
  CMIUtilString &operator=(const char *vpRhs);
  CMIUtilString &operator=(const std::string &vrRhs);

  // Overrideable:
public:
  /* dtor */ virtual ~CMIUtilString();

  // Static method:
private:
  static CMIUtilString FormatPriv(const CMIUtilString &vrFormat, va_list vArgs);
  static CMIUtilString ConvertCharValueToPrintableASCII(char vChar,
                                                        bool bEscapeQuotes);

  // Methods:
private:
  bool ExtractNumberFromHexadecimal(MIint64 &vwrNumber) const;
  CMIUtilString RemoveRepeatedCharacters(size_t vnPos, const char vChar);
  size_t FindFirstQuote(size_t vnPos) const;
};
