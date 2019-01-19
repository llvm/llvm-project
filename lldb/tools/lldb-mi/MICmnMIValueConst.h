//===-- MICmnMIValueConst.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// In-house headers:
#include "MICmnMIValue.h"

//++
//============================================================================
// Details: MI common code MI Result class. Part of the CMICmnMIValueConstRecord
//          set of objects.
//          The syntax is as follows:
//          result-record ==>  [ token ] "^" result-class ( "," result )* nl
//          token = any sequence of digits
//          * = 0 to many
//          nl = CR | CR_LF
//          result-class ==> "done" | "running" | "connected" | "error" | "exit"
//          result ==> variable "=" value
//          value ==> const | tuple | list
//          const ==> c-string (7 bit iso c string content)
//          tuple ==>  "{}" | "{" result ( "," result )* "}"
//          list ==>  "[]" | "[" value ( "," value )* "]" | "[" result ( ","
//          result )* "]"
//          More information see:
//          http://ftp.gnu.org/old-gnu/Manuals/gdb-5.1.1/html_chapter/gdb_22.html
//
//          The text formed in *this Result class is stripped of any '\n'
//          characters.
//--
class CMICmnMIValueConst : public CMICmnMIValue {
  // Methods:
public:
  /* ctor */ CMICmnMIValueConst(const CMIUtilString &vString);
  /* ctor */ CMICmnMIValueConst(const CMIUtilString &vString,
                                const bool vbNoQuotes);

  // Overridden:
public:
  // From CMICmnBase
  /* dtor */ ~CMICmnMIValueConst() override;

  // Methods:
private:
  bool BuildConst();

  // Attributes:
private:
  static const CMIUtilString ms_constStrDblQuote;
  //
  CMIUtilString m_strPartConst;
  bool m_bNoQuotes; // True = return string not surrounded with quotes, false =
                    // use quotes
};
