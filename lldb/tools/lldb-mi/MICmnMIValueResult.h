//===-- MICmnMIValueResult.h ------------------------------------*- C++ -*-===//
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
// Details: MI common code MI Result class. Part of the
// CMICmnMIValueResultRecord
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
//--
class CMICmnMIValueResult : public CMICmnMIValue {
  // Methods:
public:
  /* ctor */ CMICmnMIValueResult();
  /* ctor */ CMICmnMIValueResult(const CMIUtilString &vVariable,
                                 const CMICmnMIValue &vValue);
  /* ctor */ CMICmnMIValueResult(const CMIUtilString &vVariable,
                                 const CMICmnMIValue &vValue,
                                 const bool vbUseSpacing);
  //
  void Add(const CMIUtilString &vVariable, const CMICmnMIValue &vValue);

  // Overridden:
public:
  // From CMICmnBase
  /* dtor */ ~CMICmnMIValueResult() override;

  // Methods:
private:
  void BuildResult();
  void BuildResult(const CMIUtilString &vVariable, const CMICmnMIValue &vValue);

  // Attributes:
private:
  static const CMIUtilString ms_constStrEqual;
  //
  CMIUtilString m_strPartVariable;
  CMICmnMIValue m_partMIValue;
  bool m_bEmptyConstruction; // True = *this object used constructor with no
                             // parameters, false = constructor with parameters
  bool m_bUseSpacing; // True = put space separators into the string, false = no
                      // spaces used
};
