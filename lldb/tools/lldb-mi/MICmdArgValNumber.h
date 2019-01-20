//===-- MICmdArgValNumber.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// In-house headers:
#include "MICmdArgValBase.h"

// Declarations:
class CMICmdArgContext;

//++
//============================================================================
// Details: MI common code class. Command argument class. Arguments object
//          needing specialization derived from the CMICmdArgValBase class.
//          An argument knows what type of argument it is and how it is to
//          interpret the options (context) string to find and validate a
//          matching
//          argument and so extract a value from it .
//          Based on the Interpreter pattern.
//--
class CMICmdArgValNumber : public CMICmdArgValBaseTemplate<MIint64> {
  // Enums:
public:
  //++
  //---------------------------------------------------------------------------------
  // Details: CMICmdArgValNumber needs to know what format of argument to look
  // for in
  //          the command options text.
  //--
  enum ArgValNumberFormat_e {
    eArgValNumberFormat_Decimal = (1u << 0),
    eArgValNumberFormat_Hexadecimal = (1u << 1),
    eArgValNumberFormat_Auto =
        ((eArgValNumberFormat_Hexadecimal << 1) -
         1u) ///< Indicates to try and lookup everything up during a query.
  };

  // Methods:
public:
  /* ctor */ CMICmdArgValNumber();
  /* ctor */ CMICmdArgValNumber(
      const CMIUtilString &vrArgName, const bool vbMandatory,
      const bool vbHandleByCmd,
      const MIuint vnNumberFormatMask = eArgValNumberFormat_Decimal);
  //
  bool IsArgNumber(const CMIUtilString &vrTxt) const;

  // Overridden:
public:
  // From CMICmdArgValBase
  /* dtor */ ~CMICmdArgValNumber() override;
  // From CMICmdArgSet::IArg
  bool Validate(CMICmdArgContext &vwArgContext) override;

  // Methods:
private:
  bool ExtractNumber(const CMIUtilString &vrTxt);
  MIint64 GetNumber() const;

  // Attributes:
private:
  MIuint m_nNumberFormatMask;
  MIint64 m_nNumber;
};
