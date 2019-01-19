//===-- MICmdArgContext.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// In-house headers:
#include "MIUtilString.h"

//++
//============================================================================
// Details: MI common code class. Command arguments and options string. Holds
//          the context string.
//          Based on the Interpreter pattern.
//--
class CMICmdArgContext {
  // Methods:
public:
  /* ctor */ CMICmdArgContext();
  /* ctor */ CMICmdArgContext(const CMIUtilString &vrCmdLineArgsRaw);
  //
  const CMIUtilString &GetArgsLeftToParse() const;
  size_t GetNumberArgsPresent() const;
  CMIUtilString::VecString_t GetArgs() const;
  bool IsEmpty() const;
  bool RemoveArg(const CMIUtilString &vArg);
  bool RemoveArgAtPos(const CMIUtilString &vArg, size_t nArgIndex);
  //
  CMICmdArgContext &operator=(const CMICmdArgContext &vOther);

  // Overridden:
public:
  // From CMIUtilString
  /* dtor */ virtual ~CMICmdArgContext();

  // Attributes:
private:
  CMIUtilString m_strCmdArgsAndOptions;
};
