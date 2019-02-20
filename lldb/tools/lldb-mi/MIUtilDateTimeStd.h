//===-- MIUtilDateTimeStd.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Third party headers
#include <ctime>

// In-house headers:
#include "MIUtilString.h"

//++
//============================================================================
// Details: MI common code utility class. Used to retrieve system local date
//          time.
//--
class CMIUtilDateTimeStd {
  // Methods:
public:
  /* ctor */ CMIUtilDateTimeStd();

  CMIUtilString GetDate();
  CMIUtilString GetTime();
  CMIUtilString GetDateTimeLogFilename();

  // Overrideable:
public:
  // From CMICmnBase
  /* dtor */ virtual ~CMIUtilDateTimeStd();

  // Attributes:
private:
  std::time_t m_rawTime;
  char m_pScratch[16];
};
