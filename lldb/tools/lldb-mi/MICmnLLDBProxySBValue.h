//===-- MICmnLLDBProxySBValue.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Third Party Headers:
#include "lldb/API/SBValue.h"

// In-house headers:
#include "MIDataTypes.h"

// Declarations:
class CMIUtilString;

//++
//============================================================================
// Details: MI proxy wrapper class to lldb::SBValue. The class provides
// functionality
//          to assist in the use of SBValue's particular function usage.
//--
class CMICmnLLDBProxySBValue {
  // Statics:
public:
  static bool GetValueAsSigned(const lldb::SBValue &vrValue, MIint64 &vwValue);
  static bool GetValueAsUnsigned(const lldb::SBValue &vrValue,
                                 MIuint64 &vwValue);
  static bool GetCString(const lldb::SBValue &vrValue,
                         CMIUtilString &vwCString);
};
