//===-- MICmdArgValConsume.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmdArgValConsume.h"
#include "MICmdArgContext.h"

//++
// Details: CMICmdArgValConsume constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValConsume::CMICmdArgValConsume() {}

//++
// Details: CMICmdArgValConsume constructor.
// Type:    Method.
// Args:    vrArgName       - (R) Argument's name to search by.
//          vbMandatory     - (R) True = Yes must be present, false = optional
//          argument.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValConsume::CMICmdArgValConsume(const CMIUtilString &vrArgName,
                                         const bool vbMandatory)
    : CMICmdArgValBaseTemplate(vrArgName, vbMandatory, true) {}

//++
// Details: CMICmdArgValConsume destructor.
// Type:    Overidden.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdArgValConsume::~CMICmdArgValConsume() {}

//++
// Details: Parse the command's argument options string and try to extract the
// value *this
//          argument is looking for.
// Type:    Overridden.
// Args:    vwArgContext    - (R) The command's argument options string.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool CMICmdArgValConsume::Validate(CMICmdArgContext &vwArgContext) {
  if (vwArgContext.IsEmpty())
    return m_bMandatory ? MIstatus::failure : MIstatus::success;

  // Consume the optional file, line, linenum arguments till the mode '--'
  // argument
  const CMIUtilString::VecString_t vecOptions(vwArgContext.GetArgs());
  CMIUtilString::VecString_t::const_iterator it = vecOptions.begin();
  while (it != vecOptions.end()) {
    const CMIUtilString &rTxt(*it);

    if (rTxt == "--") {
      m_bFound = true;
      m_bValid = true;
      if (!vwArgContext.RemoveArg(rTxt))
        return MIstatus::failure;
      return MIstatus::success;
    }
    // Next
    ++it;
  }

  return MIstatus::failure;
}

//++
// Details: Nothing to examine as we just want to consume the argument or option
// (ignore
//          it).
// Type:    Method.
// Args:    None.
// Return:  bool -  True = yes ok, false = not ok.
// Throws:  None.
//--
bool CMICmdArgValConsume::IsOk() const { return true; }
