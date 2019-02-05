//===-- MICmnMIValueResult.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmnMIValueResult.h"
#include "MICmnResources.h"

// Instantiations:
const CMIUtilString CMICmnMIValueResult::ms_constStrEqual("=");

//++
//------------------------------------------------------------------------------------
// Details: CMICmnMIValueResult constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnMIValueResult::CMICmnMIValueResult() : m_bEmptyConstruction(true) {}

//++
//------------------------------------------------------------------------------------
// Details: CMICmnMIValueResult constructor.
// Type:    Method.
// Args:    vrVariable  - (R) MI value's name.
//          vrValue     - (R) The MI value.
// Return:  None.
// Throws:  None.
//--
CMICmnMIValueResult::CMICmnMIValueResult(const CMIUtilString &vrVariable,
                                         const CMICmnMIValue &vrValue)
    : m_strPartVariable(vrVariable), m_partMIValue(vrValue),
      m_bEmptyConstruction(false), m_bUseSpacing(false) {
  BuildResult();
}

//++
//------------------------------------------------------------------------------------
// Details: CMICmnMIValueResult constructor.
// Type:    Method.
// Args:    vrVariable      - (R) MI value's name.
//          vrValue         - (R) The MI value.
//          vbUseSpacing    - (R) True = put space separators into the string,
//          false = no spaces used.
// Return:  None.
// Throws:  None.
//--
CMICmnMIValueResult::CMICmnMIValueResult(const CMIUtilString &vrVariable,
                                         const CMICmnMIValue &vrValue,
                                         const bool vbUseSpacing)
    : m_strPartVariable(vrVariable), m_partMIValue(vrValue),
      m_bEmptyConstruction(false), m_bUseSpacing(vbUseSpacing) {
  BuildResult();
}

//++
//------------------------------------------------------------------------------------
// Details: CMICmnMIValueResult destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnMIValueResult::~CMICmnMIValueResult() {}

//++
//------------------------------------------------------------------------------------
// Details: Build the MI value result string.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
void CMICmnMIValueResult::BuildResult() {
  const char *pFormat = m_bUseSpacing ? "%s %s %s" : "%s%s%s";
  m_strValue = CMIUtilString::Format(pFormat, m_strPartVariable.c_str(),
                                     ms_constStrEqual.c_str(),
                                     m_partMIValue.GetString().c_str());
}

//++
//------------------------------------------------------------------------------------
// Details: Build the MI value result string.
// Type:    Method.
// Args:    vrVariable  - (R) MI value's name.
//          vrValue     - (R) The MI value.
// Return:  None.
// Throws:  None.
//--
void CMICmnMIValueResult::BuildResult(const CMIUtilString &vVariable,
                                      const CMICmnMIValue &vValue) {
  const char *pFormat = m_bUseSpacing ? "%s, %s %s %s" : "%s,%s%s%s";
  m_strValue = CMIUtilString::Format(
      pFormat, m_strValue.c_str(), vVariable.c_str(), ms_constStrEqual.c_str(),
      vValue.GetString().c_str());
}

//++
//------------------------------------------------------------------------------------
// Details: Append another MI value object to *this MI value result.
// Type:    Method.
// Args:    vrVariable  - (R) MI value's name.
//          vrValue     - (R) The MI value.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
void CMICmnMIValueResult::Add(const CMIUtilString &vrVariable,
                              const CMICmnMIValue &vrValue) {
  if (!m_bEmptyConstruction)
    BuildResult(vrVariable, vrValue);
  else {
    m_bEmptyConstruction = false;
    m_strPartVariable = vrVariable;
    m_partMIValue = vrValue;
    BuildResult();
  }
}
