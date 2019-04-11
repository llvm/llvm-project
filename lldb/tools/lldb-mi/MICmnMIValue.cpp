//===-- MICmnMIValue.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmnMIValue.h"
#include "MICmnResources.h"

//++
// Details: CMICmnMIValue constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnMIValue::CMICmnMIValue()
    : m_strValue(MIRSRC(IDS_WORD_INVALIDBRKTS)), m_bJustConstructed(true) {}

//++
// Details: CMICmnMIValue destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnMIValue::~CMICmnMIValue() {}

//++
// Details: Return the MI value as a string. The string is a direct result of
//          work done on *this value so if not enough data is added then it is
//          possible to return a malformed value. If nothing has been set or
//          added to *this MI value object then text "<Invalid>" will be
//          returned.
// Type:    Method.
// Args:    None.
// Return:  CMIUtilString & - MI output text.
// Throws:  None.
//--
const CMIUtilString &CMICmnMIValue::GetString() const { return m_strValue; }
