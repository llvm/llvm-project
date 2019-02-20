//===-- MICmnLLDBBroadcaster.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmnLLDBBroadcaster.h"

//++
//------------------------------------------------------------------------------------
// Details: CMICmnLLDBBroadcaster constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnLLDBBroadcaster::CMICmnLLDBBroadcaster()
    : lldb::SBBroadcaster("MI driver") {}

//++
//------------------------------------------------------------------------------------
// Details: CMICmnLLDBBroadcaster destructor.
// Type:    Overridable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmnLLDBBroadcaster::~CMICmnLLDBBroadcaster() { Shutdown(); }

//++
//------------------------------------------------------------------------------------
// Details: Initialize resources for *this broadcaster object.
// Type:    Method.
// Args:    None.
// Return:  MIstatus::success - Functionality succeeded.
//          MIstatus::failure - Functionality failed.
// Throws:  None.
//--
bool CMICmnLLDBBroadcaster::Initialize() {
  m_clientUsageRefCnt++;

  if (m_bInitialized)
    return MIstatus::success;

  m_bInitialized = MIstatus::success;

  return m_bInitialized;
}

//++
//------------------------------------------------------------------------------------
// Details: Release resources for *this broadcaster object.
// Type:    Method.
// Args:    None.
// Return:  MIstatus::success - Functionality succeeded.
//          MIstatus::failure - Functionality failed.
// Throws:  None.
//--
bool CMICmnLLDBBroadcaster::Shutdown() {
  if (--m_clientUsageRefCnt > 0)
    return MIstatus::success;

  if (!m_bInitialized)
    return MIstatus::success;

  m_bInitialized = false;

  return MIstatus::success;
}
