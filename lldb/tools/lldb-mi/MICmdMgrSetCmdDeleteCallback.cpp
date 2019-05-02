//===-- MICmdMgrSetCmdDeleteCallback.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// In-house headers:
#include "MICmdMgrSetCmdDeleteCallback.h"

namespace CMICmdMgrSetCmdDeleteCallback {

//++
// Details: CSetClients constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CSetClients::CSetClients() : m_bClientUnregistered(false) {}

//++
// Details: CSetClients destructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CSetClients::~CSetClients() {}

//++
// Details: Register an object to be called when a command object is deleted.
// Type:    Method.
// Args:    vObject - (R) A new interested client.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool CSetClients::Register(ICallback &vObject) {
  insert(&vObject);

  return MIstatus::success;
}

//++
// Details: Unregister an object from being called when a command object is
// deleted.
// Type:    Method.
// Args:    vObject - (R) The was interested client.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool CSetClients::Unregister(ICallback &vObject) {
  m_bClientUnregistered = true;
  erase(&vObject);

  return MIstatus::success;
}

//++
// Details: Iterate all interested clients and tell them a command is being
// deleted.
// Type:    Method.
// Args:    vCmd    - (RW) The command to be deleted.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
void CSetClients::Delete(SMICmdData &vCmd) {
  m_bClientUnregistered = false; // Reset
  iterator it = begin();
  while (it != end()) {
    ICallback *pObj = *it;
    pObj->Delete(vCmd);

    if (m_bClientUnregistered) {
      m_bClientUnregistered = false; // Reset
      it = begin();
    } else
      // Next
      ++it;
  }
}

} // namespace CMICmdMgrSetCmdDeleteCallback
