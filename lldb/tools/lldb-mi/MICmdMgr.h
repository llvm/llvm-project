//===-- MICmdMgr.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Third party headers
#include <set>

// In-house headers:
#include "MICmdBase.h"
#include "MICmdMgrSetCmdDeleteCallback.h"
#include "MICmnBase.h"
#include "MIUtilSingletonBase.h"

// Declarations:
class CMICmdInterpreter;
class CMICmdFactory;
class CMICmdInvoker;
class CMICmdBase;

//++
//============================================================================
// Details: MI command manager. Oversees command operations, controls command
//          production and the running of commands.
//          Command Invoker, Command Factory and Command Monitor while
//          independent
//          units are overseen/managed by *this manager.
//          A singleton class.
//--
class CMICmdMgr : public CMICmnBase, public MI::ISingleton<CMICmdMgr> {
  friend class MI::ISingleton<CMICmdMgr>;

  // Methods:
public:
  bool Initialize() override;
  bool Shutdown() override;

  bool CmdInterpret(const CMIUtilString &vTextLine, bool &vwbYesValid,
                    bool &vwbCmdNotInCmdFactor, SMICmdData &rwCmdData);
  bool CmdExecute(const SMICmdData &vCmdData);
  bool CmdDelete(SMICmdData vCmdData);
  bool CmdRegisterForDeleteNotification(
      CMICmdMgrSetCmdDeleteCallback::ICallback &vObject);
  bool CmdUnregisterForDeleteNotification(
      CMICmdMgrSetCmdDeleteCallback::ICallback &vObject);

  // Methods:
private:
  /* ctor */ CMICmdMgr();
  /* ctor */ CMICmdMgr(const CMICmdMgr &);
  void operator=(const CMICmdMgr &);

  // Overridden:
public:
  // From CMICmnBase
  /* dtor */ ~CMICmdMgr() override;

  // Attributes:
private:
  CMICmdInterpreter &m_interpretor;
  CMICmdFactory &m_factory;
  CMICmdInvoker &m_invoker;
  CMICmdMgrSetCmdDeleteCallback::CSetClients m_setCmdDeleteCallback;
};
