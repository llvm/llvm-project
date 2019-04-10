//===-- MICmdCmdTrace.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdTraceStatus    implementation.

// In-house headers:
#include "MICmdCmdTrace.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"

//++
// Details: CMICmdCmdTraceStatus constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdTraceStatus::CMICmdCmdTraceStatus() {
  // Command factory matches this name with that received from the stdin stream
  m_strMiCmd = "trace-status";

  // Required by the CMICmdFactory when registering *this command
  m_pSelfCreatorFn = &CMICmdCmdTraceStatus::CreateSelf;
}

//++
// Details: CMICmdCmdTraceStatus destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdTraceStatus::~CMICmdCmdTraceStatus() {}

//++
// Details: The invoker requires this function. The command does work in this
// function.
//          The command is likely to communicate with the LLDB SBDebugger in
//          here.
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool CMICmdCmdTraceStatus::Execute() {
  // Do nothing
  return MIstatus::success;
}

//++
// Details: The invoker requires this function. The command prepares a MI Record
// Result
//          for the work carried out in the Execute().
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool CMICmdCmdTraceStatus::Acknowledge() {
  const CMICmnMIValueConst miValueConst(MIRSRC(IDS_CMD_ERR_NOT_IMPLEMENTED));
  const CMICmnMIValueResult miValueResult("msg", miValueConst);
  const CMICmnMIResultRecord miRecordResult(
      m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error,
      miValueResult);
  m_miResultRecord = miRecordResult;

  return MIstatus::success;
}

//++
// Details: Required by the CMICmdFactory when registering *this command. The
// factory
//          calls this function to create an instance of *this command.
// Type:    Static method.
// Args:    None.
// Return:  CMICmdBase * - Pointer to a new command.
// Throws:  None.
//--
CMICmdBase *CMICmdCmdTraceStatus::CreateSelf() {
  return new CMICmdCmdTraceStatus();
}
