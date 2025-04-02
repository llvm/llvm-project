//===-- LLDBServerNativeProcess.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerNativeProcess.h"
#include "lldb/Host/common/NativeProcessProtocol.h"

using namespace lldb_private;
using namespace lldb_server;

LLDBServerNativeProcess::LLDBServerNativeProcess(
    NativeProcessProtocol *native_process)
    : m_native_process(native_process) {}

LLDBServerNativeProcess::~LLDBServerNativeProcess() {}
/// Set a breakpoint in the native process.
///
/// When the breakpoints gets hit, lldb-server will call
/// LLDBServerPlugin::BreakpointWasHit with this address. This will allow
/// LLDBServerPlugin plugins to synchronously handle a breakpoint hit in the
/// native process.
lldb::user_id_t LLDBServerNativeProcess::SetBreakpoint(lldb::addr_t address) {
  return LLDB_INVALID_BREAK_ID;
}

Status LLDBServerNativeProcess::RegisterSignalCatcher(int signo) {
  return Status::FromErrorString("unimplemented");
}

Status LLDBServerNativeProcess::HaltProcess() {
  return m_native_process->Halt();
}

Status LLDBServerNativeProcess::ContinueProcess() {
  ResumeActionList resume_actions;
  return m_native_process->Resume(resume_actions);
}
