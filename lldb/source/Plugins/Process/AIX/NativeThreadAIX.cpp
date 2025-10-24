//===-- NativeThreadAIX.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeThreadAIX.h"
#include "NativeProcessAIX.h"
#include "lldb/Utility/State.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_aix;

NativeThreadAIX::NativeThreadAIX(NativeProcessAIX &process, lldb::tid_t tid)
    : NativeThreadProtocol(process, tid), m_state(StateType::eStateInvalid) {}

std::string NativeThreadAIX::GetName() { return ""; }

lldb::StateType NativeThreadAIX::GetState() { return m_state; }

bool NativeThreadAIX::GetStopReason(ThreadStopInfo &stop_info,
                                    std::string &description) {
  return false;
}

Status NativeThreadAIX::SetWatchpoint(lldb::addr_t addr, size_t size,
                                      uint32_t watch_flags, bool hardware) {
  return Status("Unable to Set hardware watchpoint.");
}

Status NativeThreadAIX::RemoveWatchpoint(lldb::addr_t addr) {
  return Status("Clearing hardware watchpoint failed.");
}

Status NativeThreadAIX::SetHardwareBreakpoint(lldb::addr_t addr, size_t size) {
  return Status("Unable to set hardware breakpoint.");
}

Status NativeThreadAIX::RemoveHardwareBreakpoint(lldb::addr_t addr) {
  return Status("Clearing hardware breakpoint failed.");
}

NativeProcessAIX &NativeThreadAIX::GetProcess() {
  return static_cast<NativeProcessAIX &>(m_process);
}

const NativeProcessAIX &NativeThreadAIX::GetProcess() const {
  return static_cast<const NativeProcessAIX &>(m_process);
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
NativeThreadAIX::GetSiginfo() const {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Not implemented");
}
