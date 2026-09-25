//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadMockAccelerator.h"
#include "ProcessMockAccelerator.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;

ThreadMockAccelerator::ThreadMockAccelerator(ProcessMockAccelerator &process,
                                             lldb::tid_t tid)
    : NativeThreadProtocol(process, tid), m_reg_context(*this) {
  m_stop_info.reason = lldb::eStopReasonTrace;
}

std::string ThreadMockAccelerator::GetName() {
  return "Mock Accelerator Thread";
}

lldb::StateType ThreadMockAccelerator::GetState() {
  return lldb::eStateStopped;
}

bool ThreadMockAccelerator::GetStopReason(ThreadStopInfo &stop_info,
                                          std::string &description) {
  stop_info = m_stop_info;
  description = "mock accelerator thread stopped";
  return true;
}

Status ThreadMockAccelerator::SetWatchpoint(lldb::addr_t addr, size_t size,
                                            uint32_t watch_flags,
                                            bool hardware) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadMockAccelerator::RemoveWatchpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadMockAccelerator::SetHardwareBreakpoint(lldb::addr_t addr,
                                                    size_t size) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadMockAccelerator::RemoveHardwareBreakpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}
