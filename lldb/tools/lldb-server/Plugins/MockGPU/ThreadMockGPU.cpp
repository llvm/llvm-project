//===-- ThreadMockGPU.cpp ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ThreadMockGPU.h"
#include "ProcessMockGPU.h"

using namespace lldb_private;
using namespace lldb_server;

ThreadMockGPU::ThreadMockGPU(ProcessMockGPU &process, lldb::tid_t tid)
    : NativeThreadProtocol(process, tid), m_reg_context(*this) {
  m_stop_info.reason = lldb::eStopReasonDynammicLoader;
}

// NativeThreadProtocol Interface
std::string ThreadMockGPU::GetName() { return "Mock GPU Thread Name"; }

lldb::StateType ThreadMockGPU::GetState() { return lldb::eStateStopped; }

bool ThreadMockGPU::GetStopReason(ThreadStopInfo &stop_info,
                                  std::string &description) {
  stop_info = m_stop_info;
  description = "Mock GPU Thread Stop Reason";
  return true;
}

Status ThreadMockGPU::SetWatchpoint(lldb::addr_t addr, size_t size,
                                    uint32_t watch_flags, bool hardware) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadMockGPU::RemoveWatchpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadMockGPU::SetHardwareBreakpoint(lldb::addr_t addr, size_t size) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadMockGPU::RemoveHardwareBreakpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}

ProcessMockGPU &ThreadMockGPU::GetProcess() {
  return static_cast<ProcessMockGPU &>(m_process);
}

const ProcessMockGPU &ThreadMockGPU::GetProcess() const {
  return static_cast<const ProcessMockGPU &>(m_process);
}
