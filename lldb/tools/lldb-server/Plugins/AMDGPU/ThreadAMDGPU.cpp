//===-- ThreadAMDGPU.cpp ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ThreadAMDGPU.h"
#include "ProcessAMDGPU.h"

using namespace lldb_private;
using namespace lldb_server;

ThreadAMDGPU::ThreadAMDGPU(ProcessAMDGPU &process, lldb::tid_t tid,
                           std::optional<amd_dbgapi_wave_id_t> wave_id)
    : NativeThreadProtocol(process, tid), m_reg_context(*this),
      m_wave_id(wave_id) {
  m_stop_info.reason = lldb::eStopReasonSignal;
  m_stop_info.signo = SIGTRAP;
}

// NativeThreadProtocol Interface
std::string ThreadAMDGPU::GetName() {
  if (!m_wave_id)
    return "AMD Native Shadow Thread";
  else
    return std::string("AMD GPU Thread ") +
           std::to_string(m_wave_id.value().handle);
}

lldb::StateType ThreadAMDGPU::GetState() { return lldb::eStateStopped; }

bool ThreadAMDGPU::GetStopReason(ThreadStopInfo &stop_info,
                                 std::string &description) {
  stop_info = m_stop_info;
  description = m_description;
  return true;
}

Status ThreadAMDGPU::SetWatchpoint(lldb::addr_t addr, size_t size,
                                   uint32_t watch_flags, bool hardware) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadAMDGPU::RemoveWatchpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadAMDGPU::SetHardwareBreakpoint(lldb::addr_t addr, size_t size) {
  return Status::FromErrorString("unimplemented");
}

Status ThreadAMDGPU::RemoveHardwareBreakpoint(lldb::addr_t addr) {
  return Status::FromErrorString("unimplemented");
}

ProcessAMDGPU &ThreadAMDGPU::GetProcess() {
  return static_cast<ProcessAMDGPU &>(m_process);
}

const ProcessAMDGPU &ThreadAMDGPU::GetProcess() const {
  return static_cast<const ProcessAMDGPU &>(m_process);
}
