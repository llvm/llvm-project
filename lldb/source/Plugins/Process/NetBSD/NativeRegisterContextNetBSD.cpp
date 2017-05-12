//===-- NativeRegisterContextNetBSD.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextNetBSD.h"

#include "lldb/Host/common/NativeProcessProtocol.h"

using namespace lldb_private;
using namespace lldb_private::process_netbsd;

// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
// clang-format on

NativeRegisterContextNetBSD::NativeRegisterContextNetBSD(
    NativeThreadProtocol &native_thread, uint32_t concrete_frame_idx,
    RegisterInfoInterface *reg_info_interface_p)
    : NativeRegisterContextRegisterInfo(native_thread, concrete_frame_idx,
                                        reg_info_interface_p) {}

Status NativeRegisterContextNetBSD::ReadGPR() {
  void *buf = GetGPRBuffer();
  if (!buf)
    return Status("GPR buffer is NULL");

  return DoReadGPR(buf);
}

Status NativeRegisterContextNetBSD::WriteGPR() {
  void *buf = GetGPRBuffer();
  if (!buf)
    return Status("GPR buffer is NULL");

  return DoWriteGPR(buf);
}

Status NativeRegisterContextNetBSD::ReadFPR() {
  void *buf = GetFPRBuffer();
  if (!buf)
    return Status("FPR buffer is NULL");

  return DoReadFPR(buf);
}

Status NativeRegisterContextNetBSD::WriteFPR() {
  void *buf = GetFPRBuffer();
  if (!buf)
    return Status("FPR buffer is NULL");

  return DoWriteFPR(buf);
}

Status NativeRegisterContextNetBSD::ReadDBR() {
  void *buf = GetDBRBuffer();
  if (!buf)
    return Status("DBR buffer is NULL");

  return DoReadDBR(buf);
}

Status NativeRegisterContextNetBSD::WriteDBR() {
  void *buf = GetDBRBuffer();
  if (!buf)
    return Status("DBR buffer is NULL");

  return DoWriteDBR(buf);
}

Status NativeRegisterContextNetBSD::DoReadGPR(void *buf) {
  return NativeProcessNetBSD::PtraceWrapper(PT_GETREGS, GetProcessPid(), buf,
                                            m_thread.GetID());
}

Status NativeRegisterContextNetBSD::DoWriteGPR(void *buf) {
  return NativeProcessNetBSD::PtraceWrapper(PT_SETREGS, GetProcessPid(), buf,
                                            m_thread.GetID());
}

Status NativeRegisterContextNetBSD::DoReadFPR(void *buf) {
  return NativeProcessNetBSD::PtraceWrapper(PT_GETFPREGS, GetProcessPid(), buf,
                                            m_thread.GetID());
}

Status NativeRegisterContextNetBSD::DoWriteFPR(void *buf) {
  return NativeProcessNetBSD::PtraceWrapper(PT_SETFPREGS, GetProcessPid(), buf,
                                            m_thread.GetID());
}

Status NativeRegisterContextNetBSD::DoReadDBR(void *buf) {
  return NativeProcessNetBSD::PtraceWrapper(PT_GETDBREGS, GetProcessPid(), buf,
                                            m_thread.GetID());
}

Status NativeRegisterContextNetBSD::DoWriteDBR(void *buf) {
  return NativeProcessNetBSD::PtraceWrapper(PT_SETDBREGS, GetProcessPid(), buf,
                                            m_thread.GetID());
}

NativeProcessNetBSD &NativeRegisterContextNetBSD::GetProcess() {
  auto process_sp =
      std::static_pointer_cast<NativeProcessNetBSD>(m_thread.GetProcess());
  assert(process_sp);
  return *process_sp;
}

::pid_t NativeRegisterContextNetBSD::GetProcessPid() {
  NativeProcessNetBSD &process = GetProcess();
  lldb::pid_t pid = process.GetID();

  return pid;
}
