//===-- NativeRegisterContextOpenBSD.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextOpenBSD.h"

#include "lldb/Host/common/NativeProcessProtocol.h"

using namespace lldb_private;
using namespace lldb_private::process_openbsd;

// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
// clang-format on

NativeRegisterContextOpenBSD::NativeRegisterContextOpenBSD(
    NativeThreadProtocol &native_thread,
    RegisterInfoInterface *reg_info_interface_p)
    : NativeRegisterContextRegisterInfo(native_thread, reg_info_interface_p) {}

Status NativeRegisterContextOpenBSD::ReadGPR() {
  void *buf = GetGPRBuffer();
  if (!buf)
    return Status::FromErrorString("GPR buffer is NULL");

  return DoReadGPR(buf);
}

Status NativeRegisterContextOpenBSD::WriteGPR() {
  void *buf = GetGPRBuffer();
  if (!buf)
    return Status::FromErrorString("GPR buffer is NULL");

  return DoWriteGPR(buf);
}

Status NativeRegisterContextOpenBSD::ReadFPR() {
  void *buf = GetFPRBuffer();
  if (!buf)
    return Status::FromErrorString("FPR buffer is NULL");

  return DoReadFPR(buf);
}

Status NativeRegisterContextOpenBSD::WriteFPR() {
  void *buf = GetFPRBuffer();
  if (!buf)
    return Status::FromErrorString("FPR buffer is NULL");

  return DoWriteFPR(buf);
}

Status NativeRegisterContextOpenBSD::DoReadGPR(void *buf) {
#ifdef PT_GETREGS
  return NativeProcessOpenBSD::PtraceWrapper(PT_GETREGS, m_thread.GetID(), buf);
#else
  return Status::FromErrorString("PT_GETREGS not supported on this platform");
#endif
}

Status NativeRegisterContextOpenBSD::DoWriteGPR(void *buf) {
#ifdef PT_SETREGS
  return NativeProcessOpenBSD::PtraceWrapper(PT_SETREGS, m_thread.GetID(), buf);
#else
  return Status::FromErrorString("PT_SETREGS not supported on this platform");
#endif
}

Status NativeRegisterContextOpenBSD::DoReadFPR(void *buf) {
#ifdef PT_GETFPREGS
  return NativeProcessOpenBSD::PtraceWrapper(PT_GETFPREGS, m_thread.GetID(),
                                             buf);
#else
  return Status::FromErrorString("PT_GETFPREGS not supported on this platform");
#endif
}

Status NativeRegisterContextOpenBSD::DoWriteFPR(void *buf) {
#ifdef PT_SETFPREGS
  return NativeProcessOpenBSD::PtraceWrapper(PT_SETFPREGS, m_thread.GetID(),
                                             buf);
#else
  return Status::FromErrorString("PT_SETFPREGS not supported on this platform");
#endif
}

NativeProcessOpenBSD &NativeRegisterContextOpenBSD::GetProcess() {
  return static_cast<NativeProcessOpenBSD &>(m_thread.GetProcess());
}

::pid_t NativeRegisterContextOpenBSD::GetProcessPid() {
  return GetProcess().GetID();
}
