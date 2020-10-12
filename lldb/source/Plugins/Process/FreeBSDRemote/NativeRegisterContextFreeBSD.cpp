//===-- NativeRegisterContextFreeBSD.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextFreeBSD.h"

#include "Plugins/Process/FreeBSDRemote/NativeProcessFreeBSD.h"

#include "lldb/Host/common/NativeProcessProtocol.h"

using namespace lldb_private;
using namespace lldb_private::process_freebsd;

// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
// clang-format on

NativeRegisterContextFreeBSD::NativeRegisterContextFreeBSD(
    NativeThreadProtocol &native_thread,
    RegisterInfoInterface *reg_info_interface_p)
    : NativeRegisterContextRegisterInfo(native_thread, reg_info_interface_p) {}

Status NativeRegisterContextFreeBSD::DoRegisterSet(int ptrace_req, void *buf) {
  return NativeProcessFreeBSD::PtraceWrapper(ptrace_req, GetProcessPid(), buf,
                                             m_thread.GetID());
}

NativeProcessFreeBSD &NativeRegisterContextFreeBSD::GetProcess() {
  return static_cast<NativeProcessFreeBSD &>(m_thread.GetProcess());
}

::pid_t NativeRegisterContextFreeBSD::GetProcessPid() {
  return GetProcess().GetID();
}
