//===-- NativeRegisterContextFreeBSD.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextFreeBSD.h"
#include "lldb/Utility/ArchSpec.h"

#include "Plugins/Process/FreeBSD/NativeProcessFreeBSD.h"

#include "lldb/Host/common/NativeProcessProtocol.h"

#if defined(__powerpc__)
#include "NativeRegisterContextFreeBSD_ppc32be.h"
#endif

using namespace lldb_private;
using namespace lldb_private::process_freebsd;

// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
// clang-format on

NativeProcessFreeBSD &NativeRegisterContextFreeBSD::GetProcess() {
  return static_cast<NativeProcessFreeBSD &>(m_thread.GetProcess());
}

::pid_t NativeRegisterContextFreeBSD::GetProcessPid() {
  return GetProcess().GetID();
}

#if defined(__powerpc__)
NativeRegisterContextFreeBSD *
NativeRegisterContextFreeBSD::CreateHostNativeRegisterContextFreeBSD(
    const ArchSpec &arch, NativeThreadFreeBSD &thread) {
  switch (arch.GetMachine()) {
  case llvm::Triple::ppc:
    return new NativeRegisterContextFreeBSD_ppc32be(arch, thread);
  default:
    llvm_unreachable("unsupported native FreeBSD PowerPC architecture");
  }
}
#endif
