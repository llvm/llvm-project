//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc__)

#include "NativeRegisterContextFreeBSD_ppc32be.h"
#include "NativeThreadFreeBSD.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_ppc.h"
#include "lldb/Utility/ArchSpec.h"

using namespace lldb_private;
using namespace lldb_private::process_freebsd;

NativeRegisterContextFreeBSD_ppc32be::NativeRegisterContextFreeBSD_ppc32be(
    const ArchSpec &arch, NativeThreadFreeBSD &thread, bool has_vmx)
    : NativeRegisterContext(
          thread, arch,
          SelectLayout(RegisterLayoutFreeBSD_ppc(
                           arch, false,
                           HostInfo::GetArchitecture().GetAddressByteSize()),
                       has_vmx, false)),
      NativeRegisterContextFreeBSD_ppc(has_vmx, false) {}

llvm::Error NativeRegisterContextFreeBSD_ppc32be::CopyHardwareWatchpointsFrom(
    NativeRegisterContextFreeBSD &source) {
  return llvm::Error::success();
}

#endif // defined(__powerpc__)
