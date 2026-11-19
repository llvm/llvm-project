//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__powerpc__)

#ifndef LLDB_PLUGINS_PROCESS_FREEBSD_NATIVEREGISTERCONTEXTFREEBSD_PPC32BE_H
#define LLDB_PLUGINS_PROCESS_FREEBSD_NATIVEREGISTERCONTEXTFREEBSD_PPC32BE_H

#include "NativeRegisterContextFreeBSD_ppc.h"

namespace lldb_private::process_freebsd {

class NativeRegisterContextFreeBSD_ppc32be
    : public NativeRegisterContextFreeBSD_ppc {
public:
  NativeRegisterContextFreeBSD_ppc32be(const ArchSpec &arch,
                                       NativeThreadFreeBSD &thread,
                                       bool has_vmx = false);

  llvm::Error
  CopyHardwareWatchpointsFrom(NativeRegisterContextFreeBSD &source) override;
};

} // namespace lldb_private::process_freebsd

#endif
#endif // defined(__powerpc__)
