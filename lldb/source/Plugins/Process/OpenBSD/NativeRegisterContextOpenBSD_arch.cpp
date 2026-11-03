//===-- NativeRegisterContextOpenBSD_arch.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NativeRegisterContextOpenBSD_* contains the implementations for each
// supported architecture, and includes the static initalizer
// CreateHostNativeRegisterContextOpenBSD() implementation which returns a arch
// specific register context. This implementation contains a stub
// which just reports an error and exits on architectures which do
// not have a backend.

#if !defined(__aarch64__) && !defined(__i386__) && !defined(__x86_64__)

#include "Plugins/Process/OpenBSD/NativeRegisterContextOpenBSD.h"

using namespace lldb_private;
using namespace lldb_private::process_openbsd;

std::unique_ptr<NativeRegisterContextOpenBSD>
NativeRegisterContextOpenBSD::CreateHostNativeRegisterContextOpenBSD(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread) {
  return std::unique_ptr<NativeRegisterContextOpenBSD>{};
}

#endif
