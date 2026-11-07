//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRegisterContextOpenBSD.h"

using namespace lldb_private;
using namespace lldb_private::process_openbsd;

NativeRegisterContextOpenBSD::NativeRegisterContextOpenBSD(
    NativeThreadProtocol &native_thread,
    RegisterInfoInterface *reg_info_interface_p)
    : NativeRegisterContextRegisterInfo(native_thread, reg_info_interface_p) {}
