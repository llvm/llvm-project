//===-- NativeRegisterContextLinux_arm64dbreg.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// When debugging 32-bit processes, Arm64 lldb-server should use 64-bit ptrace
// interfaces. 32-bit ptrace interfaces should only be used by 32-bit server.
// These functions are split out to be reused in both 32-bit and 64-bit register
// context for 64-bit server.

#include "Plugins/Process/Linux/NativeProcessLinux.h"
#include "Plugins/Process/Utility/NativeRegisterContextDBReg.h"
#include "lldb/Utility/Status.h"

namespace lldb_private {
namespace process_linux {
namespace arm64 {

Status ReadHardwareDebugInfo(::pid_t tid, uint32_t &max_hwp_supported,
                             uint32_t &max_hbp_supported);

Status WriteHardwareDebugRegs(
    int hwbType, ::pid_t tid, uint32_t max_supported,
    const std::array<NativeRegisterContextDBReg::DREG, 16> &regs);

} // namespace arm64
} // namespace process_linux
} // namespace lldb_private
