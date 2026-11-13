//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_x86_64.h"
#include "lldb/Register/Common/RegisterInfo_x86.h"
#include <cstddef>
using namespace lldb_private;
namespace {
using linux_x86_64::GPR;
const RegisterStorageField g_fields[] = {
    {"r15", sizeof(((GPR *)nullptr)->r15), offsetof(GPR, r15)},
    {"r14", sizeof(((GPR *)nullptr)->r14), offsetof(GPR, r14)},
    {"r13", sizeof(((GPR *)nullptr)->r13), offsetof(GPR, r13)},
    {"r12", sizeof(((GPR *)nullptr)->r12), offsetof(GPR, r12)},
    {"rbp", sizeof(((GPR *)nullptr)->rbp), offsetof(GPR, rbp)},
    {"rbx", sizeof(((GPR *)nullptr)->rbx), offsetof(GPR, rbx)},
    {"r11", sizeof(((GPR *)nullptr)->r11), offsetof(GPR, r11)},
    {"r10", sizeof(((GPR *)nullptr)->r10), offsetof(GPR, r10)},
    {"r9", sizeof(((GPR *)nullptr)->r9), offsetof(GPR, r9)},
    {"r8", sizeof(((GPR *)nullptr)->r8), offsetof(GPR, r8)},
    {"rax", sizeof(((GPR *)nullptr)->rax), offsetof(GPR, rax)},
    {"rcx", sizeof(((GPR *)nullptr)->rcx), offsetof(GPR, rcx)},
    {"rdx", sizeof(((GPR *)nullptr)->rdx), offsetof(GPR, rdx)},
    {"rsi", sizeof(((GPR *)nullptr)->rsi), offsetof(GPR, rsi)},
    {"rdi", sizeof(((GPR *)nullptr)->rdi), offsetof(GPR, rdi)},
    {"rip", sizeof(((GPR *)nullptr)->rip), offsetof(GPR, rip)},
    {"cs", sizeof(((GPR *)nullptr)->cs), offsetof(GPR, cs)},
    {"rflags", sizeof(((GPR *)nullptr)->rflags), offsetof(GPR, rflags)},
    {"rsp", sizeof(((GPR *)nullptr)->rsp), offsetof(GPR, rsp)},
    {"ss", sizeof(((GPR *)nullptr)->ss), offsetof(GPR, ss)},
    {"fs_base", sizeof(((GPR *)nullptr)->fs_base), offsetof(GPR, fs_base)},
    {"gs_base", sizeof(((GPR *)nullptr)->gs_base), offsetof(GPR, gs_base)},
    {"ds", sizeof(((GPR *)nullptr)->ds), offsetof(GPR, ds)},
    {"es", sizeof(((GPR *)nullptr)->es), offsetof(GPR, es)},
    {"fs", sizeof(((GPR *)nullptr)->fs), offsetof(GPR, fs)},
    {"gs", sizeof(((GPR *)nullptr)->gs), offsetof(GPR, gs)},
};
} // namespace

RegisterLayout
lldb_private::CreateLinuxX86_64RegisterLayout(const ArchSpec &target_arch) {
  return CreateX86RegisterLayout(target_arch, g_fields, sizeof(GPR), true, true,
                                 X86RegisterABI::SystemV);
}
