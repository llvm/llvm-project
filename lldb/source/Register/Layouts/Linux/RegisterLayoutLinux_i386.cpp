//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_i386.h"
#include "lldb/Register/Common/RegisterInfo_x86.h"
#include <cstddef>
using namespace lldb_private;
namespace {
using linux_i386::GPR;
const RegisterStorageField g_fields[] = {
    {"ebx", sizeof(((GPR *)nullptr)->ebx), offsetof(GPR, ebx)},
    {"ecx", sizeof(((GPR *)nullptr)->ecx), offsetof(GPR, ecx)},
    {"edx", sizeof(((GPR *)nullptr)->edx), offsetof(GPR, edx)},
    {"esi", sizeof(((GPR *)nullptr)->esi), offsetof(GPR, esi)},
    {"edi", sizeof(((GPR *)nullptr)->edi), offsetof(GPR, edi)},
    {"ebp", sizeof(((GPR *)nullptr)->ebp), offsetof(GPR, ebp)},
    {"eax", sizeof(((GPR *)nullptr)->eax), offsetof(GPR, eax)},
    {"ds", sizeof(((GPR *)nullptr)->ds), offsetof(GPR, ds)},
    {"es", sizeof(((GPR *)nullptr)->es), offsetof(GPR, es)},
    {"fs", sizeof(((GPR *)nullptr)->fs), offsetof(GPR, fs)},
    {"gs", sizeof(((GPR *)nullptr)->gs), offsetof(GPR, gs)},
    {"eip", sizeof(((GPR *)nullptr)->eip), offsetof(GPR, eip)},
    {"cs", sizeof(((GPR *)nullptr)->cs), offsetof(GPR, cs)},
    {"eflags", sizeof(((GPR *)nullptr)->eflags), offsetof(GPR, eflags)},
    {"esp", sizeof(((GPR *)nullptr)->esp), offsetof(GPR, esp)},
    {"ss", sizeof(((GPR *)nullptr)->ss), offsetof(GPR, ss)},
};
} // namespace

RegisterLayout
lldb_private::CreateLinuxI386RegisterLayout(const ArchSpec &target_arch) {
  return CreateX86RegisterLayout(target_arch, g_fields, sizeof(GPR), false,
                                 false, X86RegisterABI::SystemV);
}
