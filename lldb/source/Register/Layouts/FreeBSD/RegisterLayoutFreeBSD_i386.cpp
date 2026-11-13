//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_i386.h"
#include "lldb/Register/Common/RegisterInfo_x86.h"
#include <cstddef>
using namespace lldb_private;
namespace {
struct GPR {
  uint32_t fs;
  uint32_t es;
  uint32_t ds;
  uint32_t edi;
  uint32_t esi;
  uint32_t ebp;
  uint32_t isp;
  uint32_t ebx;
  uint32_t edx;
  uint32_t ecx;
  uint32_t eax;
  uint32_t trapno;
  uint32_t err;
  uint32_t eip;
  uint32_t cs;
  uint32_t eflags;
  uint32_t esp;
  uint32_t ss;
  uint32_t gs;
};
const RegisterStorageField g_fields[] = {
    {"fs", sizeof(((GPR *)nullptr)->fs), offsetof(GPR, fs)},
    {"es", sizeof(((GPR *)nullptr)->es), offsetof(GPR, es)},
    {"ds", sizeof(((GPR *)nullptr)->ds), offsetof(GPR, ds)},
    {"edi", sizeof(((GPR *)nullptr)->edi), offsetof(GPR, edi)},
    {"esi", sizeof(((GPR *)nullptr)->esi), offsetof(GPR, esi)},
    {"ebp", sizeof(((GPR *)nullptr)->ebp), offsetof(GPR, ebp)},
    {"ebx", sizeof(((GPR *)nullptr)->ebx), offsetof(GPR, ebx)},
    {"edx", sizeof(((GPR *)nullptr)->edx), offsetof(GPR, edx)},
    {"ecx", sizeof(((GPR *)nullptr)->ecx), offsetof(GPR, ecx)},
    {"eax", sizeof(((GPR *)nullptr)->eax), offsetof(GPR, eax)},
    {"eip", sizeof(((GPR *)nullptr)->eip), offsetof(GPR, eip)},
    {"cs", sizeof(((GPR *)nullptr)->cs), offsetof(GPR, cs)},
    {"eflags", sizeof(((GPR *)nullptr)->eflags), offsetof(GPR, eflags)},
    {"esp", sizeof(((GPR *)nullptr)->esp), offsetof(GPR, esp)},
    {"ss", sizeof(((GPR *)nullptr)->ss), offsetof(GPR, ss)},
    {"gs", sizeof(((GPR *)nullptr)->gs), offsetof(GPR, gs)},
};
} // namespace

RegisterLayout
lldb_private::CreateFreeBSDI386RegisterLayout(const ArchSpec &target_arch) {
  return CreateX86RegisterLayout(target_arch, g_fields, sizeof(GPR), false,
                                 false, X86RegisterABI::SystemV);
}
