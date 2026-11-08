//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterLayoutOpenBSD_x86_64.h"
#include "lldb/Register/Common/RegisterInfo_x86.h"
#include <cstddef>
using namespace lldb_private;
namespace {
typedef struct _GPR {
  uint64_t rdi;
  uint64_t rsi;
  uint64_t rdx;
  uint64_t rcx;
  uint64_t r8;
  uint64_t r9;
  uint64_t r10;
  uint64_t r11;
  uint64_t r12;
  uint64_t r13;
  uint64_t r14;
  uint64_t r15;
  uint64_t rbp;
  uint64_t rbx;
  uint64_t rax;
  uint64_t rsp;
  uint64_t rip;
  uint64_t rflags;
  uint64_t cs;
  uint64_t ss;
  uint64_t ds;
  uint64_t es;
  uint64_t fs;
  uint64_t gs;
} GPR;
const RegisterStorageField g_fields[] = {
    {"rdi", sizeof(((GPR *)nullptr)->rdi), offsetof(GPR, rdi)},
    {"rsi", sizeof(((GPR *)nullptr)->rsi), offsetof(GPR, rsi)},
    {"rdx", sizeof(((GPR *)nullptr)->rdx), offsetof(GPR, rdx)},
    {"rcx", sizeof(((GPR *)nullptr)->rcx), offsetof(GPR, rcx)},
    {"r8", sizeof(((GPR *)nullptr)->r8), offsetof(GPR, r8)},
    {"r9", sizeof(((GPR *)nullptr)->r9), offsetof(GPR, r9)},
    {"r10", sizeof(((GPR *)nullptr)->r10), offsetof(GPR, r10)},
    {"r11", sizeof(((GPR *)nullptr)->r11), offsetof(GPR, r11)},
    {"r12", sizeof(((GPR *)nullptr)->r12), offsetof(GPR, r12)},
    {"r13", sizeof(((GPR *)nullptr)->r13), offsetof(GPR, r13)},
    {"r14", sizeof(((GPR *)nullptr)->r14), offsetof(GPR, r14)},
    {"r15", sizeof(((GPR *)nullptr)->r15), offsetof(GPR, r15)},
    {"rbp", sizeof(((GPR *)nullptr)->rbp), offsetof(GPR, rbp)},
    {"rbx", sizeof(((GPR *)nullptr)->rbx), offsetof(GPR, rbx)},
    {"rax", sizeof(((GPR *)nullptr)->rax), offsetof(GPR, rax)},
    {"rsp", sizeof(((GPR *)nullptr)->rsp), offsetof(GPR, rsp)},
    {"rip", sizeof(((GPR *)nullptr)->rip), offsetof(GPR, rip)},
    {"rflags", sizeof(((GPR *)nullptr)->rflags), offsetof(GPR, rflags)},
    {"cs", sizeof(((GPR *)nullptr)->cs), offsetof(GPR, cs)},
    {"ss", sizeof(((GPR *)nullptr)->ss), offsetof(GPR, ss)},
    {"ds", sizeof(((GPR *)nullptr)->ds), offsetof(GPR, ds)},
    {"es", sizeof(((GPR *)nullptr)->es), offsetof(GPR, es)},
    {"fs", sizeof(((GPR *)nullptr)->fs), offsetof(GPR, fs)},
    {"gs", sizeof(((GPR *)nullptr)->gs), offsetof(GPR, gs)},
};
} // namespace

RegisterLayout
lldb_private::CreateOpenBSDX86_64RegisterLayout(const ArchSpec &target_arch) {
  return CreateX86RegisterLayout(target_arch, g_fields, sizeof(GPR), true,
                                 false, X86RegisterABI::SystemV);
}
