//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/NetBSD/RegisterLayoutNetBSD_x86_64.h"
#include "lldb/Register/Common/RegisterInfo_x86.h"
#include <cstddef>
using namespace lldb_private;
namespace {
typedef struct _GPR {
  uint64_t rdi;    /*  0 */
  uint64_t rsi;    /*  1 */
  uint64_t rdx;    /*  2 */
  uint64_t rcx;    /*  3 */
  uint64_t r8;     /*  4 */
  uint64_t r9;     /*  5 */
  uint64_t r10;    /*  6 */
  uint64_t r11;    /*  7 */
  uint64_t r12;    /*  8 */
  uint64_t r13;    /*  9 */
  uint64_t r14;    /* 10 */
  uint64_t r15;    /* 11 */
  uint64_t rbp;    /* 12 */
  uint64_t rbx;    /* 13 */
  uint64_t rax;    /* 14 */
  uint64_t gs;     /* 15 */
  uint64_t fs;     /* 16 */
  uint64_t es;     /* 17 */
  uint64_t ds;     /* 18 */
  uint64_t trapno; /* 19 */
  uint64_t err;    /* 20 */
  uint64_t rip;    /* 21 */
  uint64_t cs;     /* 22 */
  uint64_t rflags; /* 23 */
  uint64_t rsp;    /* 24 */
  uint64_t ss;     /* 25 */
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
    {"gs", sizeof(((GPR *)nullptr)->gs), offsetof(GPR, gs)},
    {"fs", sizeof(((GPR *)nullptr)->fs), offsetof(GPR, fs)},
    {"es", sizeof(((GPR *)nullptr)->es), offsetof(GPR, es)},
    {"ds", sizeof(((GPR *)nullptr)->ds), offsetof(GPR, ds)},
    {"rip", sizeof(((GPR *)nullptr)->rip), offsetof(GPR, rip)},
    {"cs", sizeof(((GPR *)nullptr)->cs), offsetof(GPR, cs)},
    {"rflags", sizeof(((GPR *)nullptr)->rflags), offsetof(GPR, rflags)},
    {"rsp", sizeof(((GPR *)nullptr)->rsp), offsetof(GPR, rsp)},
    {"ss", sizeof(((GPR *)nullptr)->ss), offsetof(GPR, ss)},
};
} // namespace

RegisterLayout
lldb_private::CreateNetBSDX86_64RegisterLayout(const ArchSpec &target_arch) {
  return CreateX86RegisterLayout(target_arch, g_fields, sizeof(GPR), true,
                                 false, X86RegisterABI::SystemV);
}
