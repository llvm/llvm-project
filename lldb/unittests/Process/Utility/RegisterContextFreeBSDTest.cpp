//===-- RegisterContextFreeBSDTests.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__FreeBSD__)

// clang-format off
#include <sys/types.h>
#include <machine/reg.h>
// clang-format on

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Plugins/Process/Utility/lldb-x86-register-enums.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_i386.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_x86_64.h"

using namespace lldb;
using namespace lldb_private;

std::pair<size_t, size_t> GetRegParams(RegisterInfoInterface &ctx,
                                       uint32_t reg) {
  const RegisterInfo &info = ctx.GetRegisterInfo()[reg];
  return {info.byte_offset, info.byte_size};
}

#if defined(__x86_64__)

#define EXPECT_GPR_X86_64(regname)                                             \
  EXPECT_THAT(                                                                 \
      GetRegParams(reg_ctx, lldb_##regname##_x86_64),                          \
      ::testing::Pair(offsetof(reg, r_##regname), sizeof(reg::r_##regname)))

TEST(RegisterContextFreeBSDTest, x86_64) {
  ArchSpec arch{"x86_64-unknown-freebsd"};
  RegisterContextFreeBSD_x86_64 reg_ctx{arch};

  EXPECT_GPR_X86_64(r15);
  EXPECT_GPR_X86_64(r14);
  EXPECT_GPR_X86_64(r13);
  EXPECT_GPR_X86_64(r12);
  EXPECT_GPR_X86_64(r11);
  EXPECT_GPR_X86_64(r10);
  EXPECT_GPR_X86_64(r9);
  EXPECT_GPR_X86_64(r8);
  EXPECT_GPR_X86_64(rdi);
  EXPECT_GPR_X86_64(rsi);
  EXPECT_GPR_X86_64(rbp);
  EXPECT_GPR_X86_64(rbx);
  EXPECT_GPR_X86_64(rdx);
  EXPECT_GPR_X86_64(rcx);
  EXPECT_GPR_X86_64(rax);
  EXPECT_GPR_X86_64(fs);
  EXPECT_GPR_X86_64(gs);
  EXPECT_GPR_X86_64(es);
  EXPECT_GPR_X86_64(ds);
  EXPECT_GPR_X86_64(rip);
  EXPECT_GPR_X86_64(cs);
  EXPECT_GPR_X86_64(rflags);
  EXPECT_GPR_X86_64(rsp);
  EXPECT_GPR_X86_64(ss);
}

#endif // defined(__x86_64__)

#if defined(__i386__) || defined(__x86_64__)

#define EXPECT_GPR_I386(regname)                                               \
  EXPECT_THAT(GetRegParams(reg_ctx, lldb_##regname##_i386),                    \
              ::testing::Pair(offsetof(native_i386_regs, r_##regname),         \
                              sizeof(native_i386_regs::r_##regname)))

TEST(RegisterContextFreeBSDTest, i386) {
  ArchSpec arch{"i686-unknown-freebsd"};
  RegisterContextFreeBSD_i386 reg_ctx{arch};

#if defined(__i386__)
  using native_i386_regs = ::reg;
#else
  using native_i386_regs = ::reg32;
#endif

  EXPECT_GPR_I386(fs);
  EXPECT_GPR_I386(es);
  EXPECT_GPR_I386(ds);
  EXPECT_GPR_I386(edi);
  EXPECT_GPR_I386(esi);
  EXPECT_GPR_I386(ebp);
  EXPECT_GPR_I386(ebx);
  EXPECT_GPR_I386(edx);
  EXPECT_GPR_I386(ecx);
  EXPECT_GPR_I386(eax);
  EXPECT_GPR_I386(eip);
  EXPECT_GPR_I386(cs);
  EXPECT_GPR_I386(eflags);
  EXPECT_GPR_I386(esp);
  EXPECT_GPR_I386(ss);
  EXPECT_GPR_I386(gs);
}

#endif // defined(__i386__) || defined(__x86_64__)

#endif // defined(__FreeBSD__)
