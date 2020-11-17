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

#define EXPECT_OFF(regname, offset, size)                                      \
  EXPECT_THAT(GetRegParams(reg_ctx, lldb_##regname),                           \
              ::testing::Pair(offset + base_offset, size))

#if defined(__x86_64__)

#define EXPECT_GPR_X86_64(regname)                                             \
  EXPECT_THAT(                                                                 \
      GetRegParams(reg_ctx, lldb_##regname##_x86_64),                          \
      ::testing::Pair(offsetof(reg, r_##regname), sizeof(reg::r_##regname)))
#define EXPECT_DBR_X86_64(num)                                                 \
  EXPECT_OFF(dr##num##_x86_64, offsetof(dbreg, dr[num]), sizeof(dbreg::dr[num]))

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

  // fctrl is the first FPR field, it is used to determine offset of the whole
  // FPR struct
  size_t base_offset = reg_ctx.GetRegisterInfo()[lldb_fctrl_x86_64].byte_offset;

  // assert against FXSAVE struct
  EXPECT_OFF(fctrl_x86_64, 0x00, 2);
  EXPECT_OFF(fstat_x86_64, 0x02, 2);
  // TODO: This is a known bug, abridged ftag should is 8 bits in length.
  EXPECT_OFF(ftag_x86_64, 0x04, 2);
  EXPECT_OFF(fop_x86_64, 0x06, 2);
  // NB: Technically fiseg/foseg are 16-bit long and the higher 16 bits
  // are reserved.  However, we use them to access/recombine 64-bit FIP/FDP.
  EXPECT_OFF(fioff_x86_64, 0x08, 4);
  EXPECT_OFF(fiseg_x86_64, 0x0C, 4);
  EXPECT_OFF(fooff_x86_64, 0x10, 4);
  EXPECT_OFF(foseg_x86_64, 0x14, 4);
  EXPECT_OFF(mxcsr_x86_64, 0x18, 4);
  EXPECT_OFF(mxcsrmask_x86_64, 0x1C, 4);
  EXPECT_OFF(st0_x86_64, 0x20, 10);
  EXPECT_OFF(st1_x86_64, 0x30, 10);
  EXPECT_OFF(st2_x86_64, 0x40, 10);
  EXPECT_OFF(st3_x86_64, 0x50, 10);
  EXPECT_OFF(st4_x86_64, 0x60, 10);
  EXPECT_OFF(st5_x86_64, 0x70, 10);
  EXPECT_OFF(st6_x86_64, 0x80, 10);
  EXPECT_OFF(st7_x86_64, 0x90, 10);
  EXPECT_OFF(mm0_x86_64, 0x20, 8);
  EXPECT_OFF(mm1_x86_64, 0x30, 8);
  EXPECT_OFF(mm2_x86_64, 0x40, 8);
  EXPECT_OFF(mm3_x86_64, 0x50, 8);
  EXPECT_OFF(mm4_x86_64, 0x60, 8);
  EXPECT_OFF(mm5_x86_64, 0x70, 8);
  EXPECT_OFF(mm6_x86_64, 0x80, 8);
  EXPECT_OFF(mm7_x86_64, 0x90, 8);
  EXPECT_OFF(xmm0_x86_64, 0xA0, 16);
  EXPECT_OFF(xmm1_x86_64, 0xB0, 16);
  EXPECT_OFF(xmm2_x86_64, 0xC0, 16);
  EXPECT_OFF(xmm3_x86_64, 0xD0, 16);
  EXPECT_OFF(xmm4_x86_64, 0xE0, 16);
  EXPECT_OFF(xmm5_x86_64, 0xF0, 16);
  EXPECT_OFF(xmm6_x86_64, 0x100, 16);
  EXPECT_OFF(xmm7_x86_64, 0x110, 16);
  EXPECT_OFF(xmm8_x86_64, 0x120, 16);
  EXPECT_OFF(xmm9_x86_64, 0x130, 16);
  EXPECT_OFF(xmm10_x86_64, 0x140, 16);
  EXPECT_OFF(xmm11_x86_64, 0x150, 16);
  EXPECT_OFF(xmm12_x86_64, 0x160, 16);
  EXPECT_OFF(xmm13_x86_64, 0x170, 16);
  EXPECT_OFF(xmm14_x86_64, 0x180, 16);
  EXPECT_OFF(xmm15_x86_64, 0x190, 16);

  base_offset = reg_ctx.GetRegisterInfo()[lldb_dr0_x86_64].byte_offset;
  EXPECT_DBR_X86_64(0);
  EXPECT_DBR_X86_64(1);
  EXPECT_DBR_X86_64(2);
  EXPECT_DBR_X86_64(3);
  EXPECT_DBR_X86_64(4);
  EXPECT_DBR_X86_64(5);
  EXPECT_DBR_X86_64(6);
  EXPECT_DBR_X86_64(7);
}

#endif // defined(__x86_64__)

#if defined(__i386__) || defined(__x86_64__)

#define EXPECT_GPR_I386(regname)                                               \
  EXPECT_THAT(GetRegParams(reg_ctx, lldb_##regname##_i386),                    \
              ::testing::Pair(offsetof(native_i386_regs, r_##regname),         \
                              sizeof(native_i386_regs::r_##regname)))
#define EXPECT_DBR_I386(num)                                                   \
  EXPECT_OFF(dr##num##_i386, offsetof(native_i386_dbregs, dr[num]),            \
             sizeof(native_i386_dbregs::dr[num]))

TEST(RegisterContextFreeBSDTest, i386) {
  ArchSpec arch{"i686-unknown-freebsd"};
  RegisterContextFreeBSD_i386 reg_ctx{arch};

#if defined(__i386__)
  using native_i386_regs = ::reg;
  using native_i386_dbregs = ::dbreg;
#else
  using native_i386_regs = ::reg32;
  using native_i386_dbregs = ::dbreg32;
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

  // fctrl is the first FPR field, it is used to determine offset of the whole
  // FPR struct
  size_t base_offset = reg_ctx.GetRegisterInfo()[lldb_fctrl_i386].byte_offset;

  // assert against FXSAVE struct
  EXPECT_OFF(fctrl_i386, 0x00, 2);
  EXPECT_OFF(fstat_i386, 0x02, 2);
  // TODO: This is a known bug, abridged ftag should is 8 bits in length.
  EXPECT_OFF(ftag_i386, 0x04, 2);
  EXPECT_OFF(fop_i386, 0x06, 2);
  // NB: Technically fiseg/foseg are 16-bit long and the higher 16 bits
  // are reserved.  However, we use them to access/recombine 64-bit FIP/FDP.
  EXPECT_OFF(fioff_i386, 0x08, 4);
  EXPECT_OFF(fiseg_i386, 0x0C, 4);
  EXPECT_OFF(fooff_i386, 0x10, 4);
  EXPECT_OFF(foseg_i386, 0x14, 4);
  EXPECT_OFF(mxcsr_i386, 0x18, 4);
  EXPECT_OFF(mxcsrmask_i386, 0x1C, 4);
  EXPECT_OFF(st0_i386, 0x20, 10);
  EXPECT_OFF(st1_i386, 0x30, 10);
  EXPECT_OFF(st2_i386, 0x40, 10);
  EXPECT_OFF(st3_i386, 0x50, 10);
  EXPECT_OFF(st4_i386, 0x60, 10);
  EXPECT_OFF(st5_i386, 0x70, 10);
  EXPECT_OFF(st6_i386, 0x80, 10);
  EXPECT_OFF(st7_i386, 0x90, 10);
  EXPECT_OFF(mm0_i386, 0x20, 8);
  EXPECT_OFF(mm1_i386, 0x30, 8);
  EXPECT_OFF(mm2_i386, 0x40, 8);
  EXPECT_OFF(mm3_i386, 0x50, 8);
  EXPECT_OFF(mm4_i386, 0x60, 8);
  EXPECT_OFF(mm5_i386, 0x70, 8);
  EXPECT_OFF(mm6_i386, 0x80, 8);
  EXPECT_OFF(mm7_i386, 0x90, 8);
  EXPECT_OFF(xmm0_i386, 0xA0, 16);
  EXPECT_OFF(xmm1_i386, 0xB0, 16);
  EXPECT_OFF(xmm2_i386, 0xC0, 16);
  EXPECT_OFF(xmm3_i386, 0xD0, 16);
  EXPECT_OFF(xmm4_i386, 0xE0, 16);
  EXPECT_OFF(xmm5_i386, 0xF0, 16);
  EXPECT_OFF(xmm6_i386, 0x100, 16);
  EXPECT_OFF(xmm7_i386, 0x110, 16);

  base_offset = reg_ctx.GetRegisterInfo()[lldb_dr0_i386].byte_offset;
  EXPECT_DBR_I386(0);
  EXPECT_DBR_I386(1);
  EXPECT_DBR_I386(2);
  EXPECT_DBR_I386(3);
  EXPECT_DBR_I386(4);
  EXPECT_DBR_I386(5);
  EXPECT_DBR_I386(6);
  EXPECT_DBR_I386(7);
}

#endif // defined(__i386__) || defined(__x86_64__)

#endif // defined(__FreeBSD__)
