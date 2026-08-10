//===-- RegisterContextNetBSDTest_x86_64.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__x86_64__)

// clang-format off
#include <sys/types.h>
#include <amd64/reg.h>
// clang-format on

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Plugins/Process/Utility/lldb-x86-register-enums.h"
#include "Plugins/Process/Utility/RegisterContextNetBSD_i386.h"
#include "Plugins/Process/Utility/RegisterContextNetBSD_x86_64.h"

using namespace lldb;
using namespace lldb_private;

static std::pair<size_t, size_t> GetRegParams(RegisterInfoInterface &ctx,
                                              uint32_t reg) {
  const RegisterInfo &info = ctx.GetRegisterInfo()[reg];
  return {info.byte_offset, info.byte_size};
}

#define EXPECT_OFF(regname, offset, size)                                      \
  EXPECT_THAT(GetRegParams(reg_ctx, lldb_##regname),                           \
              ::testing::Pair(offset + base_offset, size))

#define EXPECT_GPR_X86_64(regname, regconst)                                   \
  EXPECT_THAT(                                                                 \
      GetRegParams(reg_ctx, lldb_##regname##_x86_64),                          \
      ::testing::Pair(offsetof(reg, regs[regconst]),                           \
                      sizeof(reg::regs[regconst])))
#define EXPECT_DBR_X86_64(num)                                                 \
  EXPECT_OFF(dr##num##_x86_64, offsetof(dbreg, dr[num]), sizeof(dbreg::dr[num]))

TEST(RegisterContextNetBSDTest, x86_64) {
  ArchSpec arch{"x86_64-unknown-netbsd"};
  RegisterContextNetBSD_x86_64 reg_ctx{arch};

  EXPECT_GPR_X86_64(rdi, _REG_RDI);
  EXPECT_GPR_X86_64(rsi, _REG_RSI);
  EXPECT_GPR_X86_64(rdx, _REG_RDX);
  EXPECT_GPR_X86_64(rcx, _REG_RCX);
  EXPECT_GPR_X86_64(r8, _REG_R8);
  EXPECT_GPR_X86_64(r9, _REG_R9);
  EXPECT_GPR_X86_64(r10, _REG_R10);
  EXPECT_GPR_X86_64(r11, _REG_R11);
  EXPECT_GPR_X86_64(r12, _REG_R12);
  EXPECT_GPR_X86_64(r13, _REG_R13);
  EXPECT_GPR_X86_64(r14, _REG_R14);
  EXPECT_GPR_X86_64(r15, _REG_R15);
  EXPECT_GPR_X86_64(rbp, _REG_RBP);
  EXPECT_GPR_X86_64(rbx, _REG_RBX);
  EXPECT_GPR_X86_64(rax, _REG_RAX);
  EXPECT_GPR_X86_64(gs, _REG_GS);
  EXPECT_GPR_X86_64(fs, _REG_FS);
  EXPECT_GPR_X86_64(es, _REG_ES);
  EXPECT_GPR_X86_64(ds, _REG_DS);
  EXPECT_GPR_X86_64(rip, _REG_RIP);
  EXPECT_GPR_X86_64(cs, _REG_CS);
  EXPECT_GPR_X86_64(rflags, _REG_RFLAGS);
  EXPECT_GPR_X86_64(rsp, _REG_RSP);
  EXPECT_GPR_X86_64(ss, _REG_SS);

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
  // are reserved.  However, LLDB defines them to be 32-bit long for backwards
  // compatibility, as they were used to reconstruct FIP/FDP before explicit
  // register entries for them were added.  Also, this is still how GDB does it.
  EXPECT_OFF(fioff_x86_64, 0x08, 4);
  EXPECT_OFF(fiseg_x86_64, 0x0C, 4);
  EXPECT_OFF(fip_x86_64, 0x08, 8);
  EXPECT_OFF(fooff_x86_64, 0x10, 4);
  EXPECT_OFF(foseg_x86_64, 0x14, 4);
  EXPECT_OFF(fdp_x86_64, 0x10, 8);
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
