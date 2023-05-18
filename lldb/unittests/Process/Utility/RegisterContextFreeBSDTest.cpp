//===-- RegisterContextFreeBSDTests.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
#include <sys/types.h>
#include <machine/reg.h>
#if defined(__arm__)
#include <machine/vfp.h>
#endif
// clang-format on

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Plugins/Process/Utility/RegisterContextFreeBSD_i386.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_mips64.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_powerpc.h"
#include "Plugins/Process/Utility/RegisterContextFreeBSD_x86_64.h"
#include "Plugins/Process/Utility/RegisterContextPOSIX_powerpc.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm.h"
#include "Plugins/Process/Utility/RegisterInfoPOSIX_arm64.h"
#include "Plugins/Process/Utility/lldb-arm-register-enums.h"
#include "Plugins/Process/Utility/lldb-arm64-register-enums.h"
#include "Plugins/Process/Utility/lldb-mips-freebsd-register-enums.h"
#include "Plugins/Process/Utility/lldb-x86-register-enums.h"

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

#if defined(__arm__)

#define EXPECT_GPR_ARM(lldb_reg, fbsd_reg)                                     \
  EXPECT_THAT(GetRegParams(reg_ctx, gpr_##lldb_reg##_arm),                     \
              ::testing::Pair(offsetof(reg, fbsd_reg), sizeof(reg::fbsd_reg)))
#define EXPECT_FPU_ARM(lldb_reg, fbsd_reg)                                     \
  EXPECT_THAT(GetRegParams(reg_ctx, fpu_##lldb_reg##_arm),                     \
              ::testing::Pair(offsetof(vfp_state, fbsd_reg) + base_offset,     \
                              sizeof(vfp_state::fbsd_reg)))

TEST(RegisterContextFreeBSDTest, arm) {
  ArchSpec arch{"arm-unknown-freebsd"};
  RegisterInfoPOSIX_arm reg_ctx{arch};

  EXPECT_GPR_ARM(r0, r[0]);
  EXPECT_GPR_ARM(r1, r[1]);
  EXPECT_GPR_ARM(r2, r[2]);
  EXPECT_GPR_ARM(r3, r[3]);
  EXPECT_GPR_ARM(r4, r[4]);
  EXPECT_GPR_ARM(r5, r[5]);
  EXPECT_GPR_ARM(r6, r[6]);
  EXPECT_GPR_ARM(r7, r[7]);
  EXPECT_GPR_ARM(r8, r[8]);
  EXPECT_GPR_ARM(r9, r[9]);
  EXPECT_GPR_ARM(r10, r[10]);
  EXPECT_GPR_ARM(r11, r[11]);
  EXPECT_GPR_ARM(r12, r[12]);
  EXPECT_GPR_ARM(sp, r_sp);
  EXPECT_GPR_ARM(lr, r_lr);
  EXPECT_GPR_ARM(pc, r_pc);
  EXPECT_GPR_ARM(cpsr, r_cpsr);

  size_t base_offset = reg_ctx.GetRegisterInfo()[fpu_d0_arm].byte_offset;

  EXPECT_FPU_ARM(d0, reg[0]);
  EXPECT_FPU_ARM(d1, reg[1]);
  EXPECT_FPU_ARM(d2, reg[2]);
  EXPECT_FPU_ARM(d3, reg[3]);
  EXPECT_FPU_ARM(d4, reg[4]);
  EXPECT_FPU_ARM(d5, reg[5]);
  EXPECT_FPU_ARM(d6, reg[6]);
  EXPECT_FPU_ARM(d7, reg[7]);
  EXPECT_FPU_ARM(d8, reg[8]);
  EXPECT_FPU_ARM(d9, reg[9]);
  EXPECT_FPU_ARM(d10, reg[10]);
  EXPECT_FPU_ARM(d11, reg[11]);
  EXPECT_FPU_ARM(d12, reg[12]);
  EXPECT_FPU_ARM(d13, reg[13]);
  EXPECT_FPU_ARM(d14, reg[14]);
  EXPECT_FPU_ARM(d15, reg[15]);
  EXPECT_FPU_ARM(d16, reg[16]);
  EXPECT_FPU_ARM(d17, reg[17]);
  EXPECT_FPU_ARM(d18, reg[18]);
  EXPECT_FPU_ARM(d19, reg[19]);
  EXPECT_FPU_ARM(d20, reg[20]);
  EXPECT_FPU_ARM(d21, reg[21]);
  EXPECT_FPU_ARM(d22, reg[22]);
  EXPECT_FPU_ARM(d23, reg[23]);
  EXPECT_FPU_ARM(d24, reg[24]);
  EXPECT_FPU_ARM(d25, reg[25]);
  EXPECT_FPU_ARM(d26, reg[26]);
  EXPECT_FPU_ARM(d27, reg[27]);
  EXPECT_FPU_ARM(d28, reg[28]);
  EXPECT_FPU_ARM(d29, reg[29]);
  EXPECT_FPU_ARM(d30, reg[30]);
  EXPECT_FPU_ARM(d31, reg[31]);
  EXPECT_FPU_ARM(fpscr, fpscr);
}

#endif // defined(__arm__)

#if defined(__aarch64__)

#define EXPECT_GPR_ARM64(lldb_reg, fbsd_reg)                                   \
  EXPECT_THAT(GetRegParams(reg_ctx, gpr_##lldb_reg##_arm64),                   \
              ::testing::Pair(offsetof(reg, fbsd_reg), sizeof(reg::fbsd_reg)))
#define EXPECT_FPU_ARM64(lldb_reg, fbsd_reg)                                   \
  EXPECT_THAT(GetRegParams(reg_ctx, fpu_##lldb_reg##_arm64),                   \
              ::testing::Pair(offsetof(fpreg, fbsd_reg) + base_offset,         \
                              sizeof(fpreg::fbsd_reg)))

TEST(RegisterContextFreeBSDTest, arm64) {
  Flags opt_regsets = RegisterInfoPOSIX_arm64::eRegsetMaskDefault;
  ArchSpec arch{"aarch64-unknown-freebsd"};
  RegisterInfoPOSIX_arm64 reg_ctx{arch, opt_regsets};

  EXPECT_GPR_ARM64(x0, x[0]);
  EXPECT_GPR_ARM64(x1, x[1]);
  EXPECT_GPR_ARM64(x2, x[2]);
  EXPECT_GPR_ARM64(x3, x[3]);
  EXPECT_GPR_ARM64(x4, x[4]);
  EXPECT_GPR_ARM64(x5, x[5]);
  EXPECT_GPR_ARM64(x6, x[6]);
  EXPECT_GPR_ARM64(x7, x[7]);
  EXPECT_GPR_ARM64(x8, x[8]);
  EXPECT_GPR_ARM64(x9, x[9]);
  EXPECT_GPR_ARM64(x10, x[10]);
  EXPECT_GPR_ARM64(x11, x[11]);
  EXPECT_GPR_ARM64(x12, x[12]);
  EXPECT_GPR_ARM64(x13, x[13]);
  EXPECT_GPR_ARM64(x14, x[14]);
  EXPECT_GPR_ARM64(x15, x[15]);
  EXPECT_GPR_ARM64(x16, x[16]);
  EXPECT_GPR_ARM64(x17, x[17]);
  EXPECT_GPR_ARM64(x18, x[18]);
  EXPECT_GPR_ARM64(x19, x[19]);
  EXPECT_GPR_ARM64(x20, x[20]);
  EXPECT_GPR_ARM64(x21, x[21]);
  EXPECT_GPR_ARM64(x22, x[22]);
  EXPECT_GPR_ARM64(x23, x[23]);
  EXPECT_GPR_ARM64(x24, x[24]);
  EXPECT_GPR_ARM64(x25, x[25]);
  EXPECT_GPR_ARM64(x26, x[26]);
  EXPECT_GPR_ARM64(x27, x[27]);
  EXPECT_GPR_ARM64(x28, x[28]);
  EXPECT_GPR_ARM64(fp, x[29]);
  EXPECT_GPR_ARM64(lr, lr);
  EXPECT_GPR_ARM64(sp, sp);
  EXPECT_GPR_ARM64(pc, elr);
  EXPECT_GPR_ARM64(cpsr, spsr);

  size_t base_offset = reg_ctx.GetRegisterInfo()[fpu_v0_arm64].byte_offset;

  EXPECT_FPU_ARM64(v0, fp_q[0]);
  EXPECT_FPU_ARM64(v1, fp_q[1]);
  EXPECT_FPU_ARM64(v2, fp_q[2]);
  EXPECT_FPU_ARM64(v3, fp_q[3]);
  EXPECT_FPU_ARM64(v4, fp_q[4]);
  EXPECT_FPU_ARM64(v5, fp_q[5]);
  EXPECT_FPU_ARM64(v6, fp_q[6]);
  EXPECT_FPU_ARM64(v7, fp_q[7]);
  EXPECT_FPU_ARM64(v8, fp_q[8]);
  EXPECT_FPU_ARM64(v9, fp_q[9]);
  EXPECT_FPU_ARM64(v10, fp_q[10]);
  EXPECT_FPU_ARM64(v11, fp_q[11]);
  EXPECT_FPU_ARM64(v12, fp_q[12]);
  EXPECT_FPU_ARM64(v13, fp_q[13]);
  EXPECT_FPU_ARM64(v14, fp_q[14]);
  EXPECT_FPU_ARM64(v15, fp_q[15]);
  EXPECT_FPU_ARM64(v16, fp_q[16]);
  EXPECT_FPU_ARM64(v17, fp_q[17]);
  EXPECT_FPU_ARM64(v18, fp_q[18]);
  EXPECT_FPU_ARM64(v19, fp_q[19]);
  EXPECT_FPU_ARM64(v20, fp_q[20]);
  EXPECT_FPU_ARM64(v21, fp_q[21]);
  EXPECT_FPU_ARM64(v22, fp_q[22]);
  EXPECT_FPU_ARM64(v23, fp_q[23]);
  EXPECT_FPU_ARM64(v24, fp_q[24]);
  EXPECT_FPU_ARM64(v25, fp_q[25]);
  EXPECT_FPU_ARM64(v26, fp_q[26]);
  EXPECT_FPU_ARM64(v27, fp_q[27]);
  EXPECT_FPU_ARM64(v28, fp_q[28]);
  EXPECT_FPU_ARM64(v29, fp_q[29]);
  EXPECT_FPU_ARM64(v30, fp_q[30]);
  EXPECT_FPU_ARM64(v31, fp_q[31]);
  EXPECT_FPU_ARM64(fpsr, fp_sr);
  EXPECT_FPU_ARM64(fpcr, fp_cr);
}

#endif // defined(__aarch64__)

#if defined(__mips64__)

#define EXPECT_GPR_MIPS64(lldb_reg, fbsd_regno)                                \
  EXPECT_THAT(GetRegParams(reg_ctx, gpr_##lldb_reg##_mips64),                  \
              ::testing::Pair(offsetof(reg, r_regs[fbsd_regno]),               \
                              sizeof(reg::r_regs[fbsd_regno])))
#define EXPECT_FPU_MIPS64(lldb_reg, fbsd_regno)                                \
  EXPECT_THAT(                                                                 \
      GetRegParams(reg_ctx, fpr_##lldb_reg##_mips64),                          \
      ::testing::Pair(offsetof(fpreg, r_regs[fbsd_regno]) + base_offset,       \
                      sizeof(fpreg::r_regs[fbsd_regno])))

TEST(RegisterContextFreeBSDTest, mips64) {
  ArchSpec arch{"mips64-unknown-freebsd"};
  RegisterContextFreeBSD_mips64 reg_ctx{arch};

  // we can not use aliases from <machine/regnum.h> because macros defined
  // there are not namespaced and collide a lot, e.g. 'A1'

  EXPECT_GPR_MIPS64(zero, 0);
  EXPECT_GPR_MIPS64(r1, 1);
  EXPECT_GPR_MIPS64(r2, 2);
  EXPECT_GPR_MIPS64(r3, 3);
  EXPECT_GPR_MIPS64(r4, 4);
  EXPECT_GPR_MIPS64(r5, 5);
  EXPECT_GPR_MIPS64(r6, 6);
  EXPECT_GPR_MIPS64(r7, 7);
  EXPECT_GPR_MIPS64(r8, 8);
  EXPECT_GPR_MIPS64(r9, 9);
  EXPECT_GPR_MIPS64(r10, 10);
  EXPECT_GPR_MIPS64(r11, 11);
  EXPECT_GPR_MIPS64(r12, 12);
  EXPECT_GPR_MIPS64(r13, 13);
  EXPECT_GPR_MIPS64(r14, 14);
  EXPECT_GPR_MIPS64(r15, 15);
  EXPECT_GPR_MIPS64(r16, 16);
  EXPECT_GPR_MIPS64(r17, 17);
  EXPECT_GPR_MIPS64(r18, 18);
  EXPECT_GPR_MIPS64(r19, 19);
  EXPECT_GPR_MIPS64(r20, 20);
  EXPECT_GPR_MIPS64(r21, 21);
  EXPECT_GPR_MIPS64(r22, 22);
  EXPECT_GPR_MIPS64(r23, 23);
  EXPECT_GPR_MIPS64(r24, 24);
  EXPECT_GPR_MIPS64(r25, 25);
  EXPECT_GPR_MIPS64(r26, 26);
  EXPECT_GPR_MIPS64(r27, 27);
  EXPECT_GPR_MIPS64(gp, 28);
  EXPECT_GPR_MIPS64(sp, 29);
  EXPECT_GPR_MIPS64(r30, 30);
  EXPECT_GPR_MIPS64(ra, 31);
  EXPECT_GPR_MIPS64(sr, 32);
  EXPECT_GPR_MIPS64(mullo, 33);
  EXPECT_GPR_MIPS64(mulhi, 34);
  EXPECT_GPR_MIPS64(badvaddr, 35);
  EXPECT_GPR_MIPS64(cause, 36);
  EXPECT_GPR_MIPS64(pc, 37);
  EXPECT_GPR_MIPS64(ic, 38);
  EXPECT_GPR_MIPS64(dummy, 39);

  size_t base_offset = reg_ctx.GetRegisterInfo()[fpr_f0_mips64].byte_offset;

  EXPECT_FPU_MIPS64(f0, 0);
}

#endif // defined(__mips64__)

#if defined(__powerpc__)

#define EXPECT_GPR_PPC(lldb_reg, fbsd_reg)                                     \
  EXPECT_THAT(GetRegParams(reg_ctx, gpr_##lldb_reg##_powerpc),                 \
              ::testing::Pair(offsetof(reg, fbsd_reg), sizeof(reg::fbsd_reg)))
#define EXPECT_FPU_PPC(lldb_reg, fbsd_reg)                                     \
  EXPECT_THAT(GetRegParams(reg_ctx, fpr_##lldb_reg##_powerpc),                 \
              ::testing::Pair(offsetof(fpreg, fbsd_reg) + base_offset,         \
                              sizeof(fpreg::fbsd_reg)))

TEST(RegisterContextFreeBSDTest, powerpc32) {
  ArchSpec arch{"powerpc-unknown-freebsd"};
  RegisterContextFreeBSD_powerpc32 reg_ctx{arch};

  EXPECT_GPR_PPC(r0, fixreg[0]);
  EXPECT_GPR_PPC(r1, fixreg[1]);
  EXPECT_GPR_PPC(r2, fixreg[2]);
  EXPECT_GPR_PPC(r3, fixreg[3]);
  EXPECT_GPR_PPC(r4, fixreg[4]);
  EXPECT_GPR_PPC(r5, fixreg[5]);
  EXPECT_GPR_PPC(r6, fixreg[6]);
  EXPECT_GPR_PPC(r7, fixreg[7]);
  EXPECT_GPR_PPC(r8, fixreg[8]);
  EXPECT_GPR_PPC(r9, fixreg[9]);
  EXPECT_GPR_PPC(r10, fixreg[10]);
  EXPECT_GPR_PPC(r11, fixreg[11]);
  EXPECT_GPR_PPC(r12, fixreg[12]);
  EXPECT_GPR_PPC(r13, fixreg[13]);
  EXPECT_GPR_PPC(r14, fixreg[14]);
  EXPECT_GPR_PPC(r15, fixreg[15]);
  EXPECT_GPR_PPC(r16, fixreg[16]);
  EXPECT_GPR_PPC(r17, fixreg[17]);
  EXPECT_GPR_PPC(r18, fixreg[18]);
  EXPECT_GPR_PPC(r19, fixreg[19]);
  EXPECT_GPR_PPC(r20, fixreg[20]);
  EXPECT_GPR_PPC(r21, fixreg[21]);
  EXPECT_GPR_PPC(r22, fixreg[22]);
  EXPECT_GPR_PPC(r23, fixreg[23]);
  EXPECT_GPR_PPC(r24, fixreg[24]);
  EXPECT_GPR_PPC(r25, fixreg[25]);
  EXPECT_GPR_PPC(r26, fixreg[26]);
  EXPECT_GPR_PPC(r27, fixreg[27]);
  EXPECT_GPR_PPC(r28, fixreg[28]);
  EXPECT_GPR_PPC(r29, fixreg[29]);
  EXPECT_GPR_PPC(r30, fixreg[30]);
  EXPECT_GPR_PPC(r31, fixreg[31]);
  EXPECT_GPR_PPC(lr, lr);
  EXPECT_GPR_PPC(cr, cr);
  EXPECT_GPR_PPC(xer, xer);
  EXPECT_GPR_PPC(ctr, ctr);
  EXPECT_GPR_PPC(pc, pc);

  size_t base_offset = reg_ctx.GetRegisterInfo()[fpr_f0_powerpc].byte_offset;

  EXPECT_FPU_PPC(f0, fpreg[0]);
  EXPECT_FPU_PPC(f1, fpreg[1]);
  EXPECT_FPU_PPC(f2, fpreg[2]);
  EXPECT_FPU_PPC(f3, fpreg[3]);
  EXPECT_FPU_PPC(f4, fpreg[4]);
  EXPECT_FPU_PPC(f5, fpreg[5]);
  EXPECT_FPU_PPC(f6, fpreg[6]);
  EXPECT_FPU_PPC(f7, fpreg[7]);
  EXPECT_FPU_PPC(f8, fpreg[8]);
  EXPECT_FPU_PPC(f9, fpreg[9]);
  EXPECT_FPU_PPC(f10, fpreg[10]);
  EXPECT_FPU_PPC(f11, fpreg[11]);
  EXPECT_FPU_PPC(f12, fpreg[12]);
  EXPECT_FPU_PPC(f13, fpreg[13]);
  EXPECT_FPU_PPC(f14, fpreg[14]);
  EXPECT_FPU_PPC(f15, fpreg[15]);
  EXPECT_FPU_PPC(f16, fpreg[16]);
  EXPECT_FPU_PPC(f17, fpreg[17]);
  EXPECT_FPU_PPC(f18, fpreg[18]);
  EXPECT_FPU_PPC(f19, fpreg[19]);
  EXPECT_FPU_PPC(f20, fpreg[20]);
  EXPECT_FPU_PPC(f21, fpreg[21]);
  EXPECT_FPU_PPC(f22, fpreg[22]);
  EXPECT_FPU_PPC(f23, fpreg[23]);
  EXPECT_FPU_PPC(f24, fpreg[24]);
  EXPECT_FPU_PPC(f25, fpreg[25]);
  EXPECT_FPU_PPC(f26, fpreg[26]);
  EXPECT_FPU_PPC(f27, fpreg[27]);
  EXPECT_FPU_PPC(f28, fpreg[28]);
  EXPECT_FPU_PPC(f29, fpreg[29]);
  EXPECT_FPU_PPC(f30, fpreg[30]);
  EXPECT_FPU_PPC(f31, fpreg[31]);
  EXPECT_FPU_PPC(fpscr, fpscr);
}

#endif // defined(__powerpc__)
