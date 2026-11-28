//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that the register infos used by NativeProcessOpenBSD match the layout
// of the structures that OpenBSD's ptrace(2) returns, since the register
// contexts access registers by their byte_offset into those structures.

// clang-format off
#include <sys/types.h>
#include <machine/reg.h>
// clang-format on

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Register/Arch/lldb-arm64-register-enums.h"
#include "Register/Arch/lldb-x86-register-enums.h"
#include "Register/Common/RegisterInfo_arm64.h"
#include "Register/OpenBSD/RegisterInfoOpenBSD_i386.h"
#include "Register/OpenBSD/RegisterInfoOpenBSD_x86_64.h"

#include <cstddef>
#include <optional>

using namespace lldb;
using namespace lldb_private;

static std::pair<size_t, size_t> GetRegParams(RegisterInfoInterface &ctx,
                                              uint32_t reg) {
  const RegisterInfo &info = ctx.GetRegisterInfo()[reg];
  return {info.byte_offset, info.byte_size};
}

[[maybe_unused]] static std::optional<uint32_t>
FindRegister(RegisterInfoInterface &ctx, llvm::StringRef name) {
  for (uint32_t i = 0; i < ctx.GetRegisterCount(); ++i)
    if (name == ctx.GetRegisterInfo()[i].name)
      return i;
  return std::nullopt;
}

#define EXPECT_REG(reg, offset, size)                                          \
  EXPECT_THAT(GetRegParams(reg_ctx, reg), ::testing::Pair(offset, size))

#if defined(__x86_64__)

#define EXPECT_GPR_X86_64(regname)                                             \
  EXPECT_REG(lldb_##regname##_x86_64, offsetof(reg, r_##regname),              \
             sizeof(reg::r_##regname))
#define EXPECT_FPR_X86_64(regname, offset, size)                               \
  EXPECT_REG(lldb_##regname##_x86_64, base_offset + (offset), size)

TEST(RegisterInfoOpenBSDTest, x86_64) {
  ArchSpec arch{"x86_64-unknown-openbsd"};
  RegisterInfoOpenBSD_x86_64 reg_ctx{arch};

  EXPECT_EQ(reg_ctx.GetGPRSize(), sizeof(reg));

  EXPECT_GPR_X86_64(rdi);
  EXPECT_GPR_X86_64(rsi);
  EXPECT_GPR_X86_64(rdx);
  EXPECT_GPR_X86_64(rcx);
  EXPECT_GPR_X86_64(r8);
  EXPECT_GPR_X86_64(r9);
  EXPECT_GPR_X86_64(r10);
  EXPECT_GPR_X86_64(r11);
  EXPECT_GPR_X86_64(r12);
  EXPECT_GPR_X86_64(r13);
  EXPECT_GPR_X86_64(r14);
  EXPECT_GPR_X86_64(r15);
  EXPECT_GPR_X86_64(rbp);
  EXPECT_GPR_X86_64(rbx);
  EXPECT_GPR_X86_64(rax);
  EXPECT_GPR_X86_64(rsp);
  EXPECT_GPR_X86_64(rip);
  EXPECT_GPR_X86_64(rflags);
  EXPECT_GPR_X86_64(cs);
  EXPECT_GPR_X86_64(ss);
  EXPECT_GPR_X86_64(ds);
  EXPECT_GPR_X86_64(es);
  EXPECT_GPR_X86_64(fs);
  EXPECT_GPR_X86_64(gs);

  // struct fpreg is the FXSAVE area. fctrl is its first field and gives the
  // offset of the whole area.
  EXPECT_EQ(sizeof(fpreg), 512u);
  size_t base_offset = reg_ctx.GetRegisterInfo()[lldb_fctrl_x86_64].byte_offset;

  EXPECT_FPR_X86_64(fctrl, offsetof(fxsave64, fx_fcw), 2);
  EXPECT_FPR_X86_64(fstat, offsetof(fxsave64, fx_fsw), 2);
  // TODO: This is a known bug, the abridged ftag is 8 bits in length.
  EXPECT_FPR_X86_64(ftag, offsetof(fxsave64, fx_ftw), 2);
  EXPECT_FPR_X86_64(fop, offsetof(fxsave64, fx_fop), 2);
  // fioff/fiseg and fooff/foseg are the two halves of FIP and FDP.
  EXPECT_FPR_X86_64(fioff, offsetof(fxsave64, fx_rip), 4);
  EXPECT_FPR_X86_64(fiseg, offsetof(fxsave64, fx_rip) + 4, 4);
  EXPECT_FPR_X86_64(fip, offsetof(fxsave64, fx_rip), 8);
  EXPECT_FPR_X86_64(fooff, offsetof(fxsave64, fx_rdp), 4);
  EXPECT_FPR_X86_64(foseg, offsetof(fxsave64, fx_rdp) + 4, 4);
  EXPECT_FPR_X86_64(fdp, offsetof(fxsave64, fx_rdp), 8);
  EXPECT_FPR_X86_64(mxcsr, offsetof(fxsave64, fx_mxcsr), 4);
  EXPECT_FPR_X86_64(mxcsrmask, offsetof(fxsave64, fx_mxcsr_mask), 4);

  for (uint32_t i = 0; i < 8; ++i) {
    const size_t st =
        offsetof(fxsave64, fx_st) + i * sizeof(fxsave64::fx_st[0]);
    EXPECT_REG(lldb_st0_x86_64 + i, base_offset + st, 10u) << "st" << i;
    EXPECT_REG(lldb_mm0_x86_64 + i, base_offset + st, 8u) << "mm" << i;
  }
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_REG(lldb_xmm0_x86_64 + i,
               base_offset + offsetof(fxsave64, fx_xmm) +
                   i * sizeof(fxsave64::fx_xmm[0]),
               sizeof(fxsave64::fx_xmm[0]))
        << "xmm" << i;
}

#endif // defined(__x86_64__)

#if defined(__i386__)

#define EXPECT_GPR_I386(regname)                                               \
  EXPECT_REG(lldb_##regname##_i386, offsetof(reg, r_##regname),                \
             sizeof(reg::r_##regname))
#define EXPECT_FPR_I386(regname, offset, size)                                 \
  EXPECT_REG(lldb_##regname##_i386, base_offset + (offset), size)

TEST(RegisterInfoOpenBSDTest, i386) {
  ArchSpec arch{"i386-unknown-openbsd"};
  RegisterInfoOpenBSD_i386 reg_ctx{arch};

  EXPECT_EQ(reg_ctx.GetGPRSize(), sizeof(reg));

  EXPECT_GPR_I386(eax);
  EXPECT_GPR_I386(ecx);
  EXPECT_GPR_I386(edx);
  EXPECT_GPR_I386(ebx);
  EXPECT_GPR_I386(esp);
  EXPECT_GPR_I386(ebp);
  EXPECT_GPR_I386(esi);
  EXPECT_GPR_I386(edi);
  EXPECT_GPR_I386(eip);
  EXPECT_GPR_I386(eflags);
  EXPECT_GPR_I386(cs);
  EXPECT_GPR_I386(ss);
  EXPECT_GPR_I386(ds);
  EXPECT_GPR_I386(es);
  EXPECT_GPR_I386(fs);
  EXPECT_GPR_I386(gs);

  // The FP registers are read with PT_GETXMMREGS, whose struct xmmregs is an
  // opaque FXSAVE area, so check against the FXSAVE layout. fctrl is its first
  // field and gives the offset of the whole area.
  EXPECT_EQ(sizeof(xmmregs), 512u);
  size_t base_offset = reg_ctx.GetRegisterInfo()[lldb_fctrl_i386].byte_offset;

  EXPECT_FPR_I386(fctrl, 0x00, 2);
  EXPECT_FPR_I386(fstat, 0x02, 2);
  // TODO: This is a known bug, the abridged ftag is 8 bits in length.
  EXPECT_FPR_I386(ftag, 0x04, 2);
  EXPECT_FPR_I386(fop, 0x06, 2);
  EXPECT_FPR_I386(fioff, 0x08, 4);
  EXPECT_FPR_I386(fiseg, 0x0C, 4);
  EXPECT_FPR_I386(fooff, 0x10, 4);
  EXPECT_FPR_I386(foseg, 0x14, 4);
  EXPECT_FPR_I386(mxcsr, 0x18, 4);
  EXPECT_FPR_I386(mxcsrmask, 0x1C, 4);

  for (uint32_t i = 0; i < 8; ++i) {
    EXPECT_REG(lldb_st0_i386 + i, base_offset + 0x20 + i * 0x10, 10u)
        << "st" << i;
    EXPECT_REG(lldb_mm0_i386 + i, base_offset + 0x20 + i * 0x10, 8u)
        << "mm" << i;
    EXPECT_REG(lldb_xmm0_i386 + i, base_offset + 0xA0 + i * 0x10, 16u)
        << "xmm" << i;
  }
}

#endif // defined(__i386__)

#if defined(__aarch64__)

TEST(RegisterInfoOpenBSDTest, arm64) {
  ArchSpec arch{"aarch64-unknown-openbsd"};
  // The register sets that NativeRegisterContextOpenBSD_arm64 provides.
  Flags opt_regsets =
      RegisterInfo_arm64::eRegsetMaskPAuth | RegisterInfo_arm64::eRegsetMaskTLS;
  RegisterInfo_arm64 reg_ctx{arch, opt_regsets};

  // struct reg starts with the same layout as RegisterInfo_arm64::GPR, but is
  // longer because it also holds r_tpidr.
  for (uint32_t i = 0; i <= 28; ++i)
    EXPECT_REG(gpr_x0_arm64 + i,
               offsetof(reg, r_reg) + i * sizeof(reg::r_reg[0]),
               sizeof(reg::r_reg[0]))
        << "x" << i;
  EXPECT_REG(gpr_fp_arm64, offsetof(reg, r_reg) + 29 * sizeof(reg::r_reg[0]),
             sizeof(reg::r_reg[0]));
  EXPECT_REG(gpr_lr_arm64, offsetof(reg, r_lr), sizeof(reg::r_lr));
  EXPECT_REG(gpr_sp_arm64, offsetof(reg, r_sp), sizeof(reg::r_sp));
  EXPECT_REG(gpr_pc_arm64, offsetof(reg, r_pc), sizeof(reg::r_pc));
  // cpsr is the low 32 bits of r_spsr.
  EXPECT_REG(gpr_cpsr_arm64, offsetof(reg, r_spsr), 4u);
  EXPECT_GE(sizeof(reg), sizeof(RegisterInfo_arm64::GPR));

  // The FP registers follow the GPR struct, and struct fpreg starts with the
  // same layout as RegisterInfo_arm64::FPU.
  size_t base_offset = reg_ctx.GetRegisterInfo()[fpu_v0_arm64].byte_offset;
  EXPECT_EQ(base_offset, sizeof(RegisterInfo_arm64::GPR));

  for (uint32_t i = 0; i < 32; ++i)
    EXPECT_REG(fpu_v0_arm64 + i,
               base_offset + offsetof(fpreg, fp_reg) +
                   i * sizeof(fpreg::fp_reg[0]),
               sizeof(fpreg::fp_reg[0]))
        << "v" << i;
  EXPECT_REG(fpu_fpsr_arm64, base_offset + offsetof(fpreg, fp_sr),
             sizeof(fpreg::fp_sr));
  EXPECT_REG(fpu_fpcr_arm64, base_offset + offsetof(fpreg, fp_cr),
             sizeof(fpreg::fp_cr));

  // PT_PACMASK returns the data mask followed by the code mask.
  std::optional<uint32_t> data_mask = FindRegister(reg_ctx, "data_mask");
  std::optional<uint32_t> code_mask = FindRegister(reg_ctx, "code_mask");
  ASSERT_TRUE(data_mask);
  ASSERT_TRUE(code_mask);
  EXPECT_REG(*data_mask, reg_ctx.GetPAuthOffset(), sizeof(register_t));
  EXPECT_REG(*code_mask, reg_ctx.GetPAuthOffset() + sizeof(register_t),
             sizeof(register_t));

  // The TLS set has only tpidr, which is r_tpidr in struct reg.
  std::optional<uint32_t> tpidr = FindRegister(reg_ctx, "tpidr");
  ASSERT_TRUE(tpidr);
  EXPECT_TRUE(reg_ctx.IsTLSReg(*tpidr));
  EXPECT_REG(*tpidr, reg_ctx.GetTLSOffset(), sizeof(reg::r_tpidr));
  EXPECT_FALSE(FindRegister(reg_ctx, "tpidr2"));
}

#endif // defined(__aarch64__)
