//===-- RegisterContextMinidumpTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/Utility/RegisterContextLinux_i386.h"
#include "Plugins/Process/Utility/RegisterContextLinux_x86_64.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86_32.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86_64.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_ARM.h"
#include "lldb/Utility/DataBuffer.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::minidump;

static uint32_t reg32(const DataBuffer &Buf, const RegisterInfo &Info) {
  return *reinterpret_cast<const uint32_t *>(Buf.GetBytes() + Info.byte_offset);
}

static uint64_t reg64(const DataBuffer &Buf, const RegisterInfo &Info) {
  return *reinterpret_cast<const uint64_t *>(Buf.GetBytes() + Info.byte_offset);
}

TEST(RegisterContextMinidump, ConvertMinidumpContext_x86_32) {
  MinidumpContext_x86_32 Context;
  Context.context_flags =
      static_cast<uint32_t>(MinidumpContext_x86_32_Flags::x86_32_Flag |
                            MinidumpContext_x86_32_Flags::Control |
                            MinidumpContext_x86_32_Flags::Segments |
                            MinidumpContext_x86_32_Flags::Integer);
  Context.eax = 0x00010203;
  Context.ebx = 0x04050607;
  Context.ecx = 0x08090a0b;
  Context.edx = 0x0c0d0e0f;
  Context.edi = 0x10111213;
  Context.esi = 0x14151617;
  Context.ebp = 0x18191a1b;
  Context.esp = 0x1c1d1e1f;
  Context.eip = 0x20212223;
  Context.eflags = 0x24252627;
  Context.cs = 0x2829;
  Context.fs = 0x2a2b;
  Context.gs = 0x2c2d;
  Context.ss = 0x2e2f;
  Context.ds = 0x3031;
  Context.es = 0x3233;
  llvm::ArrayRef<uint8_t> ContextRef(reinterpret_cast<uint8_t *>(&Context),
                                     sizeof(Context));

  ArchSpec arch("i386-pc-linux");
  auto RegInterface = std::make_unique<RegisterContextLinux_i386>(arch);
  lldb::DataBufferSP Buf =
      ConvertMinidumpContext_x86_32(ContextRef, RegInterface.get());
  ASSERT_EQ(RegInterface->GetGPRSize(), Buf->GetByteSize());

  const RegisterInfo *Info = RegInterface->GetRegisterInfo();
  ASSERT_NE(nullptr, Info);

  EXPECT_EQ(Context.eax, reg32(*Buf, Info[lldb_eax_i386]));
  EXPECT_EQ(Context.ebx, reg32(*Buf, Info[lldb_ebx_i386]));
  EXPECT_EQ(Context.ecx, reg32(*Buf, Info[lldb_ecx_i386]));
  EXPECT_EQ(Context.edx, reg32(*Buf, Info[lldb_edx_i386]));
  EXPECT_EQ(Context.edi, reg32(*Buf, Info[lldb_edi_i386]));
  EXPECT_EQ(Context.esi, reg32(*Buf, Info[lldb_esi_i386]));
  EXPECT_EQ(Context.ebp, reg32(*Buf, Info[lldb_ebp_i386]));
  EXPECT_EQ(Context.esp, reg32(*Buf, Info[lldb_esp_i386]));
  EXPECT_EQ(Context.eip, reg32(*Buf, Info[lldb_eip_i386]));
  EXPECT_EQ(Context.eflags, reg32(*Buf, Info[lldb_eflags_i386]));
  EXPECT_EQ(Context.cs, reg32(*Buf, Info[lldb_cs_i386]));
  EXPECT_EQ(Context.fs, reg32(*Buf, Info[lldb_fs_i386]));
  EXPECT_EQ(Context.gs, reg32(*Buf, Info[lldb_gs_i386]));
  EXPECT_EQ(Context.ss, reg32(*Buf, Info[lldb_ss_i386]));
  EXPECT_EQ(Context.ds, reg32(*Buf, Info[lldb_ds_i386]));
  EXPECT_EQ(Context.es, reg32(*Buf, Info[lldb_es_i386]));
}

TEST(RegisterContextMinidump, ConvertMinidumpContext_x86_64) {
  MinidumpContext_x86_64 Context;
  Context.context_flags =
      static_cast<uint32_t>(MinidumpContext_x86_64_Flags::x86_64_Flag |
                            MinidumpContext_x86_64_Flags::Control |
                            MinidumpContext_x86_64_Flags::Segments |
                            MinidumpContext_x86_64_Flags::Integer);
  Context.rax = 0x0001020304050607;
  Context.rbx = 0x08090a0b0c0d0e0f;
  Context.rcx = 0x1011121314151617;
  Context.rdx = 0x18191a1b1c1d1e1f;
  Context.rdi = 0x2021222324252627;
  Context.rsi = 0x28292a2b2c2d2e2f;
  Context.rbp = 0x3031323334353637;
  Context.rsp = 0x38393a3b3c3d3e3f;
  Context.r8 = 0x4041424344454647;
  Context.r9 = 0x48494a4b4c4d4e4f;
  Context.r10 = 0x5051525354555657;
  Context.r11 = 0x58595a5b5c5d5e5f;
  Context.r12 = 0x6061626364656667;
  Context.r13 = 0x68696a6b6c6d6e6f;
  Context.r14 = 0x7071727374757677;
  Context.r15 = 0x78797a7b7c7d7e7f;
  Context.rip = 0x8081828384858687;
  Context.eflags = 0x88898a8b;
  Context.cs = 0x8c8d;
  Context.fs = 0x8e8f;
  Context.gs = 0x9091;
  Context.ss = 0x9293;
  Context.ds = 0x9495;
  Context.ss = 0x9697;
  llvm::ArrayRef<uint8_t> ContextRef(reinterpret_cast<uint8_t *>(&Context),
                                     sizeof(Context));

  ArchSpec arch("x86_64-pc-linux");
  auto RegInterface = std::make_unique<RegisterContextLinux_x86_64>(arch);
  lldb::DataBufferSP Buf =
      ConvertMinidumpContext_x86_64(ContextRef, RegInterface.get());
  ASSERT_EQ(RegInterface->GetGPRSize(), Buf->GetByteSize());

  const RegisterInfo *Info = RegInterface->GetRegisterInfo();
  EXPECT_EQ(Context.rax, reg64(*Buf, Info[lldb_rax_x86_64]));
  EXPECT_EQ(Context.rbx, reg64(*Buf, Info[lldb_rbx_x86_64]));
  EXPECT_EQ(Context.rcx, reg64(*Buf, Info[lldb_rcx_x86_64]));
  EXPECT_EQ(Context.rdx, reg64(*Buf, Info[lldb_rdx_x86_64]));
  EXPECT_EQ(Context.rdi, reg64(*Buf, Info[lldb_rdi_x86_64]));
  EXPECT_EQ(Context.rsi, reg64(*Buf, Info[lldb_rsi_x86_64]));
  EXPECT_EQ(Context.rbp, reg64(*Buf, Info[lldb_rbp_x86_64]));
  EXPECT_EQ(Context.rsp, reg64(*Buf, Info[lldb_rsp_x86_64]));
  EXPECT_EQ(Context.r8, reg64(*Buf, Info[lldb_r8_x86_64]));
  EXPECT_EQ(Context.r9, reg64(*Buf, Info[lldb_r9_x86_64]));
  EXPECT_EQ(Context.r10, reg64(*Buf, Info[lldb_r10_x86_64]));
  EXPECT_EQ(Context.r11, reg64(*Buf, Info[lldb_r11_x86_64]));
  EXPECT_EQ(Context.r12, reg64(*Buf, Info[lldb_r12_x86_64]));
  EXPECT_EQ(Context.r13, reg64(*Buf, Info[lldb_r13_x86_64]));
  EXPECT_EQ(Context.r14, reg64(*Buf, Info[lldb_r14_x86_64]));
  EXPECT_EQ(Context.r15, reg64(*Buf, Info[lldb_r15_x86_64]));
  EXPECT_EQ(Context.rip, reg64(*Buf, Info[lldb_rip_x86_64]));
  EXPECT_EQ(Context.eflags, reg64(*Buf, Info[lldb_rflags_x86_64]));
  EXPECT_EQ(Context.cs, reg64(*Buf, Info[lldb_cs_x86_64]));
  EXPECT_EQ(Context.fs, reg64(*Buf, Info[lldb_fs_x86_64]));
  EXPECT_EQ(Context.gs, reg64(*Buf, Info[lldb_gs_x86_64]));
  EXPECT_EQ(Context.ss, reg64(*Buf, Info[lldb_ss_x86_64]));
  EXPECT_EQ(Context.ds, reg64(*Buf, Info[lldb_ds_x86_64]));
  EXPECT_EQ(Context.es, reg64(*Buf, Info[lldb_es_x86_64]));
}

static void TestARMRegInfo(const lldb_private::RegisterInfo *info) {
  // Make sure we have valid register numbers for eRegisterKindEHFrame and
  // eRegisterKindDWARF for GPR registers r0-r15 so that we can unwind
  // correctly when using this information.
  llvm::StringRef name(info->name);
  llvm::StringRef alt_name(info->alt_name);
  if (name.startswith("r") || alt_name.startswith("r")) {
    EXPECT_NE(info->kinds[lldb::eRegisterKindEHFrame], LLDB_INVALID_REGNUM);
    EXPECT_NE(info->kinds[lldb::eRegisterKindDWARF], LLDB_INVALID_REGNUM);
  }
  // Verify generic register are set correctly
  if (name == "r0") {
    EXPECT_EQ(info->kinds[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_ARG1);
  } else if (name == "r1") {
    EXPECT_EQ(info->kinds[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_ARG2);
  } else if (name == "r2") {
    EXPECT_EQ(info->kinds[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_ARG3);
  } else if (name == "r3") {
    EXPECT_EQ(info->kinds[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_ARG4);
  } else if (name == "sp") {
    EXPECT_EQ(info->kinds[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_SP);
  } else if (name == "fp") {
    EXPECT_EQ(info->kinds[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_FP);
  } else if (name == "lr") {
    EXPECT_EQ(info->kinds[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_RA);
  } else if (name == "pc") {
    EXPECT_EQ(info->kinds[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_PC);
  } else if (name == "cpsr") {
    EXPECT_EQ(info->kinds[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_FLAGS);
  }
}

TEST(RegisterContextMinidump, CheckRegisterContextMinidump_ARM) {
  size_t num_regs = RegisterContextMinidump_ARM::GetRegisterCountStatic();
  const lldb_private::RegisterInfo *reg_info;
  for (size_t reg=0; reg<num_regs; ++reg) {
    reg_info = RegisterContextMinidump_ARM::GetRegisterInfoAtIndexStatic(reg,
                                                                         true);
    TestARMRegInfo(reg_info);
    reg_info = RegisterContextMinidump_ARM::GetRegisterInfoAtIndexStatic(reg,
                                                                         false);
    TestARMRegInfo(reg_info);
  }
}
