//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/Windows/Common/RegisterLayoutWindows_x86.h"
#include "lldb/Register/Arch/lldb-x86_32-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Utility/ArchSpec.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(RegisterLayoutWindows_x86Test, NativeSegmentsMatchBackendValueWidths) {
  auto info = RegisterLayoutWindows_x86(
      ArchSpec("x86_64-pc-windows"), RegisterLayoutWindows_x86::Mode::X86_64);
  const auto &description = *info.GetDescription();
  for (const char *name : {"cs", "ds", "es", "fs", "gs", "ss"}) {
    const auto *reg = description.Find(name);
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->byte_size, 2u);
  }
  EXPECT_EQ(description.Find("rax")->byte_size, 8u);
  EXPECT_EQ(description.Find("rflags")->byte_size, 8u);
  EXPECT_EQ(description.Find("xmm0")->byte_size, 16u);

  auto compat = RegisterLayoutWindows_x86(ArchSpec("i686-pc-windows"),
                                          RegisterLayoutWindows_x86::Mode::X86);
  EXPECT_EQ(compat.GetDescription()->Find("cs")->byte_size, 4u);
}

TEST(RegisterLayoutWindows_x86Test, FrameX86PreservesRegisterView) {
  ArchSpec arch("i686-pc-windows");
  auto info =
      RegisterLayoutWindows_x86(arch, RegisterLayoutWindows_x86::Mode::X86,
                                RegisterLayoutWindows_x86::ContextKind::Frame);
  const char *names[] = {"eax", "ebx", "ecx", "edx", "edi",
                         "esi", "ebp", "esp", "eip", "eflags"};
  ASSERT_EQ(info.GetDescription()->GetRegisters().size(), 10u);
  for (size_t i = 0; i < std::size(names); ++i) {
    const auto &reg = info.GetDescription()->GetRegisters()[i];
    EXPECT_EQ(reg.name, names[i]);
    EXPECT_TRUE(reg.value_registers.empty());
    EXPECT_TRUE(reg.invalidate_registers.empty());
  }
  ASSERT_EQ(info.GetDescription()->GetGroups().size(), 1u);
  EXPECT_EQ(info.GetDescription()->GetGroupAtIndex(0)->registers.size(), 10u);
  for (uint32_t i = 0; i < 10; ++i)
    EXPECT_EQ(info.GetDescription()->GetGroupAtIndex(0)->registers[i],
              info.GetDescription()->GetRegisters()[i].id);
  const auto *flags = info.GetDescription()->Find("eflags");
  ASSERT_NE(flags, nullptr);
  EXPECT_EQ(flags->format, lldb::eFormatBinary);
  EXPECT_EQ(flags->alt_name, "flags");
  EXPECT_EQ(flags->byte_size, 4u);
  EXPECT_EQ(info.GetDescription()->Find("cs"), nullptr);
  EXPECT_EQ(info.GetDescription()->Find("st0"), nullptr);
  EXPECT_EQ(info.GetDescription()->Find("dr0"), nullptr);
  EXPECT_EQ(flags->numbers[lldb::eRegisterKindDWARF], 9u);
  EXPECT_EQ(flags->numbers[lldb::eRegisterKindEHFrame], 9u);
  EXPECT_EQ(flags->numbers[lldb::eRegisterKindLLDB],
            registers::x86_32::id::eflags);
  EXPECT_EQ(info.GetDescription()->Find("ax"), nullptr);
  EXPECT_EQ(info.GetDescription()->Find("xmm0"), nullptr);
}

TEST(RegisterLayoutWindows_x86Test, FrameX64PreservesRegisterView) {
  ArchSpec arch("x86_64-pc-windows");
  auto info =
      RegisterLayoutWindows_x86(arch, RegisterLayoutWindows_x86::Mode::X86_64,
                                RegisterLayoutWindows_x86::ContextKind::Frame);
  const char *names[] = {
      "rax",   "rbx",    "rcx",   "rdx",   "rdi",   "rsi",  "rbp",  "rsp",
      "r8",    "r9",     "r10",   "r11",   "r12",   "r13",  "r14",  "r15",
      "rip",   "eflags", "eax",   "ebx",   "ecx",   "edx",  "edi",  "esi",
      "ebp",   "esp",    "r8d",   "r9d",   "r10d",  "r11d", "r12d", "r13d",
      "r14d",  "r15d",   "ax",    "bx",    "cx",    "dx",   "di",   "si",
      "bp",    "sp",     "r8w",   "r9w",   "r10w",  "r11w", "r12w", "r13w",
      "r14w",  "r15w",   "ah",    "bh",    "ch",    "dh",   "al",   "bl",
      "cl",    "dl",     "dil",   "sil",   "bpl",   "spl",  "r8l",  "r9l",
      "r10l",  "r11l",   "r12l",  "r13l",  "r14l",  "r15l", "xmm0", "xmm1",
      "xmm2",  "xmm3",   "xmm4",  "xmm5",  "xmm6",  "xmm7", "xmm8", "xmm9",
      "xmm10", "xmm11",  "xmm12", "xmm13", "xmm14", "xmm15"};
  ASSERT_EQ(info.GetDescription()->GetRegisters().size(), 86u);
  for (size_t i = 0; i < std::size(names); ++i) {
    const auto &reg = info.GetDescription()->GetRegisters()[i];
    EXPECT_EQ(reg.name, names[i]);
    EXPECT_TRUE(reg.value_registers.empty());
    EXPECT_TRUE(reg.invalidate_registers.empty());
  }
  ASSERT_EQ(info.GetDescription()->GetGroups().size(), 2u);
  EXPECT_EQ(info.GetDescription()->GetGroupAtIndex(0)->registers.size(), 70u);
  for (uint32_t i = 0; i < 70; ++i)
    EXPECT_EQ(info.GetDescription()->GetGroupAtIndex(0)->registers[i],
              info.GetDescription()->GetRegisters()[i].id);
  const auto *flags = info.GetDescription()->Find("eflags");
  ASSERT_NE(flags, nullptr);
  EXPECT_EQ(flags->format, lldb::eFormatBinary);
  EXPECT_EQ(flags->alt_name, "flags");
  EXPECT_EQ(flags->byte_size, 8u);
  EXPECT_EQ(info.GetDescription()->Find("cs"), nullptr);
  EXPECT_EQ(info.GetDescription()->Find("st0"), nullptr);
  EXPECT_EQ(info.GetDescription()->Find("dr0"), nullptr);
  EXPECT_EQ(flags->numbers[lldb::eRegisterKindDWARF], LLDB_INVALID_REGNUM);
  EXPECT_EQ(flags->numbers[lldb::eRegisterKindEHFrame], LLDB_INVALID_REGNUM);
  EXPECT_EQ(flags->numbers[lldb::eRegisterKindLLDB],
            registers::x86_64::id::rflags);
  EXPECT_EQ(info.GetDescription()->Find("eax")->byte_size, 4u);
  EXPECT_EQ(info.GetDescription()->Find("ah")->byte_size, 1u);
  EXPECT_EQ(
      info.GetDescription()->Find("rcx")->numbers[lldb::eRegisterKindGeneric],
      static_cast<uint32_t>(LLDB_REGNUM_GENERIC_ARG1));
  EXPECT_EQ(
      info.GetDescription()->Find("r9")->numbers[lldb::eRegisterKindGeneric],
      static_cast<uint32_t>(LLDB_REGNUM_GENERIC_ARG4));
  EXPECT_EQ(info.GetDescription()->Find("xmm0")->encoding, lldb::eEncodingUint);
  EXPECT_EQ(info.GetDescription()->Find("xmm0")->format,
            lldb::eFormatVectorOfUInt64);
  EXPECT_EQ(info.GetDescription()->GetGroupAtIndex(1)->registers.size(), 16u);
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(info.GetDescription()->GetGroupAtIndex(1)->registers[i],
              info.GetDescription()->GetRegisters()[70 + i].id);
}

TEST(RegisterLayoutWindows_x86Test, NativeProfilesRetainSegments) {
  auto info_i386 = RegisterLayoutWindows_x86(
      ArchSpec("i686-pc-windows"), RegisterLayoutWindows_x86::Mode::X86);
  auto info_x64 = RegisterLayoutWindows_x86(
      ArchSpec("x86_64-pc-windows"), RegisterLayoutWindows_x86::Mode::X86_64);
  EXPECT_EQ(info_i386.GetDescription()->GetRegisters().size(), 16u);
  EXPECT_EQ(info_x64.GetDescription()->GetRegisters().size(), 40u);
  EXPECT_NE(info_i386.GetDescription()->Find("cs"), nullptr);
  EXPECT_NE(info_x64.GetDescription()->Find("cs"), nullptr);
  EXPECT_EQ(info_x64.GetDescription()->Find("eax"), nullptr);
  EXPECT_NE(info_x64.GetDescription()->Find("xmm0"), nullptr);
}
