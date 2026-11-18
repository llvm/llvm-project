//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-x86_32-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_x86_64.h"
#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_x86_32.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_x86_64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_x86_32.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_x86_64.h"
#include "lldb/Register/Layouts/NetBSD/RegisterLayoutNetBSD_x86_32.h"
#include "lldb/Register/Layouts/NetBSD/RegisterLayoutNetBSD_x86_64.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <array>

using namespace lldb_private;
using namespace llvm;
using FXSAVE = RegisterLayout_x86::FXSAVE;

TEST(RegisterLayout_x86Test, AliasesAndZeroExtensionAreArchitectureRules) {
  RegisterLayout draft;
  ASSERT_THAT_ERROR(draft.AddBank(0, 24), Succeeded());
  RegisterLocation parent;
  parent.kind = RegisterLocation::Stored;
  parent.pieces = {{0, 8, 8}};
  ASSERT_THAT_ERROR(draft.Add(registers::x86_64::id::rax, parent), Succeeded());
  RegisterLayout format = std::move(draft);
  llvm::cantFail(format.Finalize(std::make_shared<RegisterDescription_x86_64>(
                                     RegisterDescription_x86_64::Config{}),
                                 lldb::eByteOrderLittle));
  const auto &description = *format.GetDescription();
  std::array<uint8_t, 16> bytes;
  bytes.fill(0xcc);
  MutableRegisterData bank{0, bytes};
  ASSERT_THAT_ERROR(format.Write(description.Find("eax")->id,
                                 RegisterValue(uint32_t(0x12345678)), bank),
                    Succeeded());
  EXPECT_EQ(bytes[0], 0xcc);
  EXPECT_EQ(bytes[8], 0x78);
  EXPECT_EQ(bytes[9], 0x56);
  EXPECT_EQ(bytes[12], 0);
  EXPECT_EQ(bytes[15], 0);
  ASSERT_THAT_ERROR(format.Write(description.Find("ah")->id,
                                 RegisterValue(uint8_t(0xab)), bank),
                    Succeeded());
  EXPECT_EQ(bytes[8], 0x78);
  EXPECT_EQ(bytes[9], 0xab);
  EXPECT_EQ(bytes[10], 0x34);
}

TEST(RegisterLayout_x86Test, Compat32LayoutOwnsRegisterNames) {
  auto layout = RegisterLayoutFreeBSD_x86_32(
      ArchSpec("i386-unknown-freebsd"), RegisterLayout_x86::GPRMode::Compat64);
  EXPECT_EQ(layout.GetBankSize(0), 176u);
  EXPECT_EQ(
      layout.GetDescription()->GetRegisters()[registers::x86_32::id::eax].name,
      "eax");
  EXPECT_EQ(layout.GetDescription()
                ->GetRegisters()[registers::x86_32::id::eax]
                .byte_size,
            4u);
  EXPECT_EQ(layout.Find(layout.GetDescription()->Find("eax")->id)
                ->pieces[0]
                .byte_offset,
            14 * 8u);
  EXPECT_EQ(layout.GetDescription()->Find("eax")->byte_size, 4u);
}

template <typename Layout>
static RegisterLayout MakeLayout(const ArchSpec &arch,
                                 RegisterLayout_x86::TagWordView tag_word) {
  return Layout(arch, tag_word);
}

template <typename Layout32>
static RegisterLayout
MakeCompatLayout(const ArchSpec &arch,
                 RegisterLayout_x86::TagWordView tag_word) {
  return Layout32(arch, RegisterLayout_x86::GPRMode::Compat64, tag_word);
}

TEST(RegisterLayout_x86Test, PerOSLayoutsDescribeTheSameHardwareFormats) {
  using Factory =
      RegisterLayout (*)(const ArchSpec &, RegisterLayout_x86::TagWordView);
  const struct {
    Factory make;
    bool target64;
    bool segment_bases;
  } cases[] = {
      {MakeLayout<RegisterLayoutFreeBSD_x86_32>, false, false},
      {MakeLayout<RegisterLayoutLinux_x86_32>, false, false},
      {MakeLayout<RegisterLayoutNetBSD_x86_32>, false, false},
      {MakeCompatLayout<RegisterLayoutFreeBSD_x86_32>, false, false},
      {MakeCompatLayout<RegisterLayoutLinux_x86_32>, false, false},
      {MakeCompatLayout<RegisterLayoutNetBSD_x86_32>, false, false},
      {MakeLayout<RegisterLayoutFreeBSD_x86_64>, true, false},
      {MakeLayout<RegisterLayoutLinux_x86_64>, true, true},
      {MakeLayout<RegisterLayoutNetBSD_x86_64>, true, false},
  };

  for (const auto &test : cases) {
    ArchSpec arch(test.target64 ? "x86_64-unknown-none" : "i386-unknown-none");
    auto layout = test.make(arch, RegisterLayout_x86::TagWordView::Full);
    const auto &description = *layout.GetDescription();
    auto location = [&](const char *name) {
      const auto *reg = description.Find(name);
      return reg ? layout.Find(reg->id) : nullptr;
    };

    ASSERT_NE(location("fctrl"), nullptr);
    EXPECT_EQ(location("fctrl")->pieces.front().byte_offset,
              offsetof(FXSAVE, fctrl));
    ASSERT_NE(location("ftag"), nullptr);
    EXPECT_EQ(location("ftag")->conversion,
              &RegisterLayout_x86::GetX87TagConversion());
    ASSERT_NE(location("ymm0"), nullptr);
    ASSERT_EQ(location("ymm0")->pieces.size(), 2u);
    EXPECT_EQ(location("ymm0")->pieces[0].bank, RegisterLayout_x86::FP);
    EXPECT_EQ(location("ymm0")->pieces[1].bank, RegisterLayout_x86::YMMHigh);
    EXPECT_EQ(location("ymm0")->pieces[1].value_offset, 16u);
    EXPECT_EQ(description.Find("xmm15") != nullptr, test.target64);
    EXPECT_EQ(description.Find("fs_base") != nullptr, test.segment_bases);
  }
}
