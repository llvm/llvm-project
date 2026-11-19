//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_x86_32.h"
#include "lldb/Register/Descriptions/RegisterDescription_x86_64.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(RegisterDescription_x86Test, PreservesSparseContextNumbersAndABI) {
  RegisterDescription_x86_64::Config config;
  config.abi = RegisterDescription_x86_64::Config::ABI::Windows;
  config.features =
      Flags(RegisterDescription_x86_64::Config::eRegsetMaskSegments);
  const RegisterDescription_x86_64 description(config);

  EXPECT_EQ(description.GetRegisters().size(), 40u);
  EXPECT_EQ(description.GetRegisters()[24].numbers[lldb::eRegisterKindLLDB],
            registers::x86_64::id::xmm0);
  EXPECT_EQ(description.Find("rcx")->numbers[lldb::eRegisterKindGeneric],
            static_cast<uint32_t>(LLDB_REGNUM_GENERIC_ARG1));
  EXPECT_EQ(description.Find("r9")->numbers[lldb::eRegisterKindGeneric],
            static_cast<uint32_t>(LLDB_REGNUM_GENERIC_ARG4));
  EXPECT_EQ(description.GetGroupAtIndex(1)->registers[0],
            description.Find("xmm0")->id);
  EXPECT_EQ(
      description
          .FindByNumber(lldb::eRegisterKindLLDB, registers::x86_64::id::rip)
          ->name,
      "rip");
  EXPECT_EQ(
      description.FindGroupForRegister(
          description
              .FindByNumber(lldb::eRegisterKindLLDB, registers::x86_64::id::rip)
              ->id),
      &description.GetGroups()[0]);
}

TEST(RegisterDescription_x86Test, SegmentBaseSelectionPreservesIdentities) {
  auto base = std::make_shared<RegisterDescription_x86_64>(
      RegisterDescription_x86_64::Config{});
  RegisterDescription_x86_64::Config config;
  config.segment_bases = true;
  auto with_bases = std::make_shared<RegisterDescription_x86_64>(config);
  ASSERT_NE(base->Find("rax"), nullptr);
  ASSERT_NE(with_bases->Find("rax"), nullptr);
  EXPECT_EQ(base->Find("rax")->id, with_bases->Find("rax")->id);
  EXPECT_EQ(base->Find("eax")->id, with_bases->Find("eax")->id);
  EXPECT_EQ(base->Find("eax")->byte_size, 4u);
  EXPECT_EQ(base->Find("eax")->zero_extend_parent, base->Find("rax")->id);
  EXPECT_EQ(base->Find("fs_base"), nullptr);
  ASSERT_NE(with_bases->Find("fs_base"), nullptr);
  EXPECT_EQ(with_bases->Find("fs_base")->id, registers::x86_64::id::fs_base);
  EXPECT_EQ(base->Find("orig_rax"), nullptr);
}

TEST(RegisterDescription_x86Test,
     ExposedWidthsPreserveDebugRegisterAvailability) {
  const std::pair<RegisterID, uint32_t> sizes[] = {
      {registers::x86_64::id::cs, 2}};
  RegisterDescription_x86_64::Config config;
  auto description =
      std::make_shared<RegisterDescription_x86_64>(config, sizes);

  EXPECT_EQ(description->Find(registers::x86_64::id::cs)->byte_size, 2u);
  EXPECT_NE(description->Find(registers::x86_64::id::dr0), nullptr);
  EXPECT_FALSE(llvm::is_contained(description->GetUserRegisters(),
                                  registers::x86_64::id::dr0));
}

TEST(RegisterDescription_x86Test, SIMDIsAlwaysPresentInX86_64Descriptions) {
  using Config = RegisterDescription_x86_64::Config;
  const uint32_t selections[] = {0,
                                 Config::eRegsetMaskAliases,
                                 Config::eRegsetMaskSegments,
                                 Config::eRegsetMaskX87,
                                 Config::eRegsetMaskAVX,
                                 Config::eRegsetMaskMPX,
                                 Config::eRegsetMaskDebug,
                                 Config::eRegsetMaskAll};

  for (uint32_t selected : selections) {
    Config config;
    config.features = Flags(selected);
    auto description = std::make_shared<RegisterDescription_x86_64>(config);

    EXPECT_NE(description->Find("xmm0"), nullptr);
    EXPECT_NE(description->Find("xmm15"), nullptr);
    const auto *fpu = description->FindGroup("fpu");
    ASSERT_NE(fpu, nullptr);
    EXPECT_TRUE(
        llvm::is_contained(fpu->registers, registers::x86_64::id::xmm0));
    EXPECT_TRUE(
        llvm::is_contained(fpu->registers, registers::x86_64::id::xmm15));
  }
}

TEST(RegisterDescription_x86Test, SIMDIsOptionalInX86_32) {
  RegisterDescription_x86_32::Config config;
  config.features = Flags(0);
  auto without = std::make_shared<RegisterDescription_x86_32>(config);
  EXPECT_EQ(without->Find("xmm0"), nullptr);

  config.features = Flags(RegisterDescription_x86_32::Config::eRegsetMaskSIMD);
  auto with = std::make_shared<RegisterDescription_x86_32>(config);
  EXPECT_NE(with->Find("xmm0"), nullptr);
  EXPECT_NE(with->Find("xmm7"), nullptr);
}

TEST(RegisterDescription_x86Test,
     DebugGroupIsAvailableOutsideDefaultUserRegisters) {
  const RegisterDescription_x86_32 x86(RegisterDescription_x86_32::Config{});
  const RegisterDescription_x86_64 x86_64(RegisterDescription_x86_64::Config{});
  const RegisterDescription *descriptions[] = {&x86, &x86_64};
  for (const auto *description : descriptions) {
    const auto *debug = description->FindGroup("dbr");
    ASSERT_NE(debug, nullptr);
    ASSERT_EQ(debug->registers.size(), 8u);
    EXPECT_NE(description->Find("dr0"), nullptr);
    EXPECT_NE(description->Find("dr7"), nullptr);

    for (RegisterID id : debug->registers)
      EXPECT_FALSE(llvm::is_contained(description->GetUserRegisters(), id));
  }
}
