//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_ppc64.h"
#include "lldb/Register/Common/RegisterInfo_ppc64le.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_powerpc.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_ppc64le.h"
#include "lldb/Register/RegisterLayout.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <array>

namespace reg_ids = lldb_private::registers::ppc64le::id;
using namespace lldb_private;
using namespace llvm;

TEST(RegisterInfoPowerPCTest, IdentityDoesNotDependOnEndiannessOrLinuxState) {
  ArchSpec big_arch("powerpc64-unknown-freebsd");
  ArchSpec little_arch("powerpc64le-unknown-freebsd");
  auto big = RegisterInfo_ppc64::CreateDescription(big_arch);
  auto little = RegisterInfo_ppc64le::CreateDescription(little_arch);
  EXPECT_EQ(big->Find("pc")->id, little->Find("pc")->id);
  EXPECT_EQ(big->Find("cr")->id, little->Find("cr")->id);
  EXPECT_EQ(big->Find("vs0")->id, little->Find("vs0")->id);
  for (unsigned i = 0; i != 64; ++i) {
    std::string name = "vs" + std::to_string(i);
    ASSERT_NE(big->Find(name), nullptr);
    ASSERT_NE(little->Find(name), nullptr);
    EXPECT_EQ(big->Find(name)->id, little->Find(name)->id);
  }
  EXPECT_EQ(little->Find("origr3"), nullptr);
  EXPECT_EQ(little->Find("softe"), nullptr);
  EXPECT_EQ(little->Find("trap"), nullptr);
  auto native = std::make_unique<RegisterLayout>(
      CreateLinuxPPC64LERegisterLayout(little_arch));
  EXPECT_NE(native->GetDescription()->Find("origr3"), nullptr);
  EXPECT_NE(native->GetDescription()->Find("trap"), nullptr);
  EXPECT_EQ(native->GetBankSize(0), 352u);
  EXPECT_EQ(native->GetBankSize(1), 264u);
  EXPECT_EQ(native->GetDescription()->GetRegisters()[reg_ids::gpr_pc].name,
            "pc");
  EXPECT_EQ(native->Find(native->GetDescription()->Find("pc")->id)
                ->pieces[0]
                .byte_offset,
            256u);
  EXPECT_EQ(native->GetDescription()->Find("pc")->id, big->Find("pc")->id);
}

TEST(RegisterInfoPowerPCTest, BigEndianDescriptionUsesExplicitStorage) {
  auto description = RegisterInfo_ppc64::CreateDescription(
      ArchSpec("powerpc64-unknown-freebsd"));
  RegisterID pc = description->Find("pc")->id;
  RegisterLayout layout(lldb::eByteOrderBig);
  ASSERT_THAT_ERROR(layout.AddBank(4, 16), Succeeded());
  RegisterLocation saved;
  saved.kind = RegisterLocation::Stored;
  saved.pieces = {{4, 8, 8, 0}};
  ASSERT_THAT_ERROR(layout.Add(pc, saved), Succeeded());
  std::array<uint8_t, 16> bytes{};
  MutableRegisterData bank{4, bytes};
  layout = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(layout)));

  ASSERT_THAT_ERROR(layout.Write(pc, RegisterValue(uint64_t(0x01020304)), bank),
                    Succeeded());
  EXPECT_EQ(bytes[12], 1);
  EXPECT_EQ(bytes[15], 4);
  RegisterData data{4, bytes};
  auto value = layout.Read(pc, data);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt64(), 0x01020304u);
}

TEST(RegisterInfoPowerPCTest, FreeBSD32ValuesIn64BitBigEndianSlots) {
  auto compat =
      CreateFreeBSDPowerPC64RegisterLayout(ArchSpec("powerpc-unknown-freebsd"));
  auto *pc = compat.GetDescription()->Find("pc");
  ASSERT_NE(pc, nullptr);
  EXPECT_EQ(pc->byte_size, 4u);
  auto *location = compat.Find(pc->id);
  ASSERT_NE(location, nullptr);
  EXPECT_EQ(compat.GetBankSize(0), 296u);
  EXPECT_EQ(location->pieces[0].byte_offset, 292u);
  EXPECT_EQ(location->pieces[0].byte_size, 4u);
  auto native32 =
      CreateFreeBSDPowerPC32RegisterLayout(ArchSpec("powerpc-unknown-freebsd"));
  EXPECT_EQ(native32.GetBankSize(0), 148u);
  auto *native_pc = native32.GetDescription()->Find("pc");
  EXPECT_EQ(native32.Find(native_pc->id)->pieces[0].byte_offset, 144u);
  auto generic = RegisterInfo_ppc64::CreateDescription(
      ArchSpec("powerpc64-unknown-unknown"));
  EXPECT_EQ(native_pc->id, generic->Find("pc")->id);
}

TEST(RegisterInfoPowerPCTest, LinuxVSXUsesTheActualSplitBanks) {
  auto info = std::make_unique<RegisterLayout>(
      CreateLinuxPPC64LERegisterLayout(ArchSpec("powerpc64le-unknown-linux")));
  const auto *descriptor = info->GetDescription()->Find("vs0");
  const auto *location = info->Find(descriptor->id);
  ASSERT_NE(location, nullptr);
  ASSERT_EQ(location->pieces.size(), 2u);
  EXPECT_EQ(location->pieces[0].bank, 3u);
  EXPECT_EQ(location->pieces[1].bank, 1u);
  std::array<uint8_t, 256> upper{}, floating{};
  upper[0] = 1;
  floating[0] = 2;
  const auto *reg = info->GetDescription()->FindByNumber(
      lldb::eRegisterKindLLDB, descriptor->numbers[lldb::eRegisterKindLLDB]);
  auto value = info->Read(reg->id, {{3, upper}, {1, floating}});
  ASSERT_THAT_EXPECTED(value, Succeeded());
  const auto *bytes = static_cast<const uint8_t *>(value->GetBytes());
  EXPECT_EQ(bytes[0], 1);
  EXPECT_EQ(bytes[8], 2);
}
