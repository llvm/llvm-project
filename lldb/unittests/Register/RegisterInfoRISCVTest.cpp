//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-riscv-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_riscv32.h"
#include "lldb/Register/Common/RegisterInfo_riscv64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_riscv64.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <array>


namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb_private;
using namespace llvm;

TEST(RegisterInfoRISCVTest, OneDescriptionSupportsDifferentStorage) {
  auto description = RegisterInfo_riscv64::CreateDescription(0);
  auto linux_layout =
      CreateLinuxRISCV64RegisterLayout(ArchSpec("riscv64-unknown-linux"), 0);
  RegisterLayout other(lldb::eByteOrderBig);
  ASSERT_THAT_ERROR(other.AddBank(10, 24), Succeeded());
  RegisterLocation pc;
  pc.kind = RegisterLocation::Stored;
  pc.pieces = {{10, 16, 8, 0}};
  ASSERT_THAT_ERROR(other.Add(reg_ids::gpr_pc, pc), Succeeded());
  std::array<uint8_t, 256> linux_bytes{};
  std::array<uint8_t, 24> other_bytes{};
  MutableRegisterData linux_data{0, linux_bytes}, other_data{10, other_bytes};
  RegisterValue value(uint64_t(0x0102030405060708));
  ASSERT_THAT_ERROR(linux_layout.Write(reg_ids::gpr_pc, value, linux_data),
                    Succeeded());
  other = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(other)));

  ASSERT_THAT_ERROR(other.Write(reg_ids::gpr_pc, value, other_data),
                    Succeeded());
  EXPECT_EQ(linux_bytes[0], 8);
  EXPECT_EQ(other_bytes[16], 1);
  RegisterData linux_read{0, linux_bytes}, other_read{10, other_bytes};
  auto a = linux_layout.Read(reg_ids::gpr_pc, linux_read);
  auto b = other.Read(reg_ids::gpr_pc, other_read);
  ASSERT_THAT_EXPECTED(a, Succeeded());
  ASSERT_THAT_EXPECTED(b, Succeeded());
  EXPECT_EQ(a->GetAsUInt64(), b->GetAsUInt64());
  auto zero = linux_layout.Read(reg_ids::gpr_x0, {});
  ASSERT_THAT_EXPECTED(zero, Succeeded());
  EXPECT_EQ(zero->GetAsUInt64(), 0u);
}

TEST(RegisterInfoRISCVTest, FloatingPointAndConstantLocations) {
  Flags flags(RegisterInfo_riscv64::eRegsetMaskFP);
  auto model = CreateLinuxRISCV64RegisterLayout(
      ArchSpec("riscv64-unknown-linux"), flags);
  auto description = model.GetDescription();
  const auto &layout = model;
  auto provider = model;
  EXPECT_EQ(provider.GetDescription().get(), description.get());
  EXPECT_TRUE(provider.GetDescription()->FindGroup("fpr") != nullptr);
  EXPECT_EQ(provider.GetBankSize(0), 256u);
  EXPECT_EQ(provider.GetBankSize(1), 264u);
  EXPECT_EQ(layout.Find(reg_ids::fpr_fcsr)->pieces[0].bank, 1u);
  EXPECT_EQ(layout.Find(reg_ids::fpr_fcsr)->pieces[0].byte_offset, 256u);
  EXPECT_EQ(layout.Find(reg_ids::gpr_x0), nullptr);
  auto zero = layout.Read(reg_ids::gpr_x0, {});
  ASSERT_THAT_EXPECTED(zero, Succeeded());
  EXPECT_EQ(zero->GetAsUInt64(), 0u);
  EXPECT_EQ(provider.GetDescription()->FindGroupForRegister(
                provider.GetDescription()
                    ->FindByNumber(lldb::eRegisterKindLLDB, reg_ids::fpr_fcsr)
                    ->id),
            &provider.GetDescription()->GetGroups()[1]);
  // Padding after fcsr is not needed to decode its value.
  std::array<uint8_t, 260> fp{};
  fp[256] = 5;
  RegisterData bank{1, fp};
  auto fcsr = layout.Read(reg_ids::fpr_fcsr, bank);
  ASSERT_THAT_EXPECTED(fcsr, Succeeded());
  EXPECT_EQ(fcsr->GetAsUInt32(), 5u);
}

TEST(RegisterInfoRISCVTest,
     Rv32DescriptionDoesNotChooseAnOperatingSystemLayout) {
  ArchSpec arch("riscv32-unknown-freebsd");
  auto provider = llvm::cantFail(
      RegisterLayout::Create(RegisterInfo_riscv32::CreateDescription(),
                             RegisterLayout(arch.GetByteOrder())));
  EXPECT_EQ(provider.GetDescription()->Find(reg_ids::gpr_pc)->byte_size, 4u);
  EXPECT_EQ(provider.GetDescription()->Find(reg_ids::fpr_fcsr)->byte_size, 4u);
  EXPECT_EQ(provider.GetBankSize(0), 0u);
  EXPECT_EQ(provider.GetBankSize(1), 0u);
  EXPECT_EQ(provider.GetDescription()->GetGroups().size(), 2u);
}

TEST(RegisterInfoRISCVTest, CSRsHaveOwnedNamesAndStableIdentities) {
  llvm::SmallVector<RegisterDescriptor> base, extended;
  RegisterInfo_riscv32::BuildCSRegInfos({}, base);
  std::vector<std::string> features{"xqciint", "xqciint"};
  RegisterInfo_riscv32::BuildCSRegInfos(features, extended);
  ASSERT_EQ(base.size(), 4096u);
  ASSERT_EQ(extended.size(), base.size());
  EXPECT_EQ(base[3].name, "fcsr");
  EXPECT_EQ(base[0x7c0].name, "csr_0x7c0");
  EXPECT_EQ(extended[0x7c0].name, "qc.mmcr");
  EXPECT_EQ(extended[0x7c0].alt_name, "csr_0x7c0");
  EXPECT_EQ(base[0x7c0].id, reg_ids::csr_first + 0x7c0u);
  EXPECT_EQ(extended[0x7c0].id, base[0x7c0].id);
  features.clear();
  auto owned = extended[0x7c0];
  extended.clear();
  EXPECT_EQ(owned.name, "qc.mmcr");
  EXPECT_EQ(owned.alt_name, "csr_0x7c0");
}
