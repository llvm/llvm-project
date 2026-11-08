//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-loongarch-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_loongarch64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_loongarch64.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <array>

namespace reg_ids = lldb_private::registers::loongarch::id;
using namespace lldb_private;
using namespace llvm;

TEST(RegisterInfoLoongArchTest, LinuxStateIsAnExplicitExtension) {
  auto cpu = RegisterInfo_loongarch64::CreateDescription();
  EXPECT_EQ(cpu->Find("orig_a0"), nullptr);
  EXPECT_EQ(cpu->Find("reserved0"), nullptr);
  ASSERT_NE(cpu->Find(reg_ids::gpr_pc), nullptr);
  auto model = CreateLinuxLoongArch64RegisterLayout(
      ArchSpec("loongarch64-unknown-linux"));
  auto linux_description = model.GetDescription();
  EXPECT_NE(linux_description->Find("orig_a0"), nullptr);
  EXPECT_NE(linux_description->Find("reserved9"), nullptr);
  auto linux_provider = std::move(model);
  EXPECT_EQ(linux_provider.GetBankSize(0), 360u);
  EXPECT_EQ(linux_provider.GetBankSize(1), 272u);
  EXPECT_EQ(
      linux_provider.GetDescription()->GetRegisters()[reg_ids::gpr_pc].name,
      "pc");
  EXPECT_EQ(linux_provider.Find(reg_ids::gpr_pc)->pieces[0].byte_offset, 264u);

  auto other = llvm::cantFail(
      RegisterLayout::Create(cpu, RegisterLayout(lldb::eByteOrderLittle)));
  EXPECT_EQ(other.GetDescription()
                ->GetRegisters()[32]
                .numbers[lldb::eRegisterKindLLDB],
            reg_ids::gpr_pc);
  EXPECT_EQ(
      other.GetDescription()->FindGroupForRegister(reg_ids::fpr_fcsr),
      &other.GetDescription()->GetGroups()[RegisterInfo_loongarch64::FPRegSet]);
}

TEST(RegisterInfoLoongArchTest,
     ArchitectureDescriptionAcceptsAnotherBankLayout) {
  auto description = RegisterInfo_loongarch64::CreateDescription();
  RegisterLayout layout(lldb::eByteOrderBig);
  ASSERT_THAT_ERROR(layout.AddBank(9, 8), Succeeded());
  RegisterLocation pc;
  pc.kind = RegisterLocation::Stored;
  pc.pieces = {{9, 0, 8, 0}};
  ASSERT_THAT_ERROR(layout.Add(reg_ids::gpr_pc, pc), Succeeded());
  std::array<uint8_t, 8> bytes{1, 2, 3, 4, 5, 6, 7, 8};
  RegisterData bank{9, bytes};
  layout = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(layout)));

  auto value = layout.Read(reg_ids::gpr_pc, bank);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt64(), 0x0102030405060708ULL);
}
