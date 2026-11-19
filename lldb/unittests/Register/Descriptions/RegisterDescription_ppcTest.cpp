//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-ppc32be-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc32be-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64be-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64be-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64le-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc32be.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc64be.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc64le.h"
#include "gtest/gtest.h"

namespace reg_ids = lldb_private::registers::ppc64le::id;
using namespace lldb_private;

TEST(RegisterDescription_ppcTest, IdentitiesBelongToTheirArchitecture) {
  auto big = RegisterDescription_ppc64be::GetShared(
      RegisterDescription_ppc64be::Numbering::AIX);
  auto little = RegisterDescription_ppc64le::GetShared();
  EXPECT_EQ(big->Find("pc")->id, registers::ppc64be::id::gpr_pc);
  EXPECT_EQ(little->Find("pc")->id, reg_ids::gpr_pc);
  EXPECT_EQ(big->Find("cr")->id, registers::ppc64be::id::gpr_cr);
  EXPECT_EQ(little->Find("cr")->id, reg_ids::gpr_cr);
  EXPECT_EQ(big->Find("vs0")->id, registers::ppc64be::id::vsx_vs0);
  EXPECT_EQ(little->Find("vs0")->id, reg_ids::vsx_vs0);
  for (unsigned i = 0; i != 64; ++i) {
    std::string name = "vs" + std::to_string(i);
    ASSERT_NE(big->Find(name), nullptr);
    ASSERT_NE(little->Find(name), nullptr);
    EXPECT_EQ(big->Find(name)->byte_size, 16u);
    EXPECT_EQ(little->Find(name)->byte_size, 16u);
  }
  EXPECT_EQ(little->Find("origr3"), nullptr);
  EXPECT_EQ(little->Find("softe"), nullptr);
  EXPECT_EQ(little->Find("trap"), nullptr);
}

TEST(RegisterDescription_ppcTest, PPC32BEHas32BitGPRValues) {
  auto description = RegisterDescription_ppc32be::GetShared();
  for (const auto &reg : description->GetRegisters())
    if (description->IsInGroup(reg.id, "gpr"))
      EXPECT_EQ(reg.byte_size, 4u);
  EXPECT_EQ(description->Find("pc")->id, registers::ppc32be::id::gpr_pc);
  EXPECT_EQ(description->Find("lr")->numbers[lldb::eRegisterKindDWARF], 65u);
  EXPECT_EQ(description->Find("cr")->numbers[lldb::eRegisterKindDWARF], 68u);
}

TEST(RegisterDescription_ppcTest, SharedNumberingCatalogIsPPC64Only) {
  namespace numbers = registers::ppc64::dwarf;
  EXPECT_EQ(registers::ppc64be::dwarf::aix::lr, numbers::aix::lr);
  EXPECT_EQ(registers::ppc64be::dwarf::elf::lr, numbers::elf::lr);
  EXPECT_EQ(registers::ppc64le::dwarf::lr, numbers::elf::lr);
  EXPECT_NE(uint32_t(numbers::aix::lr), uint32_t(numbers::elf::lr));
  EXPECT_NE(uint32_t(registers::ppc32be::dwarf::vrsave),
            uint32_t(numbers::elf::vrsave));
}
