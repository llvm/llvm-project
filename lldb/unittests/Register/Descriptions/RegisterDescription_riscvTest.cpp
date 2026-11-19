//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-riscv32-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv32-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv32-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv64-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv64-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv64-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_riscv32.h"
#include "lldb/Register/Descriptions/RegisterDescription_riscv64.h"
#include "gtest/gtest.h"
#include <algorithm>

namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb_private;

TEST(RegisterDescription_riscvTest,
     Rv32DescriptionDoesNotChooseAnOperatingSystemLayout) {
  const RegisterDescription_riscv32 description(
      RegisterDescription_riscv32::Config{});
  EXPECT_EQ(description.Find(reg_ids::gpr_pc)->byte_size, 4u);
  EXPECT_EQ(description.Find(reg_ids::fpr_fcsr)->byte_size, 4u);
  EXPECT_EQ(description.GetGroups().size(), 2u);
}

TEST(RegisterDescription_riscvTest, SelectsArchitecturalGroups) {
  for (uint32_t width : {4u, 8u}) {
    for (uint32_t features : {0u, 1u, 2u, 3u}) {
      auto create = [&] {
        return width == 4 ? RegisterDescription_riscv32::GetShared(
                                RegisterDescription_riscv32::Config{features})
                          : RegisterDescription_riscv64::GetShared(
                                RegisterDescription_riscv64::Config{features});
      };
      auto description = create();
      EXPECT_EQ(description, create());
      EXPECT_EQ(description->GetRegisters().size(),
                33u + ((features & 1) ? 33u : 0u) +
                    ((features & 2) ? 32u : 0u));
      EXPECT_EQ(description->Find(reg_ids::gpr_pc)->byte_size, width);
      EXPECT_EQ(description->Find(reg_ids::gpr_x0)->constant_value,
                std::vector<uint8_t>(width, 0));
      EXPECT_EQ(description->FindGroup("fpr") != nullptr, (features & 1) != 0);
      EXPECT_EQ(description->FindGroup("vpr") != nullptr, (features & 2) != 0);
      EXPECT_EQ(description->Find(reg_ids::fpr_f0) != nullptr,
                (features & 1) != 0);
      EXPECT_EQ(description->Find(reg_ids::vpr_v0) != nullptr,
                (features & 2) != 0);
    }
  }
}

static std::optional<RegisterDescriptor>
FindCSR(uint32_t address, bool rv64 = false,
        llvm::ArrayRef<std::string> extensions = {}) {
  const uint32_t addresses[] = {address};
  auto description =
      rv64 ? RegisterDescription_riscv64::GetShared(
                 RegisterDescription_riscv64::Config{0, addresses, extensions})
           : RegisterDescription_riscv32::GetShared(
                 RegisterDescription_riscv32::Config{0, addresses, extensions});
  if (address > reg_ids::csr_last - reg_ids::csr_first)
    return std::nullopt;

  if (const auto *csr = description->Find(reg_ids::csr_first + address))
    return *csr;

  return std::nullopt;
}

TEST(RegisterDescription_riscvTest, CSRsHaveOwnedNamesAndStableIdentities) {
  auto base = FindCSR(0x7c0, false);
  std::vector<std::string> features{"xqciint", "xqciint"};
  auto extended = FindCSR(0x7c0, false, features);
  ASSERT_TRUE(base);
  ASSERT_TRUE(extended);
  auto fcsr = FindCSR(3, false);
  ASSERT_TRUE(fcsr);
  EXPECT_EQ(fcsr->name, "fcsr");
  EXPECT_EQ(fcsr->alt_name, "csr_0x003");
  EXPECT_EQ(base->name, "csr_0x7c0");
  EXPECT_EQ(extended->name, "qc.mmcr");
  EXPECT_EQ(extended->alt_name, "csr_0x7c0");
  EXPECT_EQ(base->id, reg_ids::csr_first + 0x7c0u);
  EXPECT_EQ(extended->id, base->id);
  features.clear();
  auto owned = *extended;
  extended.reset();
  EXPECT_EQ(owned.name, "qc.mmcr");
  EXPECT_EQ(owned.alt_name, "csr_0x7c0");

  std::vector<std::string> reordered{"unknown", "xqciint"};
  EXPECT_EQ(FindCSR(0x7c0, false, reordered)->name, owned.name);
  std::reverse(reordered.begin(), reordered.end());
  EXPECT_EQ(FindCSR(0x7c0, false, reordered)->name, owned.name);
}

TEST(RegisterDescription_riscvTest, CSRWidthAndNamesFollowXLEN) {
  for (uint32_t address : {0u, 3u, 0x340u, 0xfffu}) {
    auto rv32 = FindCSR(address, false);
    auto rv64 = FindCSR(address, true);
    ASSERT_TRUE(rv32);
    ASSERT_TRUE(rv64);
    EXPECT_EQ(rv32->id, rv64->id);
    EXPECT_EQ(rv32->name, rv64->name);
    EXPECT_EQ(rv32->alt_name, rv64->alt_name);
    EXPECT_EQ(rv32->numbers, rv64->numbers);
    EXPECT_EQ(rv32->byte_size, 4u);
    EXPECT_EQ(rv64->byte_size, 8u);
  }
  EXPECT_EQ(FindCSR(0x310, false)->name, "mstatush");
  EXPECT_EQ(FindCSR(0x310, true)->name, "csr_0x310");
  EXPECT_EQ(FindCSR(0x3a1, false)->name, "pmpcfg1");
  EXPECT_EQ(FindCSR(0x3a1, true)->name, "csr_0x3a1");
  std::vector<std::string> features{"xqciint"};
  EXPECT_EQ(FindCSR(0x7c0, true, features)->name, "csr_0x7c0");
  EXPECT_FALSE(FindCSR(0x1000, false));
  EXPECT_FALSE(FindCSR(0x1000, true));
}

TEST(RegisterDescription_riscvTest, FloatingControlHasNoSyntheticAlias) {
  for (auto description : {RegisterDescription_riscv32::GetShared({}),
                           RegisterDescription_riscv64::GetShared({})}) {
    const auto *fcsr = description->Find(reg_ids::fpr_fcsr);
    ASSERT_NE(fcsr, nullptr);
    EXPECT_EQ(fcsr->name, "fcsr");
    EXPECT_TRUE(fcsr->alt_name.empty());
    EXPECT_EQ(fcsr->byte_size, 4u);
  }
}

TEST(RegisterDescription_riscvTest, RequestedCSRsAreUniqueAndKeepRequestOrder) {
  const uint32_t addresses[] = {0x340, 3, 0x340, 0x1000, 0xffffffff, 0x7c0};
  auto description = RegisterDescription_riscv64::GetShared(
      RegisterDescription_riscv64::Config{0, addresses});
  const auto *group = description->FindGroup("csr");
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->registers,
            (std::vector<RegisterID>{reg_ids::csr_first + 0x340,
                                     reg_ids::csr_first + 3,
                                     reg_ids::csr_first + 0x7c0}));
  EXPECT_EQ(description->GetRegisters().size(), 36u);
  EXPECT_EQ(
      description->FindByNumber(lldb::eRegisterKindDWARF, 4096 + 0x340)->id,
      reg_ids::csr_first + 0x340);
}
