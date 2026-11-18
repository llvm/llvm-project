//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc32be.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc64be.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc64le.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_ppc.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_ppc64le.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <array>

namespace reg_ids = lldb_private::registers::ppc64le::id;
using namespace lldb_private;
using namespace llvm;

TEST(RegisterLayout_ppcTest, LinuxPPC64LEAddsSavedState) {
  ArchSpec little_arch("powerpc64le-unknown-freebsd");
  auto little = RegisterDescription_ppc64le::GetShared();
  auto native = std::make_unique<RegisterLayout>(
      RegisterLayoutLinux_ppc64le(little_arch));
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
  EXPECT_EQ(native->GetDescription()->Find("pc")->id, little->Find("pc")->id);
}

TEST(RegisterLayout_ppcTest, BigEndianDescriptionUsesExplicitStorage) {
  auto description = RegisterDescription_ppc64be::GetShared(
      RegisterDescription_ppc64be::Numbering::AIX);
  RegisterID pc = description->Find("pc")->id;
  RegisterLayout layout;
  ASSERT_THAT_ERROR(layout.AddBank(4, 16), Succeeded());
  RegisterLocation saved;
  saved.kind = RegisterLocation::Stored;
  saved.pieces = {{4, 8, 8, 0}};
  ASSERT_THAT_ERROR(layout.Add(pc, saved), Succeeded());
  std::array<uint8_t, 16> bytes{};
  MutableRegisterData bank{4, bytes};
  llvm::cantFail(
      layout.Finalize(std::make_shared<RegisterDescription>(*description),
                      lldb::eByteOrderBig));

  ASSERT_THAT_ERROR(layout.Write(pc, RegisterValue(uint64_t(0x01020304)), bank),
                    Succeeded());
  EXPECT_EQ(bytes[12], 1);
  EXPECT_EQ(bytes[15], 4);
  RegisterData data{4, bytes};
  auto value = layout.Read(pc, data);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt64(), 0x01020304u);
}

TEST(RegisterLayout_ppcTest, FreeBSD32ValuesIn64BitBigEndianSlots) {
  auto compat =
      RegisterLayoutFreeBSD_ppc(ArchSpec("powerpc-unknown-freebsd"), false, 8);
  auto *pc = compat.GetDescription()->Find("pc");
  ASSERT_NE(pc, nullptr);
  EXPECT_EQ(pc->byte_size, 4u);
  auto *location = compat.Find(pc->id);
  ASSERT_NE(location, nullptr);
  EXPECT_EQ(compat.GetBankSize(0), 296u);
  EXPECT_EQ(location->pieces[0].byte_offset, 292u);
  EXPECT_EQ(location->pieces[0].byte_size, 4u);
  auto native32 =
      RegisterLayoutFreeBSD_ppc(ArchSpec("powerpc-unknown-freebsd"));
  EXPECT_EQ(native32.GetBankSize(0), 148u);
  auto *native_pc = native32.GetDescription()->Find("pc");
  EXPECT_EQ(native32.Find(native_pc->id)->pieces[0].byte_offset, 144u);
  auto generic = RegisterDescription_ppc32be::GetShared();
  EXPECT_EQ(native_pc->id, generic->Find("pc")->id);
}

TEST(RegisterLayout_ppcTest, LinuxVSXUsesTheActualSplitBanks) {
  auto info = std::make_unique<RegisterLayout>(
      RegisterLayoutLinux_ppc64le(ArchSpec("powerpc64le-unknown-linux")));
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

TEST(RegisterLayout_ppcTest, FreeBSDVariantsShareSaveFormats) {
  struct Case {
    const char *triple;
    uint32_t slot_size;
    const char *vector_name;
    bool vsx;
  };
  const Case cases[] = {
      {"powerpc-unknown-freebsd", 4, "v31", false},
      {"powerpc-unknown-freebsd", 8, "v31", false},
      {"powerpc64-unknown-freebsd", 8, "vr31", false},
      {"powerpc64-unknown-freebsd", 8, "vr31", true},
      {"powerpc64le-unknown-freebsd", 8, "vr31", false},
      {"powerpc64le-unknown-freebsd", 8, "vr31", true},
  };

  for (const auto &test : cases) {
    SCOPED_TRACE(test.triple);
    SCOPED_TRACE(test.slot_size);
    SCOPED_TRACE(test.vsx);
    ArchSpec arch(test.triple);
    RegisterLayoutFreeBSD_ppc layout(arch, test.vsx, test.slot_size);
    EXPECT_EQ(layout.GetBankSize(0), 37u * test.slot_size);
    EXPECT_EQ(layout.GetBankSize(1), 33u * 8);
    EXPECT_EQ(layout.GetBankSize(2), 528u);
    EXPECT_EQ(layout.GetBankSize(3), test.vsx ? 256u : 0u);

    const auto *vector = layout.GetDescription()->Find(test.vector_name);
    ASSERT_NE(vector, nullptr);
    const auto *location = layout.Find(vector->id);
    ASSERT_NE(location, nullptr);
    ASSERT_EQ(location->pieces.size(), 1u);
    EXPECT_EQ(location->pieces[0].bank, 2u);
    EXPECT_EQ(location->pieces[0].byte_offset, 31u * 16);

    const auto *pc = layout.GetDescription()->Find("pc");
    ASSERT_NE(pc, nullptr);
    location = layout.Find(pc->id);
    ASSERT_NE(location, nullptr);
    const auto adjust = arch.GetByteOrder() == lldb::eByteOrderBig
                            ? test.slot_size - pc->byte_size
                            : 0u;
    EXPECT_EQ(location->pieces[0].byte_offset, 36u * test.slot_size + adjust);
  }
}
