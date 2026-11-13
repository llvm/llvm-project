//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterLayout.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "llvm/Support/Casting.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <array>

using namespace lldb_private;
using namespace llvm;

static RegisterDescriptor Register(RegisterID id, const char *name,
                                   uint32_t size = 8) {
  RegisterDescriptor reg;
  reg.id = id;
  reg.name = name;
  reg.byte_size = size;
  return reg;
}

static RegisterLocation Stored(RegisterBankID bank, uint64_t offset,
                               uint32_t size = 8) {
  RegisterLocation result;
  result.kind = RegisterLocation::Stored;
  result.pieces.push_back({bank, offset, size});

  return result;
}

TEST(RegisterDescriptionTest, OwnsNamesAndSeparatesIDsFromIndices) {
  auto first = Register(42, "pc");
  first.alt_name = "instruction_pointer";
  auto description = RegisterDescription::Create({first, Register(9, "sp")},
                                                 {{"General", "gpr", {9, 42}}});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  first.name = "changed";
  EXPECT_EQ(description->Find(42)->name, "pc");
  EXPECT_EQ(description->Find("instruction_pointer")->id, 42u);
  EXPECT_EQ(description->GetGroups()[0].registers[0], 9u);

  const RegisterID order[] = {9, 42};
  auto selected = description->WithUserRegisters(order);
  ASSERT_THAT_EXPECTED(selected, Succeeded());
  EXPECT_EQ(selected->GetUserRegisters()[0], 9u);
  EXPECT_EQ(selected->Find(selected->GetUserRegisters()[0])->name, "sp");
  EXPECT_EQ(selected->Find(9)->numbers[lldb::eRegisterKindLLDB],
            LLDB_INVALID_REGNUM);
  EXPECT_EQ(selected->GetGroups()[0].registers[0], 9u);
  EXPECT_EQ(selected->GetGroups()[0].registers[1], 42u);
}

TEST(RegisterDescriptionTest, RejectsInvalidIdentitiesAndReferences) {
  EXPECT_THAT_EXPECTED(
      RegisterDescription::Create({Register(1, "a"), Register(1, "b")}),
      Failed());
  auto alias = Register(1, "alias");
  alias.value_registers = {99};
  EXPECT_THAT_EXPECTED(RegisterDescription::Create({alias}), Failed());
  EXPECT_THAT_EXPECTED(
      RegisterDescription::Create({Register(1, "a")}, {{"gpr", "gpr", {99}}}),
      Failed());
}

TEST(RegisterLayoutBankTest, SameDescriptionDifferentLayoutsAndByteOrders) {
  auto description = RegisterDescription::Create({Register(42, "pc")});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout little(lldb::eByteOrderLittle);
  RegisterLayout big(lldb::eByteOrderBig);
  ASSERT_THAT_ERROR(little.AddBank(7, 16), Succeeded());
  ASSERT_THAT_ERROR(big.AddBank(7, 16), Succeeded());
  ASSERT_THAT_ERROR(little.Add(42, Stored(7, 0)), Succeeded());
  ASSERT_THAT_ERROR(big.Add(42, Stored(7, 8)), Succeeded());
  std::array<uint8_t, 16> a{}, b{};
  a.fill(0xcc);
  b.fill(0xdd);
  MutableRegisterData a_data{7, a}, b_data{7, b};
  RegisterValue value(uint64_t(0x0102030405060708));
  little = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(little)));

  ASSERT_THAT_ERROR(little.Write(42, value, a_data), Succeeded());
  big = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(big)));

  ASSERT_THAT_ERROR(big.Write(42, value, b_data), Succeeded());
  EXPECT_EQ(a[0], 8);
  EXPECT_EQ(a[8], 0xcc);
  EXPECT_EQ(b[0], 0xdd);
  EXPECT_EQ(b[8], 1);
  RegisterData a_read{7, a}, b_read{7, b};
  auto a_value = little.Read(42, a_read);
  auto b_value = big.Read(42, b_read);
  ASSERT_THAT_EXPECTED(a_value, Succeeded());
  ASSERT_THAT_EXPECTED(b_value, Succeeded());
  EXPECT_EQ(a_value->GetAsUInt64(), value.GetAsUInt64());
  EXPECT_EQ(b_value->GetAsUInt64(), value.GetAsUInt64());
}

TEST(RegisterLayoutBankTest, CompositeConstantAndUnavailableRegisters) {
  auto description = RegisterDescription::Create(
      {Register(1, "split"), Register(2, "zero"), Register(3, "missing")});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout(lldb::eByteOrderLittle);
  ASSERT_THAT_ERROR(layout.AddBank(8, 4), Succeeded());
  ASSERT_THAT_ERROR(layout.AddBank(9, 4), Succeeded());
  RegisterLocation split;
  split.kind = RegisterLocation::Stored;
  split.pieces = {{8, 0, 4, 0}, {9, 0, 4, 4}};
  ASSERT_THAT_ERROR(layout.Add(1, split), Succeeded());
  RegisterLocation zero;
  zero.kind = RegisterLocation::Constant;
  zero.constant.resize(8);
  ASSERT_THAT_ERROR(layout.Add(2, zero), Succeeded());
  ASSERT_THAT_ERROR(layout.Add(3, RegisterLocation{}), Succeeded());
  std::array<uint8_t, 4> low{1, 2, 3, 4}, high{5, 6, 7, 8};
  const RegisterData banks[] = {{8, low}, {9, high}};
  layout = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(layout)));

  auto result = layout.Read(1, banks);
  ASSERT_THAT_EXPECTED(result, Succeeded());
  EXPECT_EQ(result->GetAsUInt64(), 0x0807060504030201ULL);
  auto constant = layout.Read(2, {});
  ASSERT_THAT_EXPECTED(constant, Succeeded());
  EXPECT_EQ(constant->GetAsUInt64(), 0u);
  EXPECT_THAT_EXPECTED(layout.Read(3, banks), Failed());
}

TEST(RegisterLayoutBankTest, RejectsTruncationWithoutPartialWrites) {
  auto description = RegisterDescription::Create({Register(1, "split")});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout(lldb::eByteOrderLittle);
  ASSERT_THAT_ERROR(layout.AddBank(0, 4), Succeeded());
  ASSERT_THAT_ERROR(layout.AddBank(1, 4), Succeeded());
  RegisterLocation split;
  split.kind = RegisterLocation::Stored;
  split.pieces = {{0, 0, 4, 0}, {1, 0, 4, 4}};
  ASSERT_THAT_ERROR(layout.Add(1, split), Succeeded());
  std::array<uint8_t, 4> first{9, 9, 9, 9};
  std::array<uint8_t, 3> truncated{8, 8, 8};
  const MutableRegisterData banks[] = {{0, first}, {1, truncated}};
  layout = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(layout)));

  EXPECT_THAT_ERROR(layout.Write(1, RegisterValue(uint64_t(0)), banks),
                    Failed());
  EXPECT_EQ(first, (std::array<uint8_t, 4>{9, 9, 9, 9}));
}

TEST(RegisterLayoutTest, RejectsOverflowAndOverlappingPieces) {
  auto description = RegisterDescription::Create({Register(1, "r")});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout overflow(lldb::eByteOrderLittle);
  ASSERT_THAT_ERROR(overflow.AddBank(0, 8), Succeeded());
  ASSERT_THAT_ERROR(overflow.Add(1, Stored(0, UINT64_MAX)), Succeeded());
  EXPECT_THAT_ERROR(RegisterLayout::Create(
                        std::make_shared<RegisterDescription>(*description),
                        std::move(overflow))
                        .takeError(),
                    Failed());
  RegisterLayout overlap(lldb::eByteOrderLittle);
  ASSERT_THAT_ERROR(overlap.AddBank(0, 8), Succeeded());
  RegisterLocation pieces;
  pieces.kind = RegisterLocation::Stored;
  pieces.pieces = {{0, 0, 4, 0}, {0, 4, 4, 0}};
  ASSERT_THAT_ERROR(overlap.Add(1, pieces), Succeeded());
  EXPECT_THAT_ERROR(RegisterLayout::Create(
                        std::make_shared<RegisterDescription>(*description),
                        std::move(overlap))
                        .takeError(),
                    Failed());
}

TEST(RegisterLayoutBankTest, ArchitectureConstantDoesNotRequireAStorageBank) {
  auto zero = Register(51, "zero", 4);
  zero.constant_value = {0, 0, 0, 0};
  auto description = RegisterDescription::Create({zero});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout(lldb::eByteOrderBig);
  layout = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(layout)));

  auto value = layout.Read(51, {});
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt32(), 0u);
  EXPECT_THAT_ERROR(layout.Write(51, RegisterValue(uint32_t(1)), {}), Failed());
}

TEST(RegisterLayoutBankTest,
     ExplicitlyZeroFilledLanesAreReadAndValidatedOnWrite) {
  auto description = RegisterDescription::Create({Register(1, "vector", 16)});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout(lldb::eByteOrderLittle);
  ASSERT_THAT_ERROR(layout.AddBank(3, 8), Succeeded());
  auto location = Stored(3, 0, 8);
  location.zero_fill = true;
  ASSERT_THAT_ERROR(layout.Add(1, location), Succeeded());
  std::array<uint8_t, 8> bytes{};
  bytes[0] = 17;
  RegisterData input{3, bytes};
  layout = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(layout)));

  auto value = layout.Read(1, input);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetByteSize(), 16u);
  EXPECT_EQ(static_cast<const uint8_t *>(value->GetBytes())[0], 17);
  EXPECT_EQ(static_cast<const uint8_t *>(value->GetBytes())[15], 0);
  std::array<uint8_t, 16> invalid{};
  invalid[15] = 1;
  RegisterValue write(invalid, lldb::eByteOrderLittle);
  MutableRegisterData output{3, bytes};
  EXPECT_THAT_ERROR(layout.Write(1, write, output), Failed());
  EXPECT_EQ(bytes[0], 17);
}

TEST(RegisterDescriptionTest, OwnsFieldTypesAndTheirEnums) {
  auto mode = std::make_shared<RegisterTypeEnum>(
      "mode", RegisterTypeEnum::Enumerators{{0, "off"}, {1, "on"}});
  auto flags = std::make_shared<RegisterTypeFlags>(
      "control", 8,
      std::vector<RegisterTypeFlags::Field>{{"mode", 0, 1, mode.get()}});
  auto reg = Register(1, "control");
  reg.type = flags.get();
  auto description = RegisterDescription::Create({reg}, {}, {mode, flags});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  mode.reset();
  flags.reset();
  auto *type = llvm::cast<RegisterTypeFlags>(description->Find(1)->type);
  EXPECT_EQ(type->GetID(), "control");
  EXPECT_EQ(type->GetFields().back().GetEnum()->GetEnumerators()[1].m_name,
            "on");
}

TEST(RegisterDescriptionTest, ResolvesNumberingWithoutConfusingIDsAndIndices) {
  auto pc = Register(42, "pc");
  pc.numbers[lldb::eRegisterKindDWARF] = 17;
  pc.numbers[lldb::eRegisterKindEHFrame] = 31;
  pc.numbers[lldb::eRegisterKindLLDB] = 900;
  auto sp = Register(9, "sp");
  sp.numbers[lldb::eRegisterKindDWARF] = 8;
  auto description = RegisterDescription::Create({pc, sp});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  EXPECT_EQ(description->FindByNumber(lldb::eRegisterKindDWARF, 17)->id, 42u);
  EXPECT_EQ(description->GetRegisterIndex(42), 0u);
  EXPECT_EQ(description->GetRegisterIndex(9), 1u);
  EXPECT_FALSE(description->GetRegisterIndex(17));
  EXPECT_EQ(description->ConvertNumber(lldb::eRegisterKindDWARF, 17,
                                       lldb::eRegisterKindEHFrame),
            31u);
  EXPECT_EQ(description->ConvertNumber(lldb::eRegisterKindDWARF, 17,
                                       lldb::eRegisterKindLLDB),
            900u);
  EXPECT_EQ(description->ConvertNumber(lldb::eRegisterKindDWARF, 8,
                                       lldb::eRegisterKindEHFrame),
            LLDB_INVALID_REGNUM);
  EXPECT_EQ(description->FindByNumber(lldb::eRegisterKindLLDB, 42), nullptr);
  EXPECT_EQ(description->FindByNumber(lldb::eRegisterKindLLDB, 0), nullptr);
  EXPECT_EQ(description->FindByNumber(lldb::eRegisterKindGeneric,
                                      LLDB_INVALID_REGNUM),
            nullptr);
  EXPECT_EQ(description->FindByNumber(lldb::kNumRegisterKinds, 0), nullptr);
  EXPECT_EQ(description->FindByNumber(static_cast<lldb::RegisterKind>(-1), 0),
            nullptr);
}

TEST(RegisterDescriptionTest, PreservesFirstExposedExternalNumber) {
  auto first = Register(42, "first");
  auto alias = Register(9, "alias");
  first.numbers[lldb::eRegisterKindDWARF] = 17;
  alias.numbers[lldb::eRegisterKindDWARF] = 17;
  auto description = RegisterDescription::Create({first, alias});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  EXPECT_EQ(description->FindByNumber(lldb::eRegisterKindDWARF, 17)->id, 42u);
}

TEST(RegisterDescriptionTest, PreservesGenericRoleAndNameLookupPolicy) {
  auto small_sp = Register(9, "sp", 2);
  small_sp.alt_name = "small_stack";
  auto full_sp = Register(42, "rsp");
  full_sp.numbers[lldb::eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP;
  auto description = RegisterDescription::Create({small_sp, full_sp});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  EXPECT_EQ(description->FindByName("SP")->id, 42u);
  EXPECT_EQ(description->FindByName("sp", 2)->id, 42u);
  EXPECT_EQ(description->FindByName("SMALL_STACK")->id, 9u);
  EXPECT_EQ(description->FindByName("rsp")->id, 42u);
  EXPECT_EQ(description->FindByName("small_stack", 1), nullptr);
  EXPECT_EQ(description->FindByName(""), nullptr);
  EXPECT_EQ(description->FindByName("missing"), nullptr);
  // Literal format construction keeps its exact-name query separate from the
  // debugger's generic-role spelling policy.
  EXPECT_EQ(description->Find("sp")->id, 9u);
}

TEST(RegisterDescriptionTest, OwnsBoundedDefinitionInputs) {
  std::string name = "low";
  std::string alternate = "low_alias";
  RegisterID values[] = {7};
  RegisterID invalidates[] = {7};
  RegisterDescriptor::Numbering numbers;
  numbers.fill(LLDB_INVALID_REGNUM);
  numbers[lldb::eRegisterKindLLDB] = 900;
  RegisterDescriptor low(42, name, alternate.c_str(), 4, lldb::eEncodingUint,
                         lldb::eFormatHex, numbers, values, invalidates);
  auto description = RegisterDescription::Create({low, Register(7, "full")});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  name = "changed";
  alternate = "changed";
  values[0] = invalidates[0] = 999;
  EXPECT_EQ(description->FindByName("low_alias")->id, 42u);
  EXPECT_EQ(description->Find(42)->value_registers,
            (std::vector<RegisterID>{7}));
  EXPECT_EQ(description->Find(42)->invalidate_registers,
            (std::vector<RegisterID>{7}));
  EXPECT_EQ(description->FindByNumber(lldb::eRegisterKindLLDB, 900)->id, 42u);
}

TEST(RegisterLayoutTest, ComparesStorageIndependentlyOfSchemaOwnership) {
  auto make = [](uint64_t offset) {
    auto description = std::make_shared<RegisterDescription>(
        cantFail(RegisterDescription::Create({Register(42, "value")})));
    RegisterLayout draft(lldb::eByteOrderLittle);
    cantFail(draft.AddBank(3, 16));
    cantFail(draft.Add(42, Stored(3, offset)));
    return cantFail(RegisterLayout::Create(description, std::move(draft)));
  };
  auto first = make(0);
  auto same = make(0);
  EXPECT_NE(first.GetDescription(), same.GetDescription());
  EXPECT_TRUE(first.HasSameStorage(same));
  EXPECT_FALSE(first.HasSameStorage(make(8)));

  // Changing a draft after finalization requires validating it again.
  cantFail(same.AddBank(4, 8));
  EXPECT_FALSE(same.GetDescription());
  EXPECT_THAT_EXPECTED(same.GetRequiredBanks(42), Failed());
}
