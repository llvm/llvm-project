//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterDescription.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "llvm/Support/Casting.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

static RegisterDescriptor MakeRegister(RegisterID id, const char *name,
                                   uint32_t size = 8) {
  RegisterDescriptor reg;
  reg.id = id;
  reg.name = name;
  reg.byte_size = size;
  return reg;
}

TEST(RegisterDescriptionTest, OwnsNamesAndSeparatesIDsFromIndices) {
  auto first = MakeRegister(42, "pc");
  first.alt_name = "instruction_pointer";
  auto description = RegisterDescription::Create({first, MakeRegister(9, "sp")},
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
      RegisterDescription::Create({MakeRegister(1, "a"), MakeRegister(1, "b")}),
      Failed());
  auto alias = MakeRegister(1, "alias");
  alias.value_registers = {99};
  EXPECT_THAT_EXPECTED(RegisterDescription::Create({alias}), Failed());
  EXPECT_THAT_EXPECTED(
      RegisterDescription::Create({MakeRegister(1, "a")}, {{"gpr", "gpr", {99}}}),
      Failed());
}

TEST(RegisterDescriptionTest, OwnsFieldTypesAndTheirEnums) {
  auto mode = std::make_shared<RegisterTypeEnum>(
      "mode", RegisterTypeEnum::Enumerators{{0, "off"}, {1, "on"}});
  auto flags = std::make_shared<RegisterTypeFlags>(
      "control", 8,
      std::vector<RegisterTypeFlags::Field>{{"mode", 0, 1, mode.get()}});
  auto reg = MakeRegister(1, "control");
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
  auto pc = MakeRegister(42, "pc");
  pc.numbers[lldb::eRegisterKindDWARF] = 17;
  pc.numbers[lldb::eRegisterKindEHFrame] = 31;
  pc.numbers[lldb::eRegisterKindLLDB] = 900;
  auto sp = MakeRegister(9, "sp");
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
  auto first = MakeRegister(42, "first");
  auto alias = MakeRegister(9, "alias");
  first.numbers[lldb::eRegisterKindDWARF] = 17;
  alias.numbers[lldb::eRegisterKindDWARF] = 17;
  auto description = RegisterDescription::Create({first, alias});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  EXPECT_EQ(description->FindByNumber(lldb::eRegisterKindDWARF, 17)->id, 42u);
}

TEST(RegisterDescriptionTest, PreservesGenericRoleAndNameLookupPolicy) {
  auto small_sp = MakeRegister(9, "sp", 2);
  small_sp.alt_name = "small_stack";
  auto full_sp = MakeRegister(42, "rsp");
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
  auto description = RegisterDescription::Create({low, MakeRegister(7, "full")});
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

TEST(RegisterDescriptionTest, ResolvesIDsIndependentOfPresentation) {
  auto parent = MakeRegister(91, "parent");
  auto alias = MakeRegister(7, "alias");
  alias.byte_size = 4;
  alias.value_registers = {91};
  alias.invalidate_registers = {91};
  auto description =
      RegisterDescription::Create({alias, parent}, {{"values", "v", {91, 7}}});
  ASSERT_TRUE(bool(description));
  EXPECT_EQ(description->Find(7)->value_registers.front(), 91u);
  EXPECT_EQ(description->GetGroups()[0].registers.front(), 91u);
  EXPECT_EQ(description->GetRegisterIndex(91), 1u);
}

TEST(RegisterDescriptionTest, ExpandsInvalidationIndependentOfOrder) {
  auto parent = MakeRegister(100, "parent");
  auto alias = MakeRegister(90, "alias");
  alias.value_registers = {100};
  alias.invalidate_registers = {100};
  auto sibling = MakeRegister(80, "sibling");
  sibling.value_registers = {100};
  sibling.invalidate_registers = {100};
  auto leaf = MakeRegister(70, "leaf");
  leaf.value_registers = {80};
  leaf.invalidate_registers = {80};
  parent.invalidate_registers = {90, 80};
  sibling.invalidate_registers.push_back(70);
  auto first = RegisterDescription::Create({parent, alias, sibling, leaf});
  auto reordered = RegisterDescription::Create({leaf, sibling, alias, parent});
  ASSERT_TRUE(bool(first));
  ASSERT_TRUE(bool(reordered));
  EXPECT_THAT(first->Find(90)->invalidate_registers,
              testing::ElementsAre(70, 80, 100));
  for (const auto &reg : first->GetRegisters())
    EXPECT_EQ(reg.invalidate_registers,
              reordered->Find(reg.id)->invalidate_registers);
}

TEST(RegisterDescriptionTest, RejectsUnknownAndCyclicValueReferences) {
  auto first = MakeRegister(42, "first");
  first.value_registers = {99};
  auto unknown = RegisterDescription::Create({first});
  ASSERT_FALSE(bool(unknown));
  llvm::consumeError(unknown.takeError());
  auto second = MakeRegister(99, "second");
  second.value_registers = {42};
  auto cyclic = RegisterDescription::Create({second, first});
  ASSERT_FALSE(bool(cyclic));
  EXPECT_THAT(llvm::toString(cyclic.takeError()), testing::HasSubstr("cyclic"));
}

TEST(RegisterDescriptionTest, FinalizedDescriptionRetainsTypeGraph) {
  auto enumeration = std::make_shared<RegisterTypeEnum>(
      "mode", RegisterTypeEnum::Enumerators{{0, "off"}, {1, "on"}});
  auto flags = std::make_shared<RegisterTypeFlags>(
      "status", 4,
      std::vector<RegisterTypeFlags::Field>{{"mode", 0, 1, enumeration.get()}});
  std::weak_ptr<const RegisterType> enum_lifetime = enumeration;
  std::weak_ptr<const RegisterType> flags_lifetime = flags;
  {
    auto reg = MakeRegister(123, "status");
    reg.type = flags.get();
    auto description =
        RegisterDescription::Create({reg}, {}, {enumeration, flags});
    ASSERT_TRUE(bool(description));
    enumeration.reset();
    flags.reset();
    EXPECT_FALSE(enum_lifetime.expired());
    EXPECT_FALSE(flags_lifetime.expired());
    auto *type = llvm::cast<RegisterTypeFlags>(description->Find(123)->type);
    EXPECT_EQ(type->GetFields().back().GetEnum()->GetEnumerators()[1].m_name,
              "on");
  }
  EXPECT_TRUE(enum_lifetime.expired());
  EXPECT_TRUE(flags_lifetime.expired());
}
