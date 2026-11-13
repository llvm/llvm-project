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
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;

static RegisterDescriptor MakeDescriptor(RegisterID id, llvm::StringRef name) {
  RegisterDescriptor reg;
  reg.id = id;
  reg.name = name.str();
  reg.byte_size = 8;
  return reg;
}

TEST(RegisterDescriptionFinalizationTest,
     ResolvesIDsIndependentOfPresentation) {
  auto parent = MakeDescriptor(91, "parent");
  auto alias = MakeDescriptor(7, "alias");
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

TEST(RegisterDescriptionFinalizationTest,
     ExpandsInvalidationIndependentOfOrder) {
  auto parent = MakeDescriptor(100, "parent");
  auto alias = MakeDescriptor(90, "alias");
  alias.value_registers = {100};
  alias.invalidate_registers = {100};
  auto sibling = MakeDescriptor(80, "sibling");
  sibling.value_registers = {100};
  sibling.invalidate_registers = {100};
  auto leaf = MakeDescriptor(70, "leaf");
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

TEST(RegisterDescriptionFinalizationTest,
     RejectsUnknownAndCyclicValueReferences) {
  auto first = MakeDescriptor(42, "first");
  first.value_registers = {99};
  auto unknown = RegisterDescription::Create({first});
  ASSERT_FALSE(bool(unknown));
  llvm::consumeError(unknown.takeError());
  auto second = MakeDescriptor(99, "second");
  second.value_registers = {42};
  auto cyclic = RegisterDescription::Create({second, first});
  ASSERT_FALSE(bool(cyclic));
  EXPECT_THAT(llvm::toString(cyclic.takeError()), testing::HasSubstr("cyclic"));
}

TEST(RegisterDescriptionFinalizationTest,
     FinalizedDescriptionRetainsTypeGraph) {
  auto enumeration = std::make_shared<RegisterTypeEnum>(
      "mode", RegisterTypeEnum::Enumerators{{0, "off"}, {1, "on"}});
  auto flags = std::make_shared<RegisterTypeFlags>(
      "status", 4,
      std::vector<RegisterTypeFlags::Field>{{"mode", 0, 1, enumeration.get()}});
  std::weak_ptr<const RegisterType> enum_lifetime = enumeration;
  std::weak_ptr<const RegisterType> flags_lifetime = flags;
  {
    auto reg = MakeDescriptor(123, "status");
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
