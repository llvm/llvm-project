//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/gdb-remote/RegisterPacketLayout.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

namespace {
RegisterDescriptor Register(const char *name, RegisterID id, uint32_t size) {
  RegisterDescriptor result;
  result.id = id;
  result.name = name;
  result.byte_size = size;
  // An external number is neither the ID nor the packet index. Aliases may
  // even share this number; packet relationships must still use actual IDs.
  result.numbers[lldb::eRegisterKindLLDB] = 500;
  return result;
}
std::shared_ptr<const RegisterDescription>
Description(std::vector<RegisterDescriptor> registers) {
  return std::make_shared<RegisterDescription>(
      cantFail(RegisterDescription::Create(std::move(registers))));
}
} // namespace

TEST(RegisterPacketLayoutTest, ResolvesForwardAliasesWithoutCopyingMetadata) {
  auto high = Register("high", 20, 1);
  high.value_registers = {90};
  high.value_byte_offset = 1;
  auto full = Register("full", 90, 8);
  full.invalidate_registers = {20, 90};
  auto description = Description({high, Register("pc", 42, 8), full});
  auto packet = RegisterPacketLayout::Create(description, {20, 42, 90},
                                             lldb::eByteOrderLittle);
  ASSERT_THAT_EXPECTED(packet, Succeeded());
  EXPECT_EQ(packet->GetByteSize(), 16u);
  EXPECT_FALSE(packet->GetEntries()[0].primary);
  EXPECT_EQ(packet->GetEntries()[0].id, 20u);
  EXPECT_EQ(packet->GetEntries()[0].byte_offset, 9u);
  EXPECT_EQ(packet->GetEntries()[1].byte_offset, 0u);
  EXPECT_EQ(packet->GetEntries()[2].byte_offset, 8u);
  EXPECT_EQ(packet->GetWireNumber(90), 2u);
  EXPECT_EQ(packet->GetWireNumbers(description->Find(90)->invalidate_registers),
            (std::vector<uint32_t>{0}));
  // Projecting a relationship leaves its canonical IDs untouched.
  EXPECT_EQ(description->Find(20)->value_registers.front(), 90u);
  EXPECT_EQ(description->Find(20)->numbers[lldb::eRegisterKindLLDB], 500u);
  auto moved = std::move(*packet);
  auto copy = moved;
  description.reset();
  EXPECT_EQ(copy.GetDescription().Find(20)->name, "high");
  EXPECT_EQ(moved.GetWireNumber(90), 2u);
}

TEST(RegisterPacketLayoutTest, AliasPositionsUseTargetByteOrder) {
  auto byte = Register("byte", 9, 1);
  byte.value_registers = {5};
  byte.value_byte_offset = 1;
  auto description = Description({byte, Register("full", 5, 8)});
  auto little =
      RegisterPacketLayout::Create(description, {9, 5}, lldb::eByteOrderLittle);
  auto big =
      RegisterPacketLayout::Create(description, {9, 5}, lldb::eByteOrderBig);
  ASSERT_THAT_EXPECTED(little, Succeeded());
  ASSERT_THAT_EXPECTED(big, Succeeded());
  EXPECT_EQ(little->GetEntries()[0].byte_offset, 1u);
  EXPECT_EQ(big->GetEntries()[0].byte_offset, 6u);
}

TEST(RegisterPacketLayoutTest, RejectsCyclesAndOutOfBoundsAliases) {
  auto first = Register("a", 9, 8);
  auto second = Register("b", 5, 8);
  first.value_registers = {5};
  second.value_registers = {9};
  // Cycles are rejected before a description can reach packet placement.
  EXPECT_THAT_EXPECTED(RegisterDescription::Create({first, second}), Failed());
  second.value_registers.clear();
  first.value_byte_offset = 1;
  auto outside = Description({first, second});
  EXPECT_THAT_EXPECTED(
      RegisterPacketLayout::Create(outside, {9, 5}, lldb::eByteOrderLittle),
      Failed());
}

TEST(RegisterPacketLayoutTest,
     InvisibleParentsAndCompositesHaveOwnPacketSlots) {
  auto first = Register("a", 10, 4);
  first.value_registers = {99};
  auto combined = Register("both", 30, 8);
  combined.value_registers = {10, 20};
  auto description = Description(
      {first, Register("b", 20, 4), combined, Register("hidden", 99, 4)});
  auto packet = RegisterPacketLayout::Create(description, {10, 20, 30},
                                             lldb::eByteOrderLittle);
  ASSERT_THAT_EXPECTED(packet, Succeeded());
  EXPECT_TRUE(packet->GetEntries()[0].primary);
  EXPECT_TRUE(packet->GetEntries()[2].primary);
  EXPECT_EQ(packet->GetEntries()[2].byte_offset, 8u);
  EXPECT_EQ(packet->GetByteSize(), 16u);
  EXPECT_EQ(packet->GetWireNumber(99), std::nullopt);
  EXPECT_TRUE(packet->GetWireNumbers(first.value_registers).empty());
}

TEST(RegisterPacketLayoutTest, ValidatesVisibilityAndPreservesEmptyProjection) {
  auto description = Description({Register("value", 91, 8)});
  EXPECT_THAT_EXPECTED(RegisterPacketLayout::Create(description, {91, 91},
                                                    lldb::eByteOrderLittle),
                       Failed());
  EXPECT_THAT_EXPECTED(
      RegisterPacketLayout::Create(description, {3}, lldb::eByteOrderLittle),
      Failed());
  EXPECT_THAT_EXPECTED(
      RegisterPacketLayout::Create(description, {91}, lldb::eByteOrderInvalid),
      Failed());
  auto empty =
      RegisterPacketLayout::Create(description, {}, lldb::eByteOrderLittle);
  ASSERT_THAT_EXPECTED(empty, Succeeded());
  EXPECT_TRUE(empty->GetEntries().empty());
  EXPECT_EQ(empty->GetByteSize(), 0u);
}
