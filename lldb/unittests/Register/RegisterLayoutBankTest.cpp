//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterLayout.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <array>

using namespace lldb_private;
using namespace llvm;

TEST(RegisterLayoutBankTest, SelectsBanksForLogicalValuesAndWiderWrites) {
  RegisterDescriptor parent;
  parent.id = 90;
  parent.name = "parent";
  parent.byte_size = 8;
  auto alias = parent;
  alias.id = 7;
  alias.name = "alias";
  alias.byte_size = 4;
  alias.value_registers = {90};
  alias.zero_extend_parent = 90;
  auto constant = parent;
  constant.id = 200;
  constant.name = "zero";
  constant.constant_value.resize(8, 0);
  auto description = RegisterDescription::Create({alias, constant, parent});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout;
  ASSERT_THAT_ERROR(layout.AddBank(3, 8), Succeeded());
  ASSERT_THAT_ERROR(layout.AddBank(5, 8), Succeeded());
  RegisterLocation whole;
  whole.kind = RegisterLocation::Stored;
  whole.pieces = {{3, 0, 4, 0}, {5, 0, 4, 4}};
  ASSERT_THAT_ERROR(layout.Add(90, whole), Succeeded());
  RegisterLocation part;
  part.kind = RegisterLocation::Stored;
  part.pieces = {{3, 0, 4, 0}};
  ASSERT_THAT_ERROR(layout.Add(7, part), Succeeded());
  llvm::cantFail(
      layout.Finalize(std::make_shared<RegisterDescription>(*description),
                      lldb::eByteOrderLittle));

  auto read = layout.GetRequiredBanks(7);
  auto write = layout.GetRequiredBanks(7, true);
  auto zero = layout.GetRequiredBanks(200);
  ASSERT_THAT_EXPECTED(read, Succeeded());
  ASSERT_THAT_EXPECTED(write, Succeeded());
  ASSERT_THAT_EXPECTED(zero, Succeeded());
  EXPECT_EQ(*read, (std::vector<RegisterBankID>{3}));
  EXPECT_EQ(*write, (std::vector<RegisterBankID>{3, 5}));
  EXPECT_TRUE(zero->empty());
  EXPECT_THAT_EXPECTED(layout.GetRequiredBanks(200, true), Failed());
  EXPECT_THAT_EXPECTED(layout.GetRequiredBanks(99), Failed());
}

namespace {

llvm::Error ReadEncodedByte(llvm::ArrayRef<RegisterData> banks, lldb::ByteOrder,
                            llvm::MutableArrayRef<uint8_t> value) {
  auto data =
      llvm::find_if(banks, [](const auto &bank) { return bank.bank == 1; });
  auto key =
      llvm::find_if(banks, [](const auto &bank) { return bank.bank == 2; });
  value[0] = data->bytes[1] ^ key->bytes[0];

  return llvm::Error::success();
}

llvm::Error WriteEncodedByte(llvm::ArrayRef<uint8_t> value, lldb::ByteOrder,
                             llvm::ArrayRef<MutableRegisterData> banks) {
  auto data =
      llvm::find_if(banks, [](const auto &bank) { return bank.bank == 1; });
  auto key =
      llvm::find_if(banks, [](const auto &bank) { return bank.bank == 2; });
  data->bytes[1] = value[0] ^ key->bytes[0];

  return llvm::Error::success();
}

const RegisterConversion EncodedByte{1, ReadEncodedByte, WriteEncodedByte};
const RegisterConversion OtherEncodedByte{1, ReadEncodedByte, WriteEncodedByte};

} // namespace

TEST(RegisterLayoutBankTest,
     ConversionsDeclareDependenciesAndPreserveOtherBytes) {
  RegisterDescriptor reg;
  reg.id = 9;
  reg.name = "encoded";
  reg.byte_size = 1;
  auto description = std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create({reg})));
  RegisterLayout draft;
  ASSERT_THAT_ERROR(draft.AddBank(1, 3), Succeeded());
  ASSERT_THAT_ERROR(draft.AddBank(2, 1), Succeeded());
  RegisterLocation location;
  location.kind = RegisterLocation::Stored;
  location.conversion = &EncodedByte;
  location.inputs = {{1, 1, 1}, {2, 0, 1}};
  ASSERT_THAT_ERROR(draft.Add(9, location), Succeeded());
  RegisterLayout layout = draft;
  llvm::cantFail(layout.Finalize(description, lldb::eByteOrderLittle));
  auto required = layout.GetRequiredBanks(9, true);
  ASSERT_THAT_EXPECTED(required, Succeeded());
  EXPECT_EQ(*required, (std::vector<RegisterBankID>{1, 2}));
  std::array<uint8_t, 3> bytes{0xaa, 0x12, 0xbb};
  std::array<uint8_t, 1> key{0xff};
  const RegisterData read[] = {{1, bytes}, {2, key}};

  auto value = layout.Read(9, read);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt8(), 0xed);
  const RegisterData reversed[] = {{2, key}, {1, bytes}};
  auto reordered = layout.Read(9, reversed);
  ASSERT_THAT_EXPECTED(reordered, Succeeded());
  EXPECT_EQ(reordered->GetAsUInt8(), 0xed);

  const MutableRegisterData write[] = {{1, bytes}, {2, key}};
  ASSERT_THAT_ERROR(layout.Write(9, RegisterValue(uint8_t(0x34)), write),
                    Succeeded());
  EXPECT_EQ(bytes, (std::array<uint8_t, 3>{0xaa, 0xcb, 0xbb}));
  EXPECT_EQ(key[0], 0xff);

  EXPECT_THAT_EXPECTED(layout.Read(9, llvm::ArrayRef<RegisterData>(read, 1)),
                       Failed());
  const MutableRegisterData truncated[] = {
      {1, llvm::MutableArrayRef<uint8_t>(bytes.data(), 1)}, {2, key}};
  EXPECT_THAT_ERROR(layout.Write(9, RegisterValue(uint8_t(0x56)), truncated),
                    Failed());
  EXPECT_EQ(bytes[1], 0xcb);

  RegisterLayout same = draft;
  llvm::cantFail(same.Finalize(description, lldb::eByteOrderLittle));
  EXPECT_TRUE(layout.HasSameStorage(same));
  location.conversion = &OtherEncodedByte;
  RegisterLayout other;
  ASSERT_THAT_ERROR(other.AddBank(1, 3), Succeeded());
  ASSERT_THAT_ERROR(other.AddBank(2, 1), Succeeded());
  ASSERT_THAT_ERROR(other.Add(9, location), Succeeded());
  llvm::cantFail(other.Finalize(description, lldb::eByteOrderLittle));
  EXPECT_FALSE(layout.HasSameStorage(other));
}

TEST(RegisterLayoutBankTest, RejectsInvalidConversionDefinitions) {
  RegisterDescriptor reg;
  reg.id = 9;
  reg.name = "encoded";
  reg.byte_size = 1;
  auto description = std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create({reg})));
  RegisterLocation valid;
  valid.kind = RegisterLocation::Stored;
  valid.conversion = &EncodedByte;
  valid.inputs = {{1, 1, 1}, {2, 0, 1}};
  auto check = [&](RegisterLocation location) {
    RegisterLayout draft;
    EXPECT_THAT_ERROR(draft.AddBank(1, 3), Succeeded());
    EXPECT_THAT_ERROR(draft.AddBank(2, 1), Succeeded());
    EXPECT_THAT_ERROR(draft.Add(9, std::move(location)), Succeeded());
    EXPECT_THAT_ERROR(draft.Finalize(description, lldb::eByteOrderLittle),
                      Failed());
  };

  auto invalid = valid;
  invalid.inputs[0].byte_offset = 3;
  check(invalid);
  invalid = valid;
  invalid.inputs[0].bank = 99;
  check(invalid);
  invalid = valid;
  invalid.inputs[0].value_offset = 1;
  check(invalid);
  invalid = valid;
  invalid.pieces = {{1, 0, 1}};
  check(invalid);
  invalid = valid;
  invalid.zero_fill = true;
  check(invalid);
  invalid = valid;
  invalid.inputs.clear();
  check(invalid);
  invalid = valid;
  invalid.conversion = nullptr;
  check(invalid);

  static const RegisterConversion wrong_size{2, ReadEncodedByte,
                                             WriteEncodedByte};
  invalid = valid;
  invalid.conversion = &wrong_size;
  check(invalid);
  static const RegisterConversion no_read{1, nullptr, WriteEncodedByte};
  invalid.conversion = &no_read;
  check(invalid);
  static const RegisterConversion no_write{1, ReadEncodedByte, nullptr};
  invalid.conversion = &no_write;
  check(invalid);
}

TEST(RegisterLayoutBankTest, DerivesAliasChainsAcrossPiecesInEitherByteOrder) {
  RegisterDescriptor parent;
  parent.id = 90;
  parent.name = "whole";
  parent.byte_size = 8;
  auto alias = parent;
  alias.id = 7;
  alias.name = "part";
  alias.byte_size = 4;
  alias.value_registers = {90};
  alias.value_byte_offset = 2;
  auto high = alias;
  high.id = 3;
  high.name = "high";
  high.byte_size = 1;
  high.value_registers = {7};
  high.value_byte_offset = 3;
  auto description = std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create({high, alias, parent})));

  for (auto order : {lldb::eByteOrderLittle, lldb::eByteOrderBig}) {
    RegisterLayout draft;
    ASSERT_THAT_ERROR(draft.AddBank(1, 4), Succeeded());
    ASSERT_THAT_ERROR(draft.AddBank(2, 4), Succeeded());
    RegisterLocation whole;
    whole.kind = RegisterLocation::Stored;
    whole.pieces = {{1, 0, 4, 0}, {2, 0, 4, 4}};
    ASSERT_THAT_ERROR(draft.Add(90, whole), Succeeded());
    RegisterLayout layout = draft;
    llvm::cantFail(layout.Finalize(description, order));
    ASSERT_EQ(layout.Find(7)->pieces.size(), 2u);
    std::array<uint8_t, 4> first{1, 2, 3, 4}, second{5, 6, 7, 8};
    const RegisterData read[] = {{1, first}, {2, second}};
    auto value = layout.Read(7, read);
    auto top = layout.Read(3, read);
    ASSERT_THAT_EXPECTED(value, Succeeded());
    ASSERT_THAT_EXPECTED(top, Succeeded());
    EXPECT_EQ(value->GetAsUInt32(),
              order == lldb::eByteOrderLittle ? 0x06050403u : 0x03040506u);
    EXPECT_EQ(top->GetAsUInt8(), order == lldb::eByteOrderLittle ? 6u : 3u);

    const MutableRegisterData write[] = {{1, first}, {2, second}};
    ASSERT_THAT_ERROR(
        layout.Write(7, RegisterValue(uint32_t(0xaabbccdd)), write),
        Succeeded());
    EXPECT_EQ(first[0], 1u);
    EXPECT_EQ(first[1], 2u);
    EXPECT_EQ(second[2], 7u);
    EXPECT_EQ(second[3], 8u);

    RegisterLocation explicit_location;
    explicit_location.kind = RegisterLocation::Unavailable;
    ASSERT_THAT_ERROR(draft.Add(7, explicit_location), Succeeded());
    RegisterLayout unavailable = draft;
    llvm::cantFail(unavailable.Finalize(description, order));
    EXPECT_THAT_EXPECTED(unavailable.Read(7, read), Failed());
    EXPECT_EQ(unavailable.Find(3), nullptr);
  }
}
