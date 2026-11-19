//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterLayout.h"
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

TEST(RegisterLayoutTest, ConstructorStateSurvivesBaseValueCopy) {
  class OSLayout final : public RegisterLayout {
  public:
    explicit OSLayout(std::shared_ptr<const RegisterDescription> description) {
      llvm::cantFail(AddBank(7, 16));
      llvm::cantFail(Add(42, Stored(7, 8)));
      llvm::cantFail(Finalize(std::move(description), lldb::eByteOrderBig));
    }
  };

  auto description = std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create({Register(42, "pc")})));
  const OSLayout os_layout(description);
  RegisterLayout layout = os_layout;
  EXPECT_EQ(layout.GetDescription(), description);
  EXPECT_EQ(layout.GetByteOrder(), lldb::eByteOrderBig);
  EXPECT_EQ(layout.GetBankSize(7), 16u);
  ASSERT_NE(layout.Find(42), nullptr);
  EXPECT_EQ(layout.Find(42)->pieces[0].byte_offset, 8u);
}

TEST(RegisterLayoutTest, SameDescriptionDifferentLayoutsAndByteOrders) {
  auto description = RegisterDescription::Create({Register(42, "pc")});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout little;
  RegisterLayout big;
  ASSERT_THAT_ERROR(little.AddBank(7, 16), Succeeded());
  ASSERT_THAT_ERROR(big.AddBank(7, 16), Succeeded());
  ASSERT_THAT_ERROR(little.Add(42, Stored(7, 0)), Succeeded());
  ASSERT_THAT_ERROR(big.Add(42, Stored(7, 8)), Succeeded());
  std::array<uint8_t, 16> a{}, b{};
  a.fill(0xcc);
  b.fill(0xdd);
  MutableRegisterData a_data{7, a}, b_data{7, b};
  RegisterValue value(uint64_t(0x0102030405060708));
  llvm::cantFail(
      little.Finalize(std::make_shared<RegisterDescription>(*description),
                      lldb::eByteOrderLittle));

  ASSERT_THAT_ERROR(little.Write(42, value, a_data), Succeeded());
  llvm::cantFail(
      big.Finalize(std::make_shared<RegisterDescription>(*description),
                   lldb::eByteOrderBig));

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

TEST(RegisterLayoutTest, CompositeConstantAndUnavailableRegisters) {
  auto description = RegisterDescription::Create(
      {Register(1, "split"), Register(2, "zero"), Register(3, "missing")});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout;
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
  llvm::cantFail(
      layout.Finalize(std::make_shared<RegisterDescription>(*description),
                      lldb::eByteOrderLittle));

  auto result = layout.Read(1, banks);
  ASSERT_THAT_EXPECTED(result, Succeeded());
  EXPECT_EQ(result->GetAsUInt64(), 0x0807060504030201ULL);
  auto constant = layout.Read(2, {});
  ASSERT_THAT_EXPECTED(constant, Succeeded());
  EXPECT_EQ(constant->GetAsUInt64(), 0u);
  EXPECT_THAT_EXPECTED(layout.Read(3, banks), Failed());
}

TEST(RegisterLayoutTest, RejectsTruncationWithoutPartialWrites) {
  auto description = RegisterDescription::Create({Register(1, "split")});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout;
  ASSERT_THAT_ERROR(layout.AddBank(0, 4), Succeeded());
  ASSERT_THAT_ERROR(layout.AddBank(1, 4), Succeeded());
  RegisterLocation split;
  split.kind = RegisterLocation::Stored;
  split.pieces = {{0, 0, 4, 0}, {1, 0, 4, 4}};
  ASSERT_THAT_ERROR(layout.Add(1, split), Succeeded());
  std::array<uint8_t, 4> first{9, 9, 9, 9};
  std::array<uint8_t, 3> truncated{8, 8, 8};
  const MutableRegisterData banks[] = {{0, first}, {1, truncated}};
  llvm::cantFail(
      layout.Finalize(std::make_shared<RegisterDescription>(*description),
                      lldb::eByteOrderLittle));

  EXPECT_THAT_ERROR(layout.Write(1, RegisterValue(uint64_t(0)), banks),
                    Failed());
  EXPECT_EQ(first, (std::array<uint8_t, 4>{9, 9, 9, 9}));
}

TEST(RegisterLayoutTest, RejectsOverflowAndOverlappingPieces) {
  auto description = RegisterDescription::Create({Register(1, "r")});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout overflow;
  ASSERT_THAT_ERROR(overflow.AddBank(0, 8), Succeeded());
  ASSERT_THAT_ERROR(overflow.Add(1, Stored(0, UINT64_MAX)), Succeeded());
  EXPECT_THAT_ERROR(
      overflow.Finalize(std::make_shared<RegisterDescription>(*description),
                        lldb::eByteOrderLittle),
      Failed());
  RegisterLayout overlap;
  ASSERT_THAT_ERROR(overlap.AddBank(0, 8), Succeeded());
  RegisterLocation pieces;
  pieces.kind = RegisterLocation::Stored;
  pieces.pieces = {{0, 0, 4, 0}, {0, 4, 4, 0}};
  ASSERT_THAT_ERROR(overlap.Add(1, pieces), Succeeded());
  EXPECT_THAT_ERROR(
      overlap.Finalize(std::make_shared<RegisterDescription>(*description),
                       lldb::eByteOrderLittle),
      Failed());
}

TEST(RegisterLayoutTest, ArchitectureConstantDoesNotRequireAStorageBank) {
  auto zero = Register(51, "zero", 4);
  zero.constant_value = {0, 0, 0, 0};
  auto description = RegisterDescription::Create({zero});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout;
  llvm::cantFail(
      layout.Finalize(std::make_shared<RegisterDescription>(*description),
                      lldb::eByteOrderBig));

  auto value = layout.Read(51, {});
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt32(), 0u);
  EXPECT_THAT_ERROR(layout.Write(51, RegisterValue(uint32_t(1)), {}), Failed());
}

TEST(RegisterLayoutTest, ExplicitlyZeroFilledLanesAreReadAndValidatedOnWrite) {
  auto description = RegisterDescription::Create({Register(1, "vector", 16)});
  ASSERT_THAT_EXPECTED(description, Succeeded());
  RegisterLayout layout;
  ASSERT_THAT_ERROR(layout.AddBank(3, 8), Succeeded());
  auto location = Stored(3, 0, 8);
  location.zero_fill = true;
  ASSERT_THAT_ERROR(layout.Add(1, location), Succeeded());
  std::array<uint8_t, 8> bytes{};
  bytes[0] = 17;
  RegisterData input{3, bytes};
  llvm::cantFail(
      layout.Finalize(std::make_shared<RegisterDescription>(*description),
                      lldb::eByteOrderLittle));

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

TEST(RegisterLayoutTest, ComparesStorageIndependentlyOfSchemaOwnership) {
  auto make = [](uint64_t offset) {
    auto description = std::make_shared<RegisterDescription>(
        cantFail(RegisterDescription::Create({Register(42, "value")})));
    RegisterLayout draft;
    cantFail(draft.AddBank(3, 16));
    cantFail(draft.Add(42, Stored(3, offset)));
    llvm::cantFail(draft.Finalize(description, lldb::eByteOrderLittle));
    return draft;
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
