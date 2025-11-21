//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskKeyValueDB.h"
#include "CASTestConfig.h"
#include "OnDiskCommonUtils.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;
using namespace llvm::unittest::cas;

TEST_F(OnDiskCASTest, OnDiskKeyValueDBTest) {
  unittest::TempDir Temp("ondiskkv", /*Unique=*/true);
  std::unique_ptr<OnDiskKeyValueDB> DB;
  ASSERT_THAT_ERROR(OnDiskKeyValueDB::open(Temp.path(), "blake3",
                                           sizeof(HashType), "test",
                                           sizeof(ValueType))
                        .moveInto(DB),
                    Succeeded());
  {
    std::optional<ArrayRef<char>> Val;
    ASSERT_THAT_ERROR(DB->get(digest("hello")).moveInto(Val), Succeeded());
    EXPECT_FALSE(Val.has_value());
  }

  ValueType ValW = valueFromString("world");
  std::optional<ArrayRef<char>> Val;
  ASSERT_THAT_ERROR(DB->put(digest("hello"), ValW).moveInto(Val), Succeeded());
  EXPECT_EQ(*Val, ArrayRef(ValW));
  ASSERT_THAT_ERROR(
      DB->put(digest("hello"), valueFromString("other")).moveInto(Val),
      Succeeded());
  EXPECT_EQ(*Val, ArrayRef(ValW));

  {
    std::optional<ArrayRef<char>> Val;
    ASSERT_THAT_ERROR(DB->get(digest("hello")).moveInto(Val), Succeeded());
    EXPECT_TRUE(Val.has_value());
    EXPECT_EQ(*Val, ArrayRef(ValW));
  }

  // Validate
  {
    auto ValidateFunc = [](FileOffset Offset, ArrayRef<char> Data) -> Error {
      EXPECT_EQ(Data.size(), sizeof(ValueType));
      return Error::success();
    };
    ASSERT_THAT_ERROR(DB->validate(ValidateFunc), Succeeded());
  }

  // Size
  {
    size_t InitSize = DB->getStorageSize();
    unsigned InitPrecent = DB->getHardStorageLimitUtilization();

    // Insert a lot of entries.
    for (unsigned I = 0; I < 1024 * 100; ++I) {
      std::string Index = Twine(I).str();
      std::optional<ArrayRef<char>> Val;
      ASSERT_THAT_ERROR(
          DB->put(digest(Index), valueFromString(Index)).moveInto(Val),
          Succeeded());
    }

    EXPECT_GT(DB->getStorageSize(), InitSize);
    EXPECT_GT(DB->getHardStorageLimitUtilization(), InitPrecent);
  }
}
