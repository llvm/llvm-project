//===-- AddressSpaceTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/AddressSpace.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

static std::string ToString(const llvm::json::Value &value) {
  return llvm::formatv("{0}", value).str();
}

TEST(AddressSpaceTest, RoundTrip) {
  AddressSpaceInfo info{"global", 1, /*is_thread_specific=*/false};
  llvm::Expected<AddressSpaceInfo> parsed = llvm::json::parse<AddressSpaceInfo>(
      ToString(toJSON(info)), "AddressSpaceInfo");
  ASSERT_THAT_EXPECTED(parsed, llvm::Succeeded());
  EXPECT_EQ(parsed->name, "global");
  EXPECT_EQ(parsed->value, 1u);
  EXPECT_FALSE(parsed->is_thread_specific);
}

TEST(AddressSpaceTest, ArrayRoundTrip) {
  std::vector<AddressSpaceInfo> spaces = {
      {"global", 1, false},
      {"local", 2, true},
      {"private", 3, true},
  };
  llvm::json::Array array;
  for (const AddressSpaceInfo &space : spaces)
    array.push_back(toJSON(space));

  llvm::Expected<std::vector<AddressSpaceInfo>> parsed =
      llvm::json::parse<std::vector<AddressSpaceInfo>>(
          ToString(llvm::json::Value(std::move(array))), "AddressSpaceInfo");
  ASSERT_THAT_EXPECTED(parsed, llvm::Succeeded());
  ASSERT_EQ(parsed->size(), 3u);
  EXPECT_EQ((*parsed)[1].name, "local");
  EXPECT_EQ((*parsed)[1].value, 2u);
  EXPECT_TRUE((*parsed)[1].is_thread_specific);
}

TEST(AddressSpaceTest, MissingFieldFails) {
  // "is_thread_specific" is required.
  llvm::Expected<AddressSpaceInfo> parsed = llvm::json::parse<AddressSpaceInfo>(
      R"({"name":"global","value":1})", "AddressSpaceInfo");
  EXPECT_THAT_EXPECTED(parsed, llvm::Failed());
}
