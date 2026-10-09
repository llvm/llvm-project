//===----------------------------------------------------------------------===//
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
  AddressSpaceInfo info{"local", 2, /*is_thread_specific=*/true};
  llvm::Expected<AddressSpaceInfo> parsed = llvm::json::parse<AddressSpaceInfo>(
      ToString(toJSON(info)), "AddressSpaceInfo");
  ASSERT_THAT_EXPECTED(parsed, llvm::Succeeded());
  EXPECT_EQ(parsed->name, "local");
  EXPECT_EQ(parsed->space_id, 2u);
  EXPECT_TRUE(parsed->is_thread_specific);
}

TEST(AddressSpaceTest, ArrayRoundTrip) {
  std::vector<AddressSpaceInfo> spaces = {
      {"global", 1, false},
      {"local", 2, true},
      {"private", 3, false},
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
  EXPECT_EQ((*parsed)[1].space_id, 2u);
  EXPECT_TRUE((*parsed)[1].is_thread_specific);
  EXPECT_FALSE((*parsed)[0].is_thread_specific);
}

TEST(AddressSpaceTest, MissingFieldFails) {
  // "space_id" is required.
  llvm::Expected<AddressSpaceInfo> parsed = llvm::json::parse<AddressSpaceInfo>(
      R"({"name":"global"})", "AddressSpaceInfo");
  EXPECT_THAT_EXPECTED(parsed, llvm::Failed());
}

TEST(AddressSpaceTest, ApertureRoundTrip) {
  AddressSpaceInfo info{"local",
                        2,
                        /*is_thread_specific=*/false,
                        /*dwarf_address_class=*/3U,
                        /*flat_base=*/0x100000ULL,
                        /*flat_size=*/0x4000ULL,
                        /*pointer_unit=*/4ULL};
  llvm::Expected<AddressSpaceInfo> parsed = llvm::json::parse<AddressSpaceInfo>(
      ToString(toJSON(info)), "AddressSpaceInfo");
  ASSERT_THAT_EXPECTED(parsed, llvm::Succeeded());
  EXPECT_EQ(parsed->name, "local");
  EXPECT_EQ(parsed->space_id, 2u);
  EXPECT_FALSE(parsed->is_thread_specific);
  EXPECT_EQ(parsed->dwarf_address_class, std::optional<uint32_t>(3U));
  EXPECT_EQ(parsed->flat_base, std::optional<lldb::addr_t>(0x100000ULL));
  EXPECT_EQ(parsed->flat_size, 0x4000ULL);
  EXPECT_EQ(parsed->pointer_unit, 4ULL);

  // Default AddressSpaceInfo omits optional aperture fields.
  AddressSpaceInfo plain{"global", 1, /*is_thread_specific=*/false};
  llvm::Expected<AddressSpaceInfo> parsed_plain =
      llvm::json::parse<AddressSpaceInfo>(ToString(toJSON(plain)),
                                          "AddressSpaceInfo");
  ASSERT_THAT_EXPECTED(parsed_plain, llvm::Succeeded());
  EXPECT_EQ(parsed_plain->dwarf_address_class, std::nullopt);
  EXPECT_EQ(parsed_plain->flat_base, std::nullopt);
  EXPECT_EQ(parsed_plain->flat_size, 0ULL);
  EXPECT_EQ(parsed_plain->pointer_unit, 1ULL);
}
