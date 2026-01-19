//===-- LLDBUtilsTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBUtils.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBStructuredData.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;

TEST(LLDBUtilsTest, GetStringValue) {
  // Create an SBStructuredData object from JSON.
  const char *json_data = R"("test_string")";
  SBStructuredData data;
  SBError error = data.SetFromJSON(json_data);

  // Ensure the JSON was parsed successfully.
  ASSERT_TRUE(error.Success());
  ASSERT_TRUE(data.IsValid());

  // Call GetStringValue and verify the result.
  std::string result = GetStringValue(data);
  EXPECT_EQ(result, "test_string");

  // Test with invalid SBStructuredData.
  SBStructuredData invalid_data;
  result = GetStringValue(invalid_data);
  EXPECT_EQ(result, "");

  // Test with empty JSON.
  const char *empty_json = R"("")";
  SBStructuredData empty_data;
  error = empty_data.SetFromJSON(empty_json);

  ASSERT_TRUE(error.Success());
  ASSERT_TRUE(empty_data.IsValid());

  result = GetStringValue(empty_data);
  EXPECT_EQ(result, "");
}

TEST(LLDBUtilsTest, ToError) {
  // Test with a successful SBError.
  SBError success_error;
  ASSERT_TRUE(success_error.Success());
  llvm::Error llvm_error = ToError(success_error);
  EXPECT_FALSE(llvm_error);

  // Test with a failing SBError.
  SBError fail_error;
  fail_error.SetErrorString("Test error message");
  ASSERT_TRUE(fail_error.Fail());
  llvm_error = ToError(fail_error);

  std::string error_message = toString(std::move(llvm_error));
  EXPECT_EQ(error_message, "Test error message");
}

TEST(LLDBUtilsTest, UTF16Codeunits) {
  using Expect = std::optional<size_t>;

  EXPECT_EQ(UTF16CodeunitToBytes("a", 0), Expect{0});
  EXPECT_EQ(UTF16CodeunitToBytes("some word", 4), Expect{4});
  EXPECT_EQ(UTF16CodeunitToBytes("fake", 4), Expect{4});
  EXPECT_EQ(UTF16CodeunitToBytes("ƒake", 4), Expect{5});
  EXPECT_EQ(UTF16CodeunitToBytes("b", 1), Expect{1});
  EXPECT_EQ(UTF16CodeunitToBytes("💩", 0), Expect{0});
  EXPECT_EQ(UTF16CodeunitToBytes("ƒ", 1), Expect{2});
  EXPECT_EQ(UTF16CodeunitToBytes("💩ƒ", 2), Expect{4});
  EXPECT_EQ(UTF16CodeunitToBytes("√", 2), Expect{3});
  EXPECT_EQ(UTF16CodeunitToBytes("√ƒ", 4), Expect{5});
  EXPECT_EQ(UTF16CodeunitToBytes("√💩", 4), Expect{7});
  EXPECT_EQ(UTF16CodeunitToBytes("√", 1), Expect{3});

  // Index
  EXPECT_EQ(UTF16CodeunitToBytes("ƒake extra", 4), Expect{5});
  EXPECT_EQ(UTF16CodeunitToBytes("3çç ", 3), Expect{5});
  EXPECT_EQ(UTF16CodeunitToBytes("20𒂷 ", 3), std::nullopt);

  // Failures
  EXPECT_EQ(UTF16CodeunitToBytes("💩ƒ", 1), std::nullopt);
  EXPECT_NE(UTF16CodeunitToBytes("20𒂷 ", 3), Expect{5});
  EXPECT_NE(UTF16CodeunitToBytes("w💩ƒ", 2), Expect{6});
}
