//===- StringExtrasTest.cpp - Tests for utility methods in StringExtras.h -===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/StringExtras.h"
#include "gtest/gtest.h"

using namespace mlir;

static void testConvertToSnakeCase(llvm::StringRef input,
                                   llvm::StringRef expected) {
  EXPECT_EQ(convertToSnakeCase(input), expected.str());
}

TEST(StringExtras, ConvertToSnakeCase) {
  testConvertToSnakeCase("OpName", "op_name");
  testConvertToSnakeCase("opName", "op_name");
  testConvertToSnakeCase("_OpName", "_op_name");
  testConvertToSnakeCase("Op_Name", "op_name");
  testConvertToSnakeCase("", "");
  testConvertToSnakeCase("A", "a");
  testConvertToSnakeCase("_", "_");
  testConvertToSnakeCase("a", "a");
  testConvertToSnakeCase("op_name", "op_name");
  testConvertToSnakeCase("_op_name", "_op_name");
  testConvertToSnakeCase("__op_name", "__op_name");
  testConvertToSnakeCase("op__name", "op__name");
}

template <bool capitalizeFirst>
static void testConvertToCamelCase(llvm::StringRef input,
                                   llvm::StringRef expected) {
  EXPECT_EQ(convertToCamelCase(input, capitalizeFirst), expected.str());
}

TEST(StringExtras, ConvertToCamelCase) {
  testConvertToCamelCase<false>("op_name", "opName");
  testConvertToCamelCase<false>("_op_name", "_opName");
  testConvertToCamelCase<false>("__op_name", "_OpName");
  testConvertToCamelCase<false>("op__name", "op_Name");
  testConvertToCamelCase<false>("", "");
  testConvertToCamelCase<false>("A", "A");
  testConvertToCamelCase<false>("_", "_");
  testConvertToCamelCase<false>("a", "a");
  testConvertToCamelCase<false>("OpName", "OpName");
  testConvertToCamelCase<false>("opName", "opName");
  testConvertToCamelCase<false>("_OpName", "_OpName");
  testConvertToCamelCase<false>("Op_Name", "Op_Name");
  testConvertToCamelCase<true>("op_name", "OpName");
  testConvertToCamelCase<true>("_op_name", "_opName");
  testConvertToCamelCase<true>("__op_name", "_OpName");
  testConvertToCamelCase<true>("op__name", "Op_Name");
  testConvertToCamelCase<true>("", "");
  testConvertToCamelCase<true>("A", "A");
  testConvertToCamelCase<true>("_", "_");
  testConvertToCamelCase<true>("a", "A");
  testConvertToCamelCase<true>("OpName", "OpName");
  testConvertToCamelCase<true>("_OpName", "_OpName");
  testConvertToCamelCase<true>("Op_Name", "Op_Name");
  testConvertToCamelCase<true>("opName", "OpName");
}
