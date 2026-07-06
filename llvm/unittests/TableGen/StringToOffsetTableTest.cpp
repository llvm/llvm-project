//===- StringToOffsetTableTest.cpp - StringToOffsetTable tests -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(StringToOffsetTableTest, OctalEscapeFollowedByDigit) {
  // Test that when an octal escape sequence (\000) is followed by a digit,
  // the output inserts a string literal boundary to prevent MSVC C4125 warning.
  // This can happen when numeric feature names create sequences like "\000200".

  StringToOffsetTable Table;

  // Add strings that create octal escape followed by digit pattern.
  // '\0' will be emitted as \000, and "200" starts with a digit.
  std::string TestStr;
  TestStr += '\0';  // Will become \000
  TestStr += "200"; // Starts with digit

  Table.GetOrAddStringOffset(TestStr);

  std::string Output;
  raw_string_ostream OS(Output);
  Table.EmitString(OS);
  OS.flush();

  // The output should contain a string boundary after the octal escape:
  // Should be: "    "\000" "200""
  // Not:       "    "\000200""
  // This prevents line breaks from creating "\0002" + "00..." which triggers
  // MSVC warning C4125: decimal digit terminates octal escape sequence.

  EXPECT_TRUE(Output.find("\\000\" \"2") != std::string::npos)
      << "Expected string boundary after octal escape before digit, got: "
      << Output;
}

TEST(StringToOffsetTableTest, OctalEscapeNotFollowedByDigit) {
  // Test that octal escape NOT followed by digit doesn't insert boundary.

  StringToOffsetTable Table;

  std::string TestStr;
  TestStr += '\0';  // Will become \000
  TestStr += "abc"; // Starts with non-digit

  Table.GetOrAddStringOffset(TestStr);

  std::string Output;
  raw_string_ostream OS(Output);
  Table.EmitString(OS);
  OS.flush();

  // Should NOT insert a boundary when not followed by digit:
  // Should be: "    "\000abc""

  EXPECT_TRUE(Output.find("\\000abc") != std::string::npos)
      << "Expected no string boundary after octal escape when not followed by digit, got: "
      << Output;
  EXPECT_TRUE(Output.find("\\000\" \"a") == std::string::npos)
      << "Unexpected string boundary found: " << Output;
}

TEST(StringToOffsetTableTest, MultipleOctalDigitSequences) {
  // Test multiple sequences of octal escapes followed by digits.

  StringToOffsetTable Table;

  std::string TestStr;
  TestStr += '\0';
  TestStr += "100";
  TestStr += '\0';
  TestStr += "200";
  TestStr += '\0';
  TestStr += "300";

  Table.GetOrAddStringOffset(TestStr);

  std::string Output;
  raw_string_ostream OS(Output);
  Table.EmitString(OS);
  OS.flush();

  // Each octal escape followed by a digit should have a boundary.
  EXPECT_TRUE(Output.find("\\000\" \"1") != std::string::npos)
      << "Expected first boundary, got: " << Output;
  EXPECT_TRUE(Output.find("\\000\" \"2") != std::string::npos)
      << "Expected second boundary, got: " << Output;
  EXPECT_TRUE(Output.find("\\000\" \"3") != std::string::npos)
      << "Expected third boundary, got: " << Output;
}

TEST(StringToOffsetTableTest, PlainString) {
  // Test that plain strings without null characters work normally.

  StringToOffsetTable Table;
  Table.GetOrAddStringOffset("hello");

  std::string Output;
  raw_string_ostream OS(Output);
  Table.EmitString(OS);
  OS.flush();

  EXPECT_TRUE(Output.find("hello") != std::string::npos)
      << "Expected plain string, got: " << Output;
}
