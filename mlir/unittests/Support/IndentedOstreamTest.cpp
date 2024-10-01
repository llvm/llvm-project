//===- IndentedOstreamTest.cpp - Indented raw ostream Tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/IndentedOstream.h"
#include "gmock/gmock.h"

using namespace mlir;
using ::testing::StrEq;

TEST(FormatTest, SingleLine) {
  std::string str;
  llvm::raw_string_ostream os(str);
  raw_indented_ostream ros(os);
  ros << 10;
  EXPECT_THAT(str, StrEq("10"));
}

TEST(FormatTest, SimpleMultiLine) {
  std::string str;
  llvm::raw_string_ostream os(str);
  raw_indented_ostream ros(os);
  ros << "a";
  ros << "b";
  ros << "\n";
  ros << "c";
  ros << "\n";
  EXPECT_THAT(str, StrEq("ab\nc\n"));
}

TEST(FormatTest, SimpleMultiLineIndent) {
  std::string str;
  llvm::raw_string_ostream os(str);
  raw_indented_ostream ros(os);
  ros.indent(2) << "a";
  ros.indent(4) << "b";
  ros << "\n";
  ros << "c";
  ros << "\n";
  EXPECT_THAT(str, StrEq("  a    b\n    c\n"));
}

TEST(FormatTest, SingleRegion) {
  std::string str;
  llvm::raw_string_ostream os(str);
  raw_indented_ostream ros(os);
  ros << "before\n";
  {
    raw_indented_ostream::DelimitedScope scope(ros);
    ros << "inside " << 10;
    ros << "\n   two\n";
    {
      raw_indented_ostream::DelimitedScope scope(ros, "{\n", "\n}\n");
      ros << "inner inner";
    }
  }
  ros << "after";
  const auto *expected =
      R"(before
  inside 10
     two
  {
    inner inner
  }
after)";
  EXPECT_THAT(str, StrEq(expected));

  // Repeat the above with inline form.
  str.clear();
  ros << "before\n";
  ros.scope().os << "inside " << 10 << "\n   two\n";
  ros.scope().os.scope("{\n", "\n}\n").os << "inner inner";
  ros << "after";
  EXPECT_THAT(os.str(), StrEq(expected));
}

TEST(FormatTest, Reindent) {
  std::string str;
  llvm::raw_string_ostream os(str);
  raw_indented_ostream ros(os);

  // String to print with some additional empty lines at the start and lines
  // with just spaces.
  const auto *desc = R"(
       
       
         First line
                 second line
                 
                 
  )";
  ros.printReindented(desc);
  const auto *expected =
      R"(First line
        second line


)";
  EXPECT_THAT(str, StrEq(expected));
}

TEST(FormatTest, ReindentLineEndings) {
  std::string str;
  llvm::raw_string_ostream os(str);
  raw_indented_ostream ros(os);

  // Similar string as the previous test, but with \r\n (Windows style) line
  // breaks. Note that C++'s internal string representation uses \n, so just
  // running the previous test as-is on Windows is not sufficient.
  const auto *desc =
      "\r\n\r\n\r\n         First line\r\n                 second line";
  ros.printReindented(desc);
  const auto *expected = "First line\r\n        second line";
  EXPECT_THAT(str, StrEq(expected));
}
