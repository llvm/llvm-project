//===- llvm/unittest/OutputBufferTest.cpp - OutputStream unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/MicrosoftDemangleNodes.h"
#include "llvm/Demangle/Utility.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;
using llvm::itanium_demangle::OutputBuffer;

static std::string toString(OutputBuffer &OB) {
  return {OB.getBuffer(), OB.getCurrentPosition()};
}

template <typename T> static std::string printToString(const T &Value) {
  OutputBuffer OB;
  OB << Value;
  std::string s = toString(OB);
  std::free(OB.getBuffer());
  return s;
}

TEST(OutputBufferTest, Format) {
  EXPECT_EQ("0", printToString(0));
  EXPECT_EQ("1", printToString(1));
  EXPECT_EQ("-1", printToString(-1));
  EXPECT_EQ("-90", printToString(-90));
  EXPECT_EQ("109", printToString(109));
  EXPECT_EQ("400", printToString(400));

  EXPECT_EQ("a", printToString('a'));
  EXPECT_EQ("?", printToString('?'));

  EXPECT_EQ("abc", printToString("abc"));
}

TEST(OutputBufferTest, Insert) {
  OutputBuffer OB;

  OB.insert(0, "", 0);
  EXPECT_EQ("", toString(OB));

  OB.insert(0, "abcd", 4);
  EXPECT_EQ("abcd", toString(OB));

  OB.insert(0, "x", 1);
  EXPECT_EQ("xabcd", toString(OB));

  OB.insert(5, "y", 1);
  EXPECT_EQ("xabcdy", toString(OB));

  OB.insert(3, "defghi", 6);
  EXPECT_EQ("xabdefghicdy", toString(OB));

  std::free(OB.getBuffer());
}

TEST(OutputBufferTest, Prepend) {
  OutputBuffer OB;

  OB.prepend("n");
  EXPECT_EQ("n", toString(OB));

  OB << "abc";
  OB.prepend("def");
  EXPECT_EQ("defnabc", toString(OB));

  OB.setCurrentPosition(3);

  OB.prepend("abc");
  EXPECT_EQ("abcdef", toString(OB));

  std::free(OB.getBuffer());
}
