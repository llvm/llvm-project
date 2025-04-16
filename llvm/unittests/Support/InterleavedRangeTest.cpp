//===- InterleavedRangeTest.cpp - Unit tests for interleaved format -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/InterleavedRange.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(InterleavedRangeTest, VectorInt) {
  SmallVector<int> V = {0, 1, 2, 3};

  // First, make sure that the raw print API works as expected.
  std::string Buff;
  raw_string_ostream OS(Buff);
  OS << interleaved(V);
  EXPECT_EQ("0, 1, 2, 3", Buff);
  Buff.clear();
  OS << interleaved_array(V);
  EXPECT_EQ("[0, 1, 2, 3]", Buff);

  // In the rest of the tests, use `.str()` for convenience.
  EXPECT_EQ("0, 1, 2, 3", interleaved(V).str());
  EXPECT_EQ("{{0,1,2,3}}", interleaved(V, ",", "{{", "}}").str());
  EXPECT_EQ("[0, 1, 2, 3]", interleaved_array(V).str());
  EXPECT_EQ("[0;1;2;3]", interleaved_array(V, ";").str());
  EXPECT_EQ("0;1;2;3", interleaved(V, ";").str());
}

TEST(InterleavedRangeTest, VectorIntEmpty) {
  SmallVector<int> V = {};
  EXPECT_EQ("", interleaved(V).str());
  EXPECT_EQ("{{}}", interleaved(V, ",", "{{", "}}").str());
  EXPECT_EQ("[]", interleaved_array(V).str());
  EXPECT_EQ("", interleaved(V, ";").str());
}

TEST(InterleavedRangeTest, VectorIntOneElem) {
  SmallVector<int> V = {42};
  EXPECT_EQ("42", interleaved(V).str());
  EXPECT_EQ("{{42}}", interleaved(V, ",", "{{", "}}").str());
  EXPECT_EQ("[42]", interleaved_array(V).str());
  EXPECT_EQ("42", interleaved(V, ";").str());
}

struct CustomPrint {
  int N;
  friend raw_ostream &operator<<(raw_ostream &OS, const CustomPrint &CP) {
    OS << "$$" << CP.N << "##";
    return OS;
  }
};

TEST(InterleavedRangeTest, CustomPrint) {
  CustomPrint V[] = {{3}, {4}, {5}};
  EXPECT_EQ("$$3##, $$4##, $$5##", interleaved(V).str());
  EXPECT_EQ("{{$$3##;$$4##;$$5##}}", interleaved(V, ";", "{{", "}}").str());
  EXPECT_EQ("[$$3##, $$4##, $$5##]", interleaved_array(V).str());
}

} // namespace
