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

struct CustomDoublingOStream : raw_string_ostream {
  unsigned NumCalled = 0;
  using raw_string_ostream::raw_string_ostream;

  friend CustomDoublingOStream &operator<<(CustomDoublingOStream &OS, int V) {
    ++OS.NumCalled;
    static_cast<raw_string_ostream &>(OS) << (2 * V);
    return OS;
  }
};

TEST(InterleavedRangeTest, CustomOStream) {
  // Make sure that interleaved calls the stream operator on the derived class,
  // and that it returns a reference to the same stream type.
  int V[] = {3, 4, 5};
  std::string Buf;
  CustomDoublingOStream OS(Buf);
  OS << interleaved(V) << 22;
  EXPECT_EQ("6, 8, 1044", Buf);
  EXPECT_EQ(OS.NumCalled, 4u);
}

} // namespace
