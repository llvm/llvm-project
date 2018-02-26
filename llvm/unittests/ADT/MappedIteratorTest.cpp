//===------ MappedIteratorTest.cpp - Unit tests for mapped_iterator -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(MappedIteratorTest, ApplyFunctionOnDereference) {
  std::vector<int> V({0});

  auto I = map_iterator(V.begin(), [](int X) { return X + 1; });

  EXPECT_EQ(*I, 1) << "should have applied function in dereference";
}

TEST(MappedIteratorTest, ApplyFunctionOnArrow) {
  struct S {
    int Z = 0;
  };

  std::vector<int> V({0});
  S Y;
  S* P = &Y;

  auto I = map_iterator(V.begin(), [&](int X) -> S& { return *(P + X); });

  I->Z = 42;

  EXPECT_EQ(Y.Z, 42) << "should have applied function during arrow";
}

TEST(MappedIteratorTest, FunctionPreservesReferences) {
  std::vector<int> V({1});
  std::map<int, int> M({ {1, 1} });

  auto I = map_iterator(V.begin(), [&](int X) -> int& { return M[X]; });
  *I = 42;

  EXPECT_EQ(M[1], 42) << "assignment should have modified M";
}

} // anonymous namespace
