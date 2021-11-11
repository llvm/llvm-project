//===------ MappedIteratorTest.cpp - Unit tests for mapped_iterator -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

TEST(MappedIteratorTest, CustomIteratorApplyFunctionOnDereference) {
  struct CustomMapIterator
      : public llvm::mapped_iterator_base<CustomMapIterator,
                                          std::vector<int>::iterator, int> {
    using BaseT::BaseT;

    /// Map the element to the iterator result type.
    int mapElement(int X) const { return X + 1; }
  };

  std::vector<int> V({0});

  CustomMapIterator I(V.begin());

  EXPECT_EQ(*I, 1) << "should have applied function in dereference";
}

TEST(MappedIteratorTest, CustomIteratorApplyFunctionOnArrow) {
  struct S {
    int Z = 0;
  };
  struct CustomMapIterator
      : public llvm::mapped_iterator_base<CustomMapIterator,
                                          std::vector<int>::iterator, S &> {
    CustomMapIterator(std::vector<int>::iterator it, S *P) : BaseT(it), P(P) {}

    /// Map the element to the iterator result type.
    S &mapElement(int X) const { return *(P + X); }

    S *P;
  };

  std::vector<int> V({0});
  S Y;

  CustomMapIterator I(V.begin(), &Y);

  I->Z = 42;

  EXPECT_EQ(Y.Z, 42) << "should have applied function during arrow";
}

TEST(MappedIteratorTest, CustomIteratorFunctionPreservesReferences) {
  struct CustomMapIterator
      : public llvm::mapped_iterator_base<CustomMapIterator,
                                          std::vector<int>::iterator, int &> {
    CustomMapIterator(std::vector<int>::iterator it, std::map<int, int> &M)
        : BaseT(it), M(M) {}

    /// Map the element to the iterator result type.
    int &mapElement(int X) const { return M[X]; }

    std::map<int, int> &M;
  };
  std::vector<int> V({1});
  std::map<int, int> M({{1, 1}});

  auto I = CustomMapIterator(V.begin(), M);
  *I = 42;

  EXPECT_EQ(M[1], 42) << "assignment should have modified M";
}

} // anonymous namespace
