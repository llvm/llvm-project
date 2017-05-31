//===- RangeAdapterTest.cpp - Unit tests for range adapters  --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

#include <iterator>
#include <list>
#include <vector>

using namespace llvm;

namespace {

// A wrapper around vector which exposes rbegin(), rend().
class ReverseOnlyVector {
  std::vector<int> Vec;

public:
  ReverseOnlyVector(std::initializer_list<int> list) : Vec(list) {}

  typedef std::vector<int>::reverse_iterator reverse_iterator;
  reverse_iterator rbegin() { return Vec.rbegin(); }
  reverse_iterator rend() { return Vec.rend(); }
};

// A wrapper around vector which exposes begin(), end(), rbegin() and rend().
// begin() and end() don't have implementations as this ensures that we will
// get a linker error if reverse() chooses begin()/end() over rbegin(), rend().
class BidirectionalVector {
  std::vector<int> Vec;

public:
  BidirectionalVector(std::initializer_list<int> list) : Vec(list) {}

  typedef std::vector<int>::iterator iterator;
  iterator begin();
  iterator end();

  typedef std::vector<int>::reverse_iterator reverse_iterator;
  reverse_iterator rbegin() { return Vec.rbegin(); }
  reverse_iterator rend() { return Vec.rend(); }
};

template <typename R> void TestRev(const R &r) {
  int counter = 3;
  for (int i : r)
    EXPECT_EQ(i, counter--);
}

// Test fixture
template <typename T> class RangeAdapterLValueTest : public ::testing::Test {};

typedef ::testing::Types<std::vector<int>, std::list<int>, int[4]>
    RangeAdapterLValueTestTypes;
TYPED_TEST_CASE(RangeAdapterLValueTest, RangeAdapterLValueTestTypes);

TYPED_TEST(RangeAdapterLValueTest, TrivialOperation) {
  TypeParam v = {0, 1, 2, 3};
  TestRev(reverse(v));

  const TypeParam c = {0, 1, 2, 3};
  TestRev(reverse(c));
}

template <typename T> struct RangeAdapterRValueTest : testing::Test {};

typedef ::testing::Types<std::vector<int>, std::list<int>, ReverseOnlyVector,
                         BidirectionalVector> RangeAdapterRValueTestTypes;
TYPED_TEST_CASE(RangeAdapterRValueTest, RangeAdapterRValueTestTypes);

TYPED_TEST(RangeAdapterRValueTest, TrivialOperation) {
  TestRev(reverse(TypeParam({0, 1, 2, 3})));
}

} // anonymous namespace
