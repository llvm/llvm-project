//===- RangeAdapterTest.cpp - Unit tests for range adapters  --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  typedef std::vector<int>::const_reverse_iterator const_reverse_iterator;
  reverse_iterator rbegin() { return Vec.rbegin(); }
  reverse_iterator rend() { return Vec.rend(); }
  const_reverse_iterator rbegin() const { return Vec.rbegin(); }
  const_reverse_iterator rend() const { return Vec.rend(); }
};

// A wrapper around vector which exposes begin(), end(), rbegin() and rend().
// begin() and end() don't have implementations as this ensures that we will
// get a linker error if reverse() chooses begin()/end() over rbegin(), rend().
class BidirectionalVector {
  mutable std::vector<int> Vec;

public:
  BidirectionalVector(std::initializer_list<int> list) : Vec(list) {}

  typedef std::vector<int>::iterator iterator;
  iterator begin() const;
  iterator end() const;

  typedef std::vector<int>::reverse_iterator reverse_iterator;
  reverse_iterator rbegin() const { return Vec.rbegin(); }
  reverse_iterator rend() const { return Vec.rend(); }
};

/// This is the same as BidirectionalVector but with the addition of const
/// begin/rbegin methods to ensure that the type traits for has_rbegin works.
class BidirectionalVectorConsts {
  std::vector<int> Vec;

public:
  BidirectionalVectorConsts(std::initializer_list<int> list) : Vec(list) {}

  typedef std::vector<int>::iterator iterator;
  typedef std::vector<int>::const_iterator const_iterator;
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;

  typedef std::vector<int>::reverse_iterator reverse_iterator;
  typedef std::vector<int>::const_reverse_iterator const_reverse_iterator;
  reverse_iterator rbegin() { return Vec.rbegin(); }
  reverse_iterator rend() { return Vec.rend(); }
  const_reverse_iterator rbegin() const { return Vec.rbegin(); }
  const_reverse_iterator rend() const { return Vec.rend(); }
};

/// Check that types with custom iterators work.
class CustomIteratorVector {
  mutable std::vector<int> V;

public:
  CustomIteratorVector(std::initializer_list<int> list) : V(list) {}

  typedef std::vector<int>::iterator iterator;
  class reverse_iterator {
    std::vector<int>::iterator I;

  public:
    reverse_iterator() = default;
    reverse_iterator(const reverse_iterator &) = default;
    reverse_iterator &operator=(const reverse_iterator &) = default;

    explicit reverse_iterator(std::vector<int>::iterator I) : I(I) {}

    reverse_iterator &operator++() {
      --I;
      return *this;
    }
    reverse_iterator &operator--() {
      ++I;
      return *this;
    }
    int &operator*() const { return *std::prev(I); }
    int *operator->() const { return &*std::prev(I); }
    friend bool operator==(const reverse_iterator &L,
                           const reverse_iterator &R) {
      return L.I == R.I;
    }
    friend bool operator!=(const reverse_iterator &L,
                           const reverse_iterator &R) {
      return !(L == R);
    }
  };

  iterator begin() const { return V.begin(); }
  iterator end()  const { return V.end(); }
  reverse_iterator rbegin() const { return reverse_iterator(V.end()); }
  reverse_iterator rend() const { return reverse_iterator(V.begin()); }
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
TYPED_TEST_SUITE(RangeAdapterLValueTest, RangeAdapterLValueTestTypes, );

TYPED_TEST(RangeAdapterLValueTest, TrivialOperation) {
  TypeParam v = {0, 1, 2, 3};
  TestRev(reverse(v));

  const TypeParam c = {0, 1, 2, 3};
  TestRev(reverse(c));
}

template <typename T> struct RangeAdapterRValueTest : testing::Test {};

typedef ::testing::Types<std::vector<int>, std::list<int>, CustomIteratorVector,
                         ReverseOnlyVector, BidirectionalVector,
                         BidirectionalVectorConsts>
    RangeAdapterRValueTestTypes;
TYPED_TEST_SUITE(RangeAdapterRValueTest, RangeAdapterRValueTestTypes, );

TYPED_TEST(RangeAdapterRValueTest, TrivialOperation) {
  TestRev(reverse(TypeParam({0, 1, 2, 3})));
}

TYPED_TEST(RangeAdapterRValueTest, RangeType) {
  static_assert(
      std::is_same_v<decltype(reverse(std::declval<TypeParam>()).begin()),
                     decltype(std::declval<TypeParam>().rbegin())>,
      "reverse().begin() should have the same type as rbegin()");
  static_assert(
      std::is_same_v<decltype(reverse(std::declval<const TypeParam>()).begin()),
                     decltype(std::declval<const TypeParam>().rbegin())>,
      "reverse().begin() should have the same type as rbegin() [const]");
}

} // anonymous namespace
