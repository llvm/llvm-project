//===- IteratorTest.cpp - Unit tests for iterator utilities ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/iterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <int> struct Shadow;

struct WeirdIter : std::iterator<std::input_iterator_tag, Shadow<0>, Shadow<1>,
                                 Shadow<2>, Shadow<3>> {};

struct AdaptedIter : iterator_adaptor_base<AdaptedIter, WeirdIter> {};

// Test that iterator_adaptor_base forwards typedefs, if value_type is
// unchanged.
static_assert(std::is_same<typename AdaptedIter::value_type, Shadow<0>>::value,
              "");
static_assert(
    std::is_same<typename AdaptedIter::difference_type, Shadow<1>>::value, "");
static_assert(std::is_same<typename AdaptedIter::pointer, Shadow<2>>::value,
              "");
static_assert(std::is_same<typename AdaptedIter::reference, Shadow<3>>::value,
              "");

TEST(PointeeIteratorTest, Basic) {
  int arr[4] = { 1, 2, 3, 4 };
  SmallVector<int *, 4> V;
  V.push_back(&arr[0]);
  V.push_back(&arr[1]);
  V.push_back(&arr[2]);
  V.push_back(&arr[3]);

  typedef pointee_iterator<SmallVectorImpl<int *>::const_iterator> test_iterator;

  test_iterator Begin, End;
  Begin = V.begin();
  End = test_iterator(V.end());

  test_iterator I = Begin;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(*V[i], *I);

    EXPECT_EQ(I, Begin + i);
    EXPECT_EQ(I, std::next(Begin, i));
    test_iterator J = Begin;
    J += i;
    EXPECT_EQ(I, J);
    EXPECT_EQ(*V[i], Begin[i]);

    EXPECT_NE(I, End);
    EXPECT_GT(End, I);
    EXPECT_LT(I, End);
    EXPECT_GE(I, Begin);
    EXPECT_LE(Begin, I);

    EXPECT_EQ(i, I - Begin);
    EXPECT_EQ(i, std::distance(Begin, I));
    EXPECT_EQ(Begin, I - i);

    test_iterator K = I++;
    EXPECT_EQ(K, std::prev(I));
  }
  EXPECT_EQ(End, I);
}

TEST(PointeeIteratorTest, SmartPointer) {
  SmallVector<std::unique_ptr<int>, 4> V;
  V.push_back(make_unique<int>(1));
  V.push_back(make_unique<int>(2));
  V.push_back(make_unique<int>(3));
  V.push_back(make_unique<int>(4));

  typedef pointee_iterator<
      SmallVectorImpl<std::unique_ptr<int>>::const_iterator> test_iterator;

  test_iterator Begin, End;
  Begin = V.begin();
  End = test_iterator(V.end());

  test_iterator I = Begin;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(*V[i], *I);

    EXPECT_EQ(I, Begin + i);
    EXPECT_EQ(I, std::next(Begin, i));
    test_iterator J = Begin;
    J += i;
    EXPECT_EQ(I, J);
    EXPECT_EQ(*V[i], Begin[i]);

    EXPECT_NE(I, End);
    EXPECT_GT(End, I);
    EXPECT_LT(I, End);
    EXPECT_GE(I, Begin);
    EXPECT_LE(Begin, I);

    EXPECT_EQ(i, I - Begin);
    EXPECT_EQ(i, std::distance(Begin, I));
    EXPECT_EQ(Begin, I - i);

    test_iterator K = I++;
    EXPECT_EQ(K, std::prev(I));
  }
  EXPECT_EQ(End, I);
}

} // anonymous namespace
