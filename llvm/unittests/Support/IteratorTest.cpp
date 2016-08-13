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

TEST(FilterIteratorTest, Lambda) {
  auto IsOdd = [](int N) { return N % 2 == 1; };
  int A[] = {0, 1, 2, 3, 4, 5, 6};
  auto Range = make_filter_range(A, IsOdd);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, CallableObject) {
  int Counter = 0;
  struct Callable {
    int &Counter;

    Callable(int &Counter) : Counter(Counter) {}

    bool operator()(int N) {
      Counter++;
      return N % 2 == 1;
    }
  };
  Callable IsOdd(Counter);
  int A[] = {0, 1, 2, 3, 4, 5, 6};
  auto Range = make_filter_range(A, IsOdd);
  EXPECT_EQ(2, Counter);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_GE(Counter, 7);
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, FunctionPointer) {
  bool (*IsOdd)(int) = [](int N) { return N % 2 == 1; };
  int A[] = {0, 1, 2, 3, 4, 5, 6};
  auto Range = make_filter_range(A, IsOdd);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, Composition) {
  auto IsOdd = [](int N) { return N % 2 == 1; };
  std::unique_ptr<int> A[] = {make_unique<int>(0), make_unique<int>(1),
                              make_unique<int>(2), make_unique<int>(3),
                              make_unique<int>(4), make_unique<int>(5),
                              make_unique<int>(6)};
  using PointeeIterator = pointee_iterator<std::unique_ptr<int> *>;
  auto Range = make_filter_range(
      make_range(PointeeIterator(std::begin(A)), PointeeIterator(std::end(A))),
      IsOdd);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, InputIterator) {
  struct InputIterator
      : iterator_adaptor_base<InputIterator, int *, std::input_iterator_tag> {
    using BaseT =
        iterator_adaptor_base<InputIterator, int *, std::input_iterator_tag>;

    InputIterator(int *It) : BaseT(It) {}
  };

  auto IsOdd = [](int N) { return N % 2 == 1; };
  int A[] = {0, 1, 2, 3, 4, 5, 6};
  auto Range = make_filter_range(
      make_range(InputIterator(std::begin(A)), InputIterator(std::end(A))),
      IsOdd);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

} // anonymous namespace
