//===- llvm/unittest/ADT/ArrayRefTest.cpp - ArrayRef unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <limits>
#include <vector>
using namespace llvm;

// Check that the ArrayRef-of-pointer converting constructor only allows adding
// cv qualifiers (not removing them, or otherwise changing the type)
static_assert(std::is_convertible_v<ArrayRef<int *>, ArrayRef<const int *>>,
              "Adding const");
static_assert(std::is_convertible_v<ArrayRef<int *>, ArrayRef<volatile int *>>,
              "Adding volatile");
static_assert(!std::is_convertible_v<ArrayRef<int *>, ArrayRef<float *>>,
              "Changing pointer of one type to a pointer of another");
static_assert(!std::is_convertible_v<ArrayRef<const int *>, ArrayRef<int *>>,
              "Removing const");
static_assert(!std::is_convertible_v<ArrayRef<volatile int *>, ArrayRef<int *>>,
              "Removing volatile");

// Check that we can't accidentally assign a temporary location to an ArrayRef.
// (Unfortunately we can't make use of the same thing with constructors.)
static_assert(!std::is_assignable_v<ArrayRef<int *> &, int *>,
              "Assigning from single prvalue element");
static_assert(!std::is_assignable_v<ArrayRef<int *> &, int *&&>,
              "Assigning from single xvalue element");
static_assert(std::is_assignable_v<ArrayRef<int *> &, int *&>,
              "Assigning from single lvalue element");
static_assert(
    !std::is_assignable_v<ArrayRef<int *> &, std::initializer_list<int *>>,
    "Assigning from an initializer list");

namespace {

TEST(ArrayRefTest, AllocatorCopy) {
  BumpPtrAllocator Alloc;
  static const uint16_t Words1[] = { 1, 4, 200, 37 };
  ArrayRef<uint16_t> Array1 = ArrayRef(Words1, 4);
  static const uint16_t Words2[] = { 11, 4003, 67, 64000, 13 };
  ArrayRef<uint16_t> Array2 = ArrayRef(Words2, 5);
  ArrayRef<uint16_t> Array1c = Array1.copy(Alloc);
  ArrayRef<uint16_t> Array2c = Array2.copy(Alloc);
  EXPECT_TRUE(Array1.equals(Array1c));
  EXPECT_NE(Array1.data(), Array1c.data());
  EXPECT_TRUE(Array2.equals(Array2c));
  EXPECT_NE(Array2.data(), Array2c.data());

  // Check that copy can cope with uninitialized memory.
  struct NonAssignable {
    const char *Ptr;

    NonAssignable(const char *Ptr) : Ptr(Ptr) {}
    NonAssignable(const NonAssignable &RHS) = default;
    void operator=(const NonAssignable &RHS) { assert(RHS.Ptr != nullptr); }
    bool operator==(const NonAssignable &RHS) const { return Ptr == RHS.Ptr; }
  } Array3Src[] = {"hello", "world"};
  ArrayRef<NonAssignable> Array3Copy = ArrayRef(Array3Src).copy(Alloc);
  EXPECT_EQ(ArrayRef(Array3Src), Array3Copy);
  EXPECT_NE(ArrayRef(Array3Src).data(), Array3Copy.data());
}

// This test is pure UB given the ArrayRef<> implementation.
// You are not allowed to produce non-null pointers given null base pointer.
TEST(ArrayRefTest, DISABLED_SizeTSizedOperations) {
  ArrayRef<char> AR(nullptr, std::numeric_limits<ptrdiff_t>::max());

  // Check that drop_back accepts size_t-sized numbers.
  EXPECT_EQ(1U, AR.drop_back(AR.size() - 1).size());

  // Check that drop_front accepts size_t-sized numbers.
  EXPECT_EQ(1U, AR.drop_front(AR.size() - 1).size());

  // Check that slice accepts size_t-sized numbers.
  EXPECT_EQ(1U, AR.slice(AR.size() - 1).size());
  EXPECT_EQ(AR.size() - 1, AR.slice(1, AR.size() - 1).size());
}

TEST(ArrayRefTest, DropBack) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(TheNumbers, AR1.size() - 1);
  EXPECT_TRUE(AR1.drop_back().equals(AR2));
}

TEST(ArrayRefTest, DropFront) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(&TheNumbers[2], AR1.size() - 2);
  EXPECT_TRUE(AR1.drop_front(2).equals(AR2));
}

TEST(ArrayRefTest, DropWhile) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.drop_front(3);
  EXPECT_EQ(Expected, AR1.drop_while([](const int &N) { return N % 2 == 1; }));

  EXPECT_EQ(AR1, AR1.drop_while([](const int &N) { return N < 0; }));
  EXPECT_EQ(ArrayRef<int>(),
            AR1.drop_while([](const int &N) { return N > 0; }));
}

TEST(ArrayRefTest, DropUntil) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.drop_front(3);
  EXPECT_EQ(Expected, AR1.drop_until([](const int &N) { return N % 2 == 0; }));

  EXPECT_EQ(ArrayRef<int>(),
            AR1.drop_until([](const int &N) { return N < 0; }));
  EXPECT_EQ(AR1, AR1.drop_until([](const int &N) { return N > 0; }));
}

TEST(ArrayRefTest, TakeBack) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(AR1.end() - 1, 1);
  EXPECT_TRUE(AR1.take_back().equals(AR2));
}

TEST(ArrayRefTest, TakeFront) {
  static const int TheNumbers[] = {4, 8, 15, 16, 23, 42};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> AR2(AR1.data(), 2);
  EXPECT_TRUE(AR1.take_front(2).equals(AR2));
}

TEST(ArrayRefTest, TakeWhile) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.take_front(3);
  EXPECT_EQ(Expected, AR1.take_while([](const int &N) { return N % 2 == 1; }));

  EXPECT_EQ(ArrayRef<int>(),
            AR1.take_while([](const int &N) { return N < 0; }));
  EXPECT_EQ(AR1, AR1.take_while([](const int &N) { return N > 0; }));
}

TEST(ArrayRefTest, TakeUntil) {
  static const int TheNumbers[] = {1, 3, 5, 8, 10, 11};
  ArrayRef<int> AR1(TheNumbers);
  ArrayRef<int> Expected = AR1.take_front(3);
  EXPECT_EQ(Expected, AR1.take_until([](const int &N) { return N % 2 == 0; }));

  EXPECT_EQ(AR1, AR1.take_until([](const int &N) { return N < 0; }));
  EXPECT_EQ(ArrayRef<int>(),
            AR1.take_until([](const int &N) { return N > 0; }));
}

TEST(ArrayRefTest, Equals) {
  static const int A1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  ArrayRef<int> AR1(A1);
  EXPECT_TRUE(AR1.equals({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(AR1.equals({8, 1, 2, 4, 5, 6, 6, 7}));
  EXPECT_FALSE(AR1.equals({2, 4, 5, 6, 6, 7, 8, 1}));
  EXPECT_FALSE(AR1.equals({0, 1, 2, 4, 5, 6, 6, 7}));
  EXPECT_FALSE(AR1.equals({1, 2, 42, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(AR1.equals({42, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(AR1.equals({1, 2, 3, 4, 5, 6, 7, 42}));
  EXPECT_FALSE(AR1.equals({1, 2, 3, 4, 5, 6, 7}));
  EXPECT_FALSE(AR1.equals({1, 2, 3, 4, 5, 6, 7, 8, 9}));

  ArrayRef<int> AR1a = AR1.drop_back();
  EXPECT_TRUE(AR1a.equals({1, 2, 3, 4, 5, 6, 7}));
  EXPECT_FALSE(AR1a.equals({1, 2, 3, 4, 5, 6, 7, 8}));

  ArrayRef<int> AR1b = AR1a.slice(2, 4);
  EXPECT_TRUE(AR1b.equals({3, 4, 5, 6}));
  EXPECT_FALSE(AR1b.equals({2, 3, 4, 5, 6}));
  EXPECT_FALSE(AR1b.equals({3, 4, 5, 6, 7}));
}

TEST(ArrayRefTest, EmptyEquals) {
  EXPECT_TRUE(ArrayRef<unsigned>() == ArrayRef<unsigned>());
}

TEST(ArrayRefTest, ConstConvert) {
  int buf[4];
  for (int i = 0; i < 4; ++i)
    buf[i] = i;

  static int *A[] = {&buf[0], &buf[1], &buf[2], &buf[3]};
  ArrayRef<const int *> a((ArrayRef<int *>(A)));
  a = ArrayRef<int *>(A);
}

static std::vector<int> ReturnTest12() { return {1, 2}; }
static void ArgTest12(ArrayRef<int> A) {
  EXPECT_EQ(2U, A.size());
  EXPECT_EQ(1, A[0]);
  EXPECT_EQ(2, A[1]);
}

TEST(ArrayRefTest, InitializerList) {
  std::initializer_list<int> init_list = { 0, 1, 2, 3, 4 };
  ArrayRef<int> A = init_list;
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(i, A[i]);

  std::vector<int> B = ReturnTest12();
  A = B;
  EXPECT_EQ(1, A[0]);
  EXPECT_EQ(2, A[1]);

  ArgTest12({1, 2});
}

TEST(ArrayRefTest, EmptyInitializerList) {
  ArrayRef<int> A = {};
  EXPECT_TRUE(A.empty());

  A = {};
  EXPECT_TRUE(A.empty());
}

TEST(ArrayRefTest, ArrayRef) {
  static const int A1[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // A copy is expected for non-const ArrayRef (thin copy)
  ArrayRef<int> AR1(A1);
  const ArrayRef<int> &AR1Ref = ArrayRef(AR1);
  EXPECT_NE(&AR1, &AR1Ref);
  EXPECT_TRUE(AR1.equals(AR1Ref));

  // A copy is expected for non-const ArrayRef (thin copy)
  const ArrayRef<int> AR2(A1);
  const ArrayRef<int> &AR2Ref = ArrayRef(AR2);
  EXPECT_NE(&AR2Ref, &AR2);
  EXPECT_TRUE(AR2.equals(AR2Ref));
}

TEST(ArrayRefTest, OwningArrayRef) {
  static const int A1[] = {0, 1};
  OwningArrayRef<int> A{ArrayRef(A1)};
  OwningArrayRef<int> B(std::move(A));
  EXPECT_EQ(A.data(), nullptr);
}

TEST(ArrayRefTest, ArrayRefFromStdArray) {
  std::array<int, 5> A1{{42, -5, 0, 1000000, -1000000}};
  ArrayRef<int> A2 = ArrayRef(A1);

  EXPECT_EQ(A1.size(), A2.size());
  for (std::size_t i = 0; i < A1.size(); ++i) {
    EXPECT_EQ(A1[i], A2[i]);
  }
}

static_assert(std::is_trivially_copyable_v<ArrayRef<int>>,
              "trivially copyable");

TEST(ArrayRefTest, MutableArrayRefDeductionGuides) {
  // Single element
  {
    int x = 0;
    auto aref = MutableArrayRef(x);
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref)>);
    EXPECT_EQ(aref.data(), &x);
    EXPECT_EQ(aref.size(), 1u);

    // Make sure it's mutable still
    aref[0] = 1;
    EXPECT_EQ(x, 1);
  }

  // Pointer + length
  {
    int x[] = {0, 1, 2, 3};
    auto aref = MutableArrayRef(&x[0], 4);
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref)>);
    EXPECT_EQ(aref.data(), &x[0]);
    EXPECT_EQ(aref.size(), 4u);
  }

  // // Pointer + pointer
  {
    int x[] = {0, 1, 2, 3};
    auto aref = MutableArrayRef(std::begin(x), std::end(x));
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref)>);
    EXPECT_EQ(aref.data(), &x[0]);
    EXPECT_EQ(aref.size(), 4u);
  }

  // SmallVector
  {
    SmallVector<int> sv1;
    SmallVectorImpl<int> &sv2 = sv1;
    sv1.resize(5);
    auto aref1 = MutableArrayRef(sv1);
    auto aref2 = MutableArrayRef(sv2);
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref1)>);
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref2)>);
    EXPECT_EQ(aref1.data(), sv1.data());
    EXPECT_EQ(aref1.size(), sv1.size());
    EXPECT_EQ(aref2.data(), sv2.data());
    EXPECT_EQ(aref2.size(), sv2.size());
  }

  // std::vector
  {
    std::vector<int> x(5);
    auto aref = MutableArrayRef(x);
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref)>);
    EXPECT_EQ(aref.data(), x.data());
    EXPECT_EQ(aref.size(), x.size());
  }

  // std::array
  {
    std::array<int, 5> x{};
    auto aref = MutableArrayRef(x);
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref)>);
    EXPECT_EQ(aref.data(), x.data());
    EXPECT_EQ(aref.size(), x.size());
  }

  // MutableArrayRef
  {
    MutableArrayRef<int> x{};
    auto aref = MutableArrayRef(x);
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref)>);
    EXPECT_EQ(aref.data(), x.data());
    EXPECT_EQ(aref.size(), x.size());

    const MutableArrayRef<int> y{};
    auto aref2 = MutableArrayRef(y);
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref2)>);
    EXPECT_EQ(aref2.data(), y.data());
    EXPECT_EQ(aref2.size(), y.size());
  }

  // C-style array
  {
    int x[] = {0, 1, 2, 3};
    auto aref = MutableArrayRef(x);
    static_assert(std::is_same_v<MutableArrayRef<int>, decltype(aref)>);
    EXPECT_EQ(aref.data(), &x[0]);
    EXPECT_EQ(aref.size(), 4u);
  }
}

} // end anonymous namespace
