//===- llvm/unittest/ADT/PointerIntPairTest.cpp - Unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PointerIntPair.h"
#include "gtest/gtest.h"
#include <limits>
using namespace llvm;

namespace {

TEST(PointerIntPairTest, GetSet) {
  struct S {
    int i;
  };
  S s;

  PointerIntPair<S *, 2> Pair(&s, 1U);
  EXPECT_EQ(&s, Pair.getPointer());
  EXPECT_EQ(1U, Pair.getInt());

  Pair.setInt(2);
  EXPECT_EQ(&s, Pair.getPointer());
  EXPECT_EQ(2U, Pair.getInt());

  Pair.setPointer(nullptr);
  EXPECT_EQ(nullptr, Pair.getPointer());
  EXPECT_EQ(2U, Pair.getInt());

  Pair.setPointerAndInt(&s, 3U);
  EXPECT_EQ(&s, Pair.getPointer());
  EXPECT_EQ(3U, Pair.getInt());

  // Make sure that we can perform all of our operations on enum classes.
  //
  // The concern is that enum classes are only explicitly convertible to
  // integers. This means that if we assume in PointerIntPair this, a
  // compilation error will result. This group of tests exercises the enum class
  // code to make sure that we do not run into such issues in the future.
  enum class E : unsigned {
    Case1,
    Case2,
    Case3,
  };
  PointerIntPair<S *, 2, E> Pair2(&s, E::Case1);
  EXPECT_EQ(&s, Pair2.getPointer());
  EXPECT_EQ(E::Case1, Pair2.getInt());

  Pair2.setInt(E::Case2);
  EXPECT_EQ(&s, Pair2.getPointer());
  EXPECT_EQ(E::Case2, Pair2.getInt());

  Pair2.setPointer(nullptr);
  EXPECT_EQ(nullptr, Pair2.getPointer());
  EXPECT_EQ(E::Case2, Pair2.getInt());

  Pair2.setPointerAndInt(&s, E::Case3);
  EXPECT_EQ(&s, Pair2.getPointer());
  EXPECT_EQ(E::Case3, Pair2.getInt());

  static_assert(std::is_trivially_copyable<PointerIntPair<S *, 2, E>>::value,
                "trivially copyable");
}

TEST(PointerIntPairTest, DefaultInitialize) {
  PointerIntPair<float *, 2> Pair;
  EXPECT_EQ(nullptr, Pair.getPointer());
  EXPECT_EQ(0U, Pair.getInt());
}

// In real code this would be a word-sized integer limited to 31 bits.
struct Fixnum31 {
  uintptr_t Value;
};
struct FixnumPointerTraits {
  static inline void *getAsVoidPointer(Fixnum31 Num) {
    return reinterpret_cast<void *>(Num.Value << NumLowBitsAvailable);
  }
  static inline Fixnum31 getFromVoidPointer(void *P) {
    // In real code this would assert that the value is in range.
    return {reinterpret_cast<uintptr_t>(P) >> NumLowBitsAvailable};
  }
  static constexpr int NumLowBitsAvailable =
      std::numeric_limits<uintptr_t>::digits - 31;
};
TEST(PointerIntPairTest, ManyUnusedBits) {

  PointerIntPair<Fixnum31, 1, bool, FixnumPointerTraits> pair;
  EXPECT_EQ((uintptr_t)0, pair.getPointer().Value);
  EXPECT_FALSE(pair.getInt());

  pair.setPointerAndInt({ 0x7FFFFFFF }, true );
  EXPECT_EQ((uintptr_t)0x7FFFFFFF, pair.getPointer().Value);
  EXPECT_TRUE(pair.getInt());

  EXPECT_EQ(FixnumPointerTraits::NumLowBitsAvailable - 1,
            (int)PointerLikeTypeTraits<decltype(pair)>::NumLowBitsAvailable);

  static_assert(
      std::is_trivially_copyable<
          PointerIntPair<Fixnum31, 1, bool, FixnumPointerTraits>>::value,
      "trivially copyable");
}

} // end anonymous namespace
