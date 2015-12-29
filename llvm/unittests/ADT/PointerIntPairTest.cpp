//===- llvm/unittest/ADT/PointerIntPairTest.cpp - Unit tests --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/PointerIntPair.h"
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
}

TEST(PointerIntPairTest, DefaultInitialize) {
  PointerIntPair<float *, 2> Pair;
  EXPECT_EQ(nullptr, Pair.getPointer());
  EXPECT_EQ(0U, Pair.getInt());
}

TEST(PointerIntPairTest, ManyUnusedBits) {
  // In real code this would be a word-sized integer limited to 31 bits.
  struct Fixnum31 {
    uintptr_t Value;
  };
  class FixnumPointerTraits {
  public:
    static inline void *getAsVoidPointer(Fixnum31 Num) {
      return reinterpret_cast<void *>(Num.Value << NumLowBitsAvailable);
    }
    static inline Fixnum31 getFromVoidPointer(void *P) {
      // In real code this would assert that the value is in range.
      return { reinterpret_cast<uintptr_t>(P) >> NumLowBitsAvailable };
    }
    enum { NumLowBitsAvailable = std::numeric_limits<uintptr_t>::digits - 31 };
  };

  PointerIntPair<Fixnum31, 1, bool, FixnumPointerTraits> pair;
  EXPECT_EQ((uintptr_t)0, pair.getPointer().Value);
  EXPECT_FALSE(pair.getInt());

  pair.setPointerAndInt({ 0x7FFFFFFF }, true );
  EXPECT_EQ((uintptr_t)0x7FFFFFFF, pair.getPointer().Value);
  EXPECT_TRUE(pair.getInt());

  EXPECT_EQ(FixnumPointerTraits::NumLowBitsAvailable - 1,
            PointerLikeTypeTraits<decltype(pair)>::NumLowBitsAvailable);
}

} // end anonymous namespace
