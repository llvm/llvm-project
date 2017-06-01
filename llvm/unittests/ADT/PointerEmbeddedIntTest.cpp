//===- llvm/unittest/ADT/PointerEmbeddedIntTest.cpp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
using namespace llvm;

namespace {

TEST(PointerEmbeddedIntTest, Basic) {
  PointerEmbeddedInt<int, CHAR_BIT> I = 42, J = 43;

  EXPECT_EQ(42, I);
  EXPECT_EQ(43, I + 1);
  EXPECT_EQ(sizeof(uintptr_t) * CHAR_BIT - CHAR_BIT,
            PointerLikeTypeTraits<decltype(I)>::NumLowBitsAvailable);

  EXPECT_FALSE(I == J);
  EXPECT_TRUE(I != J);
  EXPECT_TRUE(I < J);
  EXPECT_FALSE(I > J);
  EXPECT_TRUE(I <= J);
  EXPECT_FALSE(I >= J);

  EXPECT_FALSE(I == 43);
  EXPECT_TRUE(I != 43);
  EXPECT_TRUE(I < 43);
  EXPECT_FALSE(I > 43);
  EXPECT_TRUE(I <= 43);
  EXPECT_FALSE(I >= 43);

  EXPECT_FALSE(42 == J);
  EXPECT_TRUE(42 != J);
  EXPECT_TRUE(42 < J);
  EXPECT_FALSE(42 > J);
  EXPECT_TRUE(42 <= J);
  EXPECT_FALSE(42 >= J);
}

TEST(PointerEmbeddedIntTest, intptr_t) {
  PointerEmbeddedInt<intptr_t, CHAR_BIT> IPos = 42, INeg = -42;
  EXPECT_EQ(42, IPos);
  EXPECT_EQ(-42, INeg);

  PointerEmbeddedInt<uintptr_t, CHAR_BIT> U = 42, USaturated = 255;
  EXPECT_EQ(42U, U);
  EXPECT_EQ(255U, USaturated);

  PointerEmbeddedInt<intptr_t, std::numeric_limits<intptr_t>::digits>
      IMax = std::numeric_limits<intptr_t>::max() >> 1,
      IMin = std::numeric_limits<intptr_t>::min() >> 1;
  EXPECT_EQ(std::numeric_limits<intptr_t>::max() >> 1, IMax);
  EXPECT_EQ(std::numeric_limits<intptr_t>::min() >> 1, IMin);

  PointerEmbeddedInt<uintptr_t, std::numeric_limits<uintptr_t>::digits - 1>
      UMax = std::numeric_limits<uintptr_t>::max() >> 1,
      UMin = std::numeric_limits<uintptr_t>::min() >> 1;
  EXPECT_EQ(std::numeric_limits<uintptr_t>::max() >> 1, UMax);
  EXPECT_EQ(std::numeric_limits<uintptr_t>::min() >> 1, UMin);
}

TEST(PointerEmbeddedIntTest, PointerLikeTypeTraits) {
  PointerEmbeddedInt<int, CHAR_BIT> I = 42;
  using ITraits = PointerLikeTypeTraits<decltype(I)>;
  EXPECT_EQ(42, ITraits::getFromVoidPointer(ITraits::getAsVoidPointer(I)));

  PointerEmbeddedInt<uintptr_t, std::numeric_limits<uintptr_t>::digits - 1>
      Max = std::numeric_limits<uintptr_t>::max() >> 1;
  using MaxTraits = PointerLikeTypeTraits<decltype(Max)>;
  EXPECT_EQ(std::numeric_limits<uintptr_t>::max() >> 1,
            MaxTraits::getFromVoidPointer(MaxTraits::getAsVoidPointer(Max)));
}

} // end anonymous namespace
