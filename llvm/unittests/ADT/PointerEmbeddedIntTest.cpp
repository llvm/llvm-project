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
  {
    PointerEmbeddedInt<intptr_t, CHAR_BIT> I = 42, J = -42;
    EXPECT_EQ(42, I);
    EXPECT_EQ(-42, J);
  }

  {
    PointerEmbeddedInt<uintptr_t, CHAR_BIT> I = 42, J = 255;
    EXPECT_EQ(42U, I);
    EXPECT_EQ(255U, J);
  }

  {
    PointerEmbeddedInt<intptr_t, std::numeric_limits<intptr_t>::digits>
        I = std::numeric_limits<intptr_t>::max() >> 1,
        J = std::numeric_limits<intptr_t>::min() >> 1;
    EXPECT_EQ(std::numeric_limits<intptr_t>::max() >> 1, I);
    EXPECT_EQ(std::numeric_limits<intptr_t>::min() >> 1, J);
  }

  {
    PointerEmbeddedInt<uintptr_t, std::numeric_limits<uintptr_t>::digits - 1>
        I = std::numeric_limits<uintptr_t>::max() >> 1,
        J = std::numeric_limits<uintptr_t>::min() >> 1;
    EXPECT_EQ(std::numeric_limits<uintptr_t>::max() >> 1, I);
    EXPECT_EQ(std::numeric_limits<uintptr_t>::min() >> 1, J);
  }
}

TEST(PointerEmbeddedIntTest, PointerLikeTypeTraits) {
  {
    PointerEmbeddedInt<int, CHAR_BIT> I = 42;
    using Traits = PointerLikeTypeTraits<decltype(I)>;
    EXPECT_EQ(42, Traits::getFromVoidPointer(Traits::getAsVoidPointer(I)));
  }

  {
    PointerEmbeddedInt<uintptr_t, std::numeric_limits<uintptr_t>::digits - 1>
        I = std::numeric_limits<uintptr_t>::max() >> 1;
    using Traits = PointerLikeTypeTraits<decltype(I)>;
    EXPECT_EQ(std::numeric_limits<uintptr_t>::max() >> 1,
              Traits::getFromVoidPointer(Traits::getAsVoidPointer(I)));
  }
}

} // end anonymous namespace
