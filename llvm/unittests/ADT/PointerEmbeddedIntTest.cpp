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

} // end anonymous namespace
