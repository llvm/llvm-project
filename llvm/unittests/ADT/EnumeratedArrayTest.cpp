//===- llvm/unittest/ADT/EnumeratedArrayTest.cpp ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EnumeratedArray unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/EnumeratedArray.h"
#include "gtest/gtest.h"

namespace llvm {

//===--------------------------------------------------------------------===//
// Test initialization and use of operator[] for both read and write.
//===--------------------------------------------------------------------===//

TEST(EnumeratedArray, InitAndIndex) {

  enum class Colors { Red, Blue, Green, Last = Green };

  EnumeratedArray<float, Colors, Colors::Last, size_t> Array1;

  Array1[Colors::Red] = 1.0;
  Array1[Colors::Blue] = 2.0;
  Array1[Colors::Green] = 3.0;

  EXPECT_EQ(Array1[Colors::Red], 1.0);
  EXPECT_EQ(Array1[Colors::Blue], 2.0);
  EXPECT_EQ(Array1[Colors::Green], 3.0);

  EnumeratedArray<bool, Colors> Array2(true);

  EXPECT_TRUE(Array2[Colors::Red]);
  EXPECT_TRUE(Array2[Colors::Blue]);
  EXPECT_TRUE(Array2[Colors::Green]);

  Array2[Colors::Red] = true;
  Array2[Colors::Blue] = false;
  Array2[Colors::Green] = true;

  EXPECT_TRUE(Array2[Colors::Red]);
  EXPECT_FALSE(Array2[Colors::Blue]);
  EXPECT_TRUE(Array2[Colors::Green]);
}

} // namespace llvm
