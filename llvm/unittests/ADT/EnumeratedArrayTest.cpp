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
#include "llvm/ADT/iterator_range.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <type_traits>

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

  EnumeratedArray<float, Colors, Colors::Last, size_t> Array3 = {10.0, 11.0,
                                                                 12.0};
  EXPECT_EQ(Array3[Colors::Red], 10.0);
  EXPECT_EQ(Array3[Colors::Blue], 11.0);
  EXPECT_EQ(Array3[Colors::Green], 12.0);
}

//===--------------------------------------------------------------------===//
// Test size and empty function
//===--------------------------------------------------------------------===//

TEST(EnumeratedArray, Size) {

  enum class Colors { Red, Blue, Green, Last = Green };

  EnumeratedArray<float, Colors, Colors::Last, size_t> Array;
  const auto &ConstArray = Array;

  EXPECT_EQ(ConstArray.size(), 3u);
  EXPECT_EQ(ConstArray.empty(), false);
}

//===--------------------------------------------------------------------===//
// Test iterators
//===--------------------------------------------------------------------===//

TEST(EnumeratedArray, Iterators) {

  enum class Colors { Red, Blue, Green, Last = Green };

  EnumeratedArray<float, Colors, Colors::Last, size_t> Array;
  const auto &ConstArray = Array;

  Array[Colors::Red] = 1.0;
  Array[Colors::Blue] = 2.0;
  Array[Colors::Green] = 3.0;

  EXPECT_THAT(Array, testing::ElementsAre(1.0, 2.0, 3.0));
  EXPECT_THAT(ConstArray, testing::ElementsAre(1.0, 2.0, 3.0));

  EXPECT_THAT(make_range(Array.rbegin(), Array.rend()),
              testing::ElementsAre(3.0, 2.0, 1.0));
  EXPECT_THAT(make_range(ConstArray.rbegin(), ConstArray.rend()),
              testing::ElementsAre(3.0, 2.0, 1.0));
}

//===--------------------------------------------------------------------===//
// Test typedefs
//===--------------------------------------------------------------------===//

namespace {

enum class Colors { Red, Blue, Green, Last = Green };

using Array = EnumeratedArray<float, Colors, Colors::Last, size_t>;

static_assert(std::is_same_v<Array::value_type, float>,
              "Incorrect value_type type");
static_assert(std::is_same_v<Array::reference, float &>,
              "Incorrect reference type!");
static_assert(std::is_same_v<Array::pointer, float *>,
              "Incorrect pointer type!");
static_assert(std::is_same_v<Array::const_reference, const float &>,
              "Incorrect const_reference type!");
static_assert(std::is_same_v<Array::const_pointer, const float *>,
              "Incorrect const_pointer type!");
} // namespace

} // namespace llvm
