//===- STLForwardCompatTest.cpp - Unit tests for STLForwardCompat ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLForwardCompat.h"
#include "gtest/gtest.h"

namespace {

TEST(STLForwardCompatTest, NegationTest) {
  EXPECT_TRUE((llvm::negation<std::false_type>::value));
  EXPECT_FALSE((llvm::negation<std::true_type>::value));
}

struct incomplete_type;

TEST(STLForwardCompatTest, ConjunctionTest) {
  EXPECT_TRUE((llvm::conjunction<>::value));
  EXPECT_FALSE((llvm::conjunction<std::false_type>::value));
  EXPECT_TRUE((llvm::conjunction<std::true_type>::value));
  EXPECT_FALSE((llvm::conjunction<std::false_type, incomplete_type>::value));
  EXPECT_FALSE((llvm::conjunction<std::false_type, std::true_type>::value));
  EXPECT_FALSE((llvm::conjunction<std::true_type, std::false_type>::value));
  EXPECT_TRUE((llvm::conjunction<std::true_type, std::true_type>::value));
  EXPECT_TRUE((llvm::conjunction<std::true_type, std::true_type,
                                 std::true_type>::value));
}

TEST(STLForwardCompatTest, DisjunctionTest) {
  EXPECT_FALSE((llvm::disjunction<>::value));
  EXPECT_FALSE((llvm::disjunction<std::false_type>::value));
  EXPECT_TRUE((llvm::disjunction<std::true_type>::value));
  EXPECT_TRUE((llvm::disjunction<std::true_type, incomplete_type>::value));
  EXPECT_TRUE((llvm::disjunction<std::false_type, std::true_type>::value));
  EXPECT_TRUE((llvm::disjunction<std::true_type, std::false_type>::value));
  EXPECT_TRUE((llvm::disjunction<std::true_type, std::true_type>::value));
  EXPECT_TRUE((llvm::disjunction<std::true_type, std::true_type,
                                 std::true_type>::value));
}

TEST(STLForwardCompatTest, MakeArrayTest) {
  ASSERT_EQ((llvm::make_array<int>()), (std::array<int, 0u>{}));
  ASSERT_EQ((llvm::make_array<size_t>()), (std::array<size_t, 0u>{}));
  ASSERT_EQ((llvm::make_array<size_t>(1u, 2u, 3u)),
            (std::array<size_t, 3u>{1, 2, 3}));
  ASSERT_EQ((llvm::make_array<size_t>(1u, 2u, 3u)),
            (llvm::make_array(size_t(1), size_t(2u), size_t(3u))));
  ASSERT_EQ((llvm::make_array(true, false, false, true)),
            (std::array<bool, 4u>{true, false, false, true}));
}

template <typename T>
class STLForwardCompatRemoveCVRefTest : public ::testing::Test {};

using STLForwardCompatRemoveCVRefTestTypes = ::testing::Types<
    // clang-format off
    std::pair<int, int>,
    std::pair<int &, int>,
    std::pair<const int, int>,
    std::pair<volatile int, int>,
    std::pair<const volatile int &, int>,
    std::pair<int *, int *>,
    std::pair<int *const, int *>,
    std::pair<const int *, const int *>,
    std::pair<int *&, int *>
    // clang-format on
    >;

TYPED_TEST_SUITE(STLForwardCompatRemoveCVRefTest,
                 STLForwardCompatRemoveCVRefTestTypes, );

TYPED_TEST(STLForwardCompatRemoveCVRefTest, RemoveCVRef) {
  using From = typename TypeParam::first_type;
  using To = typename TypeParam::second_type;
  EXPECT_TRUE(
      (std::is_same<typename llvm::remove_cvref<From>::type, To>::value));
}

TYPED_TEST(STLForwardCompatRemoveCVRefTest, RemoveCVRefT) {
  using From = typename TypeParam::first_type;
  EXPECT_TRUE((std::is_same<typename llvm::remove_cvref<From>::type,
                            llvm::remove_cvref_t<From>>::value));
}

} // namespace
