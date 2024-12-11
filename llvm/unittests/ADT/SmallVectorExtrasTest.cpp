//===- llvm/unittest/ADT/SmallVectorExtrasTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SmallVectorExtras unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVectorExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <type_traits>
#include <vector>

using testing::ElementsAre;

namespace llvm {
namespace {

TEST(SmallVectorExtrasTest, FilterToVector) {
  std::vector<int> Numbers = {0, 1, 2, 3, 4};
  auto Odd = filter_to_vector<2>(Numbers, [](int X) { return (X % 2) != 0; });
  static_assert(std::is_same_v<decltype(Odd), SmallVector<int, 2>>);
  EXPECT_THAT(Odd, ElementsAre(1, 3));

  auto Even = filter_to_vector(Numbers, [](int X) { return (X % 2) == 0; });
  static_assert(std::is_same_v<decltype(Even), SmallVector<int>>);
  EXPECT_THAT(Even, ElementsAre(0, 2, 4));
}

} // end namespace
} // namespace llvm
