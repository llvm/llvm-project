//===- llvm/unittest/ADT/CombinationGeneratorTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/CombinationGenerator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

using namespace llvm;

namespace {

TEST(CombinationGenerator, Square) {
  const std::vector<std::vector<int>> Choices{{0, 1}, {2, 3}};

  std::vector<std::vector<int>> Variants;
  CombinationGenerator<int, std::vector<int>, 4> G(Choices);
  const size_t NumVariants = G.numCombinations();
  G.generate([&](ArrayRef<int> State) -> bool {
    Variants.emplace_back(State);
    return false; // keep going
  });

  const std::vector<std::vector<int>> ExpectedVariants{
      {0, 2},
      {0, 3},
      {1, 2},
      {1, 3},
  };
  ASSERT_THAT(Variants, ::testing::SizeIs(NumVariants));
  ASSERT_THAT(Variants, ::testing::ContainerEq(ExpectedVariants));
}

TEST(CombinationGenerator, MiddleColumn) {
  const std::vector<std::vector<int>> Choices{{0}, {1, 2}, {3}};

  std::vector<std::vector<int>> Variants;
  CombinationGenerator<int, std::vector<int>, 4> G(Choices);
  const size_t NumVariants = G.numCombinations();
  G.generate([&](ArrayRef<int> State) -> bool {
    Variants.emplace_back(State);
    return false; // keep going
  });

  const std::vector<std::vector<int>> ExpectedVariants{
      {0, 1, 3},
      {0, 2, 3},
  };
  ASSERT_THAT(Variants, ::testing::SizeIs(NumVariants));
  ASSERT_THAT(Variants, ::testing::ContainerEq(ExpectedVariants));
}

TEST(CombinationGenerator, SideColumns) {
  const std::vector<std::vector<int>> Choices{{0, 1}, {2}, {3, 4}};

  std::vector<std::vector<int>> Variants;
  CombinationGenerator<int, std::vector<int>, 4> G(Choices);
  const size_t NumVariants = G.numCombinations();
  G.generate([&](ArrayRef<int> State) -> bool {
    Variants.emplace_back(State);
    return false; // keep going
  });

  const std::vector<std::vector<int>> ExpectedVariants{
      {0, 2, 3},
      {0, 2, 4},
      {1, 2, 3},
      {1, 2, 4},
  };
  ASSERT_THAT(Variants, ::testing::SizeIs(NumVariants));
  ASSERT_THAT(Variants, ::testing::ContainerEq(ExpectedVariants));
}

TEST(CombinationGenerator, LeftColumn) {
  const std::vector<std::vector<int>> Choices{{0, 1}, {2}};

  std::vector<std::vector<int>> Variants;
  CombinationGenerator<int, std::vector<int>, 4> G(Choices);
  const size_t NumVariants = G.numCombinations();
  G.generate([&](ArrayRef<int> State) -> bool {
    Variants.emplace_back(State);
    return false; // keep going
  });

  const std::vector<std::vector<int>> ExpectedVariants{
      {0, 2},
      {1, 2},
  };
  ASSERT_THAT(Variants, ::testing::SizeIs(NumVariants));
  ASSERT_THAT(Variants, ::testing::ContainerEq(ExpectedVariants));
}

TEST(CombinationGenerator, RightColumn) {
  const std::vector<std::vector<int>> Choices{{0}, {1, 2}};

  std::vector<std::vector<int>> Variants;
  CombinationGenerator<int, std::vector<int>, 4> G(Choices);
  const size_t NumVariants = G.numCombinations();
  G.generate([&](ArrayRef<int> State) -> bool {
    Variants.emplace_back(State);
    return false; // keep going
  });

  const std::vector<std::vector<int>> ExpectedVariants{
      {0, 1},
      {0, 2},
  };
  ASSERT_THAT(Variants, ::testing::SizeIs(NumVariants));
  ASSERT_THAT(Variants, ::testing::ContainerEq(ExpectedVariants));
}

TEST(CombinationGenerator, Column) {
  const std::vector<std::vector<int>> Choices{{0, 1}};

  std::vector<std::vector<int>> Variants;
  CombinationGenerator<int, std::vector<int>, 4> G(Choices);
  const size_t NumVariants = G.numCombinations();
  G.generate([&](ArrayRef<int> State) -> bool {
    Variants.emplace_back(State);
    return false; // keep going
  });

  const std::vector<std::vector<int>> ExpectedVariants{
      {0},
      {1},
  };
  ASSERT_THAT(Variants, ::testing::SizeIs(NumVariants));
  ASSERT_THAT(Variants, ::testing::ContainerEq(ExpectedVariants));
}

TEST(CombinationGenerator, Row) {
  const std::vector<std::vector<int>> Choices{{0}, {1}};

  std::vector<std::vector<int>> Variants;
  CombinationGenerator<int, std::vector<int>, 4> G(Choices);
  const size_t NumVariants = G.numCombinations();
  G.generate([&](ArrayRef<int> State) -> bool {
    Variants.emplace_back(State);
    return false; // keep going
  });

  const std::vector<std::vector<int>> ExpectedVariants{
      {0, 1},
  };
  ASSERT_THAT(Variants, ::testing::SizeIs(NumVariants));
  ASSERT_THAT(Variants, ::testing::ContainerEq(ExpectedVariants));
}

TEST(CombinationGenerator, Singleton) {
  const std::vector<std::vector<int>> Choices{{0}};

  std::vector<std::vector<int>> Variants;
  CombinationGenerator<int, std::vector<int>, 4> G(Choices);
  const size_t NumVariants = G.numCombinations();
  G.generate([&](ArrayRef<int> State) -> bool {
    Variants.emplace_back(State);
    return false; // keep going
  });

  const std::vector<std::vector<int>> ExpectedVariants{
      {0},
  };
  ASSERT_THAT(Variants, ::testing::SizeIs(NumVariants));
  ASSERT_THAT(Variants, ::testing::ContainerEq(ExpectedVariants));
}

} // end anonymous namespace
