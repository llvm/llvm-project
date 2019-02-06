//===- ReservoirSampler.cpp - Tests for the ReservoirSampler --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/Random.h"
#include "gtest/gtest.h"
#include <random>

using namespace llvm;

TEST(ReservoirSamplerTest, OneItem) {
  std::mt19937 Rand;
  auto Sampler = makeSampler(Rand, 7, 1);
  ASSERT_FALSE(Sampler.isEmpty());
  ASSERT_EQ(7, Sampler.getSelection());
}

TEST(ReservoirSamplerTest, NoWeight) {
  std::mt19937 Rand;
  auto Sampler = makeSampler(Rand, 7, 0);
  ASSERT_TRUE(Sampler.isEmpty());
}

TEST(ReservoirSamplerTest, Uniform) {
  std::mt19937 Rand;

  // Run three chi-squared tests to check that the distribution is reasonably
  // uniform.
  std::vector<int> Items = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  int Failures = 0;
  for (int Run = 0; Run < 3; ++Run) {
    std::vector<int> Counts(Items.size(), 0);

    // We need $np_s > 5$ at minimum, but we're better off going a couple of
    // orders of magnitude larger.
    int N = Items.size() * 5 * 100;
    for (int I = 0; I < N; ++I) {
      auto Sampler = makeSampler(Rand, Items);
      Counts[Sampler.getSelection()] += 1;
    }

    // Knuth. TAOCP Vol. 2, 3.3.1 (8):
    // $V = \frac{1}{n} \sum_{s=1}^{k} \left(\frac{Y_s^2}{p_s}\right) - n$
    double Ps = 1.0 / Items.size();
    double Sum = 0.0;
    for (int Ys : Counts)
      Sum += Ys * Ys / Ps;
    double V = (Sum / N) - N;

    assert(Items.size() == 10 && "Our chi-squared values assume 10 items");
    // Since we have 10 items, there are 9 degrees of freedom and the table of
    // chi-squared values is as follows:
    //
    //     | p=1%  |   5%  |  25%  |  50%  |  75%  |  95%  |  99%  |
    // v=9 | 2.088 | 3.325 | 5.899 | 8.343 | 11.39 | 16.92 | 21.67 |
    //
    // Check that we're in the likely range of results.
    //if (V < 2.088 || V > 21.67)
    if (V < 2.088 || V > 21.67)
      ++Failures;
  }
  EXPECT_LT(Failures, 3) << "Non-uniform distribution?";
}
