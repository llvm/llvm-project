//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the RandomState class, a fast and
/// lightweight pseudo-random number generator.
///
/// The implementation is based on the xorshift* generator, seeded using the
/// SplitMix64 generator for robust initialization. For more details on the
/// algorithm, see: https://en.wikipedia.org/wiki/Xorshift
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_RANDOMSTATE_HPP
#define MATHTEST_RANDOMSTATE_HPP

#include <cstdint>

struct SeedTy {
  uint64_t Value;
};

class [[nodiscard]] RandomState {
  uint64_t State;

  [[nodiscard]] static constexpr uint64_t splitMix64(uint64_t X) noexcept {
    X += 0x9E3779B97F4A7C15ULL;
    X = (X ^ (X >> 30)) * 0xBF58476D1CE4E5B9ULL;
    X = (X ^ (X >> 27)) * 0x94D049BB133111EBULL;
    X = (X ^ (X >> 31));
    return X ? X : 0x9E3779B97F4A7C15ULL;
  }

public:
  explicit constexpr RandomState(SeedTy Seed) noexcept
      : State(splitMix64(Seed.Value)) {}

  inline uint64_t next() noexcept {
    uint64_t X = State;
    X ^= X >> 12;
    X ^= X << 25;
    X ^= X >> 27;
    State = X;
    return X * 0x2545F4914F6CDD1DULL;
  }
};

#endif // MATHTEST_RANDOMSTATE_HPP
