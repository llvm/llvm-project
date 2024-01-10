//===-- Utils/ExponentialBackoff.h - Heuristic helper class ------*- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement exponential backoff counting.
// Linearly increments until given maximum, exponentially decrements based on
// given backoff factor.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_UTILS_EXPONENTIAL_BACKOFF_H
#define OMPTARGET_UTILS_EXPONENTIAL_BACKOFF_H

#include <cassert>
#include <cmath>
#include <cstdint>

namespace utils {

class ExponentialBackoff {
  int64_t Count = 0;
  const int64_t MaxCount = 0;
  const int64_t CountThreshold = 0;
  const double BackoffFactor = 0;

public:
  ExponentialBackoff(int64_t MaxCount, int64_t CountThreshold,
                     double BackoffFactor)
      : MaxCount(MaxCount), CountThreshold(CountThreshold),
        BackoffFactor(BackoffFactor) {
    assert(MaxCount >= 0 &&
           "ExponentialBackoff: maximum count value should be non-negative");
    assert(CountThreshold >= 0 &&
           "ExponentialBackoff: count threshold value should be non-negative");
    assert(BackoffFactor >= 0 && BackoffFactor < 1 &&
           "ExponentialBackoff: backoff factor should be in [0, 1) interval");
  }

  void increment() { Count = std::min(Count + 1, MaxCount); }

  void decrement() { Count *= BackoffFactor; }

  bool isAboveThreshold() const { return Count > CountThreshold; }
};

} // namespace utils

#endif // OMPTARGET_UTILS_EXPONENTIAL_BACKOFF_H
