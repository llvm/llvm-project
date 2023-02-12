//===-------- BlockFrequency.h - Block Frequency Wrapper --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Block Frequency class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_BLOCKFREQUENCY_H
#define LLVM_SUPPORT_BLOCKFREQUENCY_H

#include <cstdint>

namespace llvm {

class BranchProbability;

// This class represents Block Frequency as a 64-bit value.
class BlockFrequency {
  uint64_t Frequency;

public:
  BlockFrequency(uint64_t Freq = 0) : Frequency(Freq) { }

  /// Returns the maximum possible frequency, the saturation value.
  static uint64_t getMaxFrequency() { return -1ULL; }

  /// Returns the frequency as a fixpoint number scaled by the entry
  /// frequency.
  uint64_t getFrequency() const { return Frequency; }

  /// Multiplies with a branch probability. The computation will never
  /// overflow.
  BlockFrequency &operator*=(BranchProbability Prob);
  BlockFrequency operator*(BranchProbability Prob) const;

  /// Divide by a non-zero branch probability using saturating
  /// arithmetic.
  BlockFrequency &operator/=(BranchProbability Prob);
  BlockFrequency operator/(BranchProbability Prob) const;

  /// Adds another block frequency using saturating arithmetic.
  BlockFrequency &operator+=(BlockFrequency Freq);
  BlockFrequency operator+(BlockFrequency Freq) const;

  /// Subtracts another block frequency using saturating arithmetic.
  BlockFrequency &operator-=(BlockFrequency Freq);
  BlockFrequency operator-(BlockFrequency Freq) const;

  /// Shift block frequency to the right by count digits saturating to 1.
  BlockFrequency &operator>>=(const unsigned count);

  bool operator<(BlockFrequency RHS) const {
    return Frequency < RHS.Frequency;
  }

  bool operator<=(BlockFrequency RHS) const {
    return Frequency <= RHS.Frequency;
  }

  bool operator>(BlockFrequency RHS) const {
    return Frequency > RHS.Frequency;
  }

  bool operator>=(BlockFrequency RHS) const {
    return Frequency >= RHS.Frequency;
  }

  bool operator==(BlockFrequency RHS) const {
    return Frequency == RHS.Frequency;
  }
};

} // namespace llvm

#endif
