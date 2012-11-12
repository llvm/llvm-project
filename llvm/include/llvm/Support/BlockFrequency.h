//===-------- BlockFrequency.h - Block Frequency Wrapper --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Block Frequency class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_BLOCKFREQUENCY_H
#define LLVM_SUPPORT_BLOCKFREQUENCY_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

class raw_ostream;
class BranchProbability;

// This class represents Block Frequency as a 64-bit value.
class BlockFrequency {

  uint64_t Frequency;
  static const int64_t ENTRY_FREQ = 1024;

public:
  BlockFrequency(uint64_t Freq = 0) : Frequency(Freq) { }

  static uint64_t getEntryFrequency() { return ENTRY_FREQ; }
  uint64_t getFrequency() const { return Frequency; }

  BlockFrequency &operator*=(const BranchProbability &Prob);
  const BlockFrequency operator*(const BranchProbability &Prob) const;

  BlockFrequency &operator+=(const BlockFrequency &Freq);
  const BlockFrequency operator+(const BlockFrequency &Freq) const;

  bool operator<(const BlockFrequency &RHS) const {
    return Frequency < RHS.Frequency;
  }

  bool operator<=(const BlockFrequency &RHS) const {
    return Frequency <= RHS.Frequency;
  }

  bool operator>(const BlockFrequency &RHS) const {
    return Frequency > RHS.Frequency;
  }

  bool operator>=(const BlockFrequency &RHS) const {
    return Frequency >= RHS.Frequency;
  }

  void print(raw_ostream &OS) const;
};

raw_ostream &operator<<(raw_ostream &OS, const BlockFrequency &Freq);

}

#endif
