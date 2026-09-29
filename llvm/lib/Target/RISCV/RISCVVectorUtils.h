//===-- RISCVVectorUtils.h - RISC-V vector utilities ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines RISC-V vector utilities shared across the backend,
// including helpers for recognizing shuffle masks supported by vector
// permutation instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVVECTORUTILS_H
#define LLVM_LIB_TARGET_RISCV_RISCVVECTORUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {

inline bool isAlternating(const std::array<std::pair<int, int>, 2> &SrcInfo,
                          ArrayRef<int> Mask, unsigned Factor,
                          bool RequiredPolarity) {
  int NumElts = Mask.size();
  for (const auto &[Idx, M] : enumerate(Mask)) {
    if (M < 0)
      continue;
    int Src = M >= NumElts;
    int Diff = (int)Idx - (M % NumElts);
    bool C = Src == SrcInfo[1].first && Diff == SrcInfo[1].second;
    assert(C != (Src == SrcInfo[0].first && Diff == SrcInfo[0].second) &&
           "Must match exactly one of the two slides");
    if (RequiredPolarity != (C == (Idx / Factor) % 2))
      return false;
  }
  return true;
}

/// Given a shuffle which can be represented as a pair of two slides,
/// see if it is a pair-even idiom.
/// Pair-even is:
/// vs2: a0 a1 a2 a3
/// vs1: b0 b1 b2 b3
/// vd:  a0 b0 a2 b2
inline bool isPairEven(const std::array<std::pair<int, int>, 2> &SrcInfo,
                       ArrayRef<int> Mask, unsigned &Factor) {
  Factor = SrcInfo[1].second;
  return SrcInfo[0].second == 0 && isPowerOf2_32(Factor) &&
         Mask.size() % Factor == 0 &&
         isAlternating(SrcInfo, Mask, Factor, true);
}

/// Given a shuffle which can be represented as a pair of two slides,
/// see if it is a pair-odd idiom.
/// Pair-odd is:
/// vs2: a0 a1 a2 a3
/// vs1: b0 b1 b2 b3
/// vd:  a1 b1 a3 b3
/// Note that the operand order is swapped due to the way we canonicalize
/// the slides, so SrCInfo[0] is vs1, and SrcInfo[1] is vs2.
inline bool isPairOdd(const std::array<std::pair<int, int>, 2> &SrcInfo,
                      ArrayRef<int> Mask, unsigned &Factor) {
  Factor = -SrcInfo[1].second;
  return SrcInfo[0].second == 0 && isPowerOf2_32(Factor) &&
         Mask.size() % Factor == 0 &&
         isAlternating(SrcInfo, Mask, Factor, false);
}
} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVVECTORUTILS_H
