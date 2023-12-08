//===- llvm/unittest/Support/KnownBitsTest.h - KnownBits tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helpers for KnownBits and DemandedBits unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_SUPPORT_KNOWNBITSTEST_H
#define LLVM_UNITTESTS_SUPPORT_KNOWNBITSTEST_H

#include "llvm/Support/KnownBits.h"

namespace {

using namespace llvm;

template <typename FnTy> void ForeachKnownBits(unsigned Bits, FnTy Fn) {
  unsigned Max = 1 << Bits;
  KnownBits Known(Bits);
  for (unsigned Zero = 0; Zero < Max; ++Zero) {
    for (unsigned One = 0; One < Max; ++One) {
      Known.Zero = Zero;
      Known.One = One;
      if (Known.hasConflict())
        continue;

      Fn(Known);
    }
  }
}

template <typename FnTy>
void ForeachNumInKnownBits(const KnownBits &Known, FnTy Fn) {
  unsigned Bits = Known.getBitWidth();
  unsigned Max = 1 << Bits;
  for (unsigned N = 0; N < Max; ++N) {
    APInt Num(Bits, N);
    if ((Num & Known.Zero) != 0 || (~Num & Known.One) != 0)
      continue;

    Fn(Num);
  }
}

} // end anonymous namespace

#endif
