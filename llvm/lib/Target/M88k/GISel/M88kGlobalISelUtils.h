//===- M88kGlobalISelUtils.h -------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file APIs for M88k-specific helper functions used in the GlobalISel
/// pipeline.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_GISEL_M88KGLOBALISELUTILS_H
#define LLVM_LIB_TARGET_AARCH64_GISEL_M88KGLOBALISELUTILS_H

#include "llvm/Support/MathExtras.h"

namespace llvm {

namespace M88kGISelUtils {

// If I is a shifted mask, set the size (Width) and the first bit of the
// mask (Offset), and return true.
// For example, if I is 0x003e, then sez (Width, Offset) = (5, 1).
inline bool isShiftedMask(uint64_t I, uint64_t &Width, uint64_t &Offset) {
  if (!isShiftedMask_64(I))
    return false;

  Width = countPopulation(I);
  Offset = countTrailingZeros(I);
  return true;
}

}
} // namespace llvm

#endif