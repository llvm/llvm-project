//===-- X86ShuffleUtils.h -----------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares utility functions used in X86 shuffles
/// It is designed for X86ISelowering and X86LegalizerInfo, intends to factor
/// out common code without DAG and GIR dependencies.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86SHUFFLEUTILS_H
#define LLVM_LIB_TARGET_X86_X86SHUFFLEUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace x86shufutils {

using namespace llvm;

// This function checks if the shuffle mask has indices that are out of bounds
// for the vector type.
static inline bool hasMaskIndexOutOfBounds(const ArrayRef<int> Mask,
                                           const unsigned NumElts) {
  return llvm::any_of(Mask, [NumElts](int M) { return M >= (int)NumElts; });
}

// This function canonicalizes masks with out-of-bounds indices by setting
// indices to -1 for all indices which are out of bounds in vector type.
static inline void getCanonicalizeMask(const ArrayRef<int> Mask,
                                       const unsigned NumElts,
                                       SmallVectorImpl<int> &NewMask) {
  for (int M : Mask) {
    if (M >= (int)NumElts)
      NewMask.push_back(-1);
    else
      NewMask.push_back(M);
  }
}

// This function validates mask for legal values.
// If source 2 is undef, max legal index is size of mask vector
// else max legal index is size of mask vector * 2 since shuffle can select
// from both source vectors
static inline bool isLegalMask(const ArrayRef<int> Mask, const unsigned NumElts,
                               const unsigned MaskUpperLimit) {
  return llvm::all_of(
      Mask, [&](int M) { return -1 <= M && (unsigned)M < MaskUpperLimit; });
}
} // namespace x86shufutils

#endif // LLVM_LIB_TARGET_X86_X86SHUFFLEUTILS_H