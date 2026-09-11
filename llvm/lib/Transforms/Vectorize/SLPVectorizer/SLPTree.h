//===- SLPTree.h - SLP vectorization graph (BoUpSLP) ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header for the SLP vectorization graph. The BoUpSLP class and its
// nested types are added here in a follow-up change; this change seeds the
// header with the small helpers BoUpSLP depends on.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTREE_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTREE_H

#include "llvm/IR/ValueHandle.h"

namespace llvm::slpvectorizer {

/// If the ScheduleRegionSizeBudget is exhausted, we allow small scheduling
/// regions to be handled.
static const int MinScheduleRegionSize = 16;

/// A vectorized part of a split reduction, combined into the final reduction
/// result by the horizontal reduction emitter.
struct ReductionVectorPart {
  /// The vectorized value, tracked in case it is replaced while other parts
  /// are vectorized.
  WeakTrackingVH Vec;
  /// The number of times each lane is repeated in the reduction (emitted as a
  /// multiplication by the scale for add/fadd reductions).
  unsigned Scale = 1;
  /// Signedness of \p Vec for reductions, operating on truncated types.
  bool IsSigned = false;
  /// True if the value was already reduced in-tree.
  bool ReducedInTree = false;
  /// True if the part contribution is subtracted from (rather than added to)
  /// the final reduction result. Used for reassociated fadd reductions,
  /// flattened through fsub/fneg operations.
  bool Negated = false;
};

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPTREE_H
