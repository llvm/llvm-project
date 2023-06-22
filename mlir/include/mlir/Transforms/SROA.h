//===-- SROA.h - Scalar Replacement Of Aggregates ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SROA_H
#define MLIR_TRANSFORMS_SROA_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Statistic.h"
#include <variant>

namespace mlir {

/// Statistics collected while applying SROA.
struct SROAStatistics {
  /// Total amount of memory slots destructured.
  llvm::Statistic *destructuredAmount = nullptr;
  /// Total amount of memory slots in which the destructured size was smaller
  /// than the total size after eliminating unused fields.
  llvm::Statistic *slotsWithMemoryBenefit = nullptr;
  /// Maximal number of sub-elements a successfully destructured slot initially
  /// had.
  llvm::Statistic *maxSubelementAmount = nullptr;
};

/// Pattern applying SROA to the regions of the operations on which it
/// matches.
class SROAPattern
    : public OpInterfaceRewritePattern<DestructurableAllocationOpInterface> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  SROAPattern(MLIRContext *context, SROAStatistics statistics = {},
              PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), statistics(statistics) {}

  LogicalResult matchAndRewrite(DestructurableAllocationOpInterface allocator,
                                PatternRewriter &rewriter) const override;

private:
  SROAStatistics statistics;
};

/// Attempts to destructure the slots of destructurable allocators. Returns
/// failure if no slot was destructured.
LogicalResult tryToDestructureMemorySlots(
    ArrayRef<DestructurableAllocationOpInterface> allocators,
    RewriterBase &rewriter, SROAStatistics statistics = {});

} // namespace mlir

#endif // MLIR_TRANSFORMS_SROA_H
