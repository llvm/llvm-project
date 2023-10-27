//===-- Mem2Reg.h - Mem2Reg definitions -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_MEM2REG_H
#define MLIR_TRANSFORMS_MEM2REG_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "llvm/ADT/Statistic.h"

namespace mlir {

/// Statistics collected while applying mem2reg.
struct Mem2RegStatistics {
  /// Total amount of memory slots promoted.
  llvm::Statistic *promotedAmount = nullptr;
  /// Total amount of new block arguments inserted in blocks.
  llvm::Statistic *newBlockArgumentAmount = nullptr;
};

/// Pattern applying mem2reg to the regions of the operations on which it
/// matches.
class Mem2RegPattern
    : public OpInterfaceRewritePattern<PromotableAllocationOpInterface> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  Mem2RegPattern(MLIRContext *context, Mem2RegStatistics statistics = {},
                 PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), statistics(statistics) {}

  LogicalResult matchAndRewrite(PromotableAllocationOpInterface allocator,
                                PatternRewriter &rewriter) const override;

private:
  Mem2RegStatistics statistics;
};

/// Attempts to promote the memory slots of the provided allocators. Succeeds if
/// at least one memory slot was promoted.
LogicalResult
tryToPromoteMemorySlots(ArrayRef<PromotableAllocationOpInterface> allocators,
                        RewriterBase &rewriter,
                        Mem2RegStatistics statistics = {});

} // namespace mlir

#endif // MLIR_TRANSFORMS_MEM2REG_H
