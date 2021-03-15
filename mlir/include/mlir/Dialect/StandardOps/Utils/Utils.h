//===- Utils.h - General transformation utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the StandardOps dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARDOPS_UTILS_UTILS_H
#define MLIR_DIALECT_STANDARDOPS_UTILS_UTILS_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {

/// Matches a ConstantIndexOp.
detail::op_matcher<ConstantIndexOp> matchConstantIndex();

/// Detects the `values` produced by a ConstantIndexOp and places the new
/// constant in place of the corresponding sentinel value.
void canonicalizeSubViewPart(SmallVectorImpl<OpFoldResult> &values,
                             function_ref<bool(int64_t)> isDynamic);

void getPositionsOfShapeOne(unsigned rank, ArrayRef<int64_t> shape,
                            llvm::SmallDenseSet<unsigned> &dimsToProject);

/// Pattern to rewrite a subview op with constant arguments.
template <typename OpType, typename CastOpFunc>
class OpWithOffsetSizesAndStridesConstantArgumentFolder final
    : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    // No constant operand, just return;
    if (llvm::none_of(op.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    // At least one of offsets/sizes/strides is a new constant.
    // Form the new list of operands and constant attributes from the existing.
    SmallVector<OpFoldResult> mixedOffsets(op.getMixedOffsets());
    SmallVector<OpFoldResult> mixedSizes(op.getMixedSizes());
    SmallVector<OpFoldResult> mixedStrides(op.getMixedStrides());
    canonicalizeSubViewPart(mixedOffsets, ShapedType::isDynamicStrideOrOffset);
    canonicalizeSubViewPart(mixedSizes, ShapedType::isDynamic);
    canonicalizeSubViewPart(mixedStrides, ShapedType::isDynamicStrideOrOffset);

    // Create the new op in canonical form.
    auto newOp = rewriter.create<OpType>(op.getLoc(), op.source(), mixedOffsets,
                                         mixedSizes, mixedStrides);
    CastOpFunc func;
    func(rewriter, op, newOp);

    return success();
  }
};

} // end namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_UTILS_UTILS_H
