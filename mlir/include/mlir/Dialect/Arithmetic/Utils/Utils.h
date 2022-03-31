//===- Utils.h - General Arithmetic transformation utilities ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the Arithmetic dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITHMETIC_UTILS_UTILS_H
#define MLIR_DIALECT_ARITHMETIC_UTILS_UTILS_H

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {

/// Matches a ConstantIndexOp.
detail::op_matcher<arith::ConstantIndexOp> matchConstantIndex();

/// Detects the `values` produced by a ConstantIndexOp and places the new
/// constant in place of the corresponding sentinel value.
void canonicalizeSubViewPart(SmallVectorImpl<OpFoldResult> &values,
                             function_ref<bool(int64_t)> isDynamic);

llvm::SmallBitVector getPositionsOfShapeOne(unsigned rank,
                                            ArrayRef<int64_t> shape);

/// Pattern to rewrite a subview op with constant arguments.
template <typename OpType, typename ResultTypeFunc, typename CastOpFunc>
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
    ResultTypeFunc resultTypeFunc;
    auto resultType =
        resultTypeFunc(op, mixedOffsets, mixedSizes, mixedStrides);
    if (!resultType)
      return failure();
    auto newOp =
        rewriter.create<OpType>(op.getLoc(), resultType, op.source(),
                                mixedOffsets, mixedSizes, mixedStrides);
    CastOpFunc func;
    func(rewriter, op, newOp);

    return success();
  }
};

/// Converts an OpFoldResult to a Value. Returns the fold result if it casts to
/// a Value or creates a ConstantIndexOp if it casts to an IntegerAttribute.
/// Other attribute types are not supported.
Value getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                      OpFoldResult ofr);

/// Create a cast from an index-like value (index or integer) to another
/// index-like value. If the value type and the target type are the same, it
/// returns the original value.
Value getValueOrCreateCastToIndexLike(OpBuilder &b, Location loc,
                                      Type targetType, Value value);

/// Similar to the other overload, but converts multiple OpFoldResults into
/// Values.
SmallVector<Value>
getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                ArrayRef<OpFoldResult> valueOrAttrVec);

/// Helper struct to build simple arithmetic quantities with minimal type
/// inference support.
struct ArithBuilder {
  ArithBuilder(OpBuilder &b, Location loc) : b(b), loc(loc) {}

  Value _and(Value lhs, Value rhs);
  Value add(Value lhs, Value rhs);
  Value mul(Value lhs, Value rhs);
  Value select(Value cmp, Value lhs, Value rhs);
  Value sgt(Value lhs, Value rhs);
  Value slt(Value lhs, Value rhs);

private:
  OpBuilder &b;
  Location loc;
};
} // namespace mlir

#endif // MLIR_DIALECT_ARITHMETIC_UTILS_UTILS_H
