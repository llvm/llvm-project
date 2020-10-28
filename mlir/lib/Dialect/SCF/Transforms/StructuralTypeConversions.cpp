//===- StructuralTypeConversions.cpp - scf structural type conversions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::scf;

namespace {
class ConvertForOpTypes : public OpConversionPattern<ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ForOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 6> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // Clone and replace.
    ForOp newOp = cast<ForOp>(rewriter.clone(*op.getOperation()));
    newOp.getOperation()->setOperands(operands);
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    auto bodyArgs = newOp.getBody()->getArguments();
    for (auto t : llvm::zip(llvm::drop_begin(bodyArgs, 1), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());

    return success();
  }
};
} // namespace

namespace {
class ConvertIfOpTypes : public OpConversionPattern<IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Generalize this to any type conversion, not just 1:1.
    //
    // We need to implement something more sophisticated here that tracks which
    // types convert to which other types and does the appropriate
    // materialization logic.
    // For example, it's possible that one result type converts to 0 types and
    // another to 2 types, so newResultTypes would at least be the right size to
    // not crash in the llvm::zip call below, but then we would set the the
    // wrong type on the SSA values! These edge cases are also why we cannot
    // safely use the TypeConverter::convertTypes helper here.
    SmallVector<Type, 6> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // TODO: Write this with updateRootInPlace once the conversion infra
    // supports source materializations on ops updated in place.
    IfOp newOp = cast<IfOp>(rewriter.clone(*op.getOperation()));
    newOp.getOperation()->setOperands(operands);
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};
} // namespace

namespace {
// When the result types of a ForOp/IfOp get changed, the operand types of the
// corresponding yield op need to be changed. In order to trigger the
// appropriate type conversions / materializations, we need a dummy pattern.
class ConvertYieldOpTypes : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::YieldOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
    return success();
  }
};
} // namespace

void mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
    MLIRContext *context, TypeConverter &typeConverter,
    OwningRewritePatternList &patterns, ConversionTarget &target) {
  patterns.insert<ConvertForOpTypes, ConvertIfOpTypes, ConvertYieldOpTypes>(
      typeConverter, context);
  target.addDynamicallyLegalOp<ForOp, IfOp>([&](Operation *op) {
    return typeConverter.isLegal(op->getResultTypes());
  });
  target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
    // We only have conversions for a subset of ops that use scf.yield
    // terminators.
    if (!isa<ForOp, IfOp>(op.getParentOp()))
      return true;
    return typeConverter.isLegal(op.getOperandTypes());
  });
}
