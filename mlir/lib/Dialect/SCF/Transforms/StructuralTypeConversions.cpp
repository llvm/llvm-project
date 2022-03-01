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
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::scf;

namespace {
class ConvertForOpTypes : public OpConversionPattern<ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 6> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // Clone the op without the regions and inline the regions from the old op.
    //
    // This is a little bit tricky. We have two concerns here:
    //
    // 1. We cannot update the op in place because the dialect conversion
    // framework does not track type changes for ops updated in place, so it
    // won't insert appropriate materializations on the changed result types.
    // PR47938 tracks this issue, but it seems hard to fix. Instead, we need to
    // clone the op.
    //
    // 2. We cannot simply call `op.clone()` to get the cloned op. Besides being
    // inefficient to recursively clone the regions, there is a correctness
    // issue: if we clone with the regions, then the dialect conversion
    // framework thinks that we just inserted all the cloned child ops. But what
    // we want is to "take" the child regions and let the dialect conversion
    // framework continue recursively into ops inside those regions (which are
    // already in its worklist; inlining them into the new op's regions doesn't
    // remove the child ops from the worklist).
    ForOp newOp = cast<ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    // Take the region from the old op and put it in the new op.
    rewriter.inlineRegionBefore(op.getLoopBody(), newOp.getLoopBody(),
                                newOp.getLoopBody().end());

    // Now, update all the types.

    // Convert the type of the entry block of the ForOp's body.
    if (failed(rewriter.convertRegionTypes(&newOp.getLoopBody(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    // Change the clone to use the updated operands. We could have cloned with
    // a BlockAndValueMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());
    // Update the result types to the new converted types.
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
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
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
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

    // See comments in the ForOp pattern for why we clone without regions and
    // then inline.
    IfOp newOp = cast<IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    // Update the operands and types.
    newOp->setOperands(adaptor.getOperands());
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
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

namespace {
class ConvertWhileOpTypes : public OpConversionPattern<WhileOp> {
public:
  using OpConversionPattern<WhileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter);
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    auto newOp = rewriter.create<WhileOp>(op.getLoc(), newResultTypes,
                                          adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};
} // namespace

namespace {
class ConvertConditionOpTypes : public OpConversionPattern<ConditionOp> {
public:
  using OpConversionPattern<ConditionOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};
} // namespace

void mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  patterns.add<ConvertForOpTypes, ConvertIfOpTypes, ConvertYieldOpTypes,
               ConvertWhileOpTypes, ConvertConditionOpTypes>(
      typeConverter, patterns.getContext());
  target.addDynamicallyLegalOp<ForOp, IfOp>([&](Operation *op) {
    return typeConverter.isLegal(op->getResultTypes());
  });
  target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
    // We only have conversions for a subset of ops that use scf.yield
    // terminators.
    if (!isa<ForOp, IfOp, WhileOp>(op->getParentOp()))
      return true;
    return typeConverter.isLegal(op.getOperandTypes());
  });
  target.addDynamicallyLegalOp<WhileOp, ConditionOp>(
      [&](Operation *op) { return typeConverter.isLegal(op); });
}
