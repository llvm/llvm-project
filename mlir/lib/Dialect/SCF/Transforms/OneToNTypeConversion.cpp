//===-- OneToNTypeConversion.cpp - SCF 1:N type conversion ------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The patterns in this file are heavily inspired (and copied from)
// lib/Dialect/SCF/Transforms/StructuralTypeConversions.cpp but work for 1:N
// type conversions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Transforms.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

using namespace mlir;
using namespace mlir::scf;

class ConvertTypesInSCFIfOp : public OneToNOpConversionPattern<IfOp> {
public:
  using OneToNOpConversionPattern<IfOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();

    // Nothing to do if there is no non-identity conversion.
    if (!resultMapping.hasNonIdentityConversion())
      return failure();

    // Create new IfOp.
    TypeRange convertedResultTypes = resultMapping.getConvertedTypes();
    auto newOp = rewriter.create<IfOp>(loc, convertedResultTypes,
                                       op.getCondition(), true);
    newOp->setAttrs(op->getAttrs());

    // We do not need the empty blocks created by rewriter.
    rewriter.eraseBlock(newOp.elseBlock());
    rewriter.eraseBlock(newOp.thenBlock());

    // Inlines block from the original operation.
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
    return success();
  }
};

class ConvertTypesInSCFWhileOp : public OneToNOpConversionPattern<WhileOp> {
public:
  using OneToNOpConversionPattern<WhileOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(WhileOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    const OneToNTypeMapping &operandMapping = adaptor.getOperandMapping();
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();

    // Nothing to do if the op doesn't have any non-identity conversions for its
    // operands or results.
    if (!operandMapping.hasNonIdentityConversion() &&
        !resultMapping.hasNonIdentityConversion())
      return failure();

    // Create new WhileOp.
    TypeRange convertedResultTypes = resultMapping.getConvertedTypes();

    auto newOp = rewriter.create<WhileOp>(loc, convertedResultTypes,
                                          adaptor.getFlatOperands());
    newOp->setAttrs(op->getAttrs());

    // Update block signatures.
    std::array<OneToNTypeMapping, 2> blockMappings = {operandMapping,
                                                      resultMapping};
    for (unsigned int i : {0u, 1u}) {
      Region *region = &op.getRegion(i);
      Block *block = &region->front();

      rewriter.applySignatureConversion(block, blockMappings[i]);

      // Move updated region to new WhileOp.
      Region &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
    }

    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
    return success();
  }
};

class ConvertTypesInSCFYieldOp : public OneToNOpConversionPattern<YieldOp> {
public:
  using OneToNOpConversionPattern<YieldOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Nothing to do if there is no non-identity conversion.
    if (!adaptor.getOperandMapping().hasNonIdentityConversion())
      return failure();

    // Convert operands.
    rewriter.modifyOpInPlace(
        op, [&] { op->setOperands(adaptor.getFlatOperands()); });

    return success();
  }
};

class ConvertTypesInSCFConditionOp
    : public OneToNOpConversionPattern<ConditionOp> {
public:
  using OneToNOpConversionPattern<ConditionOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(ConditionOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Nothing to do if there is no non-identity conversion.
    if (!adaptor.getOperandMapping().hasNonIdentityConversion())
      return failure();

    // Convert operands.
    rewriter.modifyOpInPlace(
        op, [&] { op->setOperands(adaptor.getFlatOperands()); });

    return success();
  }
};

class ConvertTypesInSCFForOp final : public OneToNOpConversionPattern<ForOp> {
public:
  using OneToNOpConversionPattern<ForOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(ForOp forOp, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    const OneToNTypeMapping &operandMapping = adaptor.getOperandMapping();
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();

    // Nothing to do if there is no non-identity conversion.
    if (!operandMapping.hasNonIdentityConversion() &&
        !resultMapping.hasNonIdentityConversion())
      return failure();

    // If the lower-bound, upper-bound, or step were expanded, abort the
    // conversion. This conversion does not know what to do in such cases.
    ValueRange lbs = adaptor.getLowerBound();
    ValueRange ubs = adaptor.getUpperBound();
    ValueRange steps = adaptor.getStep();
    if (lbs.size() != 1 || ubs.size() != 1 || steps.size() != 1)
      return rewriter.notifyMatchFailure(
          forOp, "index operands converted to multiple values");

    Location loc = forOp.getLoc();

    Region *region = &forOp.getRegion();
    Block *block = &region->front();

    // Construct the new for-op with an empty body.
    ValueRange newInits = adaptor.getFlatOperands().drop_front(3);
    auto newOp =
        rewriter.create<ForOp>(loc, lbs[0], ubs[0], steps[0], newInits);
    newOp->setAttrs(forOp->getAttrs());

    // We do not need the empty blocks created by rewriter.
    rewriter.eraseBlock(newOp.getBody());

    // Convert the signature of the body region.
    OneToNTypeMapping bodyTypeMapping(block->getArgumentTypes());
    if (failed(typeConverter->convertSignatureArgs(block->getArgumentTypes(),
                                                   bodyTypeMapping)))
      return failure();

    // Perform signature conversion on the body block.
    rewriter.applySignatureConversion(block, bodyTypeMapping);

    // Splice the old body region into the new for-op.
    Region &dstRegion = newOp.getBodyRegion();
    rewriter.inlineRegionBefore(forOp.getRegion(), dstRegion, dstRegion.end());

    rewriter.replaceOp(forOp, newOp.getResults(), resultMapping);

    return success();
  }
};

namespace mlir {
namespace scf {

void populateSCFStructuralOneToNTypeConversions(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertTypesInSCFConditionOp,
      ConvertTypesInSCFForOp,
      ConvertTypesInSCFIfOp,
      ConvertTypesInSCFWhileOp,
      ConvertTypesInSCFYieldOp
      // clang-format on
      >(typeConverter, patterns.getContext());
}

} // namespace scf
} // namespace mlir
