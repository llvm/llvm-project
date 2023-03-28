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
  matchAndRewrite(IfOp op, OneToNPatternRewriter &rewriter,
                  const OneToNTypeMapping & /*operandMapping*/,
                  const OneToNTypeMapping &resultMapping,
                  const ValueRange /*convertedOperands*/) const override {
    Location loc = op->getLoc();

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

    rewriter.replaceOp(op, SmallVector<Value>(newOp->getResults()),
                       resultMapping);
    return success();
  }
};

class ConvertTypesInSCFWhileOp : public OneToNOpConversionPattern<WhileOp> {
public:
  using OneToNOpConversionPattern<WhileOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(WhileOp op, OneToNPatternRewriter &rewriter,
                  const OneToNTypeMapping &operandMapping,
                  const OneToNTypeMapping &resultMapping,
                  const ValueRange convertedOperands) const override {
    Location loc = op->getLoc();

    // Nothing to do if the op doesn't have any non-identity conversions for its
    // operands or results.
    if (!operandMapping.hasNonIdentityConversion() &&
        !resultMapping.hasNonIdentityConversion())
      return failure();

    // Create new WhileOp.
    TypeRange convertedResultTypes = resultMapping.getConvertedTypes();

    auto newOp =
        rewriter.create<WhileOp>(loc, convertedResultTypes, convertedOperands);
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

    rewriter.replaceOp(op, SmallVector<Value>(newOp->getResults()),
                       resultMapping);
    return success();
  }
};

class ConvertTypesInSCFYieldOp : public OneToNOpConversionPattern<YieldOp> {
public:
  using OneToNOpConversionPattern<YieldOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OneToNPatternRewriter &rewriter,
                  const OneToNTypeMapping &operandMapping,
                  const OneToNTypeMapping & /*resultMapping*/,
                  const ValueRange convertedOperands) const override {
    // Nothing to do if there is no non-identity conversion.
    if (!operandMapping.hasNonIdentityConversion())
      return failure();

    // Convert operands.
    rewriter.updateRootInPlace(op, [&] { op->setOperands(convertedOperands); });

    return success();
  }
};

class ConvertTypesInSCFConditionOp
    : public OneToNOpConversionPattern<ConditionOp> {
public:
  using OneToNOpConversionPattern<ConditionOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(ConditionOp op, OneToNPatternRewriter &rewriter,
                  const OneToNTypeMapping &operandMapping,
                  const OneToNTypeMapping & /*resultMapping*/,
                  const ValueRange convertedOperands) const override {
    // Nothing to do if there is no non-identity conversion.
    if (!operandMapping.hasNonIdentityConversion())
      return failure();

    // Convert operands.
    rewriter.updateRootInPlace(op, [&] { op->setOperands(convertedOperands); });

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
      ConvertTypesInSCFIfOp,
      ConvertTypesInSCFWhileOp,
      ConvertTypesInSCFYieldOp
      // clang-format on
      >(typeConverter, patterns.getContext());
}

} // namespace scf
} // namespace mlir
