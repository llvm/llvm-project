//===-- OneToNTypeFuncConversions.cpp - Func 1:N type conversion-*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The patterns in this file are heavily inspired (and copied from)
// convertFuncOpTypes in lib/Transforms/Utils/DialectConversion.cpp and the
// patterns in lib/Dialect/Func/Transforms/FuncConversions.cpp but work for 1:N
// type conversions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

using namespace mlir;
using namespace mlir::func;

namespace {

class ConvertTypesInFuncCallOp : public OneToNOpConversionPattern<CallOp> {
public:
  using OneToNOpConversionPattern<CallOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();

    // Nothing to do if the op doesn't have any non-identity conversions for its
    // operands or results.
    if (!adaptor.getOperandMapping().hasNonIdentityConversion() &&
        !resultMapping.hasNonIdentityConversion())
      return failure();

    // Create new CallOp.
    auto newOp = rewriter.create<CallOp>(loc, resultMapping.getConvertedTypes(),
                                         adaptor.getFlatOperands());
    newOp->setAttrs(op->getAttrs());

    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
    return success();
  }
};

class ConvertTypesInFuncFuncOp : public OneToNOpConversionPattern<FuncOp> {
public:
  using OneToNOpConversionPattern<FuncOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *typeConverter = getTypeConverter<OneToNTypeConverter>();

    // Construct mapping for function arguments.
    OneToNTypeMapping argumentMapping(op.getArgumentTypes());
    if (failed(typeConverter->computeTypeMapping(op.getArgumentTypes(),
                                                 argumentMapping)))
      return failure();

    // Construct mapping for function results.
    OneToNTypeMapping funcResultMapping(op.getResultTypes());
    if (failed(typeConverter->computeTypeMapping(op.getResultTypes(),
                                                 funcResultMapping)))
      return failure();

    // Nothing to do if the op doesn't have any non-identity conversions for its
    // operands or results.
    if (!argumentMapping.hasNonIdentityConversion() &&
        !funcResultMapping.hasNonIdentityConversion())
      return failure();

    // Update the function signature in-place.
    auto newType = FunctionType::get(rewriter.getContext(),
                                     argumentMapping.getConvertedTypes(),
                                     funcResultMapping.getConvertedTypes());
    rewriter.updateRootInPlace(op, [&] { op.setType(newType); });

    // Update block signatures.
    if (!op.isExternal()) {
      Region *region = &op.getBody();
      Block *block = &region->front();
      rewriter.applySignatureConversion(block, argumentMapping);
    }

    return success();
  }
};

class ConvertTypesInFuncReturnOp : public OneToNOpConversionPattern<ReturnOp> {
public:
  using OneToNOpConversionPattern<ReturnOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Nothing to do if there is no non-identity conversion.
    if (!adaptor.getOperandMapping().hasNonIdentityConversion())
      return failure();

    // Convert operands.
    rewriter.updateRootInPlace(
        op, [&] { op->setOperands(adaptor.getFlatOperands()); });

    return success();
  }
};

} // namespace

namespace mlir {

void populateFuncTypeConversionPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertTypesInFuncCallOp,
      ConvertTypesInFuncFuncOp,
      ConvertTypesInFuncReturnOp
      // clang-format on
      >(typeConverter, patterns.getContext());
}

} // namespace mlir
