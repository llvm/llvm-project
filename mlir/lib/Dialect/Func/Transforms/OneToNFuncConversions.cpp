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
    auto newOp =
        rewriter.create<CallOp>(loc, resultMapping.getConvertedTypes(),
                                adaptor.getFlatOperands(), op->getAttrs());

    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
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
    rewriter.modifyOpInPlace(
        op, [&] { op->setOperands(adaptor.getFlatOperands()); });

    return success();
  }
};

} // namespace

namespace mlir {

void populateFuncTypeConversionPatterns(const TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertTypesInFuncCallOp,
      ConvertTypesInFuncReturnOp
      // clang-format on
      >(typeConverter, patterns.getContext());
  populateOneToNFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      typeConverter, patterns);
}

} // namespace mlir
