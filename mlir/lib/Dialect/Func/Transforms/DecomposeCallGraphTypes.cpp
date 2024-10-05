//===- DecomposeCallGraphTypes.cpp - CG type decomposition ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::func;

//===----------------------------------------------------------------------===//
// DecomposeCallGraphTypesForFuncArgs
//===----------------------------------------------------------------------===//

namespace {
/// Expand function arguments according to the provided TypeConverter.
struct DecomposeCallGraphTypesForFuncArgs
    : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto functionType = op.getFunctionType();

    // Convert function arguments using the provided TypeConverter.
    TypeConverter::SignatureConversion conversion(functionType.getNumInputs());
    for (const auto &argType : llvm::enumerate(functionType.getInputs())) {
      SmallVector<Type, 2> decomposedTypes;
      if (failed(typeConverter->convertType(argType.value(), decomposedTypes)))
        return failure();
      if (!decomposedTypes.empty())
        conversion.addInputs(argType.index(), decomposedTypes);
    }

    // If the SignatureConversion doesn't apply, bail out.
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *getTypeConverter(),
                                           &conversion)))
      return failure();

    // Update the signature of the function.
    SmallVector<Type, 2> newResultTypes;
    if (failed(typeConverter->convertTypes(functionType.getResults(),
                                           newResultTypes)))
      return failure();
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                          newResultTypes));
    });
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// DecomposeCallGraphTypesForReturnOp
//===----------------------------------------------------------------------===//

namespace {
/// Expand return operands according to the provided TypeConverter.
struct DecomposeCallGraphTypesForReturnOp
    : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 2> newOperands;
    for (ValueRange operand : adaptor.getOperands())
      llvm::append_range(newOperands, operand);
    rewriter.replaceOpWithNewOp<ReturnOp>(op, newOperands);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// DecomposeCallGraphTypesForCallOp
//===----------------------------------------------------------------------===//

namespace {
/// Expand call op operands and results according to the provided TypeConverter.
struct DecomposeCallGraphTypesForCallOp : public OpConversionPattern<CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CallOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Create the operands list of the new `CallOp`.
    SmallVector<Value, 2> newOperands;
    for (ValueRange operand : adaptor.getOperands())
      llvm::append_range(newOperands, operand);

    // Create the new result types for the new `CallOp` and track the number of
    // replacement types for each original op result.
    SmallVector<Type, 2> newResultTypes;
    SmallVector<unsigned> expandedResultSizes;
    for (Type resultType : op.getResultTypes()) {
      unsigned oldSize = newResultTypes.size();
      if (failed(typeConverter->convertType(resultType, newResultTypes)))
        return failure();
      expandedResultSizes.push_back(newResultTypes.size() - oldSize);
    }

    CallOp newCallOp = rewriter.create<CallOp>(op.getLoc(), op.getCalleeAttr(),
                                               newResultTypes, newOperands);

    // Build a replacement value for each result to replace its uses.
    SmallVector<ValueRange> replacedValues;
    replacedValues.reserve(op.getNumResults());
    unsigned startIdx = 0;
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      ValueRange repl =
          newCallOp.getResults().slice(startIdx, expandedResultSizes[i]);
      replacedValues.push_back(repl);
      startIdx += expandedResultSizes[i];
    }
    rewriter.replaceOpWithMultiple(op, replacedValues);
    return success();
  }
};
} // namespace

void mlir::populateDecomposeCallGraphTypesPatterns(
    MLIRContext *context, const TypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  patterns
      .add<DecomposeCallGraphTypesForCallOp, DecomposeCallGraphTypesForFuncArgs,
           DecomposeCallGraphTypesForReturnOp>(typeConverter, context);
}
