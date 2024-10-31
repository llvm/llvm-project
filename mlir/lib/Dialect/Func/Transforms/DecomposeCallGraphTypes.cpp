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
// Helper functions
//===----------------------------------------------------------------------===//

/// If the given value can be decomposed with the type converter, decompose it.
/// Otherwise, return the given value.
// TODO: Value decomposition should happen automatically through a 1:N adaptor.
// This function will disappear when the 1:1 and 1:N drivers are merged.
static SmallVector<Value> decomposeValue(OpBuilder &builder, Location loc,
                                         Value value,
                                         const TypeConverter *converter) {
  // Try to convert the given value's type. If that fails, just return the
  // given value.
  SmallVector<Type> convertedTypes;
  if (failed(converter->convertType(value.getType(), convertedTypes)))
    return {value};
  if (convertedTypes.empty())
    return {};

  // If the given value's type is already legal, just return the given value.
  TypeRange convertedTypeRange(convertedTypes);
  if (convertedTypeRange == TypeRange(value.getType()))
    return {value};

  // Try to materialize a target conversion. If the materialization did not
  // produce values of the requested type, the materialization failed. Just
  // return the given value in that case.
  SmallVector<Value> result = converter->materializeTargetConversion(
      builder, loc, convertedTypeRange, value);
  if (result.empty())
    return {value};
  return result;
}

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
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 2> newOperands;
    for (Value operand : adaptor.getOperands()) {
      // TODO: We can directly take the values from the adaptor once this is a
      // 1:N conversion pattern.
      llvm::append_range(newOperands,
                         decomposeValue(rewriter, operand.getLoc(), operand,
                                        getTypeConverter()));
    }
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
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Create the operands list of the new `CallOp`.
    SmallVector<Value, 2> newOperands;
    for (Value operand : adaptor.getOperands()) {
      // TODO: We can directly take the values from the adaptor once this is a
      // 1:N conversion pattern.
      llvm::append_range(newOperands,
                         decomposeValue(rewriter, operand.getLoc(), operand,
                                        getTypeConverter()));
    }

    // Create the new result types for the new `CallOp` and track the indices in
    // the new call op's results that correspond to the old call op's results.
    //
    // expandedResultIndices[i] = "list of new result indices that old result i
    // expanded to".
    SmallVector<Type, 2> newResultTypes;
    SmallVector<SmallVector<unsigned, 2>, 4> expandedResultIndices;
    for (Type resultType : op.getResultTypes()) {
      unsigned oldSize = newResultTypes.size();
      if (failed(typeConverter->convertType(resultType, newResultTypes)))
        return failure();
      auto &resultMapping = expandedResultIndices.emplace_back();
      for (unsigned i = oldSize, e = newResultTypes.size(); i < e; i++)
        resultMapping.push_back(i);
    }

    CallOp newCallOp = rewriter.create<CallOp>(op.getLoc(), op.getCalleeAttr(),
                                               newResultTypes, newOperands);

    // Build a replacement value for each result to replace its uses. If a
    // result has multiple mapping values, it needs to be materialized as a
    // single value.
    SmallVector<Value, 2> replacedValues;
    replacedValues.reserve(op.getNumResults());
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      auto decomposedValues = llvm::to_vector<6>(
          llvm::map_range(expandedResultIndices[i],
                          [&](unsigned i) { return newCallOp.getResult(i); }));
      if (decomposedValues.empty()) {
        // No replacement is required.
        replacedValues.push_back(nullptr);
      } else if (decomposedValues.size() == 1) {
        replacedValues.push_back(decomposedValues.front());
      } else {
        // Materialize a single Value to replace the original Value.
        Value materialized = getTypeConverter()->materializeArgumentConversion(
            rewriter, op.getLoc(), op.getType(i), decomposedValues);
        replacedValues.push_back(materialized);
      }
    }
    rewriter.replaceOp(op, replacedValues);
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
