//===- DecomposeCallGraphTypes.cpp - CG type decomposition ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Transforms/DecomposeCallGraphTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ValueDecomposer
//===----------------------------------------------------------------------===//

void ValueDecomposer::decomposeValue(OpBuilder &builder, Location loc,
                                     Type type, Value value,
                                     SmallVectorImpl<Value> &results) {
  for (auto &conversion : decomposeValueConversions)
    if (conversion(builder, loc, type, value, results))
      return;
  results.push_back(value);
}

//===----------------------------------------------------------------------===//
// DecomposeCallGraphTypesOpConversionPattern
//===----------------------------------------------------------------------===//

namespace {
/// Base OpConversionPattern class to make a ValueDecomposer available to
/// inherited patterns.
template <typename SourceOp>
class DecomposeCallGraphTypesOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  DecomposeCallGraphTypesOpConversionPattern(TypeConverter &typeConverter,
                                             MLIRContext *context,
                                             ValueDecomposer &decomposer,
                                             PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit),
        decomposer(decomposer) {}

protected:
  ValueDecomposer &decomposer;
};
} // namespace

//===----------------------------------------------------------------------===//
// DecomposeCallGraphTypesForFuncArgs
//===----------------------------------------------------------------------===//

namespace {
/// Expand function arguments according to the provided TypeConverter and
/// ValueDecomposer.
struct DecomposeCallGraphTypesForFuncArgs
    : public DecomposeCallGraphTypesOpConversionPattern<FuncOp> {
  using DecomposeCallGraphTypesOpConversionPattern::
      DecomposeCallGraphTypesOpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto functionType = op.getType();

    // Convert function arguments using the provided TypeConverter.
    TypeConverter::SignatureConversion conversion(functionType.getNumInputs());
    for (auto argType : llvm::enumerate(functionType.getInputs())) {
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
    rewriter.updateRootInPlace(op, [&] {
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
/// Expand return operands according to the provided TypeConverter and
/// ValueDecomposer.
struct DecomposeCallGraphTypesForReturnOp
    : public DecomposeCallGraphTypesOpConversionPattern<ReturnOp> {
  using DecomposeCallGraphTypesOpConversionPattern::
      DecomposeCallGraphTypesOpConversionPattern;
  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 2> newOperands;
    for (Value operand : adaptor.getOperands())
      decomposer.decomposeValue(rewriter, op.getLoc(), operand.getType(),
                                operand, newOperands);
    rewriter.replaceOpWithNewOp<ReturnOp>(op, newOperands);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// DecomposeCallGraphTypesForCallOp
//===----------------------------------------------------------------------===//

namespace {
/// Expand call op operands and results according to the provided TypeConverter
/// and ValueDecomposer.
struct DecomposeCallGraphTypesForCallOp
    : public DecomposeCallGraphTypesOpConversionPattern<CallOp> {
  using DecomposeCallGraphTypesOpConversionPattern::
      DecomposeCallGraphTypesOpConversionPattern;

  LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Create the operands list of the new `CallOp`.
    SmallVector<Value, 2> newOperands;
    for (Value operand : adaptor.getOperands())
      decomposer.decomposeValue(rewriter, op.getLoc(), operand.getType(),
                                operand, newOperands);

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
    MLIRContext *context, TypeConverter &typeConverter,
    ValueDecomposer &decomposer, RewritePatternSet &patterns) {
  patterns
      .add<DecomposeCallGraphTypesForCallOp, DecomposeCallGraphTypesForFuncArgs,
           DecomposeCallGraphTypesForReturnOp>(typeConverter, context,
                                               decomposer);
}
