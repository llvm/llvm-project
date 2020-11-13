//===- Bufferize.cpp - Bufferization utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Bufferize.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BufferizeTypeConverter
//===----------------------------------------------------------------------===//

/// Registers conversions into BufferizeTypeConverter
BufferizeTypeConverter::BufferizeTypeConverter() {
  // Keep all types unchanged.
  addConversion([](Type type) { return type; });
  // Convert RankedTensorType to MemRefType.
  addConversion([](RankedTensorType type) -> Type {
    return MemRefType::get(type.getShape(), type.getElementType());
  });
  // Convert UnrankedTensorType to UnrankedMemRefType.
  addConversion([](UnrankedTensorType type) -> Type {
    return UnrankedMemRefType::get(type.getElementType(), 0);
  });
  addSourceMaterialization([](OpBuilder &builder, TensorType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<BaseMemRefType>());
    return builder.create<TensorLoadOp>(loc, type, inputs[0]);
  });
  addTargetMaterialization([](OpBuilder &builder, BaseMemRefType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<TensorToMemrefOp>(loc, type, inputs[0]);
  });
}

/// This method tries to decompose a value of a certain type using provided
/// decompose callback functions. If it is unable to do so, the original value
/// is returned.
void BufferizeTypeConverter::tryDecomposeValue(
    OpBuilder &builder, Location loc, Type type, Value value,
    SmallVectorImpl<Value> &results) {
  for (auto &conversion : decomposeValueConversions)
    if (conversion(builder, loc, type, value, results))
      return;
  results.push_back(value);
}

/// This method tries to decompose a type using provided decompose callback
/// functions. If it is unable to do so, the original type is returned.
void BufferizeTypeConverter::tryDecomposeType(Type type,
                                              SmallVectorImpl<Type> &types) {
  for (auto &conversion : decomposeTypeConversions)
    if (conversion(type, types))
      return;
  types.push_back(type);
}

void mlir::populateBufferizeMaterializationLegality(ConversionTarget &target) {
  target.addLegalOp<TensorLoadOp, TensorToMemrefOp>();
}

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeTensorLoadOp : public OpConversionPattern<TensorLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorLoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TensorLoadOp::Adaptor adaptor(operands);
    rewriter.replaceOp(op, adaptor.memref());
    return success();
  }
};
} // namespace

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeTensorToMemrefOp : public OpConversionPattern<TensorToMemrefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorToMemrefOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TensorToMemrefOp::Adaptor adaptor(operands);
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};
} // namespace

void mlir::populateEliminateBufferizeMaterializationsPatterns(
    MLIRContext *context, BufferizeTypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns.insert<BufferizeTensorLoadOp, BufferizeTensorToMemrefOp>(
      typeConverter, context);
}

//===----------------------------------------------------------------------===//
// BufferizeFuncOpConverter
//===----------------------------------------------------------------------===//

/// Performs the actual function signature rewriting step.
LogicalResult BufferizeFuncOpConverter::matchAndRewrite(
    mlir::FuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto funcType = funcOp.getType();

  // Convert function arguments using the provided TypeConverter.
  TypeConverter::SignatureConversion conversion(funcType.getNumInputs());
  for (auto argType : llvm::enumerate(funcType.getInputs())) {
    SmallVector<Type, 2> decomposedTypes, convertedTypes;
    converter.tryDecomposeType(argType.value(), decomposedTypes);
    converter.convertTypes(decomposedTypes, convertedTypes);
    conversion.addInputs(argType.index(), convertedTypes);
  }

  // Convert the result types of the function.
  SmallVector<Type, 2> newResultTypes;
  newResultTypes.reserve(funcOp.getNumResults());
  for (Type resultType : funcType.getResults()) {
    SmallVector<Type, 2> originTypes;
    converter.tryDecomposeType(resultType, originTypes);
    for (auto origin : originTypes)
      newResultTypes.push_back(converter.convertType(origin));
  }

  if (failed(rewriter.convertRegionTypes(&funcOp.getBody(), converter,
                                         &conversion)))
    return failure();

  // Update the signature of the function.
  rewriter.updateRootInPlace(funcOp, [&] {
    funcOp.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                            newResultTypes));
  });
  return success();
}

//===----------------------------------------------------------------------===//
// BufferizeCallOpConverter
//===----------------------------------------------------------------------===//

/// Performs the actual rewriting step.
LogicalResult BufferizeCallOpConverter::matchAndRewrite(
    CallOp callOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  Location loc = callOp.getLoc();
  SmallVector<Value, 2> newOperands;

  // TODO: if the CallOp references a FuncOp that only has a declaration (e.g.
  // to an externally defined symbol like an external library calls), only
  // convert if some special attribute is set.
  // This will allow more control of interop across ABI boundaries.

  // Create the operands list of the new `CallOp`. It unpacks the decomposable
  // values if a decompose callback function has been provided by the user.
  for (auto operand : operands)
    converter.tryDecomposeValue(rewriter, loc, operand.getType(), operand,
                                newOperands);

  // Create the new result types for the new `CallOp` and track the indices in
  // the new call op's results that correspond to the old call op's results.
  SmallVector<Type, 2> newResultTypes;
  SmallVector<SmallVector<int, 2>, 4> expandedResultIndices;
  expandedResultIndices.resize(callOp.getNumResults());
  for (auto result : llvm::enumerate(callOp.getResults())) {
    SmallVector<Type, 2> originTypes;
    converter.tryDecomposeType(result.value().getType(), originTypes);
    auto &resultMapping = expandedResultIndices[result.index()];
    for (Type origin : originTypes) {
      Type converted = converter.convertType(origin);
      newResultTypes.push_back(converted);
      // The result value is not yet available. Its index is kept and it is
      // replaced with the actual value of the new `CallOp` later.
      resultMapping.push_back(newResultTypes.size() - 1);
    }
  }

  CallOp newCallOp = rewriter.create<CallOp>(loc, callOp.getCallee(),
                                             newResultTypes, newOperands);

  // Build a replacing value for each result to replace its uses. If a result
  // has multiple mapping values, it needs to be packed to a single value.
  SmallVector<Value, 2> replacedValues;
  replacedValues.reserve(callOp.getNumResults());
  for (unsigned i = 0, e = callOp.getNumResults(); i < e; ++i) {
    auto valuesToPack = llvm::to_vector<6>(
        llvm::map_range(expandedResultIndices[i],
                        [&](int i) { return newCallOp.getResult(i); }));
    if (valuesToPack.empty()) {
      // No replacement is required.
      replacedValues.push_back(nullptr);
    } else if (valuesToPack.size() == 1) {
      replacedValues.push_back(valuesToPack.front());
    } else {
      // Values need to be packed using callback function. The same callback
      // that is used for materializeArgumentConversion is used for packing.
      Value packed = converter.materializeArgumentConversion(
          rewriter, loc, callOp.getType(i), valuesToPack);
      replacedValues.push_back(packed);
    }
  }
  rewriter.replaceOp(callOp, replacedValues);
  return success();
}
