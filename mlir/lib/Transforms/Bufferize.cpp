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
  addSourceMaterialization([](OpBuilder &builder, RankedTensorType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<BaseMemRefType>());
    return builder.create<TensorLoadOp>(loc, type, inputs[0]);
  });
  addTargetMaterialization([](OpBuilder &builder, MemRefType type,
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

/// This method returns ResultConversionKind for the input type.
BufferizeTypeConverter::ResultConversionKind
BufferizeTypeConverter::getResultConversionKind(Type origin, Type converted) {
  for (auto &conversion : resultTypeConversions)
    if (auto res = conversion(origin, converted))
      return res.getValue();
  return KeepAsFunctionResult;
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
    for (auto origin : originTypes) {
      Type converted = converter.convertType(origin);
      auto kind = converter.getResultConversionKind(origin, converted);
      if (kind == BufferizeTypeConverter::AppendToArgumentsList) {
        conversion.addInputs(converted);
      } else {
        assert(kind == BufferizeTypeConverter::KeepAsFunctionResult);
        newResultTypes.push_back(converted);
      }
    }
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

namespace {
// This class represents a mapping from a result to a list of values and some
// results that have not yet constructed. Instead, the indices of these
// results in the operation that will be constructed are known. They will be
// replaced with the actual values when they are available. The order of
// adding to this mapping is important.
class CallOpResultMapping {
public:
  CallOpResultMapping() { order = 0; };

  /// Add an available value to the mapping.
  void addMapping(Value value) { toValuesMapping.push_back({order++, value}); }

  /// Add the index of unavailble result value to the mapping.
  void addMapping(unsigned index) {
    toIndicesMapping.push_back({order++, index});
  }

  /// This method returns the mapping values list. The unknown result values
  /// that only their indicies are available are replaced with their values.
  void getMappingValues(ValueRange valuesToReplaceIndices,
                        SmallVectorImpl<Value> &values) {
    // Append available values to the list.
    SmallVector<std::pair<unsigned, Value>, 2> res(toValuesMapping.begin(),
                                                   toValuesMapping.end());
    // Replace the indices with the actual values.
    for (const std::pair<unsigned, unsigned> &entry : toIndicesMapping) {
      assert(entry.second < valuesToReplaceIndices.size() &&
             "The value index is out of range.");
      res.push_back({entry.first, valuesToReplaceIndices[entry.second]});
    }
    // Sort the values based on their adding orders.
    llvm::sort(res, [](const std::pair<unsigned, Value> &v1,
                       const std::pair<unsigned, Value> &v2) {
      return v1.first < v2.first;
    });
    // Fill the values.
    for (const std::pair<unsigned, Value> &entry : res)
      values.push_back(entry.second);
  }

private:
  /// Keeping the inserting order of mapping values.
  int order;

  /// Containing the mapping values with their inserting orders.
  SmallVector<std::pair<unsigned, Value>, 2> toValuesMapping;

  /// Containing the indices of result values with their inserting orders.
  SmallVector<std::pair<unsigned, unsigned>, 2> toIndicesMapping;
};
} // namespace

/// Performs the actual rewriting step.
LogicalResult BufferizeCallOpConverter::matchAndRewrite(
    CallOp callOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  Location loc = callOp.getLoc();
  OpBuilder builder(callOp);
  SmallVector<Value, 2> newOperands;

  // TODO: if the CallOp references a FuncOp that only has a declaration (e.g.
  // to an externally defined symbol like an external library calls), only
  // convert if some special attribute is set.
  // This will allow more control of interop across ABI boundaries.

  // Create the operands list of the new `CallOp`. It unpacks the decomposable
  // values if a decompose callback function has been provided by the user.
  for (auto operand : operands) {
    SmallVector<Value, 2> values;
    converter.tryDecomposeValue(builder, loc, operand.getType(), operand,
                                values);
    newOperands.append(values.begin(), values.end());
  }

  // Create the new result types for the new `CallOp` and a mapping from the old
  // result to new value(s).
  SmallVector<Type, 2> newResultTypes;
  SmallVector<CallOpResultMapping, 4> mappings;
  mappings.resize(callOp.getNumResults());
  for (auto result : llvm::enumerate(callOp.getResults())) {
    SmallVector<Type, 2> originTypes;
    converter.tryDecomposeType(result.value().getType(), originTypes);
    auto &resultMapping = mappings[result.index()];
    for (Type origin : originTypes) {
      Type converted = converter.convertType(origin);
      auto kind = converter.getResultConversionKind(origin, converted);
      if (kind == BufferizeTypeConverter::KeepAsFunctionResult) {
        newResultTypes.push_back(converted);
        // The result value is not yet available. Its index is kept and it is
        // replaced with the actual value of the new `CallOp` later.
        resultMapping.addMapping(newResultTypes.size() - 1);
      } else {
        // kind = BufferizeTypeConverter::AppendToArgumentsList
        MemRefType memref = converted.dyn_cast<MemRefType>();
        if (!memref)
          return callOp.emitError("Cannot allocate for a non-Memref type");
        Value alloc = rewriter.create<AllocOp>(loc, memref);
        newOperands.push_back(alloc);
        resultMapping.addMapping(alloc);
      }
    }
  }

  CallOp newCallOp = rewriter.create<CallOp>(loc, callOp.getCallee(),
                                             newResultTypes, newOperands);

  // Build a replacing value for each result to replace its uses. If a result
  // has multiple mapping values, it needs to be packed to a single value.
  OpBuilder nextBuilder(callOp.getOperation()->getNextNode());
  SmallVector<Value, 2> replacedValues;
  replacedValues.reserve(callOp.getNumResults());
  for (unsigned i = 0, e = callOp.getNumResults(); i < e; ++i) {
    SmallVector<Value, 2> valuesToPack;
    mappings[i].getMappingValues(newCallOp.getResults(), valuesToPack);
    if (valuesToPack.empty()) {
      // No replacement is required.
      replacedValues.push_back(nullptr);
    } else if (valuesToPack.size() == 1) {
      replacedValues.push_back(valuesToPack.front());
    } else {
      // Values need to be packed using callback function. The same callback
      // that is used for materializeArgumentConversion is used for packing.
      Value packed = converter.materializeArgumentConversion(
          nextBuilder, loc, callOp.getType(i), valuesToPack);
      replacedValues.push_back(packed);
    }
  }
  rewriter.replaceOp(callOp, replacedValues);
  return success();
}
