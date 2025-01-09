//===- StripFuncQuantTypes.cpp - Strip quantized types --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Strips quantized types from function headers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace quant {

#define GEN_PASS_DEF_STRIPFUNCQUANTTYPES
#include "mlir/Dialect/Quant/Transforms/Passes.h.inc"

namespace {

class QuantizedTypeConverter : public TypeConverter {

  static Type convertQuantizedType(QuantizedType quantizedType) {
    return quantizedType.getStorageType();
  }

  static Type convertTensorType(TensorType tensorType) {
    if (auto quantizedType =
            dyn_cast<QuantizedType>(tensorType.getElementType()))
      return tensorType.clone(convertQuantizedType(quantizedType));
    return tensorType;
  }

  static Value materializeConversion(OpBuilder &builder, Type type,
                                     ValueRange inputs, Location loc) {
    assert(inputs.size() == 1);
    return builder.create<quant::StorageCastOp>(loc, type, inputs[0]);
  }

public:
  explicit QuantizedTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertQuantizedType);
    addConversion(convertTensorType);

    addSourceMaterialization(materializeConversion);
    addTargetMaterialization(materializeConversion);
  }
};

// Conversion pass
class StripFuncQuantTypes
    : public impl::StripFuncQuantTypesBase<StripFuncQuantTypes> {

  // Return whether a type is considered legal when occurring in the header of
  // a function or as an operand to a 'return' op.
  static bool isLegalType(Type type) {
    if (auto tensorType = dyn_cast<TensorType>(type))
      return isLegalType(tensorType.getElementType());
    return !isa<quant::QuantizedType>(type);
  }

public:
  void runOnOperation() override {

    auto moduleOp = cast<ModuleOp>(getOperation());
    auto *context = &getContext();

    QuantizedTypeConverter typeConverter;
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    // Mark func.func, func.return, and func.call illegal if they contain any
    // quantized types.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    // Register conversion patterns
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // Apply conversion
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace quant
} // namespace mlir
