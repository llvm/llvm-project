//===- ArithToEmitCPass.cpp - Arith to EmitC Pass ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Arith dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToEmitC/ArithToEmitCPass.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertArithToEmitC
    : public impl::ConvertArithToEmitCBase<ConvertArithToEmitC> {
  void runOnOperation() override;

  /// Applies conversion to opaque types for f80 and i80 types, both unsupported
  /// in emitc. Used to test the pass with opaque types.
  void populateOpaqueTypeConversions(TypeConverter &converter);
};
} // namespace

void ConvertArithToEmitC::populateOpaqueTypeConversions(
    TypeConverter &converter) {
  converter.addConversion([](Type type) -> std::optional<Type> {
    if (type.isF80())
      return emitc::OpaqueType::get(type.getContext(), "f80");
    if (type.isInteger() && type.getIntOrFloatBitWidth() == 80)
      return emitc::OpaqueType::get(type.getContext(), "i80");
    return type;
  });

  converter.addTypeAttributeConversion(
      [](Type type,
         Attribute attrToConvert) -> TypeConverter::AttributeConversionResult {
        if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attrToConvert)) {
          if (floatAttr.getType().isF80()) {
            return emitc::OpaqueAttr::get(type.getContext(), "f80");
          }
          return {};
        }
        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attrToConvert)) {
          if (intAttr.getType().isInteger() &&
              intAttr.getType().getIntOrFloatBitWidth() == 80) {
            return emitc::OpaqueAttr::get(type.getContext(), "i80");
          }
        }
        return {};
      });
}

void ConvertArithToEmitC::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(&getContext());

  TypeConverter typeConverter;

  populateOpaqueTypeConversions(typeConverter);
  populateArithToEmitCPatterns(typeConverter, patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
