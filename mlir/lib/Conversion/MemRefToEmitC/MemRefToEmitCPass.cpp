//===- MemRefToEmitC.cpp - MemRef to EmitC conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert memref ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitCPass.h"

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertMemRefToEmitCPass
    : public impl::ConvertMemRefToEmitCBase<ConvertMemRefToEmitCPass> {
  void runOnOperation() override {
    TypeConverter converter;

    // Fallback for other types.
    converter.addConversion([](Type type) -> std::optional<Type> {
      if (emitc::isSupportedEmitCType(type))
        return type;
      return {};
    });

    populateMemRefToEmitCTypeConversion(converter);

    auto materializeAsUnrealizedCast = [](OpBuilder &builder, Type resultType,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();

      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };

    converter.addSourceMaterialization(materializeAsUnrealizedCast);
    converter.addTargetMaterialization(materializeAsUnrealizedCast);

    RewritePatternSet patterns(&getContext());
    populateMemRefToEmitCConversionPatterns(patterns, converter);

    ConversionTarget target(getContext());
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
