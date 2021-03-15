//====----- Bufferize.cpp - Bufferization of shape ops  ---------*- C++-*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct ShapeBufferizePass : public ShapeBufferizeBase<ShapeBufferizePass> {
  void runOnFunction() override {
    MLIRContext &ctx = getContext();

    OwningRewritePatternList patterns;
    BufferizeTypeConverter typeConverter;
    ConversionTarget target(getContext());

    populateBufferizeMaterializationLegality(target);
    populateShapeStructuralTypeConversionsAndLegality(&ctx, typeConverter,
                                                      patterns, target);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<FunctionPass> mlir::createShapeBufferizePass() {
  return std::make_unique<ShapeBufferizePass>();
}
