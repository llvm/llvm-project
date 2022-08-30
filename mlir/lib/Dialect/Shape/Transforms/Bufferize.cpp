//====----- Bufferize.cpp - Bufferization of shape ops  ---------*- C++-*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shape/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_SHAPEBUFFERIZEPASS
#include "mlir/Dialect/Shape/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace bufferization;

namespace {
struct ShapeBufferizePass
    : public impl::ShapeBufferizePassBase<ShapeBufferizePass> {
  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<shape::ShapeDialect>();

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    shape::ShapeDialect>();
    shape::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createShapeBufferizePass() {
  return std::make_unique<ShapeBufferizePass>();
}
