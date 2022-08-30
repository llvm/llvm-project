//===- Bufferize.cpp - Bufferization of linalg ops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGBUFFERIZEPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace bufferization;

namespace {
/// Converts Linalg operations that work on tensor-type operands or results to
/// work on buffers.
struct LinalgBufferizePass
    : public impl::LinalgBufferizePassBase<LinalgBufferizePass> {
  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<linalg::LinalgDialect>();

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect, linalg::LinalgDialect>();
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgBufferizePass() {
  return std::make_unique<LinalgBufferizePass>();
}
