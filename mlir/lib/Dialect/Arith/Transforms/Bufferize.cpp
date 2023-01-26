//===- Bufferize.cpp - Bufferization for Arith ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace arith {
#define GEN_PASS_DEF_ARITHBUFFERIZE
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace arith
} // namespace mlir

using namespace mlir;
using namespace bufferization;

namespace {
/// Pass to bufferize Arith ops.
struct ArithBufferizePass
    : public arith::impl::ArithBufferizeBase<ArithBufferizePass> {
  ArithBufferizePass(uint64_t alignment = 0, bool constantOpOnly = false)
      : constantOpOnly(constantOpOnly) {
    this->alignment = alignment;
  }

  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    if (constantOpOnly) {
      options.opFilter.allowOperation<arith::ConstantOp>();
    } else {
      options.opFilter.allowDialect<arith::ArithDialect>();
    }
    options.bufferAlignment = alignment;

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    arith::ArithDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
  }

private:
  bool constantOpOnly;
};
} // namespace

std::unique_ptr<Pass> mlir::arith::createArithBufferizePass() {
  return std::make_unique<ArithBufferizePass>();
}

std::unique_ptr<Pass>
mlir::arith::createConstantBufferizePass(uint64_t alignment) {
  return std::make_unique<ArithBufferizePass>(alignment,
                                              /*constantOpOnly=*/true);
}
