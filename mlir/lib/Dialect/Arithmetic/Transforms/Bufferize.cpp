//===- Bufferize.cpp - Bufferization for Arithmetic ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace bufferization;

namespace {
/// Pass to bufferize Arithmetic ops.
struct ArithmeticBufferizePass
    : public ArithmeticBufferizeBase<ArithmeticBufferizePass> {
  ArithmeticBufferizePass(uint64_t alignment = 0, bool constantOpOnly = false)
      : constantOpOnly(constantOpOnly) {
    this->alignment = alignment;
  }

  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    if (constantOpOnly) {
      options.opFilter.allowOperation<arith::ConstantOp>();
    } else {
      options.opFilter.allowDialect<arith::ArithmeticDialect>();
    }
    options.bufferAlignment = alignment;

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    arith::ArithmeticDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
  }

private:
  bool constantOpOnly;
};
} // namespace

std::unique_ptr<Pass> mlir::arith::createArithmeticBufferizePass() {
  return std::make_unique<ArithmeticBufferizePass>();
}

std::unique_ptr<Pass>
mlir::arith::createConstantBufferizePass(uint64_t alignment) {
  return std::make_unique<ArithmeticBufferizePass>(alignment,
                                                   /*constantOpOnly=*/true);
}
