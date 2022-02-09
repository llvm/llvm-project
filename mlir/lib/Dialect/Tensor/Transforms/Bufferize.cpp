//===- Bufferize.cpp - Bufferization for `tensor` dialect ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of `tensor` dialect ops
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace bufferization;

namespace {
struct TensorBufferizePass : public TensorBufferizeBase<TensorBufferizePass> {
  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.addToDialectFilter<tensor::TensorDialect>();

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    tensor::TensorDialect, scf::SCFDialect,
                    arith::ArithmeticDialect>();
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTensorBufferizePass() {
  return std::make_unique<TensorBufferizePass>();
}
