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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace tensor {
#define GEN_PASS_DEF_TENSORBUFFERIZEPASS
#include "mlir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace tensor
} // namespace mlir

using namespace mlir;
using namespace bufferization;

namespace {
struct TensorBufferizePass
    : public tensor::impl::TensorBufferizePassBase<TensorBufferizePass> {
  using TensorBufferizePassBase::TensorBufferizePassBase;

  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<tensor::TensorDialect>();

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
