//===- Bufferize.cpp - Bufferization for std ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of std ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
struct StdBufferizePass : public StdBufferizeBase<StdBufferizePass> {
  void runOnOperation() override {
    std::unique_ptr<BufferizationOptions> options =
        getPartialBufferizationOptions();
    options->addToDialectFilter<StandardOpsDialect>();

    if (failed(bufferizeOp(getOperation(), *options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    StandardOpsDialect, scf::SCFDialect>();
    mlir::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createStdBufferizePass() {
  return std::make_unique<StdBufferizePass>();
}
