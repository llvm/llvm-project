//===- TensorToLinalgPass.cpp - Tensor to Linalg Passes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert Tensor dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"

#include "mlir/Conversion/TensorToLinalg/TensorToLinalg.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORTOLINALGPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// A pass converting MLIR Tensor operations into the Linalg dialect.
class ConvertTensorToLinalgPass
    : public impl::ConvertTensorToLinalgPassBase<ConvertTensorToLinalgPass> {
  using ConvertTensorToLinalgPassBase::ConvertTensorToLinalgPassBase;

  void runOnOperation() override {
    auto &context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<mlir::arith::ArithmeticDialect,
                           mlir::linalg::LinalgDialect,
                           mlir::tensor::TensorDialect>();
    target.addIllegalOp<mlir::tensor::PadOp>();

    RewritePatternSet patterns(&context);
    populateTensorToLinalgPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertTensorToLinalgPass() {
  return std::make_unique<ConvertTensorToLinalgPass>();
}
