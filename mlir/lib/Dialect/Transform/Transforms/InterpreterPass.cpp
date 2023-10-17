//===- InterpreterPass.cpp - Transform dialect interpreter pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

using namespace mlir;

namespace mlir {
namespace transform {
#define GEN_PASS_DEF_INTERPRETERPASS
#include "mlir/Dialect/Transform/Transforms/Passes.h.inc"
} // namespace transform
} // namespace mlir

namespace {
class InterpreterPass
    : public transform::impl::InterpreterPassBase<InterpreterPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp transformModule =
        transform::detail::getPreloadedTransformModule(context);
    if (failed(transform::applyTransformNamedSequence(
            getOperation(), transformModule,
            options.enableExpensiveChecks(true), entryPoint)))
      return signalPassFailure();
  }

private:
  /// Transform interpreter options.
  transform::TransformOptions options;
};
} // namespace
