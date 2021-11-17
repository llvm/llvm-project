//===- CodegenStrategy.cpp - Linalg programmable codegen strategy ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic and helpers to expose Linalg transforms as
// composable rewrite patterns through a programmable CodegenStrategy object.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-codegen-strategy"

void mlir::linalg::CodegenStrategy::configurePassPipeline(
    OpPassManager &pm, MLIRContext *context, bool addEnablePass) const {
  for (unsigned stepCount = 0, e = transformationSequence.size(); stepCount < e;
       ++stepCount) {
    const std::unique_ptr<Transformation> &t =
        transformationSequence[stepCount];
    std::string currentStr = std::to_string(stepCount);
    auto currentState = StringAttr::get(context, currentStr);
    std::string nextStr = std::to_string(stepCount + 1);
    auto nextState = StringAttr::get(context, nextStr);
    auto filter = (currentState.str() == std::to_string(0))
                      ? linalg::LinalgTransformationFilter(
                            t->filter, ArrayRef<StringAttr>{}, nextState)
                      : linalg::LinalgTransformationFilter(
                            t->filter, currentState, nextState);
    t->addToPassPipeline(pm, filter);
    if (addEnablePass)
      pm.addPass(createLinalgStrategyEnablePass(linalgEnablingOptions));
  }
  pm.addPass(createLinalgStrategyRemoveMarkersPass());
}
