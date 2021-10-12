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
    OpPassManager &pm, MLIRContext *context) const {
  for (unsigned stepCount = 0, e = transformationSequence.size(); stepCount < e;
       ++stepCount) {
    const std::unique_ptr<Transformation> &t =
        transformationSequence[stepCount];
    std::string currentStr = std::to_string(stepCount);
    auto currentState = Identifier::get(currentStr, context);
    std::string nextStr = std::to_string(stepCount + 1);
    auto nextState = Identifier::get(nextStr, context);
    auto filter = (currentState.str() == std::to_string(0))
                      ? linalg::LinalgTransformationFilter(
                            t->filter, ArrayRef<Identifier>{}, nextState)
                      : linalg::LinalgTransformationFilter(
                            t->filter, currentState, nextState);
    t->addToPassPipeline(pm, filter);
    pm.addPass(createLinalgStrategyEnablePass());
  }
  LinalgVectorLoweringOptions vectorLoweringOptions;
  vectorLoweringOptions.maxTransferRank =
      lateCodegenStrategyOptions.maxTransferRank;
  vectorLoweringOptions.enableVectorTransferLowering =
      lateCodegenStrategyOptions.enableVectorTransferLowering;
  vectorLoweringOptions.enableVectorTransferPartialRewrite =
      lateCodegenStrategyOptions.enableVectorTransferPartialRewrite;
  vectorLoweringOptions.enableVectorContractLowering =
      lateCodegenStrategyOptions.enableVectorContractLowering;
  vectorLoweringOptions.enableVectorToSCFConversion =
      lateCodegenStrategyOptions.enableVectorToSCFConversion;
  vectorLoweringOptions.vectorTransformOptions = vectorTransformOptions;
  vectorLoweringOptions.vectorTransferToSCFOptions = vectorToSCFOptions;
  pm.addPass(createLinalgStrategyLowerVectorsPass(vectorLoweringOptions));
}

LogicalResult mlir::linalg::CodegenStrategy::transform(FuncOp funcOp) const {
  PassManager pm(funcOp.getContext(), funcOp.getOperationName());
  configurePassPipeline(pm, funcOp.getContext());
  LogicalResult res = pm.run(funcOp);
  // Ensure we drop the marker in the end.
  funcOp.walk([](LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
  return res;
}
