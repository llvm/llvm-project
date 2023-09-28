//===- BufferizationPipelines.cpp - Pipelines for bufferization -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace bufferization;

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

DeallocationOptions
BufferDeallocationPipelineOptions::asDeallocationOptions() const {
  DeallocationOptions opts;
  opts.privateFuncDynamicOwnership = privateFunctionDynamicOwnership.getValue();
  opts.verifyFunctionBoundaryABI = verifyFunctionBoundaryABI.getValue();
  return opts;
}

void mlir::bufferization::buildBufferDeallocationPipeline(
    OpPassManager &pm, const BufferDeallocationPipelineOptions &options) {
  pm.addPass(memref::createExpandReallocPass(/*emitDeallocs=*/false));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createOwnershipBasedBufferDeallocationPass(
      options.asDeallocationOptions()));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createBufferDeallocationSimplificationPass());
  pm.addPass(createLowerDeallocationsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void mlir::bufferization::registerBufferizationPipelines() {
  PassPipelineRegistration<BufferDeallocationPipelineOptions>(
      "buffer-deallocation-pipeline",
      "The default pipeline for automatically inserting deallocation "
      "operations after one-shot bufferization. Deallocation operations "
      "(except `memref.realloc`) may not be present already.",
      buildBufferDeallocationPipeline);
}
