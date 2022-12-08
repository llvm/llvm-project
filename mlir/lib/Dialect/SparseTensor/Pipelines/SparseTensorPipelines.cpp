//===- SparseTensorPipelines.cpp - Pipelines for sparse tensor code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

/// Return configuration options for One-Shot Bufferize.
static bufferization::OneShotBufferizationOptions
getBufferizationOptions(bool analysisOnly) {
  using namespace bufferization;
  OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  // TODO(springerm): To spot memory leaks more easily, returning dense allocs
  // should be disallowed.
  options.allowReturnAllocs = true;
  options.functionBoundaryTypeConversion = LayoutMapOption::IdentityLayoutMap;
  options.unknownTypeConverterFn = [](Value value, Attribute memorySpace,
                                      const BufferizationOptions &options) {
    return getMemRefTypeWithStaticIdentityLayout(
        value.getType().cast<TensorType>(), memorySpace);
  };
  if (analysisOnly) {
    options.testAnalysisOnly = true;
    options.printConflicts = true;
  }
  return options;
}

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void mlir::sparse_tensor::buildSparseCompiler(
    OpPassManager &pm, const SparseCompilerOptions &options) {
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizationPass());
  pm.addPass(createSparsificationAndBufferizationPass(
      getBufferizationOptions(options.testBufferizationAnalysisOnly),
      options.sparsificationOptions(), options.sparseTensorConversionOptions(),
      options.enableRuntimeLibrary, options.enableBufferInitialization,
      options.vectorLength,
      /*enableVLAVectorization=*/options.armSVE,
      /*enableSIMDIndex32=*/options.force32BitVectorIndices));
  if (options.testBufferizationAnalysisOnly)
    return;
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
  // TODO(springerm): Add sparse support to the BufferDeallocation pass and add
  // it to this pipeline.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertVectorToLLVMPass(options.lowerVectorToLLVMOptions()));
  pm.addPass(createMemRefToLLVMConversionPass());
  pm.addNestedPass<func::FuncOp>(createConvertComplexToStandardPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertComplexToLibmPass());
  pm.addPass(createConvertComplexToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void mlir::sparse_tensor::registerSparseTensorPipelines() {
  PassPipelineRegistration<SparseCompilerOptions>(
      "sparse-compiler",
      "The standard pipeline for taking sparsity-agnostic IR using the"
      " sparse-tensor type, and lowering it to LLVM IR with concrete"
      " representations and algorithms for sparse tensors.",
      buildSparseCompiler);
}
