//===- SparsificationAndBufferizationPass.cpp - Tensor to Memref Lowering -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::func;

namespace mlir {
namespace sparse_tensor {

/// Return `true` if one of the given types is a sparse tensor type.
static bool containsSparseTensor(TypeRange types) {
  for (Type t : types)
    if (getSparseTensorEncoding(t))
      return true;
  return false;
}

/// A pass that lowers tensor ops to memref ops, regardless of whether they are
/// dense or sparse.
///
/// One-Shot Analysis is used to detect RaW conflicts and to insert buffer
/// copies of the tensor level (`insertTensorCopies`). Afterwards, the lowering
/// of tensor ops to memref ops follows a different code path depending on
/// whether the op is sparse or dense:
///
/// * Sparse tensor ops are lowered through Sparsification and follow-up pass
///   that lowers sparse_tensor dialect ops.
/// * Dense tensor ops are lowered through BufferizableOpInterface
///   implementations.
class SparsificationAndBufferizationPass
    : public PassWrapper<SparsificationAndBufferizationPass,
                         OperationPass<ModuleOp>> {
public:
  SparsificationAndBufferizationPass(
      const bufferization::OneShotBufferizationOptions &bufferizationOptions,
      const SparsificationOptions &sparsificationOptions,
      const SparseTensorConversionOptions &sparseTensorConversionOptions,
      bool enableRuntimeLibrary, bool enableBufferInitialization)
      : bufferizationOptions(bufferizationOptions),
        sparsificationOptions(sparsificationOptions),
        sparseTensorConversionOptions(sparseTensorConversionOptions),
        enableRuntimeLibrary(enableRuntimeLibrary),
        enableBufferInitialization(enableBufferInitialization) {}

  /// Bufferize all dense ops. This assumes that no further analysis is needed
  /// and that all required buffer copies were already inserted by
  /// `insertTensorCopies` in the form of `bufferization.alloc_tensor` ops.
  LogicalResult runDenseBufferization() {
    bufferization::OpFilter denseOpFilter;
    denseOpFilter.allowOperation([&](Operation *op) {
      if (containsSparseTensor(TypeRange(op->getResults())) ||
          containsSparseTensor(TypeRange(op->getOperands())))
        return false;
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        FunctionType funcType = funcOp.getFunctionType();
        if (containsSparseTensor(funcType.getInputs()) ||
            containsSparseTensor(funcType.getResults()))
          return false;
      }
      return true;
    });
    return bufferization::bufferizeOp(getOperation(), bufferizationOptions,
                                      /*copyBeforeWrite=*/false,
                                      &denseOpFilter);
  }

  void runOnOperation() override {
    {
      // Run enabling transformations.
      OpPassManager pm("builtin.module");
      pm.addPass(createPreSparsificationRewritePass());
      if (failed(runPipeline(pm, getOperation())))
        return signalPassFailure();
    }

    // Insert tensor copies. This step runs One-Shot Analysis (which analyzes
    // SSA use-def chains of tensor IR) and decides where buffer copies are
    // needed and where buffers can be written to in-place. These decisions are
    // materialized in the IR in the form of `bufferization.alloc_tensor` ops.
    //
    // Note: All following steps in this pass must be careful not to modify the
    // structure of the IR (i.e., tensor use-def chains), as that could
    // invalidate the results of the analysis. From now on, only small and
    // localized rewrites are allowed, such as replacing a tensor op with its
    // memref equivalent.
    if (failed(bufferization::insertTensorCopies(getOperation(),
                                                 bufferizationOptions)))
      return signalPassFailure();

    // `testAnalysisOnly` is a debug/testing flag. If set, the results of
    // OneShotAnalysis are added to the IR via attributes. In that case, do not
    // continue with the remaining pipeline.
    if (bufferizationOptions.testAnalysisOnly)
      return;

    // Bufferize all sparse ops. No further analysis is needed. All required
    // buffer copies were already inserted by `insertTensorCopies` in the form
    // of `bufferization.alloc_tensor` ops.
    {
      OpPassManager pm("builtin.module");
      pm.addPass(createSparsificationPass(sparsificationOptions));
      pm.addPass(createPostSparsificationRewritePass(enableRuntimeLibrary));
      if (enableRuntimeLibrary) {
        pm.addPass(
            createSparseTensorConversionPass(sparseTensorConversionOptions));
      } else {
        pm.addPass(createSparseTensorCodegenPass(enableBufferInitialization));
        pm.addPass(createSparseBufferRewritePass(enableBufferInitialization));
      }
      if (failed(runPipeline(pm, getOperation())))
        return signalPassFailure();
    }

    // Bufferize all dense ops.
    if (failed(runDenseBufferization()))
      signalPassFailure();
  }

private:
  bufferization::OneShotBufferizationOptions bufferizationOptions;
  SparsificationOptions sparsificationOptions;
  SparseTensorConversionOptions sparseTensorConversionOptions;
  bool enableRuntimeLibrary;
  bool enableBufferInitialization;
};
} // namespace sparse_tensor
} // namespace mlir

std::unique_ptr<Pass> mlir::createSparsificationAndBufferizationPass(
    const bufferization::OneShotBufferizationOptions &bufferizationOptions,
    const SparsificationOptions &sparsificationOptions,
    const SparseTensorConversionOptions &sparseTensorConversionOptions,
    bool enableRuntimeLibrary, bool enableBufferInitialization) {
  return std::make_unique<
      mlir::sparse_tensor::SparsificationAndBufferizationPass>(
      bufferizationOptions, sparsificationOptions,
      sparseTensorConversionOptions, enableRuntimeLibrary,
      enableBufferInitialization);
}
