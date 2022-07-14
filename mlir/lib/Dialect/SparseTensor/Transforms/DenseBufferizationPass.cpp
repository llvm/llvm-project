//===- DenseBufferizationPass.cpp - Dense bufferization pass --------------===//
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

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

/// A pass that bufferizes only dense tensor ops and ignores all sparse tensor
/// ops. No buffer copies are inserted. All tensor OpOperands must be
/// inplacable.
class BufferizeDenseOpsPass
    : public PassWrapper<BufferizeDenseOpsPass, OperationPass<ModuleOp>> {
public:
  BufferizeDenseOpsPass(
      const bufferization::OneShotBufferizationOptions &options)
      : PassWrapper<BufferizeDenseOpsPass, OperationPass<ModuleOp>>(),
        options(options) {}

  void runOnOperation() override {
    // Disallow all sparse tensor ops, so that only dense tensor ops are
    // bufferized.
    bufferization::OpFilter opFilter;
    opFilter.allowOperation([&](Operation *op) {
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

    if (failed(bufferization::bufferizeOp(getOperation(), options,
                                          /*copyBeforeWrite=*/false,
                                          &opFilter)))
      signalPassFailure();
  }

private:
  bufferization::OneShotBufferizationOptions options;
};
} // namespace sparse_tensor
} // namespace mlir

std::unique_ptr<Pass> mlir::createDenseBufferizationPass(
    const bufferization::OneShotBufferizationOptions &options) {
  return std::make_unique<mlir::sparse_tensor::BufferizeDenseOpsPass>(options);
}
