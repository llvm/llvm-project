//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::shape;

namespace mlir {
namespace shape {
namespace {

/// Bufferization of shape.assuming.
struct AssumingOpInterface
    : public BufferizableOpInterface::ExternalModel<AssumingOpInterface,
                                                    shape::AssumingOp> {
  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    // AssumingOps do not have tensor OpOperands. The yielded value can be any
    // SSA value that is in scope. To allow for use-def chain traversal through
    // AssumingOps in the analysis, the corresponding yield value is considered
    // to be aliasing with the result.
    auto assumingOp = cast<shape::AssumingOp>(op);
    size_t resultNum = std::distance(op->getOpResults().begin(),
                                     llvm::find(op->getOpResults(), opResult));
    // TODO: Support multiple blocks.
    assert(assumingOp.getDoRegion().getBlocks().size() == 1 &&
           "expected exactly 1 block");
    auto yieldOp = dyn_cast<shape::AssumingYieldOp>(
        assumingOp.getDoRegion().front().getTerminator());
    assert(yieldOp && "expected shape.assuming_yield terminator");
    return {&yieldOp->getOpOperand(resultNum)};
  }

  // TODO: For better bufferization results, this could return `true` only if
  // there is a memory write in the region.
  bool isMemoryWrite(Operation *op, OpResult opResult,
                     const AnalysisState &state) const {
    // Similar to scf.if, results of this op are always considered memory writes
    // in the analysis. This is a useful pattern for all ops that have tensor
    // OpResults but no tensor OpOperands. By default, `isMemoryWrite` is
    // implemented in terms of `bufferizesToMemoryWrite`, which does not work on
    // ops without OpOperands.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto assumingOp = cast<shape::AssumingOp>(op);

    // Compute new result types.
    SmallVector<Type> newResultTypes;
    for (Type type : assumingOp->getResultTypes()) {
      if (auto tensorType = type.dyn_cast<TensorType>()) {
        // TODO: Infer the result type instead of computing it.
        newResultTypes.push_back(getMemRefType(tensorType, options));
      } else {
        newResultTypes.push_back(type);
      }
    }

    // Create new op and move over region.
    auto newOp = rewriter.create<shape::AssumingOp>(
        op->getLoc(), newResultTypes, assumingOp.getWitness());
    newOp.getDoRegion().takeBody(assumingOp.getRegion());

    // Update terminator.
    assert(newOp.getDoRegion().getBlocks().size() == 1 &&
           "only 1 block supported");
    Block *newBlock = &newOp.getDoRegion().front();
    auto yieldOp = cast<shape::AssumingYieldOp>(newBlock->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    SmallVector<Value> newYieldValues;
    for (const auto &it : llvm::enumerate(yieldOp.operands())) {
      Value val = it.value();
      if (val.getType().isa<TensorType>()) {
        newYieldValues.push_back(rewriter.create<bufferization::ToMemrefOp>(
            yieldOp.getLoc(), newResultTypes[it.index()], val));
      } else {
        newYieldValues.push_back(val);
      }
    }
    rewriter.replaceOpWithNewOp<shape::AssumingYieldOp>(yieldOp,
                                                        newYieldValues);

    // Update all uses of the old op.
    rewriter.setInsertionPointAfter(newOp);
    SmallVector<Value> newResults;
    for (const auto &it : llvm::enumerate(assumingOp->getResultTypes())) {
      if (it.value().isa<TensorType>()) {
        newResults.push_back(rewriter.create<bufferization::ToTensorOp>(
            assumingOp.getLoc(), newOp->getResult(it.index())));
      } else {
        newResults.push_back(newOp->getResult(it.index()));
      }
    }

    // Replace old op.
    rewriter.replaceOp(assumingOp, newResults);

    return success();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }
};

/// Bufferization of shape.assuming_yield. Bufferized as part of their enclosing
/// ops, so this is for analysis only.
struct AssumingYieldOpInterface
    : public BufferizableOpInterface::ExternalModel<AssumingYieldOpInterface,
                                                    shape::AssumingYieldOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    assert(isa<shape::AssumingOp>(op->getParentOp()) &&
           "expected that parent is an AssumingOp");
    return {op->getParentOp()->getResult(opOperand.getOperandNumber())};
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    // Yield operands always bufferize inplace. Otherwise, an alloc + copy
    // may be generated inside the block. We should not return/yield allocations
    // when possible.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // Op is bufferized as part of AssumingOp.
    return failure();
  }
};

} // namespace
} // namespace shape
} // namespace mlir

void mlir::shape::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, shape::ShapeDialect *dialect) {
    shape::AssumingOp::attachInterface<AssumingOpInterface>(*ctx);
    shape::AssumingYieldOp::attachInterface<AssumingYieldOpInterface>(*ctx);
  });
}
