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
  AliasingOpOperandList
  getAliasingOpOperands(Operation *op, OpResult opResult,
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
    return {{&yieldOp->getOpOperand(resultNum), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto assumingOp = cast<shape::AssumingOp>(op);
    assert(assumingOp.getDoRegion().getBlocks().size() == 1 &&
           "only 1 block supported");
    auto yieldOp = cast<shape::AssumingYieldOp>(
        assumingOp.getDoRegion().front().getTerminator());

    // Create new op and move over region.
    TypeRange newResultTypes(yieldOp.getOperands());
    auto newOp = rewriter.create<shape::AssumingOp>(
        op->getLoc(), newResultTypes, assumingOp.getWitness());
    newOp.getDoRegion().takeBody(assumingOp.getRegion());

    // Update all uses of the old op.
    rewriter.setInsertionPointAfter(newOp);
    SmallVector<Value> newResults;
    for (const auto &it : llvm::enumerate(assumingOp->getResultTypes())) {
      if (isa<TensorType>(it.value())) {
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    assert(isa<shape::AssumingOp>(op->getParentOp()) &&
           "expected that parent is an AssumingOp");
    OpResult opResult =
        op->getParentOp()->getResult(opOperand.getOperandNumber());
    return {{opResult, BufferRelation::Equivalent}};
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
    auto yieldOp = cast<shape::AssumingYieldOp>(op);
    SmallVector<Value> newResults;
    for (Value value : yieldOp.getOperands()) {
      if (isa<TensorType>(value.getType())) {
        FailureOr<Value> buffer = getBuffer(rewriter, value, options);
        if (failed(buffer))
          return failure();
        newResults.push_back(*buffer);
      } else {
        newResults.push_back(value);
      }
    }
    replaceOpWithNewBufferizedOp<shape::AssumingYieldOp>(rewriter, op,
                                                         newResults);
    return success();
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
