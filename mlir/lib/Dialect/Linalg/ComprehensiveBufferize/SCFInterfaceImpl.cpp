//===- SCFInterfaceImpl.cpp - SCF Impl. of BufferizableOpInterface --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/SCFInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace scf_ext {

struct ExecuteRegionOpInterface
    : public BufferizableOpInterface::ExternalModel<ExecuteRegionOpInterface,
                                                    scf::ExecuteRegionOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    // ExecuteRegionOps do not have tensor OpOperands. The yielded value can be
    // any SSA value that is in scope. To allow for use-def chain traversal
    // through ExecuteRegionOps in the analysis, the corresponding yield value
    // is considered to be aliasing with the result.
    auto executeRegionOp = cast<scf::ExecuteRegionOp>(op);
    size_t resultNum = std::distance(op->getOpResults().begin(),
                                     llvm::find(op->getOpResults(), opResult));
    assert(executeRegionOp.region().getBlocks().size() == 1 &&
           "expected exactly 1 block");
    auto yieldOp = dyn_cast<scf::YieldOp>(
        executeRegionOp.region().front().getTerminator());
    assert(yieldOp && "expected scf.yield terminator in scf.execute_region");
    return {&yieldOp->getOpOperand(resultNum)};
  }

  bool mustBufferizeInPlace(Operation *op, OpResult opResult) const {
    // ExecuteRegionOp results always bufferize in-place. Since they have no
    // OpOperands, they are mostly ignored by the analysis once alias sets are
    // set up.
    return true;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    // TODO: Add bufferization support when needed. scf.execute_region should be
    // bufferized similar to scf.if.
    auto executeRegionOp = cast<scf::ExecuteRegionOp>(op);
    bool hasTensorReturnType = any_of(
        op->getResultTypes(), [](Type t) { return t.isa<TensorType>(); });
    if (hasTensorReturnType)
      return op->emitError(
          "scf.execute_region with tensor result not supported");
    return comprehensive_bufferize::bufferize(&executeRegionOp.region(), state);
  }
};

struct IfOpInterface
    : public BufferizableOpInterface::ExternalModel<IfOpInterface, scf::IfOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    // IfOps do not have tensor OpOperands. The yielded value can be any SSA
    // value that is in scope. To allow for use-def chain traversal through
    // IfOps in the analysis, both corresponding yield values from the then/else
    // branches are considered to be aliasing with the result.
    auto ifOp = cast<scf::IfOp>(op);
    size_t resultNum = std::distance(op->getOpResults().begin(),
                                     llvm::find(op->getOpResults(), opResult));
    return {&ifOp.thenYield()->getOpOperand(resultNum),
            &ifOp.elseYield()->getOpOperand(resultNum)};
  }

  // TODO: For better bufferization results, this could return `true` only if
  // there is a memory write in one (or both) of the branches. Since this is not
  // allowed at the moment, we should never encounter scf.ifs that yield
  // unmodified tensors. Such scf.yield ops could just fold away.
  bool isMemoryWrite(Operation *op, OpResult opResult) const {
    // IfOp results are always considered memory writes in the analysis. This
    // design decision simplifies the analysis considerably. E.g., consider the
    // following test case:
    //
    // %0 = "some_writing_op" : tensor<?xf32>
    // %r = scf.if %c -> (tensor<?xf32>) {
    //   scf.yield %0
    // } else {
    //   %1 = "another_writing_op"(%0) : tensor<?xf32>
    // }
    // "some_reading_op"(%r)
    //
    // "another_writing_op" in the above example should be able to bufferize
    // inplace in the absence of another read of %0. However, if the scf.if op
    // would not be considered a "write", the analysis would detect the
    // following conflict:
    //
    // * read = some_reading_op
    // * lastWrite = %0  (Note: The last write of %r would be a set: {%0, %1}.)
    // * conflictingWrite = %1
    //
    // For more details, check the "scf.IfOp" section of the design document.
    return true;
  }

  bool mustBufferizeInPlace(Operation *op, OpResult opResult) const {
    // IfOp results always bufferize in-place. Since they have no OpOperands,
    // they are mostly ignored by the analysis once alias sets are set up.
    return true;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto ifOp = cast<scf::IfOp>(op);

    // Bufferize then/else blocks.
    if (failed(comprehensive_bufferize::bufferize(ifOp.thenBlock(), state)))
      return failure();
    if (failed(comprehensive_bufferize::bufferize(ifOp.elseBlock(), state)))
      return failure();

    for (OpResult opResult : ifOp->getResults()) {
      if (!opResult.getType().isa<TensorType>())
        continue;
      // TODO: Atm we bail on unranked TensorType because we don't know how to
      // alloc an UnrankedMemRefType + its underlying ranked MemRefType.
      assert(opResult.getType().isa<RankedTensorType>() &&
             "unsupported unranked tensor");

      Value resultBuffer = getResultBuffer(b, opResult, state);
      if (!resultBuffer)
        return failure();

      state.aliasInfo.createAliasInfoEntry(resultBuffer);
      state.mapBuffer(opResult, resultBuffer);
    }

    return success();
  }
};

struct ForOpInterface
    : public BufferizableOpInterface::ExternalModel<ForOpInterface,
                                                    scf::ForOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    // scf::ForOp alone doesn't bufferize to a memory read, one of the uses of
    // its matching bbArg may.
    auto forOp = cast<scf::ForOp>(op);
    return isValueRead(forOp.getRegionIterArgForOpOperand(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    // Tensor iter_args of scf::ForOps are always considered as a write. This is
    // to simplify the analysis.
    // TODO: Consider doing sth. like isValueWritten.
    return true;
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    auto forOp = cast<scf::ForOp>(op);
    return {&forOp.getIterOpOperands()[opResult.getResultNumber()]};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    auto forOp = cast<scf::ForOp>(op);
    if (!opOperand.get().getType().isa<RankedTensorType>())
      return OpResult();
    return forOp.getResultForOpOperand(opOperand);
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
  }

  bool isWritable(Operation *op, Value value) const {
    // Interestingly, scf::ForOp's bbArg can **always** be viewed
    // inplace from the perspective of ops nested under:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto forOp = cast<scf::ForOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);

    for (OpResult opResult : forOp->getResults()) {
      if (!opResult.getType().isa<TensorType>())
        continue;
      // TODO: Atm we bail on unranked TensorType because we don't know how to
      // alloc an UnrankedMemRefType + its underlying ranked MemRefType.
      assert(opResult.getType().isa<RankedTensorType>() &&
             "unsupported unranked tensor");

      // TODO: More general: Matching bbArg does not bufferize to a read.
      Value resultBuffer = getResultBuffer(b, opResult, state);
      if (!resultBuffer)
        return failure();

      OpOperand &opOperand = forOp.getOpOperandForResult(opResult);
      BlockArgument bbArg = forOp.getRegionIterArgForOpOperand(opOperand);
      state.aliasInfo.createAliasInfoEntry(resultBuffer);
      state.aliasInfo.insertNewBufferEquivalence(bbArg, resultBuffer);
      state.mapBuffer(bbArg, resultBuffer);
      state.mapBuffer(opResult, resultBuffer);
    }

    // Bufferize loop body.
    if (failed(comprehensive_bufferize::bufferize(&forOp.region(), state)))
      return failure();

    // Finish bufferizing scf::ForOp.
    auto yieldOp = cast<scf::YieldOp>(&forOp.region().front().back());
    for (OpOperand &operand : yieldOp->getOpOperands()) {
      auto tensorType = operand.get().getType().dyn_cast<TensorType>();
      if (!tensorType)
        continue;

      OpOperand &forOperand = forOp.getOpOperandForResult(
          forOp->getResult(operand.getOperandNumber()));
      auto bbArg = forOp.getRegionIterArgForOpOperand(forOperand);
      Value yieldedBuffer = state.lookupBuffer(operand.get());
      Value bbArgBuffer = state.lookupBuffer(bbArg);
      if (!state.aliasInfo.areEquivalentBufferizedValues(yieldedBuffer,
                                                         bbArgBuffer)) {
        // TODO: this could get resolved with copies but it can also turn into
        // swaps so we need to be careful about order of copies.
        return yieldOp->emitError()
               << "Yield operand #" << operand.getOperandNumber()
               << " does not bufferize to an equivalent buffer to the matching"
               << " enclosing scf::for operand";
      }

      // Buffers are equivalent so the work is already done and we just yield
      // the bbArg so that it later canonicalizes away.
      operand.set(bbArg);
    }
    return success();
  }
};

struct YieldOpInterface
    : public BufferizableOpInterface::ExternalModel<YieldOpInterface,
                                                    scf::YieldOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto yieldOp = cast<scf::YieldOp>(op);
    if (!isa<scf::ExecuteRegionOp, scf::IfOp, scf::ForOp>(
            yieldOp->getParentOp()))
      return yieldOp->emitError("unsupported scf::YieldOp parent");
    return success();
  }
};

} // namespace scf_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::scf_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<scf::ExecuteRegionOp,
                          scf_ext::ExecuteRegionOpInterface>();
  registry.addOpInterface<scf::ForOp, scf_ext::ForOpInterface>();
  registry.addOpInterface<scf::IfOp, scf_ext::IfOpInterface>();
  registry.addOpInterface<scf::YieldOp, scf_ext::YieldOpInterface>();
  registry.addOpInterface<scf::ParallelOp,
                          AllocationHoistingBarrierOnly<scf::ParallelOp>>();
}
