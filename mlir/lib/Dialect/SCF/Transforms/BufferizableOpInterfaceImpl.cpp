//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::scf;

namespace mlir {
namespace scf {
namespace {

// bufferization.to_memref is not allowed to change the rank.
static void ensureToMemrefOpIsValid(Value tensor, Type memrefType) {
#ifndef NDEBUG
  auto rankedTensorType = tensor.getType().dyn_cast<RankedTensorType>();
  assert((!rankedTensorType || (memrefType.cast<MemRefType>().getRank() ==
                                rankedTensorType.getRank())) &&
         "to_memref would be invalid: mismatching ranks");
#endif
}

/// Bufferization of scf.execute_region. Can be analyzed, but bufferization not
/// fully implemented at the moment.
struct ExecuteRegionOpInterface
    : public BufferizableOpInterface::ExternalModel<ExecuteRegionOpInterface,
                                                    scf::ExecuteRegionOp> {
  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    // ExecuteRegionOps do not have tensor OpOperands. The yielded value can be
    // any SSA value that is in scope. To allow for use-def chain traversal
    // through ExecuteRegionOps in the analysis, the corresponding yield value
    // is considered to be aliasing with the result.
    auto executeRegionOp = cast<scf::ExecuteRegionOp>(op);
    size_t resultNum = std::distance(op->getOpResults().begin(),
                                     llvm::find(op->getOpResults(), opResult));
    // TODO: Support multiple blocks.
    assert(executeRegionOp.getRegion().getBlocks().size() == 1 &&
           "expected exactly 1 block");
    auto yieldOp = dyn_cast<scf::YieldOp>(
        executeRegionOp.getRegion().front().getTerminator());
    assert(yieldOp && "expected scf.yield terminator in scf.execute_region");
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
    auto executeRegionOp = cast<scf::ExecuteRegionOp>(op);
    assert(executeRegionOp.getRegion().getBlocks().size() == 1 &&
           "only 1 block supported");
    auto yieldOp =
        cast<scf::YieldOp>(executeRegionOp.getRegion().front().getTerminator());
    TypeRange newResultTypes(yieldOp.getResults());

    // Create new op and move over region.
    auto newOp =
        rewriter.create<scf::ExecuteRegionOp>(op->getLoc(), newResultTypes);
    newOp.getRegion().takeBody(executeRegionOp.getRegion());

    // Update all uses of the old op.
    rewriter.setInsertionPointAfter(newOp);
    SmallVector<Value> newResults;
    for (const auto &it : llvm::enumerate(executeRegionOp->getResultTypes())) {
      if (it.value().isa<TensorType>()) {
        newResults.push_back(rewriter.create<bufferization::ToTensorOp>(
            executeRegionOp.getLoc(), newOp->getResult(it.index())));
      } else {
        newResults.push_back(newOp->getResult(it.index()));
      }
    }

    // Replace old op.
    rewriter.replaceOp(executeRegionOp, newResults);

    return success();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }
};

/// Bufferization of scf.if. Replace with a new scf.if that yields memrefs.
struct IfOpInterface
    : public BufferizableOpInterface::ExternalModel<IfOpInterface, scf::IfOp> {
  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
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
  bool isMemoryWrite(Operation *op, OpResult opResult,
                     const AnalysisState &state) const {
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

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    OpBuilder::InsertionGuard g(rewriter);
    auto ifOp = cast<scf::IfOp>(op);
    auto thenYieldOp = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYieldOp = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

    // Reconcile type mismatches between then/else branches by inserting memref
    // casts.
    SmallVector<Value> thenResults, elseResults;
    bool insertedCast = false;
    for (unsigned i = 0; i < thenYieldOp.getResults().size(); ++i) {
      Value thenValue = thenYieldOp.getResults()[i];
      Value elseValue = elseYieldOp.getResults()[i];
      if (thenValue.getType() == elseValue.getType()) {
        thenResults.push_back(thenValue);
        elseResults.push_back(elseValue);
        continue;
      }

      // Type mismatch between then/else yield value. Cast both to a memref type
      // with a fully dynamic layout map.
      auto thenMemrefType = thenValue.getType().cast<BaseMemRefType>();
      auto elseMemrefType = elseValue.getType().cast<BaseMemRefType>();
      if (thenMemrefType.getMemorySpaceAsInt() !=
          elseMemrefType.getMemorySpaceAsInt())
        return op->emitError("inconsistent memory space on then/else branches");
      rewriter.setInsertionPoint(thenYieldOp);
      BaseMemRefType memrefType = getMemRefTypeWithFullyDynamicLayout(
          ifOp.getResultTypes()[i].cast<TensorType>(),
          thenMemrefType.getMemorySpaceAsInt());
      thenResults.push_back(rewriter.create<memref::CastOp>(
          thenYieldOp.getLoc(), memrefType, thenValue));
      rewriter.setInsertionPoint(elseYieldOp);
      elseResults.push_back(rewriter.create<memref::CastOp>(
          elseYieldOp.getLoc(), memrefType, elseValue));
      insertedCast = true;
    }

    if (insertedCast) {
      rewriter.setInsertionPoint(thenYieldOp);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(thenYieldOp, thenResults);
      rewriter.setInsertionPoint(elseYieldOp);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(elseYieldOp, elseResults);
    }

    // Create new op.
    rewriter.setInsertionPoint(ifOp);
    ValueRange resultsValueRange(thenResults);
    TypeRange newTypes(resultsValueRange);
    auto newIfOp =
        rewriter.create<scf::IfOp>(ifOp.getLoc(), newTypes, ifOp.getCondition(),
                                   /*withElseRegion=*/true);

    // Move over then/else blocks.
    rewriter.mergeBlocks(ifOp.thenBlock(), newIfOp.thenBlock());
    rewriter.mergeBlocks(ifOp.elseBlock(), newIfOp.elseBlock());

    // Replace op results.
    replaceOpWithBufferizedValues(rewriter, op, newIfOp->getResults());

    return success();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    // IfOp results are equivalent to their corresponding yield values if both
    // yield values are equivalent to each other.
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    SmallVector<OpOperand *> yieldValues =
        bufferizableOp.getAliasingOpOperand(opResult, state);
    assert(yieldValues.size() == 2 && "expected 2 yield values");
    bool equivalentYields = state.areEquivalentBufferizedValues(
        yieldValues[0]->get(), yieldValues[1]->get());
    return equivalentYields ? BufferRelation::Equivalent : BufferRelation::None;
  }
};

/// Helper function for loop bufferization. Return the indices of all values
/// that have a tensor type.
static DenseSet<int64_t> getTensorIndices(ValueRange values) {
  DenseSet<int64_t> result;
  for (const auto &it : llvm::enumerate(values))
    if (it.value().getType().isa<TensorType>())
      result.insert(it.index());
  return result;
}

/// Helper function for loop bufferization. Return the indices of all
/// bbArg/yielded value pairs who's buffer relation is "Equivalent".
DenseSet<int64_t> getEquivalentBuffers(Block::BlockArgListType bbArgs,
                                       ValueRange yieldedValues,
                                       const AnalysisState &state) {
  unsigned int minSize = std::min(bbArgs.size(), yieldedValues.size());
  DenseSet<int64_t> result;
  for (unsigned int i = 0; i < minSize; ++i) {
    if (!bbArgs[i].getType().isa<TensorType>() ||
        !yieldedValues[i].getType().isa<TensorType>())
      continue;
    if (state.areEquivalentBufferizedValues(bbArgs[i], yieldedValues[i]))
      result.insert(i);
  }
  return result;
}

/// Helper function for loop bufferization. Cast the given buffer to the given
/// memref type.
static Value castBuffer(OpBuilder &b, Value buffer, Type type) {
  assert(type.isa<BaseMemRefType>() && "expected BaseMemRefType");
  assert(buffer.getType().isa<BaseMemRefType>() && "expected BaseMemRefType");
  // If the buffer already has the correct type, no cast is needed.
  if (buffer.getType() == type)
    return buffer;
  // TODO: In case `type` has a layout map that is not the fully dynamic
  // one, we may not be able to cast the buffer. In that case, the loop
  // iter_arg's layout map must be changed (see uses of `castBuffer`).
  assert(memref::CastOp::areCastCompatible(buffer.getType(), type) &&
         "scf.while op bufferization: cast incompatible");
  return b.create<memref::CastOp>(buffer.getLoc(), type, buffer).getResult();
}

/// Helper function for loop bufferization. Return the bufferized values of the
/// given OpOperands. If an operand is not a tensor, return the original value.
static SmallVector<Value> getBuffers(RewriterBase &rewriter,
                                     MutableArrayRef<OpOperand> operands,
                                     const BufferizationOptions &options) {
  SmallVector<Value> result;
  for (OpOperand &opOperand : operands) {
    if (opOperand.get().getType().isa<TensorType>()) {
      Value resultBuffer = getBuffer(rewriter, opOperand.get(), options);
      result.push_back(resultBuffer);
    } else {
      result.push_back(opOperand.get());
    }
  }
  return result;
}

/// Helper function for loop bufferization. Compute the buffer that should be
/// yielded from a loop block (loop body or loop condition).
static Value getYieldedBuffer(RewriterBase &rewriter, Value tensor,
                              BaseMemRefType type,
                              const BufferizationOptions &options) {
  assert(tensor.getType().isa<TensorType>() && "expected tensor");
  ensureToMemrefOpIsValid(tensor, type);
  Value yieldedVal = getBuffer(rewriter, tensor, options);
  return castBuffer(rewriter, yieldedVal, type);
}

/// Helper function for loop bufferization. Given a range of values, apply
/// `func` to those marked in `tensorIndices`. Otherwise, store the unmodified
/// value in the result vector.
static SmallVector<Value>
convertTensorValues(ValueRange values, const DenseSet<int64_t> &tensorIndices,
                    llvm::function_ref<Value(Value, int64_t)> func) {
  SmallVector<Value> result;
  for (const auto &it : llvm::enumerate(values)) {
    size_t idx = it.index();
    Value val = it.value();
    result.push_back(tensorIndices.contains(idx) ? func(val, idx) : val);
  }
  return result;
}

/// Helper function for loop bufferization. Given a list of pre-bufferization
/// yielded values, compute the list of bufferized yielded values.
SmallVector<Value> getYieldedValues(RewriterBase &rewriter, ValueRange values,
                                    TypeRange bufferizedTypes,
                                    const DenseSet<int64_t> &tensorIndices,
                                    const BufferizationOptions &options) {
  return convertTensorValues(
      values, tensorIndices, [&](Value val, int64_t index) {
        return getYieldedBuffer(rewriter, val,
                                bufferizedTypes[index].cast<BaseMemRefType>(),
                                options);
      });
}

/// Helper function for loop bufferization. Given a list of bbArgs of the new
/// (bufferized) loop op, wrap the bufferized tensor args (now memrefs) into
/// ToTensorOps, so that the block body can be moved over to the new op.
SmallVector<Value>
getBbArgReplacements(RewriterBase &rewriter, Block::BlockArgListType bbArgs,
                     const DenseSet<int64_t> &tensorIndices) {
  return convertTensorValues(
      bbArgs, tensorIndices, [&](Value val, int64_t index) {
        return rewriter.create<bufferization::ToTensorOp>(val.getLoc(), val);
      });
}

/// Bufferization of scf.for. Replace with a new scf.for that operates on
/// memrefs.
struct ForOpInterface
    : public BufferizableOpInterface::ExternalModel<ForOpInterface,
                                                    scf::ForOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // scf::ForOp alone doesn't bufferize to a memory read, one of the uses of
    // its matching bbArg may.
    auto forOp = cast<scf::ForOp>(op);
    return state.isValueRead(forOp.getRegionIterArgForOpOperand(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Tensor iter_args of scf::ForOps are always considered as a write.
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    auto forOp = cast<scf::ForOp>(op);
    return {forOp.getResultForOpOperand(opOperand)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    // ForOp results are equivalent to their corresponding init_args if the
    // corresponding iter_args and yield values are equivalent.
    auto forOp = cast<scf::ForOp>(op);
    OpOperand &forOperand = forOp.getOpOperandForResult(opResult);
    auto bbArg = forOp.getRegionIterArgForOpOperand(forOperand);
    auto yieldOp =
        cast<scf::YieldOp>(forOp.getLoopBody().front().getTerminator());
    bool equivalentYield = state.areEquivalentBufferizedValues(
        bbArg, yieldOp->getOperand(opResult.getResultNumber()));
    return equivalentYield ? BufferRelation::Equivalent : BufferRelation::None;
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    // Interestingly, scf::ForOp's bbArg can **always** be viewed
    // inplace from the perspective of ops nested under:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  LogicalResult resolveConflicts(Operation *op, RewriterBase &rewriter,
                                 const AnalysisState &state) const {
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    if (failed(bufferizableOp.resolveTensorOpOperandConflicts(rewriter, state)))
      return failure();

    if (!state.getOptions().enforceAliasingInvariants)
      return success();

    // According to the `getAliasing...` implementations, a bufferized OpResult
    // may alias only with the corresponding bufferized init_arg and with no
    // other buffers. I.e., the i-th OpResult may alias with the i-th init_arg;
    // but not with any other OpOperand. If a corresponding OpResult/init_arg
    // pair bufferizes to equivalent buffers, this aliasing requirement is
    // satisfied. Otherwise, we cannot be sure and must yield a new buffer copy.
    // (New buffer copies do not alias with any buffer.)
    auto forOp = cast<scf::ForOp>(op);
    auto yieldOp =
        cast<scf::YieldOp>(forOp.getLoopBody().front().getTerminator());
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(yieldOp);

    // Indices of all iter_args that have tensor type. These are the ones that
    // are bufferized.
    DenseSet<int64_t> indices = getTensorIndices(forOp.getInitArgs());
    // For every yielded value, is the value equivalent to its corresponding
    // bbArg?
    DenseSet<int64_t> equivalentYields = getEquivalentBuffers(
        forOp.getRegionIterArgs(), yieldOp.getResults(), state);
    SmallVector<Value> yieldValues;
    for (int64_t idx = 0;
         idx < static_cast<int64_t>(yieldOp.getResults().size()); ++idx) {
      Value value = yieldOp.getResults()[idx];
      if (!indices.contains(idx) || equivalentYields.contains(idx)) {
        yieldValues.push_back(value);
        continue;
      }
      Value alloc = allocateTensorForShapedValue(rewriter, yieldOp.getLoc(),
                                                 value, /*escape=*/true);
      yieldValues.push_back(alloc);
    }

    rewriter.updateRootInPlace(
        yieldOp, [&]() { yieldOp.getResultsMutable().assign(yieldValues); });
    return success();
  }

  BaseMemRefType getBufferType(Operation *op, BlockArgument bbArg,
                               const BufferizationOptions &options) const {
    auto forOp = cast<scf::ForOp>(op);
    return bufferization::getBufferType(
        forOp.getOpOperandForRegionIterArg(bbArg).get(), options);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto forOp = cast<scf::ForOp>(op);
    Block *oldLoopBody = &forOp.getLoopBody().front();

    // Indices of all iter_args that have tensor type. These are the ones that
    // are bufferized.
    DenseSet<int64_t> indices = getTensorIndices(forOp.getInitArgs());

    // The new memref init_args of the loop.
    SmallVector<Value> initArgs =
        getBuffers(rewriter, forOp.getIterOpOperands(), options);

    // Construct a new scf.for op with memref instead of tensor values.
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), initArgs);
    newForOp->setAttrs(forOp->getAttrs());
    ValueRange initArgsRange(initArgs);
    TypeRange initArgsTypes(initArgsRange);
    Block *loopBody = &newForOp.getLoopBody().front();

    // Set up new iter_args. The loop body uses tensors, so wrap the (memref)
    // iter_args of the new loop in ToTensorOps.
    rewriter.setInsertionPointToStart(loopBody);
    SmallVector<Value> iterArgs =
        getBbArgReplacements(rewriter, newForOp.getRegionIterArgs(), indices);
    iterArgs.insert(iterArgs.begin(), newForOp.getInductionVar());

    // Move loop body to new loop.
    rewriter.mergeBlocks(oldLoopBody, loopBody, iterArgs);

    // Replace loop results.
    replaceOpWithBufferizedValues(rewriter, op, newForOp->getResults());

    return success();
  }

  /// Assert that yielded values of an scf.for op are equivalent to their
  /// corresponding bbArgs. In that case, the buffer relations of the
  /// corresponding OpResults are "Equivalent".
  ///
  /// If this is not the case, an allocs+copies are inserted and yielded from
  /// the loop. This could be a performance problem, so it must be explicitly
  /// activated with `alloc-return-allocs`.
  LogicalResult verifyAnalysis(Operation *op,
                               const AnalysisState &state) const {
    const auto &options =
        static_cast<const OneShotBufferizationOptions &>(state.getOptions());
    if (options.allowReturnAllocs)
      return success();

    auto forOp = cast<scf::ForOp>(op);
    auto yieldOp =
        cast<scf::YieldOp>(forOp.getLoopBody().front().getTerminator());
    for (OpResult opResult : op->getOpResults()) {
      if (!opResult.getType().isa<TensorType>())
        continue;

      // Note: This is overly strict. We should check for aliasing bufferized
      // values. But we don't have a "must-alias" analysis yet.
      if (bufferRelation(op, opResult, state) != BufferRelation::Equivalent)
        return yieldOp->emitError()
               << "Yield operand #" << opResult.getResultNumber()
               << " is not equivalent to the corresponding iter bbArg";
    }

    return success();
  }
};

/// Bufferization of scf.while. Replace with a new scf.while that operates on
/// memrefs.
struct WhileOpInterface
    : public BufferizableOpInterface::ExternalModel<WhileOpInterface,
                                                    scf::WhileOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Tensor iter_args of scf::WhileOps are always considered as a read.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Tensor iter_args of scf::WhileOps are always considered as a write.
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    auto whileOp = cast<scf::WhileOp>(op);
    unsigned int idx = opOperand.getOperandNumber();

    // The OpResults and OpOperands may not match. They may not even have the
    // same type. The number of OpResults and OpOperands can also differ.
    if (idx >= op->getNumResults() ||
        opOperand.get().getType() != op->getResult(idx).getType())
      return {};

    // The only aliasing OpResult may be the one at the same index.
    return {whileOp->getResult(idx)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    // WhileOp results are equivalent to their corresponding init_args if the
    // corresponding iter_args and yield values are equivalent (for both the
    // "before" and the "after" block).
    unsigned int resultNumber = opResult.getResultNumber();
    auto whileOp = cast<scf::WhileOp>(op);

    // The "before" region bbArgs and the OpResults may not match.
    if (resultNumber >= whileOp.getBeforeArguments().size())
      return BufferRelation::None;
    if (opResult.getType() !=
        whileOp.getBeforeArguments()[resultNumber].getType())
      return BufferRelation::None;

    auto conditionOp = whileOp.getConditionOp();
    BlockArgument conditionBbArg = whileOp.getBeforeArguments()[resultNumber];
    Value conditionOperand = conditionOp.getArgs()[resultNumber];
    bool equivCondition =
        state.areEquivalentBufferizedValues(conditionBbArg, conditionOperand);

    auto yieldOp = whileOp.getYieldOp();
    BlockArgument bodyBbArg = whileOp.getAfterArguments()[resultNumber];
    Value yieldOperand = yieldOp.getOperand(resultNumber);
    bool equivYield =
        state.areEquivalentBufferizedValues(bodyBbArg, yieldOperand);

    return equivCondition && equivYield ? BufferRelation::Equivalent
                                        : BufferRelation::None;
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    // Interestingly, scf::WhileOp's bbArg can **always** be viewed
    // inplace from the perspective of ops nested under:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  LogicalResult resolveConflicts(Operation *op, RewriterBase &rewriter,
                                 const AnalysisState &state) const {
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    if (failed(bufferizableOp.resolveTensorOpOperandConflicts(rewriter, state)))
      return failure();

    if (!state.getOptions().enforceAliasingInvariants)
      return success();

    // According to the `getAliasing...` implementations, a bufferized OpResult
    // may alias only with the corresponding bufferized init_arg and with no
    // other buffers. I.e., the i-th OpResult may alias with the i-th init_arg;
    // but not with any other OpOperand. If a corresponding OpResult/init_arg
    // pair bufferizes to equivalent buffers, this aliasing requirement is
    // satisfied. Otherwise, we cannot be sure and must yield a new buffer copy.
    // (New buffer copies do not alias with any buffer.)
    OpBuilder::InsertionGuard g(rewriter);
    auto whileOp = cast<scf::WhileOp>(op);
    auto conditionOp = whileOp.getConditionOp();
    auto yieldOp = whileOp.getYieldOp();

    // Indices of all bbArgs that have tensor type. These are the ones that
    // are bufferized. The "before" and "after" regions may have different args.
    DenseSet<int64_t> indicesBefore = getTensorIndices(whileOp.getInits());
    DenseSet<int64_t> indicesAfter =
        getTensorIndices(whileOp.getAfterArguments());

    // For every yielded value, is the value equivalent to its corresponding
    // bbArg?
    DenseSet<int64_t> equivalentYieldsBefore = getEquivalentBuffers(
        whileOp.getBeforeArguments(), conditionOp.getArgs(), state);
    DenseSet<int64_t> equivalentYieldsAfter = getEquivalentBuffers(
        whileOp.getAfterArguments(), whileOp.getYieldOp().getResults(), state);

    // Update "before" region.
    rewriter.setInsertionPoint(conditionOp);
    SmallVector<Value> beforeYieldValues;
    for (int64_t idx = 0;
         idx < static_cast<int64_t>(conditionOp.getArgs().size()); ++idx) {
      Value value = conditionOp.getArgs()[idx];
      if (!indicesBefore.contains(idx) ||
          equivalentYieldsBefore.contains(idx)) {
        beforeYieldValues.push_back(value);
        continue;
      }
      Value alloc = allocateTensorForShapedValue(rewriter, conditionOp.getLoc(),
                                                 value, /*escape=*/true);
      beforeYieldValues.push_back(alloc);
    }
    rewriter.updateRootInPlace(conditionOp, [&]() {
      conditionOp.getArgsMutable().assign(beforeYieldValues);
    });

    // Update "after" region.
    rewriter.setInsertionPoint(yieldOp);
    SmallVector<Value> afterYieldValues;
    for (int64_t idx = 0;
         idx < static_cast<int64_t>(yieldOp.getResults().size()); ++idx) {
      Value value = yieldOp.getResults()[idx];
      if (!indicesAfter.contains(idx) || equivalentYieldsAfter.contains(idx)) {
        afterYieldValues.push_back(value);
        continue;
      }
      Value alloc = allocateTensorForShapedValue(rewriter, yieldOp.getLoc(),
                                                 value, /*escape=*/true);
      afterYieldValues.push_back(alloc);
    }
    rewriter.updateRootInPlace(yieldOp, [&]() {
      yieldOp.getResultsMutable().assign(afterYieldValues);
    });

    return success();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto whileOp = cast<scf::WhileOp>(op);

    assert(whileOp.getBefore().getBlocks().size() == 1 &&
           "regions with multiple blocks not supported");
    Block *beforeBody = &whileOp.getBefore().front();
    assert(whileOp.getAfter().getBlocks().size() == 1 &&
           "regions with multiple blocks not supported");
    Block *afterBody = &whileOp.getAfter().front();

    // Indices of all bbArgs that have tensor type. These are the ones that
    // are bufferized. The "before" and "after" regions may have different args.
    DenseSet<int64_t> indicesBefore = getTensorIndices(whileOp.getInits());
    DenseSet<int64_t> indicesAfter =
        getTensorIndices(whileOp.getAfterArguments());

    // The new memref init_args of the loop.
    SmallVector<Value> initArgs =
        getBuffers(rewriter, whileOp->getOpOperands(), options);

    // The result types of a WhileOp are the same as the "after" bbArg types.
    SmallVector<Type> argsTypesAfter = llvm::to_vector(
        llvm::map_range(whileOp.getAfterArguments(), [&](BlockArgument bbArg) {
          return bufferization::getBufferType(bbArg, options).cast<Type>();
        }));

    // Construct a new scf.while op with memref instead of tensor values.
    ValueRange argsRangeBefore(initArgs);
    TypeRange argsTypesBefore(argsRangeBefore);
    auto newWhileOp = rewriter.create<scf::WhileOp>(whileOp.getLoc(),
                                                    argsTypesAfter, initArgs);

    // Add before/after regions to the new op.
    SmallVector<Location> bbArgLocsBefore(initArgs.size(), whileOp.getLoc());
    SmallVector<Location> bbArgLocsAfter(argsTypesAfter.size(),
                                         whileOp.getLoc());
    Block *newBeforeBody = &newWhileOp.getBefore().emplaceBlock();
    newWhileOp.getBefore().addArguments(argsTypesBefore, bbArgLocsBefore);
    Block *newAfterBody = &newWhileOp.getAfter().emplaceBlock();
    newWhileOp.getAfter().addArguments(argsTypesAfter, bbArgLocsAfter);

    // Set up new iter_args and move the loop condition block to the new op.
    // The old block uses tensors, so wrap the (memref) bbArgs of the new block
    // in ToTensorOps.
    rewriter.setInsertionPointToStart(newBeforeBody);
    SmallVector<Value> newBeforeArgs = getBbArgReplacements(
        rewriter, newWhileOp.getBeforeArguments(), indicesBefore);
    rewriter.mergeBlocks(beforeBody, newBeforeBody, newBeforeArgs);

    // Update scf.condition of new loop.
    auto newConditionOp = newWhileOp.getConditionOp();
    rewriter.setInsertionPoint(newConditionOp);
    // Only equivalent buffers or new buffer allocations may be yielded to the
    // "after" region.
    // TODO: This could be relaxed for better bufferization results.
    SmallVector<Value> newConditionArgs =
        getYieldedValues(rewriter, newConditionOp.getArgs(), argsTypesAfter,
                         indicesAfter, options);
    newConditionOp.getArgsMutable().assign(newConditionArgs);

    // Set up new iter_args and move the loop body block to the new op.
    // The old block uses tensors, so wrap the (memref) bbArgs of the new block
    // in ToTensorOps.
    rewriter.setInsertionPointToStart(newAfterBody);
    SmallVector<Value> newAfterArgs = getBbArgReplacements(
        rewriter, newWhileOp.getAfterArguments(), indicesAfter);
    rewriter.mergeBlocks(afterBody, newAfterBody, newAfterArgs);

    // Update scf.yield of the new loop.
    auto newYieldOp = newWhileOp.getYieldOp();
    rewriter.setInsertionPoint(newYieldOp);
    // Only equivalent buffers or new buffer allocations may be yielded to the
    // "before" region.
    // TODO: This could be relaxed for better bufferization results.
    SmallVector<Value> newYieldValues =
        getYieldedValues(rewriter, newYieldOp.getResults(), argsTypesBefore,
                         indicesBefore, options);
    newYieldOp.getResultsMutable().assign(newYieldValues);

    // Replace loop results.
    replaceOpWithBufferizedValues(rewriter, op, newWhileOp->getResults());

    return success();
  }

  /// Assert that yielded values of an scf.while op are equivalent to their
  /// corresponding bbArgs. In that case, the buffer relations of the
  /// corresponding OpResults are "Equivalent".
  ///
  /// If this is not the case, allocs+copies are inserted and yielded from
  /// the loop. This could be a performance problem, so it must be explicitly
  /// activated with `alloc-return-allocs`.
  ///
  /// Not: In contrast to scf::ForOp, scf::WhileOp has two regions and the
  /// equivalence condition must be checked for both.
  LogicalResult verifyAnalysis(Operation *op,
                               const AnalysisState &state) const {
    auto whileOp = cast<scf::WhileOp>(op);
    const auto &options =
        static_cast<const OneShotBufferizationOptions &>(state.getOptions());
    if (options.allowReturnAllocs)
      return success();

    auto conditionOp = whileOp.getConditionOp();
    for (const auto &it : llvm::enumerate(conditionOp.getArgs())) {
      if (!it.value().getType().isa<TensorType>())
        continue;
      if (!state.areEquivalentBufferizedValues(
              it.value(), conditionOp->getBlock()->getArgument(it.index())))
        return conditionOp->emitError()
               << "Condition arg #" << it.index()
               << " is not equivalent to the corresponding iter bbArg";
    }

    auto yieldOp = whileOp.getYieldOp();
    for (const auto &it : llvm::enumerate(yieldOp.getResults())) {
      if (!it.value().getType().isa<TensorType>())
        continue;
      if (!state.areEquivalentBufferizedValues(
              it.value(), yieldOp->getBlock()->getArgument(it.index())))
        return yieldOp->emitError()
               << "Yield operand #" << it.index()
               << " is not equivalent to the corresponding iter bbArg";
    }

    return success();
  }
};

/// Bufferization of scf.yield. Bufferized as part of their enclosing ops, so
/// this is for analysis only.
struct YieldOpInterface
    : public BufferizableOpInterface::ExternalModel<YieldOpInterface,
                                                    scf::YieldOp> {
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
    if (isa<scf::IfOp>(op->getParentOp()))
      return {op->getParentOp()->getResult(opOperand.getOperandNumber())};
    if (isa<scf::ExecuteRegionOp>(op->getParentOp()))
      return {op->getParentOp()->getResult(opOperand.getOperandNumber())};
    return {};
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
    auto yieldOp = cast<scf::YieldOp>(op);
    if (!isa<scf::ExecuteRegionOp, scf::IfOp, scf::ForOp, scf::WhileOp>(
            yieldOp->getParentOp()))
      return yieldOp->emitError("unsupported scf::YieldOp parent");

    // TODO: Bufferize scf.yield inside scf.while here. (Currently bufferized
    // together with scf.while.)
    if (isa<scf::WhileOp>(yieldOp->getParentOp()))
      return success();

    SmallVector<Value> newResults;
    for (const auto &it : llvm::enumerate(yieldOp.getResults())) {
      Value value = it.value();
      if (value.getType().isa<TensorType>()) {
        Value buffer = getBuffer(rewriter, value, options);
        if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
          BaseMemRefType resultType =
              cast<BufferizableOpInterface>(forOp.getOperation())
                  .getBufferType(forOp.getRegionIterArgs()[it.index()],
                                 options);
          buffer = castBuffer(rewriter, buffer, resultType);
        }
        newResults.push_back(buffer);
      } else {
        newResults.push_back(value);
      }
    }

    replaceOpWithNewBufferizedOp<scf::YieldOp>(rewriter, op, newResults);
    return success();
  }
};

using tensor::ExtractSliceOp;

/// Return the destinations that an ForeachThreadOp is inserting into. One per
/// ParallelInsertSliceOp.
static SmallVector<OpOperand *>
getInsertionDest(ForeachThreadOp foreachThreadOp) {
  PerformConcurrentlyOp terminator = foreachThreadOp.getTerminator();
  SmallVector<OpOperand *> result;
  terminator.walk([&](ParallelInsertSliceOp insertOp) {
    result.push_back(&insertOp->getOpOperand(1) /*dest*/);
  });
  return result;
}

/// Bufferization of ForeachThreadOp. This also bufferizes the terminator of the
/// region. There are op interfaces for the terminators (PerformConcurrentlyOp
/// and ParallelInsertSliceOp), but these are only used during analysis. Not
/// for bufferization.
struct ForeachThreadOpInterface
    : public BufferizableOpInterface::ExternalModel<ForeachThreadOpInterface,
                                                    ForeachThreadOp> {
  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    // Get OpOperand (dest) from corresponding ParallelInsertSliceOp.
    auto foreachThreadOp = cast<ForeachThreadOp>(op);
    return {getInsertionDest(foreachThreadOp)[opResult.getResultNumber()]};
  }

  bool isMemoryWrite(Operation *op, OpResult opResult,
                     const AnalysisState &state) const {
    // This op is a memory write. Stop lookup here to avoid finding false
    // conflicts involving this op and one of the ops in the region. This is
    // similar to how scf.if ops are analyzed.
    return true;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult resolveConflicts(Operation *op, RewriterBase &rewriter,
                                 const AnalysisState &state) const {
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    if (failed(bufferizableOp.resolveTensorOpOperandConflicts(rewriter, state)))
      return failure();

    OpBuilder::InsertionGuard g(rewriter);
    auto foreachThreadOp = cast<ForeachThreadOp>(op);
    for (OpResult opResult : foreachThreadOp->getOpResults()) {
      SmallVector<OpOperand *> destOperands =
          state.getAliasingOpOperand(opResult);
      assert(destOperands.size() == 1 &&
             "expected exactly one aliasing OpOperand");
      assert(isa<ParallelInsertSliceOp>(destOperands.front()->getOwner()) &&
             "expected ParallelInsertSliceOp");

      // Nothing to do if there is no conflict.
      if (state.isInPlace(*destOperands.front()))
        continue;

      // Insert tensor allocation.
      bool isYielded = state.isTensorYielded(opResult);
      Value alloc = allocateTensorForShapedValue(rewriter, op->getLoc(),
                                                 destOperands.front()->get(),
                                                 /*escape=*/isYielded);

      // Update terminator operand.
      rewriter.updateRootInPlace(destOperands.front()->getOwner(),
                                 [&]() { destOperands.front()->set(alloc); });
    }

    return success();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto foreachThreadOp = cast<ForeachThreadOp>(op);

#ifndef NDEBUG
    // ParallelInsertSliceOpInterface replaces all uses.
    for (OpResult opResult : foreachThreadOp->getOpResults())
      assert(opResult.getUses().empty() &&
             "expected that all uses were already replaced");
#endif // NDEBUG

    // Create new ForeachThreadOp without any results and drop the automatically
    // introduced terminator.
    TypeRange newResultTypes;
    auto newForeachThreadOp = rewriter.create<ForeachThreadOp>(
        foreachThreadOp.getLoc(), newResultTypes,
        foreachThreadOp.getNumThreads());
    newForeachThreadOp.getBody()->getTerminator()->erase();

    // Move over block contents of the old op.
    rewriter.mergeBlocks(foreachThreadOp.getBody(),
                         newForeachThreadOp.getBody(),
                         {newForeachThreadOp.getBody()->getArguments()});

    // Remove the old op.
    rewriter.eraseOp(op);

    return success();
  }
};

/// Nothing to do for PerformConcurrentlyOp.
struct PerformConcurrentlyOpInterface
    : public BufferizableOpInterface::ExternalModel<
          PerformConcurrentlyOpInterface, PerformConcurrentlyOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &b,
                          const BufferizationOptions &options) const {
    llvm_unreachable("op does not have any tensor OpOperands / OpResults");
    return failure();
  }
};

/// Return true if the (ExtractSliceOp, ParallelInsertSliceOp) pair match (i.e.
/// equivalent operand / result and same offset/sizes/strides specification).
static bool areEquivalentExtractSliceOps(const AnalysisState &state,
                                         ExtractSliceOp st,
                                         ParallelInsertSliceOp sti) {
  if (!st || !sti)
    return false;
  if (st != sti &&
      !state.areEquivalentBufferizedValues(st.getSource(), sti.getDest()))
    return false;
  if (!sameOffsetsSizesAndStrides(st, sti, isEqualConstantIntOrValue))
    return false;
  return true;
}

/// Return true if `value` is originating from an ExtractSliceOp that matches
/// the given InsertSliceOp.
static bool hasMatchingExtractSliceOp(const AnalysisState &state, Value value,
                                      ParallelInsertSliceOp insertOp) {
  auto condition = [&](Value val) {
    if (auto extractOp = val.getDefiningOp<ExtractSliceOp>())
      if (areEquivalentExtractSliceOps(state, extractOp, insertOp))
        return true;
    return false;
  };

  return llvm::all_of(state.findValueInReverseUseDefChain(value, condition),
                      condition);
}

/// Analysis of ParallelInsertSliceOp.
struct ParallelInsertSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ParallelInsertSliceOpInterface, ParallelInsertSliceOp> {
  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (&opOperand != &op->getOpOperand(1) /*dest*/)
      return {};

    // ParallelInsertSliceOp itself has no results. Tensors are returned via
    // the parent op.
    auto foreachThreadOp = op->getParentOfType<ForeachThreadOp>();
    assert(foreachThreadOp &&
           "could not find valid owner of parallel_insert_slice");

    // The i-th ParallelInsertSliceOp result is returned via the i-th OpResult
    // of the parent ForeachThreadOp.
    Block *block = op->getBlock();
    unsigned int opIdx = 0;
    for (ParallelInsertSliceOp insertOp :
         block->getOps<ParallelInsertSliceOp>()) {
      if (insertOp.getOperation() == op)
        break;
      ++opIdx;
    }
    assert(opIdx < foreachThreadOp->getNumResults() &&
           "could not find op inside terminator op");

    return {foreachThreadOp->getResult(opIdx)};
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult resolveConflicts(Operation *op, RewriterBase &rewriter,
                                 const AnalysisState &state) const {
    return success();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    OpBuilder::InsertionGuard g(rewriter);
    auto insertOp = cast<ParallelInsertSliceOp>(op);
    auto performConcurrentlyOp = cast<PerformConcurrentlyOp>(op->getParentOp());
    auto foreachThreadOp =
        cast<ForeachThreadOp>(performConcurrentlyOp->getParentOp());

    // If the op bufferizes out-of-place, allocate the copy before the
    // ForeachThreadOp.
    rewriter.setInsertionPoint(foreachThreadOp);
    Value destBuffer = getBuffer(rewriter, insertOp.getDest(), options);

    // Bufferize the ParallelInsertSliceOp outside of the PerformConcurrentlyOp.
    rewriter.setInsertionPoint(performConcurrentlyOp);
    Value srcBuffer = getBuffer(rewriter, insertOp.getSource(), options);
    Value subview = rewriter.create<memref::SubViewOp>(
        insertOp.getLoc(), destBuffer, insertOp.getMixedOffsets(),
        insertOp.getMixedSizes(), insertOp.getMixedStrides());
    // This memcpy will fold away if everything bufferizes in-place.
    if (failed(options.createMemCpy(rewriter, insertOp.getLoc(), srcBuffer,
                                    subview)))
      return failure();
    rewriter.eraseOp(op);

    // Replace all uses of ForeachThreadOp (just the corresponding result).
    rewriter.setInsertionPointAfter(foreachThreadOp);
    Value toTensorOp =
        rewriter.create<ToTensorOp>(foreachThreadOp.getLoc(), destBuffer);
    unsigned resultNum = 0;
    for (Operation &nextOp : performConcurrentlyOp.yieldingOps()) {
      if (&nextOp == op)
        break;
      resultNum++;
    }
    assert(resultNum < foreachThreadOp->getNumResults() &&
           "ParallelInsertSliceOp not found in PerformConcurrentlyOp");
    SmallVector<OpOperand *> resultUses = llvm::to_vector(
        llvm::map_range(foreachThreadOp->getResult(resultNum).getUses(),
                        [](OpOperand &use) { return &use; }));
    for (OpOperand *use : resultUses) {
      rewriter.updateRootInPlace(use->getOwner(),
                                 [&]() { use->set(toTensorOp); });
    }
    return success();
  }

  // TODO: This is copied from TensorInterfaceImpl.cpp. Find a way to share
  // the code.
  bool isNotConflicting(Operation *op, OpOperand *uRead,
                        OpOperand *uConflictingWrite,
                        const AnalysisState &state) const {
    Operation *readingOp = uRead->getOwner();
    Operation *conflictingWritingOp = uConflictingWrite->getOwner();

    // Special rules for matching ExtractSliceOp/InsertSliceOp pairs. If
    // uRead is an InsertSliceOp...
    if (auto insertSliceOp = dyn_cast<ParallelInsertSliceOp>(readingOp)) {
      // As an example, consider the following IR.
      //
      // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
      // %1 = linalg.fill %cst, %0 {inplace= [true] }
      // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
      //     {inplace= [true] }

      // TODO: Use insertSliceOp.getDestOpOperand etc. when available.
      if (uRead == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          hasMatchingExtractSliceOp(state, uConflictingWrite->get(),
                                    insertSliceOp))
        // Case 1: The main insight is that InsertSliceOp reads only part of
        // the destination tensor. The overwritten area is not read. If
        // uConflictingWrite writes into exactly the memory location that is
        // being read by uRead, this is not a conflict.
        //
        // In the above example:
        // uRead             = OpOperand 1 (%t) of tensor.insert_slice
        // uConflictingWrite = OpOperand 1 (%0) of linalg.fill
        //
        // The read of %t does not conflict with the write of the FillOp
        // (same aliases!) because the area that the FillOp operates on is
        // exactly the one that is *not* read via %t.
        return true;

      if (uRead == &insertSliceOp->getOpOperand(0) /*source*/ &&
          uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          hasMatchingExtractSliceOp(state, uRead->get(), insertSliceOp))
        // Case 2: The read of the source tensor and the write to the dest
        // tensor via an InsertSliceOp is not a conflict if the read is
        // reading exactly that part of an equivalent tensor that the
        // InsertSliceOp is writing.
        //
        // In the above example:
        // uRead             = OpOperand 0 (%1) of tensor.insert_slice
        // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
        return true;
    }

    // If uConflictingWrite is an InsertSliceOp...
    if (auto insertSliceOp =
            dyn_cast<ParallelInsertSliceOp>(conflictingWritingOp))
      // As an example, consider the following IR.
      //
      // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
      // %1 = linalg.fill %cst, %0 {inplace= [true] }
      // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
      //     {inplace= [true] }
      // %3 = vector.transfer_read %1, %cst
      //
      // In the above example:
      // uRead             = OpOperand 0 (%1) of vector.transfer_read
      // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
      // lastWrite         = %1
      //
      // This is not a conflict because the InsertSliceOp overwrites the
      // memory segment of %1 with the exact same data. (Effectively, there
      // is no memory write here.)
      if (uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          state.areEquivalentBufferizedValues(uRead->get(),
                                              insertSliceOp.getSource()) &&
          hasMatchingExtractSliceOp(state, insertSliceOp.getSource(),
                                    insertSliceOp))
        return true;

    return false;
  }
};

} // namespace
} // namespace scf
} // namespace mlir

void mlir::scf::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    ExecuteRegionOp::attachInterface<ExecuteRegionOpInterface>(*ctx);
    ForOp::attachInterface<ForOpInterface>(*ctx);
    IfOp::attachInterface<IfOpInterface>(*ctx);
    ForeachThreadOp::attachInterface<ForeachThreadOpInterface>(*ctx);
    ParallelInsertSliceOp::attachInterface<ParallelInsertSliceOpInterface>(
        *ctx);
    PerformConcurrentlyOp::attachInterface<PerformConcurrentlyOpInterface>(
        *ctx);
    WhileOp::attachInterface<WhileOpInterface>(*ctx);
    YieldOp::attachInterface<YieldOpInterface>(*ctx);
  });
}
