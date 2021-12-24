//===- LinalgInterfaceImpl.cpp - Linalg Impl. of BufferizableOpInterface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace linalg;
using namespace comprehensive_bufferize;

namespace {

// TODO: Ops in the linalg dialect can directly implement this interface.

/// Generic conversion for any LinalgOp on tensors.
static LogicalResult bufferizeLinalgOp(OpBuilder &b, LinalgOp op,
                                       BufferizationState &state) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasBufferSemantics())
    return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (OpOperand *opOperand : op.getInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    newInputBuffers.push_back(state.lookupBuffer(opOperand->get()));
  }

  SmallVector<Value> newOutputBuffers;
  for (OpOperand *opOperand : op.getOutputOperands()) {
    OpResult opResult = op.getTiedOpResult(opOperand);
    assert(opResult && "could not find correspond OpResult");
    Value resultBuffer = state.getResultBuffer(opResult);
    if (!resultBuffer)
      return failure();
    newOutputBuffers.push_back(resultBuffer);
  }

  // Clone the newly bufferized op.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  b.setInsertionPoint(op);
  auto bufferizedOp = cast<LinalgOp>(
      op.clone(b, op.getLoc(), /*resultTypes=*/TypeRange{}, newOperands));

  // Replace the results of the old op with the new output buffers.
  state.replaceOp(op, newOutputBuffers);

  return comprehensive_bufferize::bufferize(bufferizedOp.getBlock(), state);
}

/// Linalg OpResults usually bufferize inplace with their tied (output
/// OpOperands. However, if an output OpOperand is not used in the computation,
/// it is better to bufferize inplace with an actually used input OpOperand;
/// less memory will be touched that way.
///
/// Example:
/// O(i, j) = A(i, j) + B(j)  --> bufferizes inplace to:  A(i, j) += B(j)
///
/// O(i, j) = A(j, i) + B(j)  --> cannot bufferize inplace with A because
///                               indexing maps are not identical
///
/// O(i, j) += A(i, j) + B(j) --> Output is used in computation.
/// This could bufferize inplace with A:
/// A(i, j) += O(i, j) + B(j)
/// However, we choose to bufferize inplace with O here, as there is no clear
/// benefit of choosing A. TODO: We may want to consider both options and make
/// an informed decision during analysis in the future.
static DenseMap<OpOperand *, OpResult> computeAliasingPairs(LinalgOp op) {
  DenseMap<OpOperand *, OpResult> mapping;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand *tiedOperand =
        op.getOutputTensorOperands()[opResult.getResultNumber()];
    AffineMap outputIndexingMap = op.getTiedIndexingMap(tiedOperand);
    bool onlyParallelIterators = op.getNumParallelLoops() == op.getNumLoops();
    bool tiedOperandUsed = op.payloadUsesValueFromOperand(tiedOperand);

    // If the output arg is used in the computation or at least one iterator is
    // not parallel, try to bufferize inplace with the corresponding output
    // tensor.
    if (tiedOperandUsed || !onlyParallelIterators) {
      mapping[tiedOperand] = opResult;
      continue;
    }

    // Otherwise, try to bufferize inplace with one of the inputs.
    OpOperand *chosenOperand = nullptr;
    for (OpOperand *opOperand : op.getInputTensorOperands()) {
      if (opOperand->get().getType() != opResult.getType())
        continue;
      if (!op.payloadUsesValueFromOperand(opOperand))
        continue;
      if (op.getTiedIndexingMap(opOperand) != outputIndexingMap)
        continue;
      // No other OpResult bufferizes aliases with this OpOperand.
      if (mapping.count(opOperand))
        continue;
      assert(op.getTiedIndexingMap(opOperand).isProjectedPermutation() &&
             "expected projected permutation");
      chosenOperand = opOperand;
      break;
    }

    // No suitable input tensor found. Use output tensor.
    // TODO: This operand could bufferize inplace with OpOperands that have the
    // correct type, even if they are not used inside the computation.
    if (!chosenOperand)
      chosenOperand = tiedOperand;

    mapping[chosenOperand] = opResult;
  }
  return mapping;
}

template <typename OpTy>
struct LinalgOpInterface
    : public BufferizableOpInterface::ExternalModel<LinalgOpInterface<OpTy>,
                                                    OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              BufferizationState &state) const {
    auto genericOp = cast<linalg::LinalgOp>(op);
    return genericOp.payloadUsesValueFromOperand(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    return static_cast<bool>(
        bufferizableOp.getAliasingOpResult(opOperand, state));
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       BufferizationState &state) const {
    auto genericOp = cast<linalg::LinalgOp>(op);
    DenseMap<OpOperand *, OpResult> pairs = computeAliasingPairs(genericOp);
    for (OpOperand *opOperand : genericOp.getInputAndOutputOperands())
      if (pairs[opOperand] == opResult)
        return {opOperand};
    return {};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    auto genericOp = cast<linalg::LinalgOp>(op);
    DenseMap<OpOperand *, OpResult> pairs = computeAliasingPairs(genericOp);
    return pairs[&opOperand];
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    return bufferizeLinalgOp(b, cast<LinalgOp>(op), state);
  }
};

struct InitTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<InitTensorOpInterface,
                                                    linalg::InitTensorOp> {
  bool isMemoryWrite(Operation *op, OpResult opResult,
                     BufferizationState &state) const {
    // InitTensorOps allocate but do not write.
    return false;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto initTensorOp = cast<linalg::InitTensorOp>(op);

    // The InitTensorOp may have been eliminated.
    if (initTensorOp->getUses().empty())
      return success();

    Value alloc = state.createAllocDeallocPair(b, initTensorOp->getLoc(),
                                               initTensorOp.result());
    state.replaceOp(op, alloc);
    return success();
  }
};

struct TiledLoopOpInterface
    : public BufferizableOpInterface::ExternalModel<TiledLoopOpInterface,
                                                    linalg::TiledLoopOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              BufferizationState &state) const {
    // TiledLoop alone doesn't bufferize to a memory read, one of the uses of
    // its matching bbArg may.
    auto tiledLoopOp = cast<linalg::TiledLoopOp>(op);
    return state.isValueRead(tiledLoopOp.getTiedBlockArgument(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    // TiledLoop alone doesn't bufferize to a memory write, one of the uses of
    // its matching bbArg may.
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    return static_cast<bool>(
        bufferizableOp.getAliasingOpResult(opOperand, state));
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    auto tiledLoopOp = cast<linalg::TiledLoopOp>(op);
    return tiledLoopOp.getTiedOpResult(opOperand);
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  bool isWritable(Operation *op, Value value, BufferizationState &state) const {
    // Interestingly, linalg::TiledLoopOp's bbArg can **always** be viewed
    // inplace from the perspective of ops nested under:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  bool isAllocationHoistingBarrier(Operation *op) const { return true; }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto tiledLoopOp = cast<linalg::TiledLoopOp>(op);

    // Use IRRewriter instead of OpBuilder because it has additional helper
    // functions.
    IRRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(tiledLoopOp);

    // Compute new inputs, outputs and results.
    SmallVector<Value> newInputs, newOutputs, newResults;
    for (Value value : tiledLoopOp.inputs()) {
      if (value.getType().isa<TensorType>()) {
        newInputs.push_back(state.lookupBuffer(value));
      } else {
        newInputs.push_back(value);
      }
    }
    int nextResultNum = 0;
    for (Value value : tiledLoopOp.outputs()) {
      if (value.getType().isa<TensorType>()) {
        Value buffer =
            state.getResultBuffer(tiledLoopOp->getResult(nextResultNum++));
        newOutputs.push_back(buffer);
        newResults.push_back(buffer);
      } else {
        newOutputs.push_back(value);
      }
    }

    // Create new TiledLoopOp.
    auto newTiledLoopOp = rewriter.create<TiledLoopOp>(
        tiledLoopOp.getLoc(), tiledLoopOp.lowerBound(),
        tiledLoopOp.upperBound(), tiledLoopOp.step(), newInputs, newOutputs,
        tiledLoopOp.iterator_types(), tiledLoopOp.distribution_types());

    // Remove terminator.
    if (!newTiledLoopOp.getBody()->empty())
      rewriter.eraseOp(tiledLoopOp.getBody()->getTerminator());

    // Compute new loop body arguments.
    SmallVector<Value> newBlockArgs, newRegionInOutArgs, oldRegionInOutArgs;
    ValueRange newInductionVars = newTiledLoopOp.getInductionVars();
    newBlockArgs.append(newInductionVars.begin(), newInductionVars.end());

    ValueRange newRegionInArgs = newTiledLoopOp.getRegionInputArgs();
    ValueRange newRegionOutArgs = newTiledLoopOp.getRegionOutputArgs();
    newRegionInOutArgs.append(newRegionInArgs.begin(), newRegionInArgs.end());
    newRegionInOutArgs.append(newRegionOutArgs.begin(), newRegionOutArgs.end());

    ValueRange oldRegionInArgs = tiledLoopOp.getRegionInputArgs();
    ValueRange oldRegionOutArgs = tiledLoopOp.getRegionOutputArgs();
    oldRegionInOutArgs.append(oldRegionInArgs.begin(), oldRegionInArgs.end());
    oldRegionInOutArgs.append(oldRegionOutArgs.begin(), oldRegionOutArgs.end());
    assert(newRegionInArgs.size() == oldRegionInArgs.size() &&
           "expected same number of input args");
    assert(newRegionOutArgs.size() == oldRegionOutArgs.size() &&
           "expected same number of output args");

    for (auto it : llvm::zip(oldRegionInOutArgs, newRegionInOutArgs)) {
      Value oldArg = std::get<0>(it);
      Value newArg = std::get<1>(it);
      rewriter.setInsertionPointToStart(newTiledLoopOp->getBlock());
      if (oldArg.getType().isa<TensorType>()) {
        newBlockArgs.push_back(rewriter.create<bufferization::ToTensorOp>(
            oldArg.getLoc(), newArg));
      } else {
        newBlockArgs.push_back(newArg);
      }
    }

    // Move old body into new loop.
    rewriter.mergeBlocks(tiledLoopOp.getBody(), newTiledLoopOp.getBody(),
                         newBlockArgs);

    // Replace previous terminator with a new one that does not yield anything.
    Operation *oldTerminator = newTiledLoopOp.getBody()->getTerminator();
    rewriter.setInsertionPointToEnd(newTiledLoopOp.getBody());
    rewriter.create<linalg::YieldOp>(oldTerminator->getLoc());
    rewriter.eraseOp(oldTerminator);

    // Replace results and delete old op.
    state.replaceOp(op, newResults);

    // Bufferize loop body.
    return comprehensive_bufferize::bufferize(newTiledLoopOp.getBody(), state);
  }
};

struct YieldOpInterface
    : public BufferizableOpInterface::ExternalModel<YieldOpInterface,
                                                    linalg::YieldOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto yieldOp = cast<linalg::YieldOp>(op);

    if (!yieldOp->getParentOfType<TiledLoopOp>())
      return yieldOp->emitError(
          "expected that linalg.yield terminates a tiled_loop");

    assert(yieldOp->getOpOperands().empty() &&
           "expected that linalg.yield was bufferized together with"
           " tiled_loop");
    return success();
  }
};

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... OpTys>
struct LinalgOpInterfaceHelper;

template <typename First, typename... Others>
struct LinalgOpInterfaceHelper<First, Others...> {
  static void registerOpInterface(DialectRegistry &registry) {
    registry.addOpInterface<First, LinalgOpInterface<First>>();
    LinalgOpInterfaceHelper<Others...>::registerOpInterface(registry);
  }
};

template <>
struct LinalgOpInterfaceHelper<> {
  static void registerOpInterface(DialectRegistry &registry) {}
};

} // namespace

/// Try to eliminate InitTensorOps inside `op`. An InitTensorOp is replaced
/// with the the result of `rewriteFunc` if it is anchored on a matching
/// OpOperand. "Anchored" means that there is a path on the reverse SSA use-def
/// chain, starting from the OpOperand and always following the aliasing
/// OpOperand, that eventually ends at a single InitTensorOp.
LogicalResult mlir::linalg::comprehensive_bufferize::linalg_ext::
    InitTensorEliminationStep::eliminateInitTensors(
        Operation *op, BufferizationState &state,
        BufferizationAliasInfo &aliasInfo,
        std::function<bool(OpOperand &)> anchorMatchFunc,
        std::function<Value(OpBuilder &, Location, OpOperand &)> rewriteFunc,
        SmallVector<Operation *> &newOps) {
  OpBuilder b(op->getContext());

  WalkResult status = op->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Is this a matching OpOperand?
      if (!anchorMatchFunc(operand))
        continue;

      SetVector<Value> maybeInitTensor =
          state.findValueInReverseUseDefChain(operand.get(), [&](Value val) {
            // Continue traversal until this function returns true.
            OpResult opResult = val.dyn_cast<OpResult>();
            if (!opResult)
              return true;
            if (!aliasInfo.isInPlace(opResult))
              return true;
            // Only equivalent tensors are supported at the moment.
            // TODO: Support cases such as extract_slice(init_tensor).
            SmallVector<OpOperand *> opOperands =
                state.getAliasingOpOperand(opResult);
            if (!llvm::all_of(opOperands, [&](OpOperand *operand) {
                  return aliasInfo.areEquivalentBufferizedValues(operand->get(),
                                                                 opResult);
                }))
              return true;
            return false;
          });

      // Replace only if the reverse use-def chain ends at exactly one
      // InitTensorOp.
      if (maybeInitTensor.size() != 1 ||
          !maybeInitTensor.front().getDefiningOp<InitTensorOp>())
        return WalkResult::skip();
      Value initTensor = maybeInitTensor.front();

      // Create a replacement for the InitTensorOp.
      b.setInsertionPoint(initTensor.getDefiningOp());
      Value replacement = rewriteFunc(b, initTensor.getLoc(), operand);
      if (!replacement)
        continue;

      // Uses of the InitTensorOp are replaced here, but the op is not deleted.
      // InitTensorOps without uses are ignored by the bufferization.
      initTensor.replaceAllUsesWith(replacement);
      aliasInfo.createAliasInfoEntry(replacement);
      aliasInfo.unionAliasSets(initTensor, replacement);
      aliasInfo.unionEquivalenceClasses(initTensor, replacement);

      // Register replacement ops.
      if (Operation *newOp = replacement.getDefiningOp())
        newOps.push_back(newOp);
    }

    // Advance to the next operation.
    return WalkResult::advance();
  });

  return failure(status.wasInterrupted());
}

/// Try to eliminate InitTensorOps inside `op`. An InitTensorOp can be
/// eliminated if it is eventually inserted into another tensor (and some other
/// conditions are met).
///
/// E.g.:
/// %0 = linalg.init_tensor
/// %1 = linalg.fill(%cst, %0) {inplace = [true]}
/// %2 = tensor.insert_slice %1 into %t[10][20][1]
///
/// InitTensorOp elimination will try to fill %t inplace instead of filling a
/// new allocation %0 and inserting it into %t. This is done by replacing the
/// InitTensorOp with:
///
/// %0 = tensor.extract_slice %t[10][20][1]
///
/// The analysis looks for matching ExtractSliceOp/InsertSliceOp pairs and lets
/// those bufferize inplace in the absence of other conflicts.
///
/// Starting from an InsertSliceOp, an InitTensorOp at the end of the insert
/// source's reverse use-def chain is eliminated if:
/// * The InsertSliceOp was decided to bufferize inplace.
/// * On the reverse use-def chain path from the InsertSliceOp to the
///   InitTensorOp, all ops were decided to bufferize inplace and the buffer
///   relation is "equivalent" (TODO: can be relaxed if needed).
/// * The reverse use-def chain has exactly one end, which is the InitTensorOp.
///
/// Note that the newly inserted ExtractSliceOp may have to bufferize
/// out-of-place due to RaW conflicts.
LogicalResult mlir::linalg::comprehensive_bufferize::linalg_ext::
    InsertSliceAnchoredInitTensorEliminationStep::run(
        Operation *op, BufferizationState &state,
        BufferizationAliasInfo &aliasInfo, SmallVector<Operation *> &newOps) {
  return eliminateInitTensors(
      op, state, aliasInfo,
      [&](OpOperand &operand) {
        auto insertSliceOp =
            dyn_cast<tensor::InsertSliceOp>(operand.getOwner());
        if (!insertSliceOp)
          return false;
        // Only inplace bufferized InsertSliceOps are eligible.
        if (!aliasInfo.isInPlace(insertSliceOp->getOpResult(0)))
          return false;
        return &operand == &insertSliceOp->getOpOperand(0) /*source*/;
      },
      [](OpBuilder &b, Location loc, OpOperand &operand) {
        auto insertSliceOp = cast<tensor::InsertSliceOp>(operand.getOwner());
        auto extractOp = b.create<tensor::ExtractSliceOp>(
            loc, insertSliceOp.dest(), insertSliceOp.getMixedOffsets(),
            insertSliceOp.getMixedSizes(), insertSliceOp.getMixedStrides());
        return extractOp.result();
      },
      newOps);
}

void mlir::linalg::comprehensive_bufferize::linalg_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<linalg::InitTensorOp, InitTensorOpInterface>();
  registry.addOpInterface<linalg::TiledLoopOp, TiledLoopOpInterface>();
  registry.addOpInterface<linalg::YieldOp, YieldOpInterface>();

  // Register all Linalg structured ops. `LinalgOp` is an interface and it is
  // not possible to attach an external interface to an existing interface.
  // Therefore, attach the `BufferizableOpInterface` to all ops one-by-one.
  LinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >::registerOpInterface(registry);
}
