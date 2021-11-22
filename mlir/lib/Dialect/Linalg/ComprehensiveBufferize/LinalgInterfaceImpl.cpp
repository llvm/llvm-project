//===- LinalgInterfaceImpl.cpp - Linalg Impl. of BufferizableOpInterface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace linalg;
using namespace comprehensive_bufferize;

namespace {

// TODO: Ops in the linalg dialect can directly implement this interface.

/// Helper function for LinalgOp bufferization.
/// When allocating a new buffer, analyze whether `op` wants to read form that
/// buffer. Only in that case, a copy of the result buffer may be needed.
static LogicalResult
allocateBuffersForResults(OpBuilder &b, Location loc, LinalgOp op,
                          SmallVectorImpl<Value> &resultBuffers,
                          BufferizationState &state) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);

  // TODO: provide the proper interface to iterate on OpResults and get the
  // matching OpOperands.
  for (OpOperand *opOperand : op.getOutputOperands()) {
    OpResult opResult = cast<BufferizableOpInterface>(op.getOperation())
                            .getAliasingOpResult(*opOperand);
    assert(opResult && "could not find correspond OpResult");
    Value resultBuffer = getResultBuffer(b, opResult, state);
    if (!resultBuffer)
      return failure();
    resultBuffers.push_back(resultBuffer);
  }

  if (op->getNumResults())
    state.mapBuffer(op->getResults(), resultBuffers);

  return success();
}

/// Generic conversion for any LinalgOp on tensors.
static LogicalResult bufferizeLinalgOp(OpBuilder &b, LinalgOp op,
                                       BufferizationState &state) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  Location loc = op.getLoc();
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
  // Try to allocate new buffers depending on op's inplace semantics.
  if (failed(allocateBuffersForResults(b, loc, op, newOutputBuffers, state)))
    return failure();

  // Clone the newly bufferized op.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  b.setInsertionPoint(op);
  op.clone(b, loc, /*resultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  if (op->getNumResults())
    state.mapBuffer(op->getResults(), newOutputBuffers);

  // The original op will be DCE'd away later.

  return success();
}

template <typename OpTy>
struct LinalgOpInterface
    : public BufferizableOpInterface::ExternalModel<LinalgOpInterface<OpTy>,
                                                    OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    auto genericOp = cast<linalg::LinalgOp>(op);
    return (genericOp.isInputTensor(&opOperand) ||
            genericOp.isInitTensor(&opOperand)) &&
           genericOp.payloadUsesValueFromOperand(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    auto genericOp = cast<linalg::LinalgOp>(op);
    return genericOp.isOutputTensor(&opOperand);
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    auto genericOp = cast<linalg::LinalgOp>(op);
    return {genericOp.getOutputTensorOperands()[opResult.getResultNumber()]};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    auto genericOp = cast<linalg::LinalgOp>(op);
    if (!opOperand.get().getType().isa<RankedTensorType>())
      return OpResult();
    // For now assume inputs are never inplaceable.
    // TODO: refine this.
    if (opOperand.getOperandNumber() < genericOp.getNumInputs())
      return OpResult();
    int64_t outputOperandIndex =
        opOperand.getOperandNumber() - genericOp.getNumInputs();
    int64_t numOutputBuffers = 0;
    for (unsigned idx = 0; idx < outputOperandIndex; ++idx)
      if (!genericOp.getOutputOperand(idx)->get().getType().isa<TensorType>())
        ++numOutputBuffers;
    return genericOp->getResult(outputOperandIndex - numOutputBuffers);
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
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
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {};
  }

  bool isMemoryWrite(Operation *op, OpResult opResult) const {
    // InitTensorOps allocate but do not write.
    return false;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto initTensorOp = cast<linalg::InitTensorOp>(op);

    // The InitTensorOp may have been eliminated.
    if (initTensorOp->getUses().empty())
      return success();

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(initTensorOp);

    Value alloc = state.allocationFns.createAllocDeallocFn(
        b, initTensorOp->getLoc(), initTensorOp.result(), state);
    state.mapBuffer(initTensorOp.result(), alloc);
    return success();
  }
};

struct TiledLoopOpInterface
    : public BufferizableOpInterface::ExternalModel<TiledLoopOpInterface,
                                                    linalg::TiledLoopOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    // TiledLoop alone doesn't bufferize to a memory read, one of the uses of
    // its matching bbArg may.
    auto tiledLoopOp = cast<linalg::TiledLoopOp>(op);
    return isValueRead(tiledLoopOp.getTiedBlockArgument(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    // TiledLoop alone doesn't bufferize to a memory write, one of the uses of
    // its matching bbArg may.
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    return static_cast<bool>(bufferizableOp.getAliasingOpResult(opOperand));
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    // TODO: TiledLoopOp helper method to avoid leaking impl details.
    auto tiledLoopOp = cast<linalg::TiledLoopOp>(op);
    return {&op->getOpOperand(tiledLoopOp.getNumControlOperands() +
                              tiledLoopOp.getNumInputs() +
                              opResult.getResultNumber())};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    auto tiledLoopOp = cast<linalg::TiledLoopOp>(op);
    return tiledLoopOp.getTiedOpResult(opOperand);
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
  }

  bool isWritable(Operation *op, Value value) const {
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

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);

    // Allocate output buffers if needed, forward output tensor args to the
    // terminator.
    Operation *yieldOp = tiledLoopOp.getBody()->getTerminator();
    Block *body = tiledLoopOp.getBody();

    // Take copies of the old input and output operands, so we can insert
    // inplace easily.
    auto oldInputs = llvm::to_vector<4>(tiledLoopOp.inputs());
    auto oldOutputs = llvm::to_vector<4>(tiledLoopOp.outputs());

    int numLoops = tiledLoopOp.getNumLoops();
    int numControlOperands = tiledLoopOp.getNumControlOperands();

    // Add buffers for outputs and the corresponding block arguments.
    // Keep separate iterators to increment without further leaking impl.
    // details. Start with outputs to avoid interference from new input buffers.
    int numNewOutputBuffers = 0;
    int resultIndex = 0;
    int oldOutputBBArgIndex = numLoops + oldInputs.size();
    int nextOutputBBArgIndex = numLoops + oldInputs.size() + oldOutputs.size();
    int nextOutputOperandIndex =
        numControlOperands + oldInputs.size() + oldOutputs.size();
    for (Value oldOutputTensor : oldOutputs) {
      if (!oldOutputTensor.getType().isa<TensorType>()) {
        // Skip and increment the old bbarg index only.
        ++oldOutputBBArgIndex;
        // Do not increment resultIndex as only tensors are returned.
        // TODO: better interface to avoid leaking such impl details.
        continue;
      }

      assert(oldOutputTensor.getType().isa<RankedTensorType>() &&
             "bufferizable output must be a ranked tensor");

      const OpResult &opResult = tiledLoopOp->getResult(resultIndex);
      OpOperand &yieldOperand = yieldOp->getOpOperand(resultIndex);
      Value resultBuffer = getResultBuffer(b, opResult, state);
      if (!resultBuffer)
        return failure();

      // Insert mapping and aliasing info.
      state.aliasInfo.createAliasInfoEntry(resultBuffer);
      state.aliasInfo.insertNewBufferEquivalence(opResult, resultBuffer);
      state.mapBuffer(opResult, resultBuffer);

      // Insert new operand and bbArg.
      tiledLoopOp->insertOperands(nextOutputOperandIndex, resultBuffer);
      BlockArgument newBufferBBArg =
          body->insertArgument(nextOutputBBArgIndex, resultBuffer.getType());
      BlockArgument oldTensorBBArg = body->getArgument(oldOutputBBArgIndex);
      // Insert mapping and aliasing info.
      state.aliasInfo.createAliasInfoEntry(newBufferBBArg);
      state.aliasInfo.insertNewBufferEquivalence(oldTensorBBArg,
                                                 newBufferBBArg);
      state.mapBuffer(oldTensorBBArg, newBufferBBArg);

      // Set operand of `linalg.yield` to the bbArg so it just canonicalizes
      // away later.
      yieldOperand.set(oldTensorBBArg);

      // Increment indices.
      ++numNewOutputBuffers;
      ++resultIndex;
      ++oldOutputBBArgIndex;
      ++nextOutputBBArgIndex;
      ++nextOutputOperandIndex;
    }

    // Add buffers for inputs and the corresponding block arguments.
    // Keep separate iterators to increment without further leaking impl.
    // details.
    int numNewInputBuffers = 0;
    int oldInputBBArgIndex = numLoops;
    int nextInputBBArgIndex = numLoops + oldInputs.size();
    int nextInputOperandIndex = numControlOperands + oldInputs.size();
    for (Value oldInputTensor : oldInputs) {
      if (!oldInputTensor.getType().isa<TensorType>()) {
        // Skip and increment the old bbarg index only.
        ++oldInputBBArgIndex;
        continue;
      }

      Value inputBuffer = state.lookupBuffer(oldInputTensor);

      // Insert new operand and bbArg.
      tiledLoopOp->insertOperands(nextInputOperandIndex, inputBuffer);
      BlockArgument newBufferBBArg =
          body->insertArgument(nextInputBBArgIndex, inputBuffer.getType());
      BlockArgument oldTensorBBArg = body->getArgument(oldInputBBArgIndex);

      // Insert mapping and aliasing info.
      state.aliasInfo.createAliasInfoEntry(newBufferBBArg);
      state.aliasInfo.insertNewBufferEquivalence(oldTensorBBArg,
                                                 newBufferBBArg);
      state.mapBuffer(oldTensorBBArg, newBufferBBArg);

      // Increment indices.
      ++numNewInputBuffers;
      ++oldInputBBArgIndex;
      ++nextInputBBArgIndex;
      ++nextInputOperandIndex;
    }

    // Update segment sizes.
    // TODO: Helper method to avoid leaking impl details.
    tiledLoopOp->setAttr(
        TiledLoopOp::getOperandSegmentSizeAttr(),
        b.getI32VectorAttr(
            {numLoops, numLoops, numLoops,
             static_cast<int>(oldInputs.size()) + numNewInputBuffers,
             static_cast<int>(oldOutputs.size()) + numNewOutputBuffers}));

    return success();
  }
};

struct YieldOpInterface
    : public BufferizableOpInterface::ExternalModel<YieldOpInterface,
                                                    linalg::YieldOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto yieldOp = cast<linalg::YieldOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    // Cannot create IR past a yieldOp.
    b.setInsertionPoint(yieldOp);

    // No tensors -> success.
    if (!llvm::any_of(yieldOp.getOperandTypes(),
                      [](Type t) { return t.isa<TensorType>(); }))
      return success();
    // linalg::YieldOp nested under TiledLoop must just canonicalize.
    if (yieldOp->getParentOfType<TiledLoopOp>())
      return success();
    llvm_unreachable("unexpected yieldOp");
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

/// Try to eliminate InitTensorOps inside funcOp. An InitTensorOp is replaced
/// with the the result of `rewriteFunc` if it is anchored on a matching
/// OpOperand. "Anchored" means that there is a path on the reverse SSA use-def
/// chain, starting from the OpOperand and always following the aliasing
/// OpOperand, that eventually ends at a single InitTensorOp.
LogicalResult mlir::linalg::comprehensive_bufferize::linalg_ext::
    InitTensorEliminationStep::eliminateInitTensors(
        FuncOp funcOp, BufferizationAliasInfo &aliasInfo,
        DominanceInfo &domInfo,
        std::function<bool(OpOperand &)> anchorMatchFunc,
        std::function<Value(OpBuilder &, Location, OpOperand &)> rewriteFunc,
        SmallVector<Operation *> &newOps) {
  OpBuilder b(funcOp->getContext());

  WalkResult status = funcOp->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Is this a matching OpOperand?
      if (!anchorMatchFunc(operand))
        continue;

      SetVector<Value> maybeInitTensor =
          findValueInReverseUseDefChain(operand.get(), [&](Value val) {
            // Continue traversal until this function returns true.
            OpResult opResult = val.dyn_cast<OpResult>();
            if (!opResult)
              return true;
            if (!aliasInfo.isInPlace(opResult))
              return true;
            // Only equivalent tensors are supported at the moment.
            // TODO: Support cases such as extract_slice(init_tensor).
            SmallVector<OpOperand *> opOperands =
                getAliasingOpOperand(opResult);
            if (!llvm::all_of(opOperands, [](OpOperand *operand) {
                  return bufferRelation(*operand) == BufferRelation::Equivalent;
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

/// Try to eliminate InitTensorOps inside funcOp. An InitTensorOp can be
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
        FuncOp funcOp, BufferizationAliasInfo &aliasInfo,
        DominanceInfo &domInfo, SmallVector<Operation *> &newOps) {
  return eliminateInitTensors(
      funcOp, aliasInfo, domInfo,
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
