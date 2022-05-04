//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace linalg;
using namespace mlir::bufferization;

namespace {

// TODO: Ops in the linalg dialect can directly implement this interface.

/// Generic conversion for any LinalgOp on tensors.
static LogicalResult bufferizeLinalgOp(RewriterBase &rewriter, LinalgOp op,
                                       BufferizationState &state) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasBufferSemantics())
    return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (OpOperand *opOperand : op.getInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    // Input operands are never written to.
    newInputBuffers.push_back(*state.getBuffer(
        rewriter, *opOperand,
        BufferizationState::ForceInPlacability::FORCE_INPLACE));
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    SmallVector<OpOperand *> aliasingOpOperands =
        state.getAnalysisState().getAliasingOpOperand(opResult);
    assert(aliasingOpOperands.size() == 1 && "expected 1 OpOperand");
    FailureOr<Value> resultBuffer =
        state.getBuffer(rewriter, *aliasingOpOperands.front());
    if (failed(resultBuffer))
      return failure();
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Merge input/output operands.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands. Move the existing block into the
  // new op. Since the new op does not have any tensor results, it does not
  // return anything.
  assert(op->getNumRegions() == 1 && "expected that op has 1 region");
  auto newOp = cast<LinalgOp>(op.cloneWithoutRegions(
      rewriter, op.getLoc(), /*resultTypes=*/TypeRange{}, newOperands));
  rewriter.inlineRegionBefore(op->getRegion(0), newOp->getRegion(0),
                              newOp->getRegion(0).begin());

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
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

/// Bufferization of linalg.generic. Replace with a new linalg.generic that
/// operates entirely on memrefs.
template <typename OpTy>
struct LinalgOpInterface
    : public BufferizableOpInterface::ExternalModel<LinalgOpInterface<OpTy>,
                                                    OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Operand is read if it is used in the computation.
    auto genericOp = cast<linalg::LinalgOp>(op);
    return genericOp.payloadUsesValueFromOperand(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Operand is written to if it has an aliasing OpResult.
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    return !bufferizableOp.getAliasingOpResult(opOperand, state).empty();
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    auto genericOp = cast<linalg::LinalgOp>(op);

    // By default, the i-th OpResult may alias with the i-th "out" tensor.
    if (state.getOptions().alwaysAliasingWithDest)
      return {genericOp.getOutputOperand(opResult.getResultNumber())};

    // We can try to be smart and alias in-place with an "in" tensor if the
    // corresponding "out" tensor is not used in the computation.
    // Aliasing OpOperand/OpResult pairs are computed by `computeAliasingPairs`.
    DenseMap<OpOperand *, OpResult> pairs = computeAliasingPairs(genericOp);
    for (OpOperand *opOperand : genericOp.getInputAndOutputOperands())
      if (pairs[opOperand] == opResult)
        return {opOperand};
    return {};
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    auto genericOp = cast<linalg::LinalgOp>(op);

    // By default, the i-th "out" tensor may alias with the i-th OpResult.
    if (state.getOptions().alwaysAliasingWithDest) {
      if (genericOp.isOutputTensor(&opOperand))
        return {genericOp.getTiedOpResult(&opOperand)};
      return {};
    }

    // We can try to be smart. See comment in `getAliasingOpOperand`.
    // Aliasing OpOperand/OpResult pairs are computed by `computeAliasingPairs`.
    DenseMap<OpOperand *, OpResult> pairs = computeAliasingPairs(genericOp);
    if (!pairs.count(&opOperand))
      return {};
    return {pairs[&opOperand]};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    return bufferizeLinalgOp(rewriter, cast<LinalgOp>(op), state);
  }
};

struct InitTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<InitTensorOpInterface,
                                                    linalg::InitTensorOp> {
  bool isMemoryWrite(Operation *op, OpResult opResult,
                     const AnalysisState &state) const {
    // InitTensorOps allocate but do not write.
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    auto initTensorOp = cast<linalg::InitTensorOp>(op);

    // The InitTensorOp may have been eliminated.
    if (initTensorOp->getUses().empty())
      return success();

    FailureOr<Value> alloc = state.createAlloc(rewriter, initTensorOp->getLoc(),
                                               initTensorOp.result());
    if (failed(alloc))
      return failure();
    replaceOpWithBufferizedValues(rewriter, op, *alloc);
    return success();
  }
};

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... Ops>
struct LinalgOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (void)std::initializer_list<int>{
        0, (Ops::template attachInterface<LinalgOpInterface<Ops>>(*ctx), 0)...};
  }
};
} // namespace

/// Return true if all `neededValues` are in scope at the given
/// `insertionPoint`.
static bool
neededValuesDominateInsertionPoint(const DominanceInfo &domInfo,
                                   Operation *insertionPoint,
                                   const SmallVector<Value> &neededValues) {
  for (Value val : neededValues) {
    if (auto bbArg = val.dyn_cast<BlockArgument>()) {
      Block *owner = bbArg.getOwner();
      if (!owner->findAncestorOpInBlock(*insertionPoint))
        return false;
    } else {
      auto opResult = val.cast<OpResult>();
      if (!domInfo.dominates(opResult.getOwner(), insertionPoint))
        return false;
    }
  }
  return true;
}

/// Return true if the given `insertionPoint` dominates all uses of
/// `initTensorOp`.
static bool insertionPointDominatesUses(const DominanceInfo &domInfo,
                                        Operation *insertionPoint,
                                        Operation *initTensorOp) {
  for (Operation *user : initTensorOp->getUsers())
    if (!domInfo.dominates(insertionPoint, user))
      return false;
  return true;
}

/// Find a valid insertion point for a replacement of `initTensorOp`, assuming
/// that the replacement may use any value from `neededValues`.
static Operation *
findValidInsertionPoint(Operation *initTensorOp,
                        const SmallVector<Value> &neededValues) {
  DominanceInfo domInfo;

  // Gather all possible insertion points: the location of `initTensorOp` and
  // right after the definition of each value in `neededValues`.
  SmallVector<Operation *> insertionPointCandidates;
  insertionPointCandidates.push_back(initTensorOp);
  for (Value val : neededValues) {
    // Note: The anchor op is using all of `neededValues`, so:
    // * in case of a block argument: There must be at least one op in the block
    //                                (the anchor op or one of its parents).
    // * in case of an OpResult: There must be at least one op right after the
    //                           defining op (the anchor op or one of its
    //                           parents).
    if (auto bbArg = val.dyn_cast<BlockArgument>()) {
      insertionPointCandidates.push_back(
          &bbArg.getOwner()->getOperations().front());
    } else {
      insertionPointCandidates.push_back(val.getDefiningOp()->getNextNode());
    }
  }

  // Select first matching insertion point.
  for (Operation *insertionPoint : insertionPointCandidates) {
    // Check if all needed values are in scope.
    if (!neededValuesDominateInsertionPoint(domInfo, insertionPoint,
                                            neededValues))
      continue;
    // Check if the insertion point is before all uses.
    if (!insertionPointDominatesUses(domInfo, insertionPoint, initTensorOp))
      continue;
    return insertionPoint;
  }

  // No suitable insertion point was found.
  return nullptr;
}

/// Try to eliminate InitTensorOps inside `op`. An InitTensorOp is replaced
/// with the the result of `rewriteFunc` if it is anchored on a matching
/// OpOperand. "Anchored" means that there is a path on the reverse SSA use-def
/// chain, starting from the OpOperand and always following the aliasing
/// OpOperand, that eventually ends at a single InitTensorOp.
LogicalResult mlir::linalg::eliminateInitTensors(RewriterBase &rewriter,
                                                 Operation *op,
                                                 AnalysisState &state,
                                                 AnchorMatchFn anchorMatchFunc,
                                                 RewriteFn rewriteFunc) {
  OpBuilder::InsertionGuard g(rewriter);

  WalkResult status = op->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      // Skip operands that do not bufferize inplace.
      if (!state.isInPlace(operand))
        continue;
      // All values that are needed to create the replacement op.
      SmallVector<Value> neededValues;
      // Is this a matching OpOperand?
      if (!anchorMatchFunc(operand, neededValues))
        continue;
      SetVector<Value> maybeInitTensor =
          state.findValueInReverseUseDefChain(operand.get(), [&](Value val) {
            // Continue traversal until this function returns true.
            OpResult opResult = val.dyn_cast<OpResult>();
            if (!opResult)
              return true;
            SmallVector<OpOperand *> opOperands =
                state.getAliasingOpOperand(opResult);
            if (!llvm::all_of(opOperands, [&](OpOperand *operand) {
                  return state.isInPlace(*operand);
                }))
              return true;
            // Only equivalent tensors are supported at the moment.
            // TODO: Support cases such as extract_slice(init_tensor)
            return !llvm::all_of(opOperands, [&](OpOperand *operand) {
              return state.areEquivalentBufferizedValues(operand->get(),
                                                         opResult);
            });
          });

      // Replace only if the reverse use-def chain ends at exactly one
      // InitTensorOp.
      if (maybeInitTensor.size() != 1 ||
          !maybeInitTensor.front().getDefiningOp<InitTensorOp>())
        return WalkResult::skip();
      Value initTensor = maybeInitTensor.front();

      // Find a suitable insertion point.
      Operation *insertionPoint =
          findValidInsertionPoint(initTensor.getDefiningOp(), neededValues);
      if (!insertionPoint)
        continue;

      // Create a replacement for the InitTensorOp.
      rewriter.setInsertionPoint(insertionPoint);
      Value replacement = rewriteFunc(rewriter, initTensor.getLoc(), operand);
      if (!replacement)
        continue;

      // Replace the InitTensorOp.
      rewriter.replaceOp(initTensor.getDefiningOp(), replacement);
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
/// * On the reverse use-def chain path from the InsertSliceOp to the
///   InitTensorOp, all ops were decided to bufferize inplace and the buffer
///   relation is "equivalent" (TODO: can be relaxed if needed).
/// * The reverse use-def chain has exactly one end, which is the InitTensorOp.
LogicalResult mlir::linalg::insertSliceAnchoredInitTensorEliminationStep(
    RewriterBase &rewriter, Operation *op, AnalysisState &state) {
  return eliminateInitTensors(
      rewriter, op, state,
      /*anchorMatchFunc=*/
      [&](OpOperand &operand, SmallVector<Value> &neededValues) {
        auto insertSliceOp =
            dyn_cast<tensor::InsertSliceOp>(operand.getOwner());
        if (!insertSliceOp)
          return false;
        if (&operand != &insertSliceOp->getOpOperand(0) /*source*/)
          return false;

        // Collect all values that are needed to construct the replacement op.
        neededValues.append(insertSliceOp.offsets().begin(),
                            insertSliceOp.offsets().end());
        neededValues.append(insertSliceOp.sizes().begin(),
                            insertSliceOp.sizes().end());
        neededValues.append(insertSliceOp.strides().begin(),
                            insertSliceOp.strides().end());
        neededValues.push_back(insertSliceOp.dest());

        return true;
      },
      /*rewriteFunc=*/
      [](OpBuilder &b, Location loc, OpOperand &operand) {
        auto insertOp = cast<tensor::InsertSliceOp>(operand.getOwner());
        // Expand offsets, sizes and strides to the full rank to handle the
        // rank-reducing case.
        SmallVector<OpFoldResult> mixedOffsets = insertOp.getMixedOffsets();
        SmallVector<OpFoldResult> mixedSizes = insertOp.getMixedSizes();
        SmallVector<OpFoldResult> mixedStrides = insertOp.getMixedStrides();
        OffsetSizeAndStrideOpInterface::expandToRank(
            insertOp.dest(), mixedOffsets, mixedSizes, mixedStrides,
            [&](Value target, int64_t dim) -> OpFoldResult {
              auto shapedType = target.getType().cast<ShapedType>();
              if (shapedType.isDynamicDim(dim))
                return b.create<tensor::DimOp>(loc, target, dim).result();
              return b.getIndexAttr(shapedType.getDimSize(dim));
            });
        auto t = tensor::ExtractSliceOp::inferRankReducedResultType(
            insertOp.getSourceType().getRank(),
            insertOp.dest().getType().cast<RankedTensorType>(), mixedOffsets,
            mixedSizes, mixedStrides);
        auto extractOp = b.create<tensor::ExtractSliceOp>(
            loc, t, insertOp.dest(), mixedOffsets, mixedSizes, mixedStrides);
        return extractOp.result();
      });
}

void mlir::linalg::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::InitTensorOp::attachInterface<InitTensorOpInterface>(*ctx);

    // Register all Linalg structured ops. `LinalgOp` is an interface and it is
    // not possible to attach an external interface to an existing interface.
    // Therefore, attach the `BufferizableOpInterface` to all ops one-by-one.
    LinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >::registerOpInterface(ctx);
  });
}
