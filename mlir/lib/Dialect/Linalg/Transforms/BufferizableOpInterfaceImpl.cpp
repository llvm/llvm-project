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

void mlir::linalg::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    // Register all Linalg structured ops. `LinalgOp` is an interface and it is
    // not possible to attach an external interface to an existing interface.
    // Therefore, attach the `BufferizableOpInterface` to all ops one-by-one.
    LinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >::registerOpInterface(ctx);
  });
}
