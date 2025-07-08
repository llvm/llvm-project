//===- ReifyResultShapes.cpp - Reify result shapes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform reifies result shapes of `ReifyRankedShapedTypeOpInterface`
// operations with ranked `memref` and `tensor` results.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "reify-result-shapes"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_REIFYRESULTSHAPESPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

/// Reifies the results of `op`, potentially replacing `op` with a reified
/// version. Returns `failure` if `mlir::reifyResultShapes` returned failure,
/// otherwise it always succeeds. Users of this transform should always expect
/// it to modify the IR, even when it fails. If any of the result types changes,
/// the transform will insert cast operations to the old type to keep the IR
/// consistent.
static LogicalResult reifyOpResultShapes(RewriterBase &rewriter,
                                         ReifyRankedShapedTypeOpInterface op) {
  LLVM_DEBUG({ DBGS() << " reifying op: " << op << "\n"; });
  // Get the reified out shapes.
  ReifiedRankedShapedTypeDims reifiedResultShapes;
  if (failed(mlir::reifyResultShapes(rewriter, op, reifiedResultShapes)) ||
      reifiedResultShapes.empty()) {
    return op->emitWarning() << "failed to get the reified shapes";
  }

  bool modified = false;
  // Compute the new output types.
  SmallVector<Type> outTypes;
  for (const auto &[oldTy, reifiedShape] :
       llvm::zip(op->getResultTypes(), reifiedResultShapes)) {
    // Skip if it's not a memref or tensor type.
    if (!isa<RankedTensorType, MemRefType>(oldTy)) {
      outTypes.push_back(oldTy);
      continue;
    }

    ShapedType shapedTy = dyn_cast<ShapedType>(oldTy);

    SmallVector<int64_t> shape = llvm::to_vector(shapedTy.getShape());
    for (auto &&[dim, ofr] : llvm::zip_equal(shape, reifiedShape)) {
      std::optional<int64_t> maybeCst = getConstantIntValue(ofr);
      // If the reified dim is dynamic set it appropriately.
      if (!maybeCst.has_value()) {
        dim = ShapedType::kDynamic;
        continue;
      }
      // Set the static dim.
      dim = *maybeCst;
    }

    // If the shape didn't change continue.
    if (shape == shapedTy.getShape()) {
      outTypes.push_back(oldTy);
      continue;
    }
    modified = true;
    outTypes.push_back(shapedTy.cloneWith(shape, shapedTy.getElementType()));
  }

  // Return if we don't need to update.
  if (!modified) {
    LLVM_DEBUG({ DBGS() << "- op doesn't require update\n"; });
    return success();
  }

  LLVM_DEBUG({
    DBGS() << "- oldTypes: " << llvm::interleaved_array(op->getResultTypes())
           << " \n";
    DBGS() << "- outTypes: " << llvm::interleaved_array(outTypes) << " \n";
  });

  // We now have outTypes that need to be turned to cast ops.
  Location loc = op->getLoc();
  SmallVector<Value> newResults;
  // TODO: `mlir::reifyResultShapes` and op verifiers may not agree atm.
  // This is a confluence problem that will need to be addressed.
  // For now, we know PadOp and ConcatOp are fine.
  assert((isa<tensor::PadOp, tensor::ConcatOp>(op.getOperation())) &&
         "incorrect op");
  Operation *newOp = rewriter.clone(*op);
  for (auto [reifiedTy, oldRes] : llvm::zip(outTypes, op->getResults())) {
    OpResult newRes = newOp->getResult(oldRes.getResultNumber());
    Type oldTy = oldRes.getType();
    // Continue if the type remained invariant or is not shaped.
    if (oldTy == reifiedTy || !isa<MemRefType, RankedTensorType>(oldTy)) {
      newResults.push_back(newRes);
      continue;
    }

    // Update the type.
    newRes.setType(reifiedTy);
    if (isa<RankedTensorType>(reifiedTy)) {
      newResults.push_back(rewriter.create<tensor::CastOp>(loc, oldTy, newRes));
    } else {
      assert(isa<MemRefType>(reifiedTy) && "expected a memref type");
      newResults.push_back(rewriter.create<memref::CastOp>(loc, oldTy, newRes));
    }
  }

  LLVM_DEBUG({
    DBGS() << "- reified results " << llvm::interleaved_array(newResults)
           << "\n";
  });
  rewriter.replaceOp(op, newResults);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
struct ReifyResultShapesPass final
    : public memref::impl::ReifyResultShapesPassBase<ReifyResultShapesPass> {
  void runOnOperation() override;
};
} // namespace

void ReifyResultShapesPass::runOnOperation() {
  SmallVector<ReifyRankedShapedTypeOpInterface> ops;
  getOperation()->walk([&](ReifyRankedShapedTypeOpInterface op) {
    // Handle ops that are not DPS and that do not carry an tied operand shapes.
    // For now, limit to tensor::PadOp and tensor::ConcatOp.
    if (!isa<tensor::PadOp, tensor::ConcatOp>(op.getOperation()))
      return;
    ops.push_back(op);
  });
  IRRewriter rewriter(&getContext());
  for (ReifyRankedShapedTypeOpInterface op : ops) {
    rewriter.setInsertionPoint(op);
    (void)reifyOpResultShapes(rewriter, op);
  }
}
