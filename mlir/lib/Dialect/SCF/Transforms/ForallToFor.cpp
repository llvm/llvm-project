//===- ForallToFor.cpp - scf.forall to scf.for loop conversion ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms SCF.ForallOp's into SCF.ForOp's.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_SCFFORALLTOFORLOOP
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace llvm;
using namespace mlir;
using scf::LoopNest;

LogicalResult
mlir::scf::forallToForLoop(RewriterBase &rewriter, scf::ForallOp forallOp,
                           SmallVectorImpl<Operation *> *results) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);

  Location loc = forallOp.getLoc();
  SmallVector<Value> lbs = forallOp.getLowerBound(rewriter);
  SmallVector<Value> ubs = forallOp.getUpperBound(rewriter);
  SmallVector<Value> steps = forallOp.getStep(rewriter);
  SmallVector<Value> iterArgs;
  for (auto result : forallOp->getResults()) {
    iterArgs.push_back(forallOp.getTiedOpOperand(result)->get());
  }

  InParallelOp threadReduction =
      cast<InParallelOp>(forallOp.getBody()->getTerminator());
  SmallVector<tensor::ParallelInsertSliceOp> regionArgToSlice;
  for (auto &op : threadReduction.getBody()->getOperations()) {
    auto parallelInsert = dyn_cast<tensor::ParallelInsertSliceOp>(op);
    if (!parallelInsert) {
      return op.emitOpError() << "expected parallel insert slice op";
    }
    regionArgToSlice.push_back(parallelInsert);
  }

  function_ref<ValueVector(OpBuilder &, Location, ValueRange, ValueRange)>
      build = [&](OpBuilder &rewriter, Location loc, ValueRange ivs,
                  ValueRange regionArgs) -> ValueVector {
    SmallVector<Value> res;
    for (auto [i, val] : llvm::enumerate(regionArgs)) {
      tensor::ParallelInsertSliceOp sliceOp = regionArgToSlice[i];

      // Map new induction variables where applicable.

      SmallVector<OpFoldResult> sliceOpOffsets = sliceOp.getMixedOffsets();
      for (OpFoldResult offset : sliceOpOffsets) {
        if (offset.is<Value>()) {
          Value dynamicOffset = offset.get<Value>();
          SmallVector<Value> originalInductionVars =
              forallOp.getInductionVars();
          auto *it = llvm::find(originalInductionVars, dynamicOffset);
          if (it != originalInductionVars.end()) {
            size_t index = std::distance(originalInductionVars.begin(), it);
            offset = ivs[index];
          }
        }
      }

      SmallVector<OpFoldResult> sliceOpSizes = sliceOp.getMixedSizes();
      for (OpFoldResult size : sliceOpSizes) {
        if (size.is<Value>()) {
          Value dynamicSize = size.get<Value>();
          SmallVector<Value> originalInductionVars =
              forallOp.getInductionVars();
          auto *it = llvm::find(originalInductionVars, dynamicSize);
          if (it != originalInductionVars.end()) {
            size_t index = std::distance(originalInductionVars.begin(), it);
            size = ivs[index];
          }
        }
      }

      SmallVector<OpFoldResult> sliceOpStrides = sliceOp.getMixedStrides();
      for (OpFoldResult stride : sliceOpStrides) {
        if (stride.is<Value>()) {
          Value dynamicStride = stride.get<Value>();
          SmallVector<Value> originalInductionVars =
              forallOp.getInductionVars();
          auto *it = llvm::find(originalInductionVars, dynamicStride);
          if (it != originalInductionVars.end()) {
            size_t index = std::distance(originalInductionVars.begin(), it);
            stride = ivs[index];
          }
        }
      }

      res.push_back(rewriter.create<tensor::InsertSliceOp>(
          sliceOp->getLoc(), sliceOp.getSource(), val, sliceOpOffsets,
          sliceOpSizes, sliceOpStrides));
    }
    return res;
  };

  // Now we want to create our new loops with the innermost getting the tensor
  // insert slices appropriately.
  LoopNest loopNest =
      scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, iterArgs, build);
  SmallVector<Value> ivs = llvm::map_to_vector(
      loopNest.loops, [](scf::ForOp loop) { return loop.getInductionVar(); });

  rewriter.replaceAllOpUsesWith(forallOp,
                                {loopNest.loops.front()->getResults()});
  // Erase the parallel inserts and associated shared outputs.
  for (tensor::ParallelInsertSliceOp insertSlice :
       llvm::make_early_inc_range(regionArgToSlice)) {
    auto loopBlockArg = dyn_cast<BlockArgument>(insertSlice.getDest());
    if (!loopBlockArg || loopBlockArg.getOwner()->getParentOp() != forallOp) {
      insertSlice->emitOpError()
          << "expected destination to be block argument in loop";
    }
    rewriter.eraseOp(insertSlice);
    rewriter.modifyOpInPlace(forallOp, [&]() {
      forallOp.getBody()->eraseArgument(loopBlockArg.getArgNumber());
    });
  }
  rewriter.eraseOp(forallOp.getTerminator());

  Block *innermostBlock = loopNest.loops.back().getBody();

  rewriter.inlineBlockBefore(forallOp.getBody(), innermostBlock,
                             innermostBlock->front().getIterator(), ivs);
  rewriter.eraseOp(forallOp);

  if (results) {
    llvm::move(loopNest.loops, std::back_inserter(*results));
  }

  return success();
}

namespace {
struct ForallToForLoop : public impl::SCFForallToForLoopBase<ForallToForLoop> {
  void runOnOperation() override {
    Operation *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    parentOp->walk([&](scf::ForallOp forallOp) {
      if (failed(scf::forallToForLoop(rewriter, forallOp))) {
        return signalPassFailure();
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createForallToForLoopPass() {
  return std::make_unique<ForallToForLoop>();
}
