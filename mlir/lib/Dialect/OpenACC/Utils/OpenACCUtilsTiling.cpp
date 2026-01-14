//===- OpenACCUtilsTiling.cpp - OpenACC Loop Tiling Utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for tiling OpenACC loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsTiling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/RegionUtils.h"

// Resolve unknown tile sizes (represented as -1 for tile(*)) to the default.
// Returns a value with the same type as targetType.
static mlir::Value resolveAndCastTileSize(mlir::Value tileSize,
                                          int32_t defaultTileSize,
                                          mlir::Type targetType,
                                          mlir::RewriterBase &rewriter,
                                          mlir::Location loc) {
  auto constVal = mlir::getConstantIntValue(tileSize);
  if (constVal && *constVal < 0) {
    // Create constant with the target type directly
    return mlir::arith::ConstantOp::create(
        rewriter, loc, targetType,
        rewriter.getIntegerAttr(targetType, defaultTileSize));
  }
  return mlir::getValueOrCreateCastToIndexLike(rewriter, loc, targetType,
                                               tileSize);
}

// Remove vector/worker attributes from loop
static void removeWorkerVectorFromLoop(mlir::acc::LoopOp loop) {
  if (loop.hasVector() || loop.getVectorValue()) {
    loop.removeVectorAttr();
    loop.removeVectorOperandsDeviceTypeAttr();
  } else if (loop.hasWorker() || loop.getWorkerValue()) {
    loop.removeWorkerAttr();
    loop.removeWorkerNumOperandsDeviceTypeAttr();
  }
}

// Create a new ACC loop with new steps, lb, ub from original loop
static mlir::acc::LoopOp
createACCLoopFromOriginal(mlir::acc::LoopOp origLoop,
                          mlir::RewriterBase &rewriter, mlir::ValueRange lb,
                          mlir::ValueRange ub, mlir::ValueRange step,
                          mlir::DenseBoolArrayAttr inclusiveUBAttr,
                          mlir::acc::CombinedConstructsTypeAttr combinedAttr,
                          mlir::Location loc, bool preserveCollapse) {
  mlir::ArrayAttr collapseAttr = mlir::ArrayAttr{};
  mlir::ArrayAttr collapseDeviceTypeAttr = mlir::ArrayAttr{};
  if (preserveCollapse) {
    collapseAttr = origLoop.getCollapseAttr();
    collapseDeviceTypeAttr = origLoop.getCollapseDeviceTypeAttr();
  }
  auto newLoop = mlir::acc::LoopOp::create(
      rewriter, loc, origLoop->getResultTypes(), lb, ub, step, inclusiveUBAttr,
      collapseAttr, collapseDeviceTypeAttr, origLoop.getGangOperands(),
      origLoop.getGangOperandsArgTypeAttr(),
      origLoop.getGangOperandsSegmentsAttr(),
      origLoop.getGangOperandsDeviceTypeAttr(), origLoop.getWorkerNumOperands(),
      origLoop.getWorkerNumOperandsDeviceTypeAttr(),
      origLoop.getVectorOperands(), origLoop.getVectorOperandsDeviceTypeAttr(),
      origLoop.getSeqAttr(), origLoop.getIndependentAttr(),
      origLoop.getAuto_Attr(), origLoop.getGangAttr(), origLoop.getWorkerAttr(),
      origLoop.getVectorAttr(), mlir::ValueRange{}, mlir::DenseI32ArrayAttr{},
      mlir::ArrayAttr{}, origLoop.getCacheOperands(),
      origLoop.getPrivateOperands(), origLoop.getFirstprivateOperands(),
      origLoop.getReductionOperands(), combinedAttr);
  return newLoop;
}

// Create inner loop inside input loop
static mlir::acc::LoopOp
createInnerLoop(mlir::acc::LoopOp inputLoop, mlir::RewriterBase &rewriter,
                mlir::ValueRange lb, mlir::ValueRange ub, mlir::ValueRange step,
                mlir::DenseBoolArrayAttr inclusiveUBAttr, mlir::Location loc) {
  mlir::acc::LoopOp elementLoop = createACCLoopFromOriginal(
      inputLoop, rewriter, lb, ub, step, inclusiveUBAttr,
      mlir::acc::CombinedConstructsTypeAttr{}, loc, /*preserveCollapse*/ false);

  // Remove gang/worker attributes from inner loops
  rewriter.startOpModification(elementLoop);
  if (inputLoop.hasGang() ||
      inputLoop.getGangValue(mlir::acc::GangArgType::Num) ||
      inputLoop.getGangValue(mlir::acc::GangArgType::Dim) ||
      inputLoop.getGangValue(mlir::acc::GangArgType::Static)) {
    elementLoop.removeGangAttr();
    elementLoop.removeGangOperandsArgTypeAttr();
    elementLoop.removeGangOperandsSegmentsAttr();
    elementLoop.removeGangOperandsDeviceTypeAttr();
  }
  if (inputLoop.hasVector() || inputLoop.getVectorValue()) {
    elementLoop.removeWorkerAttr();
    elementLoop.removeWorkerNumOperandsDeviceTypeAttr();
  }
  rewriter.finalizeOpModification(elementLoop);

  // Create empty block in elementLoop and add IV argument
  mlir::Block *blk = rewriter.createBlock(&elementLoop.getRegion(),
                                          elementLoop.getRegion().begin());
  rewriter.setInsertionPointToEnd(blk);
  mlir::acc::YieldOp::create(rewriter, loc);
  elementLoop.getBody().addArgument(
      inputLoop.getBody().getArgument(0).getType(), loc);

  return elementLoop;
}

// Move ops from source to target Loop and replace uses of IVs
static void moveOpsAndReplaceIVs(mlir::acc::LoopOp sourceLoop,
                                 mlir::acc::LoopOp targetLoop,
                                 llvm::ArrayRef<mlir::Value> newIVs,
                                 llvm::ArrayRef<mlir::Value> origIVs,
                                 size_t nOps, mlir::RewriterBase &rewriter) {
  // Move ops from source to target loop [begin, begin + nOps - 1)
  mlir::Block::iterator begin = sourceLoop.getBody().begin();
  targetLoop.getBody().getOperations().splice(
      targetLoop.getBody().getOperations().begin(),
      sourceLoop.getBody().getOperations(), begin, std::next(begin, nOps - 1));

  // Replace uses of origIV with newIV
  for (auto [i, newIV] : llvm::enumerate(newIVs))
    mlir::replaceAllUsesInRegionWith(origIVs[i], newIV, targetLoop.getRegion());
}

mlir::acc::LoopOp
mlir::acc::tileACCLoops(llvm::SmallVector<mlir::acc::LoopOp> &tileLoops,
                        const llvm::SmallVector<mlir::Value> &tileSizes,
                        int32_t defaultTileSize, mlir::RewriterBase &rewriter) {
  // Tile collapsed and/or nested loops
  mlir::acc::LoopOp outerLoop = tileLoops[0];
  const mlir::Location loc = outerLoop.getLoc();

  mlir::acc::LoopOp innerLoop = tileLoops[tileLoops.size() - 1];
  llvm::SmallVector<mlir::Value, 3> origIVs;
  llvm::SmallVector<mlir::Value, 3> origSteps;
  llvm::SmallVector<mlir::Value, 3> origUBs;
  llvm::SmallVector<mlir::Value, 3> newSteps;
  llvm::SmallVector<mlir::Value, 3> newUBs;
  llvm::SmallVector<mlir::Value, 3> newIVs;
  size_t nOps = innerLoop.getBody().getOperations().size();

  // Extract original inclusiveUBs
  llvm::SmallVector<bool> inclusiveUBs;
  for (auto tileLoop : tileLoops) {
    for (auto [j, step] : llvm::enumerate(tileLoop.getStep())) {
      // inclusiveUBs are present on the IR from Fortran frontend for DO loops
      // but might not be present from other frontends (python)
      // So check if it exists
      if (tileLoop.getInclusiveUpperboundAttr())
        inclusiveUBs.push_back(
            tileLoop.getInclusiveUpperboundAttr().asArrayRef()[j]);
      else
        inclusiveUBs.push_back(false);
    }
  }

  // Extract original ivs, UBs, steps, and calculate new steps
  rewriter.setInsertionPoint(outerLoop);
  for (auto [i, tileLoop] : llvm::enumerate(tileLoops)) {
    for (auto arg : tileLoop.getBody().getArguments())
      origIVs.push_back(arg);
    for (auto ub : tileLoop.getUpperbound())
      origUBs.push_back(ub);

    llvm::SmallVector<mlir::Value, 3> currentLoopSteps;
    for (auto [j, step] : llvm::enumerate(tileLoop.getStep())) {
      origSteps.push_back(step);
      if (i + j >= tileSizes.size()) {
        currentLoopSteps.push_back(step);
      } else {
        mlir::Value tileSize = resolveAndCastTileSize(
            tileSizes[i + j], defaultTileSize, step.getType(), rewriter, loc);
        auto newLoopStep =
            mlir::arith::MulIOp::create(rewriter, loc, step, tileSize);
        currentLoopSteps.push_back(newLoopStep);
        newSteps.push_back(newLoopStep);
      }
    }

    rewriter.startOpModification(tileLoop);
    tileLoop.getStepMutable().clear();
    tileLoop.getStepMutable().append(currentLoopSteps);
    rewriter.finalizeOpModification(tileLoop);
  }

  // Calculate new upper bounds for element loops
  for (size_t i = 0; i < newSteps.size(); i++) {
    rewriter.setInsertionPoint(innerLoop.getBody().getTerminator());
    // UpperBound: min(origUB, origIV+(originalStep*tile_size))
    auto stepped =
        mlir::arith::AddIOp::create(rewriter, loc, origIVs[i], newSteps[i]);
    mlir::Value newUB = stepped;
    if (inclusiveUBs[i]) {
      // Handle InclusiveUB
      // UpperBound: min(origUB, origIV+(originalStep*tile_size - 1))
      auto c1 = mlir::arith::ConstantOp::create(
          rewriter, loc, newSteps[i].getType(),
          rewriter.getIntegerAttr(newSteps[i].getType(), 1));
      newUB = mlir::arith::SubIOp::create(rewriter, loc, stepped, c1);
    }
    newUBs.push_back(
        mlir::arith::MinSIOp::create(rewriter, loc, origUBs[i], newUB));
  }

  // Create and insert nested elementLoopOps before terminator of outer loopOp
  mlir::acc::LoopOp currentLoop = innerLoop;
  for (size_t i = 0; i < tileSizes.size(); i++) {
    rewriter.setInsertionPoint(currentLoop.getBody().getTerminator());
    mlir::DenseBoolArrayAttr inclusiveUBAttr = mlir::DenseBoolArrayAttr{};
    if (inclusiveUBs[i])
      inclusiveUBAttr = rewriter.getDenseBoolArrayAttr({true});

    mlir::acc::LoopOp elementLoop =
        createInnerLoop(innerLoop, rewriter, mlir::ValueRange{origIVs[i]},
                        mlir::ValueRange{newUBs[i]},
                        mlir::ValueRange{origSteps[i]}, inclusiveUBAttr, loc);

    // Remove vector/worker attributes from inner element loops except
    // outermost element loop
    if (i > 0) {
      rewriter.startOpModification(elementLoop);
      removeWorkerVectorFromLoop(elementLoop);
      rewriter.finalizeOpModification(elementLoop);
    }
    newIVs.push_back(elementLoop.getBody().getArgument(0));
    currentLoop = elementLoop;
  }

  // Remove vector/worker attributes from outer tile loops
  for (auto tileLoop : tileLoops) {
    rewriter.startOpModification(tileLoop);
    removeWorkerVectorFromLoop(tileLoop);
    rewriter.finalizeOpModification(tileLoop);
  }

  // Move ops from inner tile loop to inner element loop and replace IV uses
  moveOpsAndReplaceIVs(innerLoop, currentLoop, newIVs, origIVs, nOps, rewriter);

  return outerLoop;
}

llvm::SmallVector<mlir::acc::LoopOp>
mlir::acc::uncollapseLoops(mlir::acc::LoopOp origLoop, unsigned tileCount,
                           unsigned collapseCount,
                           mlir::RewriterBase &rewriter) {
  llvm::SmallVector<mlir::acc::LoopOp> newLoops;
  llvm::SmallVector<mlir::Value, 3> newIVs;
  mlir::Location loc = origLoop.getLoc();
  llvm::SmallVector<bool> newInclusiveUBs;
  llvm::SmallVector<mlir::Value, 3> lbs, ubs, steps;
  for (unsigned i = 0; i < collapseCount; i++) {
    // inclusiveUpperbound attribute might not be set, default to false
    bool inclusiveUB = false;
    if (origLoop.getInclusiveUpperboundAttr())
      inclusiveUB = origLoop.getInclusiveUpperboundAttr().asArrayRef()[i];
    newInclusiveUBs.push_back(inclusiveUB);
    lbs.push_back(origLoop.getLowerbound()[i]);
    ubs.push_back(origLoop.getUpperbound()[i]);
    steps.push_back(origLoop.getStep()[i]);
  }
  mlir::acc::LoopOp outerLoop = createACCLoopFromOriginal(
      origLoop, rewriter, lbs, ubs, steps,
      rewriter.getDenseBoolArrayAttr(newInclusiveUBs),
      origLoop.getCombinedAttr(), loc, /*preserveCollapse*/ true);
  mlir::Block *blk = rewriter.createBlock(&outerLoop.getRegion(),
                                          outerLoop.getRegion().begin());
  rewriter.setInsertionPointToEnd(blk);
  mlir::acc::YieldOp::create(rewriter, loc);
  for (unsigned i = 0; i < collapseCount; i++) {
    outerLoop.getBody().addArgument(origLoop.getBody().getArgument(i).getType(),
                                    loc);
    newIVs.push_back(outerLoop.getBody().getArgument(i));
  }
  newLoops.push_back(outerLoop);

  mlir::acc::LoopOp currentLoopOp = outerLoop;
  for (unsigned i = collapseCount; i < tileCount; i++) {
    rewriter.setInsertionPoint(currentLoopOp.getBody().getTerminator());
    bool inclusiveUB = false;
    if (origLoop.getInclusiveUpperboundAttr())
      inclusiveUB = origLoop.getInclusiveUpperboundAttr().asArrayRef()[i];
    mlir::DenseBoolArrayAttr inclusiveUBAttr =
        rewriter.getDenseBoolArrayAttr({inclusiveUB});
    mlir::acc::LoopOp innerLoop = createInnerLoop(
        origLoop, rewriter, mlir::ValueRange{origLoop.getLowerbound()[i]},
        mlir::ValueRange{origLoop.getUpperbound()[i]},
        mlir::ValueRange{origLoop.getStep()[i]}, inclusiveUBAttr, loc);
    newIVs.push_back(innerLoop.getBody().getArgument(0));
    newLoops.push_back(innerLoop);
    currentLoopOp = innerLoop;
  }
  // Move ops from origLoop to innermost loop and replace uses of IVs
  size_t nOps = origLoop.getBody().getOperations().size();
  llvm::SmallVector<mlir::Value, 3> origIVs;
  for (auto arg : origLoop.getBody().getArguments())
    origIVs.push_back(arg);
  moveOpsAndReplaceIVs(origLoop, currentLoopOp, newIVs, origIVs, nOps,
                       rewriter);

  return newLoops;
}
