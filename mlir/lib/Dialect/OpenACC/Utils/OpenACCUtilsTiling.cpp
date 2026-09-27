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
#include "llvm/ADT/STLExtras.h"

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
static mlir::acc::LoopOp createACCLoopFromOriginal(
    mlir::acc::LoopOp origLoop, mlir::RewriterBase &rewriter,
    mlir::ValueRange lb, mlir::ValueRange ub, mlir::ValueRange step,
    mlir::DenseBoolArrayAttr inclusiveUBAttr,
    mlir::acc::CombinedConstructsTypeAttr combinedAttr, mlir::Location loc) {
  mlir::ArrayAttr collapseAttr = mlir::ArrayAttr{};
  mlir::ArrayAttr collapseDeviceTypeAttr = mlir::ArrayAttr{};
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

// Move ops from source to target Loop and replace uses of IVs
static void moveOpsAndReplaceIVs(mlir::acc::LoopOp sourceLoop,
                                 mlir::acc::LoopOp targetLoop,
                                 llvm::ArrayRef<mlir::Value> newIVs,
                                 llvm::ArrayRef<mlir::Value> origIVs,
                                 size_t nOps, mlir::RewriterBase &rewriter) {
  // nOps includes the terminator; move all ops except the terminator:
  // [begin, begin + nOps - 1)
  mlir::Block::iterator begin = sourceLoop.getBody().begin();
  mlir::Block::iterator end = std::next(begin, nOps - 1);

  // Notify the rewriter about all ops being moved (and their nested ops).
  // Directly moved ops have their parent block changed (rewriter fingerprint
  // tracking invalidated). Nested ops may have operands replaced by
  // replaceAllUsesInRegionWith below.
  llvm::SmallVector<mlir::Operation *> movedOps;
  for (mlir::Block::iterator it = begin; it != end; ++it)
    it->walk([&](mlir::Operation *op) {
      movedOps.push_back(op);
      rewriter.startOpModification(op);
    });

  targetLoop.getBody().getOperations().splice(
      targetLoop.getBody().getOperations().begin(),
      sourceLoop.getBody().getOperations(), begin, end);

  // Replace uses of origIV with newIV
  for (auto [i, newIV] : llvm::enumerate(newIVs))
    mlir::replaceAllUsesInRegionWith(origIVs[i], newIV, targetLoop.getRegion());

  for (mlir::Operation *op : movedOps)
    rewriter.finalizeOpModification(op);
}

// Create a single "element group" loop nested in `tileLoop`, carrying
// `ivTypes.size()` induction variables so the whole element space is one
// multi-IV loop. The element group carries vector or worker but not gang.
static mlir::acc::LoopOp
createElementGroupLoop(mlir::acc::LoopOp tileLoop, mlir::RewriterBase &rewriter,
                       mlir::ValueRange lbs, mlir::ValueRange ubs,
                       mlir::ValueRange steps,
                       mlir::DenseBoolArrayAttr inclusiveUBAttr,
                       llvm::ArrayRef<mlir::Type> ivTypes, mlir::Location loc) {
  mlir::acc::LoopOp elementLoop = createACCLoopFromOriginal(
      tileLoop, rewriter, lbs, ubs, steps, inclusiveUBAttr,
      mlir::acc::CombinedConstructsTypeAttr{}, loc);

  // Drop gang from the element group, keeping vector/worker. The operand
  // values must be cleared too, not just the attributes.
  rewriter.startOpModification(elementLoop);
  if (tileLoop.hasGang() ||
      tileLoop.getGangValue(mlir::acc::GangArgType::Num) ||
      tileLoop.getGangValue(mlir::acc::GangArgType::Dim) ||
      tileLoop.getGangValue(mlir::acc::GangArgType::Static)) {
    elementLoop.removeGangAttr();
    elementLoop.removeGangOperandsArgTypeAttr();
    elementLoop.removeGangOperandsSegmentsAttr();
    elementLoop.removeGangOperandsDeviceTypeAttr();
    elementLoop.getGangOperandsMutable().clear();
  }
  if (tileLoop.hasVector() || tileLoop.getVectorValue()) {
    elementLoop.removeWorkerAttr();
    elementLoop.removeWorkerNumOperandsDeviceTypeAttr();
    elementLoop.getWorkerNumOperandsMutable().clear();
  }
  rewriter.finalizeOpModification(elementLoop);

  // Create the element loop body: one block argument per IV plus a terminator.
  mlir::Block *blk = rewriter.createBlock(&elementLoop.getRegion(),
                                          elementLoop.getRegion().begin());
  rewriter.setInsertionPointToEnd(blk);
  mlir::acc::YieldOp::create(rewriter, loc);
  for (mlir::Type ivType : ivTypes)
    elementLoop.getBody().addArgument(ivType, loc);

  return elementLoop;
}

mlir::acc::LoopOp
mlir::acc::tileACCLoops(mlir::acc::LoopOp tileLoop,
                        const llvm::SmallVector<mlir::Value> &tileSizes,
                        int32_t defaultTileSize, mlir::RewriterBase &rewriter) {
  // Tile a single fused acc.loop that carries all associated induction
  // variables. This keeps the tile iterations as one multi-IV "tile group"
  // loop and the in-tile iterations as one multi-IV "element group" loop, each
  // spanning all of its induction variables.
  const mlir::Location loc = tileLoop.getLoc();
  const unsigned tileCount = tileSizes.size();

  llvm::SmallVector<mlir::Value, 3> origIVs(tileLoop.getBody().getArguments());
  llvm::SmallVector<mlir::Value, 3> origUBs(tileLoop.getUpperbound());
  llvm::SmallVector<mlir::Value, 3> origSteps(tileLoop.getStep());
  const unsigned numIVs = origIVs.size();
  const size_t nOps = tileLoop.getBody().getOperations().size();

  // Original inclusive-UB flags (default false when the attribute is absent).
  llvm::SmallVector<bool> inclusiveUBs;
  for (unsigned i = 0; i < numIVs; ++i) {
    if (tileLoop.getInclusiveUpperboundAttr())
      inclusiveUBs.push_back(
          tileLoop.getInclusiveUpperboundAttr().asArrayRef()[i]);
    else
      inclusiveUBs.push_back(false);
  }

  // Scale each tiled dimension's step by its tile size to form the tile group
  // loop steps.
  rewriter.setInsertionPoint(tileLoop);
  llvm::SmallVector<mlir::Value, 3> scaledSteps;
  llvm::SmallVector<mlir::Value, 3> tileLoopSteps;
  for (unsigned i = 0; i < numIVs; ++i) {
    if (i < tileCount) {
      mlir::Value tileSize = resolveAndCastTileSize(
          tileSizes[i], defaultTileSize, origSteps[i].getType(), rewriter, loc);
      mlir::Value scaled =
          mlir::arith::MulIOp::create(rewriter, loc, origSteps[i], tileSize);
      scaledSteps.push_back(scaled);
      tileLoopSteps.push_back(scaled);
    } else {
      tileLoopSteps.push_back(origSteps[i]);
    }
  }

  // Compute the element-loop upper bounds min(origUB, origIV + scaledStep).
  rewriter.setInsertionPoint(tileLoop.getBody().getTerminator());
  llvm::SmallVector<mlir::Value, 3> elemLBs, elemUBs, elemSteps;
  llvm::SmallVector<mlir::Type, 3> elemIVTypes;
  llvm::SmallVector<bool> elemInclusiveUBs;
  for (unsigned i = 0; i < tileCount; ++i) {
    mlir::Value stepped =
        mlir::arith::AddIOp::create(rewriter, loc, origIVs[i], scaledSteps[i]);
    mlir::Value newUB = stepped;
    if (inclusiveUBs[i]) {
      // Inclusive UB: min(origUB, origIV + (scaledStep - 1)).
      mlir::Value c1 = mlir::arith::ConstantOp::create(
          rewriter, loc, scaledSteps[i].getType(),
          rewriter.getIntegerAttr(scaledSteps[i].getType(), 1));
      newUB = mlir::arith::SubIOp::create(rewriter, loc, stepped, c1);
    }
    elemUBs.push_back(
        mlir::arith::MinSIOp::create(rewriter, loc, origUBs[i], newUB));
    elemLBs.push_back(origIVs[i]);
    elemSteps.push_back(origSteps[i]);
    elemIVTypes.push_back(origIVs[i].getType());
    elemInclusiveUBs.push_back(inclusiveUBs[i]);
  }

  // Only attach an inclusiveUpperbound attribute if at least one element
  // dimension is inclusive.
  mlir::DenseBoolArrayAttr elemInclAttr = mlir::DenseBoolArrayAttr{};
  if (llvm::is_contained(elemInclusiveUBs, true))
    elemInclAttr = rewriter.getDenseBoolArrayAttr(elemInclusiveUBs);

  // Create the element group loop from the unmodified tile loop.
  mlir::acc::LoopOp elementLoop =
      createElementGroupLoop(tileLoop, rewriter, elemLBs, elemUBs, elemSteps,
                             elemInclAttr, elemIVTypes, loc);

  // Move the original body into the element loop and remap the tiled IVs to the
  // element IVs.
  llvm::SmallVector<mlir::Value, 3> newIVs(
      elementLoop.getBody().getArguments());
  llvm::SmallVector<mlir::Value, 3> tiledOrigIVs(origIVs.begin(),
                                                 origIVs.begin() + tileCount);
  moveOpsAndReplaceIVs(tileLoop, elementLoop, newIVs, tiledOrigIVs, nOps,
                       rewriter);

  // Turn the fused loop into the tile group: scaled steps, gang only.
  rewriter.startOpModification(tileLoop);
  tileLoop.getStepMutable().clear();
  tileLoop.getStepMutable().append(tileLoopSteps);
  removeWorkerVectorFromLoop(tileLoop);
  rewriter.finalizeOpModification(tileLoop);

  return tileLoop;
}
