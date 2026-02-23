//===- VectorContractToPackedTypeTiledDotProduct.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/AMX/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::amx;

namespace {

static Operation *traceToVectorReadLikeParentOperation(Value v) {
  while (true) {
    // Case 1: Value defined by an operation
    if (Operation *defOp = v.getDefiningOp()) {
      if (isa<vector::TransferReadOp, vector::LoadOp>(defOp)) {
        return defOp;
      }

      if (isa<vector::ShapeCastOp, vector::ShuffleOp>(defOp)) {
        return nullptr;
      }

      return nullptr;
    }

    // Case 2: BlockArgument (scf.for iter_arg)
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      auto *parentOp = barg.getOwner()->getParentOp();

      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        unsigned argNum = barg.getArgNumber();

        // arg0 = induction variable (not an iter_arg)
        if (argNum == 0)
          return nullptr;

        unsigned iterIdx = argNum - 1;
        v = forOp.getInitArgs()[iterIdx];
        continue;
      }

      return nullptr;
    }

    return nullptr;
  }
}

static Operation *traceToVectorWriteLikeUserOperation(Value v) {
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();

    // --- TERMINAL OPS ---
    if (isa<vector::TransferWriteOp>(user) || isa<vector::StoreOp>(user)) {
      return user;
    }

    if (isa<vector::ShapeCastOp, vector::ShuffleOp>(user)) {
      return nullptr;
    }

    // --- SCF YIELD ---
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      Operation *parent = yield->getParentOp();
      unsigned idx = use.getOperandNumber();
      if (auto *res =
              traceToVectorWriteLikeUserOperation(parent->getResult(idx)))
        return res;
      continue;
    }

    // --- SCF FOR ---
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      unsigned idx = use.getOperandNumber();
      if (auto *res = traceToVectorWriteLikeUserOperation(forOp.getResult(idx)))
        return res;
      continue;
    }

    // --- GENERIC CASE ---
    for (Value res : user->getResults()) {
      if (auto *found = traceToVectorWriteLikeUserOperation(res))
        return found;
    }
  }

  return nullptr;
}

static Value collapseInnerDims(OpBuilder &builder, mlir::Location loc,
                               Value input, int64_t firstDimToCollapse) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  if (inputType.getRank() == 1)
    return input;
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < firstDimToCollapse; ++i)
    reassociation.push_back(ReassociationIndices{i});
  ReassociationIndices collapsedIndices;
  for (int64_t i = firstDimToCollapse; i < inputType.getRank(); ++i)
    collapsedIndices.push_back(i);
  reassociation.push_back(collapsedIndices);
  return memref::CollapseShapeOp::create(builder, loc, input, reassociation);
}

static std::pair<Value, SmallVector<Value>> getSrcIndxValue(OpBuilder &rewriter,
                                                            Location loc,
                                                            Value operand,
                                                            bool isNotAcc) {
  Value srcBuff;
  SmallVector<OpFoldResult> indexVals;
  llvm::TypeSwitch<Operation *>(operand.getDefiningOp())
      .Case<TransferReadOp, LoadOp>([&](auto readOp) {
        indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                              readOp.getIndices().end());
        srcBuff = readOp.getOperand(0);
      });

  if (isNotAcc) {
    indexVals.pop_back();
  }

  SmallVector<Value> indices;
  indices.reserve(indexVals.size());

  for (OpFoldResult ofr : indexVals) {
    indices.push_back(
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }

  if (isNotAcc) {
    auto subviewType = cast<ShapedType>(srcBuff.getType());
    auto subviewRank = subviewType.getRank();
    srcBuff = collapseInnerDims(rewriter, loc, srcBuff, subviewRank - 2);
  }
  return {srcBuff, indices};
}

static unsigned getIndexPosition(Value operand, scf::ForOp loop) {
  Value iv = loop.getInductionVar();

  Value srcBuff;
  SmallVector<OpFoldResult> indexVals;
  llvm::TypeSwitch<Operation *>(operand.getDefiningOp())
      .Case<TransferReadOp, LoadOp>([&](auto readOp) {
        indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                              readOp.getIndices().end());
        srcBuff = readOp.getOperand(0);
      });

  auto subview = srcBuff.getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return 0;

  auto offsets = subview.getOffsets();

  for (auto it : llvm::enumerate(offsets)) {
    if (it.value() == iv)
      return it.index();
  }

  return 0;
}

static amx::TileLoadOp createTileLoads(OpBuilder &rewriter, Location loc,
                                       Value operand, Value mat, Type ipType,
                                       bool rhs) {

  SmallVector<OpFoldResult> indexVals;
  llvm::TypeSwitch<Operation *>(operand.getDefiningOp())
      .Case<TransferReadOp, LoadOp>([&](auto readOp) {
        indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                              readOp.getIndices().end());
      });

  indexVals.pop_back();
  SmallVector<Value> indices;
  indices.reserve(indexVals.size());

  for (OpFoldResult ofr : indexVals) {
    indices.push_back(
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }

  if (rhs) {
    int offset = 4;
    if (ipType.isBF16()) {
      offset = 2;
    }

    auto c2 = arith::ConstantIndexOp::create(rewriter, loc, offset);
    indices[indices.size() - 1] =
        arith::MulIOp::create(rewriter, loc, indices[indices.size() - 1], c2);
  }

  amx::TileType tileType = amx::TileType::get({16, 64}, ipType);
  if (ipType.isBF16()) {
    tileType = amx::TileType::get({16, 32}, ipType);
  }

  auto load = amx::TileLoadOp::create(rewriter, loc, tileType, mat, indices);

  return load;
}

static SmallVector<Value> createTiledDp(OpBuilder &rewriter, Location loc,
                                        SmallVector<vector::ContractionOp> ops,
                                        Value matA, Value matB, Type ipType,
                                        Type opType, ValueRange accIterArgs) {
  auto subviewType = cast<ShapedType>(matA.getType());
  auto subviewRank = subviewType.getRank();
  auto collapsedOpnd = collapseInnerDims(rewriter, loc, matA, subviewRank - 2);

  auto subviewType1 = cast<ShapedType>(matB.getType());
  auto subviewRank1 = subviewType1.getRank();
  auto collapsedOpnd1 =
      collapseInnerDims(rewriter, loc, matB, subviewRank1 - 2);

  SmallVector<Value> accumulators;
  llvm::DenseMap<Operation *, amx::TileLoadOp> readsToTileLoads;

  for (size_t i = 0; i < ops.size(); i++) {

    Operation *readOpLhs = ops[i].getLhs().getDefiningOp();

    amx::TileLoadOp tilesLhs;

    auto it = readsToTileLoads.find(readOpLhs);
    if (it != readsToTileLoads.end()) {
      tilesLhs = it->second;
    } else {

      tilesLhs = createTileLoads(rewriter, loc, ops[i].getLhs(), collapsedOpnd,
                                 ipType, false);
      readsToTileLoads.try_emplace(readOpLhs, tilesLhs);
    }

    Operation *readOpRhs = ops[i].getRhs().getDefiningOp();

    amx::TileLoadOp tilesRhs;

    auto it1 = readsToTileLoads.find(readOpRhs);
    if (it1 != readsToTileLoads.end()) {
      tilesRhs = it1->second;
    } else {

      tilesRhs = createTileLoads(rewriter, loc, ops[i].getRhs(), collapsedOpnd1,
                                 ipType, true);
      readsToTileLoads.try_emplace(readOpRhs, tilesRhs);
    }

    auto tileType1 = amx::TileType::get({16, 16}, opType);

    Value dp;
    if (ipType.isBF16())
      dp = amx::TileMulFOp::create(rewriter, loc, tileType1, tilesLhs, tilesRhs,
                                   accIterArgs[i]);

    if (ipType.isSignlessInteger(8))
      dp = amx::TileMulIOp::create(rewriter, loc, tileType1, tilesLhs, tilesRhs,
                                   accIterArgs[i]);

    accumulators.push_back(dp);
  }
  return accumulators;
}

static SmallVector<Value> createTileZeros(OpBuilder &rewriter, Location loc,
                                          Type opType, scf::ForOp outerLoop,
                                          int64_t size) {
  rewriter.setInsertionPoint(outerLoop);

  SmallVector<Value> loopItrArgs;
  auto zeroTileType = amx::TileType::get({16, 16}, opType);

  for (int i = 0; i < size; i++) {
    auto zeroTile = amx::TileZeroOp::create(rewriter, loc, zeroTileType);
    loopItrArgs.push_back(zeroTile);
  }
  return loopItrArgs;
}

//   vector.contract <1x1xf32>, <1x16xf32> into <1x16xf32>
// ```
// to
// ```
//   vector.broadcast %lhs to <16xf32>
//   vector.fma vector<16xf32>
// ```
struct VectorContractToPackedTypeTiledDotProduct
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    Operation *accReadOp =
        traceToVectorReadLikeParentOperation(contractOp.getAcc());

    Operation *resultWriteOp =
        traceToVectorWriteLikeUserOperation(contractOp.getResult());

    if (!accReadOp || !resultWriteOp)
      return failure();

    /*if (accReadOp->getBlock() == contractOp->getBlock())
      return failure();

    if (resultWriteOp->getBlock() == contractOp->getBlock())
      return failure(); */

    // case for just one vc rewrite.
    if (accReadOp->getBlock() == contractOp->getBlock() &&
        resultWriteOp->getBlock() == contractOp->getBlock()) {
      Location loc = contractOp.getLoc();
      auto tileType = amx::TileType::get({16, 32}, rewriter.getBF16Type());

      auto [srcBuffLhs, indicesLhs] = getSrcIndxValue(
          rewriter, contractOp.getLoc(), contractOp.getLhs(), true);
      auto loadLhs = amx::TileLoadOp::create(rewriter, loc, tileType,
                                             srcBuffLhs, indicesLhs);

      auto [srcBuffRhs, indicesRhs] = getSrcIndxValue(
          rewriter, contractOp.getLoc(), contractOp.getRhs(), true);
      auto loadRhs = amx::TileLoadOp::create(rewriter, loc, tileType,
                                             srcBuffRhs, indicesRhs);

      auto [srcBuffAcc, indicesAcc] = getSrcIndxValue(
          rewriter, contractOp.getLoc(), contractOp.getAcc(), false);
      auto tileTypeAcc = amx::TileType::get({16, 16}, rewriter.getF32Type());
      auto loadAcc = amx::TileLoadOp::create(rewriter, loc, tileTypeAcc,
                                             srcBuffAcc, indicesAcc);

      auto tdp = amx::TileMulFOp::create(rewriter, loc, tileTypeAcc, loadLhs,
                                         loadRhs, loadAcc);
      amx::TileStoreOp::create(rewriter, loc, srcBuffAcc, indicesAcc, tdp);

      rewriter.eraseOp(resultWriteOp);
      return success();
    }

    SmallVector<scf::ForOp> list;
    Operation *current = contractOp;

    while (true) {
      Operation *parent = current->getParentOfType<scf::ForOp>();
      list.push_back(dyn_cast<scf::ForOp>(parent));

      if (accReadOp->getBlock() == parent->getBlock()) {
        break;
      }

      current = parent;
    }

    if (list.size() > 2 || list.size() == 0)
      return failure();

    SmallVector<vector::ContractionOp> ops;

    for (mlir::Operation &op : list[0].getBody()->getOperations()) {

      if (auto contract = llvm::dyn_cast<mlir::vector::ContractionOp>(op)) {
        ops.push_back(contract);
      }
    }

    Type ipType;
    Type opType;
    VectorType lhsTy = contractOp.getLhsType();
    if (lhsTy.getElementType().isBF16()) {
      ipType = rewriter.getBF16Type();
      opType = rewriter.getF32Type();
    }

    if (lhsTy.getElementType().isSignlessInteger(8)) {
      ipType = rewriter.getIntegerType(8);
      opType = rewriter.getIntegerType(32);
    }

    scf::ForOp outerLoop;
    scf::ForOp innerLoop;

    auto vectorReadOpLhs =
        contractOp.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto vectorReadOpRhs =
        contractOp.getRhs().getDefiningOp<vector::TransferReadOp>();

    scf::ForOp newLoop;
    if (list.size() == 2) {
      outerLoop = list[1];
      innerLoop = list[0];

      SmallVector<Value> loopItrArgs = createTileZeros(
          rewriter, outerLoop.getLoc(), opType, outerLoop, ops.size());

      newLoop = scf::ForOp::create(
          rewriter, outerLoop.getLoc(), outerLoop.getLowerBound(),
          outerLoop.getUpperBound(), outerLoop.getStep(), loopItrArgs,
          [&](OpBuilder &rewriterOuterLoop, Location locOuterLoop,
              Value ivOuterLoop, ValueRange iterArgsOuterLoop) {
            auto newInnerLoop = scf::ForOp::create(
                rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
                innerLoop.getUpperBound(), innerLoop.getStep(),
                iterArgsOuterLoop,
                [&](OpBuilder &rewriterNewInnerLoop, Location locNewInnerLoop,
                    Value ivNewInnerLoop, ValueRange iterArgsNewInnerLoop) {
                  IRMapping mapping;
                  mapping.map(
                      vectorReadOpLhs.getBase().getDefiningOp()->getOperand(
                          getIndexPosition(contractOp.getLhs(), outerLoop) + 1),
                      ivOuterLoop);
                  mapping.map(
                      vectorReadOpLhs.getBase().getDefiningOp()->getOperand(
                          getIndexPosition(contractOp.getLhs(), innerLoop) + 1),
                      ivNewInnerLoop);
                  auto lhsClone = rewriterNewInnerLoop.clone(
                      *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

                  IRMapping rhsMapping;
                  rhsMapping.map(
                      vectorReadOpRhs.getBase().getDefiningOp()->getOperand(
                          getIndexPosition(contractOp.getRhs(), outerLoop) + 1),
                      ivOuterLoop);
                  rhsMapping.map(
                      vectorReadOpRhs.getBase().getDefiningOp()->getOperand(
                          getIndexPosition(contractOp.getRhs(), innerLoop) + 1),
                      ivNewInnerLoop);
                  auto rhsClone = rewriterNewInnerLoop.clone(
                      *vectorReadOpRhs.getBase().getDefiningOp(), rhsMapping);

                  SmallVector<Value> accumulators = createTiledDp(
                      rewriter, locNewInnerLoop, ops, lhsClone->getResult(0),
                      rhsClone->getResult(0), ipType, opType,
                      iterArgsNewInnerLoop);

                  scf::YieldOp::create(rewriterNewInnerLoop, locNewInnerLoop,
                                       accumulators);
                });

            scf::YieldOp::create(rewriterOuterLoop, locOuterLoop,
                                 newInnerLoop.getResults());
          });
    }

    if (list.size() == 1) {
      outerLoop = list[0];

      SmallVector<Value> loopItrArgs = createTileZeros(
          rewriter, outerLoop.getLoc(), opType, outerLoop, ops.size());
      newLoop = scf::ForOp::create(
          rewriter, outerLoop.getLoc(), outerLoop.getLowerBound(),
          outerLoop.getUpperBound(), outerLoop.getStep(), loopItrArgs,
          [&](OpBuilder &rewriterOuterLoop, Location locOuterLoop,
              Value ivOuterLoop, ValueRange iterArgsOuterLoop) {
            IRMapping mapping;
            mapping.map(
                vectorReadOpLhs.getBase().getDefiningOp()->getOperand(
                    getIndexPosition(contractOp.getLhs(), outerLoop) + 1),
                ivOuterLoop);

            auto lhsClone = rewriterOuterLoop.clone(
                *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

            IRMapping rhsMapping;
            rhsMapping.map(
                vectorReadOpRhs.getBase().getDefiningOp()->getOperand(
                    getIndexPosition(contractOp.getRhs(), outerLoop) + 1),
                ivOuterLoop);

            auto rhsClone = rewriterOuterLoop.clone(
                *vectorReadOpRhs.getBase().getDefiningOp(), rhsMapping);

            SmallVector<Value> accumulators = createTiledDp(
                rewriter, locOuterLoop, ops, lhsClone->getResult(0),
                rhsClone->getResult(0), ipType, opType, iterArgsOuterLoop);

            scf::YieldOp::create(rewriterOuterLoop, locOuterLoop, accumulators);
          });
    }

    // post processing after the loop
    auto bufferType = MemRefType::get({16, 16}, opType);
    auto bBuffer =
        memref::AllocaOp::create(rewriter, outerLoop.getLoc(), bufferType);

    SmallVector<Value> dps = newLoop.getResults();

    for (size_t i = 0; i < ops.size(); i++) {
      vector::ContractionOp contOp = ops[i];
      Operation *resultWriteOp =
          traceToVectorWriteLikeUserOperation(contOp.getResult());
      rewriter.setInsertionPoint(resultWriteOp);

      Value indexOp_0 =
          arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 0);

      amx::TileStoreOp::create(rewriter, outerLoop.getLoc(), bBuffer,
                               ValueRange{indexOp_0, indexOp_0}, dps[i]);

      auto c0 = arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 0);
      auto one =
          arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 1);
      auto mBound =
          arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 16);

      scf::ForOp::create(
          rewriter, outerLoop.getLoc(), c0, mBound, one, ValueRange{},
          [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
            auto row1 = vector::LoadOp::create(rewriter, loc,
                                               VectorType::get(16, opType),
                                               bBuffer, ValueRange{iv, c0});

            Operation *readOp1 =
                traceToVectorReadLikeParentOperation(ops[i].getAcc());

            Value srcBuff;
            SmallVector<OpFoldResult> indexVals;
            llvm::TypeSwitch<Operation *>(readOp1).Case<TransferReadOp, LoadOp>(
                [&](auto readOp) {
                  indexVals = SmallVector<OpFoldResult>(
                      readOp.getIndices().begin(), readOp.getIndices().end());
                  srcBuff = readOp.getOperand(0);
                });

            SmallVector<Value> indices;
            indices.reserve(indexVals.size());

            for (OpFoldResult ofr : indexVals) {
              indices.push_back(
                  mlir::getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
            }

            Value sum = arith::AddIOp::create(builder, loc, iv, indices[0]);
            indices[0] = sum;

            auto row2 = vector::LoadOp::create(
                rewriter, loc, VectorType::get(16, opType), srcBuff, indices);

            Value addition;
            if (ipType.isBF16())
              addition = arith::AddFOp::create(rewriter, loc, row1, row2);

            if (ipType.isSignlessInteger(8))
              addition = arith::AddIOp::create(rewriter, loc, row1, row2);

            vector::StoreOp::create(builder, loc, addition, srcBuff, indices);

            scf::YieldOp::create(builder, outerLoop.getLoc());
          });

      rewriter.eraseOp(resultWriteOp);
    }

    return success();
  }
};

} // namespace

void amx::populateVectorContractToPackedTypeTiledDotProductPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractToPackedTypeTiledDotProduct>(
      patterns.getContext());
}
