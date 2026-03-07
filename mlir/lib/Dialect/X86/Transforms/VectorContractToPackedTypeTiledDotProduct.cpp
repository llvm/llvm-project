//===- VectorContractToPackedTypeTiledDotProduct.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/Utils/X86Utils.h"
#include "mlir/Dialect/X86/X86Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86;

namespace {

// Function to collapse the last two dimension (vnni and k) to help the
// amx.tile_load to correctly load the packed element type.
static Value collapseInnerDims(OpBuilder &builder, mlir::Location loc,
                               Value input) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  int64_t firstDimToCollapse = inputType.getRank() - 2;

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

// Get the MemRef source and offset index for the operands of
// vector.contract.
static FailureOr<std::pair<Value, SmallVector<Value>>>
getSrcIndxValue(OpBuilder &rewriter, Location loc, Value operand,
                bool isNotAcc) {
  Operation *defOp = operand.getDefiningOp();
  if (!defOp)
    return failure();

  Value srcBuff;
  SmallVector<OpFoldResult> indexVals;
  llvm::TypeSwitch<Operation *>(operand.getDefiningOp())
      .Case<TransferReadOp, LoadOp>([&](auto readOp) {
        indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                              readOp.getIndices().end());
        srcBuff = readOp.getOperand(0);
      });

  if (!srcBuff)
    return failure();

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
    srcBuff = collapseInnerDims(rewriter, loc, srcBuff);
  }

  return std::make_pair(srcBuff, indices);
}

// Function to validate the vector.contract operation.
static LogicalResult validateContractOps(OpBuilder &rewriter,
                                         vector::ContractionOp contractOp,
                                         unsigned int blockingFactor,
                                         Value srcBuffLhs, Value srcBuffRhs,
                                         bool srcValidate) {

  if (srcValidate) {
    // Get the MemRef buffer of LHS operand.
    auto srcIndxLhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                      contractOp.getLhs(), false);
    if (failed(srcIndxLhs))
      return failure();
    auto [buffLhs, indicesLhs] = *srcIndxLhs;

    // Get the MemRef buffer of RHS operand.
    auto srcIndxRhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                      contractOp.getRhs(), false);
    if (failed(srcIndxRhs))
      return failure();
    auto [buffRhs, indicesRhs] = *srcIndxRhs;

    // Return failure if the Memref buff didn't match.
    if (buffLhs != srcBuffLhs)
      return failure();

    if (buffRhs != srcBuffRhs)
      return failure();
  }

  VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
  if (!accTy)
    return failure();

  // The Accumulator dims should be 16 or 1. Like <1x16x16> or <16x16>.
  ArrayRef<int64_t> accShape = accTy.getShape();
  llvm::SmallVector<int64_t> nonUnitDimAcc;
  llvm::copy_if(accShape, std::back_inserter(nonUnitDimAcc),
                [](int64_t dim) { return (dim != 16 && dim != 1); });

  if (nonUnitDimAcc.size() != 0)
    return failure();

  // The LHS dims should be 16 or vnni or 1. Like <1x16x16x2> or
  // <16x16x4>. The vnni dims should be 2 or 4.
  VectorType lhsTy = contractOp.getLhsType();
  ArrayRef<int64_t> lhsShape = lhsTy.getShape();
  llvm::SmallVector<int64_t> nonUnitDimLhs;
  llvm::copy_if(lhsShape, std::back_inserter(nonUnitDimLhs),
                [](int64_t dim) { return (dim != 16 && dim != 1); });

  if (nonUnitDimLhs.size() != 1)
    return failure();

  if (nonUnitDimLhs[0] != blockingFactor)
    return failure();

  // The RHS dims should be 16 or vnni or 1. Like <1x16x16x2> or
  // <16x16x4>. The vnni dims should be 2 or 4.
  VectorType rhsTy = contractOp.getRhsType();
  ArrayRef<int64_t> rhsShape = rhsTy.getShape();
  llvm::SmallVector<int64_t> nonUnitDimRhs;
  llvm::copy_if(rhsShape, std::back_inserter(nonUnitDimRhs),
                [](int64_t dim) { return (dim != 16 && dim != 1); });

  if (nonUnitDimRhs.size() != 1)
    return failure();

  if (nonUnitDimRhs[0] != blockingFactor)
    return failure();

  return success();
}

// Returns the loop index position to get mapped during the
// MemRef type clone.
static unsigned getIndexPosition(Value operand, scf::ForOp loop) {
  Value iv = loop.getInductionVar();

  Value srcBuff;
  llvm::TypeSwitch<Operation *>(operand.getDefiningOp())
      .Case<TransferReadOp, LoadOp>(
          [&](auto readOp) { srcBuff = readOp.getOperand(0); });

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

// Creates amx.tile_loads.
static amx::TileLoadOp createTileLoads(OpBuilder &rewriter, Location loc,
                                       Value operand, Value mat, Type ipType,
                                       bool rhs, unsigned int offset) {

  auto srcIndx = getSrcIndxValue(rewriter, loc, operand, false);
  auto [srcBuff, indices] = *srcIndx;
  indices.pop_back();

  if (rhs) {
    auto cOffset = arith::ConstantIndexOp::create(rewriter, loc, offset);
    indices[indices.size() - 1] = arith::MulIOp::create(
        rewriter, loc, indices[indices.size() - 1], cOffset);
  }

  amx::TileType tileType = amx::TileType::get({16, (16 * offset)}, ipType);
  auto load = amx::TileLoadOp::create(rewriter, loc, tileType, mat, indices);
  return load;
}

// Creates tiled amx dot-products.
static SmallVector<Value> createTiledDp(OpBuilder &rewriter, Location loc,
                                        SmallVector<vector::ContractionOp> ops,
                                        Value matA, Value matB, Type ipType,
                                        Type opType, ValueRange accIterArgs,
                                        unsigned int offset) {

  auto subviewCollapseLhs = collapseInnerDims(rewriter, loc, matA);
  auto subviewCollapseRhs = collapseInnerDims(rewriter, loc, matB);

  SmallVector<Value> accumulators;
  // Stores the amx.tile_load operation vs it's equivalent vector tranfer_read
  // or load operations.
  llvm::DenseMap<Operation *, amx::TileLoadOp> readsToTileLoads;

  // Iterate over the contraction operations and compute the tiled dot-product.
  for (size_t i = 0; i < ops.size(); i++) {

    Operation *readOpLhs = ops[i].getLhs().getDefiningOp();
    amx::TileLoadOp tilesLhs;
    auto itLhs = readsToTileLoads.find(readOpLhs);
    if (itLhs != readsToTileLoads.end()) {
      tilesLhs = itLhs->second;
    } else {
      tilesLhs = createTileLoads(rewriter, loc, ops[i].getLhs(),
                                 subviewCollapseLhs, ipType, false, offset);
      readsToTileLoads.try_emplace(readOpLhs, tilesLhs);
    }

    Operation *readOpRhs = ops[i].getRhs().getDefiningOp();
    amx::TileLoadOp tilesRhs;
    auto itRhs = readsToTileLoads.find(readOpRhs);
    if (itRhs != readsToTileLoads.end()) {
      tilesRhs = itRhs->second;
    } else {
      tilesRhs = createTileLoads(rewriter, loc, ops[i].getRhs(),
                                 subviewCollapseRhs, ipType, true, offset);
      readsToTileLoads.try_emplace(readOpRhs, tilesRhs);
    }

    auto accTileType = amx::TileType::get({16, 16}, opType);

    Value dp;
    if (ipType.isBF16())
      dp = amx::TileMulFOp::create(rewriter, loc, accTileType, tilesLhs,
                                   tilesRhs, accIterArgs[i]);

    if (ipType.isSignlessInteger(8))
      dp = amx::TileMulIOp::create(rewriter, loc, accTileType, tilesLhs,
                                   tilesRhs, accIterArgs[i]);

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

// Implements tiled dot-product operation for a vector.contract operation or a
// sequence of vector.contracts inside the reduction loops.
//
// For example - for F32 type:
// ```
//   vector.transfer_read %arg0 {{.}*} : memref<16x32x4xi8>, vector<16x16x4xi8>
//   vector.transfer_read %arg1 {{.}*} : memref<16x32x4xi8>, vector<16x16x4xi8>
//   vector.contract <16x16x4xi8>, <16x16x4xi8> into <16x16xi32>
//   vector.transfer_write arg2 {{.}*} : vector<16x16xi32>, memref<32x32xi32>
// ```
// to
// ```
//   amx.tile_load %arg0 {{.}*} : memref<16x32x4xi8> into !amx.tile<16x64xi8>
//   amx.tile_load %arg1 {{.}*} : memref<16x32x4xi8> into !amx.tile<16x64xi8>
//   amx.tile_muli !amx.tile<16x64xi8> -> !amx.tile<16x16xi32>
//   amx.tile_store %arg2{{.}*} : memref<32x32xi32>, !amx.tile<16x16xi32>
// ```
struct VectorContractToPackedTypeTiledDotProduct
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    unsigned int blockingFactor =
        contractOp.getLhsType().getElementType().isBF16() ? 2 : 4;
    bool isVnni =
        isInVnniLayout(contractOp.getOperation(),
                       contractOp.getIndexingMapsArray(), blockingFactor);

    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isBF16() &&
        !lhsTy.getElementType().isSignlessInteger(8))
      return rewriter.notifyMatchFailure(
          contractOp, "Only BF16/Int8 lowering is supported.");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp, "Wrong accmulator type.");

    if ((lhsTy.getElementType().isBF16() && !accTy.getElementType().isF32()) ||
        (lhsTy.getElementType().isSignlessInteger(8) &&
         !accTy.getElementType().isSignlessInteger(32)))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only F32 for BF16 or Int32 for Int8 "
                                         "accumulation type is supported.");
    if (!isVnni)
      return rewriter.notifyMatchFailure(
          contractOp, "Only VNNI-packed inputs are supported.");

    Operation *accReadOp =
        traceToVectorReadLikeParentOperation(contractOp.getAcc());

    Operation *resultWriteOp =
        traceToVectorWriteLikeUserOperation(contractOp.getResult());

    if (!accReadOp || !resultWriteOp)
      return rewriter.notifyMatchFailure(
          contractOp, "The ACC operand of the vector.contract should be a "
                      "transfer_read or a load. And, the result should be "
                      "stored using transfer_write or store.");

    Type ipType;
    Type opType;

    if (lhsTy.getElementType().isBF16()) {
      ipType = rewriter.getBF16Type();
      opType = rewriter.getF32Type();
    }

    if (lhsTy.getElementType().isSignlessInteger(8)) {
      ipType = rewriter.getIntegerType(8);
      opType = rewriter.getIntegerType(32);
    }

    if (accReadOp->getBlock() == contractOp->getBlock() &&
        resultWriteOp->getBlock() != contractOp->getBlock())
      return rewriter.notifyMatchFailure(
          contractOp, "The accumulator store is in different block.");

    if (accReadOp->getBlock() != contractOp->getBlock() &&
        resultWriteOp->getBlock() == contractOp->getBlock())
      return rewriter.notifyMatchFailure(
          contractOp, "The accumulator read is in different block.");

    // Case 1: For just one VC rewrite. Where all accumulator read/write
    // within the same block.
    if (accReadOp->getBlock() == contractOp->getBlock() &&
        resultWriteOp->getBlock() == contractOp->getBlock()) {

      LogicalResult validate = validateContractOps(
          rewriter, contractOp, blockingFactor, Value(), Value(), false);

      if (failed(validate))
        return rewriter.notifyMatchFailure(
            contractOp, "The contract operation doesn't satisfy the operands "
                        "dimensions. M, N, and vnni dims are 16, 16, and 2/4. "
                        "The rest dims should be 1.");

      Location loc = contractOp.getLoc();

      auto srcIndxLhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                        contractOp.getLhs(), true);
      if (failed(srcIndxLhs))
        return rewriter.notifyMatchFailure(contractOp,
                                           "The LHS src is not a MemRef type.");
      auto [srcBuffLhs, indicesLhs] = *srcIndxLhs;

      auto srcIndxRhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                        contractOp.getRhs(), true);
      if (failed(srcIndxRhs))
        return rewriter.notifyMatchFailure(contractOp,
                                           "The RHS src is not a MemRef type.");
      auto [srcBuffRhs, indicesRhs] = *srcIndxRhs;

      auto srcIndxAcc = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                        contractOp.getAcc(), false);
      if (failed(srcIndxAcc))
        return rewriter.notifyMatchFailure(contractOp,
                                           "The ACC src is not a MemRef type.");
      auto [srcBuffAcc, indicesAcc] = *srcIndxAcc;

      // amx.tile_loads
      auto tileType = amx::TileType::get({16, (16 * blockingFactor)}, ipType);
      auto loadLhs = amx::TileLoadOp::create(rewriter, loc, tileType,
                                             srcBuffLhs, indicesLhs);
      auto loadRhs = amx::TileLoadOp::create(rewriter, loc, tileType,
                                             srcBuffRhs, indicesRhs);

      auto tileTypeAcc = amx::TileType::get({16, 16}, opType);
      auto loadAcc = amx::TileLoadOp::create(rewriter, loc, tileTypeAcc,
                                             srcBuffAcc, indicesAcc);

      // Tiled dot-product.
      Value dp;
      if (ipType.isBF16())
        dp = amx::TileMulFOp::create(rewriter, loc, tileTypeAcc, loadLhs,
                                     loadRhs, loadAcc);

      if (ipType.isSignlessInteger(8))
        dp = amx::TileMulIOp::create(rewriter, loc, tileTypeAcc, loadLhs,
                                     loadRhs, loadAcc);

      amx::TileStoreOp::create(rewriter, loc, srcBuffAcc, indicesAcc, dp);

      rewriter.eraseOp(resultWriteOp);
      return success();
    }

    // Case 2: The acc are passed as iter args through the reduction loop.
    // We support, reduction loop depth until 2. TODO: Support for n-depth
    // reduction loop.
    SmallVector<scf::ForOp> loopLists;
    Operation *current = contractOp;

    while (true) {
      Operation *parent = current->getParentOfType<scf::ForOp>();
      loopLists.push_back(dyn_cast<scf::ForOp>(parent));

      if (accReadOp->getBlock() == parent->getBlock()) {
        break;
      }

      current = parent;
    }

    if (loopLists.size() > 2 || loopLists.size() == 0)
      return rewriter.notifyMatchFailure(
          contractOp, "Rewrite is supported until reduction loop depth of 2.");

    auto srcIndxLhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                      contractOp.getLhs(), false);
    if (failed(srcIndxLhs))
      return rewriter.notifyMatchFailure(contractOp,
                                         "The LHS src is not a MemRef type.");
    auto [srcBuffLhs, indicesLhs] = *srcIndxLhs;

    auto srcIndxRhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                      contractOp.getRhs(), false);
    if (failed(srcIndxRhs))
      return rewriter.notifyMatchFailure(contractOp,
                                         "The RHS src is not a MemRef type.");
    auto [srcBuffRhs, indicesRhs] = *srcIndxRhs;

    Operation *vectorOpLhs;
    llvm::TypeSwitch<Operation *>(contractOp.getLhs().getDefiningOp())
        .Case<TransferReadOp, LoadOp>([&](auto readOp) {
          vectorOpLhs = readOp.getBase().getDefiningOp();
        });

    Operation *vectorOpRhs;
    llvm::TypeSwitch<Operation *>(contractOp.getRhs().getDefiningOp())
        .Case<TransferReadOp, LoadOp>([&](auto readOp) {
          vectorOpRhs = readOp.getBase().getDefiningOp();
        });

    // Retrive all the contaction operation within the loop.
    SmallVector<vector::ContractionOp> ops;
    for (mlir::Operation &op : loopLists[0].getBody()->getOperations()) {

      if (auto contract = llvm::dyn_cast<mlir::vector::ContractionOp>(op)) {

        LogicalResult validate = validateContractOps(
            rewriter, contract, blockingFactor, srcBuffLhs, srcBuffRhs, true);

        if (failed(validate))
          return rewriter.notifyMatchFailure(
              contractOp, "The associated contract operations doesn't satisfy "
                          "the re-write conditions either the dimensions are "
                          "wrong or MemRef source are different.");

        ops.push_back(contract);
      }
    }

    scf::ForOp outerLoop;
    scf::ForOp innerLoop;

    scf::ForOp newLoop;
    // Case 2a: Reduction loop depth is 2.
    if (loopLists.size() == 2) {
      outerLoop = loopLists[1];
      innerLoop = loopLists[0];

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
                      vectorOpLhs->getOperand(
                          getIndexPosition(contractOp.getLhs(), outerLoop) + 1),
                      ivOuterLoop);
                  mapping.map(
                      vectorOpLhs->getOperand(
                          getIndexPosition(contractOp.getLhs(), innerLoop) + 1),
                      ivNewInnerLoop);
                  auto lhsClone =
                      rewriterNewInnerLoop.clone(*vectorOpLhs, mapping);

                  IRMapping rhsMapping;
                  rhsMapping.map(
                      vectorOpRhs->getOperand(
                          getIndexPosition(contractOp.getRhs(), outerLoop) + 1),
                      ivOuterLoop);
                  rhsMapping.map(
                      vectorOpRhs->getOperand(
                          getIndexPosition(contractOp.getRhs(), innerLoop) + 1),
                      ivNewInnerLoop);
                  auto rhsClone =
                      rewriterNewInnerLoop.clone(*vectorOpRhs, rhsMapping);

                  SmallVector<Value> accumulators = createTiledDp(
                      rewriter, locNewInnerLoop, ops, lhsClone->getResult(0),
                      rhsClone->getResult(0), ipType, opType,
                      iterArgsNewInnerLoop, blockingFactor);

                  scf::YieldOp::create(rewriterNewInnerLoop, locNewInnerLoop,
                                       accumulators);
                });

            scf::YieldOp::create(rewriterOuterLoop, locOuterLoop,
                                 newInnerLoop.getResults());
          });
    }

    // Case 2a: Reduction loop depth is 1.
    if (loopLists.size() == 1) {
      outerLoop = loopLists[0];

      SmallVector<Value> loopItrArgs = createTileZeros(
          rewriter, outerLoop.getLoc(), opType, outerLoop, ops.size());
      newLoop = scf::ForOp::create(
          rewriter, outerLoop.getLoc(), outerLoop.getLowerBound(),
          outerLoop.getUpperBound(), outerLoop.getStep(), loopItrArgs,
          [&](OpBuilder &rewriterOuterLoop, Location locOuterLoop,
              Value ivOuterLoop, ValueRange iterArgsOuterLoop) {
            IRMapping mapping;
            mapping.map(
                vectorOpLhs->getOperand(
                    getIndexPosition(contractOp.getLhs(), outerLoop) + 1),
                ivOuterLoop);

            auto lhsClone = rewriterOuterLoop.clone(*vectorOpLhs, mapping);

            IRMapping rhsMapping;
            rhsMapping.map(
                vectorOpRhs->getOperand(
                    getIndexPosition(contractOp.getRhs(), outerLoop) + 1),
                ivOuterLoop);

            auto rhsClone = rewriterOuterLoop.clone(*vectorOpRhs, rhsMapping);

            SmallVector<Value> accumulators = createTiledDp(
                rewriter, locOuterLoop, ops, lhsClone->getResult(0),
                rhsClone->getResult(0), ipType, opType, iterArgsOuterLoop,
                blockingFactor);

            scf::YieldOp::create(rewriterOuterLoop, locOuterLoop, accumulators);
          });
    }

    // post processing after the loop creation.
    // Copy the amx tile accumulation results to a MemRef buffer, add the
    // initial accumulation value, and store back to the C-Matrix
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
            auto resultAcc = vector::LoadOp::create(
                rewriter, loc, VectorType::get(16, opType), bBuffer,
                ValueRange{iv, c0});

            Operation *accReadOp =
                traceToVectorReadLikeParentOperation(ops[i].getAcc());

            Value srcBuffAcc;
            SmallVector<Value> indicesAcc;

            llvm::TypeSwitch<Operation *>(accReadOp)
                .Case<TransferReadOp, LoadOp>([&](auto readOp) {
                  srcBuffAcc = readOp.getOperand(0);

                  auto indices = readOp.getIndices();
                  indicesAcc.reserve(indices.size());

                  llvm::transform(
                      indices, std::back_inserter(indicesAcc),
                      [&](OpFoldResult ofr) {
                        return mlir::getValueOrCreateConstantIndexOp(rewriter,
                                                                     loc, ofr);
                      });
                });

            Value sum = arith::AddIOp::create(builder, loc, iv, indicesAcc[0]);
            indicesAcc[0] = sum;

            auto acc = vector::LoadOp::create(rewriter, loc,
                                              VectorType::get(16, opType),
                                              srcBuffAcc, indicesAcc);
            Value addition;
            if (ipType.isBF16())
              addition = arith::AddFOp::create(rewriter, loc, resultAcc, acc);

            if (ipType.isSignlessInteger(8))
              addition = arith::AddIOp::create(rewriter, loc, resultAcc, acc);

            vector::StoreOp::create(builder, loc, addition, srcBuffAcc,
                                    indicesAcc);

            scf::YieldOp::create(builder, outerLoop.getLoc());
          });

      rewriter.eraseOp(resultWriteOp);
    }

    return success();
  }
};

} // namespace

void x86::populateVectorContractToPackedTypeTiledDotProductPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractToPackedTypeTiledDotProduct>(
      patterns.getContext());
}
