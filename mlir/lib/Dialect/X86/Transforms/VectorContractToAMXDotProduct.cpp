//===- VectorContractToAMXDotProduct.cpp ----------------------===//
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
                                       bool rhs, unsigned int offset,
                                       bool isVnni) {

  auto srcIndx = getSrcIndxValue(rewriter, loc, operand, false);
  auto [srcBuff, indices] = *srcIndx;
  if (isVnni) {
    indices.pop_back();
  }

  if (rhs && isVnni) {
    auto cOffset = arith::ConstantIndexOp::create(rewriter, loc, offset);
    indices[indices.size() - 1] = arith::MulIOp::create(
        rewriter, loc, indices[indices.size() - 1], cOffset);
  }

  amx::TileType tileType = amx::TileType::get({16, (16 * offset)}, ipType);
  auto load = amx::TileLoadOp::create(rewriter, loc, tileType, mat, indices);
  return load;
}

static void performShuffle(OpBuilder &rewriter, Location loc, Value matB,
                           Type ipType, unsigned int offset, Value bBuffer,
                           Value allocStore) {

  auto subview = matB.getDefiningOp<mlir::memref::SubViewOp>();
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> vals(subview.getOffsets().size(), c0);

  // Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value c16 = arith::ConstantIndexOp::create(rewriter, loc, 16);

  Value cStep = arith::ConstantIndexOp::create(rewriter, loc, offset);
  Value cBound = arith::ConstantIndexOp::create(rewriter, loc, (16 * offset));
  Value nLoadIndx = arith::ConstantIndexOp::create(rewriter, loc, (offset / 2));

  scf::ForOp::create(
      rewriter, loc, c0, cBound, cStep, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location loc, Value iv,
          ValueRange iterArgs) {
        Value i1_load = arith::AddIOp::create(rewriter, loc, nLoadIndx, iv);

        vals[vals.size() - 2] = iv;
        ValueRange range1(vals);
        auto vec1 = vector::LoadOp::create(
            rewriter, loc, VectorType::get((16 * offset), ipType), matB,
            range1);

        vals[vals.size() - 2] = i1_load;
        ValueRange range2(vals);
        auto vec2 = vector::LoadOp::create(
            rewriter, loc, VectorType::get((16 * offset), ipType), matB,
            range2);

        vector::ShuffleOp shuffle1;
        vector::ShuffleOp shuffle2;

        if (offset == 2) {

          shuffle1 = vector::ShuffleOp::create(
              rewriter, loc, VectorType::get({(16 * offset)}, ipType), vec1,
              vec2,
              ArrayRef<int64_t>{0,  32, 1,  33, 2,  34, 3,  35, 8,  40, 9,
                                41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50,
                                19, 51, 24, 56, 25, 57, 26, 58, 27, 59});

          shuffle2 = vector::ShuffleOp::create(
              rewriter, loc, VectorType::get({(16 * offset)}, ipType), vec1,
              vec2,
              ArrayRef<int64_t>{4,  36, 5,  37, 6,  38, 7,  39, 12, 44, 13,
                                45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54,
                                23, 55, 28, 60, 29, 61, 30, 62, 31, 63});
        }

        if (offset == 4) {

          shuffle1 = vector::ShuffleOp::create(
              rewriter, loc, VectorType::get({(16 * offset)}, ipType), vec1,
              vec2,
              ArrayRef<int64_t>{
                  0,   32,  64, 96,  1,   33,  65,  97,  2,   34,  66,  98, 3,
                  35,  67,  99, 4,   36,  68,  100, 5,   37,  69,  101, 6,  38,
                  70,  102, 7,  39,  71,  103, 8,   40,  72,  104, 9,   41, 73,
                  105, 10,  42, 74,  106, 11,  43,  75,  107, 12,  44,  76, 108,
                  13,  45,  77, 109, 14,  46,  78,  110, 15,  47,  79,  111});

          shuffle2 = vector::ShuffleOp::create(
              rewriter, loc, VectorType::get({(16 * offset)}, ipType), vec1,
              vec2,
              ArrayRef<int64_t>{
                  16, 48,  80, 112, 17, 49,  81, 113, 18, 50,  82, 114, 19, 51,
                  83, 115, 20, 52,  84, 116, 21, 53,  85, 117, 22, 54,  86, 118,
                  23, 55,  87, 119, 24, 56,  88, 120, 25, 57,  89, 121, 26, 58,
                  90, 122, 27, 59,  91, 123, 28, 60,  92, 124, 29, 61,  93, 125,
                  30, 62,  94, 126, 31, 63,  95, 127});
        }
        Value j_pos = arith::DivUIOp::create(rewriter, loc, iv, cStep);
        Value j16_pos = arith::AddIOp::create(rewriter, loc, c16, j_pos);

        vector::StoreOp::create(rewriter, loc, shuffle1, bBuffer,
                                ValueRange{allocStore, j_pos, c0});
        vector::StoreOp::create(rewriter, loc, shuffle2, bBuffer,
                                ValueRange{allocStore, j16_pos, c0});

        scf::YieldOp::create(nestedBuilder, loc);
      });
}

static llvm::DenseMap<Operation *, amx::TileLoadOp>
packInputs(OpBuilder &rewriter, Location loc,
           SmallVector<vector::ContractionOp> ops, Value matB, Type ipType,
           unsigned int offset, Value bBuffer, bool pack, Value allocStore,
           Value addIdx) {
  llvm::DenseMap<Operation *, amx::TileLoadOp> readsToTileLoads;

  for (size_t j = 0; j < ops.size(); j++) {
    for (size_t i = 0; i < ops.size(); i++) {

      if (i != j && validatePairVectorContract(ops[j], ops[i], true, 16)) {

        Operation *readOpRhs = ops[j].getRhs().getDefiningOp();
        auto itRhs = readsToTileLoads.find(readOpRhs);
        if (itRhs != readsToTileLoads.end()) {
          continue;
        }

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c16 = arith::ConstantIndexOp::create(rewriter, loc, 16);

        if (pack) {
          performShuffle(rewriter, loc, matB, ipType, offset, bBuffer,
                         allocStore);
        }

        amx::TileType tileType =
            amx::TileType::get({16, (16 * offset)}, ipType);
        auto load = amx::TileLoadOp::create(rewriter, loc, tileType, bBuffer,
                                            ValueRange{addIdx, c0, c0});

        auto load1 = amx::TileLoadOp::create(rewriter, loc, tileType, bBuffer,
                                             ValueRange{addIdx, c16, c0});

        readsToTileLoads.try_emplace(readOpRhs, load);

        readsToTileLoads.try_emplace(ops[i].getRhs().getDefiningOp(), load1);
      }
    }
  }

  return readsToTileLoads;
}

// Creates tiled amx dot-products.
static SmallVector<Value> createTiledDp(OpBuilder &rewriter, Location loc,
                                        SmallVector<vector::ContractionOp> ops,
                                        Value matA, Value matB, Type ipType,
                                        Type opType, ValueRange accIterArgs,
                                        unsigned int offset, bool isVnni,
                                        Value bBuffer, bool pack,
                                        Value allocStore, Value addIdx) {

  if (isVnni) {
    matA = collapseInnerDims(rewriter, loc, matA);
    matB = collapseInnerDims(rewriter, loc, matB);
  }

  SmallVector<Value> accumulators;
  // Stores the amx.tile_load operation vs it's equivalent vector tranfer_read
  // or load operations.
  llvm::DenseMap<Operation *, amx::TileLoadOp> readsToTileLoads;

  // function call to make the flat.
  if (!isVnni) {
    readsToTileLoads = packInputs(rewriter, loc, ops, matB, ipType, offset,
                                  bBuffer, pack, allocStore, addIdx);
  }

  // Iterate over the contraction operations and compute the tiled dot-product.
  for (size_t i = 0; i < ops.size(); i++) {

    Operation *readOpLhs = ops[i].getLhs().getDefiningOp();
    amx::TileLoadOp tilesLhs;
    auto itLhs = readsToTileLoads.find(readOpLhs);
    if (itLhs != readsToTileLoads.end()) {
      tilesLhs = itLhs->second;
    } else {
      tilesLhs = createTileLoads(rewriter, loc, ops[i].getLhs(), matA, ipType,
                                 false, offset, isVnni);
      readsToTileLoads.try_emplace(readOpLhs, tilesLhs);
    }

    Operation *readOpRhs = ops[i].getRhs().getDefiningOp();
    amx::TileLoadOp tilesRhs;
    auto itRhs = readsToTileLoads.find(readOpRhs);
    if (itRhs != readsToTileLoads.end()) {
      tilesRhs = itRhs->second;
    } else {
      tilesRhs = createTileLoads(rewriter, loc, ops[i].getRhs(), matB, ipType,
                                 true, offset, isVnni);
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

static Value bufferIndxToStore(OpBuilder &rewriter, Location loc, Value iv_K,
                               Value iv_red, bool oddDimK, bool nDimK,
                               bool pack, unsigned int blockingFactor) {

  Value c2 = arith::ConstantIndexOp::create(rewriter, loc, 2);

  Value nLoadIndx =
      arith::ConstantIndexOp::create(rewriter, loc, (16 * blockingFactor));

  Value quotient_K = arith::DivUIOp::create(rewriter, loc, iv_K, nLoadIndx);
  Value rem_K = arith::RemUIOp::create(rewriter, loc, rewriter.getIndexType(),
                                       quotient_K, c2);

  if (!nDimK && !pack) {
    rem_K = arith::RemUIOp::create(rewriter, loc, rewriter.getIndexType(),
                                   iv_red, c2);
  }

  // if K quotient is odd. Then, BR loop iv is taken
  // into consideration
  if (oddDimK) {
    auto rem_BR = arith::RemUIOp::create(rewriter, loc, rewriter.getIndexType(),
                                         iv_red, c2);
    auto remAdd = arith::AddIOp::create(rewriter, loc, rewriter.getIndexType(),
                                        rem_K, rem_BR);
    rem_K = arith::RemUIOp::create(rewriter, loc, rewriter.getIndexType(),
                                   remAdd, c2);
  }

  return rem_K;
}

static scf::ForOp
createLoops(OpBuilder &rewriter, Location loc, Value lowerBound,
            Value upperBound, Value step, SmallVector<Value> loopItrArgs,
            Type ipType, Type opType, unsigned int blockingFactor, bool isVnni,
            Operation *vectorOpLhs, Operation *vectorOpRhs,
            vector::ContractionOp contractOp, scf::ForOp outerLoop,
            scf::ForOp innerLoop, SmallVector<vector::ContractionOp> ops,
            Value ivOuterLoop, Value bBuffer, bool pack,
            arith::ConstantIndexOp innerLoopIndex, bool nDimK, bool oddDimK) {

  auto newLoop1 = scf::ForOp::create(
      rewriter, loc, lowerBound, upperBound, step, loopItrArgs,
      [&](OpBuilder &rewriterNewInnerLoop, Location locNewInnerLoop,
          Value ivNewInnerLoop, ValueRange iterArgsNewInnerLoop) {
        IRMapping mapping;
        if (outerLoop) {
          mapping.map(vectorOpLhs->getOperand(
                          getIndexPosition(contractOp.getLhs(), outerLoop) + 1),
                      ivOuterLoop);
        }
        mapping.map(vectorOpLhs->getOperand(
                        getIndexPosition(contractOp.getLhs(), innerLoop) + 1),
                    ivNewInnerLoop);
        auto lhsClone = rewriterNewInnerLoop.clone(*vectorOpLhs, mapping);

        Value c0 = arith::ConstantIndexOp::create(rewriter, locNewInnerLoop, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, locNewInnerLoop, 1);
        Value c2 = arith::ConstantIndexOp::create(rewriter, locNewInnerLoop, 2);
        Value allocStore = c0;
        Value allocGet = c0;

        if (!isVnni) {
          if (outerLoop) {
            if (innerLoopIndex.value() == 0) {
              if (pack) {
                ivNewInnerLoop = c0;
                ivOuterLoop = arith::AddIOp::create(rewriter, locNewInnerLoop,
                                                    c1, ivOuterLoop);

                if (!nDimK || oddDimK) {
                  allocStore = arith::RemUIOp::create(rewriter, locNewInnerLoop,
                                                      rewriter.getIndexType(),
                                                      ivOuterLoop, c2);
                }

                Value addIdx =
                    arith::AddIOp::create(rewriter, loc, allocStore, c1);
                allocGet = arith::RemUIOp::create(
                    rewriter, loc, rewriter.getIndexType(), addIdx, c2);
              }

            } else {
              Value nLoadIndx = arith::ConstantIndexOp::create(
                  rewriter, locNewInnerLoop, (16 * blockingFactor));
              ivNewInnerLoop = arith::AddIOp::create(rewriter, locNewInnerLoop,
                                                     nLoadIndx, ivNewInnerLoop);
              allocStore =
                  bufferIndxToStore(rewriter, loc, ivNewInnerLoop, ivOuterLoop,
                                    oddDimK, nDimK, pack, blockingFactor);
              Value addIdx =
                  arith::AddIOp::create(rewriter, loc, allocStore, c1);
              allocGet = arith::RemUIOp::create(
                  rewriter, loc, rewriter.getIndexType(), addIdx, c2);
            }
          } else {
            if (pack) {
              Value nLoadIndx = arith::ConstantIndexOp::create(
                  rewriter, locNewInnerLoop, (16 * blockingFactor));
              ivNewInnerLoop = arith::AddIOp::create(rewriter, locNewInnerLoop,
                                                     nLoadIndx, ivNewInnerLoop);
              Value quotient_K = arith::DivUIOp::create(
                  rewriter, loc, ivNewInnerLoop, nLoadIndx);
              allocStore = arith::RemUIOp::create(
                  rewriter, loc, rewriter.getIndexType(), quotient_K, c2);

              Value addIdx =
                  arith::AddIOp::create(rewriter, loc, allocStore, c1);
              allocGet = arith::RemUIOp::create(
                  rewriter, loc, rewriter.getIndexType(), addIdx, c2);
            }
          }
        }

        IRMapping rhsMapping;
        if (outerLoop) {
          rhsMapping.map(
              vectorOpRhs->getOperand(
                  getIndexPosition(contractOp.getRhs(), outerLoop) + 1),
              ivOuterLoop);
        }
        rhsMapping.map(
            vectorOpRhs->getOperand(
                getIndexPosition(contractOp.getRhs(), innerLoop) + 1),
            ivNewInnerLoop);
        auto rhsClone = rewriterNewInnerLoop.clone(*vectorOpRhs, rhsMapping);

        Value matB = rhsClone->getResult(0);

        if (!isVnni) {
          if (outerLoop) {
            if (!pack) {
              Value nLoadIndx = arith::ConstantIndexOp::create(
                  rewriter, locNewInnerLoop, (16 * blockingFactor));
              matB = Value();
              allocGet = c0;
              allocGet =
                  bufferIndxToStore(rewriter, loc, nLoadIndx, ivOuterLoop,
                                    oddDimK, nDimK, pack, blockingFactor);
            }
          } else {
            if (!pack) {
              Value nLoadIndx = arith::ConstantIndexOp::create(
                  rewriter, locNewInnerLoop, (16 * blockingFactor));
              matB = Value();
              Value quotient_K = arith::DivUIOp::create(
                  rewriter, loc, ivNewInnerLoop, nLoadIndx);
              allocGet = arith::RemUIOp::create(
                  rewriter, loc, rewriter.getIndexType(), quotient_K, c2);
            }
          }
        }

        SmallVector<Value> accumulators = createTiledDp(
            rewriter, locNewInnerLoop, ops, lhsClone->getResult(0), matB,
            ipType, opType, iterArgsNewInnerLoop, blockingFactor, isVnni,
            bBuffer, pack, allocStore, allocGet);

        scf::YieldOp::create(rewriterNewInnerLoop, locNewInnerLoop,
                             accumulators);
      });

  return newLoop1;
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
struct VectorContractToAMXDotProduct
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

    unsigned int dimValue = blockingFactor;
    if (!isVnni)
      dimValue = 16 * blockingFactor;

    // Case 1: For just one VC rewrite. Where all accumulator read/write
    // within the same block.
    if (accReadOp->getBlock() == contractOp->getBlock() &&
        resultWriteOp->getBlock() == contractOp->getBlock()) {

      bool collapse = false;
      if (isVnni)
        collapse = true;

      LogicalResult validate = validateContractOps(
          rewriter, contractOp, dimValue, Value(), Value(), false);

      if (failed(validate))
        return rewriter.notifyMatchFailure(
            contractOp, "The contract operation doesn't satisfy the operands "
                        "dimensions. M, N, and vnni dims are 16, 16, and 2/4. "
                        "The rest dims should be 1.");

      Location loc = contractOp.getLoc();

      auto srcIndxLhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                        contractOp.getLhs(), collapse);
      if (failed(srcIndxLhs))
        return rewriter.notifyMatchFailure(contractOp,
                                           "The LHS src is not a MemRef type.");
      auto [srcBuffLhs, indicesLhs] = *srcIndxLhs;

      auto srcIndxRhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                        contractOp.getRhs(), collapse);
      if (failed(srcIndxRhs))
        return rewriter.notifyMatchFailure(contractOp,
                                           "The RHS src is not a MemRef type.");
      auto rhsSrc = *srcIndxRhs;
      auto srcBuffRhs = rhsSrc.first;
      auto indicesRhs = rhsSrc.second;

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

      // Create the subview and then load.
      //
      amx::TileLoadOp loadRhs;
      if (!isVnni) {
        VectorType vecTy;
        SmallVector<OpFoldResult> indexVals;
        llvm::TypeSwitch<Operation *>(contractOp.getRhs().getDefiningOp())
            .Case<TransferReadOp, LoadOp>([&](auto readOp) {
              indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                                    readOp.getIndices().end());
              vecTy = readOp.getType();
            });
        auto one = rewriter.getIndexAttr(1);
        SmallVector<OpFoldResult> strides(indexVals.size(), one);
        SmallVector<OpFoldResult> sizes = getAsIndexOpFoldResult(
            contractOp.getRhs().getDefiningOp()->getContext(),
            vecTy.getShape());
        auto subview = memref::SubViewOp::create(rewriter, loc, srcBuffRhs,
                                                 indexVals, sizes, strides);
        auto bufferType = MemRefType::get({16, (16 * blockingFactor)}, ipType);
        auto bBuffer = memref::AllocaOp::create(rewriter, loc, bufferType);

        // create a loop that swaps them.

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);

        Value step =
            arith::ConstantIndexOp::create(rewriter, loc, blockingFactor);
        Value uBound = arith::ConstantIndexOp::create(rewriter, loc,
                                                      (blockingFactor * 16));
        Value nextLoadIndx =
            arith::ConstantIndexOp::create(rewriter, loc, (blockingFactor / 2));
        Value nextStoreIndx = arith::ConstantIndexOp::create(
            rewriter, loc, 16 * (blockingFactor / 2));

        scf::ForOp::create(
            rewriter, loc, c0, uBound, step, ValueRange{},
            [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                ValueRange iterArgs) {
              Value i1_load =
                  arith::AddIOp::create(rewriter, loc, nextLoadIndx, iv);

              indicesRhs[indicesRhs.size() - 2] = iv;
              ValueRange range1(indicesRhs);
              auto vec1 = vector::LoadOp::create(
                  rewriter, loc,
                  VectorType::get(16 * (blockingFactor / 2), ipType), subview,
                  range1);

              indicesRhs[indicesRhs.size() - 2] = i1_load;
              ValueRange range2(indicesRhs);
              auto vec2 = vector::LoadOp::create(
                  rewriter, loc,
                  VectorType::get(16 * (blockingFactor / 2), ipType), subview,
                  range2);

              vector::ShuffleOp shuffle1;
              vector::ShuffleOp shuffle2;

              if (blockingFactor == 2) {

                shuffle1 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get({16}, ipType), vec1, vec2,
                    ArrayRef<int64_t>{0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21,
                                      6, 22, 7, 23});

                shuffle2 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get({16}, ipType), vec1, vec2,
                    ArrayRef<int64_t>{8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13,
                                      29, 14, 30, 15, 31});
              }

              if (blockingFactor == 4) {
                shuffle1 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get({32}, ipType), vec1, vec2,
                    ArrayRef<int64_t>{0, 16, 32, 48, 1, 17, 33, 49,
                                      2, 18, 34, 50, 3, 19, 35, 51,
                                      4, 20, 36, 52, 5, 21, 37, 53,
                                      6, 22, 38, 54, 7, 23, 39, 55});

                shuffle2 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get({32}, ipType), vec1, vec2,
                    ArrayRef<int64_t>{8,  24, 40, 56, 9,  25, 41, 57,
                                      10, 26, 42, 58, 11, 27, 43, 59,
                                      12, 28, 44, 60, 13, 29, 45, 61,
                                      14, 30, 46, 62, 15, 31, 47, 63});
              }

              auto rem = arith::RemUIOp::create(
                  rewriter, loc, rewriter.getIndexType(), iv, step);

              vector::StoreOp::create(rewriter, loc, shuffle1, bBuffer,
                                      ValueRange{rem, c0});
              vector::StoreOp::create(rewriter, loc, shuffle2, bBuffer,
                                      ValueRange{rem, nextStoreIndx});

              scf::YieldOp::create(nestedBuilder, loc);
            });
        loadRhs = amx::TileLoadOp::create(rewriter, loc, tileType, bBuffer,
                                          ValueRange{c0, c0});
      } else {

        loadRhs = amx::TileLoadOp::create(rewriter, loc, tileType, srcBuffRhs,
                                          indicesRhs);
      }

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
            rewriter, contract, dimValue, srcBuffLhs, srcBuffRhs, true);

        if (failed(validate))
          return rewriter.notifyMatchFailure(
              contractOp, "The associated contract operations doesn't satisfy "
                          "the re-write conditions either the dimensions are "
                          "wrong or MemRef source are different.");

        ops.push_back(contract);
      }
    }

    scf::ForOp innerLoop;
    scf::ForOp outerLoop;

    scf::ForOp newLoop;
    // Case 2a: Reduction loop depth is 2.
    if (loopLists.size() == 2) {
      outerLoop = loopLists[1];
      innerLoop = loopLists[0];

      SmallVector<Value> loopItrArgs = createTileZeros(
          rewriter, outerLoop.getLoc(), opType, outerLoop, ops.size());

      if (isVnni) {
        newLoop = scf::ForOp::create(
            rewriter, outerLoop.getLoc(), outerLoop.getLowerBound(),
            outerLoop.getUpperBound(), outerLoop.getStep(), loopItrArgs,
            [&](OpBuilder &rewriterOuterLoop, Location locOuterLoop,
                Value ivOuterLoop, ValueRange iterArgsOuterLoop) {
              auto newInnerLoop = createLoops(
                  rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
                  innerLoop.getUpperBound(), innerLoop.getStep(),
                  iterArgsOuterLoop, ipType, opType, blockingFactor, isVnni,
                  vectorOpLhs, vectorOpRhs, contractOp, outerLoop, innerLoop,
                  ops, ivOuterLoop, nullptr, true, nullptr, false, false);

              scf::YieldOp::create(rewriterOuterLoop, locOuterLoop,
                                   newInnerLoop.getResults());
            });

      } else {

        bool nDimK = false;
        bool oddDimK = false;

        int64_t ubVal = 16 * blockingFactor;
        mlir::Value ub = innerLoop.getUpperBound();
        if (auto constOp = ub.getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto intAttr =
                  llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            ubVal = intAttr.getInt();
          }
        }

        nDimK = ubVal > 16 * blockingFactor;
        oddDimK = (((ubVal / (16 * blockingFactor)) % 2) == 1) && nDimK;

        rewriter.setInsertionPoint(outerLoop);

        auto c0 =
            arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 0);

        auto c1 =
            arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 1);

        auto spillLoopBound = arith::ConstantIndexOp::create(
            rewriter, outerLoop.getLoc(), 16 * blockingFactor);
        Value subBRLoop = arith::SubIOp::create(rewriter, outerLoop.getLoc(),
                                                outerLoop.getUpperBound(), c1);
        Value subKloop =
            arith::SubIOp::create(rewriter, innerLoop.getLoc(),
                                  innerLoop.getUpperBound(), spillLoopBound);
        auto bufferType =
            MemRefType::get({2, 32, (blockingFactor * 16)}, ipType);
        auto bBuffer =
            memref::AllocaOp::create(rewriter, outerLoop.getLoc(), bufferType);

        // First Shuffling outside the reduction loops
        IRMapping rhsMapping;
        rhsMapping.map(
            vectorOpRhs->getOperand(
                getIndexPosition(contractOp.getRhs(), outerLoop) + 1),
            c0);
        rhsMapping.map(
            vectorOpRhs->getOperand(
                getIndexPosition(contractOp.getRhs(), innerLoop) + 1),
            c0);
        auto rhsClone = rewriter.clone(*vectorOpRhs, rhsMapping);

        performShuffle(rewriter, outerLoop.getLoc(), rhsClone->getResult(0),
                       ipType, blockingFactor, bBuffer, c0);

        // First Set of Loops
        auto newLoop1 = scf::ForOp::create(
            rewriter, outerLoop.getLoc(), outerLoop.getLowerBound(), subBRLoop,
            outerLoop.getStep(), loopItrArgs,
            [&](OpBuilder &rewriterOuterLoop, Location locOuterLoop,
                Value ivOuterLoop, ValueRange iterArgsOuterLoop) {
              auto newInnerLoop1 = createLoops(
                  rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
                  subKloop, innerLoop.getStep(), iterArgsOuterLoop, ipType,
                  opType, blockingFactor, isVnni, vectorOpLhs, vectorOpRhs,
                  contractOp, outerLoop, innerLoop, ops, ivOuterLoop, bBuffer,
                  true, spillLoopBound, nDimK, oddDimK);

              auto newInnerLoop =
                  createLoops(rewriter, innerLoop.getLoc(), subKloop,
                              innerLoop.getUpperBound(), innerLoop.getStep(),
                              newInnerLoop1.getResults(), ipType, opType,
                              blockingFactor, isVnni, vectorOpLhs, vectorOpRhs,
                              contractOp, outerLoop, innerLoop, ops,
                              ivOuterLoop, bBuffer, true, c0, nDimK, oddDimK);

              scf::YieldOp::create(rewriterOuterLoop, locOuterLoop,
                                   newInnerLoop.getResults());
            });

        // Last set of Loops
        newLoop = scf::ForOp::create(
            rewriter, outerLoop.getLoc(), subBRLoop, outerLoop.getUpperBound(),
            outerLoop.getStep(), newLoop1.getResults(),
            [&](OpBuilder &rewriterOuterLoop, Location locOuterLoop,
                Value ivOuterLoop, ValueRange iterArgsOuterLoop) {
              auto newInnerLoop1 = createLoops(
                  rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
                  subKloop, innerLoop.getStep(), iterArgsOuterLoop, ipType,
                  opType, blockingFactor, isVnni, vectorOpLhs, vectorOpRhs,
                  contractOp, outerLoop, innerLoop, ops, ivOuterLoop, bBuffer,
                  true, spillLoopBound, nDimK, oddDimK);

              auto newInnerLoop =
                  createLoops(rewriter, innerLoop.getLoc(), subKloop,
                              innerLoop.getUpperBound(), innerLoop.getStep(),
                              newInnerLoop1.getResults(), ipType, opType,
                              blockingFactor, isVnni, vectorOpLhs, vectorOpRhs,
                              contractOp, outerLoop, innerLoop, ops,
                              ivOuterLoop, bBuffer, false, c0, nDimK, oddDimK);

              scf::YieldOp::create(rewriterOuterLoop, locOuterLoop,
                                   newInnerLoop.getResults());
            });
      }
    }

    if (loopLists.size() == 1) {
      outerLoop = loopLists[0];
      innerLoop = loopLists[0];

      SmallVector<Value> loopItrArgs = createTileZeros(
          rewriter, innerLoop.getLoc(), opType, innerLoop, ops.size());

      if (isVnni) {

        newLoop = createLoops(
            rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
            innerLoop.getUpperBound(), innerLoop.getStep(), loopItrArgs, ipType,
            opType, blockingFactor, isVnni, vectorOpLhs, vectorOpRhs,
            contractOp, nullptr, innerLoop, ops, nullptr, nullptr, true,
            nullptr, false, false);

      } else {
        bool nDimK = false;
        bool oddDimK = false;

        int64_t ubVal = 16 * blockingFactor;
        mlir::Value ub = innerLoop.getUpperBound();
        if (auto constOp = ub.getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto intAttr =
                  llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            ubVal = intAttr.getInt();
          }
        }

        nDimK = ubVal > 16 * blockingFactor;
        oddDimK = (((ubVal / (16 * blockingFactor)) % 2) == 1) && nDimK;
        rewriter.setInsertionPoint(innerLoop);
        auto c0 =
            arith::ConstantIndexOp::create(rewriter, innerLoop.getLoc(), 0);
        auto spillLoopBound = arith::ConstantIndexOp::create(
            rewriter, innerLoop.getLoc(), 16 * blockingFactor);

        Value subKloop =
            arith::SubIOp::create(rewriter, innerLoop.getLoc(),
                                  innerLoop.getUpperBound(), spillLoopBound);

        auto bufferType =
            MemRefType::get({2, 32, (blockingFactor * 16)}, ipType);
        auto bBuffer =
            memref::AllocaOp::create(rewriter, innerLoop.getLoc(), bufferType);

        // First Shuffling outside the reduction loops
        IRMapping rhsMapping;
        rhsMapping.map(
            vectorOpRhs->getOperand(
                getIndexPosition(contractOp.getRhs(), innerLoop) + 1),
            c0);
        auto rhsClone = rewriter.clone(*vectorOpRhs, rhsMapping);

        performShuffle(rewriter, innerLoop.getLoc(), rhsClone->getResult(0),
                       ipType, blockingFactor, bBuffer, c0);

        auto newLoop1 = createLoops(
            rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(), subKloop,
            innerLoop.getStep(), loopItrArgs, ipType, opType, blockingFactor,
            isVnni, vectorOpLhs, vectorOpRhs, contractOp, nullptr, innerLoop,
            ops, nullptr, bBuffer, true, spillLoopBound, nDimK, oddDimK);

        newLoop = createLoops(rewriter, innerLoop.getLoc(), subKloop,
                              innerLoop.getUpperBound(), innerLoop.getStep(),
                              newLoop1.getResults(), ipType, opType,
                              blockingFactor, isVnni, vectorOpLhs, vectorOpRhs,
                              contractOp, nullptr, innerLoop, ops, nullptr,
                              bBuffer, false, c0, nDimK, oddDimK);
      }
    }

    // Copy the amx tile accumulation results to a MemRef buffer, add the
    // initial accumulation value, and store back to the C-Matrix

    if (!isVnni) {
      Location loc = outerLoop.getLoc();
      SmallVector<Value> dps = newLoop.getResults();
      auto bufferType = MemRefType::get({32, 32}, opType);
      auto bBuffer =
          memref::AllocaOp::create(rewriter, outerLoop.getLoc(), bufferType);
      for (int i = 0, k = 0; i < 32; i = i + 16) {
        for (int j = 0; j < 32; j = j + 16) {
          Value indexOp_i =
              arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), i);
          Value indexOp_j =
              arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), j);
          amx::TileStoreOp::create(rewriter, outerLoop.getLoc(), bBuffer,
                                   ValueRange{indexOp_i, indexOp_j}, dps[k]);
          k++;
        }
      }
      auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
      auto c16 = arith::ConstantIndexOp::create(rewriter, loc, 16);
      auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto mBound = arith::ConstantIndexOp::create(rewriter, loc, 32);

      scf::ForOp::create(
          rewriter, loc, c0, mBound, one, ValueRange{},
          [&](OpBuilder &nestedBuilder, Location loc, Value iv,
              ValueRange iterArgs) {
            auto row = vector::LoadOp::create(rewriter, loc,
                                              VectorType::get(16, opType),
                                              bBuffer, ValueRange{iv, c0});

            auto row2 = vector::LoadOp::create(rewriter, loc,
                                               VectorType::get(16, opType),
                                               bBuffer, ValueRange{iv, c16});

            auto shuffle1 = vector::ShuffleOp::create(
                rewriter, loc, VectorType::get(16, opType), row, row2,
                ArrayRef<int64_t>{0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20,
                                  21, 22, 23});

            auto shuffle2 = vector::ShuffleOp::create(
                rewriter, loc, VectorType::get(16, opType), row, row2,
                ArrayRef<int64_t>{8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15,
                                  28, 29, 30, 31});

            Operation *accReadOp =
                traceToVectorReadLikeParentOperation(contractOp.getAcc());

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

            indicesAcc[indicesAcc.size() - 2] = iv;
            indicesAcc[indicesAcc.size() - 1] = c0;

            Value valueCRow1 = vector::LoadOp::create(
                rewriter, loc, VectorType::get(16, opType), srcBuffAcc,
                indicesAcc);
            indicesAcc[indicesAcc.size() - 1] = c16;

            Value valueCRow2 = vector::LoadOp::create(
                rewriter, loc, VectorType::get(16, opType), srcBuffAcc,
                indicesAcc);
            Value addOp;
            Value addOp2;

            if (ipType.isBF16()) {
              addOp =
                  arith::AddFOp::create(rewriter, loc, shuffle1, valueCRow1);

              addOp2 =
                  arith::AddFOp::create(rewriter, loc, shuffle2, valueCRow2);
            }

            if (ipType.isSignlessInteger(8)) {
              addOp =
                  arith::AddIOp::create(rewriter, loc, shuffle1, valueCRow1);

              addOp2 =
                  arith::AddIOp::create(rewriter, loc, shuffle2, valueCRow2);
            }
            indicesAcc[indicesAcc.size() - 1] = c0;
            vector::StoreOp::create(rewriter, loc, addOp, srcBuffAcc,
                                    indicesAcc);
            indicesAcc[indicesAcc.size() - 1] = c16;
            vector::StoreOp::create(rewriter, loc, addOp2, srcBuffAcc,
                                    indicesAcc);

            scf::YieldOp::create(nestedBuilder, loc);
          });
    }
    auto bufferType = MemRefType::get({16, 16}, opType);
    auto bBuffer =
        memref::AllocaOp::create(rewriter, outerLoop.getLoc(), bufferType);

    SmallVector<Value> dps = newLoop.getResults();
    for (size_t i = 0; i < ops.size(); i++) {
      vector::ContractionOp contOp = ops[i];
      Operation *resultWriteOp =
          traceToVectorWriteLikeUserOperation(contOp.getResult());
      if (isVnni) {
        rewriter.setInsertionPoint(resultWriteOp);

        Value indexOp_0 =
            arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 0);

        amx::TileStoreOp::create(rewriter, outerLoop.getLoc(), bBuffer,
                                 ValueRange{indexOp_0, indexOp_0}, dps[i]);

        auto c0 =
            arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 0);
        auto one =
            arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 1);
        auto mBound =
            arith::ConstantIndexOp::create(rewriter, outerLoop.getLoc(), 16);

        scf::ForOp::create(
            rewriter, outerLoop.getLoc(), c0, mBound, one, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value iv,
                ValueRange iterArgs) {
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
                          return mlir::getValueOrCreateConstantIndexOp(
                              rewriter, loc, ofr);
                        });
                  });

              Value sum =
                  arith::AddIOp::create(builder, loc, iv, indicesAcc[0]);
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
      }

      rewriter.eraseOp(resultWriteOp);
    }

    return success();
  }
};

} // namespace

void x86::populateVectorContractToAMXDotProductPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractToAMXDotProduct>(patterns.getContext());
}
