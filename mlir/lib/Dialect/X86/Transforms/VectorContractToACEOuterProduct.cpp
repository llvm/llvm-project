//===- VectorContractToACEOuterProduct.cpp---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/Utils/X86Utils.h"
#include "mlir/Dialect/X86/X86Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86;

namespace {
// Follows a single-use value through enclosing scf.yield ops and returns the
// value that leaves the loop nest, or null if any step has multiple uses.
static Value contractionUsersAfterYield(Value v) {
  if (!v.hasOneUse())
    return nullptr;

  OpOperand &use = *v.use_begin();
  auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
  if (!yield)
    return v;

  return contractionUsersAfterYield(
      yield->getParentOp()->getResult(use.getOperandNumber()));
}

// Returns the memref read by a transfer_read/load operand, or null.
static Value getSourceBuffer(Value operand) {
  Operation *defOp = operand.getDefiningOp();
  if (!defOp)
    return nullptr;

  Value srcBuff;
  llvm::TypeSwitch<Operation *>(defOp).Case<TransferReadOp, LoadOp>(
      [&](auto readOp) { srcBuff = readOp.getBase(); });
  return srcBuff;
}

static SmallVector<int64_t> getNonUnitDims(VectorType type) {
  SmallVector<int64_t> dims;
  llvm::copy_if(type.getShape(), std::back_inserter(dims),
                [](int64_t dim) { return dim != 16 && dim != 1; });
  return dims;
}

// Checks that a contraction reads from the expected buffers, has a single
// result chain and operates on 16x16 tiles whose only other dim is `kDim`.
static LogicalResult validateContractOp(vector::ContractionOp contractOp,
                                        int64_t kDim, Value srcBuffLhs,
                                        Value srcBuffRhs) {
  if (getSourceBuffer(contractOp.getLhs()) != srcBuffLhs ||
      getSourceBuffer(contractOp.getRhs()) != srcBuffRhs)
    return failure();

  if (!contractionUsersAfterYield(contractOp.getResult()))
    return failure();

  auto accTy = dyn_cast<VectorType>(contractOp.getAccType());
  if (!accTy || !getNonUnitDims(accTy).empty())
    return failure();

  // Flat i8/f8 operands have K == 16, so no extra dim is expected.
  auto hasValidKDim = [&](VectorType type) {
    SmallVector<int64_t> dims = getNonUnitDims(type);
    if (kDim == 16)
      return dims.empty();
    return dims.size() == 1 && dims[0] == kDim;
  };

  if (!hasValidKDim(contractOp.getLhsType()) ||
      !hasValidKDim(contractOp.getRhsType()))
    return failure();

  return success();
}

// Returns the subview offset position that uses `loop`'s induction variable.
static unsigned getIndexPosition(Value operand, scf::ForOp loop) {
  auto subview = getSourceBuffer(operand).getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return 0;

  for (auto [idx, offset] : llvm::enumerate(subview.getOffsets()))
    if (offset == loop.getInductionVar())
      return idx;

  return 0;
}

static SmallVector<Value> createTileZeros(OpBuilder &rewriter, Location loc,
                                          Type ipType, Type opType,
                                          scf::ForOp loop, int64_t size) {
  rewriter.setInsertionPoint(loop);
  // The block-scale registers are only consumed by the f8 MX outer products.
  if (ipType.isF8E5M2() || ipType.isF8E4M3FN())
    ace::BSRInitOp::create(rewriter, loc);

  SmallVector<Value> loopItrArgs;
  auto zeroTileType = amx::TileType::get({16, 16}, opType);

  for (int i = 0; i < size; i++) {
    auto zeroTile = amx::TileZeroOp::create(rewriter, loc, zeroTileType);
    loopItrArgs.push_back(zeroTile);
  }

  return loopItrArgs;
}

static SmallVector<Value> loadMasks(OpBuilder &rewriter, Location loc,
                                    MemRefType memrefTy) {
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto vecTy = VectorType::get({16}, rewriter.getI32Type());

  SmallVector<Value> masks;
  for (StringRef name : {"mask_1", "mask_2", "mask_3"}) {
    Value global = memref::GetGlobalOp::create(rewriter, loc, memrefTy, name);
    masks.push_back(
        vector::LoadOp::create(rewriter, loc, vecTy, global, ValueRange{c0}));
  }
  return masks;
}

// Loads the next row (4 x i32), broadcasts it to all four 128-bit lanes and
// selects it into `currentVec` where `cmp` is set.
static Value loadShuffleSelect(OpBuilder &rewriter, Location loc,
                               Value currentVec, Value cmp, Value memref,
                               SmallVector<Value> &loadOffsets, Value c1,
                               VectorType vecTy, VectorType vecTy4) {
  int rowIdx = loadOffsets.size() - 2;
  loadOffsets[rowIdx] =
      arith::AddIOp::create(rewriter, loc, loadOffsets[rowIdx], c1);

  Value v = vector::LoadOp::create(rewriter, loc, vecTy4, memref, loadOffsets);

  Value shuffled = vector::ShuffleOp::create(
      rewriter, loc, vecTy, v, v,
      ArrayRef<int64_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});

  return arith::SelectOp::create(rewriter, loc, cmp, shuffled, currentVec);
}

// Transposes a flat row-major [rows, K] operand into per-tile VNNI vectors.
// Each 16-row block yields 4 vectors, one per K step (index = tile * 4 + k).
static SmallVector<Value> transposeShuffle(OpBuilder &rewriter, Location loc,
                                           Value matrix, Type ipType,
                                           Value mask1, Value mask2,
                                           Value mask3, int64_t upperBound,
                                           int64_t blockingFactor) {

  SmallVector<Value> transposedValue;

  auto subview = matrix.getDefiningOp<memref::SubViewOp>();
  ValueRange offsets = subview.getOffsets();
  Value source = subview.getSource();
  Type llvmPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());

  // View the source as i32 so each element packs `blockingFactor` K values.
  auto castOp =
      UnrealizedConversionCastOp::create(rewriter, loc, llvmPtrType, source);

  SmallVector<int64_t> newShape(cast<MemRefType>(source.getType()).getShape());
  newShape.back() /= blockingFactor;
  auto targetMemRefType = MemRefType::get(newShape, rewriter.getI32Type());

  Value i32Matrix =
      UnrealizedConversionCastOp::create(rewriter, loc, targetMemRefType,
                                         ValueRange{castOp.getResult(0)})
          .getResult(0);

  SmallVector<Value> loadOffsets(offsets.begin(), offsets.end());
  size_t rowDim = loadOffsets.size() - 2;
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto vecTy = VectorType::get({16}, rewriter.getI32Type());
  auto vecTy4 = VectorType::get({4}, rewriter.getI32Type());
  Value cstOne = arith::ConstantOp::create(rewriter, loc, vecTy,
                                           DenseIntElementsAttr::get(vecTy, 1));

  Value blk = arith::ConstantIndexOp::create(rewriter, loc, blockingFactor);
  loadOffsets.back() =
      arith::DivUIOp::create(rewriter, loc, loadOffsets.back(), blk);

  int64_t elemNumber = ipType.isBF16() ? 32 : 64;
  auto elemVecTy = VectorType::get({elemNumber}, ipType);

  for (int i = 0; i < upperBound; i = i + 16) {
    SmallVector<Value> vecs;
    for (int r = 0; r < 4; r++) {
      if (r > 0)
        loadOffsets[rowDim] =
            arith::AddIOp::create(rewriter, loc, loadOffsets[rowDim], c1);
      vecs.push_back(
          vector::LoadOp::create(rewriter, loc, vecTy, i32Matrix, loadOffsets));
    }

    // Lanes [4m, 4m+3] of vecs[r] take row 4m + r of the 16-row block.
    for (Value mask : {mask1, mask2, mask3}) {
      Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                        mask, cstOne);
      for (Value &vec : vecs)
        vec = loadShuffleSelect(rewriter, loc, vec, cmp, i32Matrix, loadOffsets,
                                c1, vecTy, vecTy4);
    }

    for (Value &vec : vecs)
      vec = vector::BitCastOp::create(rewriter, loc, elemVecTy, vec);

    Value v0 = vecs[0];
    Value v1 = vecs[1];
    Value v2 = vecs[2];
    Value v3 = vecs[3];

    Value s4;
    Value s5;
    Value s6;
    Value s7;

    if (ipType.isBF16()) {
      auto s0 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v0, v1,
          ArrayRef<int64_t>{0,  1,  32, 33, 2,  3,  34, 35, 8,  9,  40,
                            41, 10, 11, 42, 43, 16, 17, 48, 49, 18, 19,
                            50, 51, 24, 25, 56, 57, 26, 27, 58, 59});
      auto s1 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v0, v1,
          ArrayRef<int64_t>{4,  5,  36, 37, 6,  7,  38, 39, 12, 13, 44,
                            45, 14, 15, 46, 47, 20, 21, 52, 53, 22, 23,
                            54, 55, 28, 29, 60, 61, 30, 31, 62, 63});
      auto s2 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v2, v3,
          ArrayRef<int64_t>{0,  1,  32, 33, 2,  3,  34, 35, 8,  9,  40,
                            41, 10, 11, 42, 43, 16, 17, 48, 49, 18, 19,
                            50, 51, 24, 25, 56, 57, 26, 27, 58, 59});
      auto s3 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v2, v3,
          ArrayRef<int64_t>{4,  5,  36, 37, 6,  7,  38, 39, 12, 13, 44,
                            45, 14, 15, 46, 47, 20, 21, 52, 53, 22, 23,
                            54, 55, 28, 29, 60, 61, 30, 31, 62, 63});

      s4 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), s0, s2,
          ArrayRef<int64_t>{0,  1,  2,  3,  32, 33, 34, 35, 8,  9,  10,
                            11, 40, 41, 42, 43, 16, 17, 18, 19, 48, 49,
                            50, 51, 24, 25, 26, 27, 56, 57, 58, 59});
      s5 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), s0, s2,
          ArrayRef<int64_t>{4,  5,  6,  7,  36, 37, 38, 39, 12, 13, 14,
                            15, 44, 45, 46, 47, 20, 21, 22, 23, 52, 53,
                            54, 55, 28, 29, 30, 31, 60, 61, 62, 63});
      s6 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), s1, s3,
          ArrayRef<int64_t>{0,  1,  2,  3,  32, 33, 34, 35, 8,  9,  10,
                            11, 40, 41, 42, 43, 16, 17, 18, 19, 48, 49,
                            50, 51, 24, 25, 26, 27, 56, 57, 58, 59});
      s7 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), s1, s3,
          ArrayRef<int64_t>{4,  5,  6,  7,  36, 37, 38, 39, 12, 13, 14,
                            15, 44, 45, 46, 47, 20, 21, 22, 23, 52, 53,
                            54, 55, 28, 29, 30, 31, 60, 61, 62, 63});
    } else {

      auto s0 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), v0, v1,
          ArrayRef<int64_t>{
              0,   1,   2,   3,   64, 65,  66,  67,  4,   5,   6,   7,  68,
              69,  70,  71,  16,  17, 18,  19,  80,  81,  82,  83,  20, 21,
              22,  23,  84,  85,  86, 87,  32,  33,  34,  35,  96,  97, 98,
              99,  36,  37,  38,  39, 100, 101, 102, 103, 48,  49,  50, 51,
              112, 113, 114, 115, 52, 53,  54,  55,  116, 117, 118, 119});

      auto s1 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), v0, v1,
          ArrayRef<int64_t>{
              8,   9,   10,  11,  72, 73,  74,  75,  12,  13,  14,  15,  76,
              77,  78,  79,  24,  25, 26,  27,  88,  89,  90,  91,  28,  29,
              30,  31,  92,  93,  94, 95,  40,  41,  42,  43,  104, 105, 106,
              107, 44,  45,  46,  47, 108, 109, 110, 111, 56,  57,  58,  59,
              120, 121, 122, 123, 60, 61,  62,  63,  124, 125, 126, 127});

      auto s2 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), v2, v3,
          ArrayRef<int64_t>{
              0,   1,   2,   3,   64, 65,  66,  67,  4,   5,   6,   7,  68,
              69,  70,  71,  16,  17, 18,  19,  80,  81,  82,  83,  20, 21,
              22,  23,  84,  85,  86, 87,  32,  33,  34,  35,  96,  97, 98,
              99,  36,  37,  38,  39, 100, 101, 102, 103, 48,  49,  50, 51,
              112, 113, 114, 115, 52, 53,  54,  55,  116, 117, 118, 119});

      auto s3 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), v2, v3,
          ArrayRef<int64_t>{
              8,   9,   10,  11,  72, 73,  74,  75,  12,  13,  14,  15,  76,
              77,  78,  79,  24,  25, 26,  27,  88,  89,  90,  91,  28,  29,
              30,  31,  92,  93,  94, 95,  40,  41,  42,  43,  104, 105, 106,
              107, 44,  45,  46,  47, 108, 109, 110, 111, 56,  57,  58,  59,
              120, 121, 122, 123, 60, 61,  62,  63,  124, 125, 126, 127});

      s4 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), s0, s2,
          ArrayRef<int64_t>{0,   1,   2,   3,   4,   5,   6,   7,   64, 65, 66,
                            67,  68,  69,  70,  71,  16,  17,  18,  19, 20, 21,
                            22,  23,  80,  81,  82,  83,  84,  85,  86, 87, 32,
                            33,  34,  35,  36,  37,  38,  39,  96,  97, 98, 99,
                            100, 101, 102, 103, 48,  49,  50,  51,  52, 53, 54,
                            55,  112, 113, 114, 115, 116, 117, 118, 119});

      s5 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), s0, s2,
          ArrayRef<int64_t>{
              8,  9,   10,  11,  12,  13,  14,  15,  72,  73,  74,  75, 76,
              77, 78,  79,  24,  25,  26,  27,  28,  29,  30,  31,  88, 89,
              90, 91,  92,  93,  94,  95,  40,  41,  42,  43,  44,  45, 46,
              47, 104, 105, 106, 107, 108, 109, 110, 111, 56,  57,  58, 59,
              60, 61,  62,  63,  120, 121, 122, 123, 124, 125, 126, 127});

      s6 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), s1, s3,
          ArrayRef<int64_t>{0,   1,   2,   3,   4,   5,   6,   7,   64, 65, 66,
                            67,  68,  69,  70,  71,  16,  17,  18,  19, 20, 21,
                            22,  23,  80,  81,  82,  83,  84,  85,  86, 87, 32,
                            33,  34,  35,  36,  37,  38,  39,  96,  97, 98, 99,
                            100, 101, 102, 103, 48,  49,  50,  51,  52, 53, 54,
                            55,  112, 113, 114, 115, 116, 117, 118, 119});

      s7 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), s1, s3,
          ArrayRef<int64_t>{
              8,  9,   10,  11,  12,  13,  14,  15,  72,  73,  74,  75, 76,
              77, 78,  79,  24,  25,  26,  27,  28,  29,  30,  31,  88, 89,
              90, 91,  92,  93,  94,  95,  40,  41,  42,  43,  44,  45, 46,
              47, 104, 105, 106, 107, 108, 109, 110, 111, 56,  57,  58, 59,
              60, 61,  62,  63,  120, 121, 122, 123, 124, 125, 126, 127});
    }

    transposedValue.append({s4, s5, s6, s7});

    loadOffsets[rowDim] =
        arith::AddIOp::create(rewriter, loc, loadOffsets[rowDim], c1);
  }

  return transposedValue;
}

/// Helper to generate a strided prefetch. Updates the selected index in
/// `loadOffsets`, emits `memref.prefetch`, and restores the original offset.
static void createStridedPrefetch(bool isStrided, bool leastOffset,
                                  int64_t offset, int64_t preOffset,
                                  Value restoreOffset, OpBuilder &rewriter,
                                  Location loc, Value matrix,
                                  SmallVectorImpl<Value> &loadOffsets) {
  if (!isStrided)
    return;

  size_t dim = leastOffset ? loadOffsets.size() - 1 : loadOffsets.size() - 2;
  Value pOffset =
      arith::ConstantIndexOp::create(rewriter, loc, preOffset + offset);
  loadOffsets[dim] = pOffset;
  memref::PrefetchOp::create(rewriter, loc, matrix, loadOffsets, false, 3,
                             true);
  loadOffsets[dim] = restoreOffset;
}

// Packs K rows `kOffset..` of a flat row-major [K, cols] operand into per-tile
// VNNI vectors. Full 32 (bf16) / 64 (i8/f8) column chunks use lane-local
// interleaves that permute columns; see the writeback fix-up for groups 3/4.
static SmallVector<Value> vnniShuffle(OpBuilder &rewriter, Location loc,
                                      Value matrix, Type ipType,
                                      int64_t upperBound, int64_t kOffset,
                                      int group) {
  int loopStep = 32;
  if (!ipType.isBF16()) {
    kOffset = kOffset * 2;
    loopStep = 64;
  }

  auto matrixTy = cast<MemRefType>(matrix.getType());
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> loadOffsets(matrixTy.getRank(), c0);

  SmallVector<Value> shuffledValue;

  Value offset_0 = arith::ConstantIndexOp::create(rewriter, loc, kOffset);
  Value offset_1 = arith::ConstantIndexOp::create(rewriter, loc, kOffset + 1);
  Value offset_2 = arith::ConstantIndexOp::create(rewriter, loc, kOffset + 2);
  Value offset_3 = arith::ConstantIndexOp::create(rewriter, loc, kOffset + 3);

  bool isStrided = true;
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(matrixTy.getStridesAndOffset(strides, offset)))
    isStrided = false;

  // Prefetch 4 KiB ahead, along rows when the row stride allows it.
  int64_t preOffset = 0;
  bool leastOffset = false;

  if (isStrided) {
    if (strides[strides.size() - 1] != 1)
      isStrided = false;

    if (strides[strides.size() - 2] < 4096) {
      preOffset = 4096 / strides[strides.size() - 2];
    } else {
      leastOffset = true;
      preOffset = 4096;
    }
  }

  for (int i = 0; i + loopStep <= upperBound; i = i + loopStep) {
    Value offset_i = arith::ConstantIndexOp::create(rewriter, loc, i);
    loadOffsets[loadOffsets.size() - 1] = offset_i;
    loadOffsets[loadOffsets.size() - 2] = offset_0;

    Value v0 = vector::LoadOp::create(rewriter, loc,
                                      VectorType::get({loopStep}, ipType),
                                      matrix, loadOffsets);
    createStridedPrefetch(isStrided, leastOffset, leastOffset ? i : kOffset,
                          preOffset, leastOffset ? offset_i : offset_0,
                          rewriter, loc, matrix, loadOffsets);

    loadOffsets[loadOffsets.size() - 2] = offset_1;
    Value v1 = vector::LoadOp::create(rewriter, loc,
                                      VectorType::get({loopStep}, ipType),
                                      matrix, loadOffsets);
    createStridedPrefetch(isStrided, leastOffset, leastOffset ? i : kOffset + 1,
                          preOffset, leastOffset ? offset_i : offset_1,
                          rewriter, loc, matrix, loadOffsets);

    Value s0;
    Value s1;
    Value s2;
    Value s3;

    if (ipType.isBF16()) {
      s0 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v0, v1,
          ArrayRef<int64_t>{0,  32, 1,  33, 2,  34, 3,  35, 8,  40, 9,
                            41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50,
                            19, 51, 24, 56, 25, 57, 26, 58, 27, 59});
      s1 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v0, v1,
          ArrayRef<int64_t>{4,  36, 5,  37, 6,  38, 7,  39, 12, 44, 13,
                            45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54,
                            23, 55, 28, 60, 29, 61, 30, 62, 31, 63});
    } else {

      loadOffsets[loadOffsets.size() - 2] = offset_2;
      Value v2 = vector::LoadOp::create(rewriter, loc,
                                        VectorType::get({loopStep}, ipType),
                                        matrix, loadOffsets);
      createStridedPrefetch(isStrided, leastOffset,
                            leastOffset ? i : kOffset + 2, preOffset,
                            leastOffset ? offset_i : offset_2, rewriter, loc,
                            matrix, loadOffsets);

      loadOffsets[loadOffsets.size() - 2] = offset_3;
      Value v3 = vector::LoadOp::create(rewriter, loc,
                                        VectorType::get({loopStep}, ipType),
                                        matrix, loadOffsets);
      createStridedPrefetch(isStrided, leastOffset,
                            leastOffset ? i : kOffset + 3, preOffset,
                            leastOffset ? offset_i : offset_3, rewriter, loc,
                            matrix, loadOffsets);

      if (group == 4) {
        auto a0 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v0, v2,
            ArrayRef<int64_t>{0,  64, 1,  65, 2,  66, 3,  67, 4,  68, 5,
                              69, 6,  70, 7,  71, 8,  72, 9,  73, 10, 74,
                              11, 75, 12, 76, 13, 77, 14, 78, 15, 79, 16,
                              80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85,
                              22, 86, 23, 87, 24, 88, 25, 89, 26, 90, 27,
                              91, 28, 92, 29, 93, 30, 94, 31, 95});

        auto a1 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v0, v2,
            ArrayRef<int64_t>{
                32,  96,  33,  97,  34,  98,  35,  99,  36,  100, 37,  101, 38,
                102, 39,  103, 40,  104, 41,  105, 42,  106, 43,  107, 44,  108,
                45,  109, 46,  110, 47,  111, 48,  112, 49,  113, 50,  114, 51,
                115, 52,  116, 53,  117, 54,  118, 55,  119, 56,  120, 57,  121,
                58,  122, 59,  123, 60,  124, 61,  125, 62,  126, 63,  127});

        auto a2 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v1, v3,
            ArrayRef<int64_t>{0,  64, 1,  65, 2,  66, 3,  67, 4,  68, 5,
                              69, 6,  70, 7,  71, 8,  72, 9,  73, 10, 74,
                              11, 75, 12, 76, 13, 77, 14, 78, 15, 79, 16,
                              80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85,
                              22, 86, 23, 87, 24, 88, 25, 89, 26, 90, 27,
                              91, 28, 92, 29, 93, 30, 94, 31, 95});

        auto a3 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v1, v3,
            ArrayRef<int64_t>{
                32,  96,  33,  97,  34,  98,  35,  99,  36,  100, 37,  101, 38,
                102, 39,  103, 40,  104, 41,  105, 42,  106, 43,  107, 44,  108,
                45,  109, 46,  110, 47,  111, 48,  112, 49,  113, 50,  114, 51,
                115, 52,  116, 53,  117, 54,  118, 55,  119, 56,  120, 57,  121,
                58,  122, 59,  123, 60,  124, 61,  125, 62,  126, 63,  127});

        s0 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a0, a2,
            ArrayRef<int64_t>{0,  1,  64, 65, 2,  3,  66, 67, 4,  5,  68,
                              69, 6,  7,  70, 71, 8,  9,  72, 73, 10, 11,
                              74, 75, 12, 13, 76, 77, 14, 15, 78, 79, 16,
                              17, 80, 81, 18, 19, 82, 83, 20, 21, 84, 85,
                              22, 23, 86, 87, 24, 25, 88, 89, 26, 27, 90,
                              91, 28, 29, 92, 93, 30, 31, 94, 95});

        s1 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a0, a2,
            ArrayRef<int64_t>{
                32,  33,  96,  97,  34,  35,  98,  99,  36,  37,  100, 101, 38,
                39,  102, 103, 40,  41,  104, 105, 42,  43,  106, 107, 44,  45,
                108, 109, 46,  47,  110, 111, 48,  49,  112, 113, 50,  51,  114,
                115, 52,  53,  116, 117, 54,  55,  118, 119, 56,  57,  120, 121,
                58,  59,  122, 123, 60,  61,  124, 125, 62,  63,  126, 127});

        s2 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a1, a3,
            ArrayRef<int64_t>{0,  1,  64, 65, 2,  3,  66, 67, 4,  5,  68,
                              69, 6,  7,  70, 71, 8,  9,  72, 73, 10, 11,
                              74, 75, 12, 13, 76, 77, 14, 15, 78, 79, 16,
                              17, 80, 81, 18, 19, 82, 83, 20, 21, 84, 85,
                              22, 23, 86, 87, 24, 25, 88, 89, 26, 27, 90,
                              91, 28, 29, 92, 93, 30, 31, 94, 95});

        s3 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a1, a3,
            ArrayRef<int64_t>{
                32,  33,  96,  97,  34,  35,  98,  99,  36,  37,  100, 101, 38,
                39,  102, 103, 40,  41,  104, 105, 42,  43,  106, 107, 44,  45,
                108, 109, 46,  47,  110, 111, 48,  49,  112, 113, 50,  51,  114,
                115, 52,  53,  116, 117, 54,  55,  118, 119, 56,  57,  120, 121,
                58,  59,  122, 123, 60,  61,  124, 125, 62,  63,  126, 127});
      } else {
        auto a0 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v0, v1,
            ArrayRef<int64_t>{
                0,  64,  1,   65,  2,   66,  3,   67,  4,   68,  5,   69, 6,
                70, 7,   71,  16,  80,  17,  81,  18,  82,  19,  83,  20, 84,
                21, 85,  22,  86,  23,  87,  32,  96,  33,  97,  34,  98, 35,
                99, 36,  100, 37,  101, 38,  102, 39,  103, 48,  112, 49, 113,
                50, 114, 51,  115, 52,  116, 53,  117, 54,  118, 55,  119});

        auto a1 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v0, v1,
            ArrayRef<int64_t>{
                8,   72,  9,   73,  10,  74,  11,  75,  12,  76,  13,  77,  14,
                78,  15,  79,  24,  88,  25,  89,  26,  90,  27,  91,  28,  92,
                29,  93,  30,  94,  31,  95,  40,  104, 41,  105, 42,  106, 43,
                107, 44,  108, 45,  109, 46,  110, 47,  111, 56,  120, 57,  121,
                58,  122, 59,  123, 60,  124, 61,  125, 62,  126, 63,  127});

        auto a2 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v2, v3,
            ArrayRef<int64_t>{
                0,  64,  1,   65,  2,   66,  3,   67,  4,   68,  5,   69, 6,
                70, 7,   71,  16,  80,  17,  81,  18,  82,  19,  83,  20, 84,
                21, 85,  22,  86,  23,  87,  32,  96,  33,  97,  34,  98, 35,
                99, 36,  100, 37,  101, 38,  102, 39,  103, 48,  112, 49, 113,
                50, 114, 51,  115, 52,  116, 53,  117, 54,  118, 55,  119});

        auto a3 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v2, v3,
            ArrayRef<int64_t>{
                8,   72,  9,   73,  10,  74,  11,  75,  12,  76,  13,  77,  14,
                78,  15,  79,  24,  88,  25,  89,  26,  90,  27,  91,  28,  92,
                29,  93,  30,  94,  31,  95,  40,  104, 41,  105, 42,  106, 43,
                107, 44,  108, 45,  109, 46,  110, 47,  111, 56,  120, 57,  121,
                58,  122, 59,  123, 60,  124, 61,  125, 62,  126, 63,  127});

        s0 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a0, a2,
            ArrayRef<int64_t>{
                0,  1,  64,  65,  2,   3,  66,  67,  4,   5,  68,  69,  6,
                7,  70, 71,  16,  17,  80, 81,  18,  19,  82, 83,  20,  21,
                84, 85, 22,  23,  86,  87, 32,  33,  96,  97, 34,  35,  98,
                99, 36, 37,  100, 101, 38, 39,  102, 103, 48, 49,  112, 113,
                50, 51, 114, 115, 52,  53, 116, 117, 54,  55, 118, 119});
        s1 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a0, a2,
            ArrayRef<int64_t>{
                8,   9,  72,  73,  10,  11, 74,  75,  12,  13,  76,  77,  14,
                15,  78, 79,  24,  25,  88, 89,  26,  27,  90,  91,  28,  29,
                92,  93, 30,  31,  94,  95, 40,  41,  104, 105, 42,  43,  106,
                107, 44, 45,  108, 109, 46, 47,  110, 111, 56,  57,  120, 121,
                58,  59, 122, 123, 60,  61, 124, 125, 62,  63,  126, 127});

        s2 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a1, a3,
            ArrayRef<int64_t>{
                0,  1,  64,  65,  2,   3,  66,  67,  4,   5,  68,  69,  6,
                7,  70, 71,  16,  17,  80, 81,  18,  19,  82, 83,  20,  21,
                84, 85, 22,  23,  86,  87, 32,  33,  96,  97, 34,  35,  98,
                99, 36, 37,  100, 101, 38, 39,  102, 103, 48, 49,  112, 113,
                50, 51, 114, 115, 52,  53, 116, 117, 54,  55, 118, 119});
        s3 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a1, a3,
            ArrayRef<int64_t>{
                8,   9,  72,  73,  10,  11, 74,  75,  12,  13,  76,  77,  14,
                15,  78, 79,  24,  25,  88, 89,  26,  27,  90,  91,  28,  29,
                92,  93, 30,  31,  94,  95, 40,  41,  104, 105, 42,  43,  106,
                107, 44, 45,  108, 109, 46, 47,  110, 111, 56,  57,  120, 121,
                58,  59, 122, 123, 60,  61, 124, 125, 62,  63,  126, 127});
      }
    }

    shuffledValue.append({s0, s1});
    if (!ipType.isBF16())
      shuffledValue.append({s2, s3});
  }

  // Remaining columns (< 64, i8/f8 only) are packed in natural column order.
  auto vecTy16 = VectorType::get({16}, ipType);
  auto vecTy32 = VectorType::get({32}, ipType);
  auto vecTy64 = VectorType::get({64}, ipType);
  SmallVector<int64_t> interleave16, interleave32;
  for (int64_t t = 0; t < 16; ++t) {
    interleave16.append({t, t + 16});
    interleave32.append({2 * t, 2 * t + 1, 32 + 2 * t, 32 + 2 * t + 1});
  }

  for (int64_t i = upperBound - upperBound % loopStep; i < upperBound;
       i += 16) {
    loadOffsets[loadOffsets.size() - 1] =
        arith::ConstantIndexOp::create(rewriter, loc, i);

    SmallVector<Value> v;
    for (Value off : {offset_0, offset_1, offset_2, offset_3}) {
      loadOffsets[loadOffsets.size() - 2] = off;
      v.push_back(
          vector::LoadOp::create(rewriter, loc, vecTy16, matrix, loadOffsets));
    }

    // Group 4 packs the K rows as (0, 2, 1, 3); keep the same order here.
    if (group == 4)
      std::swap(v[1], v[2]);

    Value p01 = vector::ShuffleOp::create(rewriter, loc, vecTy32, v[0], v[1],
                                          interleave16);
    Value p23 = vector::ShuffleOp::create(rewriter, loc, vecTy32, v[2], v[3],
                                          interleave16);
    shuffledValue.push_back(vector::ShuffleOp::create(rewriter, loc, vecTy64,
                                                      p01, p23, interleave32));
  }

  return shuffledValue;
}

// Loads one VNNI vector per 16 rows of an already VNNI-packed operand.
static SmallVector<Value> loadData(OpBuilder &rewriter, Location loc,
                                   Value matrix, Type ipType,
                                   int64_t upperBound) {
  SmallVector<Value> matrixValue;
  auto memrefType = cast<MemRefType>(matrix.getType());
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> loadOffsets(memrefType.getRank(), c0);
  auto vecTy = VectorType::get({ipType.isBF16() ? 32 : 64}, ipType);

  for (int i = 0; i < upperBound; i = i + 16) {
    loadOffsets[loadOffsets.size() - 2] =
        arith::ConstantIndexOp::create(rewriter, loc, i);
    matrixValue.push_back(
        vector::LoadOp::create(rewriter, loc, vecTy, matrix, loadOffsets));
  }

  return matrixValue;
}

// Creates the ACE outer product for one tile and K step.
static Value createTileMulOutProdOp(OpBuilder &rewriter, Location loc,
                                    Type ipType, Value lhs, Value rhs,
                                    Value acc, unsigned tileIdx,
                                    unsigned numCols) {
  if (ipType.isBF16())
    return ace::TileMulFOutProdOp::create(rewriter, loc, lhs, rhs, acc,
                                          /*scale=*/IntegerAttr());

  if (ipType.isSignlessInteger(8))
    return ace::TileMulIOutProdOp::create(rewriter, loc, lhs, rhs, acc);

  // Matched-type f8E5M2 (bf8) and f8E4M3FN (hf8) outer products; the mixed
  // bf8/hf8 forms are left for future work. The MX scale immediate selects the
  // destination tile's block-scale group: bits [2:0] hold the M-tile row and
  // bits [7:3] hold the N-tile column.
  if (ipType.isF8E5M2() || ipType.isF8E4M3FN()) {
    unsigned row = tileIdx / numCols;
    unsigned col = tileIdx % numCols;
    uint8_t scale = static_cast<uint8_t>((col << 3) | row);
    return ace::TileMulFOutProdOp::create(rewriter, loc, lhs, rhs, acc,
                                          rewriter.getI8IntegerAttr(scale));
  }

  llvm_unreachable("Unsupported input type");
}

// Builds the reduction loop that replaces `innerLoop`, re-creating the A/B
// subviews for the new induction variables and emitting the ACE outer
// products for every 16x16 accumulator tile.
static scf::ForOp createLoops(OpBuilder &rewriter, Location loc,
                              Value lowerBound, Value upperBound, Value step,
                              ValueRange loopItrArgs, Type ipType,
                              Operation *vectorOpLhs, Operation *vectorOpRhs,
                              vector::ContractionOp contractOp,
                              scf::ForOp outerLoop, scf::ForOp innerLoop,
                              Value ivOuterLoop, Value mask1, Value mask2,
                              Value mask3, int group, unsigned blockingFactor,
                              unsigned M, unsigned N) {

  auto newLoop = scf::ForOp::create(
      rewriter, loc, lowerBound, upperBound, step, loopItrArgs,
      [&](OpBuilder &rewriterNewInnerLoop, Location locNewInnerLoop,
          Value ivNewInnerLoop, ValueRange iterArgsNewInnerLoop) {
        // Clones the operand's source subview with the new loop IVs.
        auto cloneSource = [&](Operation *srcOp, Value operand) -> Value {
          if (srcOp->getNumOperands() == 0)
            return srcOp->getResult(0);

          IRMapping mapping;
          auto mapIv = [&](scf::ForOp loop, Value iv) {
            unsigned operandIdx = getIndexPosition(operand, loop) + 1;
            if (operandIdx < srcOp->getNumOperands())
              mapping.map(srcOp->getOperand(operandIdx), iv);
          };
          if (outerLoop)
            mapIv(outerLoop, ivOuterLoop);
          mapIv(innerLoop, ivNewInnerLoop);
          return rewriterNewInnerLoop.clone(*srcOp, mapping)->getResult(0);
        };

        Value matA = cloneSource(vectorOpLhs, contractOp.getLhs());
        Value matB = cloneSource(vectorOpRhs, contractOp.getRhs());
        unsigned numRows = M / 16;
        unsigned numCols = N / 16;

        // Operand vectors are indexed per 16-row/col tile; transposeShuffle
        // returns 4 K-steps per tile (tile * 4 + kStep).
        SmallVector<Value> matAValue, matBValue;
        if (group == 1) {
          matAValue = loadData(rewriter, loc, matA, ipType, M);
          matBValue = loadData(rewriter, loc, matB, ipType, N);
        }
        if (group == 4) {
          matAValue = vnniShuffle(rewriter, loc, matA, ipType, M, 0, group);
          matBValue = vnniShuffle(rewriter, loc, matB, ipType, N, 0, group);
        }
        if (group == 3 || group == 5)
          matAValue = transposeShuffle(rewriter, loc, matA, ipType, mask1,
                                       mask2, mask3, M, blockingFactor);
        if (group == 5 || group == 6)
          matBValue = transposeShuffle(rewriter, loc, matB, ipType, mask1,
                                       mask2, mask3, N, blockingFactor);

        int numKSteps = (group == 1 || group == 4) ? 1 : 4;
        SmallVector<Value> ops(iterArgsNewInnerLoop.begin(),
                               iterArgsNewInnerLoop.end());

        for (int kStep = 0; kStep < numKSteps; kStep++) {
          SmallVector<Value> aStep, bStep;
          if (group == 3)
            bStep =
                vnniShuffle(rewriter, loc, matB, ipType, N, 2 * kStep, group);
          if (group == 6)
            aStep =
                vnniShuffle(rewriter, loc, matA, ipType, M, 2 * kStep, group);

          for (unsigned i = 0; i < numRows; i++) {
            for (unsigned j = 0; j < numCols; j++) {
              Value lhs = !aStep.empty()   ? aStep[i]
                          : numKSteps == 1 ? matAValue[i]
                                           : matAValue[i * 4 + kStep];
              Value rhs = !bStep.empty()   ? bStep[j]
                          : numKSteps == 1 ? matBValue[j]
                                           : matBValue[j * 4 + kStep];
              unsigned k = i * numCols + j;
              ops[k] = createTileMulOutProdOp(rewriter, loc, ipType, lhs, rhs,
                                              ops[k], k, numCols);
            }
          }
        }

        scf::YieldOp::create(rewriterNewInnerLoop, locNewInnerLoop, ops);
      });

  return newLoop;
}

// Classifies the operand layouts of the contraction into the cases listed in
// the table above VectorContractToACEOuterProduct. Returns 0 if unsupported.
static int caseValue(vector::ContractionOp contractOp) {
  auto isVnniDim = [](int64_t dim) { return dim == 2 || dim == 4; };
  bool isLhsVnni = isVnniDim(contractOp.getLhsType().getShape().back());
  bool isRhsVnni = isVnniDim(contractOp.getRhsType().getShape().back());

  SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
  ArrayRef<AffineExpr> resA = maps[0].getResults();
  ArrayRef<AffineExpr> resB = maps[1].getResults();
  ArrayRef<AffineExpr> resAcc = maps[2].getResults();

  // A is non-transposed when M is its second-innermost non-VNNI dim, B when N
  // is its innermost non-VNNI dim.
  bool isLhsTransposed =
      resAcc[resAcc.size() - 2] != resA[resA.size() - (isLhsVnni ? 3 : 2)];
  bool isRhsTransposed =
      resAcc[resAcc.size() - 1] != resB[resB.size() - (isRhsVnni ? 2 : 1)];

  // Case 1:
  if (isLhsVnni && isRhsVnni)
    return (isLhsTransposed && !isRhsTransposed) ? 1 : 0;

  if (isLhsVnni || isRhsVnni)
    return 0;

  // Case 3:
  if (!isLhsTransposed && !isRhsTransposed)
    return 3;

  // Case 4:
  if (isLhsTransposed && !isRhsTransposed)
    return 4;

  // Case 5:
  if (!isLhsTransposed && isRhsTransposed)
    return 5;

  // Case 6:
  if (isLhsTransposed && isRhsTransposed)
    return 6;

  return 6;
}

// Implements the ACE tiled outer-product for a sequence of register-blocked
// vector.contracts (one per 16x16 accumulator tile) inside at most two
// reduction loops. The M x N accumulator block must be a multiple of 32 in
// both dims and K per contraction must be one VNNI group (cases 1/4) or four
// VNNI groups (cases 3/5/6); VNNI is 2 for bf16 and 4 for i8/f8.
//
// Supported operand layouts:
// -----------------------------------------------
// Case |  A                    |  B
// -----|-----------------------|-----------------
// 1    |  VNNI, transposed     |  VNNI
// 2    |  flat                 |  VNNI (TODO)
// 3    |  flat                 |  flat
// 4    |  flat, transposed     |  flat
// 5    |  flat                 |  flat, transposed
// 6    |  flat, transposed     |  flat, transposed
// -----------------------------------------------
//
// For example, case 3 (bf16, M = N = 32, K = 8):
// ```
//   scf.for %k = ... iter_args(%c0..%c3) {
//     %a0, %a1 = vector.transfer_read %A : vector<1x16x8xbf16>
//     %b0, %b1 = vector.transfer_read %B : vector<1x8x16xbf16>
//     %c0' = vector.contract %a0, %b0, %c0 : ... into vector<16x16xf32>
//     ... (one contraction per tile)
//   }
//   vector.transfer_write %c0..%c3, %C
// ```
// to
// ```
//   %t0..%t3 = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
//   scf.for %k = ... iter_args(%t0..%t3) {
//     - pack A/B into per-tile VNNI vector<32xbf16> (transposeShuffle for
//       flat A, vnniShuffle for flat B), one vector per K step
//     - %t0' = x86.ace.tile_mulf %a, %b, %t0 for every tile and K step
//   }
//   scf.for %row = 0 to 16 {
//     - x86.amx.tilemovrow each tile row, undo the B column permutation
//       (cases 3/4), add the initial C row and store into a 16x16 buffer
//   }
//   - forward the buffers to the users of the original results
// ```
// For f8 inputs, x86.ace.tile_mulf also carries the MX scale immediate that
// encodes the destination tile position ((col << 3) | row).
struct VectorContractToACEOuterProduct
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    Type ipType = contractOp.getLhsType().getElementType();
    bool isF8 = ipType.isF8E5M2() || ipType.isF8E4M3FN();
    if (!ipType.isBF16() && !ipType.isSignlessInteger(8) && !isF8)
      return rewriter.notifyMatchFailure(
          contractOp, "Only BF16/Int8/F8 lowering is supported.");

    if (ipType != contractOp.getRhsType().getElementType())
      return rewriter.notifyMatchFailure(
          contractOp, "Contraction should have same lhs and rhs type.");

    unsigned blockingFactor = ipType.isBF16() ? 2 : 4;
    bool isVnni =
        isInVnniLayout(contractOp.getOperation(),
                       contractOp.getIndexingMapsArray(), blockingFactor);

    int group = caseValue(contractOp);
    if (group == 0)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Unsupported operand layout.");

    auto accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp, "Wrong accumulator type.");

    Type opType = accTy.getElementType();
    if ((ipType.isSignlessInteger(8) && !opType.isSignlessInteger(32)) ||
        (!ipType.isSignlessInteger(8) && !opType.isF32()))
      return rewriter.notifyMatchFailure(
          contractOp, "Only F32 for BF16/F8 or Int32 for "
                      "Int8 accumulation type is supported.");

    Operation *accReadOp =
        traceToVectorReadLikeParentOperation(contractOp.getAcc());
    Operation *resultWriteOp =
        traceToVectorWriteLikeUserOperation(contractOp.getResult());

    if (!accReadOp || !resultWriteOp)
      return rewriter.notifyMatchFailure(
          contractOp, "The ACC operand of the vector.contract should be a "
                      "transfer_read or a load. And, the result should be "
                      "stored using transfer_write or store.");

    if (accReadOp->getBlock() == contractOp->getBlock() &&
        resultWriteOp->getBlock() != contractOp->getBlock())
      return rewriter.notifyMatchFailure(
          contractOp, "The accumulator store is in different block.");

    if (accReadOp->getBlock() != contractOp->getBlock() &&
        resultWriteOp->getBlock() == contractOp->getBlock())
      return rewriter.notifyMatchFailure(
          contractOp, "The accumulator read is in different block.");

    Value srcBuffAcc;
    SmallVector<Value> indicesAcc;
    llvm::TypeSwitch<Operation *>(accReadOp).Case<TransferReadOp, LoadOp>(
        [&](auto readOp) {
          srcBuffAcc = readOp.getBase();
          for (Value index : readOp.getIndices())
            indicesAcc.push_back(index);
        });

    ArrayRef<int64_t> outputShape =
        cast<MemRefType>(srcBuffAcc.getType()).getShape();
    unsigned M = outputShape[outputShape.size() - 2];
    unsigned N = outputShape[outputShape.size() - 1];

    if (M == 0 || N == 0 || M % 32 != 0 || N % 32 != 0)
      return rewriter.notifyMatchFailure(
          contractOp, "The M and N tile sizes should be multiples of 32.");

    // The f8 MX scale immediate encodes the tile row in 3 bits and the tile
    // column in the remaining 5 bits.
    if (isF8 && (M / 16 > 8 || N / 16 > 32))
      return rewriter.notifyMatchFailure(
          contractOp, "The f8 MX scale supports up to M=128 and N=512.");

    // K extent of each contraction: one VNNI group for cases 1/4, four for
    // the flat-K cases.
    unsigned kDim =
        (group == 1 || group == 4) ? blockingFactor : 4 * blockingFactor;

    SmallVector<int64_t> iterationBounds;
    contractOp.getIterationBounds(iterationBounds);
    int64_t contractK = 1;
    for (auto [bound, iterType] :
         llvm::zip(iterationBounds, contractOp.getIteratorTypesArray()))
      if (iterType == vector::IteratorType::reduction)
        contractK *= bound;

    if (contractK != kDim)
      return rewriter.notifyMatchFailure(
          contractOp, "Expected K=" + Twine(kDim) +
                          " per contraction for this layout and type, got K=" +
                          Twine(contractK) + ".");

    // The accumulators are carried as iter_args through at most two
    // reduction loops between the accumulator read and the contraction.
    // TODO: Support a single contraction with the accumulator read/write in
    // the same block.
    SmallVector<scf::ForOp> loopLists;
    Operation *current = contractOp;
    while (true) {
      auto parent = current->getParentOfType<scf::ForOp>();
      if (!parent)
        return rewriter.notifyMatchFailure(
            contractOp,
            "Accumulator read and contract op not within scf.for op");

      loopLists.push_back(parent);
      if (accReadOp->getBlock() == parent->getBlock())
        break;

      current = parent;
    }

    if (loopLists.size() > 2)
      return rewriter.notifyMatchFailure(
          contractOp, "Rewrite is supported until reduction loop depth of 2.");

    Value srcBuffLhs = getSourceBuffer(contractOp.getLhs());
    if (!srcBuffLhs)
      return rewriter.notifyMatchFailure(contractOp,
                                         "The LHS src is not a MemRef type.");

    Value srcBuffRhs = getSourceBuffer(contractOp.getRhs());
    if (!srcBuffRhs)
      return rewriter.notifyMatchFailure(contractOp,
                                         "The RHS src is not a MemRef type.");

    Operation *vectorOpLhs = srcBuffLhs.getDefiningOp();
    Operation *vectorOpRhs = srcBuffRhs.getDefiningOp();
    if (!vectorOpLhs || !vectorOpRhs)
      return rewriter.notifyMatchFailure(
          contractOp, "The LHS/RHS src should be defined inside the loop.");

    // Collect all the contractions of the reduction loop body.
    SmallVector<vector::ContractionOp> ops;
    for (Operation &op : loopLists[0].getBody()->getOperations()) {
      auto contract = dyn_cast<vector::ContractionOp>(op);
      if (!contract)
        continue;

      if (failed(validateContractOp(contract, kDim, srcBuffLhs, srcBuffRhs)))
        return rewriter.notifyMatchFailure(
            contractOp, "The associated contract operations doesn't satisfy "
                        "the re-write conditions either the dimensions are "
                        "wrong or MemRef source are different or many users.");

      ops.push_back(contract);
    }

    if (ops.size() != (M / 16) * (N / 16))
      return rewriter.notifyMatchFailure(
          contractOp, "Expected one 16x16 contract per accumulator tile.");

    if (!isVnni) {
      SmallVector<vector::ContractionOp> checkOps(ops.begin(), ops.end());
      unsigned pairCount = 0;
      for (size_t j = 0; j < checkOps.size(); j++) {
        if (!checkOps[j])
          continue;
        for (size_t i = j + 1; i < checkOps.size(); i++) {
          if (checkOps[i] &&
              validatePairVectorContract(checkOps[j], checkOps[i], true, 16)) {
            checkOps[i] = nullptr;
            pairCount += 2;
          }
        }
      }

      if (pairCount != ops.size())
        return rewriter.notifyMatchFailure(
            contractOp, "Couldn't find the pair vector contract.");
    }

    // Lane-group masks used by transposeShuffle to gather 4 rows per vector.
    auto module = contractOp->getParentOfType<ModuleOp>();
    OpBuilder builder(module.getBodyRegion());
    auto memrefTy = MemRefType::get({16}, builder.getI32Type());
    auto tensorTy = RankedTensorType::get({16}, builder.getI32Type());
    for (int m = 1; m <= 3; m++) {
      std::string name = "mask_" + std::to_string(m);
      if (module.lookupSymbol(name))
        continue;
      SmallVector<int32_t> values(16, 0);
      std::fill_n(values.begin() + 4 * m, 4, 1);
      memref::GlobalOp::create(
          builder, contractOp.getLoc(), name, builder.getStringAttr("private"),
          memrefTy,
          DenseIntElementsAttr::get(tensorTy, ArrayRef<int32_t>(values)),
          /*constant=*/true, IntegerAttr());
    }

    scf::ForOp innerLoop = loopLists[0];
    // With a single reduction loop the write back is placed after it.
    scf::ForOp outerLoop = loopLists.back();
    scf::ForOp newLoop;

    SmallVector<Value> loopItrArgs = createTileZeros(
        rewriter, outerLoop.getLoc(), ipType, opType, outerLoop, ops.size());
    SmallVector<Value> mask = loadMasks(rewriter, innerLoop.getLoc(), memrefTy);

    if (loopLists.size() == 2) {
      // Batch-reduce matmul: rebuild both reduction loops.
      newLoop = scf::ForOp::create(
          rewriter, outerLoop.getLoc(), outerLoop.getLowerBound(),
          outerLoop.getUpperBound(), outerLoop.getStep(), loopItrArgs,
          [&](OpBuilder &rewriterOuterLoop, Location locOuterLoop,
              Value ivOuterLoop, ValueRange iterArgsOuterLoop) {
            auto newInnerLoop = createLoops(
                rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
                innerLoop.getUpperBound(), innerLoop.getStep(),
                iterArgsOuterLoop, ipType, vectorOpLhs, vectorOpRhs, contractOp,
                outerLoop, innerLoop, ivOuterLoop, mask[0], mask[1], mask[2],
                group, blockingFactor, M, N);

            scf::YieldOp::create(rewriterOuterLoop, locOuterLoop,
                                 newInnerLoop.getResults());
          });
    } else {
      // Matmul and batch matmul: a single reduction loop.
      newLoop = createLoops(
          rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
          innerLoop.getUpperBound(), innerLoop.getStep(), loopItrArgs, ipType,
          vectorOpLhs, vectorOpRhs, contractOp, nullptr, innerLoop, nullptr,
          mask[0], mask[1], mask[2], group, blockingFactor, M, N);
    }

    // Write back: move each tile row out, add the initial C value, stage it in
    // a 16x16 buffer and forward it to the original result users.
    Location loc = outerLoop.getLoc();

    SmallVector<Value> dps = newLoop.getResults();
    auto bufferType = MemRefType::get({16, 16}, opType);

    Value c16I16 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI16Type(), rewriter.getI16IntegerAttr(16));
    Value c64I16 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI16Type(), rewriter.getI16IntegerAttr(64));

    auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto c16 = arith::ConstantIndexOp::create(rewriter, loc, 16);
    auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    SmallVector<OpFoldResult> strides(indicesAcc.size(),
                                      rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes(indicesAcc.size(),
                                    rewriter.getIndexAttr(1));
    sizes[sizes.size() - 1] = rewriter.getIndexAttr(16);
    sizes[sizes.size() - 2] = rewriter.getIndexAttr(16);

    SmallVector<memref::SubViewOp> subviews;
    SmallVector<memref::AllocaOp> allocas;

    for (unsigned i = 0; i < M; i += 16) {
      for (unsigned j = 0; j < N; j += 32) {
        auto offsetI = arith::ConstantIndexOp::create(rewriter, loc, i);
        auto offsetJ = arith::ConstantIndexOp::create(rewriter, loc, j);
        auto offsetJ16 = arith::ConstantIndexOp::create(rewriter, loc, j + 16);

        indicesAcc[indicesAcc.size() - 2] = offsetI;
        for (Value offsetCol : {Value(offsetJ), Value(offsetJ16)}) {
          indicesAcc[indicesAcc.size() - 1] = offsetCol;
          subviews.push_back(memref::SubViewOp::create(
              rewriter, loc, srcBuffAcc, getAsOpFoldResult(indicesAcc), sizes,
              strides));
        }

        allocas.push_back(memref::AllocaOp::create(rewriter, loc, bufferType));
        allocas.push_back(memref::AllocaOp::create(rewriter, loc, bufferType));
      }
    }

    // Groups 3 and 4 produce B columns permuted within each 32 (bf16, tile
    // pairs) or 64 (i8/f8, tile quads) column chunk; undo it per row.
    bool permutedB = group == 3 || group == 4;
    unsigned numCols = N / 16;
    unsigned fullQuadCols = (numCols / 4) * 4;
    auto rowTy = VectorType::get(16, opType);

    // Undoes the bf16 B column permutation across a pair of tiles.
    auto fixPair = [&](Value &r0, Value &r1) {
      const int64_t lo[] = {0,  1,  2,  3,  4,  5,  6,  7,
                            16, 17, 18, 19, 20, 21, 22, 23};
      const int64_t hi[] = {8,  9,  10, 11, 12, 13, 14, 15,
                            24, 25, 26, 27, 28, 29, 30, 31};
      auto a0 = vector::ShuffleOp::create(rewriter, loc, rowTy, r0, r1, lo);
      auto a1 = vector::ShuffleOp::create(rewriter, loc, rowTy, r0, r1, hi);
      const int64_t mask[] = {0, 1, 2, 3, 8,  9,  10, 11,
                              4, 5, 6, 7, 12, 13, 14, 15};
      r0 = vector::ShuffleOp::create(rewriter, loc, rowTy, a0, a0, mask);
      r1 = vector::ShuffleOp::create(rewriter, loc, rowTy, a1, a1, mask);
    };

    // Undoes the i8/f8 B column permutation across a quad of tiles.
    auto fixQuad = [&](MutableArrayRef<Value> r) {
      const int64_t lo[] = {0,  1,  2,  3,  4,  5,  6,  7,
                            16, 17, 18, 19, 20, 21, 22, 23};
      const int64_t hi[] = {8,  9,  10, 11, 12, 13, 14, 15,
                            24, 25, 26, 27, 28, 29, 30, 31};
      auto a0 = vector::ShuffleOp::create(rewriter, loc, rowTy, r[0], r[1], lo);
      auto a1 = vector::ShuffleOp::create(rewriter, loc, rowTy, r[0], r[1], hi);
      auto a2 = vector::ShuffleOp::create(rewriter, loc, rowTy, r[2], r[3], lo);
      auto a3 = vector::ShuffleOp::create(rewriter, loc, rowTy, r[2], r[3], hi);
      const int64_t even[] = {0,  1,  2,  3,  8,  9,  10, 11,
                              16, 17, 18, 19, 24, 25, 26, 27};
      const int64_t odd[] = {4,  5,  6,  7,  12, 13, 14, 15,
                             20, 21, 22, 23, 28, 29, 30, 31};
      r[0] = vector::ShuffleOp::create(rewriter, loc, rowTy, a0, a2, even);
      r[1] = vector::ShuffleOp::create(rewriter, loc, rowTy, a0, a2, odd);
      r[2] = vector::ShuffleOp::create(rewriter, loc, rowTy, a1, a3, even);
      r[3] = vector::ShuffleOp::create(rewriter, loc, rowTy, a1, a3, odd);
    };

    // Process up to 4 tiles of the same tile-row per loop over the 16 rows.
    for (unsigned tileRow = 0; tileRow < M / 16; tileRow++) {
      for (unsigned col = 0; col < numCols; col += 4) {
        unsigned start = tileRow * numCols + col;
        unsigned count = std::min(4u, numCols - col);

        scf::ForOp::create(
            rewriter, loc, c0, c16, one, ValueRange{},
            [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                ValueRange iterArgs) {
              Value rowIdx = arith::IndexCastOp::create(
                  rewriter, loc, rewriter.getI32Type(), iv);

              SmallVector<Value> rows;
              for (unsigned t = 0; t < count; t++) {
                auto mov = amx::TileMovRowOp::create(
                    rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
                    c16I16, c64I16, dps[start + t], rowIdx);
                rows.push_back(
                    vector::BitCastOp::create(rewriter, loc, rowTy, mov));
              }

              if (permutedB && ipType.isBF16()) {
                for (unsigned t = 0; t < count; t += 2)
                  fixPair(rows[t], rows[t + 1]);
              } else if (permutedB && col + 4 <= fullQuadCols) {
                fixQuad(rows);
              }

              indicesAcc[indicesAcc.size() - 2] = iv;
              indicesAcc[indicesAcc.size() - 1] = c0;

              for (unsigned t = 0; t < count; t++) {
                Value valueC = vector::LoadOp::create(
                    rewriter, loc, rowTy, subviews[start + t], indicesAcc);
                Value sum =
                    ipType.isSignlessInteger(8)
                        ? arith::AddIOp::create(rewriter, loc, rows[t], valueC)
                              .getResult()
                        : arith::AddFOp::create(rewriter, loc, rows[t], valueC)
                              .getResult();
                vector::StoreOp::create(rewriter, loc, sum, allocas[start + t],
                                        ValueRange{iv, c0});
              }

              scf::YieldOp::create(nestedBuilder, loc);
            });

        auto vectorType = mlir::VectorType::get({16, 16}, opType);
        Value padding = ub::PoisonOp::create(rewriter, loc, opType);
        auto map = AffineMap::getMinorIdentityMap(
            bufferType.getRank(), vectorType.getRank(), rewriter.getContext());
        SmallVector<bool> inBounds(vectorType.getRank(), true);

        for (unsigned t = start; t < start + count; t++) {
          auto vec = vector::TransferReadOp::create(
              rewriter, loc, vectorType, allocas[t], ValueRange{c0, c0},
              padding, map, inBounds);

          Value resultWrite = contractionUsersAfterYield(ops[t].getResult());
          if (auto vecType =
                  llvm::dyn_cast<VectorType>(resultWrite.getType())) {
            auto vecRow =
                mlir::vector::ShapeCastOp::create(rewriter, loc, vecType, vec);
            rewriter.replaceAllUsesWith(resultWrite, vecRow);
          }
        }
      }
    }
    return success();
  }
};
} // namespace

void x86::populateVectorContractToACEOuterProductPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractToACEOuterProduct>(patterns.getContext());
}
