//===- VectorToAMX.cpp - Convert vector to AMX dialect ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToAMX/VectorToAMX.h"

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOAMX
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Return true if vector shape is compatible with AMX tiles.
/// The validation accounts for VNNI packing.
static bool verifyAmxShape(VectorType vec) {
  // Check overall shape:
  //   - 2D for plain layout input or output
  //   - 3D for VNNI packed input
  if (vec.getRank() != 2 && vec.getRank() != 3)
    return false;

  ArrayRef<int64_t> shape = vec.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  unsigned elemBitWidth = vec.getElementType().getIntOrFloatBitWidth();

  // 3D shape indicates VNNI packed layout.
  if (vec.getRank() == 3) {
    int64_t vnniFactor = 32 / elemBitWidth;
    if (shape.back() != vnniFactor)
      return false;
    cols *= vnniFactor;
  }

  // AMX tile supports up to 16 rows of 64 bytes each.
  constexpr unsigned maxRows = 16;
  constexpr unsigned maxBitsPerRow = 64 * 8;
  return rows <= maxRows && (cols * elemBitWidth) <= maxBitsPerRow;
}

/// Checks if contraction operands are in AMX-compatible packed VNNI layout.
static LogicalResult isAmxVnniLayout(PatternRewriter &rewriter,
                                     vector::ContractionOp contractOp) {
  VectorType accType = dyn_cast<VectorType>(contractOp.getAcc().getType());
  if (!accType || accType.getRank() != 2)
    return rewriter.notifyMatchFailure(contractOp, "Expects acc 2D vector");

  // Expect 3D inputs for VNNI packed data.
  VectorType lhsType = contractOp.getLhs().getType();
  VectorType rhsType = contractOp.getRhs().getType();
  if (lhsType.getRank() != 3 || rhsType.getRank() != 3)
    return rewriter.notifyMatchFailure(contractOp,
                                       "Expects lhs and rhs 3D vectors");

  // Check if shapes are compatible with AMX tile.
  if (!verifyAmxShape(lhsType) || !verifyAmxShape(rhsType) ||
      !verifyAmxShape(accType))
    return rewriter.notifyMatchFailure(contractOp, "Invalid operand shape");

  // Validate affine maps.
  //
  // Iterators can be ordered arbitrarily. Indexing map positions are based on
  // operands' target shapes.
  // The matrix layouts must match the following:
  //   - matrix A - [M]x[K/vnniFactor]x[vnniFactor]
  //   - matrix B - [K/vnniFactor]x[N]x[vnniFactor]
  //   - matrix C - [M]x[N]
  SmallVector<AffineMap, 4> indexingMaps = contractOp.getIndexingMapsArray();
  AffineMap mapA = indexingMaps[0];
  AffineMap mapB = indexingMaps[1];
  if (mapA.getNumInputs() != 4 || mapA.getNumResults() != 3 ||
      mapB.getNumResults() != 3)
    return rewriter.notifyMatchFailure(contractOp,
                                       "Invalid input indexing maps");
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(indexingMaps);
  if (failed(dims))
    return rewriter.notifyMatchFailure(contractOp,
                                       "Failed to infer contraction dims");
  // Two reduction dimensions are expected:
  //   - one for the K dimension
  //   - one for the VNNI factor
  if (dims->k.size() != 2)
    return rewriter.notifyMatchFailure(contractOp,
                                       "Expected two reduction dims");
  assert(dims->m.size() == 1 && dims->n.size() == 1 &&
         "Invalid parallel contraction dims");

  SmallVector<vector::IteratorType> iteratorTypes =
      contractOp.getIteratorTypesArray();
  // Check VNNI dim maps - the innermost dim for A and B inputs.
  auto vnniDimA = dyn_cast<AffineDimExpr>(mapA.getResult(2));
  auto vnniDimB = dyn_cast<AffineDimExpr>(mapB.getResult(2));
  if (!vnniDimA || !vnniDimB || vnniDimA != vnniDimB ||
      iteratorTypes[vnniDimA.getPosition()] != vector::IteratorType::reduction)
    return rewriter.notifyMatchFailure(contractOp, "Invalid VNNI dim map");
  // Check K dim maps - non-transposed row-major layout.
  auto redDimA = dyn_cast<AffineDimExpr>(mapA.getResult(1));
  auto redDimB = dyn_cast<AffineDimExpr>(mapB.getResult(0));
  if (!redDimA || !redDimB || redDimA != redDimB ||
      iteratorTypes[redDimA.getPosition()] != vector::IteratorType::reduction)
    return rewriter.notifyMatchFailure(contractOp, "Invalid K dim map");
  // Check M and N dim maps - map to non-transposed output.
  AffineMap mapC = indexingMaps[2];
  auto mDimC = dyn_cast<AffineDimExpr>(mapC.getResult(0));
  auto nDimC = dyn_cast<AffineDimExpr>(mapC.getResult(1));
  if (!mDimC || !nDimC)
    return rewriter.notifyMatchFailure(contractOp, "Invalid acc maps");
  auto parallelDimA = dyn_cast<AffineDimExpr>(mapA.getResult(0));
  if (!parallelDimA ||
      iteratorTypes[parallelDimA.getPosition()] !=
          vector::IteratorType::parallel ||
      parallelDimA != mDimC)
    return rewriter.notifyMatchFailure(contractOp, "Invalid M dim map");
  auto parallelDimB = dyn_cast<AffineDimExpr>(mapB.getResult(1));
  if (!parallelDimB ||
      iteratorTypes[parallelDimB.getPosition()] !=
          vector::IteratorType::parallel ||
      parallelDimB != nDimC)
    return rewriter.notifyMatchFailure(contractOp, "Invalid N dim map");

  return success();
}

/// Validate contraction operands for AMX lowering.
static LogicalResult validateOperands(PatternRewriter &rewriter,
                                      vector::ContractionOp contractOp) {
  VectorType accType = dyn_cast<VectorType>(contractOp.getAcc().getType());
  if (!accType)
    return rewriter.notifyMatchFailure(contractOp, "Expects vector acc");

  // Check if operand types are compatible with AMX compute ops.
  bool validElemTypes = false;
  Type lhsElemType = contractOp.getLhs().getType().getElementType();
  Type rhsElemType = contractOp.getRhs().getType().getElementType();
  Type accElemType = accType.getElementType();
  if (accElemType.isInteger(32)) {
    validElemTypes = lhsElemType.isInteger(8) && rhsElemType.isInteger(8);
  } else if (accElemType.isF32()) {
    validElemTypes = (lhsElemType.isF16() && rhsElemType.isF16()) ||
                     (lhsElemType.isBF16() && rhsElemType.isBF16());
  }
  if (!validElemTypes)
    return rewriter.notifyMatchFailure(contractOp,
                                       "Invalid combination of operand types");

  if (failed(isAmxVnniLayout(rewriter, contractOp)))
    return failure();

  return success();
}

/// Collapses the two innermost dimensions together.
static Value collapseLastDim(PatternRewriter &rewriter,
                             TypedValue<MemRefType> memref) {
  int64_t rank = memref.getType().getRank();
  SmallVector<ReassociationIndices> reassocIndices;
  for (auto i : llvm::seq<int64_t>(0, rank - 2))
    reassocIndices.push_back({i});
  reassocIndices.push_back({rank - 2, rank - 1});
  return memref::CollapseShapeOp::create(rewriter, memref.getLoc(), memref,
                                         reassocIndices);
}

/// Loads vector values to an AMX tile.
static TypedValue<amx::TileType> loadTile(PatternRewriter &rewriter,
                                          TypedValue<VectorType> vec) {
  Location loc = vec.getLoc();
  Value zeroIndex = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);

  // Transfer the vector to a tile through an intermediate buffer.
  VectorType vecTy = vec.getType();
  Value buf = memref::AllocaOp::create(
      rewriter, loc, MemRefType::get(vecTy.getShape(), vecTy.getElementType()));
  SmallVector<Value> indices(vecTy.getRank(), zeroIndex);
  vector::TransferWriteOp::create(rewriter, loc, vec, buf, indices);

  // Collapse the VNNI dimension in case of packing.
  bool isPacked = vecTy.getRank() == 3;
  if (isPacked)
    buf = collapseLastDim(rewriter, cast<TypedValue<MemRefType>>(buf));

  ArrayRef<int64_t> shape = vecTy.getShape();
  int64_t rows = shape[0];
  int64_t cols = std::accumulate(shape.begin() + 1, shape.end(), 1,
                                 std::multiplies<int64_t>());
  auto tileType = amx::TileType::get({rows, cols}, vecTy.getElementType());

  return amx::TileLoadOp::create(rewriter, loc, tileType, buf,
                                 {zeroIndex, zeroIndex});
}

/// Stores an AMX tile in a vector.
static TypedValue<VectorType> storeTile(PatternRewriter &rewriter,
                                        TypedValue<amx::TileType> tile) {
  Location loc = tile.getLoc();
  Value zeroIndex = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);

  // Transfer the tile to a vector through an intermediate buffer.
  amx::TileType tileTy = tile.getType();
  Value buf = memref::AllocaOp::create(
      rewriter, loc,
      MemRefType::get(tileTy.getShape(), tileTy.getElementType()));
  SmallVector<Value> indices(2, zeroIndex);
  amx::TileStoreOp::create(rewriter, loc, buf, indices, tile);

  auto vecTy = VectorType::get(tileTy.getShape(), tileTy.getElementType());
  return vector::TransferReadOp::create(rewriter, loc, vecTy, buf, indices, {});
}

struct ContractionToAMX : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    Location loc = contractOp.getLoc();

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind");
    if (failed(validateOperands(rewriter, contractOp)))
      return failure();

    TypedValue<amx::TileType> lhsTile = loadTile(rewriter, contractOp.getLhs());
    TypedValue<amx::TileType> rhsTile = loadTile(rewriter, contractOp.getRhs());
    auto acc = dyn_cast<TypedValue<VectorType>>(contractOp.getAcc());
    assert(acc && "Invalid accumulator type");
    TypedValue<amx::TileType> accTile = loadTile(rewriter, acc);

    TypedValue<amx::TileType> tileMul;
    if (acc.getType().getElementType().isFloat()) {
      tileMul = amx::TileMulFOp::create(rewriter, loc, accTile.getType(),
                                        lhsTile, rhsTile, accTile);
    } else {
      tileMul = amx::TileMulIOp::create(rewriter, loc, accTile.getType(),
                                        lhsTile, rhsTile, accTile);
    }

    Value res = storeTile(rewriter, tileMul);
    rewriter.replaceOp(contractOp, res);

    return success();
  }
};

struct ConvertVectorToAMXPass
    : public impl::ConvertVectorToAMXBase<ConvertVectorToAMXPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    populateVectorToAMXConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::populateVectorToAMXConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ContractionToAMX>(patterns.getContext());
}
