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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/DebugLog.h"

#include <numeric>

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOAMX
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "vector-to-amx"

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
    if (shape.back() != vnniFactor) {
      LDBG() << "invalid VNNI packing factor";
      return false;
    }
    cols *= vnniFactor;
  }

  // AMX tile supports up to 16 rows of 64 bytes each.
  constexpr unsigned maxRows = 16;
  constexpr unsigned maxBitsPerRow = 64 * 8;
  return rows <= maxRows && (cols * elemBitWidth) <= maxBitsPerRow;
}

/// Check if contraction operands are in AMX-compatible packed VNNI layout.
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

/// Collapse the two innermost dimensions together.
static TypedValue<MemRefType> collapseLastDim(PatternRewriter &rewriter,
                                              TypedValue<MemRefType> memref) {
  int64_t rank = memref.getType().getRank();
  SmallVector<ReassociationIndices> reassocIndices;
  for (auto i : llvm::seq<int64_t>(0, rank - 2))
    reassocIndices.push_back({i});
  reassocIndices.push_back({rank - 2, rank - 1});
  return memref::CollapseShapeOp::create(rewriter, memref.getLoc(), memref,
                                         reassocIndices);
}

/// Attempt to create an AMX tile load/store operation equivalent to the given
/// vector transfer `xfer` op.
/// This approach allows to skip longer route through registers and a temporary
/// buffer otherwise required to move data to/from an AMX tile.
static Operation *
loadStoreFromTransfer(PatternRewriter &rewriter,
                      VectorTransferOpInterface xferOp, bool isPacked,
                      TypedValue<amx::TileType> tileToStore = nullptr) {
  if (!xferOp || !isa<vector::TransferReadOp, vector::TransferWriteOp>(xferOp))
    return nullptr;
  if (xferOp.hasOutOfBoundsDim() ||
      !xferOp.getPermutationMap().isMinorIdentity())
    return nullptr;

  // Extra checks in case of a write op.
  // Stores must not be packed.
  if (isa<vector::TransferWriteOp>(xferOp) &&
      (!tileToStore || isPacked ||
       tileToStore.getType().getShape() != xferOp.getVectorType().getShape()))
    return nullptr;

  // Check for a memref source buffer.
  // AMX data transfer requires at least 2D shape to correctly
  // infer stride between rows.
  Value base = xferOp.getBase();
  auto memTy = dyn_cast<MemRefType>(base.getType());
  int64_t memRank = memTy.getRank();
  if (!memTy || memRank < 2)
    return nullptr;

  // Check that the source buffer has enough contiguous elements to load whole
  // AMX tile row.
  //
  // To ensure correctness, the validation is conservative and expects the
  // buffer's innermost dimensions to be statically known, equal to or larger
  // than the vector row length, and equal to the VNNI dimension if applicable.
  //
  // This check could be relaxed to accept more arbitrarily shaped buffers as
  // long as there are enough contiguous elements to load a whole row.
  if (!memTy.areTrailingDimsContiguous(isPacked ? 2 : 1))
    return nullptr;
  VectorType vecTy = xferOp.getVectorType();
  ArrayRef<int64_t> vecShape = vecTy.getShape();
  ArrayRef<int64_t> memShape = memTy.getShape();
  if (memShape.back() == ShapedType::kDynamic ||
      memShape.back() < vecShape.back())
    return nullptr;
  if (isPacked &&
      (memShape.back() != vecShape.back() ||
       memShape[memShape.size() - 2] == ShapedType::kDynamic ||
       memShape[memShape.size() - 2] < vecShape[vecShape.size() - 2]))
    return nullptr;

  // Load values directly from the buffer to an AMX tile.
  PatternRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(xferOp);
  Location loc = xferOp.getLoc();

  // Create a subview of the source buffer based on the transfer op to resolve
  // offsets.
  SmallVector<OpFoldResult> strides(memRank, rewriter.getIndexAttr(1));
  int64_t vecRank = vecTy.getRank();
  assert(memRank >= vecRank &&
         "Expects buffer to be the same or greater rank than vector");
  SmallVector<int64_t> shape(memRank - vecRank, 1);
  shape.append(vecShape.begin(), vecShape.end());
  TypedValue<MemRefType> src =
      memref::SubViewOp::create(
          rewriter, loc, base, getAsOpFoldResult(xferOp.getIndices()),
          getAsOpFoldResult(rewriter.getI64ArrayAttr(shape)), strides)
          .getResult();

  // Collapse the VNNI dimension in case of packing.
  if (isPacked)
    src = collapseLastDim(rewriter, src);
  int64_t rows = vecShape[0];
  int64_t cols = std::accumulate(vecShape.begin() + 1, vecShape.end(), 1,
                                 std::multiplies<int64_t>());
  auto tileType = amx::TileType::get({rows, cols}, vecTy.getElementType());

  Value zeroIndex = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> tileIndicides(src.getType().getRank(), zeroIndex);

  Operation *amxTileOp = nullptr;
  if (isa<vector::TransferReadOp>(xferOp)) {
    amxTileOp =
        amx::TileLoadOp::create(rewriter, loc, tileType, src, tileIndicides);
  } else if (isa<vector::TransferWriteOp>(xferOp)) {
    amxTileOp = amx::TileStoreOp::create(rewriter, loc, src, tileIndicides,
                                         tileToStore);
  } else {
    llvm_unreachable("unsupported vector transfer op");
  }

  return amxTileOp;
}

/// Attempt to create an AMX tile load operation equivalent to the given
/// vector transfer `readOp`.
/// Returns loaded AMX tile if successful.
static FailureOr<TypedValue<amx::TileType>>
loadFromTransfer(PatternRewriter &rewriter, vector::TransferReadOp readOp,
                 bool isPacked) {
  amx::TileLoadOp loadOp = dyn_cast_if_present<amx::TileLoadOp>(
      loadStoreFromTransfer(rewriter, readOp, isPacked));
  if (!loadOp)
    return failure();
  return loadOp.getRes();
}

/// Attempt to create an AMX tile store operation equivalent to the given
/// vector transfer `writeOp`.
static LogicalResult storeFromTransfer(PatternRewriter &rewriter,
                                       vector::TransferWriteOp writeOp,
                                       TypedValue<amx::TileType> tileToStore) {
  return success(loadStoreFromTransfer(rewriter, writeOp, /*isPacked=*/false,
                                       tileToStore));
}

/// Load vector values to an AMX tile.
static TypedValue<amx::TileType> loadTile(PatternRewriter &rewriter,
                                          TypedValue<VectorType> vec) {
  Location loc = vec.getLoc();

  VectorType vecTy = vec.getType();
  bool isPacked = vecTy.getRank() == 3;

  // Try to load tile directly from vector producer's buffer.
  auto readOp = vec.getDefiningOp<vector::TransferReadOp>();
  FailureOr<TypedValue<amx::TileType>> tile =
      loadFromTransfer(rewriter, readOp, isPacked);
  if (succeeded(tile))
    return *tile;

  // Transfer the vector to a tile through an intermediate buffer.
  Value buf = memref::AllocaOp::create(
      rewriter, loc, MemRefType::get(vecTy.getShape(), vecTy.getElementType()));
  Value zeroIndex = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(vecTy.getRank(), zeroIndex);
  vector::TransferWriteOp::create(rewriter, loc, vec, buf, indices);

  // Collapse the VNNI dimension in case of packing.
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

/// Store an AMX tile in a vector.
static TypedValue<VectorType> storeTile(PatternRewriter &rewriter,
                                        TypedValue<amx::TileType> tile) {
  Location loc = tile.getLoc();

  // Transfer the tile to a vector through an intermediate buffer.
  amx::TileType tileTy = tile.getType();
  Value buf = memref::AllocaOp::create(
      rewriter, loc,
      MemRefType::get(tileTy.getShape(), tileTy.getElementType()));
  Value zeroIndex = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
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

    // If the contraction result is only written back to memory, try to replace
    // the vector op with an AMX store directly.
    Value res = contractOp.getResult();
    if (res.hasOneUse()) {
      auto writeOp = dyn_cast<vector::TransferWriteOp>(*res.getUsers().begin());
      LogicalResult storeRes = storeFromTransfer(rewriter, writeOp, tileMul);
      if (succeeded(storeRes)) {
        rewriter.eraseOp(writeOp);
        rewriter.eraseOp(contractOp);
        return success();
      }
    }

    // Load the result back into a vector.
    Value newResult = storeTile(rewriter, tileMul);
    rewriter.replaceOp(contractOp, newResult);

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
