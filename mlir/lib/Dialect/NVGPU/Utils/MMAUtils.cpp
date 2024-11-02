//===- MMAUtils.cpp - MLIR NVGPU dialect utils for MMA operations----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::nvgpu;

/// There are always 4 threads per [128|256|512] bit row.
static constexpr int64_t kThreadsPerRow = 4;
static constexpr int64_t kNumRowsPerTile = 8;

static bool isAccumulatorOrResult(MatMulOperandRole operandType) {
  return operandType == MatMulOperandRole::C;
}

/// Returns the number of registers which compose a matrix fragment held by a
/// single thread.
static int64_t inferNumRegistersPerMatrixFragment(const WarpMatrixInfo &type) {
  int64_t lineSize = inferTileWidthInBits(type);
  auto shape = type.vectorType.getShape();
  return (shape[0] / kNumRowsPerTile) *
         (shape[1] * type.vectorType.getElementType().getIntOrFloatBitWidth()) /
         lineSize;
}

/// Returns the number of 8 x [128|256|512] bit tiles that compose the given
/// operand shape.
static std::array<int64_t, 2> getTileShape(ArrayRef<int64_t> operandShape,
                                           Type elementType,
                                           int64_t lineSizeBits) {
  // For each 8x128bit square, a thread is responsible for one 32bit register.
  return {operandShape[0] / kNumRowsPerTile,
          (operandShape[1] * elementType.getIntOrFloatBitWidth()) /
              lineSizeBits};
}

/// Returns the first user of the `op` that is vector.contract. If no
/// vector.contract user exists, return failure.
FailureOr<vector::ContractionOp> nvgpu::getUserContract(Operation *op) {
  for (Operation *user : op->getUsers()) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(user))
      return contractOp;
  }
  return failure();
}

FailureOr<WarpMatrixInfo> nvgpu::getWarpMatrixInfo(Operation *op) {
  WarpMatrixInfo info;

  // Determine the vector type at warp-level.
  if (vector::TransferWriteOp writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    info.vectorType = writeOp.getVectorType();
  } else if (isa<vector::TransferReadOp, vector::ContractionOp,
                 vector::ExtractStridedSliceOp, arith::ConstantOp>(op)) {
    info.vectorType = op->getResult(0).getType().cast<VectorType>();
  } else {
    return op->emitError()
           << "unhandled operation type in nvgpu.mma.sync conversion path";
  }

  // Determine the operand role. We assume it is an accumulator/result unless it
  // is directly consumed by a `vector.contract` op.
  info.operandRole = MatMulOperandRole::C;
  FailureOr<vector::ContractionOp> contractOp = getUserContract(op);
  if (failed(contractOp))
    return info;

  if ((*contractOp).getLhs() == op->getResult(0))
    info.operandRole = MatMulOperandRole::A;
  else if ((*contractOp).getRhs() == op->getResult(0))
    info.operandRole = MatMulOperandRole::B;

  return info;
}

int64_t nvgpu::inferTileWidthInBits(const WarpMatrixInfo &type) {
  bool isAcc = isAccumulatorOrResult(type.operandRole);
  Type elType = type.vectorType.getElementType();
  if (isAcc && elType.getIntOrFloatBitWidth() == 32) {
    return 256;
  }
  if (elType.getIntOrFloatBitWidth() == 64) {
    return isAcc ? 512 : 256;
  }
  return 128;
}

FailureOr<FragmentElementInfo>
nvgpu::getMmaSyncRegisterType(const WarpMatrixInfo &type) {
  MLIRContext *ctx = type.vectorType.getContext();
  const bool isAccum = isAccumulatorOrResult(type.operandRole);

  Type elType = type.vectorType.getElementType();
  if (elType.isF16()) {
    return FragmentElementInfo{
        LLVM::getFixedVectorType(Float16Type::get(ctx), 2), 2, 32,
        inferNumRegistersPerMatrixFragment(type)};
  }

  // f64 operand
  Type f64Ty = Float64Type::get(ctx);
  if (elType.isF64()) {
    return isAccum
               ? FragmentElementInfo{LLVM::getFixedVectorType(f64Ty, 2), 2, 128,
                                     inferNumRegistersPerMatrixFragment(type)}
               : FragmentElementInfo{f64Ty, 1, 64,
                                     inferNumRegistersPerMatrixFragment(type)};
  }

  // int8 operand
  if (elType.isInteger(8)) {
    return FragmentElementInfo{
        LLVM::getFixedVectorType(IntegerType::get(ctx, 8), 4), 4, 32,
        inferNumRegistersPerMatrixFragment(type)};
  }

  // int4 operand
  if (elType.isInteger(4)) {
    return FragmentElementInfo{
        LLVM::getFixedVectorType(IntegerType::get(ctx, 4), 8), 8, 32,
        inferNumRegistersPerMatrixFragment(type)};
  }

  // Integer 32bit acc operands
  if (elType.isInteger(32)) {
    return FragmentElementInfo{
        LLVM::getFixedVectorType(IntegerType::get(ctx, 32), 2), 2, 64,
        inferNumRegistersPerMatrixFragment(type)};
  }

  // Floating point 32bit operands
  if (elType.isF32()) {
    Type f32Ty = Float32Type::get(ctx);
    return isAccum
               ? FragmentElementInfo{LLVM::getFixedVectorType(f32Ty, 2), 2, 64,
                                     inferNumRegistersPerMatrixFragment(type)}
               : FragmentElementInfo{f32Ty, 1, 32,
                                     inferNumRegistersPerMatrixFragment(type)};
  }
  return failure();
}

static AffineMap getRegisterIndexToTileOffsetMap(int64_t lineSize,
                                                 Type elementType,
                                                 ArrayRef<int64_t> operandShape,
                                                 bool isAccumulator,
                                                 int64_t elementsPerRegister,
                                                 AffineExpr logicalValueId) {
  const int64_t elementsPerLine =
      lineSize / elementType.getIntOrFloatBitWidth();
  const std::array<int64_t, 2> num8x128bTiles =
      getTileShape(operandShape, elementType, lineSize);
  AffineExpr registerIdx = logicalValueId.floorDiv(elementsPerRegister);
  return AffineMap::get(
      2, 0,
      {(registerIdx % num8x128bTiles[0]) * 8,
       (registerIdx.floorDiv(num8x128bTiles[0])) * elementsPerLine},
      elementType.getContext());
}

FailureOr<AffineMap>
nvgpu::getLaneIdAndValueIdToOperandCoord(Location loc, OpBuilder &builder,
                                         const WarpMatrixInfo &fragmentType) {
  Type elementType = fragmentType.vectorType.getElementType();
  ArrayRef<int64_t> operandShape = fragmentType.vectorType.getShape();
  FailureOr<nvgpu::FragmentElementInfo> regInfo =
      getMmaSyncRegisterType(fragmentType);
  if (failed(regInfo))
    return failure();

  const int64_t elementBitWidth = elementType.getIntOrFloatBitWidth();
  const int64_t elementsPerRegister =
      regInfo->registerWidthBits / elementBitWidth;
  const int64_t lineSize = inferTileWidthInBits(fragmentType);

  AffineExpr laneId, logicalValueIdDim;
  bindDims(builder.getContext(), laneId, logicalValueIdDim);

  // Determine what register logicalValueId corresponds to. Use that as a
  // linear index into the coordinate mapping `index -> (tile row, tile col)`.
  AffineMap registerIndexToTileCoord = getRegisterIndexToTileOffsetMap(
      lineSize, elementType, operandShape,
      isAccumulatorOrResult(fragmentType.operandRole), elementsPerRegister,
      logicalValueIdDim);

  auto makeMap = [&](ArrayRef<AffineExpr> dimExprs) -> AffineMap {
    return AffineMap::get(2, 0, dimExprs, builder.getContext());
  };

  auto tileRow = registerIndexToTileCoord.getResult(0);
  auto tileCol = registerIndexToTileCoord.getResult(1);
  return makeMap({tileRow + laneId.floorDiv(kThreadsPerRow),
                  tileCol + (laneId % kThreadsPerRow) * elementsPerRegister +
                      (logicalValueIdDim % elementsPerRegister)});
}

FailureOr<nvgpu::LdMatrixParams>
nvgpu::getLdMatrixParams(const WarpMatrixInfo &type, bool transpose) {
  LdMatrixParams params;
  Type elType = type.vectorType.getElementType();
  params.fragmentType = type.vectorType;
  if (type.operandRole == MatMulOperandRole::A ||
      type.operandRole == MatMulOperandRole::C) {
    params.targetLayout = NVVM::MMALayout::row;
  } else {
    params.targetLayout = NVVM::MMALayout::col;
  }
  ArrayRef<int64_t> shape = type.vectorType.getShape();
  params.contiguousDimType = transpose ? vector::IteratorType::parallel
                                       : vector::IteratorType::reduction;

  if (params.contiguousDimType == vector::IteratorType::reduction) {
    params.numTiles = (shape[0] / kNumRowsPerTile) *
                      ((shape[1] * elType.getIntOrFloatBitWidth()) / 128);
  } else {
    params.numTiles = (shape[1] / kNumRowsPerTile) *
                      ((shape[0] * elType.getIntOrFloatBitWidth()) / 128);
  }

  if (params.numTiles == 0)
    return failure();

  return params;
}

FailureOr<AffineMap>
nvgpu::getLaneIdToLdMatrixMatrixCoord(Location loc, OpBuilder &builder,
                                      const LdMatrixParams &params) {
  // One thread per 128b row.
  const int bitsPerElement = static_cast<int>(
      params.fragmentType.getElementType().getIntOrFloatBitWidth());
  const int kElementsPer128b = (128 / bitsPerElement);
  ArrayRef<int64_t> operandShape = params.fragmentType.getShape();
  AffineExpr d0 = getAffineDimExpr(0, builder.getContext());

  auto makeMap = [&](ArrayRef<AffineExpr> dimExprs) -> AffineMap {
    return AffineMap::get(1, 0, dimExprs, builder.getContext());
  };

  // Index `idx` in vectorType `operandShape` maps to the strided dimension of
  // the `srcMemref` memory of the LdMatrixOp.
  int idx =
      (params.contiguousDimType == vector::IteratorType::reduction) ? 0 : 1;

  // Affine expr in strided and contiguous dimension encodes the coordinate
  // mapping for the element a thread points to for warp-wide LdMatrixOp.
  AffineExpr strided = d0 % (operandShape[idx]);
  AffineExpr contiguous = d0.floorDiv(operandShape[idx]) * (kElementsPer128b);

  // This case corresponds to row-major matrixA or col-major matrixB or
  // row-major matrixC. This is when the memory layout in `srcMemref`
  // match mma.sync hardware vector register operand layout.
  if (params.contiguousDimType == vector::IteratorType::reduction)
    return makeMap({strided, contiguous});

  // This case corresponds to col-major matrixA or row-major matrixB or
  // col-major matrixC. This is when the memory layout in `srcMemref` does not
  // match mma.sync hardware vector register operand layout.
  if (params.contiguousDimType == vector::IteratorType::parallel)
    return makeMap({contiguous, strided});

  return failure();
}

LogicalResult nvgpu::PrepareContractToGPUMMASync::matchAndRewrite(
    vector::ContractionOp op, PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value res = op.getAcc();

  // Set up the parallel/reduction structure in right form.
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m;
  AffineExpr n;
  AffineExpr k;
  bindDims(rewriter.getContext(), m, n, k);
  static constexpr std::array<int64_t, 2> perm = {1, 0};
  auto iteratorTypes = op.getIteratorTypes().getValue();
  SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
  if (iteratorTypes.size() != 3)
    return failure();
  if (!(vector::isParallelIterator(iteratorTypes[0]) &&
        vector::isParallelIterator(iteratorTypes[1]) &&
        vector::isReductionIterator(iteratorTypes[2])))
    return failure();

  // The canonical form is "TNT" = A row-major, B col-major, C row-major.
  const auto canonicalForm = infer({{m, k}, {n, k}, {m, n}});
  if (maps == canonicalForm) {
    return failure();
  }
  if (maps == infer({{m, k}, {k, n}, {m, n}})) {
    rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
  } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
    lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
  } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
    rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
  } else if (maps == infer({{k, m}, {k, n}, {n, m}})) {
    std::swap(rhs, lhs);
    rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
  } else if (maps == infer({{k, m}, {n, k}, {n, m}})) {
    std::swap(rhs, lhs);
    rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
  } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
    std::swap(lhs, rhs);
    lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
  } else if (maps == infer({{m, k}, {n, k}, {n, m}})) {
    std::swap(lhs, rhs);
  } else {
    return failure();
  }
  rewriter.replaceOpWithNewOp<vector::ContractionOp>(
      op, lhs, rhs, res, rewriter.getAffineMapArrayAttr(canonicalForm),
      op.getIteratorTypes());
  return success();
}
