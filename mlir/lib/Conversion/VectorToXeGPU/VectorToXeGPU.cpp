//===- VectorToXeGPU.cpp - Convert vector to XeGPU dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector operations to XeGPU dialect ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToXeGPU/VectorToXeGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOXEGPU
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// Return true if value represents a zero constant.
static bool isZeroConstant(Value val) {
  auto constant = val.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;

  return TypeSwitch<Attribute, bool>(constant.getValue())
      .Case<FloatAttr>(
          [](auto floatAttr) { return floatAttr.getValue().isZero(); })
      .Case<IntegerAttr>(
          [](auto intAttr) { return intAttr.getValue().isZero(); })
      .Default([](auto) { return false; });
}

static LogicalResult storeLoadPreconditions(PatternRewriter &rewriter,
                                            Operation *op, VectorType vecTy) {
  // Validate only vector as the basic vector store and load ops guarantee
  // XeGPU-compatible memref source.
  unsigned vecRank = vecTy.getRank();
  if (!(vecRank == 1 || vecRank == 2))
    return rewriter.notifyMatchFailure(op, "Expects 1D or 2D vector");

  return success();
}

static LogicalResult transferPreconditions(PatternRewriter &rewriter,
                                           VectorTransferOpInterface xferOp) {
  if (xferOp.getMask())
    return rewriter.notifyMatchFailure(xferOp,
                                       "Masked transfer is not supported");

  auto srcTy = dyn_cast<MemRefType>(xferOp.getShapedType());
  if (!srcTy)
    return rewriter.notifyMatchFailure(xferOp, "Expects memref source");

  // Validate further transfer op semantics.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(srcTy.getStridesAndOffset(strides, offset)) || strides.back() != 1)
    return rewriter.notifyMatchFailure(
        xferOp, "Buffer must be contiguous in the innermost dimension");

  VectorType vecTy = xferOp.getVectorType();
  unsigned vecRank = vecTy.getRank();
  if (xferOp.hasOutOfBoundsDim() && vecRank < 2)
    return rewriter.notifyMatchFailure(
        xferOp, "Boundary check is available only for block instructions.");

  AffineMap map = xferOp.getPermutationMap();
  if (!map.isProjectedPermutation(/*allowZeroInResults=*/false))
    return rewriter.notifyMatchFailure(xferOp, "Unsupported permutation map");
  unsigned numInputDims = map.getNumInputs();
  for (AffineExpr expr : map.getResults().take_back(vecRank)) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (dim.getPosition() < (numInputDims - vecRank))
      return rewriter.notifyMatchFailure(
          xferOp, "Only the innermost dimensions can be accessed");
  }

  return success();
}

static xegpu::CreateNdDescOp
createNdDescriptor(PatternRewriter &rewriter, Location loc,
                   xegpu::TensorDescType descType, TypedValue<MemRefType> src,
                   Operation::operand_range offsets) {
  MemRefType srcTy = src.getType();
  auto [strides, offset] = srcTy.getStridesAndOffset();

  xegpu::CreateNdDescOp ndDesc;
  if (srcTy.hasStaticShape()) {
    ndDesc = xegpu::CreateNdDescOp::create(rewriter, loc, descType, src,
                                           getAsOpFoldResult(offsets));
  } else {
    // In case of any dynamic shapes, source's shape and strides have to be
    // explicitly provided.
    SmallVector<Value> sourceDims;
    unsigned srcRank = srcTy.getRank();
    for (unsigned i = 0; i < srcRank; ++i)
      sourceDims.push_back(memref::DimOp::create(rewriter, loc, src, i));

    SmallVector<int64_t> constOffsets;
    SmallVector<Value> dynOffsets;
    for (Value offset : offsets) {
      std::optional<int64_t> staticVal = getConstantIntValue(offset);
      if (!staticVal)
        dynOffsets.push_back(offset);
      constOffsets.push_back(staticVal.value_or(ShapedType::kDynamic));
    }

    SmallVector<Value> dynShapes;
    for (auto [idx, shape] : llvm::enumerate(srcTy.getShape())) {
      if (shape == ShapedType::kDynamic)
        dynShapes.push_back(sourceDims[idx]);
    }

    // Compute strides in reverse order.
    SmallVector<Value> dynStrides;
    Value accStride = arith::ConstantIndexOp::create(rewriter, loc, 1);
    // Last stride is guaranteed to be static and unit.
    for (int i = static_cast<int>(strides.size()) - 2; i >= 0; --i) {
      accStride =
          arith::MulIOp::create(rewriter, loc, accStride, sourceDims[i + 1]);
      if (strides[i] == ShapedType::kDynamic)
        dynStrides.push_back(accStride);
    }
    std::reverse(dynStrides.begin(), dynStrides.end());

    ndDesc = xegpu::CreateNdDescOp::create(
        rewriter, loc, descType, src, dynOffsets, dynShapes, dynStrides,
        DenseI64ArrayAttr::get(rewriter.getContext(), constOffsets),
        DenseI64ArrayAttr::get(rewriter.getContext(), srcTy.getShape()),
        DenseI64ArrayAttr::get(rewriter.getContext(), strides));
  }

  return ndDesc;
}

static LogicalResult
extraCheckForScatteredLoadStore(VectorTransferOpInterface xferOp,
                                PatternRewriter &rewriter) {
  // 1. it must be inbound access by checking in_bounds attributes, like
  // {in_bounds = [false, true]}
  if (xferOp.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(xferOp,
                                       "Out-of-bounds access is not supported "
                                       "for scatter load/store lowering");
  // 2. if the memref has static shape, its lower rank must exactly match with
  // vector shape.
  if (auto memrefType = dyn_cast<MemRefType>(xferOp.getShapedType())) {
    if (memrefType.hasStaticShape()) {
      ArrayRef<int64_t> memrefShape = memrefType.getShape();
      ArrayRef<int64_t> vectorShape = xferOp.getVectorType().getShape();
      size_t memrefRank = memrefShape.size();
      size_t vectorRank = vectorShape.size();
      if (vectorRank > memrefRank)
        return rewriter.notifyMatchFailure(
            xferOp, "Vector rank cannot exceed memref rank");
      // Compare the last vectorRank dimensions of memref with vector shape
      for (size_t i = 0; i < vectorRank; ++i) {
        if (memrefShape[memrefRank - vectorRank + i] <= vectorShape[i])
          return rewriter.notifyMatchFailure(
              xferOp, "Memref lower dimensions must match vector shape");
      }
    }
  }
  return success();
}

static LogicalResult adjustStridesForPermutation(
    Operation *op, PatternRewriter &rewriter, MemRefType memrefType,
    AffineMap permMap, VectorType vecType, SmallVectorImpl<Value> &strides) {
  unsigned vecRank;
  unsigned memrefRank = memrefType.getRank();

  if (permMap.isMinorIdentity())
    return success();
  vecRank = vecType.getRank();
  // Only adjust the last vecRank strides according to the permutation
  ArrayRef<Value> relevantStrides = ArrayRef<Value>(strides).take_back(vecRank);
  SmallVector<Value> adjustedStrides(vecRank);
  // For each output dimension in the permutation map, find which input dim it
  // refers to, and assign the corresponding stride.
  for (unsigned outIdx = 0; outIdx < vecRank; ++outIdx) {
    AffineExpr expr = permMap.getResult(outIdx);
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      return rewriter.notifyMatchFailure(op, "Unsupported permutation expr");
    }
    unsigned pos = dimExpr.getPosition();
    // Map permutation to the relevant strides (innermost dims)
    if (pos < memrefRank - vecRank) {
      return rewriter.notifyMatchFailure(op, "Permutation out of bounds");
    }
    // The stride for output dimension outIdx is the stride of input dimension
    // pos
    adjustedStrides[outIdx] = relevantStrides[pos - (memrefRank - vecRank)];
  }
  // Replace the last vecRank strides with the adjusted ones
  for (unsigned i = 0; i < vecRank; ++i)
    strides[memrefRank - vecRank + i] = adjustedStrides[i];

  return success();
}

SmallVector<Value> computeStrides(VectorTransferOpInterface xferOp,
                                  PatternRewriter &rewriter) {
  SmallVector<Value> strides;
  Value baseMemref = xferOp.getBase();
  AffineMap permMap = xferOp.getPermutationMap();
  VectorType vectorType = xferOp.getVectorType();
  MemRefType memrefType = llvm::cast<MemRefType>(baseMemref.getType());

  Location loc = xferOp.getLoc();
  if (memrefType.hasStaticShape()) {
    int64_t offset;
    SmallVector<int64_t> intStrides;
    if (failed(memrefType.getStridesAndOffset(intStrides, offset))) {
      return {};
    }
    // Wrap static strides as MLIR values
    for (int64_t s : intStrides)
      strides.push_back(arith::ConstantIndexOp::create(rewriter, loc, s));
  } else {
    // For dynamic shape memref, use memref.extract_strided_metadata to get
    // stride values
    unsigned rank = memrefType.getRank();
    Type indexType = rewriter.getIndexType();

    // Result types: [base_memref, offset, stride0, stride1, ..., strideN-1,
    // size0, size1, ..., sizeN-1]
    SmallVector<Type> resultTypes;
    resultTypes.push_back(MemRefType::get(
        {}, memrefType.getElementType())); // base memref (unranked)
    resultTypes.push_back(indexType);      // offset
    for (unsigned i = 0; i < rank; ++i) {
      resultTypes.push_back(indexType); // strides
    }
    for (unsigned i = 0; i < rank; ++i) {
      resultTypes.push_back(indexType); // sizes
    }

    auto meta = memref::ExtractStridedMetadataOp::create(
        rewriter, loc, resultTypes, baseMemref);
    strides.append(meta.getStrides().begin(), meta.getStrides().end());
  }
  // Adjust strides according to the permutation map (e.g., for transpose)
  if (failed(adjustStridesForPermutation(xferOp, rewriter, memrefType, permMap,
                                         vectorType, strides))) {
    return {};
  }
  return strides;
}

// This function compute the vectors of localOffsets for scattered load/stores.
// It is used in the lowering of vector.transfer_read/write to
// load_gather/store_scatter Example:
//   %0 = vector.transfer_read %expand_shape[%block_id_y, %c0, %c0, %c0, %c0],
//               %cst {in_bounds = [true, true, true, true]}>} :
//               memref<8x4x2x6x32xbf16>, vector<4x2x6x32xbf16>
//
//   %6 = vector.step: vector<4xindex>
//   %7 = vector.step: vector<2xindex>
//   %8 = vector.step: vector<6xindex>
//   %9 = vector.step: vector<32xindex>
//   %10 = arith.mul %6, 384
//   %11 = arith.mul %7, 192
//   %12 = arith.mul %8, 32
//   %13 = arith.mul %9, 1
//   %14 = vector.shape_cast %10: vector<4xindex> -> vector<4x1x1x1xbf16>
//   %15 = vector.shape_cast %11: vector<2xindex> -> vector<1x2x1x1xbf16>
//   %16 = vector.shape_cast %12: vector<6xindex> -> vector<1x1x6x1xbf16>
//   %17 = vector.shape_cast %13: vector<32xindex> -> vector<1x1x1x32xbf16>
//   %18 = vector.broadcast %14: vector<4x1x1x1xbf16> -> vector<4x2x6x32xindex>
//   %19 = vector.broadcast %15: vector<1x2x1x1xbf16> -> vector<4x2x6x32xindex>
//   %20 = vector.broadcast %16: vector<1x1x6x1xbf16> -> vector<4x2x6x32xindex>
//   %21 = vector.broadcast %17: vector<1x1x1x32xbf16> -> vector<4x2x6x32xindex>
//   %22 = arith.add %18, %19
//   %23 = arith.add %20, %21
//   %local_offsets = arith.add %22, %23
//   %orig_offset = %block_id_y * 4x2x6x32 // consider using affine map
//   %offsets =  orig_offset + local_offsets
static Value computeOffsets(VectorTransferOpInterface xferOp,
                            PatternRewriter &rewriter,
                            ArrayRef<Value> strides) {
  Location loc = xferOp.getLoc();
  VectorType vectorType = xferOp.getVectorType();
  SmallVector<Value> indices(xferOp.getIndices().begin(),
                             xferOp.getIndices().end());
  ArrayRef<int64_t> vectorShape = vectorType.getShape();

  // Step 1: Create vector.step operations for each dimension
  SmallVector<Value> stepVectors;
  for (int64_t dim : vectorShape) {
    auto stepType = VectorType::get({dim}, rewriter.getIndexType());
    auto stepOp = vector::StepOp::create(rewriter, loc, stepType);
    stepVectors.push_back(stepOp);
  }

  // Step 2: Multiply step vectors by corresponding strides
  size_t memrefRank = strides.size();
  size_t vectorRank = vectorShape.size();
  SmallVector<Value> strideMultiplied;
  for (size_t i = 0; i < vectorRank; ++i) {
    size_t memrefDim = memrefRank - vectorRank + i;
    Value strideValue = strides[memrefDim];
    auto mulType = llvm::cast<VectorType>(stepVectors[i].getType());
    auto bcastOp =
        vector::BroadcastOp::create(rewriter, loc, mulType, strideValue);
    auto mulOp = arith::MulIOp::create(rewriter, loc, stepVectors[i], bcastOp);
    strideMultiplied.push_back(mulOp);
  }

  // Step 3: Shape cast each multiplied vector to add singleton dimensions
  SmallVector<Value> shapeCasted;
  for (size_t i = 0; i < vectorRank; ++i) {
    SmallVector<int64_t> newShape(vectorRank, 1);
    newShape[i] = vectorShape[i];
    auto newType = VectorType::get(newShape, rewriter.getIndexType());
    auto castOp = vector::ShapeCastOp::create(rewriter, loc, newType,
                                              strideMultiplied[i]);
    shapeCasted.push_back(castOp);
  }

  // Step 4: Broadcast each shape-casted vector to full vector shape
  SmallVector<Value> broadcasted;
  auto fullIndexVectorType =
      VectorType::get(vectorShape, rewriter.getIndexType());
  for (Value shapeCastVal : shapeCasted) {
    auto broadcastOp = vector::BroadcastOp::create(
        rewriter, loc, fullIndexVectorType, shapeCastVal);
    broadcasted.push_back(broadcastOp);
  }

  // Step 5: Add all broadcasted vectors together to compute local offsets
  Value localOffsets = broadcasted[0];
  for (size_t i = 1; i < broadcasted.size(); ++i) {
    localOffsets =
        arith::AddIOp::create(rewriter, loc, localOffsets, broadcasted[i]);
  }

  // Step 6: Compute base offset from transfer read indices
  Value baseOffset = nullptr;
  if (!indices.empty()) {
    baseOffset = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (size_t i = 0; i < indices.size(); ++i) {
      Value strideVal = strides[i];
      Value offsetContrib =
          arith::MulIOp::create(rewriter, loc, indices[i], strideVal);
      baseOffset =
          arith::AddIOp::create(rewriter, loc, baseOffset, offsetContrib);
    }
    // Broadcast base offset to match vector shape
    Value bcastBase = vector::BroadcastOp::create(
        rewriter, loc, fullIndexVectorType, baseOffset);
    localOffsets =
        arith::AddIOp::create(rewriter, loc, bcastBase, localOffsets);
  }
  return localOffsets;
}

// Collapse memref shape to 1D
static Value collapseMemrefTo1D(VectorTransferOpInterface xferOp,
                                PatternRewriter &rewriter) {
  Location loc = xferOp.getLoc();

  Value baseMemref = xferOp.getBase();
  MemRefType memrefType = llvm::cast<MemRefType>(baseMemref.getType());
  Type elementType = memrefType.getElementType();

  // Compute the total number of elements in the memref
  int64_t totalElements = 1;
  bool hasDynamicDim = false;
  for (int64_t dim : memrefType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      hasDynamicDim = true;
      break;
    }
    totalElements *= dim;
  }

  MemRefType flatMemrefType;
  if (hasDynamicDim) {
    flatMemrefType = MemRefType::get({ShapedType::kDynamic}, elementType);
  } else {
    flatMemrefType = MemRefType::get({totalElements}, elementType);
  }

  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices allDims;
  for (int i = 0; i < memrefType.getRank(); ++i) {
    allDims.push_back(i);
  }
  reassociation.push_back(allDims);

  auto collapseOp = memref::CollapseShapeOp::create(
      rewriter, loc, flatMemrefType, baseMemref, reassociation);
  return collapseOp;
}

// Create XeGPU gather load operation
static LogicalResult createLoadGather(vector::TransferReadOp readOp,
                                      PatternRewriter &rewriter,
                                      Value flatMemref, Value localOffsets) {
  Location loc = readOp.getLoc();
  VectorType vectorType = readOp.getVectorType();
  ArrayRef<int64_t> vectorShape = vectorType.getShape();
  Value mask = vector::ConstantMaskOp::create(
      rewriter, loc, VectorType::get(vectorShape, rewriter.getI1Type()),
      vectorShape);
  auto gatherOp = xegpu::LoadGatherOp::create(
      rewriter, loc, vectorType, flatMemref, localOffsets, mask,
      /*chunk_size=*/IntegerAttr{},
      /*l1_hint=*/xegpu::CachePolicyAttr{},
      /*l2_hint=*/xegpu::CachePolicyAttr{},
      /*l3_hint=*/xegpu::CachePolicyAttr{});
  rewriter.replaceOp(readOp, gatherOp.getResult());
  return success();
}

// Create XeGPU store scatter operation
static LogicalResult createStoreScatter(vector::TransferWriteOp writeOp,
                                        PatternRewriter &rewriter, Value value,
                                        Value flatMemref, Value localOffsets) {
  Location loc = writeOp.getLoc();
  VectorType vectorType = writeOp.getVectorType();
  ArrayRef<int64_t> vectorShape = vectorType.getShape();
  Value mask = vector::ConstantMaskOp::create(
      rewriter, loc, VectorType::get(vectorShape, rewriter.getI1Type()),
      vectorShape);
  xegpu::StoreScatterOp::create(rewriter, loc, value, flatMemref, localOffsets,
                                mask,
                                /*chunk_size=*/IntegerAttr{},
                                /*l1_hint=*/xegpu::CachePolicyAttr{},
                                /*l2_hint=*/xegpu::CachePolicyAttr{},
                                /*l3_hint=*/xegpu::CachePolicyAttr{});
  rewriter.eraseOp(writeOp);
  return success();
}

LogicalResult lowerTransferReadToLoadOp(vector::TransferReadOp readOp,
                                        PatternRewriter &rewriter) {

  auto memrefType = dyn_cast<MemRefType>(readOp.getShapedType());
  if (!memrefType)
    return rewriter.notifyMatchFailure(readOp, "Expected memref source");

  SmallVector<Value> strides = computeStrides(readOp, rewriter);
  if (strides.empty())
    return rewriter.notifyMatchFailure(readOp, "Failed to compute strides");

  Value localOffsets = computeOffsets(readOp, rewriter, strides);

  Value flatMemref = collapseMemrefTo1D(readOp, rewriter);

  return createLoadGather(readOp, rewriter, flatMemref, localOffsets);
}

LogicalResult lowerTransferWriteToStoreOp(vector::TransferWriteOp writeOp,
                                          PatternRewriter &rewriter) {

  auto memrefType = dyn_cast<MemRefType>(writeOp.getShapedType());
  if (!memrefType)
    return rewriter.notifyMatchFailure(writeOp, "Expected memref source");

  SmallVector<Value> strides = computeStrides(writeOp, rewriter);

  Value localOffsets = computeOffsets(writeOp, rewriter, strides);

  Value flatMemref = collapseMemrefTo1D(writeOp, rewriter);

  return createStoreScatter(writeOp, rewriter, writeOp.getVector(), flatMemref,
                            localOffsets);
}

struct TransferReadLowering : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    Location loc = readOp.getLoc();

    if (failed(transferPreconditions(rewriter, readOp)))
      return failure();

    auto chip = xegpu::getXeGPUChipStr(readOp);
    if ( chip != "pvc" && chip != "bmg") {
      // perform additional checks -
      if (failed(extraCheckForScatteredLoadStore(readOp, rewriter)))
        return failure();
      // calling another function that lower TransferReadOp to regular Loadop
      return lowerTransferReadToLoadOp(readOp, rewriter);
    }

    // Perform common data transfer checks.
    VectorType vecTy = readOp.getVectorType();
    if (failed(storeLoadPreconditions(rewriter, readOp, vecTy)))
      return failure();

    bool isOutOfBounds = readOp.hasOutOfBoundsDim();
    if (isOutOfBounds && !isZeroConstant(readOp.getPadding()))
      return rewriter.notifyMatchFailure(
          readOp, "Unsupported non-zero padded out-of-bounds read");

    AffineMap readMap = readOp.getPermutationMap();
    bool isTransposeLoad = !readMap.isMinorIdentity();

    Type elementType = vecTy.getElementType();
    unsigned minTransposeBitWidth = 32;
    if (isTransposeLoad &&
        elementType.getIntOrFloatBitWidth() < minTransposeBitWidth)
      return rewriter.notifyMatchFailure(
          readOp, "Unsupported data type for transposition");

    // If load is transposed, get the base shape for the tensor descriptor.
    SmallVector<int64_t> descShape(vecTy.getShape());
    if (isTransposeLoad)
      std::reverse(descShape.begin(), descShape.end());
    auto descType = xegpu::TensorDescType::get(
        descShape, elementType, /*array_length=*/1,
        /*boundary_check=*/isOutOfBounds, xegpu::MemorySpace::Global);

    xegpu::CreateNdDescOp ndDesc =
        createNdDescriptor(rewriter, loc, descType,
                           dyn_cast<TypedValue<MemRefType>>(readOp.getBase()),
                           readOp.getIndices());

    DenseI64ArrayAttr transposeAttr =
        !isTransposeLoad ? nullptr
                         : DenseI64ArrayAttr::get(rewriter.getContext(),
                                                  ArrayRef<int64_t>{1, 0});
    // By default, no specific caching policy is assigned.
    xegpu::CachePolicyAttr hint = nullptr;
    auto loadOp = xegpu::LoadNdOp::create(rewriter, loc, vecTy, ndDesc,
                                          /*packed=*/nullptr, transposeAttr,
                                          /*l1_hint=*/hint,
                                          /*l2_hint=*/hint, /*l3_hint=*/hint);
    rewriter.replaceOp(readOp, loadOp);

    return success();
  }
};

struct TransferWriteLowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = writeOp.getLoc();

    if (failed(transferPreconditions(rewriter, writeOp)))
      return failure();

    auto chip = xegpu::getXeGPUChipStr(writeOp);
    if (chip != "pvc" && chip != "bmg") {
      // perform additional checks -
      if (failed(extraCheckForScatteredLoadStore(writeOp, rewriter)))
        return failure();
      // calling another function that lower TransferWriteOp to regular StoreOp
      return lowerTransferWriteToStoreOp(writeOp, rewriter);
    }

    // Perform common data transfer checks.
    VectorType vecTy = writeOp.getVectorType();
    if (failed(storeLoadPreconditions(rewriter, writeOp, vecTy)))
      return failure();

    AffineMap map = writeOp.getPermutationMap();
    if (!map.isMinorIdentity())
      return rewriter.notifyMatchFailure(writeOp, "Expects identity map");

    auto descType = xegpu::TensorDescType::get(
        vecTy.getShape(), vecTy.getElementType(),
        /*array_length=*/1, /*boundary_check=*/writeOp.hasOutOfBoundsDim(),
        xegpu::MemorySpace::Global);
    xegpu::CreateNdDescOp ndDesc =
        createNdDescriptor(rewriter, loc, descType,
                           dyn_cast<TypedValue<MemRefType>>(writeOp.getBase()),
                           writeOp.getIndices());

    // By default, no specific caching policy is assigned.
    xegpu::CachePolicyAttr hint = nullptr;
    auto storeOp =
        xegpu::StoreNdOp::create(rewriter, loc, writeOp.getVector(), ndDesc,
                                 /*l1_hint=*/hint,
                                 /*l2_hint=*/hint, /*l3_hint=*/hint);
    rewriter.replaceOp(writeOp, storeOp);

    return success();
  }
};

struct LoadLowering : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();

    VectorType vecTy = loadOp.getResult().getType();
    if (failed(storeLoadPreconditions(rewriter, loadOp, vecTy)))
      return failure();

    // Boundary check is available only for block instructions.
    bool boundaryCheck = vecTy.getRank() > 1;

    auto descType = xegpu::TensorDescType::get(
        vecTy.getShape(), vecTy.getElementType(), /*array_length=*/1,
        boundaryCheck, xegpu::MemorySpace::Global);
    xegpu::CreateNdDescOp ndDesc = createNdDescriptor(
        rewriter, loc, descType, loadOp.getBase(), loadOp.getIndices());

    // By default, no specific caching policy is assigned.
    xegpu::CachePolicyAttr hint = nullptr;
    auto loadNdOp = xegpu::LoadNdOp::create(
        rewriter, loc, vecTy, ndDesc, /*packed=*/nullptr, /*transpose=*/nullptr,
        /*l1_hint=*/hint,
        /*l2_hint=*/hint, /*l3_hint=*/hint);
    rewriter.replaceOp(loadOp, loadNdOp);

    return success();
  }
};

struct StoreLowering : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

    TypedValue<VectorType> vector = storeOp.getValueToStore();
    VectorType vecTy = vector.getType();
    if (failed(storeLoadPreconditions(rewriter, storeOp, vecTy)))
      return failure();

    // Boundary check is available only for block instructions.
    bool boundaryCheck = vecTy.getRank() > 1;

    auto descType = xegpu::TensorDescType::get(
        vecTy.getShape(), vecTy.getElementType(),
        /*array_length=*/1, boundaryCheck, xegpu::MemorySpace::Global);
    xegpu::CreateNdDescOp ndDesc = createNdDescriptor(
        rewriter, loc, descType, storeOp.getBase(), storeOp.getIndices());

    // By default, no specific caching policy is assigned.
    xegpu::CachePolicyAttr hint = nullptr;
    auto storeNdOp =
        xegpu::StoreNdOp::create(rewriter, loc, vector, ndDesc,
                                 /*l1_hint=*/hint,
                                 /*l2_hint=*/hint, /*l3_hint=*/hint);
    rewriter.replaceOp(storeOp, storeNdOp);

    return success();
  }
};

struct ContractionLowering : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    Location loc = contractOp.getLoc();

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind");

    TypedValue<Type> acc = contractOp.getAcc();
    VectorType accType = dyn_cast<VectorType>(acc.getType());
    if (!accType || accType.getRank() != 2)
      return rewriter.notifyMatchFailure(contractOp, "Expects acc 2D vector");

    // Accept only plain 2D data layout.
    // VNNI packing is applied to DPAS as a separate lowering step.
    TypedValue<VectorType> lhs = contractOp.getLhs();
    TypedValue<VectorType> rhs = contractOp.getRhs();
    if (lhs.getType().getRank() != 2 || rhs.getType().getRank() != 2)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects lhs and rhs 2D vectors");

    if (!isRowMajorMatmul(contractOp.getIndexingMapsAttr()))
      return rewriter.notifyMatchFailure(contractOp, "Invalid indexing maps");

    auto dpasOp = xegpu::DpasOp::create(rewriter, loc,
                                        TypeRange{contractOp.getResultType()},
                                        ValueRange{lhs, rhs, acc});
    rewriter.replaceOp(contractOp, dpasOp);

    return success();
  }
};

struct ConvertVectorToXeGPUPass
    : public impl::ConvertVectorToXeGPUBase<ConvertVectorToXeGPUPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorToXeGPUConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::populateVectorToXeGPUConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TransferReadLowering, TransferWriteLowering, LoadLowering,
               StoreLowering, ContractionLowering>(patterns.getContext());
}
