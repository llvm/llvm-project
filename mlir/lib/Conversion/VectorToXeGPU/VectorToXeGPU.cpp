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

  // Perform common data transfer checks.
  VectorType vecTy = xferOp.getVectorType();
  if (failed(storeLoadPreconditions(rewriter, xferOp, vecTy)))
    return failure();

  // Validate further transfer op semantics.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(srcTy.getStridesAndOffset(strides, offset)) || strides.back() != 1)
    return rewriter.notifyMatchFailure(
        xferOp, "Buffer must be contiguous in the innermost dimension");

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

// This function lowers vector.transfer_read to XeGPU load operation.
  // Example:
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

//   %expand_shape1 = memref.collapseshape %expand_shape:
//   memref<8x4x2x6x32xbf16> -> memref<?bf16>

//   %vec = xegpu.load_gather %expand_shape1[%offsets]:memref<?xbf16>,
//                           vector<4x2x6x32xindex> -> vector<4x2x6x32xbf16>

// Compute localOffsets for load_gather and store_scatter
static Value computeGatherOffsets(vector::TransferReadOp readOp,
                                  PatternRewriter &rewriter,
                                  ArrayRef<int64_t> strides,
                                  VectorType vectorType) {
  Location loc = readOp.getLoc();
  ArrayRef<int64_t> vectorShape = vectorType.getShape();

  // Step 1: Create vector.step operations for each dimension
  SmallVector<Value> stepVectors;
  for (int64_t dim : vectorShape) {
    auto stepType = VectorType::get({dim}, rewriter.getIndexType());
    auto stepOp = rewriter.create<vector::StepOp>(loc, stepType);
    stepVectors.push_back(stepOp);
  }

  // Step 2: Multiply step vectors by corresponding strides
  size_t memrefRank = strides.size();
  size_t vectorRank = vectorShape.size();
  SmallVector<Value> strideMultiplied;
  for (size_t i = 0; i < vectorRank; ++i) {
    size_t memrefDim = memrefRank - vectorRank + i;
    int64_t stride = strides[memrefDim];
    Value strideConstant = rewriter.create<arith::ConstantIndexOp>(loc, stride);
    auto mulType = llvm::cast<VectorType>(stepVectors[i].getType());
    auto mulOp = rewriter.create<arith::MulIOp>(
        loc, stepVectors[i],
        rewriter.create<vector::SplatOp>(loc, strideConstant, mulType));
    strideMultiplied.push_back(mulOp);
  }

  // Step 3: Shape cast each multiplied vector to add singleton dimensions
  SmallVector<Value> shapeCasted;
  for (size_t i = 0; i < vectorRank; ++i) {
    SmallVector<int64_t> newShape(vectorRank, 1);
    newShape[i] = vectorShape[i];
    auto newType = VectorType::get(newShape, rewriter.getIndexType());
    auto castOp =
        rewriter.create<vector::ShapeCastOp>(loc, newType, strideMultiplied[i]);
    shapeCasted.push_back(castOp);
  }

  // Step 4: Broadcast each shape-casted vector to full vector shape
  SmallVector<Value> broadcasted;
  auto fullIndexVectorType =
      VectorType::get(vectorShape, rewriter.getIndexType());
  for (Value shapeCastVal : shapeCasted) {
    auto broadcastOp = rewriter.create<vector::BroadcastOp>(
        loc, fullIndexVectorType, shapeCastVal);
    broadcasted.push_back(broadcastOp);
  }

  // Step 5: Add all broadcasted vectors together to compute local offsets
  Value localOffsets = broadcasted[0];
  for (size_t i = 1; i < broadcasted.size(); ++i) {
    localOffsets =
        rewriter.create<arith::AddIOp>(loc, localOffsets, broadcasted[i]);
  }

  // Step 6: Compute base offset from transfer read indices
  Value baseOffset = nullptr;
  auto indices = readOp.getIndices();
  if (!indices.empty()) {
    baseOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (size_t i = 0; i < indices.size(); ++i) {
      Value strideVal =
          rewriter.create<arith::ConstantIndexOp>(loc, strides[i]);
      Value offsetContrib =
          rewriter.create<arith::MulIOp>(loc, indices[i], strideVal);
      baseOffset =
          rewriter.create<arith::AddIOp>(loc, baseOffset, offsetContrib);
    }
    // Broadcast base offset to match vector shape
    Value splatBase =
        rewriter.create<vector::SplatOp>(loc, baseOffset, fullIndexVectorType);
    localOffsets = rewriter.create<arith::AddIOp>(loc, splatBase, localOffsets);
  }

  return localOffsets;
}

// Collapse memref shape to 1D
static Value collapseMemrefTo1D(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter,
                                MemRefType memrefType, Type elementType) {
  Location loc = readOp.getLoc();
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

  auto collapseOp = rewriter.create<memref::CollapseShapeOp>(
      loc, flatMemrefType, readOp.getBase(), reassociation);
  return collapseOp;
}

// Create XeGPU gather load operation
static LogicalResult createLoadGather(vector::TransferReadOp readOp,
                                      PatternRewriter &rewriter,
                                      Value flatMemref, Value localOffsets,
                                      VectorType vectorType) {
  Location loc = readOp.getLoc();
  ArrayRef<int64_t> vectorShape = vectorType.getShape();
  Value mask = rewriter.create<vector::ConstantMaskOp>(
      loc, VectorType::get(vectorShape, rewriter.getI1Type()), vectorShape);
  auto gatherOp = rewriter.create<xegpu::LoadGatherOp>(
      loc, vectorType, flatMemref, localOffsets, mask,
      /*chunk_size=*/IntegerAttr{},
      /*l1_hint=*/xegpu::CachePolicyAttr{},
      /*l2_hint=*/xegpu::CachePolicyAttr{},
      /*l3_hint=*/xegpu::CachePolicyAttr{});
  rewriter.replaceOp(readOp, gatherOp.getResult());
  return success();
}

// Create XeGPU store scatter operation
static LogicalResult createStoreScatter(vector::TransferWriteOp writeOp,
                                        PatternRewriter &rewriter,
                                        Value flatMemref, Value localOffsets,
                                        Value value, VectorType vectorType) {
  Location loc = writeOp.getLoc();
  ArrayRef<int64_t> vectorShape = vectorType.getShape();
  Value mask = rewriter.create<vector::ConstantMaskOp>(
      loc, VectorType::get(vectorShape, rewriter.getI1Type()), vectorShape);
  rewriter.create<xegpu::StoreScatterOp>(loc, value, flatMemref, localOffsets,
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

  VectorType vectorType = readOp.getVectorType();
  Type elementType = vectorType.getElementType();

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(memrefType.getStridesAndOffset(strides, offset)))
    return rewriter.notifyMatchFailure(readOp, "Failed to get memref strides");

  Value localOffsets =
      computeGatherOffsets(readOp, rewriter, strides, vectorType);
  Value flatMemref =
      collapseMemrefTo1D(readOp, rewriter, memrefType, elementType);
  return createLoadGather(readOp, rewriter, flatMemref, localOffsets,
                          vectorType);
}

LogicalResult lowerTransferWriteToStoreOp(vector::TransferWriteOp writeOp,
                                          PatternRewriter &rewriter) {

  auto memrefType = dyn_cast<MemRefType>(writeOp.getShapedType());
  if (!memrefType)
    return rewriter.notifyMatchFailure(writeOp, "Expected memref source");

  VectorType vectorType = writeOp.getVectorType();
  Type elementType = vectorType.getElementType();

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(memrefType.getStridesAndOffset(strides, offset)))
    return rewriter.notifyMatchFailure(writeOp, "Failed to get memref strides");

  // Compute localOffsets for store_scatter
  Value localOffsets =
      computeGatherOffsets(cast<vector::TransferReadOp>(writeOp.getOperation()),
                           rewriter, strides, vectorType);

  Value flatMemref =
      collapseMemrefTo1D(cast<vector::TransferReadOp>(writeOp.getOperation()),
                         rewriter, memrefType, elementType);

  return createStoreScatter(writeOp, rewriter, flatMemref, localOffsets,
                            writeOp.getVector(), vectorType);
}

static LogicalResult
extraCheckForScatteredLoadStore(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) {
  // 1. it must be inbound access by checking in_bounds attributes, like
  // {in_bounds = [false, true]}
  if (readOp.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(
        readOp, "Out-of-bounds access is not supported for this chip");
  // 2. if the memref has static shape, its lower rank must exactly match with
  // vector shape.
  if (auto memrefType = dyn_cast<MemRefType>(readOp.getShapedType())) {
    if (memrefType.hasStaticShape()) {
      ArrayRef<int64_t> memrefShape = memrefType.getShape();
      ArrayRef<int64_t> vectorShape = readOp.getVectorType().getShape();
      size_t memrefRank = memrefShape.size();
      size_t vectorRank = vectorShape.size();
      if (vectorRank > memrefRank)
        return rewriter.notifyMatchFailure(
            readOp, "Vector rank cannot exceed memref rank");
      // Compare the last vectorRank dimensions of memref with vector shape
      for (size_t i = 0; i < vectorRank; ++i) {
        if (memrefShape[memrefRank - vectorRank + i] != vectorShape[i])
          return rewriter.notifyMatchFailure(
              readOp, "Memref lower dimensions must match vector shape");
      }
    }
  }
  return success();
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

    bool isOutOfBounds = readOp.hasOutOfBoundsDim();
    if (isOutOfBounds && !isZeroConstant(readOp.getPadding()))
      return rewriter.notifyMatchFailure(
          readOp, "Unsupported non-zero padded out-of-bounds read");

    AffineMap readMap = readOp.getPermutationMap();
    bool isTransposeLoad = !readMap.isMinorIdentity();

    VectorType vecTy = readOp.getVectorType();
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
      // calling another function that lower TransferWriteOp to regular StoreOp
      return lowerTransferWriteToStoreOp(writeOp, rewriter);
    }

    AffineMap map = writeOp.getPermutationMap();
    if (!map.isMinorIdentity())
      return rewriter.notifyMatchFailure(writeOp, "Expects identity map");

    VectorType vecTy = writeOp.getVectorType();
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
