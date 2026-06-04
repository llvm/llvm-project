//===---- XeGPUUtils.cpp - MLIR Utilities for XeGPUOps   ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility methods for working with the XeGPU dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

/// convert ArrayRef<ValueRange> into SmallVector<Value>
SmallVector<Value> xegpu::flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const auto &vals : values)
    llvm::append_range(result, vals);
  return result;
}

FailureOr<VectorType>
mlir::xegpu::getDistributedVectorType(xegpu::TensorDescType tdescTy) {
  auto layout = llvm::dyn_cast_if_present<LayoutAttr>(tdescTy.getLayout());
  // It only works for subgroup level layout, which only has lane_layout
  // and lane_data, and is to distribute a SIMD code into SIMT code.
  if (!layout || !layout.isForSubgroup())
    return failure();

  SmallVector<int64_t> laneData(layout.getLaneData().asArrayRef());
  SmallVector<int64_t> laneLayout(layout.getLaneLayout().asArrayRef());
  auto tdescShape = tdescTy.getShape();
  auto elementType = tdescTy.getElementType();

  // compute sgSize by multiply elements of laneLayout
  // e.g. for 2D layout, sgSize = laneLayout[0] * laneLayout[1]
  // e.g. for 1D layout, sgSize = laneLayout[0]
  int64_t sgSize = llvm::product_of(laneLayout);

  // Check if the tensor descriptor shape is distributable.
  int64_t tensorSize = 1;
  for (auto [tdescDim, laneDim, laneDataDim] :
       llvm::zip_equal(tdescShape, laneLayout, laneData)) {
    assert((tdescDim % (laneDim * laneDataDim) == 0) &&
           "tensor descriptor shape is not distributable");
    tensorSize *= tdescDim;
  }
  // tensorSize must be adjusted for array_length.
  tensorSize *= tdescTy.getArrayLength();

  return VectorType::get({tensorSize / sgSize}, elementType);
}

FailureOr<VectorType>
mlir::xegpu::getDistributedVectorType(VectorType originalType,
                                      xegpu::LayoutAttr layout) {
  int64_t rank = originalType.getRank();
  // Distributed vector type is only supported for 1D, 2D and 3D vectors.
  if (rank < 1 || rank > 3)
    return failure();
  ArrayRef<int64_t> shape = originalType.getShape();
  // arrayLength is 1 for 1D and 2D vectors, and equal to the first dimension
  // of the 3D vector.
  int arrayLength = 1;
  if (rank == 3) {
    arrayLength = shape[0];
    shape = shape.drop_front();
  }
  auto helperTdescTy = xegpu::TensorDescType::get(
      shape, originalType.getElementType(), arrayLength,
      /*boundary_check=*/true,
      /*memory_space=*/xegpu::MemorySpace::Global, layout);
  return xegpu::getDistributedVectorType(helperTdescTy);
}

FailureOr<VectorType>
xegpu::getDistVecTypeBasedOnLaneLayout(xegpu::DistributeLayoutAttr layout,
                                       VectorType originalType) {
  if (!layout)
    return failure();
  assert((isa<xegpu::LayoutAttr>(layout) || isa<xegpu::SliceAttr>(layout)) &&
         "Expecting a valid layout.");

  int64_t vectorRank = originalType.getRank();
  int64_t layoutRank = layout.getRank();
  assert(vectorRank >= layoutRank && "Vector rank must be >= layout rank.");

  // When the vector has more dimensions than the layout, only the trailing
  // dimensions are distributed. Leading dimensions are preserved as-is.
  int64_t offset = vectorRank - layoutRank;
  ArrayRef<int64_t> fullShape = originalType.getShape();
  SmallVector<int64_t> trailingShape(fullShape.begin() + offset,
                                     fullShape.end());
  auto distributedShapeOrFailure =
      layout.computeDistributedShape(trailingShape);
  if (failed(distributedShapeOrFailure))
    return failure();

  SmallVector<int64_t> resultShape(fullShape.begin(),
                                   fullShape.begin() + offset);
  resultShape.append(distributedShapeOrFailure->begin(),
                     distributedShapeOrFailure->end());
  return VectorType::get(resultShape, originalType.getElementType());
}

std::string xegpu::getTemporaryLayoutName(const OpOperand &operand) {
  const StringRef prefix("layout_operand_");
  unsigned idx = const_cast<OpOperand &>(operand).getOperandNumber();
  return llvm::formatv("{0}{1}", prefix, idx).str();
}

std::string xegpu::getTemporaryLayoutName(const OpResult result) {
  const StringRef prefix = "layout_result_";
  return llvm::formatv("{0}{1}", prefix, result.getResultNumber()).str();
}

xegpu::DistributeLayoutAttr xegpu::getDistributeLayoutAttr(const Value value) {
  if (!value)
    return nullptr;

  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *defOp = result.getDefiningOp();
    assert(defOp && "result must have a defining op");

    if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(defOp)) {
      auto layout = anchorOp.getAnchorLayout();
      return layout;
    }

    std::string layoutName = getTemporaryLayoutName(result);
    if (defOp->hasAttr(layoutName)) {
      auto layout =
          defOp->getAttrOfType<xegpu::DistributeLayoutAttr>(layoutName);
      return layout;
    }
  }

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = arg.getOwner()->getParentOp();
    if (auto loop = dyn_cast_if_present<LoopLikeOpInterface>(parentOp)) {
      OpOperand *tiedInit = loop.getTiedLoopInit(arg);
      if (tiedInit)
        return getTemporaryLayout(*tiedInit);
    }
  }

  if (auto tdescTy =
          dyn_cast_if_present<xegpu::TensorDescType>(value.getType()))
    return tdescTy.getLayoutAttr();

  return nullptr;
}
xegpu::DistributeLayoutAttr
xegpu::getDistributeLayoutAttr(const OpOperand &opr) {
  Operation *op = opr.getOwner();
  unsigned idx = const_cast<OpOperand &>(opr).getOperandNumber();

  if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(op)) {
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(op)) {
      if (idx == 0) {
        return dpasOp.getLayoutAAttr();
      } else if (idx == 1) {
        return dpasOp.getLayoutBAttr();
      } else if (idx == 2) {
        return dpasOp.getLayoutCdAttr();
      }
    }
    if (auto dpasMxOp = dyn_cast<xegpu::DpasMxOp>(op)) {
      // DpasMxOp has operands: a, b, optional acc, optional scale_a, optional
      // scale_b
      unsigned currentIdx = 0;

      if (idx == currentIdx++)
        return dpasMxOp.getLayoutAAttr();

      if (idx == currentIdx++)
        return dpasMxOp.getLayoutBAttr();

      if (dpasMxOp.getAcc())
        if (idx == currentIdx++)
          return dpasMxOp.getLayoutCdAttr();

      if (dpasMxOp.getScaleA())
        if (idx == currentIdx++)
          return dpasMxOp.getLayoutAScaleAttr();

      if (dpasMxOp.getScaleB())
        if (idx == currentIdx++)
          return dpasMxOp.getLayoutBScaleAttr();

      return nullptr;
    }
    if (auto convertOp = dyn_cast<xegpu::ConvertLayoutOp>(op)) {
      return convertOp.getInputLayoutAttr();
    }
    auto layout = anchorOp.getAnchorLayout();

    if (idx == 0)
      return layout;

    // For StoreNdOp and StoreMatrixOp,
    // the layout is valid for the first two operands: value and memref/tdesc.
    if (isa<xegpu::StoreNdOp, xegpu::StoreMatrixOp>(op) && (idx < 2))
      return layout;

    if (isa<xegpu::StoreScatterOp>(op)) {
      xegpu::StoreScatterOp store(op);
      int chunkSize = store.getChunkSize().value_or(1);
      if (layout && idx >= 2 && chunkSize > 1)
        return layout.dropDims(llvm::to_vector(
            llvm::seq<int64_t>(layout.getRank() - 1, layout.getRank())));
      return layout;
    }
    if (isa<xegpu::LoadGatherOp>(op)) {
      xegpu::LoadGatherOp load(op);
      int chunkSize = load.getChunkSize().value_or(1);
      if (layout && idx >= 1 && chunkSize > 1)
        return layout.dropDims(llvm::to_vector(
            llvm::seq<int64_t>(layout.getRank() - 1, layout.getRank())));
      return layout;
    }
  }

  std::string layoutName = xegpu::getTemporaryLayoutName(opr);
  if (op->hasAttr(layoutName)) {
    auto layout = op->getAttrOfType<xegpu::DistributeLayoutAttr>(layoutName);
    return layout;
  }

  return nullptr;
}

// Returns the permanent layout attribute for the given result if it's
// available on the defining op. Otherwise returns the provided layout.
xegpu::DistributeLayoutAttr
maybePickPermanentLayout(xegpu::DistributeLayoutAttr layout,
                         const OpResult &result, mlir::Operation *owner,
                         const std::string &name) {
  xegpu::DistributeLayoutAttr candidate = layout;

  if (auto loadOp = dyn_cast<xegpu::LoadGatherOp>(owner)) {
    if (auto perm = loadOp.getLayoutAttr())
      candidate = perm;
  }

  return candidate;
}

// Returns the permanent layout attribute for the given operand if it's
// available on the defining op. Otherwise returns the provided layout.
xegpu::DistributeLayoutAttr
maybePickPermanentLayout(xegpu::DistributeLayoutAttr layout,
                         const OpOperand &operand, mlir::Operation *owner,
                         const std::string &name) {
  xegpu::DistributeLayoutAttr candidate = layout;
  unsigned idx = const_cast<OpOperand &>(operand).getOperandNumber();

  if (auto storeOp = dyn_cast<xegpu::StoreScatterOp>(owner)) {
    if (idx == 0) {
      if (auto perm = storeOp.getLayoutAttr())
        candidate = perm;
    }
  }

  return candidate;
}

// TODO-LayoutRefactor: Remove this function after replacing use
//  with setTemporaryLayout or setAnchorLayout
void xegpu::setDistributeLayoutAttr(
    const mlir::OpResult &result,
    const mlir::xegpu::DistributeLayoutAttr layout) {
  Operation *owner = result.getOwner();

  if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(owner)) {
    if (anchorOp.getAnchorLayout() == layout)
      return;
    anchorOp.setAnchorLayout(layout);
    return;
  }

  std::string name = xegpu::getTemporaryLayoutName(result);
  if (owner->hasAttrOfType<DistributeLayoutAttr>(name)) {
    return;
  }
  if (layout) {
    owner->setAttr(name, layout);
  }
}

// TODO-LayoutRefactor: Remove this function after replacing use
//  with setTemporaryLayout or setAnchorLayout
void xegpu::setDistributeLayoutAttr(const OpOperand &operand,
                                    const DistributeLayoutAttr layout) {
  Operation *owner = operand.getOwner();
  unsigned idx = const_cast<OpOperand &>(operand).getOperandNumber();

  if (!layout) {
    return;
  }
  if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(owner)) {
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(owner)) {
      if (idx == 0) {
        return dpasOp.setLayoutAAttr(layout);
      } else if (idx == 1) {
        return dpasOp.setLayoutBAttr(layout);
      } else if (idx == 2) {
        return dpasOp.setLayoutCdAttr(layout);
      }
    }
    if (auto convertOp = dyn_cast<xegpu::ConvertLayoutOp>(owner)) {
      return convertOp.setInputLayoutAttr(layout);
    }

    // For store operations (StoreScatterOp, StoreNdOp, StoreMatrixOp),
    // the layout is valid for the first two operands: value and memref/tdesc.
    // For other operations, the layout applies to the first operand only.
    if (isa<xegpu::StoreScatterOp, xegpu::StoreNdOp, xegpu::StoreMatrixOp>(
            owner)) {
      if (idx < 2) {
        anchorOp.setAnchorLayout(layout);
      }
    } else {
      if (idx == 0) {
        anchorOp.setAnchorLayout(layout);
      }
    }
  }

  std::string name = xegpu::getTemporaryLayoutName(operand);
  if (owner->hasAttrOfType<DistributeLayoutAttr>(name)) {
    return;
  }
  if (layout) {
    owner->setAttr(name, layout);
  }
}

template <typename T, typename>
xegpu::DistributeLayoutAttr
xegpu::getTemporaryLayout(const T &operandOrResult) {
  Operation *op = operandOrResult.getOwner();

  std::string layoutName = xegpu::getTemporaryLayoutName(operandOrResult);
  if (op->hasAttr(layoutName)) {
    auto layout = op->getAttrOfType<xegpu::DistributeLayoutAttr>(layoutName);
    return layout;
  }

  return nullptr;
}

template xegpu::DistributeLayoutAttr
xegpu::getTemporaryLayout<mlir::OpResult>(const OpResult &result);
template xegpu::DistributeLayoutAttr
xegpu::getTemporaryLayout<mlir::OpOperand>(const OpOperand &operand);

template <typename T, typename>
void xegpu::setTemporaryLayout(const T &operandOrResult,
                               const xegpu::DistributeLayoutAttr layout) {
  Operation *owner = operandOrResult.getOwner();
  std::string name = xegpu::getTemporaryLayoutName(operandOrResult);
  if (owner->hasAttrOfType<xegpu::DistributeLayoutAttr>(name)) {
    return;
  }
  if (layout) {
    owner->setAttr(name, layout);
  }
}

template void xegpu::setTemporaryLayout<mlir::OpResult>(
    const mlir::OpResult &result,
    const mlir::xegpu::DistributeLayoutAttr layout);

template void xegpu::setTemporaryLayout<mlir::OpOperand>(
    const mlir::OpOperand &operand,
    const mlir::xegpu::DistributeLayoutAttr layout);

SmallVector<Value>
xegpu::extractVectorsWithShapeFromValue(OpBuilder &builder, Location loc,
                                        Value value, ArrayRef<int64_t> shape) {
  auto vecTy = dyn_cast<VectorType>(value.getType());
  if (!vecTy)
    return {value};

  ArrayRef<int64_t> srcShape = vecTy.getShape();
  if (!computeShapeRatio(srcShape, shape))
    return {value};

  int64_t srcShapeRank = srcShape.size();
  int64_t targetShapeRank = shape.size();

  SmallVector<int64_t> adjustedTargetShape(srcShape.size());
  int64_t rankDiff = srcShapeRank - targetShapeRank;
  std::fill(adjustedTargetShape.begin(), adjustedTargetShape.begin() + rankDiff,
            1);
  llvm::copy(shape, adjustedTargetShape.begin() + rankDiff);

  SmallVector<Value> result;
  for (SmallVector<int64_t> offsets :
       StaticTileOffsetRange(srcShape, adjustedTargetShape)) {
    SmallVector<int64_t> staticStrides(offsets.size(), 1);
    Value slice = vector::ExtractStridedSliceOp::create(
        builder, loc, value, offsets, adjustedTargetShape, staticStrides);

    // Reshape to remove leading unit dims if needed
    if (srcShapeRank > targetShapeRank) {
      auto targetTy = VectorType::get(shape, vecTy.getElementType());
      slice = vector::ShapeCastOp::create(builder, loc, targetTy, slice);
    }
    result.push_back(slice);
  }

  return result;
}

Value xegpu::createVectorWithShapeFromValues(OpBuilder &builder, Location loc,
                                             ValueRange values,
                                             ArrayRef<int64_t> shape) {
  VectorType inputTy = dyn_cast<VectorType>(values[0].getType());
  assert(llvm::all_of(values.getTypes(),
                      [&](Type type) { return type == inputTy; }) &&
         "values must be of the same VectorType");

  Type elemTy = inputTy.getElementType();
  ArrayRef<int64_t> tileShape = inputTy.getShape();

  VectorType resultTy = VectorType::get(shape, elemTy);
  auto zeroAttr = builder.getZeroAttr(elemTy);
  Value result = arith::ConstantOp::create(
      builder, loc, resultTy, DenseElementsAttr::get(resultTy, zeroAttr));

  for (auto [src, offsets] :
       llvm::zip_equal(values, StaticTileOffsetRange(shape, tileShape))) {
    SmallVector<int64_t> staticStrides(tileShape.size(), 1);
    result = vector::InsertStridedSliceOp::create(builder, loc, src, result,
                                                  offsets, staticStrides);
  }
  return result;
}

std::optional<std::string> xegpu::getChipStr(Operation *op) {
  auto gpuModuleOp = op->getParentOfType<gpu::GPUModuleOp>();

  if (!gpuModuleOp)
    return std::nullopt;

  auto targetAttrs = gpuModuleOp.getTargets();
  if (targetAttrs) {
    for (auto &attr : *targetAttrs) {
      auto xevmAttr = llvm::dyn_cast<xevm::XeVMTargetAttr>(attr);
      if (xevmAttr)
        return xevmAttr.getChip().str();
    }
  }

  return std::nullopt;
}

/// Generates element-wise addition ops of two arrays with same length.
SmallVector<OpFoldResult> xegpu::addElementwise(OpBuilder &builder,
                                                Location loc,
                                                ArrayRef<OpFoldResult> lhs,
                                                ArrayRef<OpFoldResult> rhs) {
  assert(lhs.size() == rhs.size() && "lhs and rhs must have the same size");
  SmallVector<OpFoldResult> results;
  for (auto [l, r] : llvm::zip_equal(lhs, rhs)) {
    auto lval = getValueOrCreateConstantIndexOp(builder, loc, l);
    auto rval = getValueOrCreateConstantIndexOp(builder, loc, r);
    results.push_back(builder.createOrFold<arith::AddIOp>(loc, lval, rval));
  }
  return results;
}

/// Generates element-wise addition ops of two arrays with automatic alignment.
/// When the input arrays have different sizes, the shorter array is
/// right-aligned with the longer array, and the unmatched leading elements from
/// the longer array are preserved unchanged. This is commonly used for offset
/// computation where higher-dimensional offsets need to be added to
/// lower-dimensional adjustments.
///
/// Example:
///   lhs = [l1, l2, l3], rhs = [r1, r2]
///   Result: [11, l2+r1, l3+r2]
SmallVector<OpFoldResult>
xegpu::addWithRightAligned(OpBuilder &builder, Location loc,
                           ArrayRef<OpFoldResult> lhs,
                           ArrayRef<OpFoldResult> rhs) {
  // ensure a is longer than b
  ArrayRef<OpFoldResult> a = lhs.size() >= rhs.size() ? lhs : rhs;
  ArrayRef<OpFoldResult> b = lhs.size() >= rhs.size() ? rhs : lhs;
  SmallVector<OpFoldResult> results(a.take_front(a.size() - b.size()));
  a = a.slice(a.size() - b.size());
  results.append(addElementwise(builder, loc, a, b));
  return results;
}

template <typename T>
int xegpu::getLargestDivisor(T dim, ArrayRef<T> candidates,
                             ArrayRef<T> candidateMultiples) {
  static_assert(std::is_integral<T>::value, "T must be an integer type");
  int largest = -1;
  SmallVector<T> multiples = {1};
  if (!candidateMultiples.empty())
    multiples =
        SmallVector<T>(candidateMultiples.begin(), candidateMultiples.end());
  for (T candidate : candidates) {
    for (T multiple : multiples) {
      int value = static_cast<int>(candidate * multiple);
      if (value != 0 && dim % value == 0 && value > largest)
        largest = value;
    }
  }
  return largest;
}

Value xegpu::subgroupReduction(Location loc, OpBuilder &builder, Value input,
                               vector::CombiningKind kind, uint32_t size) {
  // First reduce on a single thread to get per lane reduction value.
  Value laneVal = vector::ReductionOp::create(builder, loc, kind, input);
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < size; i <<= 1) {
    Value shuffled =
        gpu::ShuffleOp::create(builder, loc, laneVal, i, /**  width = **/ size,
                               /**  mode = **/ gpu::ShuffleMode::XOR)
            .getShuffleResult();
    laneVal = makeArithReduction(builder, loc, kind, laneVal, shuffled);
  }
  return laneVal;
}

Value xegpu::lowerToVectorReductions(TypedValue<VectorType> src,
                                     TypedValue<VectorType> acc,
                                     vector::CombiningKind kind,
                                     int64_t reductionDim, Location loc,
                                     PatternRewriter &rewriter) {
  VectorType sourceType = src.getType();
  int64_t sourceRank = sourceType.getRank();
  // Expecting at least a 2D source vector. Leading dimensions (all except the
  // last two) must be unit.
  assert(sourceRank >= 2 && "expected at least a 2D source vector");
  for (int64_t i = 0; i < sourceRank - 2; ++i)
    assert(sourceType.getShape()[i] == 1 &&
           "expected leading dimensions to be unit");
  int64_t rowIdx = sourceRank - 2;
  int64_t columnIdx = sourceRank - 1;
  int64_t sourceH = sourceType.getShape()[rowIdx];
  int64_t sourceW = sourceType.getShape()[columnIdx];
  int nSlices = (reductionDim == rowIdx) ? sourceW : sourceH;
  // Create a constant vector to hold the result of the reduction.
  TypedAttr zeroAttr = rewriter.getZeroAttr(sourceType.getElementType());
  Value reductionResult = arith::ConstantOp::create(
      rewriter, loc, acc.getType(),
      DenseElementsAttr::get(acc.getType(), zeroAttr));
  auto srcLayout = xegpu::getTemporaryLayout(dyn_cast<OpResult>(src));
  auto accLayout = xegpu::getTemporaryLayout(dyn_cast<OpResult>(acc));
  // Reduction result should have the same layout as the accumulator.
  xegpu::setTemporaryLayout(cast<OpResult>(reductionResult), accLayout);
  // For each slice of the source, extract the slice vector, do a reduction
  // and, insert the reduced value back to the result vector.
  int64_t accRank = acc.getType().getRank();
  for (int i = 0; i < nSlices; ++i) {
    // Build nD offsets, sizes, and strides. Leading unit dims get
    // offset=0, size=1. The last two dims are set based on reductionDim.
    SmallVector<int64_t> sliceOffsets(sourceRank, 0);
    SmallVector<int64_t> sliceSizes(sourceRank, 1);
    SmallVector<int64_t> strides(sourceRank, 1);
    if (reductionDim == columnIdx) {
      sliceOffsets[rowIdx] = i;
      sliceSizes[columnIdx] = sourceW;
    } else {
      sliceOffsets[columnIdx] = i;
      sliceSizes[rowIdx] = sourceH;
    }

    vector::ExtractStridedSliceOp extractOp =
        vector::ExtractStridedSliceOp::create(rewriter, loc, src, sliceOffsets,
                                              sliceSizes, strides);
    // Extract strided slice has the same layout as src.
    xegpu::setTemporaryLayout(extractOp->getOpResult(0), srcLayout);

    int64_t nSliceElements = extractOp.getResult().getType().getNumElements();

    vector::ShapeCastOp slice = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get({nSliceElements}, sourceType.getElementType()),
        extractOp.getResult());

    // Shape cast output has the same layout as the accumulator. Shape cast
    // source has the same layout as the original reduction source.
    xegpu::setTemporaryLayout(slice->getOpOperand(0), srcLayout);
    xegpu::setTemporaryLayout(slice->getOpResult(0), accLayout);
    // Extract and reduction results in scalars, so no result layout is needed.
    // Build multi-dim index into acc (sourceRank-1 dims, i.e. source shape with
    // the reduction dim removed). Leading unit dims get index 0.
    SmallVector<int64_t> accIdx(accRank, 0);
    accIdx[accRank - 1] = i;
    Value accExtract = vector::ExtractOp::create(rewriter, loc, acc, accIdx);
    Value reduction = vector::ReductionOp::create(
        rewriter, loc, kind, slice.getResult(), accExtract);
    reductionResult = vector::InsertOp::create(rewriter, loc, reduction,
                                               reductionResult, accIdx);
    // Insert op should have the same layout as the accumulator.
    xegpu::setTemporaryLayout(cast<OpResult>(reductionResult), accLayout);
  }
  return reductionResult;
}

Value xegpu::lowerCrossLaneReductionToShuffles(
    TypedValue<VectorType> src, TypedValue<VectorType> acc,
    vector::CombiningKind kind, int64_t reductionDim, int64_t reductionSize,
    Location loc, PatternRewriter &rewriter) {
  VectorType sourceType = src.getType();
  int64_t sourceRank = sourceType.getRank();
  // Expecting at least a 2D source vector. Leading dimensions (all except the
  // last two) must be unit.
  assert(sourceRank >= 2 && "expected at least a 2D source vector");
  for (int64_t i = 0; i < sourceRank - 2; ++i)
    assert(sourceType.getShape()[i] == 1 &&
           "expected leading dimensions to be unit");
  int64_t rowIdx = sourceRank - 2;
  int64_t columnIdx = sourceRank - 1;
  int64_t sourceH = sourceType.getShape()[rowIdx];
  int64_t sourceW = sourceType.getShape()[columnIdx];

  // Create a constant vector to hold the result of the reduction.
  TypedAttr zeroAttr = rewriter.getZeroAttr(sourceType.getElementType());
  Value reductionResult = arith::ConstantOp::create(
      rewriter, loc, acc.getType(),
      DenseElementsAttr::get(acc.getType(), zeroAttr));

  // nSlices is the number of reduction operations needed to reduce the entire
  // source vector. For example, if reductionDim is the row dim, we are
  // reducing across rows, and each slice is a column. So the number of slices
  // is the number of columns, which is sourceW.
  int nSlices = (reductionDim == rowIdx) ? sourceW : sourceH;

  // For each slice of the source, extract the slice vector, do a reduction
  // and, insert the reduced value back to the result vector.
  int64_t accRank = acc.getType().getRank();
  for (int i = 0; i < nSlices; ++i) {
    // Build nD offsets, sizes, and strides. Leading unit dims get
    // offset=0, size=1. The last two dims are set based on reductionDim.
    SmallVector<int64_t> sliceOffsets(sourceRank, 0);
    SmallVector<int64_t> sliceSizes(sourceRank, 1);
    SmallVector<int64_t> strides(sourceRank, 1);
    if (reductionDim == columnIdx) {
      sliceOffsets[rowIdx] = i;
      sliceSizes[columnIdx] = sourceW;
    } else {
      sliceOffsets[columnIdx] = i;
      sliceSizes[rowIdx] = sourceH;
    }

    vector::ExtractStridedSliceOp extractOp =
        vector::ExtractStridedSliceOp::create(rewriter, loc, src, sliceOffsets,
                                              sliceSizes, strides);
    int64_t nSliceElements = extractOp.getResult().getType().getNumElements();
    vector::ShapeCastOp slice = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get({nSliceElements}, sourceType.getElementType()),
        extractOp.getResult());

    SmallVector<int64_t> accIdx(accRank, 0);
    accIdx[accRank - 1] = i;
    Value accExtract = vector::ExtractOp::create(rewriter, loc, acc, accIdx);
    Value fullReduce =
        xegpu::subgroupReduction(loc, rewriter, slice, kind, reductionSize);
    fullReduce =
        vector::makeArithReduction(rewriter, loc, kind, fullReduce, accExtract);
    reductionResult = vector::InsertOp::create(rewriter, loc, fullReduce,
                                               reductionResult, accIdx);
  }
  return reductionResult;
}

Value xegpu::createReductionNeutralValue(OpBuilder &builder, Location loc,
                                         Type type,
                                         vector::CombiningKind kind) {
  auto vecTy = dyn_cast<VectorType>(type);
  Type elemTy = vecTy ? vecTy.getElementType() : type;

  // Helper to create either a splat vector or scalar constant from an attr.
  auto makeConst = [&](Attribute scalarAttr) -> Value {
    if (vecTy)
      return arith::ConstantOp::create(
          builder, loc, vecTy, DenseElementsAttr::get(vecTy, scalarAttr));
    return arith::ConstantOp::create(builder, loc, cast<TypedAttr>(scalarAttr));
  };

  switch (kind) {
  case vector::CombiningKind::ADD:
  case vector::CombiningKind::XOR:
  case vector::CombiningKind::OR:
  case vector::CombiningKind::MAXUI:
    return makeConst(builder.getZeroAttr(elemTy));

  case vector::CombiningKind::MUL:
  case vector::CombiningKind::AND:
    return makeConst(builder.getOneAttr(elemTy));

  case vector::CombiningKind::MINSI:
    if (auto intTy = dyn_cast<IntegerType>(elemTy))
      return makeConst(builder.getIntegerAttr(
          elemTy, APInt::getSignedMaxValue(intTy.getWidth())));
    return nullptr;

  case vector::CombiningKind::MINUI:
    if (auto intTy = dyn_cast<IntegerType>(elemTy))
      return makeConst(
          builder.getIntegerAttr(elemTy, APInt::getMaxValue(intTy.getWidth())));
    return nullptr;

  case vector::CombiningKind::MAXSI:
    if (auto intTy = dyn_cast<IntegerType>(elemTy))
      return makeConst(builder.getIntegerAttr(
          elemTy, APInt::getSignedMinValue(intTy.getWidth())));
    return nullptr;

  case vector::CombiningKind::MINNUMF:
  case vector::CombiningKind::MINIMUMF:
    if (auto floatTy = dyn_cast<FloatType>(elemTy))
      return makeConst(builder.getFloatAttr(
          elemTy, APFloat::getInf(floatTy.getFloatSemantics())));
    return nullptr;

  case vector::CombiningKind::MAXNUMF:
  case vector::CombiningKind::MAXIMUMF:
    if (auto floatTy = dyn_cast<FloatType>(elemTy))
      return makeConst(builder.getFloatAttr(
          elemTy, APFloat::getInf(floatTy.getFloatSemantics(), true)));
    return nullptr;
  }
  return nullptr;
}

/// Explicit instantiations
template int xegpu::getLargestDivisor<int>(int dim, ArrayRef<int> candidates,
                                           ArrayRef<int> candidateMultiples);
template int
xegpu::getLargestDivisor<unsigned>(unsigned dim, ArrayRef<unsigned> candidates,
                                   ArrayRef<unsigned> candidateMultiples);

bool xegpu::requirePacked(const xegpu::DistributeLayoutAttr layout) {
  if (!layout)
    return false;
  auto laneData = layout.getEffectiveLaneDataAsInt();
  if (laneData.size() != 2)
    return false;
  return laneData[0] != 1;
}

bool xegpu::requireTranspose(const xegpu::DistributeLayoutAttr layout,
                             const xegpu::uArch::uArch *uArch) {
  // Return false for unsupported targets.
  // TODO: Add more support or move to target info.
  if (uArch->getName().equals_insensitive("pvc") &&
      uArch->getName().equals_insensitive("bmg") &&
      uArch->getName().equals_insensitive("cri"))
    return false;
  if (!layout)
    return false;
  auto laneLayout = layout.getEffectiveLaneLayoutAsInt();
  if (laneLayout.size() != 2)
    return false;
  return laneLayout[0] == uArch->getSubgroupSize() && laneLayout[1] == 1;
}

// Check if dst shape is an expansion of src shape by inserting unit dimensions.
// Returns true if all dimensions in src match corresponding dimensions in dst
// (after skipping unit dimensions), and populates expandedUnitDims with the
// indices of the unit dimensions in dst that were added (not present in src).
// Example: src=[2,3], dst=[1,2,3,1] -> true, expandedUnitDims=[0,3]
bool xegpu::matchUnitDimExpansion(ArrayRef<int64_t> src, ArrayRef<int64_t> dst,
                                  SmallVector<int64_t> &expandedUnitDims) {
  // All unit dimensions in dst that don't appear in src are the expanded
  // unit dimensions
  size_t srcIdx = 0;
  for (size_t dstIdx = 0; dstIdx < dst.size(); ++dstIdx)
    if (srcIdx < src.size() && src[srcIdx] == dst[dstIdx])
      srcIdx++;
    else if (dst[dstIdx] == 1)
      expandedUnitDims.push_back(dstIdx);
    else
      return false;
  return srcIdx == src.size();
}

// Checks if dst shape is an expansion of src shape where each dimension in src
// is split into one or more consecutive dimensions in dst whose product equals
// the original dimension. Populates splitDimGroups with groups of dst indices
// that correspond to each src dimension. Example: src=[6,4], dst=[2,3,2,2] ->
// true
bool xegpu::matchSplitDimExpansion(
    ArrayRef<int64_t> src, ArrayRef<int64_t> dst,
    SmallVector<SmallVector<int64_t>> &splitDimGroups) {
  // each dim in src can be mapped to one or more dims in dst whose product
  // equals to the src dim
  size_t srcIdx = 0;
  int64_t accumulatedSize = 1;
  SmallVector<int64_t> currentDstDims;

  splitDimGroups.clear();
  for (size_t dstIdx = 0; dstIdx < dst.size(); ++dstIdx) {
    if (srcIdx >= src.size())
      return false;
    accumulatedSize *= dst[dstIdx];
    currentDstDims.push_back(dstIdx);

    if (accumulatedSize == src[srcIdx]) {
      // Also collect trailing unit dims in destination, if any.
      // Leading unit dims were implicitly collected.
      if (srcIdx == src.size() - 1) {
        while (++dstIdx < dst.size() && dst[dstIdx] == 1)
          currentDstDims.push_back(dstIdx);
      }
      // Record the mapping: srcIdx -> currentDstDims
      splitDimGroups.push_back(currentDstDims);
      // move to next src dim
      srcIdx++;
      accumulatedSize = 1;
      currentDstDims.clear();
    } else if (accumulatedSize > src[srcIdx]) {
      return false;
    }
  }
  return srcIdx == src.size();
}

//===----------------------------------------------------------------------===//
// Context-aware type conversion utilities
//===----------------------------------------------------------------------===//

// Pre-computes block argument type mappings for SCF loops (scf.while,
// scf.for).
//
// Block-arg layouts ARE available in the IR (layout recovery propagates
// them onto the loop op as `layout_operand_N`). The reason we cannot rely
// on the regular `getDistributeLayoutAttr(v)` lookup during structural
// conversion is structural, not informational:
//   - For `scf.while`, `scf::WhileOpConversion` detaches the before/after
//     blocks from their parent region before invoking
//     `convertSignatureBlock`. Looking up a detached BlockArgument's layout
//     walks `v.getParentBlock()->getParent()` and trips an LLVM ilist
//     assertion.
//   - For `scf.for`, `scf::ForOpConverter` builds a new `scf.for` and moves
//     the body block into it. The new op does NOT inherit the temporary
//     `layout_operand_N` attributes that layout recovery set on the old
//     op, so any post-move query of a body block argument's layout (e.g.
//     when a pattern that consumes the iter_arg via a non-anchor op like
//     `vector.insert_strided_slice` runs after the move) returns null.
// Caching the distributed types by `Value` identity sidesteps both failure
// modes. `scf.if` has no block arguments and is therefore not covered here.
DenseMap<Value, SmallVector<Type>>
xegpu::precomputeLoopBlockArgTypes(Operation *topLevelOp,
                                   SubShapeAndCountFn getSubShapeAndCount) {
  DenseMap<Value, SmallVector<Type>> loopArgTypes;
  auto recordBlockArgTypes = [&](Value init, BlockArgument arg) {
    auto vecTy = dyn_cast<VectorType>(init.getType());
    if (!vecTy)
      return;
    auto layout = xegpu::getDistributeLayoutAttr(init);
    if (!layout)
      return;
    auto [subShape, count] = getSubShapeAndCount(vecTy, layout);
    if (count <= 0)
      return;
    auto newTy = VectorType::get(subShape, vecTy.getElementType());
    SmallVector<Type> types(count, newTy);
    loopArgTypes[arg] = std::move(types);
  };
  topLevelOp->walk([&](Operation *op) {
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      // "before" region block arguments correspond to the `inits` operands.
      for (auto [init, arg] :
           llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments()))
        recordBlockArgTypes(init, arg);
      // "after" region block arguments correspond to the operands of the
      // embedded `scf.condition` op (not the `inits`). In general the two
      // type lists may differ.
      scf::ConditionOp condOp = whileOp.getConditionOp();
      for (auto [condArg, arg] :
           llvm::zip(condOp.getArgs(), whileOp.getAfterArguments()))
        recordBlockArgTypes(condArg, arg);
      return;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Body block args (excluding the induction variable) correspond to
      // the `initArgs` operands.
      for (auto [init, arg] :
           llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs()))
        recordBlockArgTypes(init, arg);
      return;
    }
  });
  return loopArgTypes;
}

void xegpu::addVectorTypeConversion(
    TypeConverter &converter, SubShapeAndCountFn getSubShapeAndCount,
    DenseMap<Value, SmallVector<Type>> loopArgTypes) {
  // Context-aware VectorType conversion (1:1 shape-changing or 1:N). For
  // SCF loop block arguments (scf.while, scf.for), uses the pre-computed
  // map. For all other Values, retrieves the layout directly via
  // getDistributeLayoutAttr.
  auto loopArgTypeMap = std::make_shared<DenseMap<Value, SmallVector<Type>>>(
      std::move(loopArgTypes));
  converter.addConversion(
      [loopArgTypeMap, getSubShapeAndCount](
          Value v,
          SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        if (!isa<VectorType>(v.getType()))
          return std::nullopt;

        // Check pre-computed map first (for SCF loop block args).
        if (isa<BlockArgument>(v)) {
          auto it = loopArgTypeMap->find(v);
          if (it != loopArgTypeMap->end()) {
            result.append(it->second.begin(), it->second.end());
            return success();
          }
        }

        // For OpResults and other block arguments (e.g. region args of
        // non-loop ops), retrieve the layout directly.
        auto layout = xegpu::getDistributeLayoutAttr(v);
        if (!layout)
          return std::nullopt;

        auto vecType = cast<VectorType>(v.getType());
        auto [subShape, count] = getSubShapeAndCount(vecType, layout);
        if (count <= 0)
          return std::nullopt;

        auto newTy = VectorType::get(subShape, vecType.getElementType());
        result.append(count, newTy);
        return success();
      });
}

void xegpu::cleanupUnrealizedConversionCasts(
    Operation *root,
    const llvm::SmallSetVector<UnrealizedConversionCastOp, 8> &existingCasts) {
  // Structural type conversion can generate some redundant
  // UnrealizedConversionCastOps to materialize the original type from the
  // type converted (sub-tile) type. These are redundant at this point and
  // can be eliminated by either folding the cancelling cast chain or, when
  // the original and final shapes differ but their element counts match,
  // inserting a vector.shape_cast instead.
  //
  // Example (shape differs but element count matches -> shape_cast):
  //   %1 = UnrealizedConversionCastOp %0 : vector<16x1xf32>
  //                                     to vector<16x16xf32>
  //   %2 = UnrealizedConversionCastOp %1 : vector<16x16xf32>
  //                                     to vector<16xf32>
  // becomes:
  //   %2 = vector.shape_cast %0 : vector<16x1xf32> to vector<16xf32>
  //
  // For unpaired casts that emulate a pack (1:N) or unpack (N:1) between a
  // single large VectorType and N identically-typed smaller VectorTypes,
  // lower to vector.extract_strided_slice / vector.insert_strided_slice.
  auto hasIdenticalVectorTypes = [](ValueRange values) {
    auto types = values.getTypes();
    return !types.empty() && llvm::all_of(types, [&](Type type) {
      return isa<VectorType>(type) && type == types.front();
    });
  };
  OpBuilder builder(root);
  root->walk([&](UnrealizedConversionCastOp op) {
    if (existingCasts.contains(op))
      return;
    // Handle N:1 cast (N >= 1) where all inputs come from a single 1:N cast.
    if (op.getNumResults() == 1 && op.getNumOperands() >= 1) {
      auto defOp =
          op.getInputs()[0].getDefiningOp<UnrealizedConversionCastOp>();
      if (defOp && !existingCasts.contains(defOp) &&
          defOp.getNumOperands() == 1 &&
          defOp.getNumResults() == op.getNumOperands() &&
          llvm::all_of(op.getInputs(),
                       [&](Value v) { return v.getDefiningOp() == defOp; })) {
        Value orig = defOp.getInputs()[0];
        auto origTy = dyn_cast<VectorType>(orig.getType());
        auto resTy = dyn_cast<VectorType>(op.getResult(0).getType());
        if (origTy && resTy &&
            origTy.getNumElements() == resTy.getNumElements() &&
            origTy != resTy) {
          builder.setInsertionPoint(op);
          auto shapeCast =
              vector::ShapeCastOp::create(builder, op.getLoc(), resTy, orig);
          op.replaceAllUsesWith(ValueRange{shapeCast.getResult()});
        } else {
          op.replaceAllUsesWith(ValueRange{orig});
        }
        return;
      }
      // Unpaired N:1 cast emulating unpack: stitch inputs into the output
      // shape via vector.insert_strided_slice.
      auto outputTy = dyn_cast<VectorType>(op.getResult(0).getType());
      if (op.getNumOperands() > 1 && outputTy &&
          hasIdenticalVectorTypes(op.getInputs())) {
        builder.setInsertionPoint(op);
        Value result = xegpu::createVectorWithShapeFromValues(
            builder, op.getLoc(), op.getInputs(), outputTy.getShape());
        op->replaceAllUsesWith(ValueRange(result));
      }
      return;
    }
    // Handle 1:N cast where the single input comes from an N:1 cast.
    if (op.getNumOperands() == 1 && op.getNumResults() > 1) {
      auto defOp =
          op.getInputs()[0].getDefiningOp<UnrealizedConversionCastOp>();
      if (defOp && !existingCasts.contains(defOp) &&
          defOp.getNumResults() == 1 &&
          defOp.getNumOperands() == op.getNumResults() &&
          llvm::equal(ValueRange(defOp.getInputs()).getTypes(),
                      op->getResultTypes())) {
        op.replaceAllUsesWith(defOp.getInputs());
        return;
      }
      // Unpaired 1:N cast emulating pack: split the input into the output
      // tile shape via vector.extract_strided_slice.
      auto tileTy = dyn_cast<VectorType>(op.getResult(0).getType());
      if (tileTy && hasIdenticalVectorTypes(op.getResults())) {
        builder.setInsertionPoint(op);
        SmallVector<Value> results = xegpu::extractVectorsWithShapeFromValue(
            builder, op.getLoc(), op.getInputs()[0], tileTy.getShape());
        op->replaceAllUsesWith(results);
      }
      return;
    }
  });

  // Erase dead casts iteratively.
  bool changed = true;
  while (changed) {
    changed = false;
    root->walk([&](UnrealizedConversionCastOp op) {
      if (existingCasts.contains(op))
        return;
      if (op.use_empty()) {
        op.erase();
        changed = true;
      }
    });
  }
}
