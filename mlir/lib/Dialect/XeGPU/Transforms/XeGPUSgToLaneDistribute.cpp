//===- XeGPUSgToLaneDistribute.cpp - XeGPU SG to Lane Pass ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpl.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/XeGPU/uArch/uArchCommon.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUSGTOLANEDISTRIBUTE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "xegpu-sg-to-lane-distribute"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace {

/// Casts the given vector value `v` to the expected vector type `expectedTy`.
static Value castValueTo(ConversionPatternRewriter &rewriter,
                         TypedValue<VectorType> v, VectorType expectedTy) {
  // If the type matches, simply return the value itself.
  if (v.getType() == expectedTy)
    return v;
  // If only shape differs, use shape cast.
  if (isa<VectorType>(v.getType()) &&
      v.getType().getNumElements() == expectedTy.getNumElements())
    return vector::ShapeCastOp::create(rewriter, v.getLoc(), expectedTy, v);

  // Else create an unrealized cast.
  auto newOp = UnrealizedConversionCastOp::create(rewriter, v.getLoc(),
                                                  expectedTy, ValueRange{v});
  return newOp.getResult(0);
}

/// A vector::MultiDimReductionOp at subgroup level in expected form if, it has
/// exactly 1 reduction dimension, it had valid result layout attribute, and
/// result type can be distributed to lanes using the layout.
static bool isValidSubgroupMultiReductionOp(vector::MultiDimReductionOp op) {
  auto resLayout = xegpu::getTemporaryLayout(op->getOpResult(0));
  // If no layout, not valid.
  if (!resLayout || !resLayout.isForSubgroup())
    return false;
  // Scalar result (e.g., vector<32xf32> to f32) is valid.
  if (op.getType().isIntOrFloat())
    return op.getReductionDims().size() == 1;
  VectorType resTy = dyn_cast<VectorType>(op.getType());
  if (!resTy)
    return false;
  // Compute the distributed result vector type based on the layout.
  FailureOr<VectorType> resDistTypeOrFailure =
      getDistVecTypeBasedOnLaneLayout(resLayout, resTy);
  if (failed(resDistTypeOrFailure))
    return false;
  return op.getReductionDims().size() == 1;
}

/// A vector::MultiDimReductionOp is doing lane-local reduction if each lane
/// is doing its own local reduction. In this case the result layout ensures
/// that result vector is distributed to lanes, i.e. the result vector type is
/// different from the distributed result vector type.
static bool isReductionLaneLocal(vector::MultiDimReductionOp op) {
  // Must be valid MultiDimReductionOp.
  assert(isValidSubgroupMultiReductionOp(op) && "Expecting a valid subgroup "
                                                "MultiDimReductionOp");
  auto resLayout = xegpu::getTemporaryLayout(op->getOpResult(0));
  VectorType resTy = dyn_cast<VectorType>(op.getType());
  auto resDistTypeOrFailure = getDistVecTypeBasedOnLaneLayout(resLayout, resTy);
  return resTy != resDistTypeOrFailure.value();
}

/// Given a vector type and its distributed vector type, return the list of
/// dimensions that are distributed.
static SmallVector<int64_t> getDistributedDims(VectorType originalType,
                                               VectorType distributedType) {
  assert(originalType.getRank() == distributedType.getRank() &&
         "original and distributed vector types must have the same rank");
  SmallVector<int64_t> distributedDims;
  for (int64_t i = 0; i < originalType.getRank(); ++i) {
    if (distributedType.getDimSize(i) != originalType.getDimSize(i))
      distributedDims.push_back(i);
  }
  return distributedDims;
}

/// Distributes a subgroup-level CreateNdDesc op to lane-level CreateNdDesc
/// op. This simply drops the layout attribute from the tensor descriptor type.
struct SgToLaneCreateNdDesc
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::TensorDescType resultType = op.getType();
    // If no layout, nothing to do.
    if (!resultType.getLayout())
      return failure();

    auto newOp = xegpu::CreateNdDescOp::create(
        rewriter, op.getLoc(), resultType.dropLayouts(), op.getOperands(),
        op->getAttrs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Distributes a subgroup-level LoadNd op to lane-level LoadNd op. Output
/// of lane-level LoadNd op is 1D. ShapeCast is added to restore the
/// original rank.
struct SgToLaneLoadNd : public OpConversionPattern<xegpu::LoadNdOp> {
  using OpConversionPattern<xegpu::LoadNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::LoadNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout = op.getAnchorLayout();
    // If no layout, nothing to do.
    if (!layout)
      return failure();
    // Check if the layout attached to the tensor descriptor is same as the
    // anchor layout. Otherwise, this is a conflict.
    if (op.getTensorDescType().getLayout() != layout)
      return rewriter.notifyMatchFailure(
          op, "conflicting layout attributes on tensor descriptor and anchor");
    const auto *uArch =
        xegpu::uArch::getUArch(xegpu::getChipStr(op).value_or(""));
    if (!uArch)
      return rewriter.notifyMatchFailure(
          op, "xegpu::LoadNdOp require target attribute attached to "
              "determine transpose "
              "requirement");
    auto supportedLaneResultTyOrFailure =
        xegpu::getDistributedVectorType(op.getTensorDescType());
    auto expectedLaneResultTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, op.getType());
    if (failed(supportedLaneResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute the lane vector type for LoadNdOp");
    if (failed(expectedLaneResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute expected lane vector type from lane layout");
    auto newOp = xegpu::LoadNdOp::create(
        rewriter, op.getLoc(), supportedLaneResultTyOrFailure.value(),
        adaptor.getTensorDesc(), op.getMixedOffsets(), op.getPackedAttr(),
        op.getTransposeAttr(), op.getL1HintAttr(), op.getL2HintAttr(),
        op.getL3HintAttr(), /**layout**/ nullptr);
    // Set the packed attribute if the layout requires it.
    newOp.setPacked(xegpu::requirePacked(cast<xegpu::LayoutAttr>(layout)));
    // Set the transpose attribute if the layout requires it.
    if (xegpu::requireTranspose(cast<xegpu::LayoutAttr>(layout), uArch))
      newOp.setTranspose(DenseI64ArrayAttr::get(rewriter.getContext(), {1, 0}));
    rewriter.replaceOp(op, castValueTo(rewriter, newOp.getResult(),
                                       expectedLaneResultTyOrFailure.value()));
    return success();
  }
};

/// Distributes a subgroup-level StoreNd op to lane-level StoreNd op. Stored
/// value in lane-level StoreNd op is 1D. ShapeCast is added to cast the
/// incoming value to 1D.
struct SgToLaneStoreNd : public OpConversionPattern<xegpu::StoreNdOp> {
  using OpConversionPattern<xegpu::StoreNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::StoreNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout = op.getAnchorLayout();
    // If no layout, nothing to do.
    if (!layout)
      return failure();
    // Check if the layout attached to the tensor descriptor and value layout is
    // same as the anchor layout. Otherwise, this is a conflict.
    if (op.getTensorDescType().getLayout() != layout)
      return rewriter.notifyMatchFailure(
          op, "conflicting layout attributes on tensor descriptor and anchor");
    auto valueLayout = xegpu::getDistributeLayoutAttr(op->getOpOperand(0));
    if (valueLayout != layout)
      return rewriter.notifyMatchFailure(
          op, "conflicting layout attributes on value and anchor");
    auto supportedLaneValueTyOrFailure =
        xegpu::getDistributedVectorType(op.getTensorDescType());
    if (failed(supportedLaneValueTyOrFailure))
      return rewriter.notifyMatchFailure(
          op,
          "unable to compute lane vector type for StoreNdOp value from tensor "
          "descriptor");

    xegpu::StoreNdOp::create(
        rewriter, op.getLoc(),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getValue()),
                    supportedLaneValueTyOrFailure.value()),
        adaptor.getTensorDesc(), op.getMixedOffsets(), op.getL1HintAttr(),
        op.getL2HintAttr(), op.getL3HintAttr(), /**layout**/ nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Distributes a subgroup-level Dpas op to lane-level Dpas op. All inpputs
/// and output of lane-level Dpas op are 1D. Necessary casts are added to
/// convert the inputs and output to/from 1D.
struct SgToLaneDpas : public OpConversionPattern<xegpu::DpasOp> {
  using OpConversionPattern<xegpu::DpasOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::DpasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the op has A, B and CD layouts attached.
    auto layoutA = cast<xegpu::LayoutAttr>(op.getLayoutAAttr());
    auto layoutB = cast<xegpu::LayoutAttr>(op.getLayoutBAttr());
    auto layoutCd = cast<xegpu::LayoutAttr>(op.getLayoutCdAttr());
    if (!layoutA || !layoutB || !layoutCd)
      return failure();
    auto laneResultTyOrFailure =
        xegpu::getDistributedVectorType(op.getType(), layoutCd);
    auto laneATypeOrFailure =
        xegpu::getDistributedVectorType(op.getLhs().getType(), layoutA);
    auto laneBTypeOrFailure =
        xegpu::getDistributedVectorType(op.getRhs().getType(), layoutB);
    auto expectedLaneResultTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layoutCd, op.getType());
    if (failed(laneResultTyOrFailure) || failed(laneATypeOrFailure) ||
        failed(laneBTypeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to calculate supported lane vector types for DpasOp "
              "from layouts");
    if (failed(expectedLaneResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute expected lane vector type for DpasOp from "
              "lane layout");

    // Validate bit widths match uArch packed format requirements
    const auto *uArch =
        xegpu::uArch::getUArch(xegpu::getChipStr(op).value_or(""));
    if (uArch) {
      const auto *uArchInstruction =
          dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(
              uArch->getInstruction(
                  xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
      if (uArchInstruction) {
        auto laneAType = laneATypeOrFailure.value();
        auto laneBType = laneBTypeOrFailure.value();
        // Calculate total packed bit width = element bit width * vector size
        unsigned aPackedBitWidth =
            laneAType.getElementTypeBitWidth() * laneAType.getNumElements();
        unsigned bPackedBitWidth =
            laneBType.getElementTypeBitWidth() * laneBType.getNumElements();
        unsigned expectedABitSize = uArchInstruction->getPackedFormatBitSizeA();
        unsigned expectedBBitSize = uArchInstruction->getPackedFormatBitSizeB();

        if (aPackedBitWidth % expectedABitSize != 0)
          return rewriter.notifyMatchFailure(
              op,
              "A operand packed bit width must be a multiple of uArch packed "
              "format requirement");
        if (bPackedBitWidth % expectedBBitSize != 0)
          return rewriter.notifyMatchFailure(
              op,
              "B operand packed bit width must be a multiple of uArch packed "
              "format requirement");
      }
    }

    auto newOp = xegpu::DpasOp::create(
        rewriter, op->getLoc(), laneResultTyOrFailure.value(),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getLhs()),
                    laneATypeOrFailure.value()),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getRhs()),
                    laneBTypeOrFailure.value()),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getAcc()),
                    laneResultTyOrFailure.value()),
        /** layoutA**/ nullptr,
        /** layoutB**/ nullptr, /** layoutCd**/ nullptr);
    // Explicitly set the new types to enable correct type materializations.
    rewriter.replaceOp(op, castValueTo(rewriter, newOp.getResult(),
                                       expectedLaneResultTyOrFailure.value()));
    return success();
  }
};

/// Distributes elementwise ops to lane-level elementwise ops. This
/// currently handles elementwise ops with single result only.
struct SgToLaneElementWise : public ConversionPattern {
  SgToLaneElementWise(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match ops with elementwise trait and single result.
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();

    auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "operation result is not a vector type");

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(llvm::cast<OpResult>(op->getResult(0)));
    if (!layout || !layout.isForSubgroup())
      return rewriter.notifyMatchFailure(
          op, "operation result does not have subgroup distribute layout");

    auto laneShapeOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, resultType);

    if (failed(laneShapeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute lane vector type from the layout");

    VectorType newResultType = laneShapeOrFailure.value();
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(newResultType);
    // Copy all attributes except for DistributeLayoutAttr.
    for (auto attr : op->getAttrs()) {
      if (!isa<xegpu::DistributeLayoutAttr>(attr.getValue()))
        state.addAttribute(attr.getName(), attr.getValue());
    }
    Operation *newOp = rewriter.create(state);

    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

/// Distributes a subgroup-level arith ConstantOp to lane-level arith
/// ConstantOp.
///
/// Splat constants are rebuilt with the lane-local vector type. Non-splat
/// constants are distributed by extracting each lane_data-sized block from
/// the full constant and inserting it at the correct position in the
/// distributed vector using insert_strided_slice.
struct SgToLaneArithConstant : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return failure();

    // Only handle dense vector constants.
    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr)
      return rewriter.notifyMatchFailure(
          op, "only dense vector constants are supported");

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(llvm::cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForSubgroup())
      return rewriter.notifyMatchFailure(
          op, "operation result does not have subgroup distribute layout");

    auto laneShapeOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, resultType);

    if (failed(laneShapeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute lane vector type from the layout");

    VectorType newResultType = laneShapeOrFailure.value();
    Location loc = op.getLoc();

    // Splat constants: every lane gets the same value, so just rebuild the
    // splat with the distributed type.
    if (denseAttr.isSplat()) {
      auto scalarValue = denseAttr.getSplatValue<Attribute>();
      auto newDenseAttr = DenseElementsAttr::get(newResultType, scalarValue);
      auto newOp =
          arith::ConstantOp::create(rewriter, loc, newResultType, newDenseAttr);
      rewriter.replaceOp(op, newOp.getResult());
      return success();
    }

    // Non-splat constants: each lane extracts the elements it owns from the
    // full constant using the distributed coordinates from the layout.
    auto fullConst =
        arith::ConstantOp::create(rewriter, loc, resultType, denseAttr);

    Value laneId = gpu::LaneIdOp::create(rewriter, loc, rewriter.getIndexType(),
                                         /*upperBound=*/mlir::IntegerAttr());
    auto maybeCoordsVec = layout.computeDistributedCoords(
        rewriter, loc, laneId, resultType.getShape());
    if (failed(maybeCoordsVec))
      return rewriter.notifyMatchFailure(
          op, "failed to compute distributed coordinates from layout");

    SmallVector<SmallVector<Value>> coordsVec = maybeCoordsVec.value();
    SmallVector<int64_t> laneData = layout.getEffectiveLaneDataAsInt();
    ArrayRef<int64_t> distShape = newResultType.getShape();
    int64_t rank = newResultType.getRank();

    // Each lane owns one lane_data-sized block per distribution unit.
    // computeDistributedCoords returns those block starts in row-major order
    // over the block grid (distShape / laneData).
    SmallVector<int64_t> blockGridShape(rank);
    for (int64_t d = 0; d < rank; d++)
      blockGridShape[d] = distShape[d] / laneData[d];
    SmallVector<int64_t> blockGridStrides = computeStrides(blockGridShape);

    auto blockType = VectorType::get(laneData, newResultType.getElementType());
    SmallVector<int64_t> unitTile(rank, 1);
    SmallVector<int64_t> strides(rank, 1);

    Value result = arith::ConstantOp::create(
        rewriter, loc, newResultType, rewriter.getZeroAttr(newResultType));

    for (auto [blockIdx, blockStart] : llvm::enumerate(coordsVec)) {
      // Gather the block's elements from the full constant. The block start is
      // lane-dynamic, so extract element-by-element (row-major over lane_data)
      // instead.
      SmallVector<Value> blockElems;
      for (SmallVector<int64_t> off :
           StaticTileOffsetRange(laneData, unitTile)) {
        SmallVector<OpFoldResult> pos(rank);
        for (int64_t d = 0; d < rank; d++)
          pos[d] = getAsOpFoldResult(arith::AddIOp::create(
              rewriter, loc, blockStart[d],
              arith::ConstantIndexOp::create(rewriter, loc, off[d])));
        blockElems.push_back(vector::ExtractOp::create(
            rewriter, loc, fullConst.getResult(), pos));
      }

      // Rebuild the block keeping its lane_data shape, then place it with
      // insert_strided_slice so the block keeps its orientation in the
      // distributed vector (e.g. a [2, 1] block stays a vertical 2x1 slice).
      Value block =
          vector::FromElementsOp::create(rewriter, loc, blockType, blockElems);
      SmallVector<int64_t> blockGridPos =
          delinearize(blockIdx, blockGridStrides);
      SmallVector<int64_t> offsets(rank);
      for (int64_t d = 0; d < rank; d++)
        offsets[d] = blockGridPos[d] * laneData[d];
      result = vector::InsertStridedSliceOp::create(rewriter, loc, block,
                                                    result, offsets, strides);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Distributes a subgroup-level PrefetchNd op to lane-level PrefetchNd op.
struct SgToLanePrefetchNd : public OpConversionPattern<xegpu::PrefetchNdOp> {
  using OpConversionPattern<xegpu::PrefetchNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::PrefetchNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout = op.getAnchorLayout();
    // If no layout, nothing to do.
    if (!layout)
      return failure();

    xegpu::PrefetchNdOp::create(rewriter, op.getLoc(), adaptor.getTensorDesc(),
                                op.getMixedOffsets(), op.getL1HintAttr(),
                                op.getL2HintAttr(), op.getL3HintAttr(),
                                /**layout**/ nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Distributes a subgroup-level LoadGather (xegpu.load) op to lane-level.
///
/// Example 1 (1D, no chunk size):
///   layout = #xegpu.layout<lane_layout = [16], lane_data = [1]>
///   %mask = producer_op : vector<16xi1>
///   %offset = producer_op : vector<16xindex>
///   %0 = xegpu.load %src[%offset], %mask : memref<256xf16>,
///     vector<16xindex>, vector<16xi1> -> vector<16xf16>
/// Distributed to:
///   %mask = producer_op : vector<1xi1>
///   %offset = producer_op : vector<1xindex>
///   %0 = xegpu.load %src[%offset], %mask : memref<256xf16>,
///     vector<1xindex>, vector<1xi1> -> vector<1xf16>
///
/// Example 2 (2D with chunk size, same mask & offset):
///   layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>
///   %0 = xegpu.load %src[%offset], %mask <{chunk_size=8}> :
///     memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
/// Distributed to:
///   %0 = xegpu.load %src[%offset], %mask <{chunk_size=8}> :
///     memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<8xf16>
///
/// Example 3 (3D with leading unit dims):
///   layout = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>
///   %mask = producer_op : vector<1x1x16xi1>
///   %offset = producer_op : vector<1x1x16xindex>
///   %0 = xegpu.load %src[%offset], %mask : memref<256xf16>,
///     vector<1x1x16xindex>, vector<1x1x16xi1> -> vector<1x1x16xf16>
/// Distributed to:
///   %mask = producer_op : vector<1x1x1xi1>
///   %offset = producer_op : vector<1x1x1xindex>
///   %0 = xegpu.load %src[%offset], %mask : memref<256xf16>,
///     vector<1xindex>, vector<1xi1> -> vector<1xf16>
struct SgToLaneLoadGather : public OpConversionPattern<xegpu::LoadGatherOp> {
  using OpConversionPattern<xegpu::LoadGatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::LoadGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout = op.getAnchorLayout();
    if (!layout)
      return failure();

    VectorType origResultTy = op.getValueType();
    if (!origResultTy)
      return failure();

    // Check that leading dimensions are unit.
    int chunkSize = op.getChunkSize().value_or(1);
    int effectiveVecRank = (chunkSize == 1) ? 1 : 2;
    ArrayRef<int64_t> shape = origResultTy.getShape();
    if (llvm::any_of(
            shape.take_front(origResultTy.getRank() - effectiveVecRank),
            [](int64_t d) { return d != 1; }))
      return rewriter.notifyMatchFailure(
          op, "Only unit dimensions allowed for the leading "
              "dimensions of the load vector!");

    auto distResultTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, origResultTy);
    if (failed(distResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute expected lane vector type from lane layout");

    VectorType distResultTy = distResultTyOrFailure.value();
    VectorType distResultTy1D = VectorType::get({distResultTy.getNumElements()},
                                                distResultTy.getElementType());

    // Flatten offsets and mask to 1D to match the 1D result type.
    Value distOffsets = adaptor.getOffsets();
    auto distOffsetsTy = cast<VectorType>(distOffsets.getType());
    VectorType offsetsTy1D = VectorType::get({distOffsetsTy.getNumElements()},
                                             distOffsetsTy.getElementType());
    distOffsets = castValueTo(
        rewriter, cast<TypedValue<VectorType>>(distOffsets), offsetsTy1D);

    Value distMask = adaptor.getMask();
    auto distMaskTy = cast<VectorType>(distMask.getType());
    VectorType maskTy1D = VectorType::get({distMaskTy.getNumElements()},
                                          distMaskTy.getElementType());
    distMask =
        castValueTo(rewriter, cast<TypedValue<VectorType>>(distMask), maskTy1D);

    Value distSource = adaptor.getSource();
    auto newOp = xegpu::LoadGatherOp::create(
        rewriter, op.getLoc(), distResultTy1D, distSource, distOffsets,
        distMask, op.getChunkSizeAttr(), op.getL1HintAttr(), op.getL2HintAttr(),
        op.getL3HintAttr(), /*layout=*/nullptr, /*contiguity=*/nullptr);

    Value result = newOp->getResult(0);
    if (distResultTy1D != distResultTy)
      result = castValueTo(rewriter, cast<TypedValue<VectorType>>(result),
                           distResultTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// This pattern distributes a subgroup-level vector.reduction op to
/// lane-level. This require shuffling the data across the lanes (using
/// gpu::ShuffleOp) and reducing in stages until all lanes have the final
/// result.
struct SgToLaneVectorReduction
    : public OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern<vector::ReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto layout = xegpu::getDistributeLayoutAttr(op.getVector());

    // If no layout, nothing to do.
    if (!layout || !layout.isForSubgroup())
      return failure();

    VectorType srcVecType = op.getSourceVectorType();
    // Only rank 1 vectors supported.
    if (srcVecType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "Only rank 1 reductions can be distributed.");
    // Lane layout must have the same rank as the vector.
    if (layout.getRank() != srcVecType.getRank())
      return rewriter.notifyMatchFailure(
          op, "Layout rank does not match vector rank.");

    // Get the subgroup size from the layout.
    int64_t sgSize = layout.getEffectiveLaneLayoutAsInt()[0];
    const auto *uArch =
        xegpu::uArch::getUArch(xegpu::getChipStr(op).value_or(""));
    if (!uArch)
      return rewriter.notifyMatchFailure(
          op, "xegpu::ReductionOp require target attribute attached to "
              "determine subgroup size");

    // Only subgroup-sized vectors supported.
    if (sgSize != uArch->getSubgroupSize() ||
        srcVecType.getShape()[0] % sgSize != 0)
      return rewriter.notifyMatchFailure(op,
                                         "Invalid layout or reduction vector "
                                         "dimension must match subgroup size.");

    if (!op.getType().isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Reduction distribution currently only supports floats and "
              "integer types.");

    // Get the distributed vector (per lane portion).
    Value laneValVec = adaptor.getVector();

    // Distribute and reduce across lanes in the subgroup.
    Value fullReduce = xegpu::subgroupReduction(
        op.getLoc(), rewriter, laneValVec, op.getKind(), sgSize);

    // If there's an accumulator, combine it with the reduced value.
    if (adaptor.getAcc())
      fullReduce = vector::makeArithReduction(
          rewriter, op.getLoc(), op.getKind(), fullReduce, adaptor.getAcc());

    rewriter.replaceOp(op, fullReduce);
    return success();
  }
};

/// This pattern distributes a subgroup-level vector.multi_reduction op to
/// lane-level only if the reduction is lane-local. This means that
/// reduction dimension is not distributed to lanes and each lane does its own
/// local reduction.
struct SgToLaneMultiDimReduction
    : public OpConversionPattern<vector::MultiDimReductionOp> {
  using OpConversionPattern<vector::MultiDimReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MultiDimReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value result;
    ArrayRef<int64_t> reductionDims = op.getReductionDims();
    assert(reductionDims.size() == 1 &&
           "Expecting single reduction dimension for subgroup multi "
           "reduction op");
    // For rank > 2, ensure leading dimensions are unit.
    VectorType sourceType = op.getSourceVectorType();
    int64_t rank = sourceType.getRank();
    if (rank > 2) {
      ArrayRef<int64_t> shape = sourceType.getShape();
      if (llvm::any_of(shape.take_front(rank - 2),
                       [](int64_t d) { return d != 1; }))
        return rewriter.notifyMatchFailure(
            op, "only unit leading dimensions are supported for "
                "multi_reduction with rank > 2");
    }
    // Handle scalar result: full reduction of a distributed vector to a
    // scalar. First do a local vector reduction, then cross-lane shuffles.
    if (op.getType().isIntOrFloat()) {
      auto reductionDim = reductionDims[0];
      VectorType origSourceType = op.getSourceVectorType();
      int64_t reductionDimSize = origSourceType.getShape()[reductionDim];
      // Local reduction to scalar, then cross-lane butterfly shuffles.
      result =
          xegpu::subgroupReduction(op.getLoc(), rewriter, adaptor.getSource(),
                                   op.getKind(), reductionDimSize);
      // Combine with accumulator if present.
      if (adaptor.getAcc())
        result = vector::makeArithReduction(rewriter, op.getLoc(), op.getKind(),
                                            result, adaptor.getAcc());
    } else if (isReductionLaneLocal(op)) {
      // For lane-local reduction, lower to a sequence of vector.reduction ops
      // over 1D slices extracted from the distributed source vector. This is
      // required so we dont have 2D source vectors at xegpu-linearize.
      auto reductionDim = reductionDims[0];
      result = xegpu::lowerToVectorReductions(
          cast<TypedValue<VectorType>>(adaptor.getSource()),
          cast<TypedValue<VectorType>>(adaptor.getAcc()), op.getKind(),
          reductionDim, op.getLoc(), rewriter);
    } else {
      auto reductionDim = reductionDims[0];
      VectorType sourceType = op.getSourceVectorType();
      int64_t reductionDimSize = sourceType.getShape()[reductionDim];
      result = xegpu::lowerCrossLaneReductionToShuffles(
          cast<TypedValue<VectorType>>(adaptor.getSource()),
          cast<TypedValue<VectorType>>(adaptor.getAcc()), op.getKind(),
          reductionDim, reductionDimSize, op.getLoc(), rewriter);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Helper to compute distributed coordinates for matrix ops.
/// When not using subgroup_block_io, each lane computes its own
/// coordinates based on the layout and lane ID.
static SmallVector<Value> computeDistributedCoordsForMatrixOp(
    ConversionPatternRewriter &rewriter, Location loc,
    xegpu::DistributeLayoutAttr layout, ArrayRef<int64_t> payloadShape,
    ValueRange origOffsets) {
  Value laneId = gpu::LaneIdOp::create(rewriter, loc, rewriter.getIndexType(),
                                       /*upperBound=*/mlir::IntegerAttr());
  auto maybeCoords =
      layout.computeDistributedCoords(rewriter, loc, laneId, payloadShape);
  if (failed(maybeCoords))
    return {};
  assert(maybeCoords.value().size() == 1 &&
         "Expected one set of distributed offsets");
  SmallVector<OpFoldResult> ofrVec = xegpu::addWithRightAligned(
      rewriter, loc, getAsOpFoldResult(maybeCoords.value()[0]),
      getAsOpFoldResult(origOffsets));
  return llvm::map_to_vector(ofrVec, llvm::CastTo<Value>);
}

/// This pattern distributes a subgroup-level LoadMatrix op to lane-level.
struct SgToLaneLoadMatrix : public OpConversionPattern<xegpu::LoadMatrixOp> {
  using OpConversionPattern<xegpu::LoadMatrixOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::LoadMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto layout = op.getLayoutAttr();
    // If no layout, nothing to do.
    if (!layout)
      return failure();

    VectorType sgPayloadTy = dyn_cast<VectorType>(op.getResult().getType());
    if (!sgPayloadTy)
      return rewriter.notifyMatchFailure(
          op, "the matrix op payload must be a vector type");

    auto loc = op.getLoc();
    auto offsets = op.getMixedOffsets();
    if (offsets.empty())
      return rewriter.notifyMatchFailure(op, "the load op must have offsets");

    FailureOr<VectorType> distPayloadTyOrFailure =
        getDistVecTypeBasedOnLaneLayout(layout, sgPayloadTy);
    if (failed(distPayloadTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "Failed to distribute matrix op payload based on layout.");

    SmallVector<Value> offsetsAsValues =
        vector::getAsValues(rewriter, loc, offsets);

    SmallVector<Value> newCoords = offsetsAsValues;
    if (!op.getSubgroupBlockIoAttr()) {
      newCoords = computeDistributedCoordsForMatrixOp(
          rewriter, loc, layout, sgPayloadTy.getShape(), offsetsAsValues);
      if (newCoords.empty())
        return rewriter.notifyMatchFailure(
            op, "Failed to compute distributed coordinates.");
    }

    SmallVector<int64_t> newConstOffsets(op.getConstOffsets().size(),
                                         ShapedType::kDynamic);
    DenseI64ArrayAttr newConstOffsetsAttr =
        rewriter.getDenseI64ArrayAttr(newConstOffsets);

    auto newOp = xegpu::LoadMatrixOp::create(
        rewriter, loc, *distPayloadTyOrFailure, adaptor.getMemDesc(),
        ValueRange(newCoords), newConstOffsetsAttr, op.getSubgroupBlockIoAttr(),
        xegpu::DistributeLayoutAttr{});
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Distributes a subgroup-level vector.transpose op to lane-level.
struct SgToLaneVectorTranspose
    : public OpConversionPattern<vector::TransposeOp> {
  using OpConversionPattern<vector::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr sourceLayout =
        xegpu::getTemporaryLayout(op->getOpOperand(0));
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getTemporaryLayout(op->getOpResult(0));
    if (!sourceLayout || !resultLayout)
      return rewriter.notifyMatchFailure(
          op, "the source or result vector of the transpose op lacks layout "
              "attribute");
    ArrayRef<int64_t> perm = op.getPermutation();
    // Result layout must be a transpose of source layout.
    if (!resultLayout.isTransposeOf(sourceLayout, perm,
                                    xegpu::LayoutKind::Lane))
      return rewriter.notifyMatchFailure(
          op, "the source or result vector layouts must be transposes of "
              "each other");
    FailureOr<VectorType> distributedResultTypeOrFailure =
        getDistVecTypeBasedOnLaneLayout(resultLayout, op.getResultVectorType());
    if (failed(distributedResultTypeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "Failed to distribute the result vector type in "
              "vector::Transpose op");
    auto newOp = vector::TransposeOp::create(rewriter, op.getLoc(),
                                             adaptor.getVector(), perm);
    rewriter.replaceOp(op, castValueTo(rewriter, newOp.getResult(),
                                       distributedResultTypeOrFailure.value()));
    return success();
  }
};

/// Distributes a subgroup-level vector.bitcast op to lane-level.
/// Bitcast only impacts the innermost dimension of the source/result vectors.
struct SgToLaneVectorBitcast : public OpConversionPattern<vector::BitCastOp> {
  using OpConversionPattern<vector::BitCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BitCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getTemporaryLayout(op->getOpResult(0));
    if (!resultLayout)
      return rewriter.notifyMatchFailure(
          op, "result vector of the bitcast op lacks layout attribute");
    FailureOr<VectorType> distributedResultTypeOrFailure =
        getDistVecTypeBasedOnLaneLayout(resultLayout, op.getResultVectorType());
    if (failed(distributedResultTypeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "Failed to distribute the result vector type in "
              "vector::BitCast op");
    auto newOp = vector::BitCastOp::create(
        rewriter, op.getLoc(), distributedResultTypeOrFailure.value(),
        adaptor.getSource());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Distributes a subgroup-level vector.create_mask or vector.constant_mask op
/// to lane-level. Uses `computeDistributedCoords()` to obtain the
/// coordinates each lane owns, then compares each coordinate against the
/// original mask bounds using `arith.cmpi slt`. The per-element boolean
/// results are assembled into the distributed mask vector.
///
/// For multi-dimensional masks, the element is in-bounds when ALL dimensions
/// satisfy `coord[i] < bound[i]`.
///
/// Example (1D):
///   layout = #xegpu.layout<lane_layout = [16], lane_data = [1]>
///   %mask = vector.create_mask %m0 : vector<16xi1>
/// For lane k, computeDistributedCoords gives coord = [k], so:
///   %in_bounds = arith.cmpi slt, %coord, %m0  →  i1
///   %mask = vector.broadcast %in_bounds : i1 to vector<1xi1>
///
/// Example (2D):
///   layout = #xegpu.layout<lane_layout = [8, 2], lane_data = [1, 1]>
///   %mask = vector.create_mask %m0, %m1 : vector<8x4xi1>
/// Each WI owns a 1x2 slice. computeDistributedCoords returns 2 coords:
///   [[r0, c0], [r0, c1]]
/// For each coord: in_bounds = (r < m0) && (c < m1)
///   %mask = vector.from_elements %bit0, %bit1 : vector<1x2xi1>
template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<
              OpType, vector::CreateMaskOp, vector::ConstantMaskOp>::value>>
struct SgToLaneCreateMask : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(op->getOpResult(0));
    if (!layout || !layout.isForSubgroup())
      return rewriter.notifyMatchFailure(
          op, "operation result does not have subgroup distribute layout");

    VectorType origType = op.getType();
    FailureOr<VectorType> distTypeOrFailure =
        getDistVecTypeBasedOnLaneLayout(layout, origType);
    if (failed(distTypeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute lane vector type from the layout");

    VectorType distType = distTypeOrFailure.value();
    Location loc = op.getLoc();

    // Materialize the original mask bounds as Values.
    SmallVector<Value> origBounds;
    if constexpr (std::is_same_v<OpType, vector::CreateMaskOp>) {
      origBounds.append(op.getOperands().begin(), op.getOperands().end());
    } else {
      auto dimSizes = op.getMaskDimSizesAttr().asArrayRef();
      for (auto dimSize : dimSizes)
        origBounds.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, dimSize).getResult());
    }

    ArrayRef<int64_t> origShape = origType.getShape();

    // Use computeDistributedCoords to get the coordinates each WI owns.
    Value laneId = gpu::LaneIdOp::create(rewriter, loc, rewriter.getIndexType(),
                                         /*upperBound=*/mlir::IntegerAttr());
    auto maybeCoordsVec =
        layout.computeDistributedCoords(rewriter, loc, laneId, origShape);
    if (failed(maybeCoordsVec))
      return rewriter.notifyMatchFailure(
          op, "failed to compute distributed coordinates from layout");

    SmallVector<SmallVector<Value>> coordsVec = maybeCoordsVec.value();
    int64_t numElements = distType.getNumElements();
    assert(static_cast<int64_t>(coordsVec.size()) == numElements &&
           "number of coordinate sets must match number of distributed "
           "elements");

    // For each element, compare all coordinates against bounds.
    Value trueVal =
        arith::ConstantIntOp::create(rewriter, loc, /*value=*/1, /*width=*/1);
    SmallVector<Value> maskBits;
    for (auto &coords : coordsVec) {
      Value inBounds = trueVal;
      for (size_t i = 0; i < coords.size(); ++i) {
        Value cmp = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::slt, coords[i], origBounds[i]);
        inBounds = arith::AndIOp::create(rewriter, loc, inBounds, cmp);
      }
      maskBits.push_back(inBounds);
    }

    // Build the distributed mask vector.
    Value result;
    if (numElements == 1) {
      result =
          vector::BroadcastOp::create(rewriter, loc, distType, maskBits[0]);
    } else {
      result =
          vector::FromElementsOp::create(rewriter, loc, distType, maskBits);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// This pattern distributes a subgroup-level StoreMatrix op to lane-level.
struct SgToLaneStoreMatrix : public OpConversionPattern<xegpu::StoreMatrixOp> {
  using OpConversionPattern<xegpu::StoreMatrixOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::StoreMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto layout = op.getLayoutAttr();
    // If no layout, nothing to do.
    if (!layout)
      return failure();

    VectorType sgPayloadTy = dyn_cast<VectorType>(op.getData().getType());
    if (!sgPayloadTy)
      return rewriter.notifyMatchFailure(
          op, "the matrix op payload must be a vector type");

    auto loc = op.getLoc();
    auto offsets = op.getMixedOffsets();
    if (offsets.empty())
      return rewriter.notifyMatchFailure(op, "the store op must have offsets");

    FailureOr<VectorType> distPayloadTyOrFailure =
        getDistVecTypeBasedOnLaneLayout(layout, sgPayloadTy);
    if (failed(distPayloadTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "Failed to distribute matrix op payload based on layout.");

    SmallVector<Value> offsetsAsValues =
        vector::getAsValues(rewriter, loc, offsets);

    SmallVector<Value> newCoords = offsetsAsValues;
    if (!op.getSubgroupBlockIoAttr()) {
      newCoords = computeDistributedCoordsForMatrixOp(
          rewriter, loc, layout, sgPayloadTy.getShape(), offsetsAsValues);
      if (newCoords.empty())
        return rewriter.notifyMatchFailure(
            op, "Failed to compute distributed coordinates.");
    }

    SmallVector<int64_t> newConstOffsets(op.getConstOffsets().size(),
                                         ShapedType::kDynamic);
    DenseI64ArrayAttr newConstOffsetsAttr =
        rewriter.getDenseI64ArrayAttr(newConstOffsets);

    xegpu::StoreMatrixOp::create(
        rewriter, loc, TypeRange{},
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getData()),
                    distPayloadTyOrFailure.value()),
        adaptor.getMemDesc(), ValueRange(newCoords), newConstOffsetsAttr,
        op.getSubgroupBlockIoAttr(), xegpu::DistributeLayoutAttr{});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Distributes a subgroup-level StoreScatter (xegpu.store) op to
/// lane-level.
///
/// Example 1 (1D, no chunk size):
///   layout = #xegpu.layout<lane_layout = [16], lane_data = [1]>
///   %mask = producer_op : vector<16xi1>
///   %offset = producer_op : vector<16xindex>
///   xegpu.store %payload, %src[%offset], %mask : vector<16xf16>,
///     memref<256xf16>, vector<16xindex>, vector<16xi1>
/// Distributed to:
///   %mask = producer_op : vector<1xi1>
///   %offset = producer_op : vector<1xindex>
///   xegpu.store %payload, %src[%offset], %mask : vector<1xf16>,
///     memref<256xf16>, vector<1xindex>, vector<1xi1>
///
/// Example 2 (2D with chunk size, same mask & offset):
///   layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>
///   xegpu.store %payload, %src[%offset], %mask <{chunk_size=8}> :
///     vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
/// Distributed to:
///   xegpu.store %payload, %src[%offset], %mask <{chunk_size=8}> :
///     vector<8xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>
///
/// Example 3 (3D with leading unit dims):
///   layout = #xegpu.layout<lane_layout = [1, 1, 16], lane_data = [1, 1, 1]>
///   %mask = producer_op : vector<1x1x16xi1>
///   %offset = producer_op : vector<1x1x16xindex>
///   xegpu.store %payload, %src[%offset], %mask : vector<1x1x16xf16>,
///     memref<256xf16>, vector<1x1x16xindex>, vector<1x1x16xi1>
/// Distributed to:
///   %mask = producer_op : vector<1x1x1xi1>
///   %offset = producer_op : vector<1x1x1xindex>
///   xegpu.store %payload, %src[%offset], %mask : vector<1xf16>,
///     memref<256xf16>, vector<1xindex>, vector<1xi1>
struct SgToLaneStoreScatter
    : public OpConversionPattern<xegpu::StoreScatterOp> {
  using OpConversionPattern<xegpu::StoreScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::StoreScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout = op.getAnchorLayout();
    if (!layout)
      return failure();

    VectorType origValueTy = op.getValueType();
    if (!origValueTy)
      return failure();

    // Check that all leading dimensions are unit dimensions.
    int chunkSize = op.getChunkSize().value_or(1);
    int effectiveVecRank = (chunkSize == 1) ? 1 : 2;
    ArrayRef<int64_t> shape = origValueTy.getShape();
    if (llvm::any_of(shape.take_front(origValueTy.getRank() - effectiveVecRank),
                     [](int64_t d) { return d != 1; }))
      return rewriter.notifyMatchFailure(
          op, "Only unit dimensions allowed for the leading "
              "dimensions of the store vector!");

    auto distValueTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, origValueTy);
    if (failed(distValueTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute expected lane vector type from lane layout");

    VectorType distValueTy = distValueTyOrFailure.value();
    VectorType distValueTy1D = VectorType::get({distValueTy.getNumElements()},
                                               distValueTy.getElementType());

    Value distValue = adaptor.getValue();
    if (distValue.getType() != distValueTy1D)
      distValue = castValueTo(rewriter, cast<TypedValue<VectorType>>(distValue),
                              distValueTy1D);

    // Flatten offsets and mask to 1D to match the 1D value type.
    Value distOffsets = adaptor.getOffsets();
    auto distOffsetsTy = cast<VectorType>(distOffsets.getType());
    VectorType offsetsTy1D = VectorType::get({distOffsetsTy.getNumElements()},
                                             distOffsetsTy.getElementType());
    distOffsets = castValueTo(
        rewriter, cast<TypedValue<VectorType>>(distOffsets), offsetsTy1D);

    Value distMask = adaptor.getMask();
    auto distMaskTy = cast<VectorType>(distMask.getType());
    VectorType maskTy1D = VectorType::get({distMaskTy.getNumElements()},
                                          distMaskTy.getElementType());
    distMask =
        castValueTo(rewriter, cast<TypedValue<VectorType>>(distMask), maskTy1D);

    Value distDest = adaptor.getDest();
    xegpu::StoreScatterOp::create(rewriter, op.getLoc(), distValue, distDest,
                                  distOffsets, distMask, op.getChunkSizeAttr(),
                                  op.getL1HintAttr(), op.getL2HintAttr(),
                                  op.getL3HintAttr(), /*layout=*/nullptr,
                                  /*contiguity=*/nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Distribute a vector::StepOp to lane-level.
/// The layout must have exactly 1 effective lane dimension.
/// We completely resolve the vector::StepOp by computing the lane_data-sized
/// subranges.
struct SgToLaneVectorStep : public OpConversionPattern<vector::StepOp> {
  using OpConversionPattern<vector::StepOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::StepOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getTemporaryLayout(op->getResult(0));
    if (!resultLayout || !resultLayout.isForSubgroup())
      return rewriter.notifyMatchFailure(
          op, "the result vector of the step op lacks subgroup layout");

    auto loc = op.getLoc();
    auto stepResultVecTy = op.getResult().getType();
    auto laneShapeOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(resultLayout, stepResultVecTy);
    if (failed(laneShapeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute lane vector type from the layout");
    VectorType newVecTy = laneShapeOrFailure.value();

    Value laneId = gpu::LaneIdOp::create(rewriter, loc, rewriter.getIndexType(),
                                         /*upperBound=*/mlir::IntegerAttr());
    auto laneDataBlockCoords = resultLayout.computeDistributedCoords(
        rewriter, loc, laneId, stepResultVecTy.getShape());
    if (failed(laneDataBlockCoords))
      return rewriter.notifyMatchFailure(
          op, "failed to compute lane data block coordinates");

    auto laneDataBlockCoordsVec = laneDataBlockCoords.value();
    auto laneDataBlockLength = resultLayout.getEffectiveLaneDataAsInt()[0];
    assert(static_cast<int64_t>(laneDataBlockCoordsVec.size()) ==
           newVecTy.getNumElements() / laneDataBlockLength);
    SmallVector<Value> stepVals;
    // For each lane_data block, reconstruct its sub-range
    // from the range of SG-level vector.step.Example: vector.step
    // {slice<layout<lane_layout=[2,4,2], lane_data=[1,2,1]>, dims=[0,2]>} :
    // vector<16xindex>
    // Each logical lane holds 4 elements as 2 blocks of 2 elements each.
    // The blocks are round-robin distributed, so logical lane id 0
    // holds values [0,1, 8,9].
    for (auto &laneDataBlockCoords : laneDataBlockCoordsVec) {
      auto laneDataBlockStartCoord = laneDataBlockCoords[0];
      stepVals.push_back(laneDataBlockStartCoord);
      for (int i = 1; i < laneDataBlockLength; ++i) {
        auto offset = arith::ConstantIndexOp::create(rewriter, loc, i);
        stepVals.push_back(arith::AddIOp::create(
            rewriter, loc, laneDataBlockStartCoord, offset));
      }
    }
    assert(static_cast<int64_t>(stepVals.size()) == newVecTy.getNumElements() &&
           "Expecting the number of step values to match the number of "
           "elements in the vector");
    auto stepOpVal =
        vector::FromElementsOp::create(rewriter, loc, newVecTy, stepVals);
    rewriter.replaceOp(op, stepOpVal);
    return success();
  }
};

/// Distributes a subgroup-level vector.extract op to lane-level. Only
/// handles sub-vector extraction (result is VectorType, not scalar).
struct SgToLaneVectorExtract : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern<vector::ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle vector results (not scalar extraction).
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "scalar extract not supported");

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(op->getOpResult(0));
    if (!layout || !layout.isForSubgroup())
      return failure();

    // This implementation assumes distribution only happens on the innermost
    // dimension. Verify that lane_layout[0...n-2] are all unit.
    auto laneLayout = layout.getEffectiveLaneLayoutAsInt();
    if (llvm::any_of(ArrayRef<int64_t>(laneLayout).drop_back(1),
                     [](int64_t v) { return v != 1; }))
      return rewriter.notifyMatchFailure(
          op, "only innermost dimension distribution is supported for "
              "vector.extract");

    auto newOp = vector::ExtractOp::create(
        rewriter, op.getLoc(), adaptor.getSource(), op.getMixedPosition());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// This pattern distributes a subgroup-level ShapeCast op to lane-level.
struct SgToLaneVectorShapeCast
    : public OpConversionPattern<vector::ShapeCastOp> {
  using OpConversionPattern<vector::ShapeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ShapeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getTemporaryLayout(op->getOpResult(0));
    if (!resultLayout || !resultLayout.isForSubgroup())
      return rewriter.notifyMatchFailure(
          op, "the result vector of the shape_cast op lacks subgroup layout");

    auto resultDistTypeOrFailure = xegpu::getDistVecTypeBasedOnLaneLayout(
        resultLayout, op.getResultVectorType());
    if (failed(resultDistTypeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to get distributed vector type for result");

    Value source = adaptor.getSource();
    auto newShapeCast = vector::ShapeCastOp::create(
        rewriter, op.getLoc(), resultDistTypeOrFailure.value(), source);
    rewriter.replaceOp(op, newShapeCast);
    return success();
  }
};

/// Distributes a subgroup-level vector.extract_strided_slice op to
/// lane-level. If the result is distributed, the offsets and sizes are
/// adjusted to match the distributed types.
struct SgToLaneVectorExtractStridedSlice
    : public OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern<vector::ExtractStridedSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getTemporaryLayout(op->getOpResult(0));
    if (!resultLayout || !resultLayout.isForSubgroup())
      return failure();

    VectorType resultType = op.getType();
    auto distResultTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(resultLayout, resultType);
    if (failed(distResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute distributed vector type from lane layout");
    VectorType distResultTy = *distResultTyOrFailure;

    SmallVector<int64_t> distributedDims =
        getDistributedDims(resultType, distResultTy);

    // Collect updated sizes, offsets, strides. Pad to full source rank.
    int64_t sourceRank = op.getSourceVectorType().getRank();
    SmallVector<Attribute> updatedSizes =
        llvm::map_to_vector(op.getSizes(), [](Attribute attr) { return attr; });
    SmallVector<Attribute> updatedOffsets = llvm::map_to_vector(
        op.getOffsets(), [](Attribute attr) { return attr; });
    SmallVector<Attribute> updatedStrides = llvm::map_to_vector(
        op.getStrides(), [](Attribute attr) { return attr; });
    for (int64_t i = op.getSizes().size(); i < sourceRank; ++i) {
      updatedSizes.push_back(
          rewriter.getI64IntegerAttr(op.getSourceVectorType().getDimSize(i)));
      updatedOffsets.push_back(rewriter.getI64IntegerAttr(0));
      updatedStrides.push_back(rewriter.getI64IntegerAttr(1));
    }

    // If the result is distributed, adjust offsets and sizes in the
    // distributed dimension.
    if (!distributedDims.empty()) {
      if (distributedDims.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "only single dimension distribution is supported");
      int64_t distDim = distributedDims[0];
      const auto *uArch =
          xegpu::uArch::getUArch(xegpu::getChipStr(op).value_or(""));
      if (!uArch)
        return rewriter.notifyMatchFailure(
            op, "target attribute required to determine subgroup size");
      int subgroupSize = uArch->getSubgroupSize();
      auto sourceLayout = xegpu::getTemporaryLayout(op->getOpOperand(0));
      if (!sourceLayout || sourceLayout.getEffectiveLaneLayoutAsInt().empty())
        return rewriter.notifyMatchFailure(
            op, "source of extract_strided_slice lacks distribution layout");
      int sourceDistrDimSize = op.getSourceVectorType().getShape()[distDim];
      auto laneLayout = sourceLayout.getEffectiveLaneLayoutAsInt();
      // Effective subgroup size needs to be adjusted if laneLayout along
      // the distributed dimension is smaller than subgroup size.
      if (laneLayout[distDim] < subgroupSize &&
          subgroupSize % laneLayout[distDim] == 0)
        subgroupSize = laneLayout[distDim];
      if (sourceDistrDimSize % subgroupSize != 0)
        return rewriter.notifyMatchFailure(
            op, "source size along distributed dim is not a multiple of "
                "subgroup size");
      auto sourceLaneData = sourceLayout.getEffectiveLaneDataAsInt();
      // Only check lane_data for the distributed dimension. Non-distributed
      // dimensions may have non-unit lane_data (e.g., packed layouts).
      if (distDim < static_cast<int64_t>(sourceLaneData.size()) &&
          sourceLaneData[distDim] != 1)
        return rewriter.notifyMatchFailure(
            op, "expecting unit lane data along the distributed dimension");
      int64_t distrDimOffset =
          cast<IntegerAttr>(updatedOffsets[distDim]).getInt();
      if (distrDimOffset % subgroupSize != 0)
        return rewriter.notifyMatchFailure(
            op, "offset along distributed dim is not a multiple of "
                "subgroup size");
      // Adjust sizes and offsets for the distributed dimension.
      updatedSizes[distDim] =
          rewriter.getI64IntegerAttr(distResultTy.getDimSize(distDim));
      updatedOffsets[distDim] =
          rewriter.getI64IntegerAttr(distrDimOffset / subgroupSize);
    }

    auto newOp = vector::ExtractStridedSliceOp::create(
        rewriter, op.getLoc(), distResultTy, adaptor.getSource(),
        ArrayAttr::get(rewriter.getContext(), updatedOffsets),
        ArrayAttr::get(rewriter.getContext(), updatedSizes),
        ArrayAttr::get(rewriter.getContext(), updatedStrides));
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// This pattern distributes a subgroup-level `vector.broadcast` op to
/// lane-level. The pattern supports three cases:
///
/// 1) Broadcast a low-rank vector to high-rank vector: The low-rank input
///    vector must have a slice layout of the result. If the distributed source
///    and target vector types are identical, this lowers to a no-op; otherwise,
///    it remains a broadcast but operates on distributed vectors.
///
/// 2) Broadcast a same-rank vector with identical layouts for source and
///    target: The source vector must have unit dimensions, and lane_data must
///    be unit size for those unit dims. This always lowers to a no-op.
///
/// 3) Broadcast a scalar with no layout: This always lowers to a broadcast
///    from scalar to distributed result type.
///
/// Example 1 (low-rank to high-rank broadcast):
/// ```
///   %0 = "some_op"() {layout_result_0 =
///     #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
///     dims = [0]>} : () -> vector<16xf16>
///   %1 = vector.broadcast %0 {layout_result_0 =
///     #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
///     : vector<16xf16> to vector<16x16xf16>
/// ```
/// is distributed to:
/// ```
///   %0 = "some_op"() : () -> vector<1xf16>
///   %1 = vector.broadcast %0 : vector<1xf16> to vector<16x1xf16>
/// ```
///
/// Example 2 (same-rank broadcast, no-op):
/// ```
///   %0 = "some_op"() {layout_result_0 =
///     #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
///     : () -> vector<16x1xf16>
///   %1 = vector.broadcast %0 {layout_result_0 =
///     #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
///     : vector<16x1xf16> to vector<16x16xf16>
/// ```
/// is distributed to (no-op, source already matches distributed result type):
/// ```
///   %0 = "some_op"() : () -> vector<16x1xf16>
///   // broadcast is eliminated, %0 is used directly
/// ```
///
/// Example 3 (scalar to vector broadcast):
/// ```
///   %0 = "some_op"() : () -> f16
///   %1 = vector.broadcast %0 {layout_result_0 =
///     #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
///     : f16 to vector<16x16xf16>
/// ```
/// is distributed to:
/// ```
///   %0 = "some_op"() : f16
///   %1 = vector.broadcast %0 : f16 to vector<16x1xf16>
/// ```
struct SgToLaneBroadcast : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getTemporaryLayout(cast<OpResult>(op.getResult()));
    if (!resultLayout || !resultLayout.isForSubgroup())
      return rewriter.notifyMatchFailure(
          op, "result does not have subgroup distribute layout");

    VectorType destType = op.getResultVectorType();
    VectorType sourceType = dyn_cast<VectorType>(op.getSourceType());

    xegpu::DistributeLayoutAttr sourceLayout =
        xegpu::getTemporaryLayout(op->getOpOperand(0));

    if (sourceType) {
      int64_t rankDiff = destType.getRank() - sourceType.getRank();
      if (rankDiff > 0) {
        // Case 1: Low-rank to high-rank broadcast.
        if (!sourceLayout || !sourceLayout.isSliceOf(resultLayout))
          op.emitWarning(
              "broadcast source layout must be a slice of result layout");
      } else if (rankDiff == 0) {
        // Case 2: Same-rank broadcast.
        auto broadcastUnitDimsSet = op.computeBroadcastedUnitDims();
        SmallVector<int64_t> broadcastUnitDims(broadcastUnitDimsSet.begin(),
                                               broadcastUnitDimsSet.end());
        assert(sourceLayout.isEqualTo(
                   sourceLayout.setUnitDimData(broadcastUnitDims)) &&
               "The sg_data for unit dimensions should be set as 1");
        sourceLayout = sourceLayout.setUnitDimLayout(broadcastUnitDims);
      }
    } else {
      // Case 3: Scalar to vector broadcast.
      if (sourceLayout)
        return rewriter.notifyMatchFailure(
            op, "broadcast from scalar must not have a layout attribute");
    }

    auto destDistType =
        xegpu::getDistVecTypeBasedOnLaneLayout(resultLayout, destType);
    if (failed(destDistType))
      return rewriter.notifyMatchFailure(
          op, "failed to distribute the result vector type");

    Value source = adaptor.getSource();
    // If the adapted source already matches the dest dist type, it's a no-op.
    if (source.getType() == destDistType.value()) {
      rewriter.replaceOp(op, source);
      return success();
    }

    auto newOp = vector::BroadcastOp::create(rewriter, op.getLoc(),
                                             destDistType.value(), source);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/// Distributes a subgroup-level vector.insert_strided_slice op to
/// lane-level. If the dest is distributed, the offsets are adjusted to
/// match the distributed types.
struct SgToLaneVectorInsertStridedSlice
    : public OpConversionPattern<vector::InsertStridedSliceOp> {
  using OpConversionPattern<vector::InsertStridedSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertStridedSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr resultLayout =
        xegpu::getTemporaryLayout(op->getOpResult(0));
    if (!resultLayout || !resultLayout.isForSubgroup())
      return failure();

    VectorType destType = op.getDestVectorType();
    auto distDestTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(resultLayout, destType);
    if (failed(distDestTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute distributed vector type from lane layout");
    VectorType distDestTy = *distDestTyOrFailure;

    SmallVector<int64_t> destDistributedDims =
        getDistributedDims(destType, distDestTy);

    SmallVector<Attribute> updatedOffsets = llvm::map_to_vector(
        op.getOffsets(), [](Attribute attr) { return attr; });

    if (!destDistributedDims.empty()) {
      if (destDistributedDims.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "only single dimension distribution is supported");
      int64_t destDistDim = destDistributedDims[0];

      const auto *uArch =
          xegpu::uArch::getUArch(xegpu::getChipStr(op).value_or(""));
      if (!uArch)
        return rewriter.notifyMatchFailure(
            op, "target attribute required to determine subgroup size");
      int subgroupSize = uArch->getSubgroupSize();

      VectorType srcType = op.getSourceVectorType();
      // The distributed dim must be in the last k (source rank) dims of dest.
      int64_t sourceDistDim =
          destDistDim - (destType.getRank() - srcType.getRank());
      if (sourceDistDim < 0)
        return rewriter.notifyMatchFailure(
            op, "distributed dimension must be in the last k dims of dest");

      auto destLayout = xegpu::getTemporaryLayout(op->getOpOperand(1));
      auto sourceLayout = xegpu::getTemporaryLayout(op->getOpOperand(0));
      if (!destLayout || !sourceLayout ||
          destLayout.getEffectiveLaneLayoutAsInt().empty() ||
          sourceLayout.getEffectiveLaneLayoutAsInt().empty())
        return rewriter.notifyMatchFailure(
            op, "source or dest of insert_strided_slice lacks distribution "
                "layout");

      auto destLaneData = destLayout.getEffectiveLaneDataAsInt();
      auto sourceLaneData = sourceLayout.getEffectiveLaneDataAsInt();
      // Only check lane_data for the distributed dimension. Non-distributed
      // dimensions may have non-unit lane_data (e.g., packed layouts).
      if ((destDistDim < static_cast<int64_t>(destLaneData.size()) &&
           destLaneData[destDistDim] != 1) ||
          (sourceDistDim < static_cast<int64_t>(sourceLaneData.size()) &&
           sourceLaneData[sourceDistDim] != 1))
        return rewriter.notifyMatchFailure(
            op, "expecting unit lane data along the distributed dimension");

      int64_t srcDistrDimSize = srcType.getDimSize(sourceDistDim);
      if (srcDistrDimSize % subgroupSize != 0)
        return rewriter.notifyMatchFailure(
            op, "source distributed dim size is not a multiple of "
                "subgroup size");

      int64_t destDistrDimOffset =
          cast<IntegerAttr>(op.getOffsets()[destDistDim]).getInt();
      if (destDistrDimOffset % subgroupSize != 0)
        return rewriter.notifyMatchFailure(
            op, "offset along distributed dim is not a multiple of "
                "subgroup size");
      // Adjust offset for the distributed dimension.
      updatedOffsets[destDistDim] =
          rewriter.getI64IntegerAttr(destDistrDimOffset / subgroupSize);
    }

    auto newOp = vector::InsertStridedSliceOp::create(
        rewriter, op.getLoc(), distDestTy, adaptor.getValueToStore(),
        adaptor.getDest(),
        ArrayAttr::get(rewriter.getContext(), updatedOffsets), op.getStrides());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Distributes a subgroup-level vector.insert op to lane-level. Only
/// handles sub-vector insertion (value to store is VectorType, not scalar).
struct SgToLaneVectorInsert : public OpConversionPattern<vector::InsertOp> {
  using OpConversionPattern<vector::InsertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle vector value-to-store (not scalar insertion).
    auto valueType = dyn_cast<VectorType>(op.getValueToStoreType());
    if (!valueType)
      return rewriter.notifyMatchFailure(op, "scalar insert not supported");

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(op->getOpResult(0));
    if (!layout || !layout.isForSubgroup())
      return failure();

    // verify that the outer k dimensions (for offsets)
    // don't have non-unit lane_layout.
    auto laneLayout = layout.getEffectiveLaneLayoutAsInt();
    if (llvm::any_of(ArrayRef<int64_t>(laneLayout).drop_back(1),
                     [](int64_t v) { return v != 1; }))
      return rewriter.notifyMatchFailure(
          op, "only innermost dimension distribution is supported for "
              "vector.insert");

    auto newOp = vector::InsertOp::create(
        rewriter, op.getLoc(), adaptor.getValueToStore(), adaptor.getDest(),
        op.getMixedPosition());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Redistributes `src` for a `convert_layout` that changes only the
/// `lane_layout` along the outer (distributed) dimension, shrinking it from
/// `currentLaneNum` to `targetLaneNum` lanes (a partial-subgroup
/// distribution). Because the data is no longer replicated across all lanes,
/// each surviving lane must gather the values that previously lived in the
/// lanes that are dropped. The values are gathered with `gpu.shuffle` and
/// concatenated with the lane-local data using `vector.shuffle`, which doubles
/// the distributed outer dimension when the lane count is halved.
///
/// Only halving the lane count (a factor of two) is currently supported.
/// Returns the redistributed value on success, or failure if `src` cannot be
/// shuffled (e.g. it is not a rank-2 vector or its bit width is not a multiple
/// of 32).
static FailureOr<Value>
shuffleDataAsLaneLayoutChange(ConversionPatternRewriter &rewriter, Location loc,
                              Value src, int64_t currentLaneNum,
                              int64_t targetLaneNum) {
  VectorType srcTy = dyn_cast<VectorType>(src.getType());
  if (!srcTy || srcTy.getRank() != 2)
    return failure();
  // Only halving the lane count (factor of two) is supported for now.
  if (targetLaneNum <= 0 || currentLaneNum != targetLaneNum * 2)
    return failure();
  // gpu.shuffle operates on i32, so the data must be a multiple of 32 bits.
  int64_t vectorBitWidth =
      srcTy.getNumElements() * srcTy.getElementTypeBitWidth();
  if (vectorBitWidth % 32 != 0)
    return failure();

  // A vector cannot be shuffled across lanes directly:
  // -- cast the source to a 1D vector of i32
  // -- create a temp 1D vector of i32 initialized to zero
  // -- for each i32 element:
  // ---- extract it from the source bundle
  // ---- gpu.shuffle to gather the value from the partner lane
  // ---- insert it into the temp bundle
  // -- cast the temp back to the source vector type
  // -- vector.shuffle the source and temp to concatenate along the outer dim
  Type shuffleElemTy = rewriter.getI32Type();
  int64_t numShuffles = vectorBitWidth / 32;
  VectorType shuffleBundleTy = VectorType::get({numShuffles}, shuffleElemTy);
  // Initialize temp to zero.
  Value temp = arith::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(shuffleBundleTy,
                             IntegerAttr::get(shuffleElemTy, 0)));
  VectorType flatSrcTy =
      VectorType::get({srcTy.getNumElements()}, srcTy.getElementType());
  Value flatSrc = vector::ShapeCastOp::create(rewriter, loc, flatSrcTy, src);
  Value shuffleBundle =
      vector::BitCastOp::create(rewriter, loc, shuffleBundleTy, flatSrc);
  for (int64_t i = 0; i < numShuffles; i++) {
    Value shuffleElem =
        vector::ExtractOp::create(rewriter, loc, shuffleBundle, i);
    shuffleElem = gpu::ShuffleOp::create(rewriter, loc, shuffleElem, 0,
                                         targetLaneNum, gpu::ShuffleMode::UP)
                      .getResult(0);
    temp = vector::InsertOp::create(rewriter, loc, shuffleElem, temp, i);
  }
  temp = vector::BitCastOp::create(rewriter, loc, flatSrcTy, temp);
  temp = vector::ShapeCastOp::create(rewriter, loc, srcTy, temp);

  // Concatenate the lane-local and gathered data along the outer dimension.
  SmallVector<int64_t> indices(srcTy.getShape()[0] * 2);
  std::iota(indices.begin(), indices.end(), 0);
  Value res = vector::ShuffleOp::create(rewriter, loc, src, temp, indices);
  return res;
}

/// Repacks `src`'s `lane_data` along `repackDim` between round-robin and
/// contiguous form with an `xegpu.lane_shuffle`, which moves each lane's run of
/// `k` elements across lanes while preserving the element type.
///
/// `inputData`/`targetData` are the `repackDim` `lane_data` of the input and
/// target layouts; exactly one must be 1 (round-robin) and the other `k`
/// (contiguous). Returns failure if that does not hold.
static FailureOr<Value> repackLaneData(ConversionPatternRewriter &rewriter,
                                       Location loc, Value src,
                                       int64_t repackDim, int64_t inputData,
                                       int64_t targetData) {
  auto srcTy = dyn_cast<VectorType>(src.getType());
  if (!srcTy)
    return failure();
  int64_t rank = srcTy.getRank();
  Type elemTy = srcTy.getElementType();
  int64_t k = srcTy.getShape()[repackDim];

  bool roundRobinToContig = inputData == 1 && targetData == k;
  bool contigToRoundRobin = inputData == k && targetData == 1;
  if (!roundRobinToContig && !contigToRoundRobin)
    return failure();

  // Round-robin -> contiguous gathers a lane's strided elements into
  // consecutive positions (pack); the reverse scatters them back (unpack).
  xegpu::LaneShuffleMode mode = roundRobinToContig
                                    ? xegpu::LaneShuffleMode::Pack
                                    : xegpu::LaneShuffleMode::Unpack;
  VectorType runTy = VectorType::get({k}, elemTy);

  // Common case: the lane fragment is a single run (every dimension other than
  // `repackDim` is unit), so collapse it to 1D, shuffle once, and restore it.
  if (srcTy.getNumElements() == k) {
    if (rank == 1)
      return Value(
          xegpu::LaneShuffleOp::create(rewriter, loc, runTy, src, mode));
    Value flat = vector::ShapeCastOp::create(rewriter, loc, runTy, src);
    Value shuffled =
        xegpu::LaneShuffleOp::create(rewriter, loc, runTy, flat, mode);
    return Value(vector::ShapeCastOp::create(rewriter, loc, srcTy, shuffled));
  }

  // When `repackDim` is innermost each run is a contiguous sub-vector, so it is
  // extracted and re-inserted as a whole.
  if (repackDim == rank - 1) {
    SmallVector<int64_t> outerShape(srcTy.getShape().drop_back());
    int64_t numRuns = computeProduct(outerShape);
    SmallVector<int64_t> outerStrides = computeStrides(outerShape);
    Value result = arith::ConstantOp::create(rewriter, loc, srcTy,
                                             rewriter.getZeroAttr(srcTy));
    for (int64_t i = 0; i < numRuns; ++i) {
      SmallVector<int64_t> pos = delinearize(i, outerStrides);
      Value run = vector::ExtractOp::create(rewriter, loc, src, pos);
      Value shuffled =
          xegpu::LaneShuffleOp::create(rewriter, loc, runTy, run, mode);
      result = vector::InsertOp::create(rewriter, loc, shuffled, result, pos);
    }
    return result;
  }

  // Otherwise each run is strided along `repackDim`: extract the `k`-long slice
  // (a sub-vector that is unit along every other dim), flatten it to 1D,
  // shuffle, and insert it back.
  SmallVector<int64_t> keptShape;
  SmallVector<int64_t> keptDims;
  for (int64_t d = 0; d < rank; ++d)
    if (d != repackDim) {
      keptShape.push_back(srcTy.getShape()[d]);
      keptDims.push_back(d);
    }
  int64_t numRuns = computeProduct(keptShape);
  SmallVector<int64_t> keptStrides = computeStrides(keptShape);
  SmallVector<int64_t> sliceSizes(rank, 1);
  sliceSizes[repackDim] = k;
  SmallVector<int64_t> sliceStrides(rank, 1);
  VectorType sliceTy = VectorType::get(sliceSizes, elemTy);
  Value result = arith::ConstantOp::create(rewriter, loc, srcTy,
                                           rewriter.getZeroAttr(srcTy));
  for (int64_t i = 0; i < numRuns; ++i) {
    SmallVector<int64_t> keptPos = delinearize(i, keptStrides);
    SmallVector<int64_t> offsets(rank, 0);
    for (auto [dim, coord] : llvm::zip_equal(keptDims, keptPos))
      offsets[dim] = coord;
    Value slice = vector::ExtractStridedSliceOp::create(
        rewriter, loc, src, offsets, sliceSizes, sliceStrides);
    Value run = vector::ShapeCastOp::create(rewriter, loc, runTy, slice);
    Value repacked =
        xegpu::LaneShuffleOp::create(rewriter, loc, runTy, run, mode);
    Value repackedSlice =
        vector::ShapeCastOp::create(rewriter, loc, sliceTy, repacked);
    result = vector::InsertStridedSliceOp::create(
        rewriter, loc, repackedSlice, result, offsets, sliceStrides);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Broadcast redistribution
//===----------------------------------------------------------------------===//
//
// Lowers a `convert_layout` that only relocates data between lanes: the two
// layouts agree on `lane_data`, and differ only in which lane holds which part
// of the value. Every element already exists in some lane, so nothing is
// recomputed. The scale operands of scaled matrix multiplication need this,
// being produced broadcasted and consumed distributed.
//
// Terms
// -----
//
// These follow from the layouts, for a value of shape `Sh` on a subgroup of `S`
// lanes. `#xegpu.` prefixes are dropped for width throughout.
//
//   distribution unit  what a lane owns as one block: `lane_data`
//   fragment           everything one lane owns. Slicing drops dimensions from
//                      `lane_layout`; over what is left a lane owns
//                      `Sh[i] / lane_layout[i]` per dimension, flattened
//                      row-major
//   lane period        `product(lane_layout)` counting the dimensions slicing
//                      dropped, so slicing does not reduce it. Lanes this far
//                      apart hold the same fragment. `getLanePeriod`,
//                      `lanePeriod` below
//   slot               `lane % lanePeriod`, a lane's index within one period
//   ownership table    every lane's fragment, tabulated as element coordinates,
//                      one table per side. `computeOwnedCoords`, `inputOwned`
//                      and `targetOwned` below
//
// These name the moving parts of the scheme:
//
//   donor              the lane a value is read from, `slot + donorDelta`, with
//                      `donorDelta` a multiple of `lanePeriod`
//   donor group        the lanes one `donorDelta` names, one per slot:
//                      `donorDelta` through `donorDelta + lanePeriod - 1`.
//                      Candidates are tried a group at a time, since every slot
//                      has to be served by the same `donorDelta`
//   element source     where one element of the target fragment comes from: a
//                      donor, and the index to extract from that donor's
//                      fragment. One per element, and it needs a shuffle
//                      exactly when the donor is not the lane itself.
//                      `ElementSource` below
//
// Either layout may replicate -- a set of lanes all holding the same fragment
// -- and a layout has two ways to spell that. Both of these are on a
// `vector<8x2>` over 16 lanes:
//
//   slice<layout<[8, 1, 2], order = [0, 2, 1]>, dims = [0]>    ("sliced")
//     slicing leaves `lane_layout` [1, 2], so a fragment is 8x1, one whole
//     column, while the lane period still counts the sliced dimension and is
//     8 * 1 * 2 = 16. Sixteen lanes hold two distinct columns between them,
//     eight lanes to a column.
//
//   layout<[8, 1]>                                             ("short")
//     a `lane_layout` naming fewer lanes than the subgroup has. A fragment is
//     1x2, one whole row, and the lane period is 8: the lane id is delinearized
//     modulo each extent, so a lane's coordinates depend on `lane % 8` alone
//     and lanes 8-15 repeat lanes 0-7.
//
// The two spellings are treated alike throughout.
//
// The two sides are not symmetric in what they mean:
//
//   input   supplies the values. Any lane may be a donor, so every lane's
//           fragment has to be known -- which it is, for any lane_layout.
//   target  states the obligation. Only the lanes it distributes to have to end
//           up holding the fragment it assigns them; the rest are left
//           unspecified.
//
// What it handles
// ---------------
//
// Two properties decide it, one per side. Examples stay on a `vector<8x2>` over
// 16 lanes; the case numbers are used throughout the rest of this comment.
//
// The input's replication decides what a lane arrives with, and so what has to
// travel:
//
//   fully broadcast      every lane already holds the whole value, so nothing
//                        travels and the result is extracts alone. Spelled by
//                        slicing away every distributed dimension:
//                        slice<layout<[1, 1, 16]>, dims = [2]>
//
//   partially broadcast  groups of lanes each hold part of it, so a lane has to
//                        be given whatever its group lacks. Either spelling of
//                        replication from Terms above.
//
// The target's lane period decides where a lane may be given that from. Donors
// sit at multiples of the period, so a period `P` over `S` lanes offers `S / P`
// donor groups:
//
//   period < S   several groups, so elements can cross lanes. `layout<[8, 1]>`
//                offers 2, `layout<[2, 1]>` offers 8.
//
//   period == S  one group, `donorDelta` 0, so nothing can cross a lane: only
//                an input already holding each lane's target fragment works.
//                A replicating target can have this period, slicing not
//                reducing it.
//
// One example per combination, and what the tests show it emitting. The target
// is given as its `lane_layout`, the period following from it; the input as its
// category, `sliced` and `short` being the two spellings named in Terms:
//
//   case  input            target             period  emits
//   1     fully broadcast  [4, 1]                  4  4 extracts
//   3     fully broadcast  [8, 1, 2] sliced       16  8 extracts
//   2     partial, sliced  [8, 1]                  8  1 extract, 1 shuffle
//   4     partial, short   [2, 1]                  2  2 extracts, 6 shuffles
//   --    partial, short   [8, 1, 2] sliced       16  declined
//
// Only the target varies within a category, so the test suite carries more than
// one of some -- @convert_layout_broadcast_all_lanes is case 1 with `[8, 1]` --
// but they behave alike and only one is described here. Case 1 is the one whose
// constants are all distinct, which the index expression below is read off.
//
// The last row is the boundary the two properties imply: with only `donorDelta`
// 0 available, a lane arriving with less than its target fragment cannot be
// completed, and the match fails with NoDonorDelta. There a lane arrives with 2
// of the 8 elements it needs.
//
// Case 2 is the running example below: lanes 0-7 arrive holding all of column
// 0, lanes 8-15 all of column 1, and lane `i` leaves holding row `i`. Lane 3
// wants (3, 0) and (3, 1); it holds (3, 0) already, while (3, 1) is in lane 11.
//
// Nested slices are coalesced before anything else looks at a layout, so only
// the lane_layout of the underlying layout and the union of the sliced dims
// matter; @convert_layout_broadcast_nested_slice covers a two-level input.
//
// Why donors sit a whole period apart
// -----------------------------------
//
// `gpu.shuffle idx` is the only way to move data between lanes and it exchanges
// one value per lane, so every lane runs the same extract. A donor can
// therefore only serve lanes of its own slot, which is why `donorDelta` is a
// multiple of `lanePeriod`. It also means the donor extracts the right thing
// without knowing who asked: its own slot is
// `(slot + donorDelta) % lanePeriod`, which is `slot`, so it computes the same
// index.
//
// Case 2 has two element sources, one per element of its 1x2 fragment. Slot 3
// wants (3, 0) and (3, 1):
//
//   element 0 -> donorDelta = 0, donor = 3,  index = 3   already local
//   element 1 -> donorDelta = 8, donor = 11, index = 3   shuffled
//
// What it declines
// ----------------
//
// The layout combinations, in the terms above, with the `RedistributionLimit` a
// match failure names. None is a limit of `convert_layout`, which is meaningful
// for any pair and so always lowerable; each is a limit of the scheme here, one
// shuffle per element from a donor a whole number of lane periods away. Another
// pattern or a more general scheme lowers any of them.
//
// Decided by inspecting the layouts:
//
//   they differ in `lane_data`                         LaneDataDiffers
//
//      layout<[8, 1], [1, 1]>  ->  layout<[1, 8], [1, 2]>
//
//    A `lane_data`-only change is the repack pattern's; both changing is
//    neither's.
//
//   their common `lane_data` is not all ones           LaneDataNotUnit
//
//   the target's lane period does not divide `S`       LanePeriodNotDivisor
//
// Decided by building the ownership tables, so these name layout combinations
// known to fail rather than a full characterisation -- others fail the same
// way:
//
//   a partially broadcast input into a target of       NoDonorDelta
//   period `S`, the last row of the table above
//
//   a target whose distributed dimension the index     IndexNotLaneAffine
//   cannot walk, `order` having transposed it
//
//      slice<layout<[1, 1, 16]>, dims = [2]>
//        ->  layout<[4, 4], order = [0, 1]>        on vector<4x4>
//
// Not layout conditions: SubByteElement, `gpu.shuffle` having no type for an
// element narrower than a byte, and FragmentSizeMismatch, an invariant check
// for a layout disagreeing with the vector type the caller derived from it.
//
// The index expression
// --------------------
//
// The index has to be computable from the lane id, so it is restricted to
//
//   index(slot) = stride * ((slot / dimStride) % dimExtent) + offset
//
// `(slot / dimStride) % dimExtent` is the lane's coordinate along one dimension
// of the target lane_layout: `dimExtent` is that dimension's size and
// `dimStride` how many consecutive slots share a coordinate. `stride` and
// `offset` place that coordinate in the donor's fragment. An arbitrary index
// per slot would need a materialized table and a gather; this costs a couple of
// `arith` ops on the lane id.
//
// The cases, `p` being the position in the target fragment:
//
//   case  lanePeriod  index(slot)     stride  dimStride  dimExtent  offset
//   1         4       2*slot + off      2         1          4      0,1,8,9
//   2         8       slot              1         1          8        0
//   3        16       slot/8 + 2*p      1         8          2       2*p
//   4         2       p % 2             0         1          1      p % 2
//
// Where the constants come from: the `2`s multiplying `slot` are the size of
// the input fragment's innermost dimension, which in cases 1 and 3 is the whole
// 8x2 value, so stepping one row costs 2 elements. In case 2 the fragment is a
// single column, its row stride is 1, and no `2` appears. Case 4's index does
// not depend on the slot at all: each target row arrives whole from one donor,
// so `stride` is 0 and `offset` alone selects the column.
//
// Read the expression off case 1, where the four quantities are 4, 2, 1 and 4.
// On case 2's `layout<[8, 1]>` target they collide: the lane period (8), its
// own `donorDelta` (8) and case 3's `dimStride` (8) all have the same value
// while being three different things -- a lane count, a lane offset and an
// index divisor.
//
// Algorithm
// ---------
//
//   tabulate inputOwned[lane] and targetOwned[lane]  // element coordinates
//   for each element `pos` of the target fragment:
//     for donorDelta = 0, lanePeriod, 2 * lanePeriod, ...  // smallest first
//       for each slot:
//         needed[slot] = index of the element targetOwned[slot][pos] in the
//                        fragment of its donor, lane `slot + donorDelta`
//         if that donor does not hold the element: give up on this donorDelta
//       look for one stride/offset/dimStride/dimExtent whose index reproduces
//       every needed[slot]; if there is none: next donorDelta
//       accept (donorDelta, index) and stop
//     if no donorDelta fits: fail, the layout change is not of this form
//   emit, per element source: an extract at index(slot), a `gpu.shuffle idx`
//     from `slot + donorDelta` unless that is the lane itself, and an insert.
//
// The shuffle is dropped exactly when `donorDelta` is 0, since the extract then
// lands on an element the lane owns. Replication makes that a choice: several
// donor groups may hold the element and all give the same value, so the
// smallest `donorDelta` is taken, which is what leaves a fully broadcast input
// shuffle-free.
//
// Only the first `lanePeriod` lanes are required to end up correct -- each is
// the one lane of its slot the target layout distributes to, and the layout
// assigns the rest no elements. Lanes sharing a slot run the same arithmetic on
// their own fragment: when a shuffle is emitted they all receive the donor's
// value and are equally correct, and when it is dropped their local extract can
// land on a different element. Dropping one therefore relies on nobody reading
// them.
//
// An index is derived from the tables and then verified against them, so a
// layout change that is not of this form is reported as a match failure rather
// than lowered incorrectly.
//
//===----------------------------------------------------------------------===//

/// The lane period of `layout`: the modulus after which its assignment of
/// elements to lanes repeats, so that lane `l` and lane `l + getLanePeriod()`
/// hold the same *fragment* -- the set of elements one lane owns.
///
/// It repeats because `computeStaticDistributedCoords` delinearizes the lane id
/// into digits, one per lane_layout dimension, taking each modulo that
/// dimension's extent. Nothing a layout derives from the lane id can therefore
/// depend on it beyond the product of those extents, which is what this
/// returns. A subgroup larger than the product simply repeats.
///
/// This is not the number of lanes holding distinct data. A sliced dimension
/// contributes its lanes to the period even though they replicate one fragment:
/// `slice<layout<lane_layout = [8, 1, 2]>, dims = [0]>` has period 16 and hands
/// out 2 distinct fragments. Nothing here needs the latter count -- the
/// ownership tables are consulted per lane -- and the period is what the lane
/// id has to be reduced by.
static int64_t getLanePeriod(xegpu::DistributeLayoutAttr layout) {
  // flatten() coalesces nested slices, so the parent it leaves is always a
  // plain LayoutAttr carrying the lane_layout of the whole subgroup.
  if (auto sliceAttr = dyn_cast<xegpu::SliceAttr>(layout))
    layout = cast<xegpu::LayoutAttr>(sliceAttr.flatten().getParent());
  return computeProduct(layout.getEffectiveLaneLayoutAsInt());
}

/// Why a `convert_layout` is not the redistribution
/// `redistributeBroadcastedValue` implements. Every failure path below reports
/// one of these, so the limitations listed in the section header are exactly
/// the values here.
enum class RedistributionLimit {
  LaneDataDiffers,
  LaneDataNotUnit,
  LanePeriodNotDivisor,
  SubByteElement,
  FragmentSizeMismatch,
  NoDonorDelta,
  IndexNotLaneAffine,
};

static StringRef describe(RedistributionLimit limit) {
  switch (limit) {
  case RedistributionLimit::LaneDataDiffers:
    return "input and target lane_data differ";
  case RedistributionLimit::LaneDataNotUnit:
    return "lane_data is not all ones";
  case RedistributionLimit::LanePeriodNotDivisor:
    return "the target lane period does not divide the subgroup size";
  case RedistributionLimit::SubByteElement:
    return "no gpu.shuffle type carries the element";
  case RedistributionLimit::FragmentSizeMismatch:
    return "a layout does not distribute what its vector type accounts for";
  case RedistributionLimit::NoDonorDelta:
    return "no donor lane holds the element for every slot";
  case RedistributionLimit::IndexNotLaneAffine:
    return "the fragment index is not an affine function of the lane id";
  }
  llvm_unreachable("unhandled RedistributionLimit");
}

/// Relational rule: both layouts have to hand the lanes the same distribution
/// unit -- the `lane_data`-sized block of elements a lane owns as a whole -- so
/// that only the assignment of those blocks to lanes changes.
static bool haveSameDistributionUnit(xegpu::DistributeLayoutAttr inputLayout,
                                     xegpu::DistributeLayoutAttr targetLayout) {
  return inputLayout.getEffectiveLaneDataAsInt() ==
         targetLayout.getEffectiveLaneDataAsInt();
}

/// Target-side rule: the target's lane period has to divide the subgroup size.
///
/// A value is moved between lanes by having every lane extract one element and
/// `gpu.shuffle idx` it from a chosen *donor* lane. All lanes run the same
/// code, so a lane can only be served by a donor that agrees with it on which
/// element to extract -- one whose lane id differs by a whole number of lane
/// periods. The lowering therefore searches donors at `donorDelta = 0,
/// lanePeriod, 2 * lanePeriod, ...` up to the subgroup size, reading lane `slot
/// + donorDelta`, where `slot` is `lane % lanePeriod` and so below the period.
/// If the period did not divide the subgroup size that sum could name a lane
/// past the last one: with a period of 3 on 16 lanes, `donorDelta` reaches 15
/// and a slot reaches 2, naming lane 17.
///
/// The input has no counterpart: it is only ever indexed, never used to derive
/// a donor, so its own period is unconstrained.
static bool isSupportedLanePeriod(xegpu::DistributeLayoutAttr targetLayout,
                                  int64_t subgroupSize) {
  int64_t lanePeriod = getLanePeriod(targetLayout);
  return lanePeriod > 0 && subgroupSize % lanePeriod == 0;
}

/// Matches the layout change `redistributeBroadcastedValue` implements.
///
/// Neither layout has to lay out the whole subgroup. A layout whose lane_layout
/// covers fewer lanes replicates its fragments over the rest, because
/// `computeStaticDistributedCoords` delinearizes the lane id modulo each
/// lane_layout extent; the fragment of every lane is therefore defined either
/// way. `redistributeBroadcastedValue` verifies the assignment element-wise, so
/// this only has to be permissive enough not to reject what it can lower.
///
/// That the layouts differ is not checked: they are already known to be
/// incompatible here, and the difference may equally be in `order`.
static bool
isBroadcastRedistribution(xegpu::DistributeLayoutAttr inputLayout,
                          xegpu::DistributeLayoutAttr targetLayout,
                          int64_t subgroupSize,
                          std::optional<RedistributionLimit> &limit) {
  if (!haveSameDistributionUnit(inputLayout, targetLayout)) {
    limit = RedistributionLimit::LaneDataDiffers;
    return false;
  }
  if (!isSupportedLanePeriod(targetLayout, subgroupSize)) {
    limit = RedistributionLimit::LanePeriodNotDivisor;
    return false;
  }
  return true;
}

/// Ownership table: `owned[lane][i]` is the coordinate, in the undistributed
/// shape, of element `i` of the fragment `lane` holds.
using OwnedCoords = SmallVector<SmallVector<SmallVector<int64_t>>>;

/// Tabulates the fragment every lane of the subgroup owns of `shape` under
/// `layout`: `owned[lane][i]` is the coordinate of element `i` of that
/// fragment, numbered as the flattened fragment `vector.extract` indexes.
///
/// Fails for non-unit lane_data, `LaneDataNotUnit` above.
static FailureOr<OwnedCoords>
computeOwnedCoords(xegpu::DistributeLayoutAttr layout, ArrayRef<int64_t> shape,
                   int64_t subgroupSize,
                   std::optional<RedistributionLimit> &limit) {
  SmallVector<int64_t> laneData = layout.getEffectiveLaneDataAsInt();
  if (!llvm::all_of(laneData, [](int64_t d) { return d == 1; })) {
    limit = RedistributionLimit::LaneDataNotUnit;
    return failure();
  }
  OwnedCoords owned;
  for (int64_t lane = 0; lane < subgroupSize; lane++)
    owned.push_back(layout.computeStaticDistributedCoords(lane, shape));
  return owned;
}

/// Which element of the donor's fragment a lane extracts -- the donor being the
/// lane it reads from, `slot + donorDelta` -- as a function of its `slot`
/// (`lane % lanePeriod`).
///
/// `at()` is restricted to one shape: the lane's coordinate along a single
/// dimension of the target lane_layout, scaled and offset into the fragment.
/// That keeps the index a couple of `arith` ops on the lane id; an arbitrary
/// index per slot would need a materialized table and a gather. A `stride` of 0
/// is the degenerate case where no dimension is walked at all and every slot
/// reads `offset`.
///
/// For a `vector<8x2>` whose input layout gives every lane all 16 elements,
/// row-major, and whose target gives lane `i` row `i`, element `p` of the
/// target fragment sits at `2 * slot + p`: `stride` 2, `offset` p, `dimStride`
/// 1, `dimExtent` 8.
struct FragmentIndex {
  /// Distance in the fragment between consecutive coordinates along the
  /// dimension being walked, or 0 when every slot reads the same element.
  int64_t stride;
  /// Fragment index that coordinate 0 maps to, and the whole index when
  /// `stride` is 0.
  int64_t offset;
  /// Slots apart for the coordinate to advance by one.
  int64_t dimStride;
  /// How many values the coordinate takes before wrapping. This is the extent
  /// of the target `lane_layout` dimension being walked, except when `stride`
  /// is 0 and no dimension is: then it is 1, as is `dimStride`.
  int64_t dimExtent;

  /// Compile-time twin of the arithmetic the emitter builds.
  int64_t at(int64_t slot) const {
    return stride * ((slot / dimStride) % dimExtent) + offset;
  }

  /// Key identifying the value this extracts, for reuse across elements.
  std::tuple<int64_t, int64_t, int64_t, int64_t> asTuple() const {
    return {stride, offset, dimStride, dimExtent};
  }
};

/// Where one element of the target fragment comes from.
struct ElementSource {
  /// Which element of the donor's fragment to extract.
  FragmentIndex index;
  /// Lanes from the reader to the donor it reads through `gpu.shuffle idx`: the
  /// donor is lane `slot + donorDelta`. Always a multiple of the lane period,
  /// so that donor and reader agree on the index to extract.
  int64_t donorDelta;
  /// False when the donor is the reader itself, which is exactly `donorDelta`
  /// being 0, and the extract is already local.
  bool needsShuffle;
};

/// Fits `needed[slot]`, the fragment index each slot has to read, to the
/// restricted form `FragmentIndex` can emit. A slot is a lane reduced modulo
/// the lane period, so there are `lanePeriod` of them. Fails when no index in
/// that form reproduces the whole table.
///
/// For `needed = [0, 0, 2, 2]`: the first change is at slot 2, so pairs of
/// slots share a coordinate (`dimStride` 2) and consecutive coordinates are 2
/// elements apart (`stride` 2); the table reaches 2, one step, so the
/// coordinate takes 2 values (`dimExtent` 2); and slot 0 reads 0 (`offset` 0).
static FailureOr<FragmentIndex> fitFragmentIndex(ArrayRef<int64_t> needed,
                                                 int64_t lanePeriod) {
  // Slots sharing an index are adjacent, so the first change in `needed` gives
  // both how many of them there are and how far apart their elements sit.
  int64_t offset = needed[0];
  int64_t dimStride = 1;
  while (dimStride < lanePeriod && needed[dimStride] == offset)
    dimStride++;
  int64_t stride = dimStride < lanePeriod ? needed[dimStride] - offset : 0;
  if (stride < 0 || lanePeriod % dimStride != 0)
    return failure();

  // The index wraps once it has taken all its values, so how far `needed` gets
  // from `offset` gives the dimExtent.
  int64_t dimExtent = 1;
  if (stride == 0) {
    // Every slot reads `offset`; no dimension is involved.
    dimStride = 1;
  } else {
    for (int64_t index : needed) {
      int64_t diff = index - offset;
      if (diff < 0 || diff % stride != 0)
        return failure();
      dimExtent = std::max(dimExtent, diff / stride + 1);
    }
  }

  // Derived from the table, so verify against the table before relying on it.
  FragmentIndex index{stride, offset, dimStride, dimExtent};
  if (!llvm::all_of(llvm::seq<int64_t>(0, lanePeriod), [&](int64_t slot) {
        return needed[slot] == index.at(slot);
      }))
    return failure();
  return index;
}

/// Works out where element `pos` of the target fragment comes from: which donor
/// lane, at `slot + donorDelta` for some multiple of the lane period, and which
/// element of that donor's fragment. Candidates are tried a donor group at a
/// time -- the lanes one `donorDelta` names, one per slot -- because every slot
/// has to be served by the same `donorDelta`. The smallest is preferred, so a
/// lane that already holds the element needs no shuffle.
///
/// Fails when no donor group holds the element for every slot, or when the
/// indices they would be read at are not of `FragmentIndex`'s form.
static FailureOr<ElementSource>
deriveElementSource(const OwnedCoords &inputOwned,
                    const OwnedCoords &targetOwned, int64_t pos,
                    int64_t subgroupSize, int64_t lanePeriod,
                    std::optional<RedistributionLimit> &limit) {
  // Index of the element with coordinate `coord` in the fragment `lane` holds,
  // or -1 if it holds none.
  auto findInFragment = [&](int64_t lane, ArrayRef<int64_t> coord) -> int64_t {
    for (auto [idx, candidate] : llvm::enumerate(inputOwned[lane]))
      if (ArrayRef<int64_t>(candidate) == coord)
        return idx;
    return -1;
  };

  // Donor groups, smallest `donorDelta` first. `sawCompleteDonor` records
  // whether any group held the element at all, which is what distinguishes the
  // two ways this fails.
  bool sawCompleteDonor = false;
  for (int64_t donorDelta = 0; donorDelta < subgroupSize;
       donorDelta += lanePeriod) {
    // Where in its fragment each slot's donor keeps the element it needs.
    SmallVector<int64_t> needed;
    for (int64_t slot = 0; slot < lanePeriod; slot++) {
      int64_t index = findInFragment(slot + donorDelta, targetOwned[slot][pos]);
      if (index < 0)
        break;
      needed.push_back(index);
    }
    if (static_cast<int64_t>(needed.size()) != lanePeriod)
      continue;
    sawCompleteDonor = true;

    FailureOr<FragmentIndex> index = fitFragmentIndex(needed, lanePeriod);
    if (failed(index))
      continue;

    // The donor is the lane itself exactly when `donorDelta` is 0:
    // `needed[slot]` was located in the fragment of lane `slot + donorDelta`,
    // so at 0 every slot's extract is local by construction.
    return ElementSource{*index, donorDelta, /*needsShuffle=*/donorDelta != 0};
  }
  limit = sawCompleteDonor ? RedistributionLimit::IndexNotLaneAffine
                           : RedistributionLimit::NoDonorDelta;
  return failure();
}

/// Redistributes `src`, the fragment the lane holds under `inputLayout`, into
/// the `resTy` fragment it owns under `targetLayout`. Fails if the layout
/// change is not of the form described above.
static FailureOr<Value> redistributeBroadcastedValue(
    ConversionPatternRewriter &rewriter, Location loc, Value src,
    VectorType resTy, xegpu::DistributeLayoutAttr inputLayout,
    xegpu::DistributeLayoutAttr targetLayout, ArrayRef<int64_t> shape,
    int64_t subgroupSize, std::optional<RedistributionLimit> &limit) {
  auto srcTy = dyn_cast<VectorType>(src.getType());
  if (!srcTy)
    return failure();
  int64_t srcNumElems = srcTy.getNumElements();
  int64_t resNumElems = resTy.getNumElements();
  int64_t lanePeriod = getLanePeriod(targetLayout);
  if (!isSupportedLanePeriod(targetLayout, subgroupSize)) {
    limit = RedistributionLimit::LanePeriodNotDivisor;
    return failure();
  }
  // gpu.shuffle only moves the integer widths a lane can hold.
  int64_t elemBitWidth = srcTy.getElementTypeBitWidth();
  if (!llvm::isPowerOf2_64(elemBitWidth) || elemBitWidth < 8 ||
      elemBitWidth > 64) {
    limit = RedistributionLimit::SubByteElement;
    return failure();
  }

  // Tabulate ownership: what each lane brings and what each slot needs.
  auto inputOwned = computeOwnedCoords(inputLayout, shape, subgroupSize, limit);
  auto targetOwned =
      computeOwnedCoords(targetLayout, shape, subgroupSize, limit);
  if (failed(inputOwned) || failed(targetOwned))
    return failure();
  // Require the layouts to distribute what the vector types account for. The
  // lanes past the first `lanePeriod` are replicas by construction: both
  // computeStaticDistributedCoords implementations delinearize the lane id
  // modulo the lane_layout extents, whose product is `lanePeriod`.
  for (int64_t lane = 0; lane < subgroupSize; lane++) {
    assert((*targetOwned)[lane] == (*targetOwned)[lane % lanePeriod] &&
           "target ownership is not periodic in the lane period");
    if (static_cast<int64_t>((*inputOwned)[lane].size()) != srcNumElems ||
        static_cast<int64_t>((*targetOwned)[lane].size()) != resNumElems) {
      limit = RedistributionLimit::FragmentSizeMismatch;
      return failure();
    }
  }

  // Work out where every element of the result comes from before emitting
  // anything, so that an unsupported layout change leaves no ops behind.
  SmallVector<ElementSource> sources;
  for (int64_t pos = 0; pos < resNumElems; pos++) {
    FailureOr<ElementSource> source = deriveElementSource(
        *inputOwned, *targetOwned, pos, subgroupSize, lanePeriod, limit);
    if (failed(source))
      return failure();
    sources.push_back(*source);
  }

  // The lane's fragment, flattened and bitcast to integers: shuffles move
  // same-width integers, which any lane data type bitcasts to.
  Type elemTy = srcTy.getElementType();
  Type shuffleTy = rewriter.getIntegerType(elemBitWidth);
  Value fragment = src;
  auto flatFragmentTy = VectorType::get({srcNumElems}, elemTy);
  if (srcTy != flatFragmentTy)
    fragment =
        vector::ShapeCastOp::create(rewriter, loc, flatFragmentTy, fragment);
  if (elemTy != shuffleTy)
    fragment = vector::BitCastOp::create(
        rewriter, loc, VectorType::get({srcNumElems}, shuffleTy), fragment);

  // The lane's slot, which every fragment index is a function of.
  Value slot = gpu::LaneIdOp::create(rewriter, loc, rewriter.getIndexType(),
                                     /*upperBound=*/mlir::IntegerAttr());
  if (lanePeriod != subgroupSize)
    slot = arith::RemUIOp::create(
        rewriter, loc, slot,
        arith::ConstantIndexOp::create(rewriter, loc, lanePeriod));

  Type i32Ty = rewriter.getI32Type();
  auto flatResTy = VectorType::get({resNumElems}, shuffleTy);
  Value res = arith::ConstantOp::create(rewriter, loc, flatResTy,
                                        rewriter.getZeroAttr(flatResTy));
  Value width;
  Value slotI32;

  // The lane's index along the selected dimension, shared by sources over the
  // same one. Either step is skipped when it is a no-op over `slot`.
  llvm::DenseMap<std::pair<int64_t, int64_t>, Value> dimIndices;
  auto getDimIndex = [&](const FragmentIndex &index) -> Value {
    Value &dimIndex = dimIndices[{index.dimStride, index.dimExtent}];
    if (dimIndex)
      return dimIndex;
    dimIndex = slot;
    if (index.dimStride != 1)
      dimIndex = arith::DivUIOp::create(
          rewriter, loc, dimIndex,
          arith::ConstantIndexOp::create(rewriter, loc, index.dimStride));
    if (index.dimExtent < lanePeriod / index.dimStride)
      dimIndex = arith::RemUIOp::create(
          rewriter, loc, dimIndex,
          arith::ConstantIndexOp::create(rewriter, loc, index.dimExtent));
    return dimIndex;
  };

  // The index the lane extracts.
  auto getFragmentIndex = [&](const FragmentIndex &index) -> OpFoldResult {
    // Every slot reads the same element; no dimension is involved.
    if (index.stride == 0)
      return rewriter.getIndexAttr(index.offset);
    Value value = getDimIndex(index);
    if (index.stride != 1)
      value = arith::MulIOp::create(
          rewriter, loc, value,
          arith::ConstantIndexOp::create(rewriter, loc, index.stride));
    if (index.offset != 0)
      value = arith::AddIOp::create(
          rewriter, loc, value,
          arith::ConstantIndexOp::create(rewriter, loc, index.offset));
    return value;
  };

  // Extract, shuffle and insert one element of the result at a time: one
  // extract per distinct fragment index, one shuffle per element the lane does
  // not already hold.
  // TODO: elements sharing a donor could travel in one shuffle, `gpu.shuffle`
  // moving a whole 32-bit lane word.
  llvm::DenseMap<std::tuple<int64_t, int64_t, int64_t, int64_t>, Value>
      extracted;
  for (auto [pos, source] : llvm::enumerate(sources)) {
    // Elements reading the same index of the fragment share the extract.
    Value &value = extracted[source.index.asTuple()];
    if (!value)
      value = vector::ExtractOp::create(rewriter, loc, fragment,
                                        getFragmentIndex(source.index));

    Value owned = value;
    if (source.needsShuffle) {
      if (!width) {
        width =
            arith::ConstantIntOp::create(rewriter, loc, i32Ty, subgroupSize);
        slotI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, slot);
      }
      Value donor = slotI32;
      if (source.donorDelta != 0)
        donor =
            arith::AddIOp::create(rewriter, loc, donor,
                                  arith::ConstantIntOp::create(
                                      rewriter, loc, i32Ty, source.donorDelta));
      owned = gpu::ShuffleOp::create(rewriter, loc, owned, donor, width,
                                     gpu::ShuffleMode::IDX)
                  .getResult(0);
    }
    res = vector::InsertOp::create(rewriter, loc, owned, res, pos);
  }

  if (elemTy != shuffleTy)
    res = vector::BitCastOp::create(
        rewriter, loc, VectorType::get({resNumElems}, elemTy), res);
  if (res.getType() != resTy)
    res = vector::ShapeCastOp::create(rewriter, loc, resTy, res);
  return res;
}

/// Folds a subgroup-level ConvertLayout op with compatible lane layouts.
struct SgToLaneConvertLayout
    : public OpConversionPattern<xegpu::ConvertLayoutOp> {
  using OpConversionPattern<xegpu::ConvertLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputLayout = op.getEffectiveInputLayout();
    auto targetLayout = op.getTargetLayoutAttr();
    Type valType = op.getResult().getType();

    if (valType.isIntOrFloat()) {
      rewriter.replaceOp(op, op.getSource());
      return success();
    }

    auto resShape = cast<VectorType>(valType).getShape();
    SmallVector<int64_t> resShapeVec(resShape.begin(), resShape.end());

    // Equivalent layouts: the convert_layout is a no-op and folds to its
    // source.
    if (inputLayout.isCompatibleWith(targetLayout, resShapeVec,
                                     xegpu::LayoutKind::Lane)) {
      rewriter.replaceOp(op, adaptor.getSource());
      return success();
    }

    // Handle the special case where the conversion redistributes a value
    // across a fraction of the subgroup: the lane_layout shrinks along the
    // outer (distributed) dimension while lane_data stays the same. Only a
    // pure outer-dimension lane_layout change is supported, so the inner
    // lane_layout must be unit (making the outer dim the only distributed one)
    // and the outer lane_layout must be genuinely distributed (> 1), which
    // also rules out the degenerate [1, 1] layout.
    if (inputLayout.getEffectiveOrderAsInt() ==
            targetLayout.getEffectiveOrderAsInt() &&
        inputLayout.getRank() == 2 && targetLayout.getRank() == 2) {
      auto laneLayout = inputLayout.getEffectiveLaneLayoutAsInt();
      auto targetLaneLayout = targetLayout.getEffectiveLaneLayoutAsInt();
      auto laneData = inputLayout.getEffectiveLaneDataAsInt();
      auto targetLaneData = targetLayout.getEffectiveLaneDataAsInt();
      if (laneLayout.size() == 2 && targetLaneLayout.size() == 2 &&
          laneData == targetLaneData && laneLayout[1] == 1 &&
          targetLaneLayout[1] == 1 && laneLayout[0] > 1 &&
          laneLayout[0] != targetLaneLayout[0]) {
        FailureOr<Value> res = shuffleDataAsLaneLayoutChange(
            rewriter, op.getLoc(), adaptor.getSource(), laneLayout[0],
            targetLaneLayout[0]);
        if (succeeded(res)) {
          rewriter.replaceOp(op, *res);
          return success();
        }
      }
    }

    // Handle a pure `lane_data` repack: `lane_layout` and `order` are unchanged
    // and exactly one dimension's `lane_data` switches between round-robin
    // (lane_data 1) and contiguous (lane_data == run length). The elements per
    // lane are unchanged, but their assignment to lanes is not, so the data is
    // moved across lanes with `xegpu.lane_shuffle`. The changed dimension must
    // be one of the two innermost ones, since sg-to-lane distribution is 2D.
    if (inputLayout.getEffectiveOrderAsInt() ==
            targetLayout.getEffectiveOrderAsInt() &&
        inputLayout.getEffectiveLaneLayoutAsInt() ==
            targetLayout.getEffectiveLaneLayoutAsInt()) {
      auto laneLayout = inputLayout.getEffectiveLaneLayoutAsInt();
      auto laneData = inputLayout.getEffectiveLaneDataAsInt();
      auto targetLaneData = targetLayout.getEffectiveLaneDataAsInt();
      // Find the single dimension whose lane_data changed; bail out if more
      // than one differs.
      int64_t rank = laneData.size();
      int64_t repackDim = -1;
      bool multipleChanged = false;
      for (int64_t d = 0; d < rank; ++d)
        if (laneData[d] != targetLaneData[d]) {
          if (repackDim != -1)
            multipleChanged = true;
          repackDim = d;
        }

      // `repackDim` must be the distributed dim (lane_layout != 1) and the
      // other innermost dim non-distributed (lane_layout == 1).
      int64_t otherDim = repackDim == rank - 1 ? rank - 2 : rank - 1;
      bool laneLayoutOk = repackDim != -1 && laneLayout[repackDim] != 1 &&
                          (rank < 2 || laneLayout[otherDim] == 1);

      // Exactly one dimension must change, and it must be one of the two
      // innermost (>= rank - 2).
      if (repackDim != -1 && repackDim >= rank - 2 && !multipleChanged &&
          laneLayoutOk) {
        FailureOr<Value> res = repackLaneData(
            rewriter, op.getLoc(), adaptor.getSource(), repackDim,
            laneData[repackDim], targetLaneData[repackDim]);
        if (succeeded(res)) {
          rewriter.replaceOp(op, *res);
          return success();
        }
      }
    }

    // The two layouts assign the same distribution unit to different lanes, so
    // data has to move across lanes.
    const auto *uArch =
        xegpu::uArch::getUArch(xegpu::getChipStr(op).value_or(""));
    FailureOr<VectorType> resDistTy = xegpu::getDistVecTypeBasedOnLaneLayout(
        targetLayout, cast<VectorType>(valType));
    std::optional<RedistributionLimit> limit;
    if (uArch && succeeded(resDistTy)) {
      int64_t subgroupSize = uArch->getSubgroupSize();
      if (isBroadcastRedistribution(inputLayout, targetLayout, subgroupSize,
                                    limit)) {
        FailureOr<Value> res = redistributeBroadcastedValue(
            rewriter, op.getLoc(), adaptor.getSource(), *resDistTy, inputLayout,
            targetLayout, resShapeVec, subgroupSize, limit);
        if (succeeded(res)) {
          rewriter.replaceOp(op, *res);
          return success();
        }
      }
    }

    if (limit)
      return rewriter.notifyMatchFailure(
          op, Twine("convert_layout is not a redistribution this pattern "
                    "lowers: ") +
                  describe(*limit));
    return rewriter.notifyMatchFailure(
        op, "lowering incompatible convert_layout not yet supported");
  }
};

// Trivially distribute `vector.interleave`
struct SgToLaneVectorInterleave
    : public OpConversionPattern<vector::InterleaveOp> {
  using OpConversionPattern<vector::InterleaveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InterleaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto newOp = vector::InterleaveOp::create(
        rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Trivially distribute `vector.deinterleave`
struct SgToLaneVectorDeinterleave
    : public OpConversionPattern<vector::DeinterleaveOp> {
  using OpConversionPattern<vector::DeinterleaveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::DeinterleaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto newOp = vector::DeinterleaveOp::create(rewriter, op.getLoc(),
                                                adaptor.getSource());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct SgToLaneDpasMx : public OpConversionPattern<xegpu::DpasMxOp> {
  using OpConversionPattern<xegpu::DpasMxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::DpasMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto *uArch =
        xegpu::uArch::getUArch(xegpu::getChipStr(op).value_or(""));
    if (!uArch)
      return failure();
    if (!uArch->isSupportedInstruction(
            xegpu::uArch::InstructionKind::SubgroupScaledMatrixMultiplyAcc))
      return rewriter.notifyMatchFailure(
          op, "target uArch does not support scaled subgroup mma");
    // Check if the op has A, B and CD layouts attached.
    auto layoutA = cast<xegpu::LayoutAttr>(op.getLayoutAAttr());
    auto layoutB = cast<xegpu::LayoutAttr>(op.getLayoutBAttr());
    auto layoutCd = cast<xegpu::LayoutAttr>(op.getLayoutCdAttr());
    if (!layoutA || !layoutB || !layoutCd)
      return rewriter.notifyMatchFailure(
          op, "missing required layout attributes for DpasMxOp distribution");

    // Retrieve expected types, according to anchor layouts.
    auto expected1DTypeResult =
        xegpu::getDistributedVectorType(op.getType(), layoutCd);
    auto expected1DTypeA =
        xegpu::getDistributedVectorType(op.getA().getType(), layoutA);
    auto expected1DTypeB =
        xegpu::getDistributedVectorType(op.getB().getType(), layoutB);

    VectorType expected1DTypeScaleA, expected1DTypeScaleB;
    if (op.getScaleA()) {
      auto layoutScaleA = cast<xegpu::LayoutAttr>(op.getLayoutAScaleAttr());
      auto expected1DTypeScaleAOrFailure = xegpu::getDistributedVectorType(
          cast<VectorType>(op.getScaleA().getType()), layoutScaleA);
      if (failed(expected1DTypeScaleAOrFailure))
        return rewriter.notifyMatchFailure(
            op, "failed to calculate expected 1D vector type for scale A");
      expected1DTypeScaleA = expected1DTypeScaleAOrFailure.value();
    }
    if (op.getScaleB()) {
      auto layoutScaleB = cast<xegpu::LayoutAttr>(op.getLayoutBScaleAttr());
      auto expected1DTypeScaleBOrFailure = xegpu::getDistributedVectorType(
          cast<VectorType>(op.getScaleB().getType()), layoutScaleB);
      if (failed(expected1DTypeScaleBOrFailure))
        return rewriter.notifyMatchFailure(
            op, "failed to calculate expected 1D vector type for scale B");
      expected1DTypeScaleB = expected1DTypeScaleBOrFailure.value();
    }

    auto expectedNDTypeResult =
        xegpu::getDistVecTypeBasedOnLaneLayout(layoutCd, op.getType());
    if (failed(expected1DTypeResult) || failed(expected1DTypeA) ||
        failed(expected1DTypeB))
      return rewriter.notifyMatchFailure(
          op,
          "failed to calculate supported workitem 1D vector types for DpasOp "
          "from layouts");
    if (failed(expectedNDTypeResult))
      return rewriter.notifyMatchFailure(
          op, "unable to compute expected workitem vector type for DpasOp from "
              "lane layout");

    // Validate bit widths match uArch packed format requirements
    const auto *uArchInstruction = dyn_cast<
        xegpu::uArch::SubgroupScaledMatrixMultiplyAcc>(uArch->getInstruction(
        xegpu::uArch::InstructionKind::SubgroupScaledMatrixMultiplyAcc));
    assert(uArchInstruction);
    auto wiAType = expected1DTypeA.value();
    auto wiBType = expected1DTypeB.value();
    // Calculate total packed bit width = element bit width * vector size
    unsigned aPackedBitWidth =
        wiAType.getElementTypeBitWidth() * wiAType.getNumElements();
    unsigned bPackedBitWidth =
        wiBType.getElementTypeBitWidth() * wiBType.getNumElements();
    if (aPackedBitWidth % uArchInstruction->getPackedFormatBitSizeA())
      return rewriter.notifyMatchFailure(
          op, "A operand packed bit width must be a multiple of uArch packed "
              "format requirement");
    if (bPackedBitWidth % uArchInstruction->getPackedFormatBitSizeB())
      return rewriter.notifyMatchFailure(
          op, "B operand packed bit width must be a multiple of uArch packed "
              "format requirement");

    auto newOp = xegpu::DpasMxOp::create(
        rewriter, op->getLoc(), expected1DTypeResult.value(),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getA()),
                    expected1DTypeA.value()),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getB()),
                    expected1DTypeB.value()),
        op.getAcc()
            ? castValueTo(rewriter,
                          cast<TypedValue<VectorType>>(adaptor.getAcc()),
                          expected1DTypeResult.value())
            : nullptr,

        op.getScaleA()
            ? castValueTo(rewriter,
                          cast<TypedValue<VectorType>>(adaptor.getScaleA()),
                          expected1DTypeScaleA)
            : nullptr,
        op.getScaleB()
            ? castValueTo(rewriter,
                          cast<TypedValue<VectorType>>(adaptor.getScaleB()),
                          expected1DTypeScaleB)
            : nullptr,
        /** layoutA**/ nullptr,
        /** layoutB**/ nullptr, /** layoutCd**/ nullptr,
        /** layoutAScale**/ nullptr, /** layoutBScale**/ nullptr);
    // Explicitly set the new types to enable correct type materializations.
    rewriter.replaceOp(op, castValueTo(rewriter, newOp.getResult(),
                                       expectedNDTypeResult.value()));
    return success();
  }
};

struct XeGPUSgToLaneDistributePass
    : public xegpu::impl::XeGPUSgToLaneDistributeBase<
          XeGPUSgToLaneDistributePass> {
  void runOnOperation() override;
};

} // namespace

void XeGPUSgToLaneDistributePass::runOnOperation() {

  // Recover temporary operand layouts for usage in patterns.
  Operation *root = getOperation();
  if (!xegpu::recoverTemporaryLayouts(root)) {
    signalPassFailure();
    return;
  }

  // Collect existing UnrealizedConversionCastOps. These must be preserved.
  llvm::SmallSetVector<UnrealizedConversionCastOp, 8> existingCasts;
  root->walk(
      [&](UnrealizedConversionCastOp castOp) { existingCasts.insert(castOp); });
  // Perform a structural type conversion to convert structural ops to have WI
  // types. This will insert UnrealizedConversionCastOps to make the IR
  // valid.
  {
    ConversionTarget target(getContext());
    TypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    // Source (N:1) and target (1:1) materializations using
    // UnrealizedConversionCastOp.
    auto materializeCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);
    xegpu::populateXeGPUSgToLaneDistributeTypeConversions(typeConverter, root);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);
    xegpu::populateXeGPUSgToLaneDistributeTypeConversionAndLegality(
        typeConverter, patterns, target, root);
    target.addLegalOp<UnrealizedConversionCastOp>();
    (void)applyPartialConversion(root, target, std::move(patterns));
  }
  // Fold cancelling cast chains and erase dead casts.
  xegpu::cleanupUnrealizedConversionCasts(root, existingCasts);
  xegpu::removeTemporaryLayoutAttrs(getOperation());
}

void xegpu::populateXeGPUSgToLaneDistributeTypeConversions(
    TypeConverter &typeConverter, Operation *topLevelOp) {
  // Pass through any type by default; more specific conversions registered
  // below override this for TensorDescType and (distributing) VectorType.
  typeConverter.addConversion([](Type type) -> Type { return type; });
  // For TensorDescType, drop the layout attribute if any.
  typeConverter.addConversion([](TensorDescType type) -> Type {
    if (type.getLayoutAttr()) {
      return type.dropLayouts();
    }
    return type;
  });
  // For VectorType, distribute based on the lane layout (1:1 shape-changing
  // conversion). Uses xegpu::addVectorTypeConversion with a pre-computed
  // map for SCF loop block args (see precomputeLoopBlockArgTypes for the
  // rationale).
  auto getSubShapeAndCount = [](VectorType vecTy,
                                xegpu::DistributeLayoutAttr layout)
      -> std::pair<SmallVector<int64_t>, int> {
    auto distTyOrFailure = getDistVecTypeBasedOnLaneLayout(layout, vecTy);
    if (failed(distTyOrFailure))
      return {{}, 0};
    return {SmallVector<int64_t>(distTyOrFailure->getShape()), 1};
  };
  auto loopArgTypes =
      xegpu::precomputeLoopBlockArgTypes(topLevelOp, getSubShapeAndCount);
  xegpu::addVectorTypeConversion(typeConverter, getSubShapeAndCount,
                                 std::move(loopArgTypes));
}

void xegpu::populateXeGPUSgToLaneDistributeTypeConversionAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, Operation *topLevelOp) {
  populateXeGPUSgToLaneDistributeTypeConversions(typeConverter, topLevelOp);
  // CreateNdDescOp is legal only if its result type has no layout attribute.
  target.addDynamicallyLegalOp<xegpu::CreateNdDescOp>(
      [&](xegpu::CreateNdDescOp op) { return !op.getType().getLayoutAttr(); });
  // Any anchor XeGPU op is legal only if it has no anchor layout.
  target.addDynamicallyLegalDialect<xegpu::XeGPUDialect>([](Operation *op) {
    if (isa<xegpu::ConvertLayoutOp>(op))
      return false;
    auto anchorOp = dyn_cast<AnchorLayoutInterface>(op);
    if (!anchorOp)
      return true;
    return !anchorOp.getAnchorLayout();
  });
  // Arith constants are legal only if they have no temporary layout attribute.
  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [=](arith::ConstantOp op) -> bool {
        // If the result type is not a vector, it's legal.
        if (!isa<VectorType>(op.getResult().getType()))
          return true;
        return !xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
      });
  // In math and arith dialects, only handle elementwise ops with a single
  // result and with a result layout attribute.
  target.addDynamicallyLegalDialect<math::MathDialect, arith::ArithDialect>(
      [=](Operation *op) -> std::optional<bool> {
        // Only handle elementwise mappable ops
        if (!OpTrait::hasElementwiseMappableTraits(op))
          return true;
        // Only handle ops with single vector result
        if (op->getNumResults() != 1)
          return true;

        VectorType resultType =
            dyn_cast<VectorType>(op->getResult(0).getType());
        if (!resultType)
          return true;

        // Check if all operands are vectors of the same shape
        for (Value operand : op->getOperands()) {
          VectorType operandType = dyn_cast<VectorType>(operand.getType());
          if (!operandType || operandType.getShape() != resultType.getShape()) {
            return true;
          }
        }
        return !xegpu::getTemporaryLayout(dyn_cast<OpResult>(op->getResult(0)));
      });
  // vector::ReductionOp is legal only if its source has no distribute layout
  // attribute.
  target.addDynamicallyLegalOp<vector::ReductionOp>(
      [=](vector::ReductionOp op) -> bool {
        auto layout = xegpu::getDistributeLayoutAttr(op.getVector());
        return !layout;
      });
  // vector::MultiDimReductionOp op legality.
  target.addDynamicallyLegalOp<vector::MultiDimReductionOp>(
      [=](vector::MultiDimReductionOp op) -> bool {
        return !isValidSubgroupMultiReductionOp(op);
      });
  target.addDynamicallyLegalOp<vector::CreateMaskOp, vector::ConstantMaskOp,
                               vector::TransposeOp, vector::BitCastOp,
                               vector::ShapeCastOp, vector::StepOp,
                               vector::BroadcastOp>([=](Operation *op) -> bool {
    return !xegpu::getTemporaryLayout(op->getOpResult(0));
  });
  target.addDynamicallyLegalOp<vector::ExtractOp>(
      [=](vector::ExtractOp op) -> bool {
        if (!isa<VectorType>(op.getType()))
          return true;
        return !xegpu::getTemporaryLayout(op->getOpResult(0));
      });
  target.addDynamicallyLegalOp<vector::InsertOp>(
      [=](vector::InsertOp op) -> bool {
        return !xegpu::getTemporaryLayout(op->getOpResult(0));
      });
  target.addDynamicallyLegalOp<vector::ExtractStridedSliceOp>(
      [=](vector::ExtractStridedSliceOp op) -> bool {
        return !xegpu::getTemporaryLayout(op->getOpResult(0));
      });
  target.addDynamicallyLegalOp<vector::InsertStridedSliceOp>(
      [=](vector::InsertStridedSliceOp op) -> bool {
        return !xegpu::getTemporaryLayout(op->getOpResult(0));
      });
  target.addDynamicallyLegalOp<vector::InterleaveOp, vector::DeinterleaveOp>(
      [=](Operation *op) -> bool {
        return !xegpu::getTemporaryLayout(op->getOpResult(0));
      });
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  patterns.add<
      SgToLaneCreateNdDesc, SgToLaneLoadNd, SgToLaneStoreNd, SgToLaneDpas,
      SgToLaneElementWise, SgToLaneArithConstant, SgToLanePrefetchNd,
      SgToLaneLoadGather, SgToLaneStoreScatter, SgToLaneVectorReduction,
      SgToLaneMultiDimReduction, SgToLaneVectorExtract, SgToLaneVectorInsert,
      SgToLaneVectorExtractStridedSlice, SgToLaneVectorInsertStridedSlice,
      SgToLaneLoadMatrix, SgToLaneStoreMatrix, SgToLaneConvertLayout,
      SgToLaneVectorTranspose, SgToLaneVectorBitcast, SgToLaneVectorStep,
      SgToLaneVectorShapeCast, SgToLaneBroadcast,
      SgToLaneCreateMask<vector::CreateMaskOp>,
      SgToLaneCreateMask<vector::ConstantMaskOp>, SgToLaneVectorDeinterleave,
      SgToLaneVectorInterleave, SgToLaneDpasMx>(typeConverter,
                                                patterns.getContext());
}
