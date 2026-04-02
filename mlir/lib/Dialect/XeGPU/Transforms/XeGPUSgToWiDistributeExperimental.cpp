//===- XeGPUSgToWiDistributeExperimental.cpp - XeGPU SG to WI Pass --------===//
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpl.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
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
#define GEN_PASS_DEF_XEGPUSGTOWIDISTRIBUTEEXPERIMENTAL
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "xegpu-sg-to-wi-distribute-experimental"
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

/// Checks if all XeGPU anchor ops and vector results have valid layouts.
static LogicalResult verifyLayouts(Operation *root) {
  auto walkResult = root->walk([&](Operation *nestedOp) -> WalkResult {
    if (auto anchorOp = dyn_cast<xegpu::AnchorLayoutInterface>(nestedOp)) {
      auto layout = anchorOp.getAnchorLayout();
      if (!layout) {
        nestedOp->emitError("expected anchor layout attribute on operation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    // For each vector result, check if the op contains a result layout
    // attribute.
    for (OpResult result : nestedOp->getResults()) {
      if (isa<VectorType>(result.getType())) {
        auto layout = xegpu::getDistributeLayoutAttr(result);
        if (!layout) {
          nestedOp->emitError(
              "expected result layout attribute on vector result");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

/// A vector::MultiDimReductionOp at subgroup level in expected form if, it has
/// exactly 1 reduction dimension, it had valid result layout attribute, and
/// result type can be distributed to lanes using the layout.
static bool isValidSubgroupMultiReductionOp(vector::MultiDimReductionOp op) {
  auto resLayout = xegpu::getTemporaryLayout(op->getOpResult(0));
  // If no layout, not valid.
  if (!resLayout || !resLayout.isForSubgroup())
    return false;
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

/// A vector::MultiDimReductionOp is doing lane-local reduction if each workitem
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

/// Distributes a subgroup-level CreateNdDesc op to workitem-level CreateNdDesc
/// op. This simply drops the layout attribute from the tensor descriptor type.
struct SgToWiCreateNdDesc : public OpConversionPattern<xegpu::CreateNdDescOp> {
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

/// Distributes a subgroup-level LoadNd op to workitem-level LoadNd op. Output
/// of workitem-level LoadNd op is 1D. ShapeCast is added to restore the
/// original rank.
struct SgToWiLoadNd : public OpConversionPattern<xegpu::LoadNdOp> {
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
    auto uArch = getUArch(xegpu::getChipStr(op).value_or(""));
    if (!uArch)
      return rewriter.notifyMatchFailure(
          op, "xegpu::LoadNdOp require target attribute attached to "
              "determine transpose "
              "requirement");
    auto supportedWiResultTyOrFailure =
        xegpu::getDistributedVectorType(op.getTensorDescType());
    auto expectedWiResultTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, op.getType());
    if (failed(supportedWiResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute the workitem vector type for LoadNdOp");
    if (failed(expectedWiResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op,
          "unable to compute expected workitem vector type from lane layout");
    auto newOp = xegpu::LoadNdOp::create(
        rewriter, op.getLoc(), supportedWiResultTyOrFailure.value(),
        adaptor.getTensorDesc(), op.getMixedOffsets(), op.getPackedAttr(),
        op.getTransposeAttr(), op.getL1HintAttr(), op.getL2HintAttr(),
        op.getL3HintAttr(), /**layout**/ nullptr);
    // Set the packed attribute if the layout requires it.
    newOp.setPacked(xegpu::requirePacked(cast<xegpu::LayoutAttr>(layout)));
    // Set the transpose attribute if the layout requires it.
    if (xegpu::requireTranspose(cast<xegpu::LayoutAttr>(layout), uArch))
      newOp.setTranspose(DenseI64ArrayAttr::get(rewriter.getContext(), {1, 0}));
    rewriter.replaceOp(op, castValueTo(rewriter, newOp.getResult(),
                                       expectedWiResultTyOrFailure.value()));
    return success();
  }
};

/// Distributes a subgroup-level StoreNd op to workitem-level StoreNd op. Stored
/// value in workitem-level StoreNd op is 1D. ShapeCast is added to cast the
/// incoming value to 1D.
struct SgToWiStoreNd : public OpConversionPattern<xegpu::StoreNdOp> {
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
    auto supportedWiValueTyOrFailure =
        xegpu::getDistributedVectorType(op.getTensorDescType());
    if (failed(supportedWiValueTyOrFailure))
      return rewriter.notifyMatchFailure(
          op,
          "unable to compute wi vector type for StoreNdOp value from tensor "
          "descriptor");

    xegpu::StoreNdOp::create(
        rewriter, op.getLoc(),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getValue()),
                    supportedWiValueTyOrFailure.value()),
        adaptor.getTensorDesc(), op.getMixedOffsets(), op.getL1HintAttr(),
        op.getL2HintAttr(), op.getL3HintAttr(), /**layout**/ nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Distributes a subgroup-level Dpas op to workitem-level Dpas op. All inpputs
/// and output of workitem-level Dpas op are 1D. Necessary casts are added to
/// convert the inputs and output to/from 1D.
struct SgToWiDpas : public OpConversionPattern<xegpu::DpasOp> {
  using OpConversionPattern<xegpu::DpasOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::DpasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // llvm::errs() << "DpasOpPattern matchAndRewrite called\n";
    // Check if the op has A, B and CD layouts attached.
    auto layoutA = cast<xegpu::LayoutAttr>(op.getLayoutAAttr());
    auto layoutB = cast<xegpu::LayoutAttr>(op.getLayoutBAttr());
    auto layoutCd = cast<xegpu::LayoutAttr>(op.getLayoutCdAttr());
    if (!layoutA || !layoutB || !layoutCd)
      return failure();
    // llvm::errs() << "tryning to calculate wi types for dpas op\n";
    auto wiResultTyOrFailure =
        xegpu::getDistributedVectorType(op.getType(), layoutCd);
    auto wiATypeOrFailure =
        xegpu::getDistributedVectorType(op.getLhs().getType(), layoutA);
    auto wiBTypeOrFailure =
        xegpu::getDistributedVectorType(op.getRhs().getType(), layoutB);
    auto expectedWiResultTyOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layoutCd, op.getType());
    if (failed(wiResultTyOrFailure) || failed(wiATypeOrFailure) ||
        failed(wiBTypeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to calculate supported workitem vector types for DpasOp "
              "from layouts");
    if (failed(expectedWiResultTyOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute expected workitem vector type for DpasOp from "
              "lane layout");
    auto newOp = xegpu::DpasOp::create(
        rewriter, op->getLoc(), wiResultTyOrFailure.value(),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getLhs()),
                    wiATypeOrFailure.value()),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getRhs()),
                    wiBTypeOrFailure.value()),
        castValueTo(rewriter, cast<TypedValue<VectorType>>(adaptor.getAcc()),
                    wiResultTyOrFailure.value()),
        /** layoutA**/ nullptr,
        /** layoutB**/ nullptr, /** layoutCd**/ nullptr);
    // Explicitly set the new types to enable correct type materializations.
    rewriter.replaceOp(op, castValueTo(rewriter, newOp.getResult(),
                                       expectedWiResultTyOrFailure.value()));
    return success();
  }
};

/// Distributes elementwise ops to workitem-level elementwise ops. This
/// currently handles elementwise ops with single result only.
struct SgToWiElementWise : public ConversionPattern {
  SgToWiElementWise(TypeConverter &typeConverter, MLIRContext *ctx)
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

    auto wiShapeOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, resultType);

    if (failed(wiShapeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute workitem vector type from the layout");

    VectorType newResultType = wiShapeOrFailure.value();
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

/// Distributes a subgroup-level arith ConstantOp to workitem-level arith
/// ConstantOp.
struct SgToWiArithConstant : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType)
      return failure();

    // Only handle dense vector constants
    auto dense = dyn_cast<SplatElementsAttr>(op.getValue());
    if (!dense)
      return rewriter.notifyMatchFailure(
          op, "only dense splat vector constants are supported");

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(llvm::cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForSubgroup())
      return rewriter.notifyMatchFailure(
          op, "operation result does not have subgroup distribute layout");

    auto wiShapeOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(layout, resultType);

    if (failed(wiShapeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute workitem vector type from the layout");

    VectorType newResultType = wiShapeOrFailure.value();
    auto sclarValue = dense.getSplatValue<Attribute>();
    auto newDenseAttr = DenseElementsAttr::get(newResultType, sclarValue);

    auto newOp = arith::ConstantOp::create(rewriter, op.getLoc(), newResultType,
                                           newDenseAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Distributes a subgroup-level PrefetchNd op to workitem-level PrefetchNd op.
struct SgToWiPrefetchNd : public OpConversionPattern<xegpu::PrefetchNdOp> {
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

/// Distributes a subgroup-level LoadGather (xegpu.load) op to workitem-level.
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
struct SgToWiLoadGather : public OpConversionPattern<xegpu::LoadGatherOp> {
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
          op,
          "unable to compute expected workitem vector type from lane layout");

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
        op.getL3HintAttr(), /*layout=*/nullptr);

    Value result = newOp->getResult(0);
    if (distResultTy1D != distResultTy)
      result = castValueTo(rewriter, cast<TypedValue<VectorType>>(result),
                           distResultTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// This pattern distributes a subgroup-level vector.reduction op to
/// workitem-level. This require shuffling the data across the workitems (using
/// gpu::ShuffleOp) and reducing in stages until all workitems have the final
/// result.
struct SgToWiVectorReduction : public OpConversionPattern<vector::ReductionOp> {
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
    const uArch *uArch = getUArch(xegpu::getChipStr(op).value_or(""));
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

    // Get the distributed vector (per work-item portion).
    Value laneValVec = adaptor.getVector();

    // Distribute and reduce across work-items in the subgroup.
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
/// workitem-level only if the reduction is lane-local. This means that
/// reduction dimension is not distributed to lanes and each lane does its own
/// local reduction.
struct SgToWiMultiDimReduction
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
    if (isReductionLaneLocal(op)) {
      auto resLayout = xegpu::getTemporaryLayout(op->getOpResult(0));
      VectorType resVecTy = dyn_cast<VectorType>(op.getType());
      auto resDistVecTyOrFailure =
          getDistVecTypeBasedOnLaneLayout(resLayout, resVecTy);
      // For lane local reduction, simply create a new MultiDimReductionOp using
      // adaptor operands and the new result type.
      result = vector::MultiDimReductionOp::create(
          rewriter, op.getLoc(), resDistVecTyOrFailure.value(), op.getKind(),
          adaptor.getSource(), adaptor.getAcc(), op.getReductionDims());
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
/// When not using subgroup_block_io, each workitem computes its own
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

/// This pattern distributes a subgroup-level LoadMatrix op to workitem-level.
struct SgToWiLoadMatrix : public OpConversionPattern<xegpu::LoadMatrixOp> {
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

/// Distributes a subgroup-level vector.transpose op to workitem-level.
struct SgToWiVectorTranspose : public OpConversionPattern<vector::TransposeOp> {
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

/// Distributes a subgroup-level vector.bitcast op to workitem-level.
/// Bitcast only impacts the innermost dimension of the source/result vectors.
struct SgToWiVectorBitcast : public OpConversionPattern<vector::BitCastOp> {
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
/// to workitem-level. Uses `computeDistributedCoords()` to obtain the
/// coordinates each workitem owns, then compares each coordinate against the
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
struct SgToWiCreateMask : public OpConversionPattern<OpType> {
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
          op, "unable to compute workitem vector type from the layout");

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

/// This pattern distributes a subgroup-level StoreMatrix op to workitem-level.
struct SgToWiStoreMatrix : public OpConversionPattern<xegpu::StoreMatrixOp> {
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
/// workitem-level.
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
struct SgToWiStoreScatter : public OpConversionPattern<xegpu::StoreScatterOp> {
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
          op,
          "unable to compute expected workitem vector type from lane layout");

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
                                  op.getL3HintAttr(), /*layout=*/nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Distribute a vector::StepOp to workitem-level.
/// The layout must have exactly 1 effective lane dimension.
/// We completely resolve the vector::StepOp by computing the lane_data-sized
/// subranges.
struct SgToWiVectorStep : public OpConversionPattern<vector::StepOp> {
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
    auto wiShapeOrFailure =
        xegpu::getDistVecTypeBasedOnLaneLayout(resultLayout, stepResultVecTy);
    if (failed(wiShapeOrFailure))
      return rewriter.notifyMatchFailure(
          op, "unable to compute workitem vector type from the layout");
    VectorType newVecTy = wiShapeOrFailure.value();

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

/// Distributes a subgroup-level vector.extract op to workitem-level. Only
/// handles sub-vector extraction (result is VectorType, not scalar).
struct SgToWiVectorExtract : public OpConversionPattern<vector::ExtractOp> {
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

/// This pattern distributes a subgroup-level ShapeCast op to workitem-level.
struct SgToWiVectorShapeCast : public OpConversionPattern<vector::ShapeCastOp> {
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
/// workitem-level. If the result is distributed, the offsets and sizes are
/// adjusted to match the distributed types.
struct SgToWiVectorExtractStridedSlice
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
      const uArch *uArch = getUArch(xegpu::getChipStr(op).value_or(""));
      if (!uArch)
        return rewriter.notifyMatchFailure(
            op, "target attribute required to determine subgroup size");
      int subgroupSize = uArch->getSubgroupSize();
      auto sourceLayout = xegpu::getTemporaryLayout(op->getOpOperand(0));
      if (!sourceLayout || sourceLayout.getEffectiveLaneLayoutAsInt().empty())
        return rewriter.notifyMatchFailure(
            op, "source of extract_strided_slice lacks distribution layout");
      int sourceDistrDimSize = op.getSourceVectorType().getShape()[distDim];
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
/// workitem-level. The pattern supports three cases:
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
struct SgToWiBroadcast : public OpConversionPattern<vector::BroadcastOp> {
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
/// workitem-level. If the dest is distributed, the offsets are adjusted to
/// match the distributed types.
struct SgToWiVectorInsertStridedSlice
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

      const uArch *uArch = getUArch(xegpu::getChipStr(op).value_or(""));
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

/// Distributes a subgroup-level vector.insert op to workitem-level. Only
/// handles sub-vector insertion (value to store is VectorType, not scalar).
struct SgToWiVectorInsert : public OpConversionPattern<vector::InsertOp> {
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

/// Folds a subgroup-level ConvertLayout op with compatible lane layouts.
struct SgToWiConvertLayout
    : public OpConversionPattern<xegpu::ConvertLayoutOp> {
  using OpConversionPattern<xegpu::ConvertLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputLayout = op.getInputLayoutAttr();
    auto targetLayout = op.getTargetLayoutAttr();
    Type valType = op.getResult().getType();

    if (valType.isIntOrFloat()) {
      rewriter.replaceOp(op, op.getSource());
      return success();
    }

    auto resShape = cast<VectorType>(valType).getShape();
    SmallVector<int64_t> resShapeVec(resShape.begin(), resShape.end());
    if (!inputLayout.isCompatibleWith(targetLayout, resShapeVec,
                                      xegpu::LayoutKind::Lane)) {
      return rewriter.notifyMatchFailure(
          op, "lowering incompatible convert_layout not yet supported");
    }

    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

struct XeGPUSgToWiDistributeExperimentalPass
    : public xegpu::impl::XeGPUSgToWiDistributeExperimentalBase<
          XeGPUSgToWiDistributeExperimentalPass> {
  void runOnOperation() override;
};

} // namespace

void XeGPUSgToWiDistributeExperimentalPass::runOnOperation() {

  // Recover temporary operand layouts for usage in patterns.
  Operation *root = getOperation();
  if (!xegpu::recoverTemporaryLayouts(root)) {
    signalPassFailure();
    return;
  }

  // Verify if all XeGPU anchor ops and vector ops have result layouts.
  // TODO: This can be removed once the full layout refactoring is done.
  if (failed(verifyLayouts(root))) {
    LLVM_DEBUG(DBGS() << "XeGPUSgToWiDistributeExperimentalPass: layout "
                         "verification failed\n");
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
  auto materializeCast = [&](mlir::OpBuilder &builder, mlir::Type type,
                             mlir::ValueRange inputs,
                             mlir::Location loc) -> mlir::Value {
    UnrealizedConversionCastOp castOp =
        UnrealizedConversionCastOp::create(builder, loc, type, inputs);
    return castOp.getResult(0);
  };
  {
    ConversionTarget target(getContext());
    TypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);
    xegpu::populateXeGPUSgToWiDistributeTypeConversions(typeConverter);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);
    xegpu::populateXeGPUSgToWiDistributeTypeConversionAndLegality(
        typeConverter, patterns, target);
    target.addLegalOp<UnrealizedConversionCastOp>();
    (void)applyPartialConversion(root, target, std::move(patterns));
  }
  // Structural type conversion can generate some redundant
  // UnrealizedConversionCastOps to materialize the SG type from type converted
  // WI type. These are redundant at this point and can be eliminated by
  // inserting shape casts instead.
  // Example:
  // %1 = UnrealizedConversionCastOp %0 : vector<16x1xf32> to vector<16x16xf32>
  // %2 = UnrealizedConversionCastOp %1 : vector<16x16xf32> to vector<16xf32>
  // This can be replaced with:
  // %2 = vector.shape_cast %0 : vector<16x1xf32> to vector<16xf32>
  OpBuilder builder(root);
  root->walk([&](UnrealizedConversionCastOp op) {
    // If this op existed before, nothing to do.
    if (existingCasts.contains(op))
      return;
    // number of inputs and outputs must be 1.
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return;
    // Both input and output types must be vector types.
    auto singleInput = op.getInputs()[0];
    auto inputTy = dyn_cast<VectorType>(singleInput.getType());
    auto outputTy = dyn_cast<VectorType>(op.getResult(0).getType());
    if (!inputTy || !outputTy)
      return;

    // Check if the defining op of the input is also an
    // UnrealizedConversionCastOp and it has a single user (which is this
    // op).
    auto definingOp = singleInput.getDefiningOp<UnrealizedConversionCastOp>();
    if (!definingOp || !definingOp->hasOneUse())
      return;
    auto inputOfDefiningOp = definingOp.getInputs()[0];
    // If the input of the defining op and output type are both vector types
    // have same number of elements, insert a shape cast.
    auto inputOfDefiningOpTy =
        dyn_cast<VectorType>(inputOfDefiningOp.getType());
    if (inputOfDefiningOpTy &&
        inputOfDefiningOpTy.getNumElements() == outputTy.getNumElements()) {
      builder.setInsertionPoint(op);
      auto shapeCast = vector::ShapeCastOp::create(builder, op.getLoc(),
                                                   outputTy, inputOfDefiningOp);
      op.replaceAllUsesWith(ValueRange{shapeCast.getResult()});
      return;
    }
  });
  // At this point, we will have some dead UnrealizedConversionCastOps. Just
  // erase them.
  bool changed = true;
  while (changed) {
    changed = false;
    root->walk([&](UnrealizedConversionCastOp op) {
      // Skip existing casts.
      if (existingCasts.contains(op))
        return;
      if (op.use_empty()) {
        op.erase();
        changed = true;
      }
    });
  }
}

void xegpu::populateXeGPUSgToWiDistributeTypeConversions(
    TypeConverter &typeConverter) {
  // Any type other than TensorDescType and VectorType are legal as is.
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (!isa<TensorDescType, VectorType>(type))
      return type;
    return std::nullopt;
  });
  // For TensorDescType, drop the layout attribute if any.
  typeConverter.addConversion([](TensorDescType type) -> Type {
    if (type.getLayoutAttr()) {
      return type.dropLayouts();
    }
    return type;
  });
  // For VectorType, check if there is a distribute layout attribute on the
  // value. If so, convert to the distributed vector type based on the layout.
  typeConverter.addConversion([](Value v) -> std::optional<Type> {
    auto type = v.getType();
    // If value is not vector type, nothing to do.
    if (!isa<VectorType>(type))
      return std::nullopt;
    auto layout = xegpu::getDistributeLayoutAttr(v);
    if (!layout || !layout.isForSubgroup())
      return type;
    // Vector type is distributed based on lane layout.
    auto newTyOrFailure =
        getDistVecTypeBasedOnLaneLayout(layout, cast<VectorType>(type));
    if (failed(newTyOrFailure))
      return type;
    return *newTyOrFailure;
  });
}

void xegpu::populateXeGPUSgToWiDistributeTypeConversionAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  populateXeGPUSgToWiDistributeTypeConversions(typeConverter);
  // CreateNdDescOp is legal only if its result type has no layout attribute.
  target.addDynamicallyLegalOp<xegpu::CreateNdDescOp>(
      [&](xegpu::CreateNdDescOp op) { return !op.getType().getLayoutAttr(); });
  // Any anchor XeGPU op is legal only if it has no anchor layout.
  target.addDynamicallyLegalDialect<xegpu::XeGPUDialect>([](Operation *op) {
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
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  patterns.add<SgToWiCreateNdDesc, SgToWiLoadNd, SgToWiStoreNd, SgToWiDpas,
               SgToWiElementWise, SgToWiArithConstant, SgToWiPrefetchNd,
               SgToWiLoadGather, SgToWiStoreScatter, SgToWiVectorReduction,
               SgToWiMultiDimReduction, SgToWiVectorExtract, SgToWiVectorInsert,
               SgToWiVectorExtractStridedSlice, SgToWiVectorInsertStridedSlice,
               SgToWiLoadMatrix, SgToWiStoreMatrix, SgToWiConvertLayout,
               SgToWiVectorTranspose, SgToWiVectorBitcast, SgToWiVectorStep,
               SgToWiVectorShapeCast, SgToWiBroadcast,
               SgToWiCreateMask<vector::CreateMaskOp>,
               SgToWiCreateMask<vector::ConstantMaskOp>>(typeConverter,
                                                         patterns.getContext());
}
