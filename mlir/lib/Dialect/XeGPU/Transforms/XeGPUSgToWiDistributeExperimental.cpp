//===- XeGPUSgToWiDistributeExperimental.cpp - XeGPU SG to WI Pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
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
    const auto *uArch = getUArch(xegpu::getChipStr(op).value_or(""));
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
    // Only lane-local reduction is handled here.
    if (!isReductionLaneLocal(op))
      return rewriter.notifyMatchFailure(
          op, "Only lane-local reduction is supported, expected reduction "
              "dimension to be "
              "not distributed.");
    auto resLayout = xegpu::getTemporaryLayout(op->getOpResult(0));
    VectorType resVecTy = dyn_cast<VectorType>(op.getType());
    auto resDistVecTyOrFailure =
        getDistVecTypeBasedOnLaneLayout(resLayout, resVecTy);
    // Simply create a new MultiDimReductionOp using adaptor operands and the
    // new result type.
    auto newOp = vector::MultiDimReductionOp::create(
        rewriter, op.getLoc(), resDistVecTyOrFailure.value(), op.getKind(),
        adaptor.getSource(), adaptor.getAcc(), op.getReductionDims());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// This pattern rewrites a subgroup-level vector.multi_reduction op to a series
/// of vector.extract_strided_slice, vector.reduction and
/// vector.insert_strided_slice ops. This is used when the reduction dimension
/// is distributed to lanes and a naive (lane-local) distribution is not
/// possible. Then later on, these partially lowered subgroup-level ops are
/// further lowered to workitem-level by respective patterns.
struct LowerVectorMultiReductionPattern
    : public OpConversionPattern<vector::MultiDimReductionOp> {
  using OpConversionPattern<vector::MultiDimReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MultiDimReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only non-lane-local reduction is handled here.
    if (isReductionLaneLocal(op))
      return rewriter.notifyMatchFailure(
          op, "Reduction is lane-local, it does not require rewrite.");
    ArrayRef<int64_t> reductionDims = op.getReductionDims();
    assert(
        reductionDims.size() == 1 &&
        "Expecting single reduction dimension for subgroup multi reduction op");

    // Rewrite MultiDimReductionOp into a sequence of ReductionOps.
    Value result = xegpu::lowerToVectorReductions(
        cast<TypedValue<VectorType>>(op.getSource()),
        cast<TypedValue<VectorType>>(op.getAcc()), op.getKind(),
        reductionDims[0], op.getLoc(), rewriter);

    rewriter.replaceOp(op, result);
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

  // Verify if all XeGPU anchor ops and vector ops have result layouts.
  // TODO: This can be removed once the full layout refactoring is done.
  Operation *root = getOperation();
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
        // Check common conditions for subgroup multi reduction op.
        if (!isValidSubgroupMultiReductionOp(op))
          return true;
        // Lane local reductions are illegal at this point and must be lowered.
        return !isReductionLaneLocal(op);
      });
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  patterns.add<SgToWiCreateNdDesc, SgToWiLoadNd, SgToWiStoreNd, SgToWiDpas,
               SgToWiElementWise, SgToWiArithConstant, SgToWiPrefetchNd,
               SgToWiVectorReduction, SgToWiMultiDimReduction>(
      typeConverter, patterns.getContext());
}

void xegpu::populateXeGPUSgToWiLowerVectorMultiReductionAndLegality(
    RewritePatternSet &patterns, ConversionTarget &target) {
  // vector::MultiDimReductionOp legality.
  target.addDynamicallyLegalOp<vector::MultiDimReductionOp>(
      [&](vector::MultiDimReductionOp op) {
        // Check common conditions for subgroup multi reduction op.
        if (!isValidSubgroupMultiReductionOp(op))
          return true;
        // Lane local reductions are legal. We only rewrite non-lane-local
        // reductions.
        return isReductionLaneLocal(op);
      });
  // vector::ReductionOp is legal.
  target.addDynamicallyLegalOp<vector::ReductionOp>(
      [&](vector::ReductionOp op) { return true; });
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  patterns.add<LowerVectorMultiReductionPattern>(patterns.getContext());
}
