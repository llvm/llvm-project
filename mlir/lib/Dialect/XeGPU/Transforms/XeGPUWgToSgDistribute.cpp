//===- XeGPUWgToSgDistribute.cpp - XeGPU Workgroup to Subgroup Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpl.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUWGTOSGDISTRIBUTE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;

namespace {

// Marker attribute carrying the common workgroup layout onto create_mem_desc so
// the conversion pattern can shrink the private buffer to the per-subgroup size.
// Set by the SLM privatization pre-phase and consumed by WgToSgCreateMemDescOp.
static constexpr StringLiteral kPrivatizeLayoutAttrName =
    "__xegpu_privatize_layout__";

// Retrieve the RangeAttr if it is specified.
static xegpu::RangeAttr getRangeSpecAttr(Operation *op) {
  Operation *parent = op->getParentOfType<scf::IfOp>();
  while (parent) {
    if (auto attr = llvm::dyn_cast_if_present<xegpu::RangeAttr>(
            parent->getAttr("sg_id_range")))
      return attr;
    parent = parent->getParentOfType<scf::IfOp>();
  }
  return {};
}

static std::pair<SmallVector<int64_t>, int>
getSgShapeAndCount(ArrayRef<int64_t> shape,
                   xegpu::DistributeLayoutAttr layout) {
  int count = 1;
  SmallVector<int64_t> sgShape(shape);
  auto distributedShape = layout.computeDistributedShape(
      SmallVector<int64_t>(shape.begin(), shape.end()));
  if (failed(distributedShape))
    return std::make_pair(sgShape, count);
  auto sgData = layout.getEffectiveSgDataAsInt();
  count = computeProduct(distributedShape.value()) / computeProduct(sgData);
  return std::make_pair(sgData, count);
}

/// Utility helper for deriving a list of offsets for each sub-TensorDescs
/// or sub-MemDescs to be accessed by current subgroup (sgId) based on the
/// associated distribute layout attribute, the shape, subgroup id and the
/// original offsets of the op
template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<
              OpType, xegpu::LoadNdOp, xegpu::StoreNdOp, xegpu::PrefetchNdOp,
              xegpu::LoadMatrixOp, xegpu::StoreMatrixOp>::value>>
static LogicalResult
genOffsetsList(ConversionPatternRewriter &rewriter, OpType op,
               SmallVector<SmallVector<OpFoldResult>> &offsetsList) {
  Location loc = op.getLoc();
  SmallVector<OpFoldResult> origOffsets = op.getMixedOffsets();
  // not applicable to ops without offsets operands.
  if (origOffsets.empty())
    return failure();

  // if op is xegpu::CreateNdDescOp, call op.getDescLayoutAttr()
  xegpu::DistributeLayoutAttr layout;
  if constexpr (std::is_same_v<OpType, xegpu::LoadMatrixOp> ||
                std::is_same_v<OpType, xegpu::StoreMatrixOp>) {
    layout = op.getLayoutAttr();
  } else {
    layout = op.getDescLayoutAttr();
  }

  // not applicable to ops without workgroup layout attributes
  if (!layout || !layout.isForWorkgroup())
    return failure();

  Value sgId =
      gpu::SubgroupIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);

  // verify and adjust the sgId if the range specifier is present
  xegpu::RangeAttr sgIdRange = getRangeSpecAttr(op);
  if (sgIdRange) {
    int64_t startOfRange = sgIdRange.getStart().getInt();
    int64_t endOfRange = sgIdRange.getEnd().getInt();
    // verify the RangeAttr against the layout attribute
    if (layout.getNumSubgroups() != endOfRange - startOfRange)
      return rewriter.notifyMatchFailure(
          op, "sg_layout size must match the sg_id_range");
    // adjust the sgId if necessary
    if (startOfRange > 0) {
      Value startOfRangeVal =
          arith::ConstantIndexOp::create(rewriter, loc, startOfRange);
      sgId = index::SubOp::create(rewriter, loc, sgId, startOfRangeVal);
    }
  }

  // Compute the list of subgroup-relative offsets for sub-tensors or sub-memory
  // descriptors to be accessed, based on the layout information.
  ArrayRef<int64_t> wgShape = op.getDataShape();
  auto maybeDescOffsets =
      layout.computeDistributedCoords(rewriter, loc, sgId, wgShape);
  if (failed(maybeDescOffsets))
    return failure();

  // Compute the final global offsets for each accessed sub-tensor
  // or sub-memory descriptor.
  for (const auto &sgOffsets : *maybeDescOffsets) {
    SmallVector<OpFoldResult> newOffsets = xegpu::addWithRightAligned(
        rewriter, loc, getAsOpFoldResult(sgOffsets), origOffsets);
    offsetsList.push_back(std::move(newOffsets));
  }

  // callback(offsetsList);
  return success();
}

/// This pattern transforms the CreateNdDescOp to create a subgroup descriptor
/// from a workgroup descriptor. It replaces the offsets and sizes with
/// appropriate values for the subgroup.
/// It uses round-robin assignment to distribute the work to the subgroups.
/// Following create_nd_desc operation:
///    %tdesc = xegpu.create_nd_tdesc %src : memref<24x24xf32>
///       -> !xegpu.tensor_desc<24x24xf32, #xegpu.layout<sg_layout = [4, 4],
///           sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
/// is converted to 9 subgroup level operations based on the sg_layout &
/// sg_data:
///    %tdesc = xegpu.create_nd_tdesc %src : memref<24x24xf32> ->
///           !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2],
///           lane_data = [1, 1]>>
///
/// The sg_layout and sg_data attributes are dropped after the pass as they are
/// no longer needed.
///
/// 24x24 matrix distribution example:
/// sg_layout = [4, 4], sg_data = [2, 2]
/// Each 8x8 matrix within the 24x24 matrix is called a distribution unit.
/// dist_unit_shape = [8, 8] --> sg_layout[i] * sg_data[i]
///
/// +------------------------+
/// | 8x8 | 8x8 | 8x8 |      <- 3 tiles across
/// |-----+-----+-----|
/// | 8x8 | 8x8 | 8x8 |      <- 3 tiles down
/// |-----+-----+-----|
/// | 8x8 | 8x8 | 8x8 |
/// +------------------------+
///
/// Each 8x8 tile is further subdivided among subgroups:
/// +------------------------+
/// | 2x2 2x2 2x2 2x2 |  <- 4 subgroups across (each handles 2 columns)
/// | 2x2 2x2 2x2 2x2 |  <- 4 subgroups down (each handles 2 rows)
/// | 2x2 2x2 2x2 2x2 |
/// | 2x2 2x2 2x2 2x2 |
/// +------------------------+
///
/// Since the 24x24 matrix is divided into 8x8 distribution units, there will be
/// 9 distribution units (3x3) in total. Hence the 9 subgroup level operations.

/// The pass currently has entire distribution logic in the WgToSgCreateNdOp
/// pattern and all the other ops just follow.
/// TODO: Decouple the distribution logic from WgToSgCreateNdOp for all the
/// ops in the pass.
// This pattern transforms the CreateNdDescOp to create a
// subgroup descriptor from a workgroup descriptor.
struct WgToSgCreateNdOp : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    xegpu::TensorDescType tdescTy = op.getType();
    auto layout = dyn_cast<xegpu::LayoutAttr>(tdescTy.getLayout());
    if (!layout || !layout.isForWorkgroup())
      return failure();

    Type elemTy = tdescTy.getElementType();
    ArrayRef<int64_t> wgShape = tdescTy.getShape();

    SmallVector<int64_t> sgShape;
    int count;
    std::tie(sgShape, count) = getSgShapeAndCount(wgShape, layout);
    xegpu::TensorDescType newTdescTy =
        xegpu::TensorDescType::get(ctx, sgShape, elemTy, tdescTy.getEncoding(),
                                   layout.dropSgLayoutAndData());

    SmallVector<Value> newCreateNdOps(count);
    std::generate(newCreateNdOps.begin(), newCreateNdOps.end(), [&]() {
      return xegpu::CreateNdDescOp::create(rewriter, loc, newTdescTy,
                                           op.getSource(), op.getMixedSizes(),
                                           op.getMixedStrides());
    });

    rewriter.replaceOpWithMultiple(op, {newCreateNdOps});
    return success();
  }
};

/// This pattern transforms the LoadNdOp to load subgroup data.
struct WgToSgLoadNdOp : public OpConversionPattern<xegpu::LoadNdOp> {
  using OpConversionPattern<xegpu::LoadNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::LoadNdOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (failed(genOffsetsList(rewriter, op, offsetsList)))
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropSgLayoutAndData();
    SmallVector<Value> newOps;
    for (auto [tdesc, offsets] :
         llvm::zip(adaptor.getTensorDesc(), offsetsList)) {
      auto tdescTy = dyn_cast<xegpu::TensorDescType>(tdesc.getType());
      VectorType newResTy =
          VectorType::get(tdescTy.getShape(), tdescTy.getElementType());
      auto newOp = xegpu::LoadNdOp::create(
          rewriter, op.getLoc(), newResTy, tdesc, offsets,
          /*packed = */ nullptr, /*transpose = */ nullptr, op.getL1HintAttr(),
          op.getL2HintAttr(), op.getL3HintAttr(), layout);
      newOps.push_back(newOp);
    }
    rewriter.replaceOpWithMultiple(op, {newOps});

    return success();
  }
};

/// This pattern transforms the StoreNdOp to store subgroup data.
struct WgToSgStoreNdOp : public OpConversionPattern<xegpu::StoreNdOp> {
  using OpConversionPattern<xegpu::StoreNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::StoreNdOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (failed(genOffsetsList(rewriter, op, offsetsList)))
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropSgLayoutAndData();
    for (auto [v, tdesc, offsets] :
         llvm::zip(adaptor.getValue(), adaptor.getTensorDesc(), offsetsList)) {
      xegpu::StoreNdOp::create(rewriter, op.getLoc(), v, tdesc, offsets,
                               op.getL1HintAttr(), op.getL2HintAttr(),
                               op.getL3HintAttr(), layout);
    }
    rewriter.eraseOp(op);

    return success();
  }
};

/// This pattern transforms the PrefetchNdOp to prefetch subgroup data.
struct WgToSgPrefetchNdOp : public OpConversionPattern<xegpu::PrefetchNdOp> {
  using OpConversionPattern<xegpu::PrefetchNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::PrefetchNdOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (failed(genOffsetsList(rewriter, op, offsetsList)))
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropSgLayoutAndData();
    for (auto [tdesc, offsets] :
         llvm::zip(adaptor.getTensorDesc(), offsetsList)) {
      xegpu::PrefetchNdOp::create(rewriter, op.getLoc(), tdesc, offsets,
                                  op.getL1HintAttr(), op.getL2HintAttr(),
                                  op.getL3HintAttr(), layout);
    }
    rewriter.eraseOp(op);

    return success();
  }
};

/// This pattern transforms the DpasOp to work at subgroup level.
struct WgToSgDpasOp : public OpConversionPattern<xegpu::DpasOp> {
  using OpConversionPattern<xegpu::DpasOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::DpasOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType resultTy = op.getResult().getType();
    if (resultTy.getRank() < 2)
      return failure();

    auto layoutCd = op.getLayoutCdAttr();
    auto layoutA = op.getLayoutAAttr();
    auto layoutB = op.getLayoutBAttr();
    if (!layoutCd || !layoutA || !layoutB)
      return failure();
    size_t i = 0;
    SmallVector<Value> newDpasOps;
    for (auto aVec : adaptor.getLhs()) {
      for (auto bVec : adaptor.getRhs()) {

        llvm::SmallVector<Value> operands({aVec, bVec});
        Value tmpC;
        if (op.getAcc()) {
          tmpC = adaptor.getAcc()[i++];
          operands.push_back(tmpC);
        }

        ArrayRef<int64_t> aVecShape =
            cast<VectorType>(aVec.getType()).getShape();
        ArrayRef<int64_t> bVecShape =
            cast<VectorType>(bVec.getType()).getShape();
        // Build result shape: batch dims from A + [M, N] from last dims of
        // A and B.
        SmallVector<int64_t> resShape(aVecShape.drop_back(2));
        resShape.push_back(aVecShape[aVecShape.size() - 2]);
        resShape.push_back(bVecShape[bVecShape.size() - 1]);
        VectorType resTy = VectorType::get(resShape, resultTy.getElementType());
        auto newDpasOp = xegpu::DpasOp::create(rewriter, loc, resTy, operands);
        newDpasOp.setLayoutCdAttr(layoutCd.dropSgLayoutAndData());
        newDpasOp.setLayoutAAttr(layoutA.dropSgLayoutAndData());
        newDpasOp.setLayoutBAttr(layoutB.dropSgLayoutAndData());

        newDpasOps.push_back(newDpasOp);
      }
    }
    rewriter.replaceOpWithMultiple(op, {newDpasOps});
    return success();
  }
};

/// This pattern transforms the DpasMxOp to work at subgroup level.
struct WgToSgDpasMxOp : public OpConversionPattern<xegpu::DpasMxOp> {
  using OpConversionPattern<xegpu::DpasMxOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::DpasMxOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    VectorType resultTy = op.getResult().getType();

    if (resultTy.getRank() < 2)
      return failure();

    auto layoutCd = op.getLayoutCdAttr();
    auto layoutA = op.getLayoutAAttr();
    auto layoutB = op.getLayoutBAttr();
    auto layoutAScale = op.getLayoutAScaleAttr();
    auto layoutBScale = op.getLayoutBScaleAttr();

    if (!layoutCd || !layoutA || !layoutB || !layoutAScale || !layoutBScale)
      return failure();

    size_t index_c = 0;
    SmallVector<Value> newDpasMxOps;
    for (auto [index_a, aVec] : llvm::enumerate(adaptor.getA())) {
      for (auto [index_b, bVec] : llvm::enumerate(adaptor.getB())) {
        Value accVal = (op.getAcc()) ? adaptor.getAcc()[index_c++] : Value();
        Value scaleAVal =
            (op.getScaleA()) ? adaptor.getScaleA()[index_a] : Value();
        Value scaleBVal =
            (op.getScaleB()) ? adaptor.getScaleB()[index_b] : Value();

        ArrayRef<int64_t> aVecShape =
            cast<VectorType>(aVec.getType()).getShape();
        ArrayRef<int64_t> bVecShape =
            cast<VectorType>(bVec.getType()).getShape();
        // Build result shape: batch dims from A + [M, N]
        SmallVector<int64_t> resShape(aVecShape.drop_back(2));
        resShape.push_back(aVecShape[aVecShape.size() - 2]);
        resShape.push_back(bVecShape[bVecShape.size() - 1]);
        VectorType resTy = VectorType::get(resShape, resultTy.getElementType());
        auto newDpasMxOp = xegpu::DpasMxOp::create(
            rewriter, loc, resTy, aVec, bVec, accVal, scaleAVal, scaleBVal,
            layoutA.dropSgLayoutAndData(), layoutB.dropSgLayoutAndData(),
            layoutCd.dropSgLayoutAndData(), layoutAScale.dropSgLayoutAndData(),
            layoutBScale.dropSgLayoutAndData());

        newDpasMxOps.push_back(newDpasMxOp);
      }
    }
    rewriter.replaceOpWithMultiple(op, {newDpasMxOps});
    return success();
  }
};

/// This pattern transforms vector.broadcast ops to work at subgroup level.
struct WgToSgVectorBroadcastOp
    : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    VectorType resultType = op.getResult().getType();
    ArrayRef<int64_t> wgShape = resultType.getShape();

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(llvm::cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForWorkgroup())
      return failure();

    SmallVector<int64_t> sgShape;
    int count;
    std::tie(sgShape, count) = getSgShapeAndCount(wgShape, layout);
    VectorType newResultType =
        VectorType::get(sgShape, resultType.getElementType());

    SmallVector<Value> newBroadcastOps;
    auto distSource = adaptor.getOperands().front();
    int numDistributions = count / distSource.size();
    for (int i = 0; i < numDistributions; ++i) {
      for (auto operand : distSource) {
        auto newBroadcast = vector::BroadcastOp::create(rewriter, op.getLoc(),
                                                        newResultType, operand);

        newBroadcastOps.push_back(newBroadcast.getResult());
      }
    }
    rewriter.replaceOpWithMultiple(op, {newBroadcastOps});
    return success();
  }
};

// This pattern transforms elementwise ops to work at subgroup level.
struct WgToSgElementwiseOp : public ConversionPattern {
  WgToSgElementwiseOp(MLIRContext *ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match ops with elementwise trait and single result.
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();

    auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
    assert(resultType && "Expected result to be a VectorType");

    ArrayRef<int64_t> wgShape = resultType.getShape();

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(llvm::cast<OpResult>(op->getResult(0)));
    if (!layout || !layout.isForWorkgroup())
      return failure();

    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;

    size_t numVariants = operands.empty() ? 0 : operands.front().size();

    if (llvm::any_of(operands, [&](const ValueRange &operandVec) {
          return operandVec.size() != numVariants;
        }))
      return failure();

    SmallVector<Value> newResults;
    VectorType newResultType =
        VectorType::get(sgShape, resultType.getElementType());

    for (size_t i = 0; i < numVariants; ++i) {
      SmallVector<Value> opOperands;
      for (auto &operandVec : operands)
        opOperands.push_back(operandVec[i]);

      OperationState state(op->getLoc(), op->getName());
      state.addOperands(opOperands);
      state.addTypes(newResultType);
      state.addAttributes(op->getAttrs());
      Operation *newOp = rewriter.create(state);
      xegpu::removeLayoutAttrs(newOp);
      newResults.push_back(newOp->getResult(0));
    }

    rewriter.replaceOpWithMultiple(op, {newResults});
    return success();
  }
};

// clang-format off
// Pattern for lowering ConvertLayoutOp based on sg_layout and sg_data.
// If input_layout and target_layout have identical sg_layout and sg_data,
// the op is rewritten to a subgroup-level ConvertLayoutOp with these fields
// dropped. For example:
//   #a = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [16, 16]>
//   #b = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 16], inst_data = [8, 16]>
//   xegpu.convert_layout %1 <{input_layout = #a, target_layout = #b}> : vector<32x64xf32>
// becomes:
//   #a = #xegpu.layout<inst_data = [16, 16]>
//   #b = #xegpu.layout<inst_data = [8, 16]>
//   xegpu.convert_layout %1 <{input_layout = #a, target_layout = #b}> : vector<16x16xf32>
// (vector<16x16xf32> is determined by sg_data = [16, 16])
//
// If sg_layout or sg_data differ, SLM is used to redistribute data across subgroups.
// For example:
//   #a = #xegpu.layout<sg_layout = [1, 4], sg_data = [32, 16], inst_data = [16, 16]>
//   #b = #xegpu.layout<sg_layout = [2, 2], sg_data = [16, 32], inst_data = [8, 16]>
//   xegpu.convert_layout %1 <{input_layout = #a, target_layout = #b}> : vector<32x64xf32>
// is lowered to:
//   #a = #xegpu.layout<inst_data = [16, 16]>
//   #b = #xegpu.layout<inst_data = [8, 16]>
//   store_matrix %1, %slm <{layout_input_0 = #a}> : vector<32x16>, mem_desc<32x64xf32>
//   %d = load_matrix %slm <{layout_result_0 = #a}> : mem_desc<32x64xf32> -> vector<16x32xf32>
//   xegpu.convert_layout %d <{input_layout = #a, target_layout = #b}> : vector<16x32xf32>
// clang-format on
struct WgToSgConvertLayoutOp
    : public OpConversionPattern<xegpu::ConvertLayoutOp> {
  using OpConversionPattern<xegpu::ConvertLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::ConvertLayoutOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputLayout = op.getInputLayout();
    auto targetLayout = op.getTargetLayout();

    if (!inputLayout || !targetLayout || !inputLayout.isForWorkgroup() ||
        !targetLayout.isForWorkgroup())
      return rewriter.notifyMatchFailure(
          op, "Input and target layouts must have subgroup layout");

    Type resultType = op.getResult().getType();
    if (resultType.isIntOrFloat()) {
      rewriter.replaceOp(op, op.getSource());
      assert(!inputLayout.dropSgLayoutAndData() &&
             !targetLayout.dropSgLayoutAndData() &&
             "unexpected layout attributes for scalar type");
      return success();
    }

    ArrayRef<int64_t> wgShape = cast<VectorType>(resultType).getShape();
    SmallVector<int64_t> inputSgLayout =
        inputLayout.getEffectiveSgLayoutAsInt();
    SmallVector<int64_t> inputSgData = inputLayout.getEffectiveSgDataAsInt();
    SmallVector<int64_t> targetSgLayout =
        targetLayout.getEffectiveSgLayoutAsInt();
    SmallVector<int64_t> targetSgData = targetLayout.getEffectiveSgDataAsInt();

    // Fast path: if sg_layout and sg_data are identical, no SLM needed
    SmallVector<int64_t> wgShapeVec(wgShape.begin(), wgShape.end());
    if (inputLayout.isCompatibleWith(targetLayout, wgShapeVec,
                                     xegpu::LayoutKind::Subgroup)) {
      inputLayout = inputLayout.dropSgLayoutAndData();
      targetLayout = targetLayout.dropSgLayoutAndData();

      SmallVector<Value> newOps(adaptor.getSource());
      if (inputLayout && targetLayout) {
        for (auto [i, src] : llvm::enumerate(adaptor.getSource())) {
          auto newOp = xegpu::ConvertLayoutOp::create(
              rewriter, loc, src.getType(), src, inputLayout, targetLayout);
          newOps[i] = newOp;
        }
      }
      rewriter.replaceOpWithMultiple(op, {newOps});
      return success();
    }

    // SLM path: layouts differ, need cross-subgroup data redistribution
    Type elemTy = cast<VectorType>(op.getSource().getType()).getElementType();

    SmallVector<int64_t> slmShape = llvm::to_vector(wgShape);

    // Calculate SLM size requirements
    auto bitWidth = elemTy.getIntOrFloatBitWidth();
    auto bytesPerElement = bitWidth / 8;
    auto slmSize = computeProduct(slmShape) * bytesPerElement;

    // Allocate SLM
    auto slmTy = MemRefType::get({slmSize}, rewriter.getI8Type(), {}, 3);
    auto slm = memref::AllocaOp::create(rewriter, loc, slmTy);

    auto memDescType = xegpu::MemDescType::get(rewriter.getContext(), slmShape,
                                               elemTy, nullptr);
    auto memDesc =
        xegpu::CreateMemDescOp::create(rewriter, loc, memDescType, slm);

    auto sgId = gpu::SubgroupIdOp::create(rewriter, loc,
                                          rewriter.getIndexType(), nullptr);

    // STORE PHASE: Each subgroup stores in SLM using input layout
    auto storeCoords = inputLayout.computeDistributedCoords(
        rewriter, loc, sgId.getResult(), wgShape);
    if (failed(storeCoords))
      return failure();

    // Store to SLM
    for (auto [src, coords] : llvm::zip(adaptor.getSource(), *storeCoords)) {
      SmallVector<OpFoldResult> storeMatrixOffsets;
      for (Value coord : coords) {
        storeMatrixOffsets.push_back(coord);
      }
      xegpu::StoreMatrixOp::create(rewriter, loc, src, memDesc.getResult(),
                                   storeMatrixOffsets, nullptr /*layout*/);
    }

    gpu::BarrierOp::create(rewriter, loc);

    // LOAD PHASE: Each target subgroup loads from SLM using target layout
    auto loadCoords = targetLayout.computeDistributedCoords(
        rewriter, loc, sgId.getResult(), wgShape);
    if (failed(loadCoords))
      return failure();

    VectorType loadType = VectorType::get(targetSgData, elemTy);

    // Load vectors from SLM
    SmallVector<Value> finalResults;
    for (auto coords : *loadCoords) {
      SmallVector<OpFoldResult> loadMatrixOffsets;
      for (Value coord : coords) {
        loadMatrixOffsets.push_back(coord);
      }
      auto loadOp = xegpu::LoadMatrixOp::create(
          rewriter, loc, loadType, memDesc.getResult(), loadMatrixOffsets,
          targetLayout.dropSgLayoutAndData());

      finalResults.push_back(loadOp.getResult());
    }

    rewriter.replaceOpWithMultiple(op, {finalResults});
    return success();
  }
};

// This pattern distributes arith.constant op into subgroup-level constants
struct WgToSgArithConstantOp : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vecAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    auto vecType = dyn_cast<VectorType>(op.getType());
    if (!vecAttr || !vecType)
      return failure();

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForWorkgroup())
      return failure();

    ArrayRef<int64_t> wgShape = vecType.getShape();
    SmallVector<int64_t> sgShape;
    int count;
    std::tie(sgShape, count) = getSgShapeAndCount(wgShape, layout);

    auto newType = VectorType::get(sgShape, vecType.getElementType());
    Location loc = op.getLoc();
    auto eltType = vecType.getElementType();

    if (vecAttr.isSplat()) {
      // Splat: single value for all subgroups
      Attribute singleVal = vecAttr.getSplatValue<Attribute>();
      auto sgAttr = DenseElementsAttr::get(newType, singleVal);
      SmallVector<Value> newConstOps;
      for (int i = 0; i < count; ++i) {
        auto cstOp = arith::ConstantOp::create(rewriter, loc, newType, sgAttr);
        newConstOps.push_back(cstOp);
      }
      rewriter.replaceOpWithMultiple(op, {newConstOps});
      return success();
    } else if (sgShape == wgShape) { // if the entire vector is shared by all
                                     // subgroups, don't distribute
      auto newConstOp =
          arith::ConstantOp::create(rewriter, op.getLoc(), vecType, vecAttr);
      rewriter.replaceOp(op, newConstOp);
      return success();
    } else {
      // Non-splat constant
      // Only supports 1D & 2D
      // TODO: support other cases that require SLM access
      if (!eltType.isIndex())
        return rewriter.notifyMatchFailure(
            op, "Unsupported element type for non-splat constant op.");

      if (wgShape.size() > 2)
        return rewriter.notifyMatchFailure(
            op, "Only 1D & 2D vector constant supported");

      SmallVector<Attribute> values(vecAttr.getValues<Attribute>());
      int64_t rowStride = 0, colStride = 0;
      int64_t rows = wgShape.size() == 1 ? 1 : wgShape[0];
      int64_t cols = wgShape.size() == 1 ? wgShape[0] : wgShape[1];

      // Compute colStride and rowStride, and check for constant strides.
      if (cols > 1) {
        colStride = cast<IntegerAttr>(values[1]).getInt() -
                    cast<IntegerAttr>(values[0]).getInt();
      }
      if (rows > 1) {
        rowStride = cast<IntegerAttr>(values[cols]).getInt() -
                    cast<IntegerAttr>(values[0]).getInt();
      }

      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          int64_t idx = r * cols + c;
          // Check column stride
          if (c > 0 && cols > 1) {
            int64_t prevIdx = r * cols + (c - 1);
            int64_t diff = cast<IntegerAttr>(values[idx]).getInt() -
                           cast<IntegerAttr>(values[prevIdx]).getInt();
            if (diff != colStride)
              return rewriter.notifyMatchFailure(
                  op, "Non-constant column stride in constant op.");
          }
          // Check row stride
          if (r > 0 && rows > 1) {
            int64_t prevIdx = (r - 1) * cols + c;
            int64_t diff = cast<IntegerAttr>(values[idx]).getInt() -
                           cast<IntegerAttr>(values[prevIdx]).getInt();
            if (diff != rowStride)
              return rewriter.notifyMatchFailure(
                  op, "Non-constant row stride in constant op.");
          }
        }
      }

      // Create a constant for the base tile.
      // For 2D case, extract the top-left sgShape[0] x sgShape[1] submatrix.
      // For 1D case, extract the first sgShape[0] elements.
      SmallVector<Attribute> baseTileValues;
      int baseTileCols = sgShape[sgShape.size() - 1];
      int64_t baseTileRows = sgShape.size() == 1 ? 1 : sgShape[0];
      for (int64_t r = 0; r < baseTileRows; ++r) {
        for (int64_t c = 0; c < baseTileCols; ++c) {
          baseTileValues.push_back(values[r * cols + c]);
        }
      }

      auto tileAttr = DenseElementsAttr::get(VectorType::get(sgShape, eltType),
                                             baseTileValues);
      auto baseConstVec = arith::ConstantOp::create(rewriter, loc, tileAttr);

      // Get subgroup id
      Value sgId =
          gpu::SubgroupIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);
      auto sgOffsets =
          layout.computeDistributedCoords(rewriter, loc, sgId, wgShape);
      if (failed(sgOffsets))
        return failure();

      SmallVector<Value, 2> strideConsts;
      strideConsts.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, colStride));
      if (rows > 1)
        strideConsts.insert(
            strideConsts.begin(),
            arith::ConstantIndexOp::create(rewriter, loc, rowStride));

      SmallVector<Value> newConstOps;
      for (auto offsets : *sgOffsets) {
        // Multiply offset with stride, broadcast it and add to baseConstVec
        Value mulOffset = arith::ConstantIndexOp::create(rewriter, loc, 0);
        for (size_t i = 0; i < strideConsts.size(); ++i) {
          Value mul =
              arith::MulIOp::create(rewriter, loc, rewriter.getIndexType(),
                                    offsets[i], strideConsts[i]);
          mulOffset = arith::AddIOp::create(
              rewriter, loc, rewriter.getIndexType(), mulOffset, mul);
        }
        // Broadcast to baseConstVec size
        auto bcastOffset = vector::BroadcastOp::create(
            rewriter, loc, baseConstVec.getType(), mulOffset);
        auto finalConst =
            arith::AddIOp::create(rewriter, loc, baseConstVec, bcastOffset);
        newConstOps.push_back(finalConst);
      }
      rewriter.replaceOpWithMultiple(op, {newConstOps});
      return success();
    }
  }
};

// This pattern transforms the LoadGatherOp with explicit offsets to load
// subgroup data
struct WgToSgLoadGatherOp : public OpConversionPattern<xegpu::LoadGatherOp> {
  using OpConversionPattern<xegpu::LoadGatherOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::LoadGatherOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    VectorType resultType = dyn_cast<VectorType>(op.getResult().getType());
    if (!resultType)
      return failure();
    ArrayRef<int64_t> wgShape = resultType.getShape();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();

    if (!layout || !layout.isForWorkgroup())
      return failure();

    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;

    // The offsets need to be distributed
    auto offsetsVecType =
        dyn_cast<VectorType>(adaptor.getOffsets().front().getType());
    auto maskVecType =
        dyn_cast<VectorType>(adaptor.getMask().front().getType());
    if (!offsetsVecType || !maskVecType ||
        offsetsVecType.getShape() != maskVecType.getShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "offsets have not been distributed");
    }

    SmallVector<Value> newLoadOps;
    auto chunkSizeAttr =
        rewriter.getI64IntegerAttr(op.getChunkSize().value_or(1));
    VectorType newTy = VectorType::get(sgShape, resultType.getElementType());
    for (auto [offsets, mask] :
         llvm::zip(adaptor.getOffsets(), adaptor.getMask())) {
      auto newLayout = layout.dropSgLayoutAndData();
      auto newLoadOp = xegpu::LoadGatherOp::create(
          rewriter, loc, newTy, op.getSource(), offsets, mask, chunkSizeAttr,
          op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr(), newLayout,
          /*contiguity=*/nullptr);
      newLoadOps.push_back(newLoadOp);
    }
    rewriter.replaceOpWithMultiple(op, {newLoadOps});
    return success();
  }
};

// This pattern transforms the StoreScatterOp with explicit offsets to store
// subgroup data
struct WgToSgStoreScatterOp
    : public OpConversionPattern<xegpu::StoreScatterOp> {
  using OpConversionPattern<xegpu::StoreScatterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::StoreScatterOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    VectorType valueType = dyn_cast<VectorType>(op.getValue().getType());
    if (!valueType)
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();

    if (!layout || !layout.isForWorkgroup())
      return failure();

    // The offsets need to be distributed
    auto offsetsVecType =
        dyn_cast<VectorType>(adaptor.getOffsets().front().getType());
    auto maskVecType =
        dyn_cast<VectorType>(adaptor.getMask().front().getType());
    if (!offsetsVecType || !maskVecType ||
        offsetsVecType.getShape() != maskVecType.getShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "offsets have not been distributed");
    }

    auto chunkSizeOpt = op.getChunkSize();
    int64_t chunkSize = chunkSizeOpt ? static_cast<int64_t>(*chunkSizeOpt) : 1;
    auto chunkSizeAttr = rewriter.getI64IntegerAttr(chunkSize);
    for (auto [val, offs, mask] : llvm::zip(
             adaptor.getValue(), adaptor.getOffsets(), adaptor.getMask())) {
      xegpu::StoreScatterOp::create(rewriter, loc, val, op.getDest(), offs,
                                    mask, chunkSizeAttr, op.getL1HintAttr(),
                                    op.getL2HintAttr(), op.getL3HintAttr(),
                                    layout.dropSgLayoutAndData(),
                                    /*contiguity=*/nullptr);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Returns true if `memDesc` was produced by a create_mem_desc whose source
// memref lives in private memory (i.e. it was privatized by the pre-phase).
static bool isPrivateMemDesc(Value memDesc) {
  auto createOp = memDesc.getDefiningOp<xegpu::CreateMemDescOp>();
  if (!createOp)
    return false;
  auto srcTy = dyn_cast<MemRefType>(createOp.getSource().getType());
  return srcTy && xegpu::XeGPUDialect::isPrivateMemory(srcTy);
}

// Computes the per-subgroup local offset lists into a privatized (shrunk)
// mem_desc. Because every subgroup owns a private copy sized to the distributed
// shape, offsets are subgroup-id independent: round `r` maps to `r * sg_data`.
static SmallVector<SmallVector<OpFoldResult>>
genPrivateOffsetsList(ConversionPatternRewriter &rewriter, Location loc,
                      ArrayRef<int64_t> wgShape,
                      xegpu::DistributeLayoutAttr layout) {
  SmallVector<int64_t> sgData = layout.getEffectiveSgDataAsInt();
  SmallVector<int64_t> sgLayout = layout.getEffectiveSgLayoutAsInt();

  // The private buffer holds the per-subgroup distributed shape, which spans all
  // distribution rounds stacked along each dim (roundShape[d] = wgShape[d] /
  // sg_layout[d] = sg_data[d] * rounds). This matches the shape allocated by
  // WgToSgCreateMemDescOp, so stepping by sg_data yields in-bounds local offsets.
  SmallVector<int64_t> roundShape(wgShape.size());
  for (auto [i, dim] : llvm::enumerate(wgShape))
    roundShape[i] = dim / sgLayout[i];

  SmallVector<SmallVector<OpFoldResult>> offsetsList;
  for (SmallVector<int64_t> off : StaticTileOffsetRange(roundShape, sgData)) {
    SmallVector<OpFoldResult> offsets;
    for (int64_t o : off)
      offsets.push_back(rewriter.getIndexAttr(o));
    offsetsList.push_back(std::move(offsets));
  }
  return offsetsList;
}

struct WgToSgLoadMatrixOp : public OpConversionPattern<xegpu::LoadMatrixOp> {
  using OpConversionPattern<xegpu::LoadMatrixOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::LoadMatrixOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ArrayRef<int64_t> wgShape = op.getDataShape();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getRes().getType());
    assert(valueTy && "the value type must be vector type!");
    Type elemTy = valueTy.getElementType();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();

    // Privatized buffers are shrunk to the per-subgroup size and indexed with
    // local, subgroup-id-free offsets; SLM buffers keep the sg-relative offsets.
    bool isPrivate = isPrivateMemDesc(op.getMemDesc());
    auto memDesc = cast<TypedValue<xegpu::MemDescType>>(
        isPrivate ? adaptor.getMemDesc()[0] : op.getMemDesc());
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (isPrivate) {
      offsetsList = genPrivateOffsetsList(rewriter, op.getLoc(), wgShape,
                                          layout);
    } else if (failed(genOffsetsList(rewriter, op, offsetsList))) {
      return failure();
    }

    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;
    VectorType newResTy = VectorType::get(sgShape, elemTy);
    SmallVector<Value> newOps;
    for (auto offsets : offsetsList) {
      auto newOp = xegpu::LoadMatrixOp::create(rewriter, op.getLoc(), newResTy,
                                               memDesc, offsets,
                                               layout.dropSgLayoutAndData());
      newOps.push_back(newOp);
    }
    rewriter.replaceOpWithMultiple(op, {newOps});

    return success();
  }
};

struct WgToSgStoreMatrixOp : public OpConversionPattern<xegpu::StoreMatrixOp> {
  using OpConversionPattern<xegpu::StoreMatrixOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::StoreMatrixOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();

    bool isPrivate = isPrivateMemDesc(op.getMemDesc());
    auto memDesc = cast<TypedValue<xegpu::MemDescType>>(
        isPrivate ? adaptor.getMemDesc()[0] : op.getMemDesc());
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (isPrivate) {
      offsetsList = genPrivateOffsetsList(rewriter, op.getLoc(),
                                          op.getDataShape(), layout);
    } else if (failed(genOffsetsList(rewriter, op, offsetsList))) {
      return failure();
    }

    for (auto [v, offsets] : llvm::zip(adaptor.getData(), offsetsList))
      xegpu::StoreMatrixOp::create(rewriter, op.getLoc(), v, memDesc, offsets,
                                   layout.dropSgLayoutAndData());
    rewriter.eraseOp(op);
    return success();
  }
};

// Shrinks a privatized create_mem_desc (marked by the pre-phase) to the
// per-subgroup size: it allocates a smaller private buffer and builds a
// mem_desc of the distributed shape, so each subgroup owns a private copy.
struct WgToSgCreateMemDescOp
    : public OpConversionPattern<xegpu::CreateMemDescOp> {
  using OpConversionPattern<xegpu::CreateMemDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateMemDescOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto layout = op->getAttrOfType<xegpu::DistributeLayoutAttr>(
        kPrivatizeLayoutAttrName);
    if (!layout)
      return failure();

    Location loc = op.getLoc();
    xegpu::MemDescType mdescTy = op.getMemDesc().getType();
    Type elemTy = mdescTy.getElementType();

    // Per-subgroup distributed shape: the workgroup shape divided by sg_layout,
    // holding every distribution round this subgroup owns (sg_data * rounds).
    auto distShape =
        layout.computeDistributedShape(llvm::to_vector(mdescTy.getShape()));
    if (failed(distShape))
      return failure();
    SmallVector<int64_t> sgShape = *distShape;

    // Allocate a private buffer just large enough for one subgroup's copy.
    auto bytesPerElement = elemTy.getIntOrFloatBitWidth() / 8;
    auto slmSize = computeProduct(sgShape) * bytesPerElement;
    auto memrefTy =
        MemRefType::get({slmSize}, rewriter.getI8Type(), {}, /*space=*/4);
    auto buffer = memref::AllocaOp::create(rewriter, loc, memrefTy);

    auto newMdescTy = xegpu::MemDescType::get(rewriter.getContext(), sgShape,
                                              elemTy, mdescTy.getMemLayout());
    auto newOp =
        xegpu::CreateMemDescOp::create(rewriter, loc, newMdescTy, buffer);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// This pattern distributes the vector.step ops to work at subgroup level
struct WgToSgVectorStepOp : public OpConversionPattern<vector::StepOp> {
  using OpConversionPattern<vector::StepOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::StepOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForWorkgroup())
      return failure();

    Location loc = op.getLoc();
    VectorType type = op.getResult().getType();
    auto wgShape = type.getShape();
    std::optional<SmallVector<int64_t>> sgShape =
        getSgShapeAndCount(wgShape, layout).first;
    if (!sgShape)
      return failure();

    Value sgId =
        gpu::SubgroupIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);
    auto sgOffsets =
        layout.computeDistributedCoords(rewriter, loc, sgId, wgShape);
    if (failed(sgOffsets))
      return failure();

    VectorType newTy = type.cloneWith(*sgShape, type.getElementType());
    auto steps = vector::StepOp::create(rewriter, loc, newTy);
    SmallVector<Value> newOps;
    for (auto offsets : *sgOffsets) {
      // Broadcast the offset scalar to a vector & add to the base steps
      auto bcastOffset =
          vector::BroadcastOp::create(rewriter, loc, newTy, offsets[0]);
      auto finalSteps =
          arith::AddIOp::create(rewriter, loc, steps, bcastOffset);
      newOps.push_back(finalSteps);
    }

    rewriter.replaceOpWithMultiple(op, {newOps});
    return success();
  }
};

// This pattern transforms vector.shape_cast ops to work at subgroup level.
struct WgToSgVectorShapeCastOp
    : public OpConversionPattern<vector::ShapeCastOp> {
  using OpConversionPattern<vector::ShapeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ShapeCastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    VectorType resultType = dyn_cast<VectorType>(op.getResult().getType());
    if (!resultType)
      return failure();

    ArrayRef<int64_t> wgShape = resultType.getShape();
    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForWorkgroup())
      return failure();

    // Check that srcShape and destShape, if they differ, only differ by
    // expand of unit dimensions.
    auto srcType = dyn_cast<VectorType>(op.getSource().getType());
    if (!srcType)
      return failure();

    ArrayRef<int64_t> srcShape = srcType.getShape();

    xegpu::DistributeLayoutAttr layoutToDistribute = layout;
    SmallVector<int64_t> expandedUnitDims;
    if (xegpu::matchUnitDimExpansion(srcShape, wgShape, expandedUnitDims)) {
      xegpu::DistributeLayoutAttr sourceLayout =
          xegpu::getTemporaryLayout(op->getOpOperand(0));

      if (!sourceLayout.isSliceOf(layout))
        return rewriter.notifyMatchFailure(
            op, "The ShapeCast op only expands dimensions, the input layout "
                "must be a slice of the result layout.");

      assert(layoutToDistribute.isEqualTo(
                 layoutToDistribute.setUnitDimData(expandedUnitDims)) &&
             "The sg_data for unit dimensions should be set as 1");
    }

    SmallVector<int64_t> sgShape =
        getSgShapeAndCount(wgShape, layoutToDistribute).first;
    VectorType newResultType =
        VectorType::get(sgShape, resultType.getElementType());

    SmallVector<Value> newShapeCastOps;
    for (auto src : adaptor.getSource()) {
      auto newShapeCast = vector::ShapeCastOp::create(rewriter, op.getLoc(),
                                                      newResultType, src);
      newShapeCastOps.push_back(newShapeCast.getResult());
    }

    rewriter.replaceOpWithMultiple(op, {newShapeCastOps});
    return success();
  }
};

/// This pattern transforms vector.multi_dim_reduction operations from
/// workgroup-level to subgroup-level execution with support for multiple
/// reduction dimensions.
///
/// Steps include:
/// 1. LOCAL REDUCTION :
///    - Each subgroup performs local reduction on its data slice
///    - Uses ZERO accumulator to avoid double-counting during cross-subgroup
///    phase
///
/// 2. CROSS-SUBGROUP :
///    - Determines if cross-subgroup reduction is needed (when sg_layout > 1 in
///      reduction dims & sgData[reduction dims] < wgData[reduction dims])
///    - If not needed, adds original accumulator and returns local results
///
/// 3. SHARED LOCAL MEMORY (SLM) PHASE (when cross-subgroup reduction needed):
///    a) SLM Layout Design:
///       - Rows: subgroups participating in reduction (product of sg_layout in
///       reduction dims)
///       - Cols: total result elements across non-reduction dimensions
///
///    b) Store Phase:
///       - Each subgroup stores its local reduction result to SLM
///       - Row offset: linearized index of subgroup in reduction dimensions
///       - Col offset: linearized index of subgroup in non-reduction dimensions
///
///    c) Load and Final Reduction Phase:
///       - Each subgroup loads a column of data (all reduction participants for
///       its position)
///       - Performs final reduction along the loaded dimension
///       - Adds original accumulator to get final result
///
struct WgToSgMultiDimReductionOp
    : public OpConversionPattern<vector::MultiDimReductionOp> {
  using OpConversionPattern<vector::MultiDimReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MultiDimReductionOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    VectorType srcType = op.getSourceVectorType();
    Type resultTy = op.getResult().getType();
    VectorType dstVecType = dyn_cast<VectorType>(resultTy);
    bool isScalarResult = !dstVecType;

    auto originalSrcShape = srcType.getShape();
    Type elemTy = srcType.getElementType();

    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForWorkgroup())
      return failure();

    auto reductionDims = llvm::to_vector(op.getReductionDims());

    // Get sg_layout and sg_data from the parent layout
    SmallVector<int64_t> sgLayout;
    SmallVector<int64_t> sgData;
    xegpu::DistributeLayoutAttr parentLayout;
    if (auto sliceAttr = dyn_cast<xegpu::SliceAttr>(layout)) {
      parentLayout = sliceAttr.getParent();
      sgLayout = parentLayout.getEffectiveSgLayoutAsInt();
      sgData = parentLayout.getEffectiveSgDataAsInt();
    } else
      return rewriter.notifyMatchFailure(
          op, "Reduction should have SliceAttr layout");

    // Step 1: perform local subgroup reductions with neutral accumulator
    SmallVector<Value> localReductions;
    auto sgSrcs = adaptor.getSource();
    auto sgSrcType = dyn_cast<VectorType>(sgSrcs.front().getType());
    SmallVector<int64_t> sgSrcShape(sgSrcType.getShape().begin(),
                                    sgSrcType.getShape().end());

    // Determine the SG-level destination type.
    // For scalar results (all dims reduced), the sg result is also scalar.
    // For vector results, compute the sg destination shape from layout.
    Type sgDstType;
    if (dstVecType) {
      auto originalDstShape = dstVecType.getShape();
      SmallVector<int64_t> sgDstShape =
          getSgShapeAndCount(originalDstShape, layout).first;
      sgDstType = VectorType::get(sgDstShape, elemTy);
    } else {
      sgDstType = elemTy;
    }

    for (auto sgSrc : sgSrcs) {
      // Create neutral accumulator for local reduction
      Value neutralLocalAcc = xegpu::createReductionNeutralValue(
          rewriter, loc, sgDstType, op.getKind());
      // Local reduction with neutral accumulator
      auto localReduce = vector::MultiDimReductionOp::create(
          rewriter, loc, sgDstType, op.getKind(), sgSrc, neutralLocalAcc,
          reductionDims);
      localReductions.push_back(localReduce.getResult());
    }

    // Check if cross-subgroup reduction is needed for any reduction dimension
    SmallVector<int64_t> crossSgReductionDims;
    for (int64_t reductionDim : reductionDims) {
      bool needsCrossSubgroupReduction =
          (sgLayout[reductionDim] > 1) &&
          (sgData[reductionDim] < originalSrcShape[reductionDim]);

      if (needsCrossSubgroupReduction) {
        crossSgReductionDims.push_back(reductionDim);
      }
    }

    // If no cross-subgroup reduction needed, add accumulator and return
    if (crossSgReductionDims.empty()) {
      SmallVector<Value> results;
      for (auto localResult : localReductions) {
        auto finalResult = vector::makeArithReduction(
            rewriter, loc, op.getKind(), localResult, adaptor.getAcc()[0]);
        results.push_back(finalResult);
      }
      rewriter.replaceOpWithMultiple(op, {results});
      return success();
    }

    // Step 2: cross-subgroup reduction using SLM - allocating slm memory
    auto slmStoreDataShape = sgSrcShape;
    for (int64_t dim : reductionDims)
      slmStoreDataShape[dim] = 1;
    VectorType slmStoreDataType = VectorType::get(slmStoreDataShape, elemTy);
    SmallVector<Value> slmStoreData;
    for (auto localResult : localReductions) {
      if (isScalarResult) {
        // Scalar result: broadcast scalar to vector<1x...x1> for SLM store
        slmStoreData.push_back(vector::BroadcastOp::create(
            rewriter, loc, slmStoreDataType, localResult));
      } else {
        slmStoreData.push_back(vector::ShapeCastOp::create(
            rewriter, loc, slmStoreDataType, localResult));
      }
    }
    // for reduction dimension, SLM stores partial results from each subgroup
    SmallVector<int64_t> slmShape(originalSrcShape.begin(),
                                  originalSrcShape.end());
    SmallVector<int> slmSgData(sgData.begin(), sgData.end());
    SmallVector<int> slmSgLayout(sgLayout.begin(), sgLayout.end());
    for (int dim : reductionDims) {
      slmShape[dim] = sgLayout[dim];
      slmSgData[dim] = 1;
    }
    xegpu::LayoutAttr slmStoreLayout =
        xegpu::LayoutAttr::get(rewriter.getContext(), slmSgLayout, slmSgData);

    // Allocate SLM
    auto bitWidth = elemTy.getIntOrFloatBitWidth();
    auto bytesPerElement = bitWidth / 8;
    auto slmSize = computeProduct(slmShape) * bytesPerElement;
    auto slmTy = MemRefType::get({slmSize}, rewriter.getI8Type(), {}, 3);
    auto slm = memref::AllocaOp::create(rewriter, loc, slmTy);

    auto memDescType = xegpu::MemDescType::get(rewriter.getContext(), slmShape,
                                               elemTy, nullptr);
    auto memDesc =
        xegpu::CreateMemDescOp::create(rewriter, loc, memDescType, slm);

    // Step 3: Store local results to SLM
    auto sgId = gpu::SubgroupIdOp::create(rewriter, loc,
                                          rewriter.getIndexType(), nullptr);

    auto slmStoreCoords =
        slmStoreLayout.computeDistributedCoords(rewriter, loc, sgId, slmShape);
    if (failed(slmStoreCoords))
      return failure();
    for (auto [data, coord] : llvm::zip(slmStoreData, *slmStoreCoords)) {
      SmallVector<OpFoldResult> coordOfr(coord.begin(), coord.end());
      xegpu::StoreMatrixOp::create(rewriter, loc, data, memDesc.getResult(),
                                   coordOfr,
                                   /*layout=*/nullptr);
    }

    gpu::BarrierOp::create(rewriter, loc);

    // Step 4: Load from SLM for final reduction
    SmallVector<int64_t> slmLoadDataShape(sgSrcShape.begin(), sgSrcShape.end());
    for (int64_t dim : reductionDims) {
      slmLoadDataShape[dim] = slmShape[dim];
      slmSgData[dim] = slmShape[dim];
    }
    xegpu::LayoutAttr slmLoadLayout =
        xegpu::LayoutAttr::get(rewriter.getContext(), slmSgLayout, slmSgData);
    auto slmLoadCoords =
        slmLoadLayout.computeDistributedCoords(rewriter, loc, sgId, slmShape);
    if (failed(slmLoadCoords))
      return failure();

    VectorType slmLoadType = VectorType::get(slmLoadDataShape, elemTy);
    SmallVector<Value> slmLoadData;
    for (auto coord : *slmLoadCoords) {
      SmallVector<OpFoldResult> coordOfr(coord.begin(), coord.end());
      slmLoadData.push_back(xegpu::LoadMatrixOp::create(
          rewriter, loc, slmLoadType, memDesc.getResult(), coordOfr,
          /*layout=*/nullptr));
    }

    // Step 5: Perform final reduction with neutral accumulator and add the
    // original accumulator at the end
    Value neutralFinalAcc = xegpu::createReductionNeutralValue(
        rewriter, loc, sgDstType, op.getKind());

    SmallVector<Value> finalResults;
    for (size_t i = 0; i < slmLoadData.size(); ++i) {
      auto loaded = slmLoadData[i];
      auto finalReduce = vector::MultiDimReductionOp::create(
          rewriter, loc, sgDstType, op.getKind(), loaded, neutralFinalAcc,
          reductionDims);
      finalResults.push_back(vector::makeArithReduction(
          rewriter, loc, op.getKind(), finalReduce.getResult(),
          adaptor.getAcc()[i]));
    }
    rewriter.replaceOpWithMultiple(op, {finalResults});
    return success();
  }
};

// This pattern transforms vector.transpose ops to work at subgroup level.
struct WgToSgVectorTransposeOp
    : public OpConversionPattern<vector::TransposeOp> {
  using OpConversionPattern<vector::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransposeOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();

    ArrayRef<int64_t> wgShape = resultType.getShape();
    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForWorkgroup())
      return failure();
    xegpu::DistributeLayoutAttr sourceLayout =
        xegpu::getTemporaryLayout(op->getOpOperand(0));
    if (!sourceLayout || !sourceLayout.isForWorkgroup())
      return failure();

    SmallVector<int64_t> sourceSgLayout =
        sourceLayout.getEffectiveSgLayoutAsInt();
    SmallVector<int64_t> resultSgLayout = layout.getEffectiveSgLayoutAsInt();

    ArrayRef<int64_t> permutation = op.getPermutation();
    size_t permutationSize = permutation.size();
    if (sourceSgLayout.size() != permutationSize ||
        resultSgLayout.size() != permutationSize) {
      return rewriter.notifyMatchFailure(
          op, "Layouts and permutation must have the same rank");
    }

    // Check that sgLayout, sgData & order are properly transposed for source
    // and result
    if (!layout.isTransposeOf(sourceLayout, permutation,
                              xegpu::LayoutKind::Subgroup))
      return rewriter.notifyMatchFailure(
          op, "Result layout is not a valid transpose of source layout "
              "according to permutation");

    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;
    VectorType newResultType =
        VectorType::get(sgShape, resultType.getElementType());

    SmallVector<Value> newTransposeOps;
    for (auto src : adaptor.getVector()) {
      auto newTranspose = vector::TransposeOp::create(
          rewriter, op.getLoc(), newResultType, src, permutation);
      newTransposeOps.push_back(newTranspose.getResult());
    }
    rewriter.replaceOpWithMultiple(op, {newTransposeOps});
    return success();
  }
};

// Distribute vector mask ops to work at subgroup level.
template <typename MaskOpType>
struct WgToSgVectorMaskOp : public OpConversionPattern<MaskOpType> {
  using OpConversionPattern<MaskOpType>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MaskOpType op,
      typename OpConversionPattern<MaskOpType>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForWorkgroup())
      return failure();

    Location loc = op.getLoc();
    VectorType type = op.getResult().getType();
    auto wgShape = type.getShape();

    SmallVector<Value> wgMaskDimSizes;
    if constexpr (std::is_same_v<MaskOpType, vector::ConstantMaskOp>) {
      for (int64_t maskSize : op.getMaskDimSizes()) {
        wgMaskDimSizes.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, maskSize));
      }
    } else if constexpr (std::is_same_v<MaskOpType, vector::CreateMaskOp>) {
      wgMaskDimSizes = llvm::to_vector(op.getOperands());
    }

    Value sgId =
        gpu::SubgroupIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);
    auto sgOffsets =
        layout.computeDistributedCoords(rewriter, loc, sgId, wgShape);
    if (failed(sgOffsets))
      return failure();

    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;
    VectorType resultType = VectorType::get(sgShape, type.getElementType());

    // In each dimension, each subgroup computes its local mask size as:
    // min(max(wgMaskDimSize[d] - offset[d], 0), sgDimSize[d])
    SmallVector<Value> newCreateMaskOps;
    for (auto offsetSet : *sgOffsets) {
      SmallVector<Value> maskOperands;

      for (auto [i, wgMaskDimSize] : llvm::enumerate(wgMaskDimSizes)) {
        Value dimSizeVal =
            arith::ConstantIndexOp::create(rewriter, loc, sgShape[i]);
        Value offset = offsetSet[i];
        Value adjustedMaskSize =
            arith::SubIOp::create(rewriter, loc, wgMaskDimSize, offset);
        Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value nonNegative =
            arith::MaxSIOp::create(rewriter, loc, adjustedMaskSize, zero);
        Value sgMaskSize =
            arith::MinSIOp::create(rewriter, loc, nonNegative, dimSizeVal);
        maskOperands.push_back(sgMaskSize);
      }

      auto newCreateMaskOp =
          vector::CreateMaskOp::create(rewriter, loc, resultType, maskOperands);
      newCreateMaskOps.push_back(newCreateMaskOp.getResult());
    }

    rewriter.replaceOpWithMultiple(op, {newCreateMaskOps});
    return success();
  }
};

using WgToSgVectorConstantMaskOp = WgToSgVectorMaskOp<vector::ConstantMaskOp>;
using WgToSgVectorCreateMaskOp = WgToSgVectorMaskOp<vector::CreateMaskOp>;

// This pattern transforms vector.bitcast ops to work at subgroup level.
struct WgToSgVectorBitCastOp : public OpConversionPattern<vector::BitCastOp> {
  using OpConversionPattern<vector::BitCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BitCastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();

    ArrayRef<int64_t> wgShape = resultType.getShape();
    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForWorkgroup())
      return failure();

    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;
    VectorType newResultType =
        VectorType::get(sgShape, resultType.getElementType());

    SmallVector<Value> newBitCastOps;
    for (auto src : adaptor.getSource()) {
      auto newBitCast =
          vector::BitCastOp::create(rewriter, op.getLoc(), newResultType, src);
      newBitCastOps.push_back(newBitCast.getResult());
    }

    rewriter.replaceOpWithMultiple(op, {newBitCastOps});
    return success();
  }
};

// This pattern transforms vector.interleave ops to work at subgroup level.
struct WgToSgVectorInterleaveOp
    : public OpConversionPattern<vector::InterleaveOp> {
  using OpConversionPattern<vector::InterleaveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InterleaveOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();

    ArrayRef<int64_t> wgShape = resultType.getShape();
    xegpu::DistributeLayoutAttr layout =
        xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
    if (!layout || !layout.isForWorkgroup())
      return failure();

    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;
    VectorType newResultType =
        VectorType::get(sgShape, resultType.getElementType());

    SmallVector<Value> newInterleaveOps;
    // Interleave operates pairwise: each lhs value is interleaved with
    // corresponding rhs value
    for (auto [lhs, rhs] : llvm::zip(adaptor.getLhs(), adaptor.getRhs())) {
      auto newInterleave = vector::InterleaveOp::create(
          rewriter, op.getLoc(), newResultType, lhs, rhs);
      newInterleaveOps.push_back(newInterleave.getResult());
    }

    rewriter.replaceOpWithMultiple(op, {newInterleaveOps});
    return success();
  }
};

// This pattern transforms vector.deinterleave ops to work at subgroup level.
struct WgToSgVectorDeinterleaveOp
    : public OpConversionPattern<vector::DeinterleaveOp> {
  using OpConversionPattern<vector::DeinterleaveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::DeinterleaveOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> newRes1Ops;
    SmallVector<Value> newRes2Ops;

    for (auto src : adaptor.getSource()) {
      auto newDeinterleave =
          vector::DeinterleaveOp::create(rewriter, op.getLoc(), src);
      newRes1Ops.push_back(newDeinterleave.getRes1());
      newRes2Ops.push_back(newDeinterleave.getRes2());
    }

    SmallVector<SmallVector<Value>> results = {newRes1Ops, newRes2Ops};
    rewriter.replaceOpWithMultiple(op, results);
    return success();
  }
};

} // namespace

namespace mlir {
namespace xegpu {
void populateXeGPUWgToSgDistributeTypeConversions(TypeConverter &converter,
                                                  Operation *topLevelOp) {
  // Pass through all types by default.
  converter.addConversion([](Type type) -> Type { return type; });

  // For TensorDescType, convert WG-level tensor descs to N SG-level descs.
  converter.addConversion(
      [](xegpu::TensorDescType type,
         SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        xegpu::DistributeLayoutAttr layout = type.getLayoutAttr();
        if (!layout || !layout.isForWorkgroup())
          return std::nullopt;

        Type elemTy = type.getElementType();
        ArrayRef<int64_t> shape = type.getShape();

        int count;
        SmallVector<int64_t> subShape;
        std::tie(subShape, count) = getSgShapeAndCount(shape, layout);

        layout = layout.dropSgLayoutAndData();

        auto newTy = xegpu::TensorDescType::get(
            type.getContext(), subShape, elemTy, type.getEncoding(), layout);
        result.append(count, newTy);
        return success();
      });

  // Context-aware VectorType conversion based on sg_layout/sg_data
  // (1:1 shape-changing or 1:N).
  auto getSubShapeAndCount = [](VectorType vecTy,
                                xegpu::DistributeLayoutAttr layout)
      -> std::pair<SmallVector<int64_t>, int> {
    if (!layout.isForWorkgroup())
      return {{}, 0};
    return getSgShapeAndCount(vecTy.getShape(), layout);
  };
  auto loopArgTypes =
      xegpu::precomputeLoopBlockArgTypes(topLevelOp, getSubShapeAndCount);
  xegpu::addVectorTypeConversion(converter, getSubShapeAndCount,
                                 std::move(loopArgTypes));
}

void populateXeGPUWgToSgDistributePatterns(RewritePatternSet &patterns) {
  patterns.add<WgToSgCreateNdOp, WgToSgLoadNdOp, WgToSgStoreNdOp, WgToSgDpasOp,
               WgToSgDpasMxOp, WgToSgPrefetchNdOp, WgToSgElementwiseOp,
               WgToSgVectorBroadcastOp, WgToSgConvertLayoutOp,
               WgToSgArithConstantOp, WgToSgLoadGatherOp, WgToSgStoreScatterOp,
               WgToSgLoadMatrixOp, WgToSgStoreMatrixOp, WgToSgCreateMemDescOp,
               WgToSgVectorStepOp,
               WgToSgVectorShapeCastOp, WgToSgMultiDimReductionOp,
               WgToSgVectorTransposeOp, WgToSgVectorConstantMaskOp,
               WgToSgVectorCreateMaskOp, WgToSgVectorBitCastOp,
               WgToSgVectorInterleaveOp, WgToSgVectorDeinterleaveOp>(
      patterns.getContext());
}
} // namespace xegpu
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// SLM privatization
//===----------------------------------------------------------------------===//
// An SLM buffer whose every matrix access (load_matrix/store_matrix) uses the
// same subgroup layout, the same data (vector) shape, and the same offsets is
// accessed identically by every subgroup. Because the same mem_desc implies the
// same physical layout, an identical (sg_layout, sg_data, offset) means every
// subgroup reads and writes the exact same SLM region, so the region is
// effectively private to each subgroup and can be demoted to private memory
// (space 4, backed by registers).
//
// This runs as a pre-phase of WG-to-SG distribution. For a qualifying buffer it
// (a) flips the source memref's memory space from 3 (SLM) to 4 (private), and
// (b) stashes the common workgroup layout on the create_mem_desc op as a
// discardable marker attribute so the WgToSgCreateMemDescOp pattern can later
// shrink the memref/mem_desc to the per-subgroup size. The matrix ops are left
// intact for the regular distribution patterns that follow.

// Compares two offset lists for structural equality. Static offsets compare by
// value; dynamic offsets compare by SSA value identity.
static bool sameOffsets(ArrayRef<OpFoldResult> a, ArrayRef<OpFoldResult> b) {
  if (a.size() != b.size())
    return false;
  for (auto [lhs, rhs] : llvm::zip(a, b)) {
    auto lhsAttr = dyn_cast<Attribute>(lhs);
    auto rhsAttr = dyn_cast<Attribute>(rhs);
    if (static_cast<bool>(lhsAttr) != static_cast<bool>(rhsAttr))
      return false;
    if (lhsAttr) {
      if (lhsAttr != rhsAttr)
        return false;
    } else if (cast<Value>(lhs) != cast<Value>(rhs)) {
      return false;
    }
  }
  return true;
}

// Returns the single create_mem_desc op that views `source` if `source` is a
// privatizable local buffer. The buffer must be a local alloca/alloc in SLM
// whose only user is exactly one create_mem_desc op; any other user (or a
// second view) means the buffer might escape or be aliased, so it is rejected.
static xegpu::CreateMemDescOp getPrivatizableView(Value source) {
  Operation *defOp = source.getDefiningOp();
  if (!defOp || !isa<memref::AllocaOp, memref::AllocOp>(defOp))
    return nullptr;

  auto memrefTy = dyn_cast<MemRefType>(source.getType());
  if (!memrefTy || !xegpu::XeGPUDialect::isSharedMemory(memrefTy))
    return nullptr;

  // The buffer must be viewed by exactly one create_mem_desc and nothing else,
  // so it neither escapes nor aliases another mem_desc.
  if (!source.hasOneUse())
    return nullptr;
  return dyn_cast<xegpu::CreateMemDescOp>(*source.getUsers().begin());
}

// If every load_matrix/store_matrix on `createOp`'s mem_desc is a workgroup
// access sharing the same sg_layout, sg_data, data shape, and offsets, and the
// workgroup tile is evenly distributed to the subgroups (no broadcast/overlap),
// returns that common layout. Otherwise returns nullptr.
static xegpu::DistributeLayoutAttr
getSubgroupPrivateLayout(xegpu::CreateMemDescOp createOp) {
  xegpu::DistributeLayoutAttr commonLayout;
  SmallVector<OpFoldResult> commonOffsets;
  ArrayRef<int64_t> commonDataShape;
  bool seen = false;

  auto checkAccess = [&](xegpu::DistributeLayoutAttr layout,
                         SmallVector<OpFoldResult> offsets,
                         ArrayRef<int64_t> dataShape) -> bool {
    // Every access must be a distributed workgroup-level access.
    if (!layout || !layout.isForWorkgroup())
      return false;

    // The workgroup tile must be evenly distributed across the subgroups with
    // no broadcast: sg_layout[d] * sg_data[d] must not exceed the tile in any
    // dim. A larger product means a dimension wraps around and is broadcast to
    // multiple subgroups, so their regions overlap and are not private.
    SmallVector<int64_t> sgLayout = layout.getEffectiveSgLayoutAsInt();
    SmallVector<int64_t> sgData = layout.getEffectiveSgDataAsInt();
    if (sgLayout.size() != dataShape.size() ||
        sgData.size() != dataShape.size())
      return false;
    for (auto [l, d, tile] : llvm::zip_equal(sgLayout, sgData, dataShape))
      if (l * d > tile)
        return false;

    if (!seen) {
      commonLayout = layout;
      commonOffsets = std::move(offsets);
      commonDataShape = dataShape;
      seen = true;
      return true;
    }
    return commonLayout.getEffectiveSgLayoutAsInt() == sgLayout &&
           commonLayout.getEffectiveSgDataAsInt() == sgData &&
           commonDataShape == dataShape && sameOffsets(commonOffsets, offsets);
  };

  for (Operation *memUser : createOp.getMemDesc().getUsers()) {
    if (auto loadOp = dyn_cast<xegpu::LoadMatrixOp>(memUser)) {
      if (!checkAccess(loadOp.getLayoutAttr(), loadOp.getMixedOffsets(),
                       loadOp.getDataShape()))
        return nullptr;
    } else if (auto storeOp = dyn_cast<xegpu::StoreMatrixOp>(memUser)) {
      if (!checkAccess(storeOp.getLayoutAttr(), storeOp.getMixedOffsets(),
                       storeOp.getDataShape()))
        return nullptr;
    } else {
      // Any other use of the mem_desc is unexpected; be conservative.
      return nullptr;
    }
  }

  // Require at least one access to have been observed.
  return seen ? commonLayout : nullptr;
}

// Rewrites the buffer defined by `defOp` (an alloca/alloc) so that its memory
// space is private (4) instead of shared (3), updates the source type of the
// create_mem_desc that consumes it, and marks that op with the common workgroup
// layout so the conversion pattern can shrink it to the per-subgroup size.
static void privatizeSlmBuffer(Operation *defOp, xegpu::CreateMemDescOp view,
                               xegpu::DistributeLayoutAttr layout) {
  OpBuilder builder(defOp);
  Value oldBuffer = defOp->getResult(0);
  auto oldTy = cast<MemRefType>(oldBuffer.getType());
  auto privateSpace = builder.getI64IntegerAttr(4);
  auto newTy = MemRefType::get(oldTy.getShape(), oldTy.getElementType(),
                               oldTy.getLayout(), privateSpace);

  Operation *newDefOp = builder.clone(*defOp);
  newDefOp->getResult(0).setType(newTy);
  oldBuffer.replaceAllUsesWith(newDefOp->getResult(0));
  defOp->erase();

  view->setAttr(kPrivatizeLayoutAttrName, layout);
}

// Entry point for the SLM privatization pre-phase.
static void privatizeSharedLocalMemory(Operation *root) {
  SmallVector<std::tuple<Operation *, xegpu::CreateMemDescOp,
                         xegpu::DistributeLayoutAttr>>
      toPrivatize;

  root->walk([&](xegpu::CreateMemDescOp createOp) {
    Value source = createOp.getSource();
    if (getPrivatizableView(source) != createOp)
      return;
    if (auto layout = getSubgroupPrivateLayout(createOp))
      toPrivatize.push_back({source.getDefiningOp(), createOp, layout});
  });

  for (auto [defOp, view, layout] : toPrivatize)
    privatizeSlmBuffer(defOp, view, layout);
}

struct XeGPUWgToSgDistributePass
    : public xegpu::impl::XeGPUWgToSgDistributeBase<XeGPUWgToSgDistributePass> {
  void runOnOperation() override;
};
} // namespace

void XeGPUWgToSgDistributePass::runOnOperation() {

  Operation *op = getOperation();

  if (!xegpu::recoverTemporaryLayouts(op)) {
    signalPassFailure();
    return;
  }

  // Pre-phase: demote SLM buffers that are accessed identically by every
  // subgroup (same sg_layout, sg_data, data shape and offsets) to subgroup-
  // private memory. Run after recoverTemporaryLayouts, which strips discardable
  // DistributeLayoutAttr-valued attrs (and would otherwise remove the marker
  // attribute this phase sets on create_mem_desc).
  privatizeSharedLocalMemory(op);

  // Collect existing UnrealizedConversionCastOps. These must be preserved.
  llvm::SmallSetVector<UnrealizedConversionCastOp, 8> existingCasts;
  getOperation()->walk(
      [&](UnrealizedConversionCastOp castOp) { existingCasts.insert(castOp); });

  // Perform workgroup to subgroup distribution for TensorDesc and Vector
  // values, as well as XeGPU, Arith, and Vector operations. Uses a
  // context-aware type converter that inspects Values to retrieve the
  // distribute layout attribute for 1:N type conversion.
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ConversionTarget target(*ctx);
  TypeConverter converter;
  // Source (N:1) and target (1:1) materializations using
  // UnrealizedConversionCastOp.
  auto materializeCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                            Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  };
  converter.addSourceMaterialization(materializeCast);
  converter.addTargetMaterialization(materializeCast);
  xegpu::populateXeGPUWgToSgDistributeTypeConversions(converter,
                                                      getOperation());

  auto getTensorDescType = [](Operation *op) -> xegpu::TensorDescType {
    if (auto createOp = dyn_cast<xegpu::CreateNdDescOp>(op))
      return createOp.getType();
    if (auto loadOp = dyn_cast<xegpu::LoadNdOp>(op))
      return loadOp.getTensorDescType();
    if (auto storeOp = dyn_cast<xegpu::StoreNdOp>(op))
      return storeOp.getTensorDescType();
    if (auto prefetchOp = dyn_cast<xegpu::PrefetchNdOp>(op))
      return prefetchOp.getTensorDescType();
    return xegpu::TensorDescType();
  };

  auto isLegal = [&](xegpu::DistributeLayoutAttr layout) -> bool {
    return !layout || !layout.isForWorkgroup();
  };

  target.addDynamicallyLegalOp<xegpu::CreateNdDescOp, xegpu::LoadNdOp,
                               xegpu::StoreNdOp, xegpu::PrefetchNdOp>(
      [=](Operation *op) -> bool {
        auto tdescTy = getTensorDescType(op);
        auto layout =
            dyn_cast_if_present<xegpu::LayoutAttr>(tdescTy.getLayout());
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<xegpu::DpasOp>([=](xegpu::DpasOp op) -> bool {
    auto layout = op.getLayoutCdAttr();
    return isLegal(layout);
  });

  target.addDynamicallyLegalOp<xegpu::DpasMxOp>(
      [=](xegpu::DpasMxOp op) -> bool {
        auto layout = op.getLayoutCdAttr();
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<xegpu::LoadMatrixOp>(
      [=](xegpu::LoadMatrixOp op) -> bool {
        return isLegal(op.getLayoutAttr());
      });

  target.addDynamicallyLegalOp<xegpu::StoreMatrixOp>(
      [=](xegpu::StoreMatrixOp op) -> bool {
        return isLegal(op.getLayoutAttr());
      });

  // A create_mem_desc marked by the privatization pre-phase must be shrunk to
  // the per-subgroup size; all others are already legal.
  target.addDynamicallyLegalOp<xegpu::CreateMemDescOp>(
      [=](xegpu::CreateMemDescOp op) -> bool {
        return !op->hasAttr(kPrivatizeLayoutAttrName);
      });

  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [=](arith::ConstantOp op) -> bool {
        auto vecType = dyn_cast<VectorType>(op.getType());
        if (!vecType)
          return true;

        auto layout =
            xegpu::getTemporaryLayout(dyn_cast<OpResult>(op.getResult()));
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<
      vector::ShapeCastOp, vector::StepOp, vector::TransposeOp,
      vector::BroadcastOp, vector::MultiDimReductionOp, vector::ConstantMaskOp,
      vector::CreateMaskOp, vector::BitCastOp, vector::InterleaveOp,
      vector::DeinterleaveOp>([=](Operation *op) -> bool {
    // Check for either a SliceAttr or LayoutAttr on the result.
    auto layout =
        xegpu::getTemporaryLayout(dyn_cast<OpResult>(op->getResult(0)));
    return isLegal(layout);
  });

  target.addDynamicallyLegalOp<xegpu::LoadGatherOp>(
      [=](xegpu::LoadGatherOp op) -> bool {
        auto layout = op.getLayoutAttr();
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<xegpu::StoreScatterOp>(
      [=](xegpu::StoreScatterOp op) -> bool {
        auto layout = op.getLayoutAttr();
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<xegpu::ConvertLayoutOp>(
      [=](xegpu::ConvertLayoutOp op) -> bool {
        return isLegal(op.getInputLayout()) && isLegal(op.getTargetLayout());
      });

  target.addDynamicallyLegalDialect<math::MathDialect, arith::ArithDialect>(
      [=](Operation *op) -> std::optional<bool> {
        // Only handle elementwise mappable ops
        if (!OpTrait::hasElementwiseMappableTraits(op))
          return true;

        VectorType resultType =
            dyn_cast<VectorType>(op->getResult(0).getType());
        if (!resultType)
          return true;

        // Check if all operands are vectors of the same shape
        // TODO: Support other types.
        for (Value operand : op->getOperands()) {
          VectorType operandType = dyn_cast<VectorType>(operand.getType());
          if (!operandType || operandType.getShape() != resultType.getShape()) {
            return true;
          }
        }

        xegpu::DistributeLayoutAttr layout =
            xegpu::getTemporaryLayout(op->getResult(0));
        return isLegal(layout);
      });

  target.addLegalOp<UnrealizedConversionCastOp>();

  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                       target);
  xegpu::populateXeGPUWgToSgDistributePatterns(patterns);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();

  // Fold cancelling cast chains and erase dead casts.
  xegpu::cleanupUnrealizedConversionCasts(getOperation(), existingCasts);
  xegpu::removeTemporaryLayoutAttrs(getOperation());

  // Erase the original full-size buffers left dead by privatization: the
  // WgToSgCreateMemDescOp pattern allocates fresh per-subgroup buffers, so the
  // pre-phase's space-4 allocas no longer have any users.
  getOperation()->walk([](Operation *op) {
    if (isa<memref::AllocaOp, memref::AllocOp>(op) && op->use_empty())
      op->erase();
  });
}
