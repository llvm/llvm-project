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
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUWGTOSGDISTRIBUTE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;

namespace {

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
  if (layout && layout.isForWorkgroup()) {
    SmallVector<int64_t> sgLayout = layout.getEffectiveSgLayoutAsInt();
    if (!layout.getEffectiveSgDataAsInt().empty())
      sgShape = layout.getEffectiveSgDataAsInt();
    else if (auto maybeDerivedSgData = computeShapeRatio(shape, sgLayout))
      sgShape = *maybeDerivedSgData;
    SmallVector<int64_t> distUnit = computeElementwiseMul(sgLayout, sgShape);
    // Clamp distUnit to the original shape to handle cases where data is
    // shared among subgroups, which may cause distUnit to exceed the original
    // shape.
    for (size_t i = 0; i < distUnit.size(); ++i)
      distUnit[i] = std::min(shape[i], distUnit[i]);
    count = computeProduct(shape) / computeProduct(distUnit);
  }
  return std::make_pair(sgShape, count);
}

/// Utility helper for deriving a list of offsets for each sub-TensorDescs
/// or sub-MemDescs to be accessed by current subgroup (sgId) based on the
/// associated distribute layout attribute, the shape, subgroup id and the
/// original offsets of the op
template <
    typename OpType,
    typename = std::enable_if_t<llvm::is_one_of<
        OpType, xegpu::CreateNdDescOp, xegpu::LoadNdOp, xegpu::StoreNdOp,
        xegpu::PrefetchNdOp, xegpu::LoadMatrixOp, xegpu::StoreMatrixOp>::value>>
static LogicalResult
genOffsetsList(ConversionPatternRewriter &rewriter, OpType op,
               SmallVector<SmallVector<OpFoldResult>> &offsetsList) {
  Location loc = op.getLoc();
  SmallVector<OpFoldResult> origOffsets = op.getMixedOffsets();
  // not applicable to ops without offsets operands.
  if (origOffsets.empty())
    return failure();

  // not applicable to ops without workgroup layout attributes
  xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
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
  auto maybeDescOffsets = layout.getOffsets(rewriter, loc, sgId, wgShape);
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
/// Following create_nd_desc operation:,
///    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x24xf32>
///       -> !xegpu.tensor_desc<24x24xf32, #xegpu.layout<sg_layout = [4, 4],
///           sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
/// is converted to 9 subgroup level operations based on the sg_layout &
/// sg_data:
///    %tdesc = xegpu.create_nd_tdesc %src[off1, off2] : memref<24x24xf32> ->
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
struct WgToSgCreateNdOp : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (failed(genOffsetsList(rewriter, op, offsetsList)))
      return failure();

    MLIRContext *ctx = op.getContext();
    xegpu::TensorDescType tdescTy = op.getType();
    ArrayRef<int64_t> wgShape = tdescTy.getShape();
    Type elemTy = tdescTy.getElementType();
    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;
    auto newTdescTy =
        xegpu::TensorDescType::get(ctx, sgShape, elemTy, tdescTy.getEncoding(),
                                   layout.dropSgLayoutAndData());

    SmallVector<Value> newOps;
    for (auto offsets : offsetsList) {
      auto newOp = xegpu::CreateNdDescOp::create(
          rewriter, op.getLoc(), newTdescTy, op.getSource(), offsets,
          op.getMixedSizes(), op.getMixedStrides());

      newOps.push_back(newOp);
    }
    rewriter.replaceOpWithMultiple(op, {newOps});

    return success();
  }
};

// This pattern transforms the CreateNdDescOp without offsets to create a
// subgroup descriptor from a workgroup descriptor
struct WgToSgCreateNdOpNoOffset
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check no offsets are specified.
    if (!op.getMixedOffsets().empty())
      return failure();

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
    if (!op.getMixedOffsets().empty())
      return failure();

    SmallVector<Value> newLoadOps;
    for (auto src : adaptor.getTensorDesc()) {
      xegpu::TensorDescType tdescTy =
          dyn_cast<xegpu::TensorDescType>(src.getType());
      ArrayRef<int64_t> srcShape = tdescTy.getShape();
      VectorType newResTy = VectorType::get(srcShape, tdescTy.getElementType());
      auto newLoadOp = xegpu::LoadNdOp::create(rewriter, op.getLoc(), newResTy,
                                               src, op->getAttrs());
      newLoadOps.push_back(newLoadOp);
    }
    rewriter.replaceOpWithMultiple(op, {newLoadOps});
    return mlir::success();
  }
};

/// This pattern transforms the StoreNdOp to store to a subgroup descriptor
/// It creates a StoreNdOp op to store the updated values to the new subgroup
/// src tensor descriptors.
struct WgToSgStoreNdOp : public OpConversionPattern<xegpu::StoreNdOp> {
  using OpConversionPattern<xegpu::StoreNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::StoreNdOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getMixedOffsets().empty())
      return failure();

    for (auto [v, t] : llvm::zip(adaptor.getValue(), adaptor.getTensorDesc()))
      xegpu::StoreNdOp::create(rewriter, op.getLoc(), v, t, op.getL1HintAttr(),
                               op.getL2HintAttr(), op.getL3HintAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

// This pattern transforms the LoadNdOp with explicit offsets to load
// subgroup data.
struct WgToSgLoadNdOpWithOffset : public OpConversionPattern<xegpu::LoadNdOp> {
  using OpConversionPattern<xegpu::LoadNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::LoadNdOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (failed(genOffsetsList(rewriter, op, offsetsList)))
      return failure();

    SmallVector<Value> newOps;
    for (auto [tdesc, offsets] :
         llvm::zip(adaptor.getTensorDesc(), offsetsList)) {
      auto tdescTy = dyn_cast<xegpu::TensorDescType>(tdesc.getType());
      VectorType newResTy =
          VectorType::get(tdescTy.getShape(), tdescTy.getElementType());
      auto newOp = xegpu::LoadNdOp::create(
          rewriter, op.getLoc(), newResTy, tdesc, offsets,
          /*packed = */ nullptr, /*transpose = */ nullptr, op.getL1HintAttr(),
          op.getL2HintAttr(), op.getL3HintAttr());
      newOps.push_back(newOp);
    }
    rewriter.replaceOpWithMultiple(op, {newOps});

    return success();
  }
};

// This pattern transforms the StoreNdOp with explicit offsets to store
// subgroup data.
struct WgToSgStoreNdOpWithOffset
    : public OpConversionPattern<xegpu::StoreNdOp> {
  using OpConversionPattern<xegpu::StoreNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::StoreNdOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (failed(genOffsetsList(rewriter, op, offsetsList)))
      return failure();

    for (auto [v, tdesc, offsets] :
         llvm::zip(adaptor.getValue(), adaptor.getTensorDesc(), offsetsList)) {
      xegpu::StoreNdOp::create(rewriter, op.getLoc(), v, tdesc, offsets,
                               op.getL1HintAttr(), op.getL2HintAttr(),
                               op.getL3HintAttr());
    }
    rewriter.eraseOp(op);

    return success();
  }
};

// This pattern transforms the PrefetchNdOp with explicit offsets to prefetch
// subgroup data.
struct WgToSgPrefetchNdOpWithOffset
    : public OpConversionPattern<xegpu::PrefetchNdOp> {
  using OpConversionPattern<xegpu::PrefetchNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::PrefetchNdOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (failed(genOffsetsList(rewriter, op, offsetsList)))
      return failure();

    for (auto [tdesc, offsets] :
         llvm::zip(adaptor.getTensorDesc(), offsetsList)) {
      xegpu::PrefetchNdOp::create(rewriter, op.getLoc(), tdesc, offsets,
                                  op.getL1HintAttr(), op.getL2HintAttr(),
                                  op.getL3HintAttr());
    }
    rewriter.eraseOp(op);

    return success();
  }
};

/// This pattern transforms the UpdateNdOffsetOp to update the offsets of a
/// subgroup descriptor. It creates an UpdateNdOffsetOp op to update the
/// offsets of the new subgroup src tensor descriptors.
struct WgToSgUpdateNdOffsetOp
    : public OpConversionPattern<xegpu::UpdateNdOffsetOp> {
  using OpConversionPattern<xegpu::UpdateNdOffsetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::UpdateNdOffsetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> newUpdateTileOffsetOps;
    for (auto tDesc : adaptor.getTensorDesc()) {
      auto newUpdateTileOffsetOp = xegpu::UpdateNdOffsetOp::create(
          rewriter, op.getLoc(), tDesc.getType(), tDesc, op.getOffsets(),
          op.getConstOffsets());
      newUpdateTileOffsetOps.push_back(newUpdateTileOffsetOp);
    }

    rewriter.replaceOpWithMultiple(op, {newUpdateTileOffsetOps});
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
    if (resultTy.getRank() != 2)
      return failure();

    auto originalLayout = xegpu::getDistributeLayoutAttr(op.getResult());
    if (!originalLayout)
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
            llvm::cast<VectorType>(aVec.getType()).getShape();
        ArrayRef<int64_t> bVecShape =
            llvm::cast<VectorType>(bVec.getType()).getShape();
        VectorType resTy = VectorType::get({aVecShape[0], bVecShape[1]},
                                           resultTy.getElementType());
        tmpC = xegpu::DpasOp::create(rewriter, loc, resTy, operands);
        xegpu::setDistributeLayoutAttr(cast<OpResult>(tmpC),
                                       originalLayout.dropSgLayoutAndData());

        newDpasOps.push_back(tmpC);
      }
    }
    rewriter.replaceOpWithMultiple(op, {newDpasOps});
    return success();
  }
};

/// This pattern transforms the PrefetchNdOp to prefetch the subgroup data.
struct WgToSgPrefetchNdOp : public OpConversionPattern<xegpu::PrefetchNdOp> {
  using OpConversionPattern<xegpu::PrefetchNdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::PrefetchNdOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    int64_t offsetSize = static_cast<int64_t>(op.getOffsets().size());
    if ((offsetSize != 0) || op.getConstOffsetsAttr())
      return failure();

    for (auto src : adaptor.getTensorDesc())
      xegpu::PrefetchNdOp::create(rewriter, op.getLoc(), TypeRange(), src,
                                  op->getAttrs());
    rewriter.eraseOp(op);
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
        xegpu::getDistributeLayoutAttr(op.getResult());
    if (!layout || !layout.isForWorkgroup())
      return failure();

    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;
    VectorType newResultType =
        VectorType::get(sgShape, resultType.getElementType());

    if (!xegpu::XeGPUDialect::isEvenlyDistributable(wgShape, layout))
      return failure();

    SmallVector<Value> newBroadcastOps;
    for (auto operand : adaptor.getOperands().front()) {
      auto newBroadcast = vector::BroadcastOp::create(rewriter, op.getLoc(),
                                                      newResultType, operand);
      if (!layout.getEffectiveLaneLayoutAsInt().empty() ||
          !layout.getEffectiveInstDataAsInt().empty())
        xegpu::setDistributeLayoutAttr(newBroadcast->getResult(0),
                                       layout.dropSgLayoutAndData());

      newBroadcastOps.push_back(newBroadcast.getResult());
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
        xegpu::getDistributeLayoutAttr(op->getResult(0));
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
      // Copy all attributes, but update "layout_result_0" to drop
      // sgLayout/sgData
      for (auto attr : op->getAttrs()) {
        if (auto layout =
                dyn_cast<xegpu::DistributeLayoutAttr>(attr.getValue())) {
          if (!layout.getEffectiveLaneLayoutAsInt().empty() ||
              !layout.getEffectiveInstDataAsInt().empty())
            state.addAttribute(attr.getName(), layout.dropSgLayoutAndData());
        } else {
          state.addAttribute(attr.getName(), attr.getValue());
        }
      }
      Operation *newOp = rewriter.create(state);
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
    // TODO: currently, we only support LayoutAttr
    auto input = dyn_cast<xegpu::LayoutAttr>(op.getInputLayout());
    auto target = dyn_cast<xegpu::LayoutAttr>(op.getTargetLayout());

    if (!input || !target || !input.isForWorkgroup() ||
        !target.isForWorkgroup())
      return rewriter.notifyMatchFailure(
          op, "Input and target layouts must have subgroup layout");

    DenseI32ArrayAttr inputSgLayout = input.getSgLayout();
    DenseI32ArrayAttr inputSgData = input.getSgData();
    DenseI32ArrayAttr inputOrder = input.getOrder();
    DenseI32ArrayAttr targetSgLayout = target.getSgLayout();
    DenseI32ArrayAttr targetSgData = target.getSgData();
    DenseI32ArrayAttr targetOrder = target.getOrder();

    // TODO: currently we only support for optimal case, where input and
    // output has the same sg_layout and sg_data, so SLM is not involved.
    if (inputSgLayout != targetSgLayout || inputSgData != targetSgData ||
        inputOrder != targetOrder)
      return failure();

    input = input.dropSgLayoutAndData();
    target = target.dropSgLayoutAndData();

    SmallVector<Value> newOps(adaptor.getSource());
    if (input && target) {
      // keep the ConvertLayoutOp for rest fields, e.g., inst_data.
      for (auto [i, src] : llvm::enumerate(adaptor.getSource())) {
        auto newOp = xegpu::ConvertLayoutOp::create(
            rewriter, op.getLoc(), src.getType(), src, input, target);
        newOps[i] = newOp;
      }
    }
    rewriter.replaceOpWithMultiple(op, {newOps});
    return success();
  }
};

// Handles UnrealizedConversionCastOp generated during
// SCFStructuralTypeConversions (step 1). This op may appear as either a
// target or source materialization for Vector values, e.g.:
// 1. unrealized_cast %1 : vector<256xf32> to vector<16xf32>, ...
// 2. unrealized_cast %1 : vector<16xf32>, ... to vector<256xf32>
// it could be either 1:N or N:1 cast. In both cases, the pattern
// simply forwards the inputs to the outputs using 1:1 or 1:N interface.
// for example, the following scf::forOp
// ```
// %for = scf.for ... iter_args(%arg1 = %0)->(vector<128x128xf16>) {
//     %n = use(%arg1): vector<128x128xf16>
//     scf.yield %n : vector<128x128xf16>
// }
// ```
// Could be converted to:
// ```
// %1 = unrealized_conversion_cast %0
//          : vector<128x128xf16> to vector<16x16xf16>, vector<16x16xf16>
// %for:2 = scf.for ... iter_args(%arg1 = %1#1, %arg2 = %1#2)
//                    -> (vector<16x16xf16>, vector<16x16xf16) {
//     %m = unrealized_conversion_cast %arg1, %arg2
//            : vector<16x16xf16>, vector<16x16xf16> to vector<128x128xf16>
//     %n = use(%m): vector<128x128xf16>
//     %b = unrealized_conversion_cast %n
//            : vector<128x128xf16> to vector<16x16xf16>, vector<16x16xf16>
//     scf.yield %b#1, %b#2 : vector<16x16xf16>, vector<16x16xf16>
// }
// %cast = unrealized_conversion_cast %for:2
//          : vector<16x16xf16>, vector<16x16xf16> to vector<128x128xf16>
// ```
// TODO: remove it when context-aware type converter is ready.
struct UnrealizedConversionCastOpPattern
    : public OpConversionPattern<mlir::UnrealizedConversionCastOp> {
  using OpConversionPattern<
      mlir::UnrealizedConversionCastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = xegpu::flattenValues(adaptor.getInputs());

    auto inputTy = dyn_cast<VectorType>(inputs[0].getType());
    auto outputTy = dyn_cast<VectorType>(op->getOpResult(0).getType());

    if (!inputTy || !outputTy || !llvm::all_equal(op->getResultTypes()) ||
        !llvm::all_equal(ValueRange(inputs).getTypes()))
      return failure();

    // Handles the case "cast %1 : vector<256xf32> to vector<16xf32>, ...".
    // It is generated by source materialization (e.g., inits to scf forOp).
    // The input values provided by the adaptor should already be distributed,
    // and their types should correspond exactly to the result types of the
    // operation.
    if (op.getNumOperands() == 1 &&
        llvm::equal(ValueRange(inputs).getTypes(), op->getResultTypes())) {
      rewriter.replaceOp(op, inputs);
      return success();
    }

    // Handles the case "cast %1 : vector<16xf32>, ... to vector<256xf32>".
    // It is generated by target materialization (e.g., arguments/results
    // of scf forOp). All input values must have the same vector type, and
    // their shape must be evenly divisible by the output vector's shape
    // (determined by the nature of the workgroup to subgroup distribution).
    // TODO: it is not safe to do such forward, since such N:1 cast could be
    // from others.
    if (op.getNumResults() == 1 &&
        computeShapeRatio(outputTy.getShape(), inputTy.getShape())) {
      rewriter.replaceOpWithMultiple(op, {inputs});
      return success();
    }

    return mlir::failure();
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
        xegpu::getDistributeLayoutAttr(op.getResult());
    if (!layout || !layout.isForWorkgroup())
      return failure();

    ArrayRef<int64_t> wgShape = vecType.getShape();
    SmallVector<int64_t> sgShape;
    int count;
    std::tie(sgShape, count) = getSgShapeAndCount(wgShape, layout);

    auto newType = VectorType::get(sgShape, vecType.getElementType());
    Location loc = op.getLoc();
    auto eltType = vecType.getElementType();

    auto setLayoutIfNeeded = [&](Value val) {
      if (!layout.getEffectiveLaneLayoutAsInt().empty() ||
          !layout.getEffectiveInstDataAsInt().empty()) {
        xegpu::setDistributeLayoutAttr(llvm::dyn_cast<OpResult>(val),
                                       layout.dropSgLayoutAndData());
      }
    };

    if (vecAttr.isSplat()) {
      // Splat: single value for all subgroups
      Attribute singleVal = vecAttr.getSplatValue<Attribute>();
      auto sgAttr = DenseElementsAttr::get(newType, singleVal);
      auto cstOp = arith::ConstantOp::create(rewriter, loc, newType, sgAttr);
      setLayoutIfNeeded(cstOp->getResult(0));
      rewriter.replaceOp(op, cstOp);
      return success();
    } else if (sgShape == wgShape) { // if the entire vector is shared by all
                                     // subgroups, don't distribute
      auto newConstOp =
          arith::ConstantOp::create(rewriter, op.getLoc(), vecType, vecAttr);
      setLayoutIfNeeded(newConstOp->getResult(0));
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
      auto baseConstVec = rewriter.create<arith::ConstantOp>(loc, tileAttr);

      // Get subgroup id
      Value sgId =
          gpu::SubgroupIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);

      auto sgOffsets = layout.getOffsets(rewriter, loc, sgId, wgShape);
      if (failed(sgOffsets))
        return failure();

      SmallVector<Value, 2> strideConsts;
      strideConsts.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, colStride));
      if (rows > 1)
        strideConsts.insert(
            strideConsts.begin(),
            rewriter.create<arith::ConstantIndexOp>(loc, rowStride));

      SmallVector<Value> newConstOps;
      for (auto offsets : *sgOffsets) {
        // Multiply offset with stride, broadcast it and add to baseConstVec
        Value mulOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        for (size_t i = 0; i < strideConsts.size(); ++i) {
          Value mul = rewriter.create<arith::MulIOp>(
              loc, rewriter.getIndexType(), offsets[i], strideConsts[i]);
          mulOffset = rewriter.create<arith::AddIOp>(
              loc, rewriter.getIndexType(), mulOffset, mul);
        }
        // Broadcast to baseConstVec size
        auto bcastOffset = rewriter.create<vector::BroadcastOp>(
            loc, baseConstVec.getType(), mulOffset);
        auto finalConst =
            arith::AddIOp::create(rewriter, loc, baseConstVec, bcastOffset);
        setLayoutIfNeeded(baseConstVec);
        setLayoutIfNeeded(bcastOffset);
        setLayoutIfNeeded(finalConst);
        newConstOps.push_back(finalConst);
      }
      rewriter.replaceOpWithMultiple(op, {newConstOps});
      return success();
    }
  }
};

// This pattern transforms the LoadGatherOp with explicit offsets to load
// subgroup data
struct WgToSgLoadGatherOpWithOffset
    : public OpConversionPattern<xegpu::LoadGatherOp> {
  using OpConversionPattern<xegpu::LoadGatherOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::LoadGatherOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getOffsets())
      return failure();

    Location loc = op.getLoc();
    VectorType resultType = dyn_cast<VectorType>(op.getResult().getType());
    if (!resultType)
      return failure();
    ArrayRef<int64_t> wgShape = resultType.getShape();

    xegpu::DistributeLayoutAttr layout =
        xegpu::getDistributeLayoutAttr(op.getResult());
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
      auto newLoadOp = xegpu::LoadGatherOp::create(
          rewriter, loc, newTy, op.getSource(), offsets, mask, chunkSizeAttr,
          op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr());
      xegpu::setDistributeLayoutAttr(newLoadOp->getResult(0),
                                     layout.dropSgLayoutAndData());
      newLoadOps.push_back(newLoadOp);
    }
    rewriter.replaceOpWithMultiple(op, {newLoadOps});
    return success();
  }
};

// This pattern transforms the StoreScatterOp with explicit offsets to store
// subgroup data
struct WgToSgStoreScatterOpWithOffset
    : public OpConversionPattern<xegpu::StoreScatterOp> {
  using OpConversionPattern<xegpu::StoreScatterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::StoreScatterOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getOffsets())
      return failure();

    Location loc = op.getLoc();
    VectorType valueType = dyn_cast<VectorType>(op.getValue().getType());
    if (!valueType)
      return failure();

    xegpu::DistributeLayoutAttr layout =
        xegpu::getDistributeLayoutAttr(op.getOperand(0));
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
      auto store = xegpu::StoreScatterOp::create(
          rewriter, loc, val, op.getDest(), offs, mask, chunkSizeAttr,
          op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr());
      // Update the layout attribute to drop sg_layout and sg_data.
      if (!layout.getEffectiveLaneLayoutAsInt().empty() ||
          !layout.getEffectiveInstDataAsInt().empty()) {
        for (OpOperand &operand : store->getOpOperands()) {
          // Skip for operand one (memref)
          if (operand.getOperandNumber() == 1)
            continue;
          xegpu::setDistributeLayoutAttr(operand, layout.dropSgLayoutAndData());
        }
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct WgToSgLoadMatrixOp : public OpConversionPattern<xegpu::LoadMatrixOp> {
  using OpConversionPattern<xegpu::LoadMatrixOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::LoadMatrixOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (failed(genOffsetsList(rewriter, op, offsetsList)))
      return failure();

    ArrayRef<int64_t> wgShape = op.getDataShape();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getRes().getType());
    assert(valueTy && "the value type must be vector type!");
    Type elemTy = valueTy.getElementType();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;
    VectorType newResTy = VectorType::get(sgShape, elemTy);
    SmallVector<Value> newOps;
    for (auto offsets : offsetsList) {
      auto newOp = xegpu::LoadMatrixOp::create(rewriter, op.getLoc(), newResTy,
                                               op.getMemDesc(), offsets,
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

    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    if (failed(genOffsetsList(rewriter, op, offsetsList)))
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    for (auto [v, offsets] : llvm::zip(adaptor.getData(), offsetsList))
      xegpu::StoreMatrixOp::create(rewriter, op.getLoc(), v, op.getMemDesc(),
                                   offsets, layout.dropSgLayoutAndData());
    rewriter.eraseOp(op);
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
        xegpu::getDistributeLayoutAttr(op.getResult());
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
    auto sgOffsets = layout.getOffsets(rewriter, loc, sgId, wgShape);
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
      if (!layout.getEffectiveLaneLayoutAsInt().empty() ||
          !layout.getEffectiveInstDataAsInt().empty()) {
        xegpu::setDistributeLayoutAttr(steps->getResult(0),
                                       layout.dropSgLayoutAndData());
        xegpu::setDistributeLayoutAttr(bcastOffset->getResult(0),
                                       layout.dropSgLayoutAndData());
        xegpu::setDistributeLayoutAttr(finalSteps->getResult(0),
                                       layout.dropSgLayoutAndData());
      }
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
        xegpu::getDistributeLayoutAttr(op.getResult());
    if (!layout || !layout.isForWorkgroup())
      return failure();

    SmallVector<int64_t> sgShape = getSgShapeAndCount(wgShape, layout).first;
    VectorType newResultType =
        VectorType::get(sgShape, resultType.getElementType());

    // TODO: Add check for compatible layouts in layout attr.
    auto srcType = dyn_cast<VectorType>(adaptor.getSource()[0].getType());
    if (!srcType)
      return failure();

    // Check that shape_cast only adds/removes unit dimensions,
    auto onlyUnitDims = [](ArrayRef<int64_t> src, ArrayRef<int64_t> dst) {
      // Remove all 1s from both shapes and compare the rest.
      SmallVector<int64_t> srcNonUnit, dstNonUnit;
      for (int64_t d : src)
        if (d != 1)
          srcNonUnit.push_back(d);
      for (int64_t d : dst)
        if (d != 1)
          dstNonUnit.push_back(d);
      return srcNonUnit == dstNonUnit;
    };

    if (!onlyUnitDims(srcType.getShape(), sgShape))
      return failure();

    // For rank reducing or increasing shape_cast ops, the lower rank layout
    // must be a slice of higher rank layout.
    int64_t sourceRank = srcType.getRank();
    int64_t resultRank = sgShape.size();
    xegpu::DistributeLayoutAttr sourceLayout =
        xegpu::getDistributeLayoutAttr(op.getSource());
    if (sourceRank < resultRank && !sourceLayout.isSliceOf(layout))
      return failure();
    if (sourceRank > resultRank && !layout.isSliceOf(sourceLayout))
      return failure();

    SmallVector<Value> newShapeCastOps;
    for (auto src : adaptor.getSource()) {
      auto newShapeCast =
          rewriter.create<vector::ShapeCastOp>(op.getLoc(), newResultType, src);
      if (!layout.getEffectiveLaneLayoutAsInt().empty() ||
          !layout.getEffectiveInstDataAsInt().empty())
        xegpu::setDistributeLayoutAttr(newShapeCast->getResult(0),
                                       layout.dropSgLayoutAndData());
      newShapeCastOps.push_back(newShapeCast.getResult());
    }

    rewriter.replaceOpWithMultiple(op, {newShapeCastOps});
    return success();
  }
};

/// Pattern for lowering vector.multi_reduction op to subgroup level.
/// Current limitation: the sg_layout in the reduced dimension being 1
/// so that reduction is local to subgroup & no cross-subgroup communication is
/// needed.
/// TODO: Add cases to handle more general situations which require SLM access.
struct WgToSgMultiDimReductionOp
    : public OpConversionPattern<vector::MultiDimReductionOp> {
  using OpConversionPattern<vector::MultiDimReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MultiDimReductionOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = op.getSourceVectorType();
    VectorType dstType = dyn_cast<VectorType>(op.getResult().getType());
    if (!dstType)
      return failure();

    auto srcShape = srcType.getShape();
    xegpu::DistributeLayoutAttr layout =
        xegpu::getDistributeLayoutAttr(op.getResult());
    if (!layout || !layout.isForWorkgroup())
      return failure();

    auto reductionDims = llvm::to_vector(op.getReductionDims());

    SmallVector<int64_t> sgLayout = llvm::cast<xegpu::SliceAttr>(layout)
                                        .getParent()
                                        .getEffectiveSgLayoutAsInt();
    SmallVector<int64_t> sgData = llvm::cast<xegpu::SliceAttr>(layout)
                                      .getParent()
                                      .getEffectiveSgDataAsInt();

    // Check that the sgLayout in the reduced dimension is 1 and
    // each sg gets the entire slice to reduce.
    for (int64_t dim : reductionDims) {
      if (sgLayout[dim] != 1 || sgData[dim] != srcShape[dim])
        return rewriter.notifyMatchFailure(
            op,
            "sgLayout in each reduced dimension must be 1 and sgData in the "
            "reduced dim must match srcShape in that dim");
    }

    SmallVector<int64_t> sgShape = getSgShapeAndCount(srcShape, layout).first;

    VectorType newDstType =
        VectorType::get({sgShape}, dstType.getElementType());

    SmallVector<Value> newReductions;
    for (auto sgSrc : adaptor.getSource()) {
      auto newOp = rewriter.create<vector::MultiDimReductionOp>(
          op.getLoc(), newDstType, op.getKind(), sgSrc, adaptor.getAcc()[0],
          op.getReductionDims());
      if (!layout.getEffectiveLaneLayoutAsInt().empty() ||
          !layout.getEffectiveInstDataAsInt().empty())
        xegpu::setDistributeLayoutAttr(newOp->getResult(0),
                                       layout.dropSgLayoutAndData());
      newReductions.push_back(newOp.getResult());
    }

    rewriter.replaceOpWithMultiple(op, {newReductions});
    return success();
  }
};

} // namespace

namespace mlir {
namespace xegpu {
void populateXeGPUWgToSgDistributePatterns(RewritePatternSet &patterns) {
  patterns
      .add<WgToSgCreateNdOp, WgToSgCreateNdOpNoOffset, WgToSgLoadNdOp,
           WgToSgLoadNdOpWithOffset, WgToSgStoreNdOp, WgToSgStoreNdOpWithOffset,
           WgToSgUpdateNdOffsetOp, WgToSgDpasOp, WgToSgPrefetchNdOp,
           WgToSgPrefetchNdOpWithOffset, UnrealizedConversionCastOpPattern,
           WgToSgElementwiseOp, WgToSgVectorBroadcastOp, WgToSgConvertLayoutOp,
           WgToSgArithConstantOp, WgToSgLoadGatherOpWithOffset,
           WgToSgStoreScatterOpWithOffset, WgToSgLoadMatrixOp,
           WgToSgStoreMatrixOp, WgToSgVectorStepOp, WgToSgVectorShapeCastOp,
           WgToSgMultiDimReductionOp>(patterns.getContext());
}
} // namespace xegpu
} // namespace mlir

namespace {
struct XeGPUWgToSgDistributePass
    : public xegpu::impl::XeGPUWgToSgDistributeBase<XeGPUWgToSgDistributePass> {
  void runOnOperation() override;
};
} // namespace

void XeGPUWgToSgDistributePass::runOnOperation() {
  // Track existing UnrealizedConversionCastOps
  SmallVector<Operation *> existingCastOps;
  getOperation()->walk([&](UnrealizedConversionCastOp castOp) {
    existingCastOps.push_back(castOp.getOperation());
  });

  {
    // Step 1: Apply SCFStructuralTypeConversions to SCF operations with
    // VectorType operands. This first converts such operands to
    // RankedTensorType, propagates the layout attribute into the encoding
    // attribute, and finally converts the RankedTensorType to VectorType based
    // on the encoding.

    TypeConverter converter;
    converter.addConversion([&](Type type) -> Type { return type; });
    converter.addConversion(
        [&](RankedTensorType type,
            SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
          Type elemTy = type.getElementType();
          ArrayRef<int64_t> shape = type.getShape();

          int count;
          SmallVector<int64_t> subShape;
          std::tie(subShape, count) = getSgShapeAndCount(
              shape,
              dyn_cast_if_present<xegpu::LayoutAttr>(type.getEncoding()));

          auto newTy = VectorType::get(subShape, elemTy);
          result.append(count, newTy);
          return success();
        });

    xegpu::doSCFStructuralTypeConversionWithTensorType(getOperation(),
                                                       converter);
  }

  // Step 2: Perform workgroup to subgroup distribution for TensorDesc values,
  // as well as XeGPU, Arith, and Vector operations.
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ConversionTarget target(*ctx);
  TypeConverter converter;
  converter.addConversion([&](Type type) -> Type { return type; });
  converter.addConversion(
      [&](xegpu::TensorDescType type,
          SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        Type elemTy = type.getElementType();
        ArrayRef<int64_t> shape = type.getShape();

        int count;
        SmallVector<int64_t> subShape;
        xegpu::LayoutAttr layout = type.getLayoutAttr();
        std::tie(subShape, count) = getSgShapeAndCount(shape, layout);

        if (layout)
          layout = layout.dropSgLayoutAndData();

        auto newTy = xegpu::TensorDescType::get(
            type.getContext(), subShape, elemTy, type.getEncoding(), layout);
        result.append(count, newTy);
        return success();
      });

  auto getTensorDescType = [](Operation *op) -> xegpu::TensorDescType {
    if (auto createOp = dyn_cast<xegpu::CreateNdDescOp>(op))
      return createOp.getType();
    if (auto loadOp = dyn_cast<xegpu::LoadNdOp>(op))
      return loadOp.getTensorDescType();
    if (auto storeOp = dyn_cast<xegpu::StoreNdOp>(op))
      return storeOp.getTensorDescType();
    if (auto updateOp = dyn_cast<xegpu::UpdateNdOffsetOp>(op))
      return updateOp.getType();
    if (auto prefetchOp = dyn_cast<xegpu::PrefetchNdOp>(op))
      return prefetchOp.getTensorDescType();
    return xegpu::TensorDescType();
  };

  auto isLegal = [&](xegpu::DistributeLayoutAttr layout) -> bool {
    return !layout || !layout.isForWorkgroup();
  };

  target.addDynamicallyLegalOp<xegpu::CreateNdDescOp, xegpu::LoadNdOp,
                               xegpu::StoreNdOp, xegpu::UpdateNdOffsetOp,
                               xegpu::PrefetchNdOp>([=](Operation *op) -> bool {
    auto tdescTy = getTensorDescType(op);
    auto layout = dyn_cast_if_present<xegpu::LayoutAttr>(tdescTy.getLayout());
    return isLegal(layout);
  });

  target.addDynamicallyLegalOp<xegpu::DpasOp>([=](xegpu::DpasOp op) -> bool {
    auto layout = xegpu::getDistributeLayoutAttr(op.getResult());
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

  target.addDynamicallyLegalOp<arith::ConstantOp>(
      [=](arith::ConstantOp op) -> bool {
        auto vecType = dyn_cast<VectorType>(op.getType());
        if (!vecType)
          return true;

        auto layout = xegpu::getDistributeLayoutAttr(op.getResult());
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<vector::ShapeCastOp, vector::StepOp>(
      [=](Operation *op) -> bool {
        // Check for either a SliceAttr or LayoutAttr on the result.
        auto layout = xegpu::getDistributeLayoutAttr(op->getResult(0));
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<xegpu::LoadGatherOp>(
      [=](xegpu::LoadGatherOp op) -> bool {
        auto layout = xegpu::getDistributeLayoutAttr(op.getResult());
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<xegpu::StoreScatterOp>(
      [=](xegpu::StoreScatterOp op) -> bool {
        auto layout = xegpu::getDistributeLayoutAttr(op.getOperand(0));
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<vector::BroadcastOp>(
      [=](vector::BroadcastOp op) -> bool {
        return isLegal(xegpu::getDistributeLayoutAttr(op.getResult()));
      });

  target.addDynamicallyLegalOp<vector::MultiDimReductionOp>(
      [=](vector::MultiDimReductionOp op) -> bool {
        return isLegal(xegpu::getDistributeLayoutAttr(op.getResult()));
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
            xegpu::getDistributeLayoutAttr(op->getResult(0));
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [=](UnrealizedConversionCastOp op) {
        return llvm::is_contained(existingCastOps, op.getOperation());
      });

  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                       target);
  xegpu::populateXeGPUWgToSgDistributePatterns(patterns);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();

  // Remove sg_layout and sg_data attributes from the Layout
  // attribute for each VectorType result of the operation.
  // For Structured Control Flow ops, the layout is simply removed,
  // since in 1:N case, the layout for new results are missing.
  // Layout propagation pass will activated.
  getOperation()->walk([](Operation *op) {
    for (OpResult result : op->getOpResults()) {
      std::string name = xegpu::getLayoutName(result);
      if (auto layout = op->getAttrOfType<xegpu::LayoutAttr>(name)) {
        op->removeAttr(name);
        if (!isa<scf::IfOp, scf::ForOp, scf::WhileOp, scf::ConditionOp>(op)) {
          if (auto newLayout = layout.dropSgLayoutAndData())
            op->setAttr(name, newLayout);
        }
      }
    }
  });
}
