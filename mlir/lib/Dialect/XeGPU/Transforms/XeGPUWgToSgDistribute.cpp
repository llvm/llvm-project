//===- XeGPUWgToSgDistribute.cpp - XeGPU Workgroup to Subgroup Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUWGTOSGDISTRIBUTE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

using namespace mlir;

namespace {

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

  // Calculate offset for each subgroup
  SmallVector<OpFoldResult>
  calculateGlobalOffsets(ConversionPatternRewriter &rewriter, Location loc,
                         const SmallVector<OpFoldResult> &originalOffsets,
                         const SmallVector<Value> &localOffset,
                         const SmallVector<int64_t> &distUnitBaseAddr,
                         const SmallVector<int64_t> &distUnitShape) const {
    assert(localOffset.size() == distUnitBaseAddr.size() &&
           "localOffset and distUnitBaseAddr must have the same rank");

    SmallVector<OpFoldResult> globalOffsets(originalOffsets.begin(),
                                            originalOffsets.end());
    size_t rank = localOffset.size();
    for (size_t i = 0; i < rank; ++i) {
      size_t dimIdx = originalOffsets.size() - rank + i;
      Value constOffset =
          rewriter.create<arith::ConstantIndexOp>(loc, distUnitBaseAddr[i]);
      Value offset =
          rewriter.createOrFold<index::AddOp>(loc, localOffset[i], constOffset);
      Value modValue =
          rewriter.create<arith::ConstantIndexOp>(loc, distUnitShape[i]);
      Value offsetMod =
          rewriter.createOrFold<index::RemUOp>(loc, offset, modValue);
      Value origOffset = getValueOrCreateConstantIndexOp(
          rewriter, loc, originalOffsets[dimIdx]);
      Value globalOffset =
          rewriter.createOrFold<index::AddOp>(loc, origOffset, offsetMod);
      globalOffsets[dimIdx] = globalOffset;
    }

    return globalOffsets;
  }

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    xegpu::TensorDescType tdescTy = op.getType();
    auto layout = dyn_cast<xegpu::LayoutAttr>(tdescTy.getLayout());
    if (!layout)
      return failure();
    Type elemTy = tdescTy.getElementType();
    ArrayRef<int64_t> wgShape = tdescTy.getShape();
    // sgLayout must be present for workgroup-level distribution.
    SmallVector<int64_t> sgLayout;
    if (auto sgLayoutAttr = layout.getSgLayout())
      sgLayout = llvm::to_vector_of<int64_t>(sgLayoutAttr.asArrayRef());
    else
      return rewriter.notifyMatchFailure(
          op, "sgLayout attribute is required in layout");

    SmallVector<int64_t> sgShape;
    if (auto sgDataAttr = layout.getSgData()) {
      sgShape = llvm::to_vector_of<int64_t>(sgDataAttr.asArrayRef());
    } else {
      assert(wgShape.size() == sgLayout.size() &&
             "sgLayout and wgShape must have the same rank");
      sgShape.reserve(wgShape.size());
      for (size_t i = 0; i < wgShape.size(); ++i) {
        assert(sgLayout[i] != 0 && "sgLayout elements must be non-zero");
        sgShape.push_back(wgShape[i] / sgLayout[i]);
      }
    }

    // TODO : Handle order attribute
    // Get the subgroup ID
    auto linearSgId =
        rewriter.create<gpu::SubgroupIdOp>(loc, /*upper_bound=*/nullptr);

    // Create constants for layout dimensions
    SmallVector<Value> sgLayoutDim(sgLayout.size());
    SmallVector<Value> sgDataDim(sgShape.size());

    for (size_t i = 0; i < sgLayout.size(); i++) {
      sgLayoutDim[i] =
          rewriter.create<arith::ConstantIndexOp>(loc, sgLayout[i]);
      sgDataDim[i] = rewriter.create<arith::ConstantIndexOp>(loc, sgShape[i]);
    }

    auto deLinearizeSgId =
        affine::delinearizeIndex(rewriter, loc, linearSgId, sgLayoutDim);
    if (failed(deLinearizeSgId))
      return failure();
    SmallVector<Value> sgIds = *deLinearizeSgId;

    // Calculate distribution unit shape and local offsets for subgroup
    SmallVector<int64_t> distUnitShape(sgLayout.size());
    SmallVector<Value> localOffset(sgLayout.size());
    for (size_t i = 0; i < sgLayout.size(); i++) {
      distUnitShape[i] = std::min(sgLayout[i] * sgShape[i], wgShape[i]);
      localOffset[i] =
          rewriter.createOrFold<index::MulOp>(loc, sgIds[i], sgDataDim[i]);
    }

    SmallVector<OpFoldResult> originalOffsets = op.getMixedOffsets();

    xegpu::TensorDescType newTdescTy =
        xegpu::TensorDescType::get(ctx, sgShape, elemTy, tdescTy.getEncoding(),
                                   layout.dropSgLayoutAndData());
    SmallVector<Value> newCreateNdOps;
    for (SmallVector<int64_t> distUnitBaseAddr :
         StaticTileOffsetRange(wgShape, distUnitShape)) {
      SmallVector<OpFoldResult> globalOffsets =
          calculateGlobalOffsets(rewriter, loc, originalOffsets, localOffset,
                                 distUnitBaseAddr, distUnitShape);

      auto newCreateNdOp = rewriter.create<xegpu::CreateNdDescOp>(
          loc, newTdescTy, op.getSource(), globalOffsets, op.getMixedSizes(),
          op.getMixedStrides());
      newCreateNdOps.push_back(newCreateNdOp);
    }

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
    SmallVector<Value> newLoadOps;
    for (auto src : adaptor.getTensorDesc()) {
      xegpu::TensorDescType tdescTy =
          dyn_cast<xegpu::TensorDescType>(src.getType());
      ArrayRef<int64_t> srcShape = tdescTy.getShape();
      VectorType newResTy = VectorType::get(srcShape, tdescTy.getElementType());
      auto newLoadOp = rewriter.create<xegpu::LoadNdOp>(op.getLoc(), newResTy,
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
    for (auto [v, t] : llvm::zip(adaptor.getValue(), adaptor.getTensorDesc()))
      rewriter.create<xegpu::StoreNdOp>(op.getLoc(), v, t, op.getL1HintAttr(),
                                        op.getL2HintAttr(), op.getL3HintAttr());

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
      auto newUpdateTileOffsetOp = rewriter.create<xegpu::UpdateNdOffsetOp>(
          op.getLoc(), tDesc.getType(), tDesc, op.getOffsets(),
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

    auto originalLayout =
        llvm::dyn_cast_or_null<xegpu::LayoutAttr>(op->getAttr("layout"));
    if (!originalLayout)
      return failure();

    SmallVector<Value> newDpasOps;
    size_t i = 0;
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
        tmpC = rewriter.create<xegpu::DpasOp>(
            loc, resTy, operands,
            llvm::ArrayRef<NamedAttribute>(
                {"layout_result_0", originalLayout.dropSgLayoutAndData()}));
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
    for (auto src : adaptor.getTensorDesc())
      rewriter.create<xegpu::PrefetchNdOp>(op.getLoc(), TypeRange(), src,
                                           op->getAttrs());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir {
namespace xegpu {
void populateXeGPUWgToSgDistributePatterns(RewritePatternSet &patterns) {
  patterns.add<WgToSgCreateNdOp, WgToSgLoadNdOp, WgToSgStoreNdOp,
               WgToSgUpdateNdOffsetOp, WgToSgDpasOp, WgToSgPrefetchNdOp>(
      patterns.getContext());
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
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ConversionTarget target(*ctx);

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

  auto isLegal = [&](xegpu::LayoutAttr layout) -> bool {
    return !layout || layout.getSgLayout() == nullptr;
  };

  target.addDynamicallyLegalOp<xegpu::CreateNdDescOp, xegpu::LoadNdOp,
                               xegpu::StoreNdOp, xegpu::UpdateNdOffsetOp,
                               xegpu::PrefetchNdOp>([=](Operation *op) -> bool {
    auto tdescTy = getTensorDescType(op);
    auto layout = dyn_cast_or_null<xegpu::LayoutAttr>(tdescTy.getLayout());
    return isLegal(layout);
  });

  target.addDynamicallyLegalOp<xegpu::DpasOp>([=](xegpu::DpasOp op) -> bool {
    auto layout = dyn_cast_or_null<xegpu::LayoutAttr>(op->getAttr("layout"));
    return isLegal(layout);
  });

  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  xegpu::populateXeGPUWgToSgDistributePatterns(patterns);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
