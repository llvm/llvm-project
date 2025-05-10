//===- XeGPUWgToSg.cpp - XeGPU WorkGroup to Subgroup Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <numeric>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUWGTOSG
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-wg-to-sg"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

// clang-format off
/// This pattern transform the CreateNdDescOp to create a subgroup descriptor
/// from a workgroup descriptor. It replaces the offsets and sizes with
/// appropriate values for the subgroup.
/// It uses round-robin distribution to create the subgroup descriptor.

/// Following create_nd_desc operation:,
///    %tdesc = xegpu.create_nd_tdesc %src[0, 0] : memref<24x24xf32>
///       -> !xegpu.tensor_desc<24x24xf32, #xegpu.layout<sg_layout = [4, 4],
///           sg_data = [2, 2], lane_layout = [2, 2], lane_data = [1, 1]>>
/// is converted to 9 subgroup level operations based on the sg_layout & sg_data:
///    %tdesc = xegpu.create_nd_tdesc %src[off1, off2] : memref<24x24xf32> ->
///           !xegpu.tensor_desc<2x2xf32, #xegpu.layout<lane_layout = [2, 2], lane_data = [1, 1]>>
///
/// The sg_layout and sg_data are dropped from the layout attribute as they are no longer needed.
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
/// Since the 24x24 matrix is divided into 8x8 distribution units, there will be 9
/// distribution units (3x3) in total. Hence the 9 subgroup level operations.
/// Each 8x8 matrix within the 24x24 matrix is called a distribution unit.
// clang-format on
struct WgToSgCreateNdOp : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;

  // Helper to extract mixed offsets into a Value array
  SmallVector<Value> extractOffsets(ConversionPatternRewriter &rewriter,
                                    xegpu::CreateNdDescOp op) const {
    llvm::SmallVector<Value> offsets;
    auto staticOffsets = op.getStaticOffsets();
    auto dynamicOffsets = op.getOffsets();

    for (size_t i = 0, j = 0; i != staticOffsets.size(); i++) {
      if (ShapedType::isDynamic(staticOffsets[i])) {
        offsets.push_back(dynamicOffsets[j++]);
      } else {
        offsets.push_back(rewriter.create<arith::ConstantIndexOp>(
            op.getLoc(), staticOffsets[i]));
      }
    }
    return offsets;
  }

  // Convert linear subgroup ID to 2D coordinates
  // TODO: Delinearize for nD
  SmallVector<Value> delinearizeSubgroupId(ConversionPatternRewriter &rewriter,
                                           Location loc, Value sgID,
                                           Value sgDimX, Value sgDimY) const {
    return {rewriter.create<index::DivUOp>(loc, sgID, sgDimY),
            rewriter.create<index::RemUOp>(loc, sgID, sgDimY)};
  }

  // Create a constant index value
  Value createConstantIndex(ConversionPatternRewriter &rewriter, Location loc,
                            int64_t value) const {
    return rewriter.create<arith::ConstantIndexOp>(loc, value);
  }

  // Calculate global offset for each subgroup
  SmallVector<OpFoldResult>
  calculateGlobalOffsets(ConversionPatternRewriter &rewriter, Location loc,
                         const SmallVector<Value> &originalOffsets,
                         const SmallVector<Value> &localOffset,
                         const SmallVector<int64_t> &distUnitBaseAddr) const {

    Value constOffsetX =
        createConstantIndex(rewriter, loc, distUnitBaseAddr[0]);
    Value constOffsetY =
        createConstantIndex(rewriter, loc, distUnitBaseAddr[1]);

    // Compute offsets within entire tile
    Value offsetX =
        rewriter.createOrFold<index::AddOp>(loc, localOffset[0], constOffsetX);
    Value offsetY =
        rewriter.createOrFold<index::AddOp>(loc, localOffset[1], constOffsetY);

    // Add to global offsets
    size_t lastDimIndex = originalOffsets.size() - 1;
    size_t secondLastDimIndex = lastDimIndex - 1;

    Value globalOffsetX = rewriter.createOrFold<index::AddOp>(
        loc, originalOffsets[secondLastDimIndex], offsetX);
    Value globalOffsetY = rewriter.createOrFold<index::AddOp>(
        loc, originalOffsets[lastDimIndex], offsetY);

    // Create final offset list
    SmallVector<OpFoldResult> globalOffsets(originalOffsets.begin(),
                                            originalOffsets.end());
    globalOffsets[secondLastDimIndex] = globalOffsetX;
    globalOffsets[lastDimIndex] = globalOffsetY;

    return globalOffsets;
  }

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    xegpu::TensorDescType tdescTy = op.getType();
    auto layout = dyn_cast<xegpu::LayoutAttr>(tdescTy.getLayout());
    Type elemTy = tdescTy.getElementType();
    ArrayRef<int64_t> wgShape = tdescTy.getShape();
    ArrayRef<int64_t> sgShape =
        llvm::to_vector_of<int64_t>(layout.getSgData().asArrayRef());
    ArrayRef<int64_t> sgLayout =
        llvm::to_vector_of<int64_t>(layout.getSgLayout().asArrayRef());

    // Get the subgroup ID
    auto linearSgId = rewriter.create<gpu::SubgroupIdOp>(loc, nullptr);

    // Create constants for layout dimensions
    SmallVector<Value> sgLayoutDim(sgLayout.size());
    SmallVector<Value> sgDataDim(sgShape.size());

    for (size_t i = 0; i < sgLayout.size(); i++) {
      sgLayoutDim[i] = createConstantIndex(rewriter, loc, sgLayout[i]);
      sgDataDim[i] = createConstantIndex(rewriter, loc, sgShape[i]);
    }

    // Delinearize the 1D subgroup id into nd coordinates
    SmallVector<Value> sgIds = delinearizeSubgroupId(
        rewriter, loc, linearSgId, sgLayoutDim[0], sgLayoutDim[1]);

    // Calculate distribution unit shape and local offsets for subgroup
    SmallVector<int64_t> distUnitShape(sgLayout.size());
    SmallVector<Value> localOffset(sgLayout.size());
    for (size_t i = 0; i < sgLayout.size(); i++) {
      distUnitShape[i] = sgLayout[i] * sgShape[i];
      localOffset[i] =
          rewriter.createOrFold<index::MulOp>(loc, sgIds[i], sgDataDim[i]);
    }

    SmallVector<Value> originalOffsets = extractOffsets(rewriter, op);

    xegpu::TensorDescType newTdescTy =
        xegpu::TensorDescType::get(ctx, sgShape, elemTy, tdescTy.getEncoding(),
                                   layout.dropSgLayoutAndData());
    SmallVector<Value> newCreateNdOps;
    for (const SmallVector<int64_t> &distUnitBaseAddr :
         StaticTileOffsetRange(wgShape, distUnitShape)) {
      SmallVector<OpFoldResult> globalOffsets = calculateGlobalOffsets(
          rewriter, loc, originalOffsets, localOffset, distUnitBaseAddr);

      auto newCreateNdOp = rewriter.create<xegpu::CreateNdDescOp>(
          loc, newTdescTy, op.getSource(), globalOffsets, op.getMixedSizes(),
          op.getMixedStrides());
      newCreateNdOps.push_back(newCreateNdOp);
    }

    rewriter.replaceOpWithMultiple(op, {newCreateNdOps});
    return success();
  }
};

/// This pattern transforms the LoadNdOp to load from a subgroup descriptor
/// It creates a LoadNdOp op to load the new subgroup src tensor descriptors.
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
                {"layout", originalLayout.dropSgLayoutAndData()}));
        newDpasOps.push_back(tmpC);
      }
    }
    rewriter.replaceOpWithMultiple(op, {newDpasOps});
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace xegpu {
void populateXeGPUWgToSgPatterns(RewritePatternSet &patterns) {
  patterns.add<WgToSgCreateNdOp, WgToSgLoadNdOp, WgToSgStoreNdOp,
               WgToSgUpdateNdOffsetOp, WgToSgDpasOp>(patterns.getContext());
}
} // namespace xegpu
} // namespace mlir

namespace {
struct XeGPUWgToSgPass : public xegpu::impl::XeGPUWgToSgBase<XeGPUWgToSgPass> {
  void runOnOperation() override;
};
} // namespace

void XeGPUWgToSgPass::runOnOperation() {
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
    return xegpu::TensorDescType();
  };

  auto isLegal = [&](xegpu::LayoutAttr layout) -> bool {
    return !layout || layout.getSgLayout() == nullptr;
  };

  target.addDynamicallyLegalOp<xegpu::CreateNdDescOp, xegpu::LoadNdOp,
                               xegpu::StoreNdOp, xegpu::UpdateNdOffsetOp>(
      [=](Operation *op) -> bool {
        auto tdescTy = getTensorDescType(op);
        auto layout = dyn_cast_or_null<xegpu::LayoutAttr>(tdescTy.getLayout());
        return isLegal(layout);
      });

  target.addDynamicallyLegalOp<xegpu::DpasOp>([=](xegpu::DpasOp op) -> bool {
    auto layout = dyn_cast_or_null<xegpu::LayoutAttr>(op->getAttr("layout"));
    return isLegal(layout);
  });

  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  xegpu::populateXeGPUWgToSgPatterns(patterns);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
