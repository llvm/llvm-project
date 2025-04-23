//===- XeGPUUnroll.cpp - patterns to do unrolling ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <numeric>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUUNROLL
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

static const char *const packAttrName = "__xetile_blocking_pack__";
static const char *const unpackAttrName = "__xetile_blocking_unpack__";
static const char *const blockAttrName = "__xetile_blocking_inner_block__";

// emulate the the unpack behavior using insert_strided_slice for VectorType
// values and unrealized_conversion_cast for TileType values.
static Value addUnpackOp(ValueRange srcs, Type destTy,
                         llvm::ArrayRef<int64_t> innerBlock, Location loc,
                         PatternRewriter &rewriter) {
  if (auto vecTy = dyn_cast<VectorType>(destTy)) {
    assert(vecTy.getRank() == 2 && innerBlock.size() == 2 &&
           "Expecting innerBlock size to match the rank of destTy.");
    auto shape = vecTy.getShape();
    auto zeroAttr = rewriter.getZeroAttr(vecTy.getElementType());

    Value result = rewriter.create<arith::ConstantOp>(
        loc, vecTy, DenseElementsAttr::get(vecTy, zeroAttr));
    int64_t idx = 0;
    for (int64_t i = 0; i < shape[0]; i += innerBlock[0]) {
      for (int64_t j = 0; j < shape[1]; j += innerBlock[1]) {
        result = rewriter.create<vector::InsertStridedSliceOp>(
            loc, srcs[idx++], result, llvm::ArrayRef<int64_t>({i, j}),
            llvm::ArrayRef<int64_t>({1, 1}));
      }
    }
    return result;

  } else if (isa<xegpu::TensorDescType>(destTy)) {
    auto attr = NamedAttribute(rewriter.getStringAttr(unpackAttrName),
                               rewriter.getUnitAttr());
    auto innerBlkAttr =
        NamedAttribute(rewriter.getStringAttr(blockAttrName),
                       rewriter.getDenseI64ArrayAttr(innerBlock));
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, destTy, srcs,
        llvm::ArrayRef<NamedAttribute>({attr, innerBlkAttr}));
    return castOp.getResult(0);
  }

  llvm_unreachable("Unexpected destTy.");
  return Value();
}

// emulate the the pack behavior using extract_strided_slice for VectorType
// values and unrealized_conversion_cast for TensorDescType values.
static llvm::SmallVector<Value> addPackOp(Value src, TypeRange destTypes,
                                          llvm::ArrayRef<int64_t> innerBlock,
                                          Location loc,
                                          PatternRewriter &rewriter) {
  if (auto vecTy = dyn_cast<VectorType>(src.getType())) {
    assert(vecTy.getRank() == 2 && innerBlock.size() == 2 &&
           "Expecting innerBlock size to match the rank of src.");
    auto shape = vecTy.getShape();
    llvm::SmallVector<Value> results;
    for (int64_t i = 0; i < shape[0]; i += innerBlock[0]) {
      for (int64_t j = 0; j < shape[1]; j += innerBlock[1]) {
        auto slice = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, src, llvm::ArrayRef<int64_t>({i, j}), innerBlock,
            llvm::ArrayRef<int64_t>({1, 1}));
        results.push_back(slice);
      }
    }
    return results;
  } else if (isa<xegpu::TensorDescType>(src.getType())) {
    auto attr = NamedAttribute(rewriter.getStringAttr(packAttrName),
                               rewriter.getUnitAttr());
    auto innerBlkAttr =
        NamedAttribute(rewriter.getStringAttr(blockAttrName),
                       rewriter.getDenseI64ArrayAttr(innerBlock));
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        loc, destTypes, src,
        llvm::ArrayRef<NamedAttribute>({attr, innerBlkAttr}));
    return castOp.getResults();
  }

  llvm_unreachable("Unexpected src type.");
  return llvm::SmallVector<Value>();
}

template <typename SourceOp>
struct UnrollPattern : public OpRewritePattern<SourceOp> {
  UnrollPattern(MLIRContext *context,
                const vector::UnrollVectorOptions &options,
                PatternBenefit benefit = 1)
      : OpRewritePattern<SourceOp>(context, benefit), options(options) {}

protected:
  std::optional<SmallVector<int64_t>>
  getTargetShape(const vector::UnrollVectorOptions &options,
                 Operation *op) const {
    LDBG("");
    LDBG("Get unroll shape for: " << *op);
    assert(options.nativeShape &&
           "expects the native shape for native shape call back function.");
    auto nativeShape = options.nativeShape(op);
    return nativeShape;
  }

  std::optional<SmallVector<int64_t>>
  computeGrids(llvm::ArrayRef<int64_t> shape,
               llvm::ArrayRef<int64_t> subShape) const {
    // if the shape == subshape, we don't need to unroll.
    if (shape == subShape)
      return std::nullopt;
    return computeShapeRatio(shape, subShape);
  }

  bool isUnrollable(Attribute attr) const {
    auto layout = dyn_cast_if_present<xegpu::LayoutAttr>(attr);
    return layout && layout.isSgLayout() && layout.getInstData() != nullptr;
  }

  xegpu::LayoutAttr getLaneLayoutAttr(Attribute attr) const {
    auto layout = dyn_cast_if_present<xegpu::LayoutAttr>(attr);
    if (!layout || layout.getLaneLayout() == nullptr)
      return xegpu::LayoutAttr();
    return xegpu::LayoutAttr::get(
        layout.getContext(), nullptr /* sg_layout */, nullptr /* sg_data */,
        nullptr /* inst_data */, layout.getLaneLayout(), layout.getLaneData(),
        layout.getOrder());
  }

  vector::UnrollVectorOptions options;
};

struct UnrollCreateNdOp : public UnrollPattern<xegpu::CreateNdDescOp> {
  using UnrollPattern<xegpu::CreateNdDescOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto tdescTy = op.getType();
    auto shape = tdescTy.getShape();
    auto layout = tdescTy.getLayout();

    if (!isUnrollable(layout))
      return failure();

    auto maybeTargetShape = getTargetShape(options, op);
    if (!maybeTargetShape)
      return failure();
    auto targetShape = *maybeTargetShape;

    auto maybeGrids = computeGrids(shape, targetShape);
    if (!maybeGrids)
      return failure();
    auto grids = *maybeGrids;

    auto encoding = tdescTy.getEncoding();
    auto newLayout = getLaneLayoutAttr(layout);
    auto newTdescTy = xegpu::TensorDescType::get(
        ctx, targetShape, tdescTy.getElementType(), encoding, newLayout);

    auto addi = [&](OpFoldResult a, int64_t b) -> Value {
      auto maybeInt = getConstantIntValue(a);
      if (maybeInt) {
        return rewriter.create<arith::ConstantIndexOp>(loc, *maybeInt + b);
      } else {
        auto aV = llvm::cast<Value>(a);
        auto bV = rewriter.create<arith::ConstantIndexOp>(loc, b);
        return rewriter.createOrFold<arith::AddIOp>(loc, aV, bV);
      }
    };

    auto mixedOffsets = op.getMixedOffsets();
    // For n-D memrefs where n > 2, we need to handle the last two
    // dimensions, and keep the first n-2 dimensions as is.
    int64_t x = mixedOffsets.size() - 2;
    int64_t y = mixedOffsets.size() - 1;
    OpFoldResult oldX = mixedOffsets[x];
    OpFoldResult oldY = mixedOffsets[y];

    SmallVector<Value> newOps;
    for (int64_t i = 0; i < grids[0]; i++) {
      for (int64_t j = 0; j < grids[1]; j++) {
        auto subOffX = targetShape[0] * i;
        auto subOffY = targetShape[1] * j;
        mixedOffsets[x] = addi(oldX, subOffX);
        mixedOffsets[y] = addi(oldY, subOffY);
        auto newOp = rewriter.create<xegpu::CreateNdDescOp>(
          loc, newTdescTy, op.getSource(), mixedOffsets, op.getMixedSizes(), op.getMixedStrides());
        newOps.push_back(newOp);
      }
    }
    auto castOp = addUnpackOp(newOps, tdescTy, targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);

    return success();
  }
};

struct UnrollPrefetchNdOp : public UnrollPattern<xegpu::PrefetchNdOp> {
  using UnrollPattern<xegpu::PrefetchNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::PrefetchNdOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct UnrollLoadNdOp : public UnrollPattern<xegpu::LoadNdOp> {
  using UnrollPattern<xegpu::LoadNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadNdOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto valueTy = op.getType();
    auto tdescTy = op.getTensorDescType();
    auto layout = tdescTy.getLayout();

    if (!isUnrollable(layout))
      return failure();

    auto maybeTargetShape = getTargetShape(options, op);
    if (!maybeTargetShape)
      return failure();
    auto targetShape = *maybeTargetShape;

    auto maybeGrids = computeGrids(tdescTy.getShape(), targetShape);
    if (!maybeGrids)
      return failure();
    auto grids = *maybeGrids;

    auto elemTy = tdescTy.getElementType();
    auto newValueTy = valueTy.cloneWith(targetShape, elemTy);
    auto newTdescTy = xegpu::TensorDescType::get(ctx, targetShape, elemTy,
                                                 tdescTy.getEncoding(),
                                                 getLaneLayoutAttr(layout));

    auto numNewOps = computeProduct(grids);
    llvm::SmallVector<Type> convertedTdescTypes(numNewOps, newTdescTy);
    auto convertedTdescs = addPackOp(op.getTensorDesc(), convertedTdescTypes,
                                     targetShape, loc, rewriter);

    llvm::SmallVector<Value> newOps;
    for (auto t : convertedTdescs) {
      auto newOp =
          rewriter.create<xegpu::LoadNdOp>(loc, newValueTy, t, op->getAttrs());
      newOps.push_back(newOp);
    }

    auto castOp = addUnpackOp(newOps, op.getType(), targetShape, loc, rewriter);

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollStoreNdOp : public UnrollPattern<xegpu::StoreNdOp> {
  using UnrollPattern<xegpu::StoreNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreNdOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto valueTy = op.getValueType();
    auto tdescTy = op.getTensorDescType();
    auto layout = tdescTy.getLayout();

    if (!isUnrollable(layout))
      return failure();

    auto maybeTargetShape = getTargetShape(options, op);
    if (!maybeTargetShape)
      return failure();
    auto targetShape = *maybeTargetShape;

    auto maybeGrids = computeGrids(tdescTy.getShape(), targetShape);
    if (!maybeGrids)
      return failure();
    auto grids = *maybeGrids;

    auto elemTy = tdescTy.getElementType();
    auto newValueTy = valueTy.cloneWith(targetShape, elemTy);
    auto newTdescTy = xegpu::TensorDescType::get(ctx, targetShape, elemTy, tdescTy.getEncoding(),
        getLaneLayoutAttr(layout));

    auto numNewOps = computeProduct(grids);
    llvm::SmallVector<Type> convertedValTypes(numNewOps, newValueTy);
    llvm::SmallVector<Type> convertedTdescTypes(numNewOps, newTdescTy);
    auto convertedValues = addPackOp(op.getValue(), convertedValTypes, targetShape, loc, rewriter);
    auto convertedTdescs = addPackOp(op.getTensorDesc(), convertedTdescTypes,
                                     targetShape, loc, rewriter);

    for (auto [v, t] : llvm::zip(convertedValues, convertedTdescs)) {
      rewriter.create<xegpu::StoreNdOp>(loc, v, t, op.getL1HintAttr(),
                                           op.getL2HintAttr(),
                                           op.getL3HintAttr());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollUpdateNdOffsetOp : public UnrollPattern<xegpu::UpdateNdOffsetOp> {
  using UnrollPattern<xegpu::UpdateNdOffsetOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::UpdateNdOffsetOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct UnrollCreateDescOp : public UnrollPattern<xegpu::CreateDescOp> {
  using UnrollPattern<xegpu::CreateDescOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::CreateDescOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct UnrollPrefetchOp : public UnrollPattern<xegpu::PrefetchOp> {
  using UnrollPattern<xegpu::PrefetchOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::PrefetchOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct UnrollLoadOp : public UnrollPattern<xegpu::LoadGatherOp> {
  using UnrollPattern<xegpu::LoadGatherOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadGatherOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct UnrollStoreOp : public UnrollPattern<xegpu::StoreScatterOp> {
  using UnrollPattern<xegpu::StoreScatterOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreScatterOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct UnrollUpdateOffsetOp : public UnrollPattern<xegpu::UpdateOffsetOp> {
  using UnrollPattern<xegpu::UpdateOffsetOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::UpdateOffsetOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct UnrollDpasOp : public UnrollPattern<xegpu::DpasOp> {
  using UnrollPattern<xegpu::DpasOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::DpasOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct UnrollAtomicRMWOp : public UnrollPattern<xegpu::AtomicRMWOp> {
  using UnrollPattern<xegpu::AtomicRMWOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct XeGPUUnrollPass final
    : public xegpu::impl::XeGPUUnrollBase<XeGPUUnrollPass> {
  XeGPUUnrollPass() = default;
  XeGPUUnrollPass(const XeGPUUnrollPass &pass) = default;

  void runOnOperation() override {
    vector::UnrollVectorOptions options;
    options.setNativeShapeFn(
        [&](Operation *op) -> std::optional<SmallVector<int64_t>> {
          if (isa<xegpu::CreateNdDescOp, xegpu::LoadNdOp, xegpu::StoreNdOp>(op)) {
            xegpu::TensorDescType tdescTy;
            if (auto createNdOp = dyn_cast<xegpu::CreateNdDescOp>(op)) {
              tdescTy = createNdOp.getType();
            } else if (auto loadNdOp = dyn_cast<xegpu::LoadNdOp>(op)) {
              tdescTy = loadNdOp.getTensorDescType();
            } else if (auto storeNdOp = dyn_cast<xegpu::StoreNdOp>(op)) {
              tdescTy = storeNdOp.getTensorDescType();
            }

            if (auto layout = tdescTy.getLayoutAttr()) {
              if (auto inst_data = layout.getInstData())
                return SmallVector<int64_t>(inst_data.asArrayRef().begin(),
                                            inst_data.asArrayRef().end());
            }
          }

          return std::nullopt;
        });

    auto funcOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<UnrollCreateNdOp, UnrollLoadNdOp, UnrollStoreNdOp>(
        patterns.getContext(), options);

    // GreedyRewriteConfig config;
    // config.fold = false;
    // config.cseConstants = false;
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
    return;
  }
};
} // namespace

void mlir::xegpu::populateXeGPUUnrollPatterns(
    RewritePatternSet &patterns, const mlir::vector::UnrollVectorOptions &options) {
  patterns.add<UnrollCreateNdOp, UnrollLoadNdOp, UnrollStoreNdOp>(
        patterns.getContext(), options);
}
