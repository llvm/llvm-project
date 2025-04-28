//===----- FlattenMemRefs.cpp - MemRef ops flattener pass  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns for flattening an multi-rank memref-related
// ops into 1-d memref ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_FLATTENMEMREFSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

static void setInsertionPointToStart(OpBuilder &builder, Value val) {
  if (auto *parentOp = val.getDefiningOp()) {
    builder.setInsertionPointAfter(parentOp);
  } else {
    builder.setInsertionPointToStart(val.getParentBlock());
  }
}

OpFoldResult computeMemRefSpan(Value memref, OpBuilder &builder) {
  Location loc = memref.getLoc();
  MemRefType type = cast<MemRefType>(memref.getType());
  ArrayRef<int64_t> shape = type.getShape();
  
  // Check for empty memref
  if (type.hasStaticShape() && 
      llvm::any_of(shape, [](int64_t dim) { return dim == 0; })) {
    return builder.getIndexAttr(0);
  }
  
  // Get strides of the memref
  SmallVector<int64_t, 4> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset))) {
    // Cannot extract strides, return a dynamic value
    return Value();
  }
  
  // Static case: compute at compile time if possible
  if (type.hasStaticShape()) {
    int64_t span = 0;
    for (unsigned i = 0; i < type.getRank(); ++i) {
      span += (shape[i] - 1) * strides[i];
    }
    return builder.getIndexAttr(span);
  }
  
  // Dynamic case: emit IR to compute at runtime
  Value result = builder.create<arith::ConstantIndexOp>(loc, 0);
  
  for (unsigned i = 0; i < type.getRank(); ++i) {
    // Get dimension size
    Value dimSize;
    if (shape[i] == ShapedType::kDynamic) {
      dimSize = builder.create<memref::DimOp>(loc, memref, i);
    } else {
      dimSize = builder.create<arith::ConstantIndexOp>(loc, shape[i]);
    }
    
    // Compute (dim - 1)
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value dimMinusOne = builder.create<arith::SubIOp>(loc, dimSize, one);
    
    // Get stride
    Value stride;
    if (strides[i] == ShapedType::kDynamicStrideOrOffset) {
      // For dynamic strides, need to extract from memref descriptor
      // This would require runtime support, possibly using extractStride
      // As a placeholder, return a dynamic value
      return Value();
    } else {
      stride = builder.create<arith::ConstantIndexOp>(loc, strides[i]);
    }
    
    // Add (dim - 1) * stride to result
    Value term = builder.create<arith::MulIOp>(loc, dimMinusOne, stride);
    result = builder.create<arith::AddIOp>(loc, result, term);
  }
  
  return result;
}

static std::tuple<Value, OpFoldResult, SmallVector<OpFoldResult>, OpFoldResult,
                  OpFoldResult>
getFlatOffsetAndStrides(OpBuilder &rewriter, Location loc, Value source,
                        ArrayRef<OpFoldResult> subOffsets,
                        ArrayRef<OpFoldResult> subStrides = std::nullopt) {
  auto sourceType = cast<MemRefType>(source.getType());
  auto sourceRank = static_cast<unsigned>(sourceType.getRank());

  memref::ExtractStridedMetadataOp newExtractStridedMetadata;
  {
    OpBuilder::InsertionGuard g(rewriter);
    setInsertionPointToStart(rewriter, source);
    newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, source);
  }

  auto &&[sourceStrides, sourceOffset] = sourceType.getStridesAndOffset();

  auto getDim = [&](int64_t dim, Value dimVal) -> OpFoldResult {
    return ShapedType::isDynamic(dim) ? getAsOpFoldResult(dimVal)
                                      : rewriter.getIndexAttr(dim);
  };

  OpFoldResult origOffset =
      getDim(sourceOffset, newExtractStridedMetadata.getOffset());
  ValueRange sourceStridesVals = newExtractStridedMetadata.getStrides();
  OpFoldResult outmostDim =
      getDim(sourceType.getShape().front(),
             newExtractStridedMetadata.getSizes().front());

  SmallVector<OpFoldResult> origStrides;
  origStrides.reserve(sourceRank);

  SmallVector<OpFoldResult> strides;
  strides.reserve(sourceRank);

  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
  for (auto i : llvm::seq(0u, sourceRank)) {
    OpFoldResult origStride = getDim(sourceStrides[i], sourceStridesVals[i]);

    if (!subStrides.empty()) {
      strides.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 * s1, {subStrides[i], origStride}));
    }

    origStrides.emplace_back(origStride);
  }

  // Compute linearized index:
  auto &&[expr, values] =
      computeLinearIndex(rewriter.getIndexAttr(0), origStrides, subOffsets);
  OpFoldResult linearizedIndex =
      affine::makeComposedFoldedAffineApply(rewriter, loc, expr, values);

  // Compute collapsed size: (the outmost stride * outmost dimension).
  //SmallVector<OpFoldResult> ops{origStrides.front(), outmostDim};
  //OpFoldResult collapsedSize = affine::computeProduct(loc, rewriter, ops);
  OpFoldResult collapsedSize = computeMemRefSpan(source, rewriter);

  return {newExtractStridedMetadata.getBaseBuffer(), linearizedIndex,
          origStrides, origOffset, collapsedSize};
}

static Value getValueFromOpFoldResult(OpBuilder &rewriter, Location loc,
                                      OpFoldResult in) {
  if (Attribute offsetAttr = dyn_cast<Attribute>(in)) {
    return rewriter.create<arith::ConstantIndexOp>(
        loc, cast<IntegerAttr>(offsetAttr).getInt());
  }
  return cast<Value>(in);
}

/// Returns a collapsed memref and the linearized index to access the element
/// at the specified indices.
static std::pair<Value, Value> getFlattenMemrefAndOffset(OpBuilder &rewriter,
                                                         Location loc,
                                                         Value source,
                                                         ValueRange indices) {
  auto &&[base, index, strides, offset, collapsedShape] =
      getFlatOffsetAndStrides(rewriter, loc, source,
                              getAsOpFoldResult(indices));

  return std::make_pair(
      rewriter.create<memref::ReinterpretCastOp>(
          loc, source,
          /* offset = */ offset,
          /* shapes = */ ArrayRef<OpFoldResult>{collapsedShape},
          /* strides = */ ArrayRef<OpFoldResult>{strides.back()}),
      getValueFromOpFoldResult(rewriter, loc, index));
}

static bool needFlattening(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getRank() > 1;
}

static bool checkLayout(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getLayout().isIdentity() ||
         isa<StridedLayoutAttr>(type.getLayout());
}

namespace {
static Value getTargetMemref(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .template Case<memref::LoadOp, memref::StoreOp>(
          [](auto op) { return op.getMemref(); })
      .template Case<vector::LoadOp, vector::StoreOp, vector::MaskedLoadOp,
                     vector::MaskedStoreOp>(
          [](auto op) { return op.getBase(); })
      .template Case<vector::TransferReadOp, vector::TransferWriteOp>(
          [](auto op) { return op.getSource(); })
      .Default([](auto) { return Value{}; });
}

static void replaceOp(Operation *op, PatternRewriter &rewriter,
                      Value flatMemref, Value offset) {
  auto loc = op->getLoc();
  llvm::TypeSwitch<Operation *>(op)
      .Case<memref::LoadOp>([&](auto op) {
        auto newLoad = rewriter.create<memref::LoadOp>(
            loc, op->getResultTypes(), flatMemref, ValueRange{offset});
        newLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newLoad.getResult());
      })
      .Case<memref::StoreOp>([&](auto op) {
        auto newStore = rewriter.create<memref::StoreOp>(
            loc, op->getOperands().front(), flatMemref, ValueRange{offset});
        newStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newStore);
      })
      .Case<vector::LoadOp>([&](auto op) {
        auto newLoad = rewriter.create<vector::LoadOp>(
            loc, op->getResultTypes(), flatMemref, ValueRange{offset});
        newLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newLoad.getResult());
      })
      .Case<vector::StoreOp>([&](auto op) {
        auto newStore = rewriter.create<vector::StoreOp>(
            loc, op->getOperands().front(), flatMemref, ValueRange{offset});
        newStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newStore);
      })
      .Case<vector::MaskedLoadOp>([&](auto op) {
        auto newMaskedLoad = rewriter.create<vector::MaskedLoadOp>(
            loc, op.getType(), flatMemref, ValueRange{offset}, op.getMask(),
            op.getPassThru());
        newMaskedLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newMaskedLoad.getResult());
      })
      .Case<vector::MaskedStoreOp>([&](auto op) {
        auto newMaskedStore = rewriter.create<vector::MaskedStoreOp>(
            loc, flatMemref, ValueRange{offset}, op.getMask(),
            op.getValueToStore());
        newMaskedStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newMaskedStore);
      })
      .Case<vector::TransferReadOp>([&](auto op) {
        auto newTransferRead = rewriter.create<vector::TransferReadOp>(
            loc, op.getType(), flatMemref, ValueRange{offset}, op.getPadding());
        rewriter.replaceOp(op, newTransferRead.getResult());
      })
      .Case<vector::TransferWriteOp>([&](auto op) {
        auto newTransferWrite = rewriter.create<vector::TransferWriteOp>(
            loc, op.getVector(), flatMemref, ValueRange{offset});
        rewriter.replaceOp(op, newTransferWrite);
      })
      .Default([&](auto op) {
        op->emitOpError("unimplemented: do not know how to replace op.");
      });
}

template <typename T>
struct MemRefRewritePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    Value memref = getTargetMemref(op);
    if (!needFlattening(memref) || !checkLayout(memref))
      return failure();
    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op->getLoc(), memref, op.getIndices());
    replaceOp(op, rewriter, flatMemref, offset);
    return success();
  }
};

struct FlattenSubview : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    Value memref = op.getSource();
    if (!needFlattening(memref))
      return rewriter.notifyMatchFailure(op, "already flattened");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    Location loc = op.getLoc();
    SmallVector<OpFoldResult> subOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> subSizes = op.getMixedSizes();
    SmallVector<OpFoldResult> subStrides = op.getMixedStrides();
    auto &&[base, finalOffset, strides, _, __] =
        getFlatOffsetAndStrides(rewriter, loc, memref, subOffsets, subStrides);

    auto srcType = cast<MemRefType>(memref.getType());
    auto resultType = cast<MemRefType>(op.getType());
    unsigned subRank = static_cast<unsigned>(resultType.getRank());

    llvm::SmallBitVector droppedDims = op.getDroppedDims();

    SmallVector<OpFoldResult> finalSizes;
    finalSizes.reserve(subRank);

    SmallVector<OpFoldResult> finalStrides;
    finalStrides.reserve(subRank);

    for (auto i : llvm::seq(0u, static_cast<unsigned>(srcType.getRank()))) {
      if (droppedDims.test(i))
        continue;

      finalSizes.push_back(subSizes[i]);
      finalStrides.push_back(strides[i]);
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resultType, base, finalOffset, finalSizes, finalStrides);
    return success();
  }
};

struct FlattenMemrefsPass
    : public mlir::memref::impl::FlattenMemrefsPassBase<FlattenMemrefsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    memref::populateFlattenMemrefsPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void memref::populateFlattenMemrefsPatterns(RewritePatternSet &patterns) {
  patterns
      .insert<MemRefRewritePattern<memref::LoadOp>,
              MemRefRewritePattern<memref::StoreOp>,
              MemRefRewritePattern<vector::LoadOp>,
              MemRefRewritePattern<vector::StoreOp>,
              MemRefRewritePattern<vector::TransferReadOp>,
              MemRefRewritePattern<vector::TransferWriteOp>, FlattenSubview>(
          patterns.getContext());
}
