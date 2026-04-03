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
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_FLATTENMEMREFSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

static Value getValueFromOpFoldResult(OpBuilder &rewriter, Location loc,
                                      OpFoldResult in) {
  if (Attribute offsetAttr = dyn_cast<Attribute>(in)) {
    return arith::ConstantIndexOp::create(
        rewriter, loc, cast<IntegerAttr>(offsetAttr).getInt());
  }
  return cast<Value>(in);
}

/// Returns a collapsed memref and the linearized index to access the element
/// at the specified indices.
static std::pair<Value, Value> getFlattenMemrefAndOffset(OpBuilder &rewriter,
                                                         Location loc,
                                                         Value source,
                                                         ValueRange indices) {
  int64_t sourceOffset;
  SmallVector<int64_t, 4> sourceStrides;
  auto sourceType = cast<MemRefType>(source.getType());
  if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset))) {
    assert(false);
  }

  memref::ExtractStridedMetadataOp stridedMetadata =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, source);

  auto typeBit = sourceType.getElementType().getIntOrFloatBitWidth();
  OpFoldResult linearizedIndices;
  memref::LinearizedMemRefInfo linearizedInfo;
  std::tie(linearizedInfo, linearizedIndices) =
      memref::getLinearizedMemRefOffsetAndSize(
          rewriter, loc, typeBit, typeBit,
          stridedMetadata.getConstifiedMixedOffset(),
          stridedMetadata.getConstifiedMixedSizes(),
          stridedMetadata.getConstifiedMixedStrides(),
          getAsOpFoldResult(indices));

  return std::make_pair(
      memref::ReinterpretCastOp::create(
          rewriter, loc, source,
          /* offset = */ linearizedInfo.linearizedOffset,
          /* shapes = */
          ArrayRef<OpFoldResult>{linearizedInfo.linearizedSize},
          /* strides = */
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)}),
      getValueFromOpFoldResult(rewriter, loc, linearizedIndices));
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
      .template Case<memref::LoadOp, memref::StoreOp, memref::AllocaOp,
                     memref::AllocOp>([](auto op) { return op.getMemref(); })
      .template Case<vector::LoadOp, vector::StoreOp, vector::MaskedLoadOp,
                     vector::MaskedStoreOp, vector::TransferReadOp,
                     vector::TransferWriteOp>(
          [](auto op) { return op.getBase(); })
      .Default(nullptr);
}

template <typename T>
static void replaceOp(T op, PatternRewriter &rewriter, Value flatMemref,
                      Value offset) {
  Location loc = op->getLoc();
  llvm::TypeSwitch<Operation *>(op.getOperation())
      .Case([&](memref::LoadOp op) {
        auto newLoad =
            memref::LoadOp::create(rewriter, loc, op->getResultTypes(),
                                   flatMemref, ValueRange{offset});
        newLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newLoad.getResult());
      })
      .Case([&](memref::StoreOp op) {
        auto newStore =
            memref::StoreOp::create(rewriter, loc, op->getOperands().front(),
                                    flatMemref, ValueRange{offset});
        newStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newStore);
      })
      .Case([&](vector::LoadOp op) {
        auto newLoad =
            vector::LoadOp::create(rewriter, loc, op->getResultTypes(),
                                   flatMemref, ValueRange{offset});
        newLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newLoad.getResult());
      })
      .Case([&](vector::StoreOp op) {
        auto newStore =
            vector::StoreOp::create(rewriter, loc, op->getOperands().front(),
                                    flatMemref, ValueRange{offset});
        newStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newStore);
      })
      .Case([&](vector::MaskedLoadOp op) {
        auto newMaskedLoad = vector::MaskedLoadOp::create(
            rewriter, loc, op.getType(), flatMemref, ValueRange{offset},
            op.getMask(), op.getPassThru());
        newMaskedLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newMaskedLoad.getResult());
      })
      .Case([&](vector::MaskedStoreOp op) {
        auto newMaskedStore = vector::MaskedStoreOp::create(
            rewriter, loc, flatMemref, ValueRange{offset}, op.getMask(),
            op.getValueToStore());
        newMaskedStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newMaskedStore);
      })
      .Case([&](vector::TransferReadOp op) {
        auto newTransferRead = vector::TransferReadOp::create(
            rewriter, loc, op.getType(), flatMemref, ValueRange{offset},
            op.getPadding());
        rewriter.replaceOp(op, newTransferRead.getResult());
      })
      .Case([&](vector::TransferWriteOp op) {
        auto newTransferWrite = vector::TransferWriteOp::create(
            rewriter, loc, op.getVector(), flatMemref, ValueRange{offset});
        rewriter.replaceOp(op, newTransferWrite);
      })
      .Default([&](auto op) {
        op->emitOpError("unimplemented: do not know how to replace op.");
      });
}

template <typename T>
static ValueRange getIndices(T op) {
  return op.getIndices();
}

template <typename T>
static LogicalResult canBeFlattened(T op, PatternRewriter &rewriter) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op.getOperation())
      .template Case<vector::TransferReadOp, vector::TransferWriteOp>(
          [&](auto oper) {
            // For vector.transfer_read/write, must make sure:
            // 1. all accesses are inbound, and
            // 2. has an identity or minor identity permutation map.
            auto permutationMap = oper.getPermutationMap();
            if (!permutationMap.isIdentity() &&
                !permutationMap.isMinorIdentity()) {
              return rewriter.notifyMatchFailure(
                  oper, "only identity permutation map is supported");
            }
            mlir::ArrayAttr inbounds = oper.getInBounds();
            if (llvm::any_of(inbounds, [](Attribute attr) {
                  return !cast<BoolAttr>(attr).getValue();
                })) {
              return rewriter.notifyMatchFailure(oper,
                                                 "only inbounds are supported");
            }
            return success();
          })
      .Default([&](auto op) { return success(); });
}

// Pattern for memref::AllocOp and memref::AllocaOp.
//
// The "source" memref for these ops IS the op's own result, so the generic
// MemRefRewritePattern cannot be used: getFlattenMemrefAndOffset would insert
// ExtractStridedMetadataOp and ReinterpretCastOp that use op.result BEFORE op
// in the block. After replaceOpWithNewOp the original result is RAUW'd to the
// new ReinterpretCastOp, leaving the earlier ops with forward references
// (domination violations) caught by MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS.
//
// Instead, sizes and strides are computed from the op's operands and type
// (which all dominate the op), avoiding any reference to op.result until the
// final replaceOpWithNewOp.
template <typename AllocLikeOp>
struct AllocLikeFlattenPattern : public OpRewritePattern<AllocLikeOp> {
  using OpRewritePattern<AllocLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AllocLikeOp op,
                                PatternRewriter &rewriter) const override {
    if (!needFlattening(op.getMemref()) || !checkLayout(op.getMemref()))
      return failure();

    Location loc = op->getLoc();
    auto memrefType = cast<MemRefType>(op.getType());
    auto elemType = memrefType.getElementType();
    if (!elemType.isIntOrFloat())
      return failure();
    unsigned elemBitWidth = elemType.getIntOrFloatBitWidth();

    SmallVector<OpFoldResult> sizes = op.getMixedSizes();

    int64_t staticOffset;
    SmallVector<int64_t> staticStrides;
    if (failed(memrefType.getStridesAndOffset(staticStrides, staticOffset)))
      return failure();
    if (staticOffset == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(op, "dynamic offset not supported");
    SmallVector<OpFoldResult> strides;
    strides.reserve(staticStrides.size());
    for (int64_t stride : staticStrides) {
      if (stride == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(op,
                                           "dynamic stride cannot be computed");
      strides.push_back(rewriter.getIndexAttr(stride));
    }

    // Compute the linearized flat extent from sizes and strides (no SSA ops
    // referencing op.result are created here).
    memref::LinearizedMemRefInfo linearizedInfo;
    OpFoldResult linearizedOffset;
    std::tie(linearizedInfo, linearizedOffset) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, elemBitWidth, elemBitWidth, rewriter.getIndexAttr(0),
            sizes, strides);
    (void)linearizedOffset;

    // The total allocation must cover [0, staticOffset + linearizedExtent).
    // When the offset is non-zero, add it to the computed extent so that the
    // buffer is large enough for elements accessed at positions
    // [staticOffset, staticOffset + linearizedExtent).
    OpFoldResult flatSizeOfr = linearizedInfo.linearizedSize;
    if (staticOffset != 0) {
      AffineExpr s0;
      bindSymbols(rewriter.getContext(), s0);
      flatSizeOfr = affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 + staticOffset, {flatSizeOfr});
    }

    // Build the flat 1-D MemRefType. The linearized size may be static or
    // dynamic (OpFoldResult of either IntegerAttr or a Value).
    int64_t flatDimSize = ShapedType::kDynamic;
    if (auto attr = dyn_cast<Attribute>(flatSizeOfr))
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        flatDimSize = intAttr.getInt();

    auto flatMemrefType =
        MemRefType::get({flatDimSize}, memrefType.getElementType(),
                        StridedLayoutAttr::get(rewriter.getContext(), 0, {1}),
                        memrefType.getMemorySpace());

    // Collect the flat dynamic-size operand (empty for fully-static case).
    SmallVector<Value, 1> dynSizes;
    if (flatDimSize == ShapedType::kDynamic)
      dynSizes.push_back(getValueFromOpFoldResult(rewriter, loc, flatSizeOfr));

    auto newOp = AllocLikeOp::create(rewriter, loc, flatMemrefType, dynSizes,
                                     op.getAlignmentAttr());
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, cast<MemRefType>(op.getType()), newOp,
        rewriter.getIndexAttr(staticOffset), sizes, strides);
    return success();
  }
};

template <typename T>
struct MemRefRewritePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    LogicalResult canFlatten = canBeFlattened(op, rewriter);
    if (failed(canFlatten))
      return canFlatten;

    Value memref = getTargetMemref(op);
    if (!needFlattening(memref) || !checkLayout(memref))
      return failure();

    auto &&[flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op->getLoc(), memref, getIndices<T>(op));
    replaceOp<T>(op, rewriter, flatMemref, offset);
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

void memref::populateFlattenVectorOpsOnMemrefPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<MemRefRewritePattern<vector::LoadOp>,
                  MemRefRewritePattern<vector::StoreOp>,
                  MemRefRewritePattern<vector::TransferReadOp>,
                  MemRefRewritePattern<vector::TransferWriteOp>,
                  MemRefRewritePattern<vector::MaskedLoadOp>,
                  MemRefRewritePattern<vector::MaskedStoreOp>>(
      patterns.getContext());
}

void memref::populateFlattenMemrefOpsPatterns(RewritePatternSet &patterns) {
  patterns.insert<MemRefRewritePattern<memref::LoadOp>,
                  MemRefRewritePattern<memref::StoreOp>,
                  AllocLikeFlattenPattern<memref::AllocOp>,
                  AllocLikeFlattenPattern<memref::AllocaOp>>(
      patterns.getContext());
}

void memref::populateFlattenMemrefsPatterns(RewritePatternSet &patterns) {
  populateFlattenMemrefOpsPatterns(patterns);
  populateFlattenVectorOpsOnMemrefPatterns(patterns);
}
