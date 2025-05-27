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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

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
  int64_t sourceOffset;
  SmallVector<int64_t, 4> sourceStrides;
  auto sourceType = cast<MemRefType>(source.getType());
  if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset))) {
    assert(false);
  }

  memref::ExtractStridedMetadataOp stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, source);

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
      rewriter.create<memref::ReinterpretCastOp>(
          loc, source,
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
      .Default([](auto) { return Value{}; });
}

template <typename T>
static void castAllocResult(T oper, T newOper, Location loc,
                            PatternRewriter &rewriter) {
  memref::ExtractStridedMetadataOp stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, oper);
  rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
      oper, cast<MemRefType>(oper.getType()), newOper,
      /*offset=*/rewriter.getIndexAttr(0),
      stridedMetadata.getConstifiedMixedSizes(),
      stridedMetadata.getConstifiedMixedStrides());
}

template <typename T>
static void replaceOp(T op, PatternRewriter &rewriter, Value flatMemref,
                      Value offset) {
  Location loc = op->getLoc();
  llvm::TypeSwitch<Operation *>(op.getOperation())
      .template Case<memref::AllocOp>([&](auto oper) {
        auto newAlloc = rewriter.create<memref::AllocOp>(
            loc, cast<MemRefType>(flatMemref.getType()),
            oper.getAlignmentAttr());
        castAllocResult(oper, newAlloc, loc, rewriter);
      })
      .template Case<memref::AllocaOp>([&](auto oper) {
        auto newAlloca = rewriter.create<memref::AllocaOp>(
            loc, cast<MemRefType>(flatMemref.getType()),
            oper.getAlignmentAttr());
        castAllocResult(oper, newAlloca, loc, rewriter);
      })
      .template Case<memref::LoadOp>([&](auto op) {
        auto newLoad = rewriter.create<memref::LoadOp>(
            loc, op->getResultTypes(), flatMemref, ValueRange{offset});
        newLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newLoad.getResult());
      })
      .template Case<memref::StoreOp>([&](auto op) {
        auto newStore = rewriter.create<memref::StoreOp>(
            loc, op->getOperands().front(), flatMemref, ValueRange{offset});
        newStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newStore);
      })
      .template Case<vector::LoadOp>([&](auto op) {
        auto newLoad = rewriter.create<vector::LoadOp>(
            loc, op->getResultTypes(), flatMemref, ValueRange{offset});
        newLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newLoad.getResult());
      })
      .template Case<vector::StoreOp>([&](auto op) {
        auto newStore = rewriter.create<vector::StoreOp>(
            loc, op->getOperands().front(), flatMemref, ValueRange{offset});
        newStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newStore);
      })
      .template Case<vector::MaskedLoadOp>([&](auto op) {
        auto newMaskedLoad = rewriter.create<vector::MaskedLoadOp>(
            loc, op.getType(), flatMemref, ValueRange{offset}, op.getMask(),
            op.getPassThru());
        newMaskedLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newMaskedLoad.getResult());
      })
      .template Case<vector::MaskedStoreOp>([&](auto op) {
        auto newMaskedStore = rewriter.create<vector::MaskedStoreOp>(
            loc, flatMemref, ValueRange{offset}, op.getMask(),
            op.getValueToStore());
        newMaskedStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newMaskedStore);
      })
      .template Case<vector::TransferReadOp>([&](auto op) {
        auto newTransferRead = rewriter.create<vector::TransferReadOp>(
            loc, op.getType(), flatMemref, ValueRange{offset}, op.getPadding());
        rewriter.replaceOp(op, newTransferRead.getResult());
      })
      .template Case<vector::TransferWriteOp>([&](auto op) {
        auto newTransferWrite = rewriter.create<vector::TransferWriteOp>(
            loc, op.getVector(), flatMemref, ValueRange{offset});
        rewriter.replaceOp(op, newTransferWrite);
      })
      .Default([&](auto op) {
        op->emitOpError("unimplemented: do not know how to replace op.");
      });
}

template <typename T>
static ValueRange getIndices(T op) {
  if constexpr (std::is_same_v<T, memref::AllocaOp> ||
                std::is_same_v<T, memref::AllocOp>) {
    return ValueRange{};
  } else {
    return op.getIndices();
  }
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

template <typename T>
struct MemRefRewritePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    LogicalResult canFlatten = canBeFlattened(op, rewriter);
    if (failed(canFlatten)) {
      return canFlatten;
    }

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

void memref::populateFlattenMemrefsPatterns(RewritePatternSet &patterns) {
  patterns.insert<MemRefRewritePattern<memref::LoadOp>,
                  MemRefRewritePattern<memref::StoreOp>,
                  MemRefRewritePattern<memref::AllocOp>,
                  MemRefRewritePattern<memref::AllocaOp>,
                  MemRefRewritePattern<vector::LoadOp>,
                  MemRefRewritePattern<vector::StoreOp>,
                  MemRefRewritePattern<vector::TransferReadOp>,
                  MemRefRewritePattern<vector::TransferWriteOp>,
                  MemRefRewritePattern<vector::MaskedLoadOp>,
                  MemRefRewritePattern<vector::MaskedStoreOp>>(
      patterns.getContext());
}
