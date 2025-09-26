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
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
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

static bool checkLayout(MemRefType type) {
  return type.getLayout().isIdentity() ||
         isa<StridedLayoutAttr>(type.getLayout());
}

static bool checkLayout(Value val) {
  return checkLayout(cast<MemRefType>(val.getType()));
}

namespace {
static Value getTargetMemref(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .template Case<memref::LoadOp, memref::StoreOp, memref::AllocaOp,
                     memref::AllocOp, memref::DeallocOp>(
          [](auto op) { return op.getMemref(); })
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
      memref::ExtractStridedMetadataOp::create(rewriter, loc, oper);
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
        auto newAlloc = memref::AllocOp::create(
            rewriter, loc, cast<MemRefType>(flatMemref.getType()),
            oper.getAlignmentAttr());
        castAllocResult(oper, newAlloc, loc, rewriter);
      })
      .template Case<memref::AllocaOp>([&](auto oper) {
        auto newAlloca = memref::AllocaOp::create(
            rewriter, loc, cast<MemRefType>(flatMemref.getType()),
            oper.getAlignmentAttr());
        castAllocResult(oper, newAlloca, loc, rewriter);
      })
      .template Case<memref::LoadOp>([&](auto op) {
        auto newLoad =
            memref::LoadOp::create(rewriter, loc, op->getResultTypes(),
                                   flatMemref, ValueRange{offset});
        newLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newLoad.getResult());
      })
      .template Case<memref::StoreOp>([&](auto op) {
        auto newStore =
            memref::StoreOp::create(rewriter, loc, op->getOperands().front(),
                                    flatMemref, ValueRange{offset});
        newStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newStore);
      })
      .template Case<vector::LoadOp>([&](auto op) {
        auto newLoad =
            vector::LoadOp::create(rewriter, loc, op->getResultTypes(),
                                   flatMemref, ValueRange{offset});
        newLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newLoad.getResult());
      })
      .template Case<vector::StoreOp>([&](auto op) {
        auto newStore =
            vector::StoreOp::create(rewriter, loc, op->getOperands().front(),
                                    flatMemref, ValueRange{offset});
        newStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newStore);
      })
      .template Case<vector::MaskedLoadOp>([&](auto op) {
        auto newMaskedLoad = vector::MaskedLoadOp::create(
            rewriter, loc, op.getType(), flatMemref, ValueRange{offset},
            op.getMask(), op.getPassThru());
        newMaskedLoad->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newMaskedLoad.getResult());
      })
      .template Case<vector::MaskedStoreOp>([&](auto op) {
        auto newMaskedStore = vector::MaskedStoreOp::create(
            rewriter, loc, flatMemref, ValueRange{offset}, op.getMask(),
            op.getValueToStore());
        newMaskedStore->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newMaskedStore);
      })
      .template Case<vector::TransferReadOp>([&](auto op) {
        auto newTransferRead = vector::TransferReadOp::create(
            rewriter, loc, op.getType(), flatMemref, ValueRange{offset},
            op.getPadding());
        rewriter.replaceOp(op, newTransferRead.getResult());
      })
      .template Case<vector::TransferWriteOp>([&](auto op) {
        auto newTransferWrite = vector::TransferWriteOp::create(
            rewriter, loc, op.getVector(), flatMemref, ValueRange{offset});
        rewriter.replaceOp(op, newTransferWrite);
      })
      .template Case<memref::DeallocOp>([&](auto op) {
        auto newDealloc = memref::DeallocOp::create(rewriter, loc, flatMemref);
        rewriter.replaceOp(op, newDealloc);
      })
      .Default([&](auto op) {
        op->emitOpError("unimplemented: do not know how to replace op.");
      });
}

template <typename T>
static ValueRange getIndices(T op) {
  if constexpr (std::is_same_v<T, memref::AllocaOp> ||
                std::is_same_v<T, memref::AllocOp> ||
                std::is_same_v<T, memref::DeallocOp>) {
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

/// Flattens memref global ops with more than 1 dimensions to 1 dimension.
struct FlattenGlobal final : public OpRewritePattern<memref::GlobalOp> {
  using OpRewritePattern::OpRewritePattern;

  static Attribute flattenAttribute(Attribute value, ShapedType newType) {
    if (!value)
      return value;
    if (auto splatAttr = llvm::dyn_cast<SplatElementsAttr>(value)) {
      return splatAttr.reshape(newType);
    } else if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(value)) {
      return denseAttr.reshape(newType);
    } else if (auto denseResourceAttr =
                   llvm::dyn_cast<DenseResourceElementsAttr>(value)) {
      return DenseResourceElementsAttr::get(newType,
                                            denseResourceAttr.getRawHandle());
    }
    return {};
  }

  LogicalResult matchAndRewrite(memref::GlobalOp globalOp,
                                PatternRewriter &rewriter) const override {
    auto oldType = llvm::dyn_cast<MemRefType>(globalOp.getType());
    if (!oldType || !oldType.getLayout().isIdentity() || oldType.getRank() <= 1)
      return failure();

    auto tensorType = RankedTensorType::get({oldType.getNumElements()},
                                            oldType.getElementType());
    auto memRefType =
        MemRefType::get({oldType.getNumElements()}, oldType.getElementType(),
                        AffineMap(), oldType.getMemorySpace());
    auto newInitialValue =
        flattenAttribute(globalOp.getInitialValueAttr(), tensorType);
    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        globalOp, globalOp.getSymName(), globalOp.getSymVisibilityAttr(),
        memRefType, newInitialValue, globalOp.getConstant(),
        /*alignment=*/IntegerAttr());
    return success();
  }
};

struct FlattenCollapseShape final
    : public OpRewritePattern<memref::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CollapseShapeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    memref::ExtractStridedMetadataOp metadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, op.getSrc());

    SmallVector<OpFoldResult> origSizes = metadata.getConstifiedMixedSizes();
    SmallVector<OpFoldResult> origStrides =
        metadata.getConstifiedMixedStrides();
    OpFoldResult offset = metadata.getConstifiedMixedOffset();

    SmallVector<OpFoldResult> collapsedSizes;
    SmallVector<OpFoldResult> collapsedStrides;
    unsigned numGroups = op.getReassociationIndices().size();
    collapsedSizes.reserve(numGroups);
    collapsedStrides.reserve(numGroups);
    for (unsigned i = 0; i < numGroups; ++i) {
      SmallVector<OpFoldResult> groupSizes =
          memref::getCollapsedSize(op, rewriter, origSizes, i);
      SmallVector<OpFoldResult> groupStrides =
          memref::getCollapsedStride(op, rewriter, origSizes, origStrides, i);
      collapsedSizes.append(groupSizes.begin(), groupSizes.end());
      collapsedStrides.append(groupStrides.begin(), groupStrides.end());
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, op.getType(), op.getSrc(), offset, collapsedSizes,
        collapsedStrides);
    return success();
  }
};

struct FlattenExpandShape final
    : public OpRewritePattern<memref::ExpandShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExpandShapeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    memref::ExtractStridedMetadataOp metadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, op.getSrc());

    SmallVector<OpFoldResult> origSizes = metadata.getConstifiedMixedSizes();
    SmallVector<OpFoldResult> origStrides =
        metadata.getConstifiedMixedStrides();
    OpFoldResult offset = metadata.getConstifiedMixedOffset();

    SmallVector<OpFoldResult> expandedSizes;
    SmallVector<OpFoldResult> expandedStrides;
    unsigned numGroups = op.getReassociationIndices().size();
    expandedSizes.reserve(op.getResultType().getRank());
    expandedStrides.reserve(op.getResultType().getRank());

    for (unsigned i = 0; i < numGroups; ++i) {
      SmallVector<OpFoldResult> groupSizes =
          memref::getExpandedSizes(op, rewriter, origSizes, i);
      SmallVector<OpFoldResult> groupStrides =
          memref::getExpandedStrides(op, rewriter, origSizes, origStrides, i);
      expandedSizes.append(groupSizes.begin(), groupSizes.end());
      expandedStrides.append(groupStrides.begin(), groupStrides.end());
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, op.getType(), op.getSrc(), offset, expandedSizes, expandedStrides);
    return success();
  }
};

// Flattens memref subview ops with more than 1 dimension into 1-D accesses.
struct FlattenSubView final : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<MemRefType>(op.getSource().getType());
    if (!sourceType || sourceType.getRank() <= 1)
      return failure();
    if (!checkLayout(sourceType))
      return failure();

    MemRefType resultType = op.getType();
    if (resultType.getRank() <= 1 || !checkLayout(resultType))
      return failure();

    unsigned elementBitWidth = sourceType.getElementTypeBitWidth();
    if (!elementBitWidth)
      return failure();

    Location loc = op.getLoc();

    // Materialize offsets as values so they can participate in linearization.
    SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = op.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = op.getMixedStrides();

    SmallVector<Value> offsetValues;
    offsetValues.reserve(mixedOffsets.size());
    for (OpFoldResult ofr : mixedOffsets)
      offsetValues.push_back(getValueFromOpFoldResult(rewriter, loc, ofr));

    auto [flatSource, linearOffset] = getFlattenMemrefAndOffset(
        rewriter, loc, op.getSource(), ValueRange(offsetValues));

    memref::ExtractStridedMetadataOp sourceMetadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, op.getSource());

    SmallVector<OpFoldResult> sourceStrides =
        sourceMetadata.getConstifiedMixedStrides();
    OpFoldResult sourceOffset = sourceMetadata.getConstifiedMixedOffset();

    llvm::SmallBitVector droppedDims = op.getDroppedDims();

    SmallVector<OpFoldResult> resultSizes;
    SmallVector<OpFoldResult> resultStrides;
    resultSizes.reserve(resultType.getRank());
    resultStrides.reserve(resultType.getRank());

    OpFoldResult resultOffset = sourceOffset;
    for (auto zipped : llvm::enumerate(llvm::zip_equal(
             mixedOffsets, sourceStrides, mixedSizes, mixedStrides))) {
      auto idx = zipped.index();
      auto it = zipped.value();
      auto offsetOfr = std::get<0>(it);
      auto strideOfr = std::get<1>(it);
      auto sizeOfr = std::get<2>(it);
      auto relativeStrideOfr = std::get<3>(it);
      OpFoldResult contribution = [&]() -> OpFoldResult {
        if (Attribute offsetAttr = dyn_cast<Attribute>(offsetOfr)) {
          if (Attribute strideAttr = dyn_cast<Attribute>(strideOfr)) {
            auto offsetInt = cast<IntegerAttr>(offsetAttr).getInt();
            auto strideInt = cast<IntegerAttr>(strideAttr).getInt();
            return rewriter.getIndexAttr(offsetInt * strideInt);
          }
        }
        Value offsetVal = getValueFromOpFoldResult(rewriter, loc, offsetOfr);
        Value strideVal = getValueFromOpFoldResult(rewriter, loc, strideOfr);
        return rewriter.create<arith::MulIOp>(loc, offsetVal, strideVal)
            .getResult();
      }();
      resultOffset = [&]() -> OpFoldResult {
        if (Attribute offsetAttr = dyn_cast<Attribute>(resultOffset)) {
          if (Attribute contribAttr = dyn_cast<Attribute>(contribution)) {
            auto offsetInt = cast<IntegerAttr>(offsetAttr).getInt();
            auto contribInt = cast<IntegerAttr>(contribAttr).getInt();
            return rewriter.getIndexAttr(offsetInt + contribInt);
          }
        }
        Value offsetVal = getValueFromOpFoldResult(rewriter, loc, resultOffset);
        Value contribVal =
            getValueFromOpFoldResult(rewriter, loc, contribution);
        return rewriter.create<arith::AddIOp>(loc, offsetVal, contribVal)
            .getResult();
      }();

      if (droppedDims.test(idx))
        continue;

      resultSizes.push_back(sizeOfr);
      OpFoldResult combinedStride = [&]() -> OpFoldResult {
        if (Attribute relStrideAttr = dyn_cast<Attribute>(relativeStrideOfr)) {
          if (Attribute strideAttr = dyn_cast<Attribute>(strideOfr)) {
            auto relStrideInt = cast<IntegerAttr>(relStrideAttr).getInt();
            auto strideInt = cast<IntegerAttr>(strideAttr).getInt();
            return rewriter.getIndexAttr(relStrideInt * strideInt);
          }
        }
        Value relStrideVal =
            getValueFromOpFoldResult(rewriter, loc, relativeStrideOfr);
        Value strideVal = getValueFromOpFoldResult(rewriter, loc, strideOfr);
        return rewriter.create<arith::MulIOp>(loc, relStrideVal, strideVal)
            .getResult();
      }();
      resultStrides.push_back(combinedStride);
    }

    memref::LinearizedMemRefInfo linearizedInfo;
    [[maybe_unused]] OpFoldResult linearizedIndex;
    std::tie(linearizedInfo, linearizedIndex) =
        memref::getLinearizedMemRefOffsetAndSize(rewriter, loc, elementBitWidth,
                                                 elementBitWidth, resultOffset,
                                                 resultSizes, resultStrides);

    Value flattenedSize =
        getValueFromOpFoldResult(rewriter, loc, linearizedInfo.linearizedSize);
    Value strideOne = arith::ConstantIndexOp::create(rewriter, loc, 1);

    Value flattenedSubview = memref::SubViewOp::create(
        rewriter, loc, flatSource, ValueRange{linearOffset},
        ValueRange{flattenedSize}, ValueRange{strideOne});

    Value replacement = memref::ReinterpretCastOp::create(
        rewriter, loc, resultType, flattenedSubview, resultOffset, resultSizes,
        resultStrides);

    rewriter.replaceOp(op, replacement);
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

struct FlattenGetGlobal : public OpRewritePattern<memref::GetGlobalOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::GetGlobalOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this get_global references a multi-dimensional global
    auto module = op->template getParentOfType<ModuleOp>();
    auto globalOp =
        module.template lookupSymbol<memref::GlobalOp>(op.getName());
    if (!globalOp) {
      return failure();
    }

    auto globalType = globalOp.getType();
    auto resultType = op.getType();

    // Only apply if the global has been flattened but the get_global hasn't
    if (globalType.getRank() == 1 && resultType.getRank() > 1) {
      auto newGetGlobal = memref::GetGlobalOp::create(rewriter, op.getLoc(),
                                                      globalType, op.getName());

      // Cast the flattened result back to the original shape
      memref::ExtractStridedMetadataOp stridedMetadata =
          memref::ExtractStridedMetadataOp::create(rewriter, op.getLoc(),
                                                   op.getResult());
      auto castResult = memref::ReinterpretCastOp::create(
          rewriter, op.getLoc(), resultType, newGetGlobal,
          /*offset=*/rewriter.getIndexAttr(0),
          stridedMetadata.getConstifiedMixedSizes(),
          stridedMetadata.getConstifiedMixedStrides());
      rewriter.replaceOp(op, castResult);
      return success();
    }

    return failure();
  }
};

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
                  MemRefRewritePattern<memref::AllocOp>,
                  MemRefRewritePattern<memref::AllocaOp>,
                  MemRefRewritePattern<memref::DeallocOp>, FlattenExpandShape,
                  FlattenCollapseShape, FlattenSubView, FlattenGetGlobal,
                  FlattenGlobal>(patterns.getContext());
}

void memref::populateFlattenMemrefsPatterns(RewritePatternSet &patterns) {
  populateFlattenMemrefOpsPatterns(patterns);
  populateFlattenVectorOpsOnMemrefPatterns(patterns);
}
