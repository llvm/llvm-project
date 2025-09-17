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
#include "mlir/IR/DialectResourceBlobManager.h"
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

/// Produce an OpFoldResult representing the product of the values or constants
/// referenced by `indices`. `staticShape` provides the statically known sizes
/// for the source memref, while `values` contains the mixed (value/attribute)
/// representation produced by `memref.extract_strided_metadata`.
static OpFoldResult getProductOfValues(ArrayRef<int64_t> indices,
                                       OpBuilder &builder, Location loc,
                                       ArrayRef<int64_t> staticShape,
                                       ArrayRef<OpFoldResult> values) {
  AffineExpr product = builder.getAffineConstantExpr(1);
  SmallVector<OpFoldResult> inputs;
  unsigned numSymbols = 0;
  for (int64_t idx : indices) {
    product = product * builder.getAffineSymbolExpr(numSymbols++);
    if (ShapedType::isDynamic(staticShape[idx]))
      inputs.push_back(values[idx]);
    else
      inputs.push_back(builder.getIndexAttr(staticShape[idx]));
  }
  return affine::makeComposedFoldedAffineApply(builder, loc, product, inputs);
}

/// Return the collapsed size (as OpFoldResult) for the reassociation group
/// `groupId` of `collapseShapeOp`.
static SmallVector<OpFoldResult>
getCollapsedSize(memref::CollapseShapeOp collapseShapeOp, OpBuilder &builder,
                 ArrayRef<OpFoldResult> origSizes, unsigned groupId) {
  SmallVector<OpFoldResult> collapsedSize;

  MemRefType resultType = collapseShapeOp.getResultType();
  int64_t dimSize = resultType.getDimSize(groupId);
  if (!ShapedType::isDynamic(dimSize)) {
    collapsedSize.push_back(builder.getIndexAttr(dimSize));
    return collapsedSize;
  }

  auto sourceType = collapseShapeOp.getSrcType();
  ArrayRef<int64_t> staticShape = sourceType.getShape();
  ArrayRef<int64_t> reassocGroup =
      collapseShapeOp.getReassociationIndices()[groupId];

  collapsedSize.push_back(getProductOfValues(reassocGroup, builder,
                                             collapseShapeOp.getLoc(),
                                             staticShape, origSizes));
  return collapsedSize;
}

/// Return the collapsed stride (as OpFoldResult) for the reassociation group
/// `groupId` of `collapseShapeOp`.
static SmallVector<OpFoldResult> getCollapsedStride(
    memref::CollapseShapeOp collapseShapeOp, OpBuilder &builder,
    ArrayRef<OpFoldResult> origSizes, ArrayRef<OpFoldResult> origStrides,
    unsigned groupId) {
  ArrayRef<int64_t> reassocGroup =
      collapseShapeOp.getReassociationIndices()[groupId];
  assert(!reassocGroup.empty() &&
         "reassociation group must contain at least one dimension");

  auto sourceType = collapseShapeOp.getSrcType();
  auto [strides, offset] = sourceType.getStridesAndOffset();
  (void)offset;
  ArrayRef<int64_t> srcShape = sourceType.getShape();

  OpFoldResult lastValidStride = nullptr;
  for (int64_t dim : reassocGroup) {
    if (srcShape[dim] == 1)
      continue;
    int64_t currentStride = strides[dim];
    if (ShapedType::isDynamic(currentStride))
      lastValidStride = origStrides[dim];
    else
      lastValidStride = builder.getIndexAttr(currentStride);
  }

  if (!lastValidStride) {
    MemRefType collapsedType = collapseShapeOp.getResultType();
    auto [collapsedStrides, collapsedOffset] =
        collapsedType.getStridesAndOffset();
    (void)collapsedOffset;
    int64_t finalStride = collapsedStrides[groupId];
    if (ShapedType::isDynamic(finalStride)) {
      for (int64_t dim : reassocGroup) {
        assert(srcShape[dim] == 1 && "expected size-one dimensions");
        if (ShapedType::isDynamic(strides[dim]))
          return {origStrides[dim]};
      }
      llvm_unreachable("expected to find a dynamic stride");
    }
    return {builder.getIndexAttr(finalStride)};
  }

  return {lastValidStride};
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

  LogicalResult
  matchAndRewrite(memref::GlobalOp globalOp,
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
    SmallVector<OpFoldResult> origStrides = metadata.getConstifiedMixedStrides();
    OpFoldResult offset = metadata.getConstifiedMixedOffset();

    SmallVector<OpFoldResult> collapsedSizes;
    SmallVector<OpFoldResult> collapsedStrides;
    unsigned numGroups = op.getReassociationIndices().size();
    collapsedSizes.reserve(numGroups);
    collapsedStrides.reserve(numGroups);
    for (unsigned i = 0; i < numGroups; ++i) {
      SmallVector<OpFoldResult> groupSizes =
          getCollapsedSize(op, rewriter, origSizes, i);
      SmallVector<OpFoldResult> groupStrides =
          getCollapsedStride(op, rewriter, origSizes, origStrides, i);
      collapsedSizes.append(groupSizes.begin(), groupSizes.end());
      collapsedStrides.append(groupStrides.begin(), groupStrides.end());
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, op.getType(), op.getSrc(), offset, collapsedSizes,
        collapsedStrides);
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

/// Special pattern for GetGlobalOp to avoid infinite loops
struct FlattenGetGlobal : public OpRewritePattern<memref::GetGlobalOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::GetGlobalOp op,
                               PatternRewriter &rewriter) const override {
    // Check if this get_global references a multi-dimensional global
    auto module = op->template getParentOfType<ModuleOp>();
    auto globalOp = module.template lookupSymbol<memref::GlobalOp>(op.getName());
    if (!globalOp) {
      return failure();
    }

    auto globalType = globalOp.getType();
    auto resultType = op.getType();

    // Only apply if the global has been flattened but the get_global hasn't
    if (globalType.getRank() == 1 && resultType.getRank() > 1) {
      auto newGetGlobal = memref::GetGlobalOp::create(
          rewriter, op.getLoc(), globalType, op.getName());

      // Cast the flattened result back to the original shape
      memref::ExtractStridedMetadataOp stridedMetadata =
          memref::ExtractStridedMetadataOp::create(rewriter, op.getLoc(), op.getResult());
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

void memref::populateFlattenMemrefOpsPatterns(RewritePatternSet &patterns) {
  patterns.insert<MemRefRewritePattern<memref::LoadOp>,
                  MemRefRewritePattern<memref::StoreOp>,
                  MemRefRewritePattern<memref::AllocOp>,
                  MemRefRewritePattern<memref::AllocaOp>,
                  MemRefRewritePattern<memref::DeallocOp>,
                  FlattenCollapseShape,
                  FlattenGetGlobal,
                  FlattenGlobal>(
      patterns.getContext());
}

void memref::populateFlattenMemrefsPatterns(RewritePatternSet &patterns) {
  populateFlattenMemrefOpsPatterns(patterns);
  populateFlattenVectorOpsOnMemrefPatterns(patterns);
}
