//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"

using namespace mlir;
using namespace mlir::bufferization;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

FailureOr<Value>
mlir::bufferization::castOrReallocMemRefValue(OpBuilder &b, Value value,
                                              MemRefType destType) {
  auto srcType = value.getType().cast<MemRefType>();

  // Casting to the same type, nothing to do.
  if (srcType == destType)
    return value;

  // Element type, rank and memory space must match.
  if (srcType.getElementType() != destType.getElementType())
    return failure();
  if (srcType.getMemorySpaceAsInt() != destType.getMemorySpaceAsInt())
    return failure();
  if (srcType.getRank() != destType.getRank())
    return failure();

  // In case the affine maps are different, we may need to use a copy if we go
  // from dynamic to static offset or stride (the canonicalization cannot know
  // at this point that it is really cast compatible).
  auto isGuaranteedCastCompatible = [](MemRefType source, MemRefType target) {
    int64_t sourceOffset, targetOffset;
    SmallVector<int64_t, 4> sourceStrides, targetStrides;
    if (failed(getStridesAndOffset(source, sourceStrides, sourceOffset)) ||
        failed(getStridesAndOffset(target, targetStrides, targetOffset)))
      return false;
    auto dynamicToStatic = [](int64_t a, int64_t b) {
      return a == MemRefType::getDynamicStrideOrOffset() &&
             b != MemRefType::getDynamicStrideOrOffset();
    };
    if (dynamicToStatic(sourceOffset, targetOffset))
      return false;
    for (auto it : zip(sourceStrides, targetStrides))
      if (dynamicToStatic(std::get<0>(it), std::get<1>(it)))
        return false;
    return true;
  };

  // Note: If `areCastCompatible`, a cast is valid, but may fail at runtime. To
  // ensure that we only generate casts that always succeed at runtime, we check
  // a fix extra conditions in `isGuaranteedCastCompatible`.
  if (memref::CastOp::areCastCompatible(srcType, destType) &&
      isGuaranteedCastCompatible(srcType, destType)) {
    Value casted = b.create<memref::CastOp>(value.getLoc(), destType, value);
    return casted;
  }

  auto loc = value.getLoc();
  SmallVector<Value, 4> dynamicOperands;
  for (int i = 0; i < destType.getRank(); ++i) {
    if (destType.getShape()[i] != ShapedType::kDynamicSize)
      continue;
    auto index = b.createOrFold<arith::ConstantIndexOp>(loc, i);
    Value size = b.create<memref::DimOp>(loc, value, index);
    dynamicOperands.push_back(size);
  }
  // TODO: Use alloc/memcpy callback from BufferizationOptions if called via
  // BufferizableOpInterface impl of ToMemrefOp.
  Value copy = b.create<memref::AllocOp>(loc, destType, dynamicOperands);
  b.create<memref::CopyOp>(loc, value, copy);
  return copy;
}

//===----------------------------------------------------------------------===//
// CloneOp
//===----------------------------------------------------------------------===//

void CloneOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), input(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), output(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(), output(),
                       SideEffects::DefaultResource::get());
}

OpFoldResult CloneOp::fold(ArrayRef<Attribute> operands) {
  return succeeded(memref::foldMemRefCast(*this)) ? getResult() : Value();
}

namespace {

/// Merge the clone and its source (by converting the clone to a cast) when
/// possible.
struct SimplifyClones : public OpRewritePattern<CloneOp> {
  using OpRewritePattern<CloneOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CloneOp cloneOp,
                                PatternRewriter &rewriter) const override {
    if (cloneOp.use_empty()) {
      rewriter.eraseOp(cloneOp);
      return success();
    }

    Value source = cloneOp.input();

    // This only finds dealloc operations for the immediate value. It should
    // also consider aliases. That would also make the safety check below
    // redundant.
    llvm::Optional<Operation *> maybeCloneDeallocOp =
        memref::findDealloc(cloneOp.output());
    // Skip if either of them has > 1 deallocate operations.
    if (!maybeCloneDeallocOp.hasValue())
      return failure();
    llvm::Optional<Operation *> maybeSourceDeallocOp =
        memref::findDealloc(source);
    if (!maybeSourceDeallocOp.hasValue())
      return failure();
    Operation *cloneDeallocOp = *maybeCloneDeallocOp;
    Operation *sourceDeallocOp = *maybeSourceDeallocOp;

    // If both are deallocated in the same block, their in-block lifetimes
    // might not fully overlap, so we cannot decide which one to drop.
    if (cloneDeallocOp && sourceDeallocOp &&
        cloneDeallocOp->getBlock() == sourceDeallocOp->getBlock())
      return failure();

    Block *currentBlock = cloneOp->getBlock();
    Operation *redundantDealloc = nullptr;
    if (cloneDeallocOp && cloneDeallocOp->getBlock() == currentBlock) {
      redundantDealloc = cloneDeallocOp;
    } else if (sourceDeallocOp && sourceDeallocOp->getBlock() == currentBlock) {
      redundantDealloc = sourceDeallocOp;
    }

    if (!redundantDealloc)
      return failure();

    // Safety check that there are no other deallocations inbetween
    // cloneOp and redundantDealloc, as otherwise we might deallocate an alias
    // of source before the uses of the clone. With alias information, we could
    // restrict this to only fail of the dealloc's operand is an alias
    // of the source.
    for (Operation *pos = cloneOp->getNextNode(); pos != redundantDealloc;
         pos = pos->getNextNode()) {
      auto effectInterface = dyn_cast<MemoryEffectOpInterface>(pos);
      if (!effectInterface)
        continue;
      if (effectInterface.hasEffect<MemoryEffects::Free>())
        return failure();
    }

    rewriter.replaceOpWithNewOp<memref::CastOp>(cloneOp, cloneOp.getType(),
                                                source);
    rewriter.eraseOp(redundantDealloc);
    return success();
  }
};

} // namespace

void CloneOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<SimplifyClones>(context);
}

//===----------------------------------------------------------------------===//
// ToTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult ToTensorOp::fold(ArrayRef<Attribute>) {
  if (auto toMemref = memref().getDefiningOp<ToMemrefOp>())
    // Approximate alias analysis by conservatively folding only when no there
    // is no interleaved operation.
    if (toMemref->getBlock() == this->getOperation()->getBlock() &&
        toMemref->getNextNode() == this->getOperation())
      return toMemref.tensor();
  return {};
}

namespace {

struct DimOfToTensorFolder : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto memrefToTensorOp = dimOp.source().getDefiningOp<ToTensorOp>();
    if (!memrefToTensorOp)
      return failure();

    rewriter.replaceOpWithNewOp<memref::DimOp>(dimOp, memrefToTensorOp.memref(),
                                               dimOp.index());
    return success();
  }
};

} // namespace

void ToTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<DimOfToTensorFolder>(context);
}

//===----------------------------------------------------------------------===//
// ToMemrefOp
//===----------------------------------------------------------------------===//

OpFoldResult ToMemrefOp::fold(ArrayRef<Attribute>) {
  if (auto memrefToTensor = tensor().getDefiningOp<ToTensorOp>())
    if (memrefToTensor.memref().getType() == getType())
      return memrefToTensor.memref();
  return {};
}

namespace {

/// Replace tensor.cast + to_memref by to_memref + memref.cast.
struct ToMemrefOfCast : public OpRewritePattern<ToMemrefOp> {
  using OpRewritePattern<ToMemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToMemrefOp toMemref,
                                PatternRewriter &rewriter) const final {
    auto tensorCastOperand =
        toMemref.getOperand().getDefiningOp<tensor::CastOp>();
    if (!tensorCastOperand)
      return failure();
    auto srcTensorType =
        tensorCastOperand.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!srcTensorType)
      return failure();
    auto memrefType = MemRefType::get(srcTensorType.getShape(),
                                      srcTensorType.getElementType());
    Value memref = rewriter.create<ToMemrefOp>(toMemref.getLoc(), memrefType,
                                               tensorCastOperand.getOperand());
    rewriter.replaceOpWithNewOp<memref::CastOp>(toMemref, toMemref.getType(),
                                                memref);
    return success();
  }
};

/// Try to fold to_memref(to_tensor(x)). If x's type and the result type of the
/// to_memref op are different, a memref.cast is needed.
static LogicalResult foldToMemrefToTensorPair(RewriterBase &rewriter,
                                              ToMemrefOp toMemref,
                                              bool allowSameType = true) {
  auto memrefToTensor = toMemref.tensor().getDefiningOp<ToTensorOp>();
  if (!memrefToTensor)
    return failure();

  Type srcType = memrefToTensor.memref().getType();
  Type destType = toMemref.getType();

  // Function can be configured to only handle cases where a cast is needed.
  if (!allowSameType && srcType == destType)
    return failure();

  auto rankedSrcType = srcType.dyn_cast<MemRefType>();
  auto rankedDestType = destType.dyn_cast<MemRefType>();
  auto unrankedSrcType = srcType.dyn_cast<UnrankedMemRefType>();

  // Ranked memref -> Ranked memref cast.
  if (rankedSrcType && rankedDestType) {
    FailureOr<Value> replacement = castOrReallocMemRefValue(
        rewriter, memrefToTensor.memref(), rankedDestType);
    if (failed(replacement))
      return failure();

    rewriter.replaceOp(toMemref, *replacement);
    return success();
  }

  // Unranked memref -> Ranked memref cast: May require a copy.
  // TODO: Not implemented at the moment.
  if (unrankedSrcType && rankedDestType)
    return failure();

  // Unranked memref -> unranked memref cast
  // Ranked memref -> unranked memref cast: No copy needed.
  assert(memref::CastOp::areCastCompatible(srcType, destType) &&
         "expected that types are cast compatible");
  rewriter.replaceOpWithNewOp<memref::CastOp>(toMemref, destType,
                                              memrefToTensor.memref());
  return success();
}

/// Canonicalize bufferization.to_tensor + bufferization.to_memref to
/// memref.cast when type mismatches prevent `ToMemrefOp::fold` to kick in.
struct TensorLoadToMemref : public OpRewritePattern<ToMemrefOp> {
  using OpRewritePattern<ToMemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToMemrefOp toMemref,
                                PatternRewriter &rewriter) const final {
    // Only handle cases where a cast is needed. The other case is handled by
    // the folder.
    return foldToMemrefToTensorPair(rewriter, toMemref,
                                    /*allowSameType=*/false);
  }
};

/// Fold a load on a to_memref operation into an tensor.extract on the
/// corresponding tensor.
struct LoadOfToMemref : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load,
                                PatternRewriter &rewriter) const override {
    auto toMemref = load.memref().getDefiningOp<ToMemrefOp>();
    if (!toMemref)
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(load, toMemref.tensor(),
                                                   load.indices());
    return success();
  }
};

/// Fold dim of a to_memref into the dim of the tensor.
struct DimOfCastOp : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern<memref::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = dimOp.source().getDefiningOp<ToMemrefOp>();
    if (!castOp)
      return failure();
    Value newSource = castOp.getOperand();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(dimOp, newSource, dimOp.index());
    return success();
  }
};

} // namespace

void ToMemrefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<DimOfCastOp, LoadOfToMemref, ToMemrefOfCast, TensorLoadToMemref>(
      context);
}

LogicalResult ToMemrefOp::bufferize(RewriterBase &rewriter,
                                    BufferizationState &state) {
  // Fold to_memref(to_tensor(x)) to x. Insert a cast if necessary.
  return foldToMemrefToTensorPair(rewriter, *this);
}

Optional<Operation *> CloneOp::buildDealloc(OpBuilder &builder, Value alloc) {
  return builder.create<memref::DeallocOp>(alloc.getLoc(), alloc)
      .getOperation();
}

Optional<Value> CloneOp::buildClone(OpBuilder &builder, Value alloc) {
  return builder.create<CloneOp>(alloc.getLoc(), alloc).getResult();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Bufferization/IR/BufferizationOps.cpp.inc"
