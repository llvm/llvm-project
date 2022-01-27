
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
        findDealloc(cloneOp.output());
    // Skip if either of them has > 1 deallocate operations.
    if (!maybeCloneDeallocOp.hasValue())
      return failure();
    llvm::Optional<Operation *> maybeSourceDeallocOp = findDealloc(source);
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
  results.insert<SimplifyClones>(context);
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

  // A memref_to_tensor + tensor_to_memref with same types can be folded without
  // inserting a cast.
  if (memrefToTensor.memref().getType() == toMemref.getType()) {
    if (!allowSameType)
      // Function can be configured to only handle cases where a cast is needed.
      return failure();
    rewriter.replaceOp(toMemref, memrefToTensor.memref());
    return success();
  }

  // If types are definitely not cast-compatible, bail.
  if (!memref::CastOp::areCastCompatible(memrefToTensor.memref().getType(),
                                         toMemref.getType()))
    return failure();

  // We already know that the types are potentially cast-compatible. However
  // in case the affine maps are different, we may need to use a copy if we go
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

  auto memrefToTensorType =
      memrefToTensor.memref().getType().dyn_cast<MemRefType>();
  auto toMemrefType = toMemref.getType().dyn_cast<MemRefType>();
  if (memrefToTensorType && toMemrefType &&
      !isGuaranteedCastCompatible(memrefToTensorType, toMemrefType)) {
    MemRefType resultType = toMemrefType;
    auto loc = toMemref.getLoc();
    SmallVector<Value, 4> dynamicOperands;
    for (int i = 0; i < resultType.getRank(); ++i) {
      if (resultType.getShape()[i] != ShapedType::kDynamicSize)
        continue;
      auto index = rewriter.createOrFold<arith::ConstantIndexOp>(loc, i);
      Value size = rewriter.create<tensor::DimOp>(loc, memrefToTensor, index);
      dynamicOperands.push_back(size);
    }
    // TODO: Use alloc/memcpy callback from BufferizationOptions if called via
    // BufferizableOpInterface impl of ToMemrefOp.
    auto copy =
        rewriter.create<memref::AllocOp>(loc, resultType, dynamicOperands);
    rewriter.create<memref::CopyOp>(loc, memrefToTensor.memref(), copy);
    rewriter.replaceOp(toMemref, {copy});
  } else
    rewriter.replaceOpWithNewOp<memref::CastOp>(toMemref, toMemref.getType(),
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
                                    const BufferizationState &state) {
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
