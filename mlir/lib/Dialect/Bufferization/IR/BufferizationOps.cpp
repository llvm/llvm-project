//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::bufferization;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

FailureOr<Value>
mlir::bufferization::castOrReallocMemRefValue(OpBuilder &b, Value value,
                                              MemRefType destType) {
  auto srcType = value.getType().cast<MemRefType>();

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

/// Try to fold to_memref(to_tensor(x)). If x's type and the result type of the
/// to_memref op are different, a memref.cast is needed.
LogicalResult
mlir::bufferization::foldToMemrefToTensorPair(RewriterBase &rewriter,
                                              ToMemrefOp toMemref) {
  auto memrefToTensor = toMemref.getTensor().getDefiningOp<ToTensorOp>();
  if (!memrefToTensor)
    return failure();

  Type srcType = memrefToTensor.getMemref().getType();
  Type destType = toMemref.getType();

  // Directly rewrite if the type did not change.
  if (srcType == destType) {
    rewriter.replaceOp(toMemref, memrefToTensor.getMemref());
    return success();
  }

  auto rankedSrcType = srcType.dyn_cast<MemRefType>();
  auto rankedDestType = destType.dyn_cast<MemRefType>();
  auto unrankedSrcType = srcType.dyn_cast<UnrankedMemRefType>();

  // Ranked memref -> Ranked memref cast.
  if (rankedSrcType && rankedDestType) {
    FailureOr<Value> replacement = castOrReallocMemRefValue(
        rewriter, memrefToTensor.getMemref(), rankedDestType);
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
                                              memrefToTensor.getMemref());
  return success();
}

void mlir::bufferization::populateDynamicDimSizes(
    OpBuilder &b, Location loc, Value shapedValue,
    SmallVector<Value> &dynamicDims) {
  auto shapedType = shapedValue.getType().cast<ShapedType>();
  for (int64_t i = 0; i < shapedType.getRank(); ++i) {
    if (shapedType.isDynamicDim(i)) {
      if (shapedType.isa<MemRefType>()) {
        dynamicDims.push_back(b.create<memref::DimOp>(loc, shapedValue, i));
      } else {
        assert(shapedType.isa<RankedTensorType>() && "expected tensor");
        dynamicDims.push_back(b.create<tensor::DimOp>(loc, shapedValue, i));
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// AllocTensorOp
//===----------------------------------------------------------------------===//

LogicalResult AllocTensorOp::bufferize(RewriterBase &rewriter,
                                       const BufferizationOptions &options) {
  OpBuilder::InsertionGuard g(rewriter);
  Operation *op = this->getOperation();
  Location loc = getLoc();

  // Nothing to do for dead AllocTensorOps.
  if (getOperation()->getUses().empty()) {
    rewriter.eraseOp(getOperation());
    return success();
  }

  // Get "copy" buffer.
  Value copyBuffer;
  if (getCopy()) {
    FailureOr<Value> maybeCopyBuffer = getBuffer(rewriter, getCopy(), options);
    if (failed(maybeCopyBuffer))
      return failure();
    copyBuffer = *maybeCopyBuffer;
  }

  // Compute memory space of this allocation.
  unsigned memorySpace;
  if (getMemorySpace().hasValue()) {
    memorySpace = *getMemorySpace();
  } else if (getCopy()) {
    memorySpace =
        copyBuffer.getType().cast<BaseMemRefType>().getMemorySpaceAsInt();
  } else if (options.defaultMemorySpace.hasValue()) {
    memorySpace = *options.defaultMemorySpace;
  } else {
    return op->emitError("could not infer memory space");
  }

  // Create memory allocation.
  auto allocType =
      MemRefType::get(getType().getShape(), getType().getElementType(),
                      AffineMap(), memorySpace);
  SmallVector<Value> dynamicDims = getDynamicSizes();
  if (getCopy()) {
    assert(dynamicDims.empty() && "expected either `copy` or `dynamicDims`");
    populateDynamicDimSizes(rewriter, loc, copyBuffer, dynamicDims);
  }
  FailureOr<Value> alloc =
      options.createAlloc(rewriter, loc, allocType, dynamicDims);
  if (failed(alloc))
    return failure();

  // Create memory copy (if any).
  if (getCopy()) {
    if (failed(options.createMemCpy(rewriter, loc, copyBuffer, *alloc)))
      return failure();
  }

  // Should the buffer be deallocated?
  AnalysisState analysisState(options);
  bool dealloc;
  if (op->hasAttr(BufferizationDialect::kEscapeAttrName)) {
    // AllocTensorOp has one result.
    ArrayAttr escapeAttr =
        op->getAttr(BufferizationDialect::kEscapeAttrName).cast<ArrayAttr>();
    dealloc = !escapeAttr[0].cast<BoolAttr>().getValue();
  } else {
    // No "escape" annotation found.
    if (options.createDeallocs) {
      // Perform an ad-hoc analysis.
      dealloc = !analysisState.isTensorYielded(getResult());
    } else {
      dealloc = false;
    }
  }

  // Replace op.
  replaceOpWithBufferizedValues(rewriter, getOperation(), *alloc);

  // Create buffer deallocation (if requested).
  if (!dealloc)
    return success();

  rewriter.setInsertionPoint(rewriter.getInsertionBlock()->getTerminator());
  if (failed(options.createDealloc(rewriter, loc, *alloc)))
    return failure();
  return success();
}

bool AllocTensorOp::isMemoryWrite(OpResult opResult,
                                  const AnalysisState &state) {
  // AllocTensorOps do not write unless they have a `copy` value.
  return static_cast<bool>(getCopy());
}

bool AllocTensorOp::bufferizesToMemoryRead(OpOperand &opOperand,
                                           const AnalysisState &state) {
  assert(opOperand.getOperandNumber() == getNumOperands() - 1 &&
         "expected copy operand");
  return true;
}

bool AllocTensorOp::bufferizesToMemoryWrite(OpOperand &opOperand,
                                            const AnalysisState &state) {
  assert(opOperand.getOperandNumber() == getNumOperands() - 1 &&
         "expected copy operand");
  return false;
}

SmallVector<OpResult>
AllocTensorOp::getAliasingOpResult(OpOperand &opOperand,
                                   const AnalysisState &state) {
  // This is a new allocation. It does not alias with any other buffer.
  return {};
}

LogicalResult AllocTensorOp::verify() {
  if (getCopy() && !getDynamicSizes().empty())
    return emitError("dynamic sizes not needed when copying a tensor");
  if (!getCopy() && getType().getNumDynamicDims() !=
                        static_cast<int64_t>(getDynamicSizes().size()))
    return emitError("expected ")
           << getType().getNumDynamicDims() << " dynamic sizes";
  if (getCopy() && getCopy().getType() != getType())
    return emitError("expected that `copy` and return type match");

  // For sparse tensor allocation, we require that none of its
  // uses escapes the function boundary directly.
  if (sparse_tensor::getSparseTensorEncoding(getType())) {
    for (auto &use : getOperation()->getUses())
      if (isa<func::ReturnOp, func::CallOp, func::CallIndirectOp>(
              use.getOwner()))
        return emitError("sparse tensor allocation should not escape function");
  }

  return success();
}

void AllocTensorOp::build(OpBuilder &builder, OperationState &result,
                          RankedTensorType type, ValueRange dynamicSizes) {
  build(builder, result, type, dynamicSizes, /*copy=*/Value(),
        /*memory_space=*/IntegerAttr());
}

void AllocTensorOp::build(OpBuilder &builder, OperationState &result,
                          RankedTensorType type, ValueRange dynamicSizes,
                          Value copy) {
  build(builder, result, type, dynamicSizes, copy,
        /*memory_space=*/IntegerAttr());
}

namespace {
/// Change the type of the result of a `bufferization.alloc_tensor` by making
/// the result type statically sized along dimension that in the original
/// operation where defined as dynamic, but the size was defined using a
/// `constant` op. For example:
///
///  %c5 = arith.constant 5: index
///  %0 = bufferization.alloc_tensor(%arg0, %c5) : tensor<?x?xf32>
///
///  to
///
///  %0 = bufferization.alloc_tensor(%arg0) : tensor<?x5xf32>
struct ReplaceStaticShapeDims : OpRewritePattern<AllocTensorOp> {
  using OpRewritePattern<AllocTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocTensorOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCopy())
      return failure();
    SmallVector<int64_t> newShape = llvm::to_vector(op.getType().getShape());
    SmallVector<Value> newDynamicSizes;
    unsigned int dynValCounter = 0;
    for (int64_t i = 0; i < op.getType().getRank(); ++i) {
      if (!op.isDynamicDim(i))
        continue;
      Value value = op.getDynamicSizes()[dynValCounter++];
      APInt intVal;
      if (matchPattern(value, m_ConstantInt(&intVal))) {
        newShape[i] = intVal.getSExtValue();
      } else {
        newDynamicSizes.push_back(value);
      }
    }
    RankedTensorType newType = RankedTensorType::get(
        newShape, op.getType().getElementType(), op.getType().getEncoding());
    if (newType == op.getType())
      return failure();
    auto newOp = rewriter.create<AllocTensorOp>(
        op.getLoc(), newType, newDynamicSizes, /*copy=*/Value());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), newOp);
    return success();
  }
};

struct FoldDimOfAllocTensorOp : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    Optional<int64_t> maybeConstantIndex = dimOp.getConstantIndex();
    auto allocTensorOp = dimOp.getSource().getDefiningOp<AllocTensorOp>();
    if (!allocTensorOp || !maybeConstantIndex)
      return failure();
    if (!allocTensorOp.getType().isDynamicDim(*maybeConstantIndex))
      return failure();
    rewriter.replaceOp(
        dimOp, allocTensorOp.getDynamicSize(rewriter, *maybeConstantIndex));
    return success();
  }
};
} // namespace

void AllocTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *ctx) {
  results.add<FoldDimOfAllocTensorOp, ReplaceStaticShapeDims>(ctx);
}

LogicalResult AllocTensorOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  auto shapes = llvm::to_vector<4>(llvm::map_range(
      llvm::seq<int64_t>(0, getType().getRank()), [&](int64_t dim) -> Value {
        if (isDynamicDim(dim))
          return getDynamicSize(builder, dim);
        return builder.create<arith::ConstantIndexOp>(getLoc(),
                                                      getStaticSize(dim));
      }));
  reifiedReturnShapes.emplace_back(std::move(shapes));
  return success();
}

ParseResult AllocTensorOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizesOperands;
  if (parser.parseLParen() || parser.parseOperandList(dynamicSizesOperands) ||
      parser.parseRParen())
    return failure();
  ParseResult copyKeyword = parser.parseOptionalKeyword("copy");
  OpAsmParser::UnresolvedOperand copyOperand;
  if (copyKeyword.succeeded())
    if (parser.parseLParen() || parser.parseOperand(copyOperand) ||
        parser.parseRParen())
      return failure();
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();

  TensorType type;
  if (parser.parseCustomTypeWithFallback(type))
    return failure();
  result.addTypes(type);

  Type indexType = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(dynamicSizesOperands, indexType, result.operands))
    return failure();
  if (copyKeyword.succeeded())
    if (parser.resolveOperand(copyOperand, type, result.operands))
      return failure();
  result.addAttribute(AllocTensorOp::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(dynamicSizesOperands.size()),
                           static_cast<int32_t>(copyKeyword.succeeded())}));
  return success();
}

void AllocTensorOp::print(OpAsmPrinter &p) {
  p << "(" << getDynamicSizes() << ")";
  if (getCopy())
    p << " copy(" << getCopy() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{
                              AllocTensorOp::getOperandSegmentSizeAttr()});
  p << " : ";
  auto type = getResult().getType();
  if (auto validType = type.dyn_cast<::mlir::TensorType>())
    p.printStrippedAttrOrType(validType);
  else
    p << type;
}

Value AllocTensorOp::getDynamicSize(OpBuilder &b, unsigned idx) {
  assert(isDynamicDim(idx) && "expected dynamic dim");
  if (getCopy())
    return b.create<tensor::DimOp>(getLoc(), getCopy(), idx);
  return getOperand(getIndexOfDynamicSize(idx));
}

//===----------------------------------------------------------------------===//
// CloneOp
//===----------------------------------------------------------------------===//

void CloneOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getInput(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), getOutput(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(), getOutput(),
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

    Value source = cloneOp.getInput();

    // This only finds dealloc operations for the immediate value. It should
    // also consider aliases. That would also make the safety check below
    // redundant.
    llvm::Optional<Operation *> maybeCloneDeallocOp =
        memref::findDealloc(cloneOp.getOutput());
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
  if (auto toMemref = getMemref().getDefiningOp<ToMemrefOp>())
    // Approximate alias analysis by conservatively folding only when no there
    // is no interleaved operation.
    if (toMemref->getBlock() == this->getOperation()->getBlock() &&
        toMemref->getNextNode() == this->getOperation())
      return toMemref.getTensor();
  return {};
}

namespace {
struct DimOfToTensorFolder : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto memrefToTensorOp = dimOp.getSource().getDefiningOp<ToTensorOp>();
    if (!memrefToTensorOp)
      return failure();

    rewriter.replaceOpWithNewOp<memref::DimOp>(
        dimOp, memrefToTensorOp.getMemref(), dimOp.getIndex());
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
  if (auto memrefToTensor = getTensor().getDefiningOp<ToTensorOp>())
    if (memrefToTensor.getMemref().getType() == getType())
      return memrefToTensor.getMemref();
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

/// Canonicalize bufferization.to_tensor + bufferization.to_memref. Insert a
/// cast if necessary.
struct ToMemrefToTensorFolding : public OpRewritePattern<ToMemrefOp> {
  using OpRewritePattern<ToMemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToMemrefOp toMemref,
                                PatternRewriter &rewriter) const final {
    return foldToMemrefToTensorPair(rewriter, toMemref);
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

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(load, toMemref.getTensor(),
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
  results.add<DimOfCastOp, LoadOfToMemref, ToMemrefOfCast,
              ToMemrefToTensorFolding>(context);
}

LogicalResult ToMemrefOp::bufferize(RewriterBase &rewriter,
                                    const BufferizationOptions &options) {
  // Fold to_memref(to_tensor(x)) to x. Insert a cast if necessary.
  (void)foldToMemrefToTensorPair(rewriter, *this);
  // Note: The return value of `bufferize` indicates whether there was an error
  // or not. (And not whether the pattern matched or not.)
  return success();
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
