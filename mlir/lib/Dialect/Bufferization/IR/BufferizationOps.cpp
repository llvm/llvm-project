//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include <optional>

using namespace mlir;
using namespace mlir::bufferization;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

FailureOr<Value> mlir::bufferization::castOrReallocMemRefValue(
    OpBuilder &b, Value value, MemRefType destType,
    const BufferizationOptions &options) {
  auto srcType = llvm::cast<MemRefType>(value.getType());

  // Element type, rank and memory space must match.
  if (srcType.getElementType() != destType.getElementType())
    return failure();
  if (srcType.getMemorySpace() != destType.getMemorySpace())
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
      return ShapedType::isDynamic(a) && !ShapedType::isDynamic(b);
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
    if (destType.getShape()[i] != ShapedType::kDynamic)
      continue;
    Value size = b.create<memref::DimOp>(loc, value, i);
    dynamicOperands.push_back(size);
  }

  FailureOr<Value> copy =
      options.createAlloc(b, loc, destType, dynamicOperands);
  if (failed(copy))
    return failure();
  if (failed(options.createMemCpy(b, loc, value, *copy)))
    return failure();
  return copy;
}

/// Try to fold to_memref(to_tensor(x)). If x's type and the result type of the
/// to_memref op are different, a memref.cast is needed.
LogicalResult mlir::bufferization::foldToMemrefToTensorPair(
    RewriterBase &rewriter, ToMemrefOp toMemref,
    const BufferizationOptions &options) {
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

  auto rankedSrcType = llvm::dyn_cast<MemRefType>(srcType);
  auto rankedDestType = llvm::dyn_cast<MemRefType>(destType);
  auto unrankedSrcType = llvm::dyn_cast<UnrankedMemRefType>(srcType);

  // Ranked memref -> Ranked memref cast.
  if (rankedSrcType && rankedDestType) {
    FailureOr<Value> replacement = castOrReallocMemRefValue(
        rewriter, memrefToTensor.getMemref(), rankedDestType, options);
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
  auto shapedType = llvm::cast<ShapedType>(shapedValue.getType());
  for (int64_t i = 0; i < shapedType.getRank(); ++i) {
    if (shapedType.isDynamicDim(i)) {
      if (llvm::isa<MemRefType>(shapedType)) {
        dynamicDims.push_back(b.create<memref::DimOp>(loc, shapedValue, i));
      } else {
        assert(llvm::isa<RankedTensorType>(shapedType) && "expected tensor");
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

  // Create memory allocation.
  auto allocType = bufferization::getBufferType(getResult(), options);
  if (failed(allocType))
    return failure();
  SmallVector<Value> dynamicDims = getDynamicSizes();
  if (getCopy()) {
    assert(dynamicDims.empty() && "expected either `copy` or `dynamicDims`");
    populateDynamicDimSizes(rewriter, loc, copyBuffer, dynamicDims);
  }
  FailureOr<Value> alloc = options.createAlloc(
      rewriter, loc, llvm::cast<MemRefType>(*allocType), dynamicDims);
  if (failed(alloc))
    return failure();

  // Create memory copy (if any).
  if (getCopy()) {
    if (failed(options.createMemCpy(rewriter, loc, copyBuffer, *alloc)))
      return failure();
  }

  // Replace op.
  replaceOpWithBufferizedValues(rewriter, getOperation(), *alloc);

  return success();
}

bool AllocTensorOp::resultBufferizesToMemoryWrite(OpResult opResult,
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

AliasingValueList AllocTensorOp::getAliasingValues(OpOperand &opOperand,
                                                   const AnalysisState &state) {
  // This is a new allocation. It does not alias with any other buffer.
  return {};
}

FailureOr<BaseMemRefType>
AllocTensorOp::getBufferType(Value value, const BufferizationOptions &options,
                             SmallVector<Value> &invocationStack) {
  assert(value == getResult() && "invalid value");

  // Compute memory space of this allocation.
  Attribute memorySpace;
  if (getMemorySpace().has_value()) {
    memorySpace = *getMemorySpace();
  } else if (getCopy()) {
    auto copyBufferType =
        bufferization::getBufferType(getCopy(), options, invocationStack);
    if (failed(copyBufferType))
      return failure();
    memorySpace = copyBufferType->getMemorySpace();
  } else if (auto ms = options.defaultMemorySpaceFn(getType())) {
    memorySpace = *ms;
  } else {
    return getOperation()->emitError("could not infer memory space");
  }

  return getMemRefTypeWithStaticIdentityLayout(getType(), memorySpace);
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
  return success();
}

void AllocTensorOp::build(OpBuilder &builder, OperationState &result,
                          RankedTensorType type, ValueRange dynamicSizes) {
  build(builder, result, type, dynamicSizes, /*copy=*/Value(),
        /*size_hint=*/Value(),
        /*memory_space=*/IntegerAttr());
}

void AllocTensorOp::build(OpBuilder &builder, OperationState &result,
                          RankedTensorType type, ValueRange dynamicSizes,
                          Value copy) {
  build(builder, result, type, dynamicSizes, copy, /*size_hint=*/Value(),
        /*memory_space=*/IntegerAttr());
}

void AllocTensorOp::build(OpBuilder &builder, OperationState &result,
                          TensorType type, ValueRange dynamicSizes, Value copy,
                          IntegerAttr memorySpace) {
  build(builder, result, type, dynamicSizes, copy, /*size_hint=*/Value(),
        memorySpace);
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
        int64_t dim = intVal.getSExtValue();
        if (dim >= 0)
          newShape[i] = intVal.getSExtValue();
        else
          newDynamicSizes.push_back(value);
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
    std::optional<int64_t> maybeConstantIndex = dimOp.getConstantIndex();
    auto allocTensorOp = dimOp.getSource().getDefiningOp<AllocTensorOp>();
    if (!allocTensorOp || !maybeConstantIndex)
      return failure();
    if (*maybeConstantIndex < 0 ||
        *maybeConstantIndex >= allocTensorOp.getType().getRank())
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
  auto shapes = llvm::to_vector<4>(
      llvm::map_range(llvm::seq<int64_t>(0, getType().getRank()),
                      [&](int64_t dim) -> OpFoldResult {
                        if (isDynamicDim(dim))
                          return getDynamicSize(builder, dim);
                        return builder.getIndexAttr(getStaticSize(dim));
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
  ParseResult sizeHintKeyword = parser.parseOptionalKeyword("size_hint");
  OpAsmParser::UnresolvedOperand sizeHintOperand;
  if (sizeHintKeyword.succeeded())
    if (parser.parseEqual() || parser.parseOperand(sizeHintOperand))
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
  if (sizeHintKeyword.succeeded())
    if (parser.resolveOperand(sizeHintOperand, indexType, result.operands))
      return failure();
  result.addAttribute(AllocTensorOp::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(dynamicSizesOperands.size()),
                           static_cast<int32_t>(copyKeyword.succeeded()),
                           static_cast<int32_t>(sizeHintKeyword.succeeded())}));
  return success();
}

void AllocTensorOp::print(OpAsmPrinter &p) {
  p << "(" << getDynamicSizes() << ")";
  if (getCopy())
    p << " copy(" << getCopy() << ")";
  if (getSizeHint())
    p << " size_hint=" << getSizeHint();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{
                              AllocTensorOp::getOperandSegmentSizeAttr()});
  p << " : ";
  auto type = getResult().getType();
  if (auto validType = llvm::dyn_cast<::mlir::TensorType>(type))
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

OpFoldResult CloneOp::fold(FoldAdaptor adaptor) {
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
    if (source.getType() != cloneOp.getType() &&
        !memref::CastOp::areCastCompatible({source.getType()},
                                           {cloneOp.getType()}))
      return failure();

    // Aims to find the dealloc op for the canonical source
    // which otherwise could prevent removal of unnecessary allocs.
    Value canonicalSource = source;
    while (auto iface = dyn_cast_or_null<ViewLikeOpInterface>(
               canonicalSource.getDefiningOp()))
      canonicalSource = iface.getViewSource();

    std::optional<Operation *> maybeCloneDeallocOp =
        memref::findDealloc(cloneOp.getOutput());
    // Skip if either of them has > 1 deallocate operations.
    if (!maybeCloneDeallocOp.has_value())
      return failure();
    std::optional<Operation *> maybeSourceDeallocOp =
        memref::findDealloc(canonicalSource);
    if (!maybeSourceDeallocOp.has_value())
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
      // Bail if we run out of operations while looking for a deallocation op.
      if (!pos)
        return failure();
      auto effectInterface = dyn_cast<MemoryEffectOpInterface>(pos);
      if (!effectInterface)
        continue;
      if (effectInterface.hasEffect<MemoryEffects::Free>())
        return failure();
    }

    if (source.getType() != cloneOp.getType())
      source = rewriter.create<memref::CastOp>(cloneOp.getLoc(),
                                               cloneOp.getType(), source);
    rewriter.replaceOp(cloneOp, source);
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
// DeallocTensorOp
//===----------------------------------------------------------------------===//

LogicalResult DeallocTensorOp::bufferize(RewriterBase &rewriter,
                                         const BufferizationOptions &options) {
  FailureOr<Value> buffer = getBuffer(rewriter, getTensor(), options);
  if (failed(buffer))
    return failure();
  rewriter.create<memref::DeallocOp>(getLoc(), *buffer);
  rewriter.eraseOp(getOperation());
  return success();
}

//===----------------------------------------------------------------------===//
// MaterializeInDestinationOp
//===----------------------------------------------------------------------===//

bool MaterializeInDestinationOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const AnalysisState &state) {
  return opOperand == getSourceMutable();
}

bool MaterializeInDestinationOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const AnalysisState &state) {
  if (opOperand == getDestMutable()) {
    assert(isa<TensorType>(getDest().getType()) && "expected tensor type");
    return true;
  }
  return false;
}

bool MaterializeInDestinationOp::mustBufferizeInPlace(
    OpOperand &opOperand, const AnalysisState &state) {
  // The source is only read and not written, so it always bufferizes in-place
  // by default. The destination is written and is forced to bufferize in-place
  // (if it is a tensor).
  return true;
}

AliasingValueList
MaterializeInDestinationOp::getAliasingValues(OpOperand &opOperand,
                                              const AnalysisState &state) {
  if (opOperand == getDestMutable()) {
    assert(isa<TensorType>(getDest().getType()) && "expected tensor type");
    return {{getOperation()->getResult(0), BufferRelation::Equivalent}};
  }
  return {};
}

LogicalResult
MaterializeInDestinationOp::bufferize(RewriterBase &rewriter,
                                      const BufferizationOptions &options) {
  bool tensorDest = isa<TensorType>(getDest().getType());
  Value buffer;
  if (tensorDest) {
    FailureOr<Value> maybeBuffer = getBuffer(rewriter, getDest(), options);
    if (failed(maybeBuffer))
      return failure();
    buffer = *maybeBuffer;
  } else {
    assert(isa<BaseMemRefType>(getDest().getType()) && "expected memref type");
    buffer = getDest();
  }
  auto srcBuffer = getBuffer(rewriter, getSource(), options);
  if (failed(srcBuffer))
    return failure();
  if (failed(options.createMemCpy(rewriter, getLoc(), *srcBuffer, buffer)))
    return failure();
  replaceOpWithBufferizedValues(rewriter, getOperation(),
                                tensorDest ? ValueRange(buffer) : ValueRange());
  return success();
}

bool MaterializeInDestinationOp::bufferizesToElementwiseAccess(
    const AnalysisState &state, ArrayRef<OpOperand *> opOperands) {
  // As elements are copied from the "source" buffer to the "dest" buffer,
  // already copied elements are not read a second time.
  return true;
}

LogicalResult MaterializeInDestinationOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getOperation()->getNumResults() == 1) {
    assert(isa<TensorType>(getDest().getType()) && "expected tensor type");
    reifiedReturnShapes.resize(1,
                               SmallVector<OpFoldResult>(getType().getRank()));
    reifiedReturnShapes[0] =
        tensor::getMixedSizes(builder, getLoc(), getDest());
  }
  return success();
}

Value MaterializeInDestinationOp::buildSubsetExtraction(OpBuilder &builder,
                                                        Location loc) {
  if (isa<TensorType>(getDest().getType())) {
    // The subset is the entire destination tensor.
    return getDest();
  }

  // The "restrict" attribute is transferred from this op to the newly created
  // to_tensor op. If this op does not the "restrict" attribute, the subset
  // extraction cannot be built because there is no guarantee that there is no
  // pre-existing "restrict" to_tensor op with the same/an aliasing destination.
  if (!getRestrict())
    return {};

  // Build a bufferization.to_tensor op.
  assert(isa<BaseMemRefType>(getDest().getType()) && "expected memref type");
  assert(getRestrict() &&
         "expected that ops with memrefs dest have 'restrict'");
  setRestrict(false);
  return builder.create<ToTensorOp>(loc, getDest(), /*restrict=*/true,
                                    getWritable());
}

bool MaterializeInDestinationOp::isEquivalentSubset(
    Value candidate, function_ref<bool(Value, Value)> equivalenceFn) {
  return equivalenceFn(getDest(), candidate);
}

SmallVector<Value>
MaterializeInDestinationOp::getValuesNeededToBuildSubsetExtraction() {
  return {getDest()};
}

OpOperand &MaterializeInDestinationOp::getSourceOperand() {
  return getOperation()->getOpOperand(0) /*source*/;
}

bool MaterializeInDestinationOp::operatesOnEquivalentSubset(
    SubsetOpInterface subsetOp,
    function_ref<bool(Value, Value)> equivalenceFn) {
  return false;
}

bool MaterializeInDestinationOp::operatesOnDisjointSubset(
    SubsetOpInterface subsetOp,
    function_ref<bool(Value, Value)> equivalenceFn) {
  return false;
}

LogicalResult MaterializeInDestinationOp::verify() {
  if (!isa<TensorType, BaseMemRefType>(getDest().getType()))
    return emitOpError("'dest' must be a tensor or a memref");
  if (auto destType = dyn_cast<TensorType>(getDest().getType())) {
    if (getOperation()->getNumResults() != 1)
      return emitOpError("tensor 'dest' implies exactly one tensor result");
    if (destType != getResult().getType())
      return emitOpError("result and 'dest' types must match");
  }
  if (isa<BaseMemRefType>(getDest().getType()) &&
      getOperation()->getNumResults() != 0)
    return emitOpError("memref 'dest' implies zero results");
  if (getRestrict() && !isa<BaseMemRefType>(getDest().getType()))
    return emitOpError("'restrict' is valid only for memref destinations");
  if (getWritable() != isa<BaseMemRefType>(getDest().getType()))
    return emitOpError("'writable' must be specified if and only if the "
                       "destination is of memref type");
  TensorType srcType = getSource().getType();
  ShapedType destType = cast<ShapedType>(getDest().getType());
  if (srcType.hasRank() != destType.hasRank())
    return emitOpError("source/destination shapes are incompatible");
  if (srcType.hasRank()) {
    if (srcType.getRank() != destType.getRank())
      return emitOpError("rank mismatch between source and destination shape");
    for (auto [src, dest] :
         llvm::zip(srcType.getShape(), destType.getShape())) {
      if (src == ShapedType::kDynamic || dest == ShapedType::kDynamic) {
        // Cannot verify dynamic dimension size. Assume that that they match at
        // runtime.
        continue;
      }
      if (src != dest)
        return emitOpError("source/destination shapes are incompatible");
    }
  }
  return success();
}

void MaterializeInDestinationOp::build(OpBuilder &builder,
                                       OperationState &state, Value source,
                                       Value dest) {
  auto destTensorType = dyn_cast<TensorType>(dest.getType());
  build(builder, state, /*result=*/destTensorType ? destTensorType : Type(),
        source, dest);
}

bool MaterializeInDestinationOp::isWritable(Value value,
                                            const AnalysisState &state) {
  return isa<TensorType>(getDest().getType()) ? true : getWritable();
}

MutableOperandRange MaterializeInDestinationOp::getDpsInitsMutable() {
  return getDestMutable();
}

void MaterializeInDestinationOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isa<BaseMemRefType>(getDest().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable(),
                         SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ToTensorOp
//===----------------------------------------------------------------------===//

bool ToTensorOp::isWritable(Value value, const AnalysisState &state) {
  return getWritable();
}

OpFoldResult ToTensorOp::fold(FoldAdaptor) {
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

OpFoldResult ToMemrefOp::fold(FoldAdaptor) {
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
    auto srcTensorType = llvm::dyn_cast<RankedTensorType>(
        tensorCastOperand.getOperand().getType());
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
    BufferizationOptions options;
    options.bufferAlignment = 0;
    return foldToMemrefToTensorPair(rewriter, toMemref, options);
  }
};

/// Fold a load on a to_memref operation into an tensor.extract on the
/// corresponding tensor.
struct LoadOfToMemref : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load,
                                PatternRewriter &rewriter) const override {
    auto toMemref = load.getMemref().getDefiningOp<ToMemrefOp>();
    if (!toMemref)
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(load, toMemref.getTensor(),
                                                   load.getIndices());
    return success();
  }
};

/// Fold dim of a to_memref into the dim of the tensor.
struct DimOfCastOp : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern<memref::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = dimOp.getSource().getDefiningOp<ToMemrefOp>();
    if (!castOp)
      return failure();
    Value newSource = castOp.getOperand();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(dimOp, newSource,
                                               dimOp.getIndex());
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
  (void)foldToMemrefToTensorPair(rewriter, *this, options);
  // Note: The return value of `bufferize` indicates whether there was an error
  // or not. (And not whether the pattern matched or not.)
  return success();
}

std::optional<Operation *> CloneOp::buildDealloc(OpBuilder &builder,
                                                 Value alloc) {
  return builder.create<memref::DeallocOp>(alloc.getLoc(), alloc)
      .getOperation();
}

std::optional<Value> CloneOp::buildClone(OpBuilder &builder, Value alloc) {
  return builder.create<CloneOp>(alloc.getLoc(), alloc).getResult();
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

LogicalResult DeallocOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  DeallocOpAdaptor adaptor(operands, attributes, properties, regions);
  inferredReturnTypes = SmallVector<Type>(adaptor.getRetained().size(),
                                          IntegerType::get(context, 1));
  return success();
}

LogicalResult DeallocOp::verify() {
  if (getMemrefs().size() != getConditions().size())
    return emitOpError(
        "must have the same number of conditions as memrefs to deallocate");
  if (getRetained().size() != getUpdatedConditions().size())
    return emitOpError("must have the same number of updated conditions "
                       "(results) as retained operands");
  return success();
}

static LogicalResult updateDeallocIfChanged(DeallocOp deallocOp,
                                            ValueRange memrefs,
                                            ValueRange conditions,
                                            PatternRewriter &rewriter) {
  if (deallocOp.getMemrefs() == memrefs &&
      deallocOp.getConditions() == conditions)
    return failure();

  rewriter.modifyOpInPlace(deallocOp, [&]() {
    deallocOp.getMemrefsMutable().assign(memrefs);
    deallocOp.getConditionsMutable().assign(conditions);
  });
  return success();
}

namespace {

/// Remove duplicate values in the list of memrefs to be deallocated. We need to
/// make sure the corresponding condition value is updated accordingly since
/// their two conditions might not cover the same set of cases. In that case, we
/// have to combine them (by computing the disjunction of them).
/// Example:
/// ```mlir
/// bufferization.dealloc (%arg0, %arg0 : ...) if (%arg1, %arg2)
/// ```
/// is canonicalized to
/// ```mlir
/// %0 = arith.ori %arg1, %arg2 : i1
/// bufferization.dealloc (%arg0 : memref<2xi32>) if (%0)
/// ```
struct DeallocRemoveDuplicateDeallocMemrefs
    : public OpRewritePattern<DeallocOp> {
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    // Unique memrefs to be deallocated.
    DenseMap<Value, unsigned> memrefToCondition;
    SmallVector<Value> newMemrefs, newConditions;
    for (auto [i, memref, cond] :
         llvm::enumerate(deallocOp.getMemrefs(), deallocOp.getConditions())) {
      if (memrefToCondition.count(memref)) {
        // If the dealloc conditions don't match, we need to make sure that the
        // dealloc happens on the union of cases.
        Value &newCond = newConditions[memrefToCondition[memref]];
        if (newCond != cond)
          newCond =
              rewriter.create<arith::OrIOp>(deallocOp.getLoc(), newCond, cond);
      } else {
        memrefToCondition.insert({memref, newConditions.size()});
        newMemrefs.push_back(memref);
        newConditions.push_back(cond);
      }
    }

    // Return failure if we don't change anything such that we don't run into an
    // infinite loop of pattern applications.
    return updateDeallocIfChanged(deallocOp, newMemrefs, newConditions,
                                  rewriter);
  }
};

/// Remove duplicate values in the list of retained memrefs. We need to make
/// sure the corresponding result condition value is replaced properly.
/// Example:
/// ```mlir
/// %0:2 = bufferization.dealloc retain (%arg3, %arg3 : ...)
/// ```
/// is canonicalized to
/// ```mlir
/// %0 = bufferization.dealloc retain (%arg3 : memref<2xi32>)
/// ```
struct DeallocRemoveDuplicateRetainedMemrefs
    : public OpRewritePattern<DeallocOp> {
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    // Unique retained values
    DenseMap<Value, unsigned> seen;
    SmallVector<Value> newRetained;
    SmallVector<unsigned> resultReplacementIdx;
    unsigned i = 0;
    for (auto retained : deallocOp.getRetained()) {
      if (seen.count(retained)) {
        resultReplacementIdx.push_back(seen[retained]);
        continue;
      }

      seen[retained] = i;
      newRetained.push_back(retained);
      resultReplacementIdx.push_back(i++);
    }

    // Return failure if we don't change anything such that we don't run into an
    // infinite loop of pattern applications.
    if (newRetained.size() == deallocOp.getRetained().size())
      return failure();

    // We need to create a new op because the number of results is always the
    // same as the number of condition operands.
    auto newDeallocOp =
        rewriter.create<DeallocOp>(deallocOp.getLoc(), deallocOp.getMemrefs(),
                                   deallocOp.getConditions(), newRetained);
    SmallVector<Value> replacements(
        llvm::map_range(resultReplacementIdx, [&](unsigned idx) {
          return newDeallocOp.getUpdatedConditions()[idx];
        }));
    rewriter.replaceOp(deallocOp, replacements);
    return success();
  }
};

/// Erase deallocation operations where the variadic list of memrefs to
/// deallocate is empty. Example:
/// ```mlir
/// %0 = bufferization.dealloc retain (%arg0: memref<2xi32>)
/// ```
struct EraseEmptyDealloc : public OpRewritePattern<DeallocOp> {
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    if (deallocOp.getMemrefs().empty()) {
      Value constFalse = rewriter.create<arith::ConstantOp>(
          deallocOp.getLoc(), rewriter.getBoolAttr(false));
      rewriter.replaceOp(
          deallocOp, SmallVector<Value>(deallocOp.getUpdatedConditions().size(),
                                        constFalse));
      return success();
    }
    return failure();
  }
};

/// Removes memrefs from the deallocation list if their associated condition is
/// always 'false'.
///
/// Example:
/// ```
/// bufferization.dealloc (%arg0, %arg1 : memref<2xi32>, memref<2xi32>)
///                           if (%arg2, %false)
/// ```
/// becomes
/// ```
/// bufferization.dealloc (%arg0 : memref<2xi32>) if (%arg2)
/// ```
struct EraseAlwaysFalseDealloc : public OpRewritePattern<DeallocOp> {
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newMemrefs, newConditions;
    for (auto [memref, cond] :
         llvm::zip(deallocOp.getMemrefs(), deallocOp.getConditions())) {
      if (!matchPattern(cond, m_Zero())) {
        newMemrefs.push_back(memref);
        newConditions.push_back(cond);
      }
    }

    return updateDeallocIfChanged(deallocOp, newMemrefs, newConditions,
                                  rewriter);
  }
};

/// The `memref.extract_strided_metadata` is often inserted to get the base
/// memref if the operand is not already guaranteed to be the result of a memref
/// allocation operation. This canonicalization pattern removes this extraction
/// operation if the operand is now produced by an allocation operation (e.g.,
/// due to other canonicalizations simplifying the IR).
///
/// Example:
/// ```mlir
/// %alloc = memref.alloc() : memref<2xi32>
/// %base_memref, %offset, %size, %stride = memref.extract_strided_metadata
///   %alloc : memref<2xi32> -> memref<i32>, index, index, index
/// bufferization.dealloc (%base_memref : memref<i32>) if (%cond)
/// ```
/// is canonicalized to
/// ```mlir
/// %alloc = memref.alloc() : memref<2xi32>
/// bufferization.dealloc (%alloc : memref<2xi32>) if (%cond)
/// ```
struct SkipExtractMetadataOfAlloc : public OpRewritePattern<DeallocOp> {
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newMemrefs(
        llvm::map_range(deallocOp.getMemrefs(), [&](Value memref) {
          auto extractStridedOp =
              memref.getDefiningOp<memref::ExtractStridedMetadataOp>();
          if (!extractStridedOp)
            return memref;
          Value allocMemref = extractStridedOp.getOperand();
          auto allocOp = allocMemref.getDefiningOp<MemoryEffectOpInterface>();
          if (!allocOp)
            return memref;
          if (allocOp.getEffectOnValue<MemoryEffects::Allocate>(allocMemref))
            return allocMemref;
          return memref;
        }));

    return updateDeallocIfChanged(deallocOp, newMemrefs,
                                  deallocOp.getConditions(), rewriter);
  }
};

/// Removes pairs of `bufferization.dealloc` and alloc operations if there is no
/// other user of the allocated value and the allocating operation can be safely
/// removed. If the same value is present multiple times, this pattern relies on
/// other canonicalization patterns to remove the duplicate first.
///
/// Example:
/// ```mlir
/// %alloc = memref.alloc() : memref<2xi32>
/// bufferization.dealloc (%alloc, %arg0, : ...) if (%true, %true)
/// ```
/// is canonicalized to
/// ```mlir
/// bufferization.dealloc (%arg0 : ...) if (%true)
/// ```
struct RemoveAllocDeallocPairWhenNoOtherUsers
    : public OpRewritePattern<DeallocOp> {
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newMemrefs, newConditions;
    SmallVector<Operation *> toDelete;
    for (auto [memref, cond] :
         llvm::zip(deallocOp.getMemrefs(), deallocOp.getConditions())) {
      if (auto allocOp = memref.getDefiningOp<MemoryEffectOpInterface>()) {
        // Check that it is indeed an allocate effect, that the op has no other
        // side effects (which would not allow us to remove the op), and that
        // there are no other users.
        if (allocOp.getEffectOnValue<MemoryEffects::Allocate>(memref) &&
            hasSingleEffect<MemoryEffects::Allocate>(allocOp, memref) &&
            memref.hasOneUse()) {
          toDelete.push_back(allocOp);
          continue;
        }
      }

      newMemrefs.push_back(memref);
      newConditions.push_back(cond);
    }

    if (failed(updateDeallocIfChanged(deallocOp, newMemrefs, newConditions,
                                      rewriter)))
      return failure();

    for (Operation *op : toDelete)
      rewriter.eraseOp(op);

    return success();
  }
};

} // anonymous namespace

void DeallocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  populateDeallocOpCanonicalizationPatterns(results, context);
}

void bufferization::populateDeallocOpCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<DeallocRemoveDuplicateDeallocMemrefs,
               DeallocRemoveDuplicateRetainedMemrefs, EraseEmptyDealloc,
               EraseAlwaysFalseDealloc, SkipExtractMetadataOfAlloc,
               RemoveAllocDeallocPairWhenNoOtherUsers>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Bufferization/IR/BufferizationOps.cpp.inc"
