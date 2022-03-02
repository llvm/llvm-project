//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;
using namespace mlir::memref;

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *MemRefDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  if (arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<arith::ConstantOp>(loc, value, type);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Common canonicalization pattern support logic
//===----------------------------------------------------------------------===//

/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref.cast
/// into the root operation directly.
LogicalResult mlir::memref::foldMemRefCast(Operation *op, Value inner) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto cast = operand.get().getDefiningOp<CastOp>();
    if (cast && operand.get() != inner &&
        !cast.getOperand().getType().isa<UnrankedMemRefType>()) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

/// Return an unranked/ranked tensor type for the given unranked/ranked memref
/// type.
Type mlir::memref::getTensorTypeFromMemRefType(Type type) {
  if (auto memref = type.dyn_cast<MemRefType>())
    return RankedTensorType::get(memref.getShape(), memref.getElementType());
  if (auto memref = type.dyn_cast<UnrankedMemRefType>())
    return UnrankedTensorType::get(memref.getElementType());
  return NoneType::get(type.getContext());
}

//===----------------------------------------------------------------------===//
// AllocOp / AllocaOp
//===----------------------------------------------------------------------===//

template <typename AllocLikeOp>
static LogicalResult verifyAllocLikeOp(AllocLikeOp op) {
  static_assert(llvm::is_one_of<AllocLikeOp, AllocOp, AllocaOp>::value,
                "applies to only alloc or alloca");
  auto memRefType = op.getResult().getType().template dyn_cast<MemRefType>();
  if (!memRefType)
    return op.emitOpError("result must be a memref");

  if (static_cast<int64_t>(op.dynamicSizes().size()) !=
      memRefType.getNumDynamicDims())
    return op.emitOpError("dimension operand count does not equal memref "
                          "dynamic dimension count");

  unsigned numSymbols = 0;
  if (!memRefType.getLayout().isIdentity())
    numSymbols = memRefType.getLayout().getAffineMap().getNumSymbols();
  if (op.symbolOperands().size() != numSymbols)
    return op.emitOpError("symbol operand count does not equal memref symbol "
                          "count: expected ")
           << numSymbols << ", got " << op.symbolOperands().size();

  return success();
}

LogicalResult AllocOp::verify() { return verifyAllocLikeOp(*this); }

LogicalResult AllocaOp::verify() {
  // An alloca op needs to have an ancestor with an allocation scope trait.
  if (!(*this)->getParentWithTrait<OpTrait::AutomaticAllocationScope>())
    return emitOpError(
        "requires an ancestor op with AutomaticAllocationScope trait");

  return verifyAllocLikeOp(*this);
}

namespace {
/// Fold constant dimensions into an alloc like operation.
template <typename AllocLikeOp>
struct SimplifyAllocConst : public OpRewritePattern<AllocLikeOp> {
  using OpRewritePattern<AllocLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocLikeOp alloc,
                                PatternRewriter &rewriter) const override {
    // Check to see if any dimensions operands are constants.  If so, we can
    // substitute and drop them.
    if (llvm::none_of(alloc.dynamicSizes(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    auto memrefType = alloc.getType();

    // Ok, we have one or more constant operands.  Collect the non-constant ones
    // and keep track of the resultant memref type to build.
    SmallVector<int64_t, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType.getRank());
    SmallVector<Value, 4> dynamicSizes;

    unsigned dynamicDimPos = 0;
    for (unsigned dim = 0, e = memrefType.getRank(); dim < e; ++dim) {
      int64_t dimSize = memrefType.getDimSize(dim);
      // If this is already static dimension, keep it.
      if (dimSize != -1) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto dynamicSize = alloc.dynamicSizes()[dynamicDimPos];
      auto *defOp = dynamicSize.getDefiningOp();
      if (auto constantIndexOp =
              dyn_cast_or_null<arith::ConstantIndexOp>(defOp)) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp.value());
      } else {
        // Dynamic shape dimension not folded; copy dynamicSize from old memref.
        newShapeConstants.push_back(-1);
        dynamicSizes.push_back(dynamicSize);
      }
      dynamicDimPos++;
    }

    // Create new memref type (which will have fewer dynamic dimensions).
    MemRefType newMemRefType =
        MemRefType::Builder(memrefType).setShape(newShapeConstants);
    assert(static_cast<int64_t>(dynamicSizes.size()) ==
           newMemRefType.getNumDynamicDims());

    // Create and insert the alloc op for the new memref.
    auto newAlloc = rewriter.create<AllocLikeOp>(
        alloc.getLoc(), newMemRefType, dynamicSizes, alloc.symbolOperands(),
        alloc.alignmentAttr());
    // Insert a cast so we have the same type as the old alloc.
    auto resultCast =
        rewriter.create<CastOp>(alloc.getLoc(), alloc.getType(), newAlloc);

    rewriter.replaceOp(alloc, {resultCast});
    return success();
  }
};

/// Fold alloc operations with no users or only store and dealloc uses.
template <typename T>
struct SimplifyDeadAlloc : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T alloc,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(alloc->getUsers(), [&](Operation *op) {
          if (auto storeOp = dyn_cast<StoreOp>(op))
            return storeOp.value() == alloc;
          return !isa<DeallocOp>(op);
        }))
      return failure();

    for (Operation *user : llvm::make_early_inc_range(alloc->getUsers()))
      rewriter.eraseOp(user);

    rewriter.eraseOp(alloc);
    return success();
  }
};
} // namespace

void AllocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<SimplifyAllocConst<AllocOp>, SimplifyDeadAlloc<AllocOp>>(context);
}

void AllocaOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<SimplifyAllocConst<AllocaOp>, SimplifyDeadAlloc<AllocaOp>>(
      context);
}

//===----------------------------------------------------------------------===//
// AllocaScopeOp
//===----------------------------------------------------------------------===//

void AllocaScopeOp::print(OpAsmPrinter &p) {
  bool printBlockTerminators = false;

  p << ' ';
  if (!results().empty()) {
    p << " -> (" << getResultTypes() << ")";
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(bodyRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult AllocaScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create a region for the body.
  result.regions.reserve(1);
  Region *bodyRegion = result.addRegion();

  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Parse the body region.
  if (parser.parseRegion(*bodyRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  AllocaScopeOp::ensureTerminator(*bodyRegion, parser.getBuilder(),
                                  result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

LogicalResult AllocaScopeOp::verify() {
  return RegionBranchOpInterface::verifyTypes(*this);
}

void AllocaScopeOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  regions.push_back(RegionSuccessor(&bodyRegion()));
}

/// Given an operation, return whether this op is guaranteed to
/// allocate an AutomaticAllocationScopeResource
static bool isGuaranteedAutomaticAllocationScope(Operation *op) {
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return false;
  for (auto res : op->getResults()) {
    if (auto effect =
            interface.getEffectOnValue<MemoryEffects::Allocate>(res)) {
      if (isa<SideEffects::AutomaticAllocationScopeResource>(
              effect->getResource()))
        return true;
    }
  }
  return false;
}

/// Given an operation, return whether this op could to
/// allocate an AutomaticAllocationScopeResource
static bool isPotentialAutomaticAllocationScope(Operation *op) {
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return true;
  for (auto res : op->getResults()) {
    if (auto effect =
            interface.getEffectOnValue<MemoryEffects::Allocate>(res)) {
      if (isa<SideEffects::AutomaticAllocationScopeResource>(
              effect->getResource()))
        return true;
    }
  }
  return false;
}

/// Return whether this op is the last non terminating op
/// in a region. That is to say, it is in a one-block region
/// and is only followed by a terminator. This prevents
/// extending the lifetime of allocations.
static bool lastNonTerminatorInRegion(Operation *op) {
  return op->getNextNode() == op->getBlock()->getTerminator() &&
         op->getParentRegion()->getBlocks().size() == 1;
}

/// Inline an AllocaScopeOp if either the direct parent is an allocation scope
/// or it contains no allocation.
struct AllocaScopeInliner : public OpRewritePattern<AllocaScopeOp> {
  using OpRewritePattern<AllocaScopeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocaScopeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->getParentOp()->hasTrait<OpTrait::AutomaticAllocationScope>()) {
      bool hasPotentialAlloca =
          op->walk([&](Operation *alloc) {
              if (isPotentialAutomaticAllocationScope(alloc))
                return WalkResult::interrupt();
              return WalkResult::skip();
            }).wasInterrupted();
      if (hasPotentialAlloca)
        return failure();
    }

    // Only apply to if this is this last non-terminator
    // op in the block (lest lifetime be extended) of a one
    // block region
    if (!lastNonTerminatorInRegion(op))
      return failure();

    Block *block = &op.getRegion().front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.mergeBlockBefore(block, op);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);
    return success();
  }
};

/// Move allocations into an allocation scope, if it is legal to
/// move them (e.g. their operands are available at the location
/// the op would be moved to).
struct AllocaScopeHoister : public OpRewritePattern<AllocaScopeOp> {
  using OpRewritePattern<AllocaScopeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocaScopeOp op,
                                PatternRewriter &rewriter) const override {

    if (!op->getParentWithTrait<OpTrait::AutomaticAllocationScope>())
      return failure();

    Operation *lastParentWithoutScope = op->getParentOp();

    if (!lastParentWithoutScope ||
        lastParentWithoutScope->hasTrait<OpTrait::AutomaticAllocationScope>())
      return failure();

    // Only apply to if this is this last non-terminator
    // op in the block (lest lifetime be extended) of a one
    // block region
    if (!lastNonTerminatorInRegion(op) ||
        !lastNonTerminatorInRegion(lastParentWithoutScope))
      return failure();

    while (!lastParentWithoutScope->getParentOp()
                ->hasTrait<OpTrait::AutomaticAllocationScope>()) {
      lastParentWithoutScope = lastParentWithoutScope->getParentOp();
      if (!lastParentWithoutScope ||
          !lastNonTerminatorInRegion(lastParentWithoutScope))
        return failure();
    }
    assert(lastParentWithoutScope->getParentOp()
               ->hasTrait<OpTrait::AutomaticAllocationScope>());

    Region *containingRegion = nullptr;
    for (auto &r : lastParentWithoutScope->getRegions()) {
      if (r.isAncestor(op->getParentRegion())) {
        assert(containingRegion == nullptr &&
               "only one region can contain the op");
        containingRegion = &r;
      }
    }
    assert(containingRegion && "op must be contained in a region");

    SmallVector<Operation *> toHoist;
    op->walk([&](Operation *alloc) {
      if (!isGuaranteedAutomaticAllocationScope(alloc))
        return WalkResult::skip();

      // If any operand is not defined before the location of
      // lastParentWithoutScope (i.e. where we would hoist to), skip.
      if (llvm::any_of(alloc->getOperands(), [&](Value v) {
            return containingRegion->isAncestor(v.getParentRegion());
          }))
        return WalkResult::skip();
      toHoist.push_back(alloc);
      return WalkResult::advance();
    });

    if (!toHoist.size())
      return failure();
    rewriter.setInsertionPoint(lastParentWithoutScope);
    for (auto op : toHoist) {
      auto cloned = rewriter.clone(*op);
      rewriter.replaceOp(op, cloned->getResults());
    }
    return success();
  }
};

void AllocaScopeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<AllocaScopeInliner, AllocaScopeHoister>(context);
}

//===----------------------------------------------------------------------===//
// AssumeAlignmentOp
//===----------------------------------------------------------------------===//

LogicalResult AssumeAlignmentOp::verify() {
  if (!llvm::isPowerOf2_32(alignment()))
    return emitOpError("alignment must be power of 2");
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

/// Determines whether MemRef_CastOp casts to a more dynamic version of the
/// source memref. This is useful to to fold a memref.cast into a consuming op
/// and implement canonicalization patterns for ops in different dialects that
/// may consume the results of memref.cast operations. Such foldable memref.cast
/// operations are typically inserted as `view` and `subview` ops are
/// canonicalized, to preserve the type compatibility of their uses.
///
/// Returns true when all conditions are met:
/// 1. source and result are ranked memrefs with strided semantics and same
/// element type and rank.
/// 2. each of the source's size, offset or stride has more static information
/// than the corresponding result's size, offset or stride.
///
/// Example 1:
/// ```mlir
///   %1 = memref.cast %0 : memref<8x16xf32> to memref<?x?xf32>
///   %2 = consumer %1 ... : memref<?x?xf32> ...
/// ```
///
/// may fold into:
///
/// ```mlir
///   %2 = consumer %0 ... : memref<8x16xf32> ...
/// ```
///
/// Example 2:
/// ```
///   %1 = memref.cast %0 : memref<?x16xf32, affine_map<(i, j)->(16 * i + j)>>
///          to memref<?x?xf32>
///   consumer %1 : memref<?x?xf32> ...
/// ```
///
/// may fold into:
///
/// ```
///   consumer %0 ... : memref<?x16xf32, affine_map<(i, j)->(16 * i + j)>>
/// ```
bool CastOp::canFoldIntoConsumerOp(CastOp castOp) {
  MemRefType sourceType = castOp.source().getType().dyn_cast<MemRefType>();
  MemRefType resultType = castOp.getType().dyn_cast<MemRefType>();

  // Requires ranked MemRefType.
  if (!sourceType || !resultType)
    return false;

  // Requires same elemental type.
  if (sourceType.getElementType() != resultType.getElementType())
    return false;

  // Requires same rank.
  if (sourceType.getRank() != resultType.getRank())
    return false;

  // Only fold casts between strided memref forms.
  int64_t sourceOffset, resultOffset;
  SmallVector<int64_t, 4> sourceStrides, resultStrides;
  if (failed(getStridesAndOffset(sourceType, sourceStrides, sourceOffset)) ||
      failed(getStridesAndOffset(resultType, resultStrides, resultOffset)))
    return false;

  // If cast is towards more static sizes along any dimension, don't fold.
  for (auto it : llvm::zip(sourceType.getShape(), resultType.getShape())) {
    auto ss = std::get<0>(it), st = std::get<1>(it);
    if (ss != st)
      if (ShapedType::isDynamic(ss) && !ShapedType::isDynamic(st))
        return false;
  }

  // If cast is towards more static offset along any dimension, don't fold.
  if (sourceOffset != resultOffset)
    if (ShapedType::isDynamicStrideOrOffset(sourceOffset) &&
        !ShapedType::isDynamicStrideOrOffset(resultOffset))
      return false;

  // If cast is towards more static strides along any dimension, don't fold.
  for (auto it : llvm::zip(sourceStrides, resultStrides)) {
    auto ss = std::get<0>(it), st = std::get<1>(it);
    if (ss != st)
      if (ShapedType::isDynamicStrideOrOffset(ss) &&
          !ShapedType::isDynamicStrideOrOffset(st))
        return false;
  }

  return true;
}

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  auto aT = a.dyn_cast<MemRefType>();
  auto bT = b.dyn_cast<MemRefType>();

  auto uaT = a.dyn_cast<UnrankedMemRefType>();
  auto ubT = b.dyn_cast<UnrankedMemRefType>();

  if (aT && bT) {
    if (aT.getElementType() != bT.getElementType())
      return false;
    if (aT.getLayout() != bT.getLayout()) {
      int64_t aOffset, bOffset;
      SmallVector<int64_t, 4> aStrides, bStrides;
      if (failed(getStridesAndOffset(aT, aStrides, aOffset)) ||
          failed(getStridesAndOffset(bT, bStrides, bOffset)) ||
          aStrides.size() != bStrides.size())
        return false;

      // Strides along a dimension/offset are compatible if the value in the
      // source memref is static and the value in the target memref is the
      // same. They are also compatible if either one is dynamic (see
      // description of MemRefCastOp for details).
      auto checkCompatible = [](int64_t a, int64_t b) {
        return (a == MemRefType::getDynamicStrideOrOffset() ||
                b == MemRefType::getDynamicStrideOrOffset() || a == b);
      };
      if (!checkCompatible(aOffset, bOffset))
        return false;
      for (const auto &aStride : enumerate(aStrides))
        if (!checkCompatible(aStride.value(), bStrides[aStride.index()]))
          return false;
    }
    if (aT.getMemorySpace() != bT.getMemorySpace())
      return false;

    // They must have the same rank, and any specified dimensions must match.
    if (aT.getRank() != bT.getRank())
      return false;

    for (unsigned i = 0, e = aT.getRank(); i != e; ++i) {
      int64_t aDim = aT.getDimSize(i), bDim = bT.getDimSize(i);
      if (aDim != -1 && bDim != -1 && aDim != bDim)
        return false;
    }
    return true;
  } else {
    if (!aT && !uaT)
      return false;
    if (!bT && !ubT)
      return false;
    // Unranked to unranked casting is unsupported
    if (uaT && ubT)
      return false;

    auto aEltType = (aT) ? aT.getElementType() : uaT.getElementType();
    auto bEltType = (bT) ? bT.getElementType() : ubT.getElementType();
    if (aEltType != bEltType)
      return false;

    auto aMemSpace = (aT) ? aT.getMemorySpace() : uaT.getMemorySpace();
    auto bMemSpace = (bT) ? bT.getMemorySpace() : ubT.getMemorySpace();
    return aMemSpace == bMemSpace;
  }

  return false;
}

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  return succeeded(foldMemRefCast(*this)) ? getResult() : Value();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

namespace {
/// If the source/target of a CopyOp is a CastOp that does not modify the shape
/// and element type, the cast can be skipped. Such CastOps only cast the layout
/// of the type.
struct FoldCopyOfCast : public OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    bool modified = false;

    // Check source.
    if (auto castOp = copyOp.source().getDefiningOp<CastOp>()) {
      auto fromType = castOp.source().getType().dyn_cast<MemRefType>();
      auto toType = castOp.source().getType().dyn_cast<MemRefType>();

      if (fromType && toType) {
        if (fromType.getShape() == toType.getShape() &&
            fromType.getElementType() == toType.getElementType()) {
          rewriter.updateRootInPlace(
              copyOp, [&] { copyOp.sourceMutable().assign(castOp.source()); });
          modified = true;
        }
      }
    }

    // Check target.
    if (auto castOp = copyOp.target().getDefiningOp<CastOp>()) {
      auto fromType = castOp.source().getType().dyn_cast<MemRefType>();
      auto toType = castOp.source().getType().dyn_cast<MemRefType>();

      if (fromType && toType) {
        if (fromType.getShape() == toType.getShape() &&
            fromType.getElementType() == toType.getElementType()) {
          rewriter.updateRootInPlace(
              copyOp, [&] { copyOp.targetMutable().assign(castOp.source()); });
          modified = true;
        }
      }
    }

    return success(modified);
  }
};

/// Fold memref.copy(%x, %x).
struct FoldSelfCopy : public OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp.source() != copyOp.target())
      return failure();

    rewriter.eraseOp(copyOp);
    return success();
  }
};
} // namespace

void CopyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldCopyOfCast, FoldSelfCopy>(context);
}

LogicalResult CopyOp::fold(ArrayRef<Attribute> cstOperands,
                           SmallVectorImpl<OpFoldResult> &results) {
  /// copy(memrefcast) -> copy
  bool folded = false;
  Operation *op = *this;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

LogicalResult DeallocOp::fold(ArrayRef<Attribute> cstOperands,
                              SmallVectorImpl<OpFoldResult> &results) {
  /// dealloc(memrefcast) -> dealloc
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

void DimOp::build(OpBuilder &builder, OperationState &result, Value source,
                  int64_t index) {
  auto loc = result.location;
  Value indexValue = builder.create<arith::ConstantIndexOp>(loc, index);
  build(builder, result, source, indexValue);
}

void DimOp::build(OpBuilder &builder, OperationState &result, Value source,
                  Value index) {
  auto indexTy = builder.getIndexType();
  build(builder, result, indexTy, source, index);
}

Optional<int64_t> DimOp::getConstantIndex() {
  if (auto constantOp = index().getDefiningOp<arith::ConstantOp>())
    return constantOp.getValue().cast<IntegerAttr>().getInt();
  return {};
}

LogicalResult DimOp::verify() {
  // Assume unknown index to be in range.
  Optional<int64_t> index = getConstantIndex();
  if (!index.hasValue())
    return success();

  // Check that constant index is not knowingly out of range.
  auto type = source().getType();
  if (auto memrefType = type.dyn_cast<MemRefType>()) {
    if (index.getValue() >= memrefType.getRank())
      return emitOpError("index is out of range");
  } else if (type.isa<UnrankedMemRefType>()) {
    // Assume index to be in range.
  } else {
    llvm_unreachable("expected operand with memref type");
  }
  return success();
}

/// Return a map with key being elements in `vals` and data being number of
/// occurences of it. Use std::map, since the `vals` here are strides and the
/// dynamic stride value is the same as the tombstone value for
/// `DenseMap<int64_t>`.
static std::map<int64_t, unsigned> getNumOccurences(ArrayRef<int64_t> vals) {
  std::map<int64_t, unsigned> numOccurences;
  for (auto val : vals)
    numOccurences[val]++;
  return numOccurences;
}

/// Given the `originalType` and a `candidateReducedType` whose shape is assumed
/// to be a subset of `originalType` with some `1` entries erased, return the
/// set of indices that specifies which of the entries of `originalShape` are
/// dropped to obtain `reducedShape`.
/// This accounts for cases where there are multiple unit-dims, but only a
/// subset of those are dropped. For MemRefTypes these can be disambiguated
/// using the strides. If a dimension is dropped the stride must be dropped too.
static llvm::Optional<llvm::SmallBitVector>
computeMemRefRankReductionMask(MemRefType originalType, MemRefType reducedType,
                               ArrayRef<OpFoldResult> sizes) {
  llvm::SmallBitVector unusedDims(originalType.getRank());
  if (originalType.getRank() == reducedType.getRank())
    return unusedDims;

  for (const auto &dim : llvm::enumerate(sizes))
    if (auto attr = dim.value().dyn_cast<Attribute>())
      if (attr.cast<IntegerAttr>().getInt() == 1)
        unusedDims.set(dim.index());

  SmallVector<int64_t> originalStrides, candidateStrides;
  int64_t originalOffset, candidateOffset;
  if (failed(
          getStridesAndOffset(originalType, originalStrides, originalOffset)) ||
      failed(
          getStridesAndOffset(reducedType, candidateStrides, candidateOffset)))
    return llvm::None;

  // For memrefs, a dimension is truly dropped if its corresponding stride is
  // also dropped. This is particularly important when more than one of the dims
  // is 1. Track the number of occurences of the strides in the original type
  // and the candidate type. For each unused dim that stride should not be
  // present in the candidate type. Note that there could be multiple dimensions
  // that have the same size. We dont need to exactly figure out which dim
  // corresponds to which stride, we just need to verify that the number of
  // reptitions of a stride in the original + number of unused dims with that
  // stride == number of repititions of a stride in the candidate.
  std::map<int64_t, unsigned> currUnaccountedStrides =
      getNumOccurences(originalStrides);
  std::map<int64_t, unsigned> candidateStridesNumOccurences =
      getNumOccurences(candidateStrides);
  for (size_t dim = 0, e = unusedDims.size(); dim != e; ++dim) {
    if (!unusedDims.test(dim))
      continue;
    int64_t originalStride = originalStrides[dim];
    if (currUnaccountedStrides[originalStride] >
        candidateStridesNumOccurences[originalStride]) {
      // This dim can be treated as dropped.
      currUnaccountedStrides[originalStride]--;
      continue;
    }
    if (currUnaccountedStrides[originalStride] ==
        candidateStridesNumOccurences[originalStride]) {
      // The stride for this is not dropped. Keep as is.
      unusedDims.reset(dim);
      continue;
    }
    if (currUnaccountedStrides[originalStride] <
        candidateStridesNumOccurences[originalStride]) {
      // This should never happen. Cant have a stride in the reduced rank type
      // that wasnt in the original one.
      return llvm::None;
    }
  }

  if ((int64_t)unusedDims.count() + reducedType.getRank() !=
      originalType.getRank())
    return llvm::None;
  return unusedDims;
}

llvm::SmallBitVector SubViewOp::getDroppedDims() {
  MemRefType sourceType = getSourceType();
  MemRefType resultType = getType();
  llvm::Optional<llvm::SmallBitVector> unusedDims =
      computeMemRefRankReductionMask(sourceType, resultType, getMixedSizes());
  assert(unusedDims && "unable to find unused dims of subview");
  return *unusedDims;
}

OpFoldResult DimOp::fold(ArrayRef<Attribute> operands) {
  // All forms of folding require a known index.
  auto index = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (!index)
    return {};

  // Folding for unranked types (UnrankedMemRefType) is not supported.
  auto memrefType = source().getType().dyn_cast<MemRefType>();
  if (!memrefType)
    return {};

  // Fold if the shape extent along the given index is known.
  if (!memrefType.isDynamicDim(index.getInt())) {
    Builder builder(getContext());
    return builder.getIndexAttr(memrefType.getShape()[index.getInt()]);
  }

  // The size at the given index is now known to be a dynamic size.
  unsigned unsignedIndex = index.getValue().getZExtValue();

  // Fold dim to the size argument for an `AllocOp`, `ViewOp`, or `SubViewOp`.
  Operation *definingOp = source().getDefiningOp();

  if (auto alloc = dyn_cast_or_null<AllocOp>(definingOp))
    return *(alloc.getDynamicSizes().begin() +
             memrefType.getDynamicDimIndex(unsignedIndex));

  if (auto alloca = dyn_cast_or_null<AllocaOp>(definingOp))
    return *(alloca.getDynamicSizes().begin() +
             memrefType.getDynamicDimIndex(unsignedIndex));

  if (auto view = dyn_cast_or_null<ViewOp>(definingOp))
    return *(view.getDynamicSizes().begin() +
             memrefType.getDynamicDimIndex(unsignedIndex));

  if (auto subview = dyn_cast_or_null<SubViewOp>(definingOp)) {
    llvm::SmallBitVector unusedDims = subview.getDroppedDims();
    unsigned resultIndex = 0;
    unsigned sourceRank = subview.getSourceType().getRank();
    unsigned sourceIndex = 0;
    for (auto i : llvm::seq<unsigned>(0, sourceRank)) {
      if (unusedDims.test(i))
        continue;
      if (resultIndex == unsignedIndex) {
        sourceIndex = i;
        break;
      }
      resultIndex++;
    }
    assert(subview.isDynamicSize(sourceIndex) &&
           "expected dynamic subview size");
    return subview.getDynamicSize(sourceIndex);
  }

  if (auto sizeInterface =
          dyn_cast_or_null<OffsetSizeAndStrideOpInterface>(definingOp)) {
    assert(sizeInterface.isDynamicSize(unsignedIndex) &&
           "Expected dynamic subview size");
    return sizeInterface.getDynamicSize(unsignedIndex);
  }

  // dim(memrefcast) -> dim
  if (succeeded(foldMemRefCast(*this)))
    return getResult();

  return {};
}

namespace {
/// Fold dim of a memref reshape operation to a load into the reshape's shape
/// operand.
struct DimOfMemRefReshape : public OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DimOp dim,
                                PatternRewriter &rewriter) const override {
    auto reshape = dim.source().getDefiningOp<ReshapeOp>();

    if (!reshape)
      return failure();

    // Place the load directly after the reshape to ensure that the shape memref
    // was not mutated.
    rewriter.setInsertionPointAfter(reshape);
    Location loc = dim.getLoc();
    Value load = rewriter.create<LoadOp>(loc, reshape.shape(), dim.index());
    if (load.getType() != dim.getType())
      load = rewriter.create<arith::IndexCastOp>(loc, dim.getType(), load);
    rewriter.replaceOp(dim, load);
    return success();
  }
};

} // namespace

void DimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<DimOfMemRefReshape>(context);
}

// ---------------------------------------------------------------------------
// DmaStartOp
// ---------------------------------------------------------------------------

void DmaStartOp::build(OpBuilder &builder, OperationState &result,
                       Value srcMemRef, ValueRange srcIndices, Value destMemRef,
                       ValueRange destIndices, Value numElements,
                       Value tagMemRef, ValueRange tagIndices, Value stride,
                       Value elementsPerStride) {
  result.addOperands(srcMemRef);
  result.addOperands(srcIndices);
  result.addOperands(destMemRef);
  result.addOperands(destIndices);
  result.addOperands({numElements, tagMemRef});
  result.addOperands(tagIndices);
  if (stride)
    result.addOperands({stride, elementsPerStride});
}

void DmaStartOp::print(OpAsmPrinter &p) {
  p << " " << getSrcMemRef() << '[' << getSrcIndices() << "], "
    << getDstMemRef() << '[' << getDstIndices() << "], " << getNumElements()
    << ", " << getTagMemRef() << '[' << getTagIndices() << ']';
  if (isStrided())
    p << ", " << getStride() << ", " << getNumElementsPerStride();

  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getSrcMemRef().getType() << ", " << getDstMemRef().getType()
    << ", " << getTagMemRef().getType();
}

// Parse DmaStartOp.
// Ex:
//   %dma_id = dma_start %src[%i, %j], %dst[%k, %l], %size,
//                       %tag[%index], %stride, %num_elt_per_stride :
//                     : memref<3076 x f32, 0>,
//                       memref<1024 x f32, 2>,
//                       memref<1 x i32>
//
ParseResult DmaStartOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType srcMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> srcIndexInfos;
  OpAsmParser::OperandType dstMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> dstIndexInfos;
  OpAsmParser::OperandType numElementsInfo;
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> tagIndexInfos;
  SmallVector<OpAsmParser::OperandType, 2> strideInfo;

  SmallVector<Type, 3> types;
  auto indexType = parser.getBuilder().getIndexType();

  // Parse and resolve the following list of operands:
  // *) source memref followed by its indices (in square brackets).
  // *) destination memref followed by its indices (in square brackets).
  // *) dma size in KiB.
  if (parser.parseOperand(srcMemRefInfo) ||
      parser.parseOperandList(srcIndexInfos, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(dstMemRefInfo) ||
      parser.parseOperandList(dstIndexInfos, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(numElementsInfo) ||
      parser.parseComma() || parser.parseOperand(tagMemrefInfo) ||
      parser.parseOperandList(tagIndexInfos, OpAsmParser::Delimiter::Square))
    return failure();

  // Parse optional stride and elements per stride.
  if (parser.parseTrailingOperandList(strideInfo))
    return failure();

  bool isStrided = strideInfo.size() == 2;
  if (!strideInfo.empty() && !isStrided) {
    return parser.emitError(parser.getNameLoc(),
                            "expected two stride related operands");
  }

  if (parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 3)
    return parser.emitError(parser.getNameLoc(), "fewer/more types expected");

  if (parser.resolveOperand(srcMemRefInfo, types[0], result.operands) ||
      parser.resolveOperands(srcIndexInfos, indexType, result.operands) ||
      parser.resolveOperand(dstMemRefInfo, types[1], result.operands) ||
      parser.resolveOperands(dstIndexInfos, indexType, result.operands) ||
      // size should be an index.
      parser.resolveOperand(numElementsInfo, indexType, result.operands) ||
      parser.resolveOperand(tagMemrefInfo, types[2], result.operands) ||
      // tag indices should be index.
      parser.resolveOperands(tagIndexInfos, indexType, result.operands))
    return failure();

  if (isStrided) {
    if (parser.resolveOperands(strideInfo, indexType, result.operands))
      return failure();
  }

  return success();
}

LogicalResult DmaStartOp::verify() {
  unsigned numOperands = getNumOperands();

  // Mandatory non-variadic operands are: src memref, dst memref, tag memref and
  // the number of elements.
  if (numOperands < 4)
    return emitOpError("expected at least 4 operands");

  // Check types of operands. The order of these calls is important: the later
  // calls rely on some type properties to compute the operand position.
  // 1. Source memref.
  if (!getSrcMemRef().getType().isa<MemRefType>())
    return emitOpError("expected source to be of memref type");
  if (numOperands < getSrcMemRefRank() + 4)
    return emitOpError() << "expected at least " << getSrcMemRefRank() + 4
                         << " operands";
  if (!getSrcIndices().empty() &&
      !llvm::all_of(getSrcIndices().getTypes(),
                    [](Type t) { return t.isIndex(); }))
    return emitOpError("expected source indices to be of index type");

  // 2. Destination memref.
  if (!getDstMemRef().getType().isa<MemRefType>())
    return emitOpError("expected destination to be of memref type");
  unsigned numExpectedOperands = getSrcMemRefRank() + getDstMemRefRank() + 4;
  if (numOperands < numExpectedOperands)
    return emitOpError() << "expected at least " << numExpectedOperands
                         << " operands";
  if (!getDstIndices().empty() &&
      !llvm::all_of(getDstIndices().getTypes(),
                    [](Type t) { return t.isIndex(); }))
    return emitOpError("expected destination indices to be of index type");

  // 3. Number of elements.
  if (!getNumElements().getType().isIndex())
    return emitOpError("expected num elements to be of index type");

  // 4. Tag memref.
  if (!getTagMemRef().getType().isa<MemRefType>())
    return emitOpError("expected tag to be of memref type");
  numExpectedOperands += getTagMemRefRank();
  if (numOperands < numExpectedOperands)
    return emitOpError() << "expected at least " << numExpectedOperands
                         << " operands";
  if (!getTagIndices().empty() &&
      !llvm::all_of(getTagIndices().getTypes(),
                    [](Type t) { return t.isIndex(); }))
    return emitOpError("expected tag indices to be of index type");

  // Optional stride-related operands must be either both present or both
  // absent.
  if (numOperands != numExpectedOperands &&
      numOperands != numExpectedOperands + 2)
    return emitOpError("incorrect number of operands");

  // 5. Strides.
  if (isStrided()) {
    if (!getStride().getType().isIndex() ||
        !getNumElementsPerStride().getType().isIndex())
      return emitOpError(
          "expected stride and num elements per stride to be of type index");
  }

  return success();
}

LogicalResult DmaStartOp::fold(ArrayRef<Attribute> cstOperands,
                               SmallVectorImpl<OpFoldResult> &results) {
  /// dma_start(memrefcast) -> dma_start
  return foldMemRefCast(*this);
}

// ---------------------------------------------------------------------------
// DmaWaitOp
// ---------------------------------------------------------------------------

LogicalResult DmaWaitOp::fold(ArrayRef<Attribute> cstOperands,
                              SmallVectorImpl<OpFoldResult> &results) {
  /// dma_wait(memrefcast) -> dma_wait
  return foldMemRefCast(*this);
}

LogicalResult DmaWaitOp::verify() {
  // Check that the number of tag indices matches the tagMemRef rank.
  unsigned numTagIndices = tagIndices().size();
  unsigned tagMemRefRank = getTagMemRefRank();
  if (numTagIndices != tagMemRefRank)
    return emitOpError() << "expected tagIndices to have the same number of "
                            "elements as the tagMemRef rank, expected "
                         << tagMemRefRank << ", but got " << numTagIndices;
  return success();
}

//===----------------------------------------------------------------------===//
// GenericAtomicRMWOp
//===----------------------------------------------------------------------===//

void GenericAtomicRMWOp::build(OpBuilder &builder, OperationState &result,
                               Value memref, ValueRange ivs) {
  result.addOperands(memref);
  result.addOperands(ivs);

  if (auto memrefType = memref.getType().dyn_cast<MemRefType>()) {
    Type elementType = memrefType.getElementType();
    result.addTypes(elementType);

    Region *bodyRegion = result.addRegion();
    bodyRegion->push_back(new Block());
    bodyRegion->addArgument(elementType, memref.getLoc());
  }
}

LogicalResult GenericAtomicRMWOp::verify() {
  auto &body = getRegion();
  if (body.getNumArguments() != 1)
    return emitOpError("expected single number of entry block arguments");

  if (getResult().getType() != body.getArgument(0).getType())
    return emitOpError("expected block argument of the same type result type");

  bool hasSideEffects =
      body.walk([&](Operation *nestedOp) {
            if (MemoryEffectOpInterface::hasNoEffect(nestedOp))
              return WalkResult::advance();
            nestedOp->emitError(
                "body of 'memref.generic_atomic_rmw' should contain "
                "only operations with no side effects");
            return WalkResult::interrupt();
          })
          .wasInterrupted();
  return hasSideEffects ? failure() : success();
}

ParseResult GenericAtomicRMWOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::OperandType memref;
  Type memrefType;
  SmallVector<OpAsmParser::OperandType, 4> ivs;

  Type indexType = parser.getBuilder().getIndexType();
  if (parser.parseOperand(memref) ||
      parser.parseOperandList(ivs, OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(memrefType) ||
      parser.resolveOperand(memref, memrefType, result.operands) ||
      parser.resolveOperands(ivs, indexType, result.operands))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, llvm::None) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.types.push_back(memrefType.cast<MemRefType>().getElementType());
  return success();
}

void GenericAtomicRMWOp::print(OpAsmPrinter &p) {
  p << ' ' << memref() << "[" << indices() << "] : " << memref().getType()
    << ' ';
  p.printRegion(getRegion());
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// AtomicYieldOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicYieldOp::verify() {
  Type parentType = (*this)->getParentOp()->getResultTypes().front();
  Type resultType = result().getType();
  if (parentType != resultType)
    return emitOpError() << "types mismatch between yield op: " << resultType
                         << " and its parent: " << parentType;
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static void printGlobalMemrefOpTypeAndInitialValue(OpAsmPrinter &p, GlobalOp op,
                                                   TypeAttr type,
                                                   Attribute initialValue) {
  p << type;
  if (!op.isExternal()) {
    p << " = ";
    if (op.isUninitialized())
      p << "uninitialized";
    else
      p.printAttributeWithoutType(initialValue);
  }
}

static ParseResult
parseGlobalMemrefOpTypeAndInitialValue(OpAsmParser &parser, TypeAttr &typeAttr,
                                       Attribute &initialValue) {
  Type type;
  if (parser.parseType(type))
    return failure();

  auto memrefType = type.dyn_cast<MemRefType>();
  if (!memrefType || !memrefType.hasStaticShape())
    return parser.emitError(parser.getNameLoc())
           << "type should be static shaped memref, but got " << type;
  typeAttr = TypeAttr::get(type);

  if (parser.parseOptionalEqual())
    return success();

  if (succeeded(parser.parseOptionalKeyword("uninitialized"))) {
    initialValue = UnitAttr::get(parser.getContext());
    return success();
  }

  Type tensorType = getTensorTypeFromMemRefType(memrefType);
  if (parser.parseAttribute(initialValue, tensorType))
    return failure();
  if (!initialValue.isa<ElementsAttr>())
    return parser.emitError(parser.getNameLoc())
           << "initial value should be a unit or elements attribute";
  return success();
}

LogicalResult GlobalOp::verify() {
  auto memrefType = type().dyn_cast<MemRefType>();
  if (!memrefType || !memrefType.hasStaticShape())
    return emitOpError("type should be static shaped memref, but got ")
           << type();

  // Verify that the initial value, if present, is either a unit attribute or
  // an elements attribute.
  if (initial_value().hasValue()) {
    Attribute initValue = initial_value().getValue();
    if (!initValue.isa<UnitAttr>() && !initValue.isa<ElementsAttr>())
      return emitOpError("initial value should be a unit or elements "
                         "attribute, but got ")
             << initValue;

    // Check that the type of the initial value is compatible with the type of
    // the global variable.
    if (initValue.isa<ElementsAttr>()) {
      Type initType = initValue.getType();
      Type tensorType = getTensorTypeFromMemRefType(memrefType);
      if (initType != tensorType)
        return emitOpError("initial value expected to be of type ")
               << tensorType << ", but was of type " << initType;
    }
  }

  if (Optional<uint64_t> alignAttr = alignment()) {
    uint64_t alignment = alignAttr.getValue();

    if (!llvm::isPowerOf2_64(alignment))
      return emitError() << "alignment attribute value " << alignment
                         << " is not a power of 2";
  }

  // TODO: verify visibility for declarations.
  return success();
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult
GetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify that the result type is same as the type of the referenced
  // memref.global op.
  auto global =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, nameAttr());
  if (!global)
    return emitOpError("'")
           << name() << "' does not reference a valid global memref";

  Type resultType = result().getType();
  if (global.type() != resultType)
    return emitOpError("result type ")
           << resultType << " does not match type " << global.type()
           << " of the global memref @" << name();
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  if (getNumOperands() != 1 + getMemRefType().getRank())
    return emitOpError("incorrect number of indices for load");
  return success();
}

OpFoldResult LoadOp::fold(ArrayRef<Attribute> cstOperands) {
  /// load(memrefcast) -> load
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return OpFoldResult();
}

//===----------------------------------------------------------------------===//
// PrefetchOp
//===----------------------------------------------------------------------===//

void PrefetchOp::print(OpAsmPrinter &p) {
  p << " " << memref() << '[';
  p.printOperands(indices());
  p << ']' << ", " << (isWrite() ? "write" : "read");
  p << ", locality<" << localityHint();
  p << ">, " << (isDataCache() ? "data" : "instr");
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"localityHint", "isWrite", "isDataCache"});
  p << " : " << getMemRefType();
}

ParseResult PrefetchOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  IntegerAttr localityHint;
  MemRefType type;
  StringRef readOrWrite, cacheType;

  auto indexTy = parser.getBuilder().getIndexType();
  auto i32Type = parser.getBuilder().getIntegerType(32);
  if (parser.parseOperand(memrefInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseKeyword(&readOrWrite) ||
      parser.parseComma() || parser.parseKeyword("locality") ||
      parser.parseLess() ||
      parser.parseAttribute(localityHint, i32Type, "localityHint",
                            result.attributes) ||
      parser.parseGreater() || parser.parseComma() ||
      parser.parseKeyword(&cacheType) || parser.parseColonType(type) ||
      parser.resolveOperand(memrefInfo, type, result.operands) ||
      parser.resolveOperands(indexInfo, indexTy, result.operands))
    return failure();

  if (!readOrWrite.equals("read") && !readOrWrite.equals("write"))
    return parser.emitError(parser.getNameLoc(),
                            "rw specifier has to be 'read' or 'write'");
  result.addAttribute(
      PrefetchOp::getIsWriteAttrName(),
      parser.getBuilder().getBoolAttr(readOrWrite.equals("write")));

  if (!cacheType.equals("data") && !cacheType.equals("instr"))
    return parser.emitError(parser.getNameLoc(),
                            "cache type has to be 'data' or 'instr'");

  result.addAttribute(
      PrefetchOp::getIsDataCacheAttrName(),
      parser.getBuilder().getBoolAttr(cacheType.equals("data")));

  return success();
}

LogicalResult PrefetchOp::verify() {
  if (getNumOperands() != 1 + getMemRefType().getRank())
    return emitOpError("too few indices");

  return success();
}

LogicalResult PrefetchOp::fold(ArrayRef<Attribute> cstOperands,
                               SmallVectorImpl<OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

OpFoldResult RankOp::fold(ArrayRef<Attribute> operands) {
  // Constant fold rank when the rank of the operand is known.
  auto type = getOperand().getType();
  auto shapedType = type.dyn_cast<ShapedType>();
  if (shapedType && shapedType.hasRank())
    return IntegerAttr::get(IndexType::get(getContext()), shapedType.getRank());
  return IntegerAttr();
}

//===----------------------------------------------------------------------===//
// ReinterpretCastOp
//===----------------------------------------------------------------------===//

/// Build a ReinterpretCastOp with all dynamic entries: `staticOffsets`,
/// `staticSizes` and `staticStrides` are automatically filled with
/// source-memref-rank sentinel values that encode dynamic entries.
void ReinterpretCastOp::build(OpBuilder &b, OperationState &result,
                              MemRefType resultType, Value source,
                              OpFoldResult offset, ArrayRef<OpFoldResult> sizes,
                              ArrayRef<OpFoldResult> strides,
                              ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offset, dynamicOffsets, staticOffsets,
                             ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             ShapedType::kDynamicStrideOrOffset);
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getI64ArrayAttr(staticOffsets),
        b.getI64ArrayAttr(staticSizes), b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

void ReinterpretCastOp::build(OpBuilder &b, OperationState &result,
                              MemRefType resultType, Value source,
                              int64_t offset, ArrayRef<int64_t> sizes,
                              ArrayRef<int64_t> strides,
                              ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> sizeValues =
      llvm::to_vector<4>(llvm::map_range(sizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, resultType, source, b.getI64IntegerAttr(offset), sizeValues,
        strideValues, attrs);
}

void ReinterpretCastOp::build(OpBuilder &b, OperationState &result,
                              MemRefType resultType, Value source, Value offset,
                              ValueRange sizes, ValueRange strides,
                              ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, resultType, source, offset, sizeValues, strideValues, attrs);
}

// TODO: ponder whether we want to allow missing trailing sizes/strides that are
// completed automatically, like we have for subview and extract_slice.
LogicalResult ReinterpretCastOp::verify() {
  // The source and result memrefs should be in the same memory space.
  auto srcType = source().getType().cast<BaseMemRefType>();
  auto resultType = getType().cast<MemRefType>();
  if (srcType.getMemorySpace() != resultType.getMemorySpace())
    return emitError("different memory spaces specified for source type ")
           << srcType << " and result memref type " << resultType;
  if (srcType.getElementType() != resultType.getElementType())
    return emitError("different element types specified for source type ")
           << srcType << " and result memref type " << resultType;

  // Match sizes in result memref type and in static_sizes attribute.
  for (auto &en : llvm::enumerate(llvm::zip(
           resultType.getShape(), extractFromI64ArrayAttr(static_sizes())))) {
    int64_t resultSize = std::get<0>(en.value());
    int64_t expectedSize = std::get<1>(en.value());
    if (!ShapedType::isDynamic(resultSize) &&
        !ShapedType::isDynamic(expectedSize) && resultSize != expectedSize)
      return emitError("expected result type with size = ")
             << expectedSize << " instead of " << resultSize
             << " in dim = " << en.index();
  }

  // Match offset and strides in static_offset and static_strides attributes. If
  // result memref type has no affine map specified, this will assume an
  // identity layout.
  int64_t resultOffset;
  SmallVector<int64_t, 4> resultStrides;
  if (failed(getStridesAndOffset(resultType, resultStrides, resultOffset)))
    return emitError("expected result type to have strided layout but found ")
           << resultType;

  // Match offset in result memref type and in static_offsets attribute.
  int64_t expectedOffset = extractFromI64ArrayAttr(static_offsets()).front();
  if (!ShapedType::isDynamicStrideOrOffset(resultOffset) &&
      !ShapedType::isDynamicStrideOrOffset(expectedOffset) &&
      resultOffset != expectedOffset)
    return emitError("expected result type with offset = ")
           << resultOffset << " instead of " << expectedOffset;

  // Match strides in result memref type and in static_strides attribute.
  for (auto &en : llvm::enumerate(llvm::zip(
           resultStrides, extractFromI64ArrayAttr(static_strides())))) {
    int64_t resultStride = std::get<0>(en.value());
    int64_t expectedStride = std::get<1>(en.value());
    if (!ShapedType::isDynamicStrideOrOffset(resultStride) &&
        !ShapedType::isDynamicStrideOrOffset(expectedStride) &&
        resultStride != expectedStride)
      return emitError("expected result type with stride = ")
             << expectedStride << " instead of " << resultStride
             << " in dim = " << en.index();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Reassociative reshape ops
//===----------------------------------------------------------------------===//

SmallVector<AffineMap, 4> CollapseShapeOp::getReassociationMaps() {
  return getSymbolLessAffineMaps(getReassociationExprs());
}
SmallVector<ReassociationExprs, 4> CollapseShapeOp::getReassociationExprs() {
  return convertReassociationIndicesToExprs(getContext(),
                                            getReassociationIndices());
}

SmallVector<AffineMap, 4> ExpandShapeOp::getReassociationMaps() {
  return getSymbolLessAffineMaps(getReassociationExprs());
}
SmallVector<ReassociationExprs, 4> ExpandShapeOp::getReassociationExprs() {
  return convertReassociationIndicesToExprs(getContext(),
                                            getReassociationIndices());
}

/// Detect whether memref dims [dim, dim + extent) can be reshaped without
/// copies.
static bool isReshapableDimBand(unsigned dim, unsigned extent,
                                ArrayRef<int64_t> sizes,
                                ArrayRef<AffineExpr> strides) {
  // Bands of extent one can be reshaped, as they are not reshaped at all.
  if (extent == 1)
    return true;
  // Otherwise, the size of the first dimension needs to be known.
  if (ShapedType::isDynamic(sizes[dim]))
    return false;
  assert(sizes.size() == strides.size() && "mismatched ranks");
  // off by 1 indexing to avoid out of bounds
  //                       V
  for (auto idx = dim, e = dim + extent; idx + 1 < e; ++idx) {
    // Only bands of static shapes are reshapable. This is due to the fact that
    // there is no relation between dynamic sizes and dynamic strides: we do not
    // have enough information to know whether a "-1" size corresponds to the
    // proper symbol in the AffineExpr of a stride.
    if (ShapedType::isDynamic(sizes[idx + 1]))
      return false;
    // TODO: Refine this by passing the proper nDims and nSymbols so we can
    // simplify on the fly and catch more reshapable cases.
    if (strides[idx] != strides[idx + 1] * sizes[idx + 1])
      return false;
  }
  return true;
}

/// Compute the MemRefType obtained by applying the `reassociation` (which is
/// expected to be valid) to `type`.
/// If `type` is Contiguous MemRefType, this always produce a contiguous
/// MemRefType.
static MemRefType
computeReshapeCollapsedType(MemRefType type,
                            ArrayRef<AffineMap> reassociation) {
  auto sizes = type.getShape();
  AffineExpr offset;
  SmallVector<AffineExpr, 4> strides;
  auto status = getStridesAndOffset(type, strides, offset);
  auto isIdentityLayout = type.getLayout().isIdentity();
  (void)status;
  assert(succeeded(status) && "expected strided memref");

  SmallVector<int64_t, 4> newSizes;
  newSizes.reserve(reassociation.size());
  SmallVector<AffineExpr, 4> newStrides;
  newStrides.reserve(reassociation.size());

  // Use the fact that reassociation is valid to simplify the logic: only use
  // each map's rank.
  assert(isReassociationValid(reassociation) && "invalid reassociation");
  unsigned currentDim = 0;
  for (AffineMap m : reassociation) {
    unsigned dim = m.getNumResults();
    int64_t size = 1;
    AffineExpr stride = strides[currentDim + dim - 1];
    if (isIdentityLayout ||
        isReshapableDimBand(currentDim, dim, sizes, strides)) {
      for (unsigned d = 0; d < dim; ++d) {
        int64_t currentSize = sizes[currentDim + d];
        if (ShapedType::isDynamic(currentSize)) {
          size = ShapedType::kDynamicSize;
          break;
        }
        size *= currentSize;
      }
    } else {
      size = ShapedType::kDynamicSize;
      stride = AffineExpr();
    }
    newSizes.push_back(size);
    newStrides.push_back(stride);
    currentDim += dim;
  }

  // Early-exit: if `type` is contiguous, the result must be contiguous.
  if (canonicalizeStridedLayout(type).getLayout().isIdentity())
    return MemRefType::Builder(type).setShape(newSizes).setLayout({});

  // Convert back to int64_t because we don't have enough information to create
  // new strided layouts from AffineExpr only. This corresponds to a case where
  // copies may be necessary.
  int64_t intOffset = ShapedType::kDynamicStrideOrOffset;
  if (auto o = offset.dyn_cast<AffineConstantExpr>())
    intOffset = o.getValue();
  SmallVector<int64_t, 4> intStrides;
  intStrides.reserve(strides.size());
  for (auto stride : newStrides) {
    if (auto cst = stride.dyn_cast_or_null<AffineConstantExpr>())
      intStrides.push_back(cst.getValue());
    else
      intStrides.push_back(ShapedType::kDynamicStrideOrOffset);
  }
  auto layout =
      makeStridedLinearLayoutMap(intStrides, intOffset, type.getContext());
  return canonicalizeStridedLayout(
      MemRefType::Builder(type).setShape(newSizes).setLayout(
          AffineMapAttr::get(layout)));
}

void ExpandShapeOp::build(OpBuilder &b, OperationState &result, Value src,
                          ArrayRef<ReassociationIndices> reassociation,
                          ArrayRef<NamedAttribute> attrs) {
  auto memRefType = src.getType().cast<MemRefType>();
  auto resultType = computeReshapeCollapsedType(
      memRefType, getSymbolLessAffineMaps(convertReassociationIndicesToExprs(
                      b.getContext(), reassociation)));
  build(b, result, resultType, src, attrs);
  result.addAttribute(getReassociationAttrName(),
                      getReassociationIndicesAttribute(b, reassociation));
}

void CollapseShapeOp::build(OpBuilder &b, OperationState &result, Value src,
                            ArrayRef<ReassociationIndices> reassociation,
                            ArrayRef<NamedAttribute> attrs) {
  auto memRefType = src.getType().cast<MemRefType>();
  auto resultType = computeReshapeCollapsedType(
      memRefType, getSymbolLessAffineMaps(convertReassociationIndicesToExprs(
                      b.getContext(), reassociation)));
  build(b, result, resultType, src, attrs);
  result.addAttribute(getReassociationAttrName(),
                      getReassociationIndicesAttribute(b, reassociation));
}

template <typename ReshapeOp,
          bool isExpansion = std::is_same<ReshapeOp, ExpandShapeOp>::value>
static LogicalResult verifyReshapeOp(ReshapeOp op, MemRefType expandedType,
                                     MemRefType collapsedType) {
  if (failed(
          verifyReshapeLikeTypes(op, expandedType, collapsedType, isExpansion)))
    return failure();
  auto maps = op.getReassociationMaps();
  MemRefType expectedType = computeReshapeCollapsedType(expandedType, maps);
  if (collapsedType != expectedType)
    return op.emitOpError("expected collapsed type to be ")
           << expectedType << ", but got " << collapsedType;
  return success();
}

LogicalResult ExpandShapeOp::verify() {
  return verifyReshapeOp(*this, getResultType(), getSrcType());
}

void ExpandShapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<CollapseReshapeOps<ExpandShapeOp>,
              CollapseMixedReshapeOps<ExpandShapeOp, CollapseShapeOp>>(context);
}

LogicalResult CollapseShapeOp::verify() {
  return verifyReshapeOp(*this, getSrcType(), getResultType());
}

struct CollapseShapeOpMemRefCastFolder
    : public OpRewritePattern<CollapseShapeOp> {
public:
  using OpRewritePattern<CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CollapseShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto cast = op.getOperand().getDefiningOp<CastOp>();
    if (!cast)
      return failure();

    if (!CastOp::canFoldIntoConsumerOp(cast))
      return failure();

    Type newResultType = computeReshapeCollapsedType(
        cast.getOperand().getType().cast<MemRefType>(),
        op.getReassociationMaps());

    if (newResultType == op.getResultType()) {
      rewriter.updateRootInPlace(
          op, [&]() { op.srcMutable().assign(cast.source()); });
    } else {
      Value newOp = rewriter.create<CollapseShapeOp>(
          op->getLoc(), cast.source(), op.getReassociationIndices());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(), newOp);
    }
    return success();
  }
};

void CollapseShapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<CollapseReshapeOps<CollapseShapeOp>,
              CollapseMixedReshapeOps<CollapseShapeOp, ExpandShapeOp>,
              CollapseShapeOpMemRefCastFolder>(context);
}
OpFoldResult ExpandShapeOp::fold(ArrayRef<Attribute> operands) {
  return foldReshapeOp<ExpandShapeOp, CollapseShapeOp>(*this, operands);
}
OpFoldResult CollapseShapeOp::fold(ArrayRef<Attribute> operands) {
  return foldReshapeOp<CollapseShapeOp, ExpandShapeOp>(*this, operands);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  Type operandType = source().getType();
  Type resultType = result().getType();

  Type operandElementType = operandType.cast<ShapedType>().getElementType();
  Type resultElementType = resultType.cast<ShapedType>().getElementType();
  if (operandElementType != resultElementType)
    return emitOpError("element types of source and destination memref "
                       "types should be the same");

  if (auto operandMemRefType = operandType.dyn_cast<MemRefType>())
    if (!operandMemRefType.getLayout().isIdentity())
      return emitOpError("source memref type should have identity affine map");

  int64_t shapeSize = shape().getType().cast<MemRefType>().getDimSize(0);
  auto resultMemRefType = resultType.dyn_cast<MemRefType>();
  if (resultMemRefType) {
    if (!resultMemRefType.getLayout().isIdentity())
      return emitOpError("result memref type should have identity affine map");
    if (shapeSize == ShapedType::kDynamicSize)
      return emitOpError("cannot use shape operand with dynamic length to "
                         "reshape to statically-ranked memref type");
    if (shapeSize != resultMemRefType.getRank())
      return emitOpError(
          "length of shape operand differs from the result's memref rank");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  if (getNumOperands() != 2 + getMemRefType().getRank())
    return emitOpError("store index operand count not equal to memref rank");

  return success();
}

LogicalResult StoreOp::fold(ArrayRef<Attribute> cstOperands,
                            SmallVectorImpl<OpFoldResult> &results) {
  /// store(memrefcast) -> store
  return foldMemRefCast(*this, getValueToStore());
}

//===----------------------------------------------------------------------===//
// SubViewOp
//===----------------------------------------------------------------------===//

namespace {
/// Helpers to write more idiomatic operations.
namespace saturated_arith {
struct Wrapper {
  explicit Wrapper(int64_t v) : v(v) {}
  operator int64_t() { return v; }
  int64_t v;
};
Wrapper operator+(Wrapper a, int64_t b) {
  if (ShapedType::isDynamicStrideOrOffset(a) ||
      ShapedType::isDynamicStrideOrOffset(b))
    return Wrapper(ShapedType::kDynamicStrideOrOffset);
  return Wrapper(a.v + b);
}
Wrapper operator*(Wrapper a, int64_t b) {
  if (ShapedType::isDynamicStrideOrOffset(a) ||
      ShapedType::isDynamicStrideOrOffset(b))
    return Wrapper(ShapedType::kDynamicStrideOrOffset);
  return Wrapper(a.v * b);
}
} // namespace saturated_arith
} // namespace

/// A subview result type can be fully inferred from the source type and the
/// static representation of offsets, sizes and strides. Special sentinels
/// encode the dynamic case.
Type SubViewOp::inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<int64_t> staticOffsets,
                                ArrayRef<int64_t> staticSizes,
                                ArrayRef<int64_t> staticStrides) {
  unsigned rank = sourceMemRefType.getRank();
  (void)rank;
  assert(staticOffsets.size() == rank && "staticOffsets length mismatch");
  assert(staticSizes.size() == rank && "staticSizes length mismatch");
  assert(staticStrides.size() == rank && "staticStrides length mismatch");

  // Extract source offset and strides.
  int64_t sourceOffset;
  SmallVector<int64_t, 4> sourceStrides;
  auto res = getStridesAndOffset(sourceMemRefType, sourceStrides, sourceOffset);
  assert(succeeded(res) && "SubViewOp expected strided memref type");
  (void)res;

  // Compute target offset whose value is:
  //   `sourceOffset + sum_i(staticOffset_i * sourceStrides_i)`.
  int64_t targetOffset = sourceOffset;
  for (auto it : llvm::zip(staticOffsets, sourceStrides)) {
    auto staticOffset = std::get<0>(it), targetStride = std::get<1>(it);
    using namespace saturated_arith;
    targetOffset = Wrapper(targetOffset) + Wrapper(staticOffset) * targetStride;
  }

  // Compute target stride whose value is:
  //   `sourceStrides_i * staticStrides_i`.
  SmallVector<int64_t, 4> targetStrides;
  targetStrides.reserve(staticOffsets.size());
  for (auto it : llvm::zip(sourceStrides, staticStrides)) {
    auto sourceStride = std::get<0>(it), staticStride = std::get<1>(it);
    using namespace saturated_arith;
    targetStrides.push_back(Wrapper(sourceStride) * staticStride);
  }

  // The type is now known.
  return MemRefType::get(
      staticSizes, sourceMemRefType.getElementType(),
      makeStridedLinearLayoutMap(targetStrides, targetOffset,
                                 sourceMemRefType.getContext()),
      sourceMemRefType.getMemorySpace());
}

Type SubViewOp::inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<OpFoldResult> offsets,
                                ArrayRef<OpFoldResult> sizes,
                                ArrayRef<OpFoldResult> strides) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             ShapedType::kDynamicStrideOrOffset);
  return SubViewOp::inferResultType(sourceMemRefType, staticOffsets,
                                    staticSizes, staticStrides);
}

Type SubViewOp::inferRankReducedResultType(unsigned resultRank,
                                           MemRefType sourceRankedTensorType,
                                           ArrayRef<int64_t> offsets,
                                           ArrayRef<int64_t> sizes,
                                           ArrayRef<int64_t> strides) {
  auto inferredType =
      inferResultType(sourceRankedTensorType, offsets, sizes, strides)
          .cast<MemRefType>();
  assert(inferredType.getRank() >= resultRank && "expected ");
  int rankDiff = inferredType.getRank() - resultRank;
  if (rankDiff > 0) {
    auto shape = inferredType.getShape();
    llvm::SmallBitVector dimsToProject =
        getPositionsOfShapeOne(rankDiff, shape);
    SmallVector<int64_t> projectedShape;
    for (unsigned pos = 0, e = shape.size(); pos < e; ++pos)
      if (!dimsToProject.test(pos))
        projectedShape.push_back(shape[pos]);

    AffineMap map = inferredType.getLayout().getAffineMap();
    if (!map.isIdentity())
      map = getProjectedMap(map, dimsToProject);
    inferredType =
        MemRefType::get(projectedShape, inferredType.getElementType(), map,
                        inferredType.getMemorySpace());
  }
  return inferredType;
}

Type SubViewOp::inferRankReducedResultType(unsigned resultRank,
                                           MemRefType sourceRankedTensorType,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes,
                                           ArrayRef<OpFoldResult> strides) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             ShapedType::kDynamicStrideOrOffset);
  return SubViewOp::inferRankReducedResultType(
      resultRank, sourceRankedTensorType, staticOffsets, staticSizes,
      staticStrides);
}
// Build a SubViewOp with mixed static and dynamic entries and custom result
// type. If the type passed is nullptr, it is inferred.
void SubViewOp::build(OpBuilder &b, OperationState &result,
                      MemRefType resultType, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides,
                      ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             ShapedType::kDynamicStrideOrOffset);
  auto sourceMemRefType = source.getType().cast<MemRefType>();
  // Structuring implementation this way avoids duplication between builders.
  if (!resultType) {
    resultType = SubViewOp::inferResultType(sourceMemRefType, staticOffsets,
                                            staticSizes, staticStrides)
                     .cast<MemRefType>();
  }
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getI64ArrayAttr(staticOffsets),
        b.getI64ArrayAttr(staticSizes), b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

// Build a SubViewOp with mixed static and dynamic entries and inferred result
// type.
void SubViewOp::build(OpBuilder &b, OperationState &result, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides,
                      ArrayRef<NamedAttribute> attrs) {
  build(b, result, MemRefType(), source, offsets, sizes, strides, attrs);
}

// Build a SubViewOp with static entries and inferred result type.
void SubViewOp::build(OpBuilder &b, OperationState &result, Value source,
                      ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
                      ArrayRef<int64_t> strides,
                      ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sizeValues =
      llvm::to_vector<4>(llvm::map_range(sizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, source, offsetValues, sizeValues, strideValues, attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the
// type passed is nullptr, it is inferred.
void SubViewOp::build(OpBuilder &b, OperationState &result,
                      MemRefType resultType, Value source,
                      ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
                      ArrayRef<int64_t> strides,
                      ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> sizeValues =
      llvm::to_vector<4>(llvm::map_range(sizes, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues,
        attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the type
// passed is nullptr, it is inferred.
void SubViewOp::build(OpBuilder &b, OperationState &result,
                      MemRefType resultType, Value source, ValueRange offsets,
                      ValueRange sizes, ValueRange strides,
                      ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues);
}

// Build a SubViewOp with dynamic entries and inferred result type.
void SubViewOp::build(OpBuilder &b, OperationState &result, Value source,
                      ValueRange offsets, ValueRange sizes, ValueRange strides,
                      ArrayRef<NamedAttribute> attrs) {
  build(b, result, MemRefType(), source, offsets, sizes, strides, attrs);
}

/// For ViewLikeOpInterface.
Value SubViewOp::getViewSource() { return source(); }

/// Return true if t1 and t2 have equal offsets (both dynamic or of same static
/// value).
static bool haveCompatibleOffsets(MemRefType t1, MemRefType t2) {
  AffineExpr t1Offset, t2Offset;
  SmallVector<AffineExpr> t1Strides, t2Strides;
  auto res1 = getStridesAndOffset(t1, t1Strides, t1Offset);
  auto res2 = getStridesAndOffset(t2, t2Strides, t2Offset);
  return succeeded(res1) && succeeded(res2) && t1Offset == t2Offset;
}

/// Checks if `original` Type type can be rank reduced to `reduced` type.
/// This function is slight variant of `is subsequence` algorithm where
/// not matching dimension must be 1.
static SliceVerificationResult
isRankReducedMemRefType(MemRefType originalType,
                        MemRefType candidateRankReducedType,
                        ArrayRef<OpFoldResult> sizes) {
  auto partialRes = isRankReducedType(originalType, candidateRankReducedType);
  if (partialRes != SliceVerificationResult::Success)
    return partialRes;

  auto optionalUnusedDimsMask = computeMemRefRankReductionMask(
      originalType, candidateRankReducedType, sizes);

  // Sizes cannot be matched in case empty vector is returned.
  if (!optionalUnusedDimsMask.hasValue())
    return SliceVerificationResult::LayoutMismatch;

  if (originalType.getMemorySpace() !=
      candidateRankReducedType.getMemorySpace())
    return SliceVerificationResult::MemSpaceMismatch;

  // No amount of stride dropping can reconcile incompatible offsets.
  if (!haveCompatibleOffsets(originalType, candidateRankReducedType))
    return SliceVerificationResult::LayoutMismatch;

  return SliceVerificationResult::Success;
}

template <typename OpTy>
static LogicalResult produceSubViewErrorMsg(SliceVerificationResult result,
                                            OpTy op, Type expectedType) {
  auto memrefType = expectedType.cast<ShapedType>();
  switch (result) {
  case SliceVerificationResult::Success:
    return success();
  case SliceVerificationResult::RankTooLarge:
    return op.emitError("expected result rank to be smaller or equal to ")
           << "the source rank. ";
  case SliceVerificationResult::SizeMismatch:
    return op.emitError("expected result type to be ")
           << expectedType
           << " or a rank-reduced version. (mismatch of result sizes) ";
  case SliceVerificationResult::ElemTypeMismatch:
    return op.emitError("expected result element type to be ")
           << memrefType.getElementType();
  case SliceVerificationResult::MemSpaceMismatch:
    return op.emitError("expected result and source memory spaces to match.");
  case SliceVerificationResult::LayoutMismatch:
    return op.emitError("expected result type to be ")
           << expectedType
           << " or a rank-reduced version. (mismatch of result layout) ";
  }
  llvm_unreachable("unexpected subview verification result");
}

/// Verifier for SubViewOp.
LogicalResult SubViewOp::verify() {
  MemRefType baseType = getSourceType();
  MemRefType subViewType = getType();

  // The base memref and the view memref should be in the same memory space.
  if (baseType.getMemorySpace() != subViewType.getMemorySpace())
    return emitError("different memory spaces specified for base memref "
                     "type ")
           << baseType << " and subview memref type " << subViewType;

  // Verify that the base memref type has a strided layout map.
  if (!isStrided(baseType))
    return emitError("base type ") << baseType << " is not strided";

  // Verify result type against inferred type.
  auto expectedType = SubViewOp::inferResultType(
      baseType, extractFromI64ArrayAttr(static_offsets()),
      extractFromI64ArrayAttr(static_sizes()),
      extractFromI64ArrayAttr(static_strides()));

  auto result = isRankReducedMemRefType(expectedType.cast<MemRefType>(),
                                        subViewType, getMixedSizes());
  return produceSubViewErrorMsg(result, *this, expectedType);
}

raw_ostream &mlir::operator<<(raw_ostream &os, const Range &range) {
  return os << "range " << range.offset << ":" << range.size << ":"
            << range.stride;
}

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> mlir::getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                              OpBuilder &b, Location loc) {
  std::array<unsigned, 3> ranks = op.getArrayAttrMaxRanks();
  assert(ranks[0] == ranks[1] && "expected offset and sizes of equal ranks");
  assert(ranks[1] == ranks[2] && "expected sizes and strides of equal ranks");
  SmallVector<Range, 8> res;
  unsigned rank = ranks[0];
  res.reserve(rank);
  for (unsigned idx = 0; idx < rank; ++idx) {
    Value offset =
        op.isDynamicOffset(idx)
            ? op.getDynamicOffset(idx)
            : b.create<arith::ConstantIndexOp>(loc, op.getStaticOffset(idx));
    Value size =
        op.isDynamicSize(idx)
            ? op.getDynamicSize(idx)
            : b.create<arith::ConstantIndexOp>(loc, op.getStaticSize(idx));
    Value stride =
        op.isDynamicStride(idx)
            ? op.getDynamicStride(idx)
            : b.create<arith::ConstantIndexOp>(loc, op.getStaticStride(idx));
    res.emplace_back(Range{offset, size, stride});
  }
  return res;
}

/// Compute the canonical result type of a SubViewOp. Call `inferResultType` to
/// deduce the result type for the given `sourceType`. Additionally, reduce the
/// rank of the inferred result type if `currentResultType` is lower rank than
/// `currentSourceType`. Use this signature if `sourceType` is updated together
/// with the result type. In this case, it is important to compute the dropped
/// dimensions using `currentSourceType` whose strides align with
/// `currentResultType`.
static MemRefType getCanonicalSubViewResultType(
    MemRefType currentResultType, MemRefType currentSourceType,
    MemRefType sourceType, ArrayRef<OpFoldResult> mixedOffsets,
    ArrayRef<OpFoldResult> mixedSizes, ArrayRef<OpFoldResult> mixedStrides) {
  auto nonRankReducedType = SubViewOp::inferResultType(sourceType, mixedOffsets,
                                                       mixedSizes, mixedStrides)
                                .cast<MemRefType>();
  llvm::Optional<llvm::SmallBitVector> unusedDims =
      computeMemRefRankReductionMask(currentSourceType, currentResultType,
                                     mixedSizes);
  // Return nullptr as failure mode.
  if (!unusedDims)
    return nullptr;
  SmallVector<int64_t> shape;
  for (const auto &sizes : llvm::enumerate(nonRankReducedType.getShape())) {
    if (unusedDims->test(sizes.index()))
      continue;
    shape.push_back(sizes.value());
  }
  AffineMap layoutMap = nonRankReducedType.getLayout().getAffineMap();
  if (!layoutMap.isIdentity())
    layoutMap = getProjectedMap(layoutMap, unusedDims.getValue());
  return MemRefType::get(shape, nonRankReducedType.getElementType(), layoutMap,
                         nonRankReducedType.getMemorySpace());
}

/// Compute the canonical result type of a SubViewOp. Call `inferResultType` to
/// deduce the result type. Additionally, reduce the rank of the inferred result
/// type if `currentResultType` is lower rank than `sourceType`.
static MemRefType getCanonicalSubViewResultType(
    MemRefType currentResultType, MemRefType sourceType,
    ArrayRef<OpFoldResult> mixedOffsets, ArrayRef<OpFoldResult> mixedSizes,
    ArrayRef<OpFoldResult> mixedStrides) {
  return getCanonicalSubViewResultType(currentResultType, sourceType,
                                       sourceType, mixedOffsets, mixedSizes,
                                       mixedStrides);
}

/// Helper method to check if a `subview` operation is trivially a no-op. This
/// is the case if the all offsets are zero, all strides are 1, and the source
/// shape is same as the size of the subview. In such cases, the subview can be
/// folded into its source.
static bool isTrivialSubViewOp(SubViewOp subViewOp) {
  if (subViewOp.getSourceType().getRank() != subViewOp.getType().getRank())
    return false;

  auto mixedOffsets = subViewOp.getMixedOffsets();
  auto mixedSizes = subViewOp.getMixedSizes();
  auto mixedStrides = subViewOp.getMixedStrides();

  // Check offsets are zero.
  if (llvm::any_of(mixedOffsets, [](OpFoldResult ofr) {
        Optional<int64_t> intValue = getConstantIntValue(ofr);
        return !intValue || intValue.getValue() != 0;
      }))
    return false;

  // Check strides are one.
  if (llvm::any_of(mixedStrides, [](OpFoldResult ofr) {
        Optional<int64_t> intValue = getConstantIntValue(ofr);
        return !intValue || intValue.getValue() != 1;
      }))
    return false;

  // Check all size values are static and matches the (static) source shape.
  ArrayRef<int64_t> sourceShape = subViewOp.getSourceType().getShape();
  for (const auto &size : llvm::enumerate(mixedSizes)) {
    Optional<int64_t> intValue = getConstantIntValue(size.value());
    if (!intValue || intValue.getValue() != sourceShape[size.index()])
      return false;
  }
  // All conditions met. The `SubViewOp` is foldable as a no-op.
  return true;
}

namespace {
/// Pattern to rewrite a subview op with MemRefCast arguments.
/// This essentially pushes memref.cast past its consuming subview when
/// `canFoldIntoConsumerOp` is true.
///
/// Example:
/// ```
///   %0 = memref.cast %V : memref<16x16xf32> to memref<?x?xf32>
///   %1 = memref.subview %0[0, 0][3, 4][1, 1] :
///     memref<?x?xf32> to memref<3x4xf32, offset:?, strides:[?, 1]>
/// ```
/// is rewritten into:
/// ```
///   %0 = memref.subview %V: memref<16x16xf32> to memref<3x4xf32, #[[map0]]>
///   %1 = memref.cast %0: memref<3x4xf32, offset:0, strides:[16, 1]> to
///     memref<3x4xf32, offset:?, strides:[?, 1]>
/// ```
class SubViewOpMemRefCastFolder final : public OpRewritePattern<SubViewOp> {
public:
  using OpRewritePattern<SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubViewOp subViewOp,
                                PatternRewriter &rewriter) const override {
    // Any constant operand, just return to let SubViewOpConstantFolder kick in.
    if (llvm::any_of(subViewOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    auto castOp = subViewOp.source().getDefiningOp<CastOp>();
    if (!castOp)
      return failure();

    if (!CastOp::canFoldIntoConsumerOp(castOp))
      return failure();

    // Compute the SubViewOp result type after folding the MemRefCastOp. Use the
    // MemRefCastOp source operand type to infer the result type and the current
    // SubViewOp source operand type to compute the dropped dimensions if the
    // operation is rank-reducing.
    auto resultType = getCanonicalSubViewResultType(
        subViewOp.getType(), subViewOp.getSourceType(),
        castOp.source().getType().cast<MemRefType>(),
        subViewOp.getMixedOffsets(), subViewOp.getMixedSizes(),
        subViewOp.getMixedStrides());
    if (!resultType)
      return failure();

    Value newSubView = rewriter.create<SubViewOp>(
        subViewOp.getLoc(), resultType, castOp.source(), subViewOp.offsets(),
        subViewOp.sizes(), subViewOp.strides(), subViewOp.static_offsets(),
        subViewOp.static_sizes(), subViewOp.static_strides());
    rewriter.replaceOpWithNewOp<CastOp>(subViewOp, subViewOp.getType(),
                                        newSubView);
    return success();
  }
};

/// Canonicalize subview ops that are no-ops. When the source shape is not same
/// as a result shape due to use of `affine_map`.
class TrivialSubViewOpFolder final : public OpRewritePattern<SubViewOp> {
public:
  using OpRewritePattern<SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubViewOp subViewOp,
                                PatternRewriter &rewriter) const override {
    if (!isTrivialSubViewOp(subViewOp))
      return failure();
    if (subViewOp.getSourceType() == subViewOp.getType()) {
      rewriter.replaceOp(subViewOp, subViewOp.source());
      return success();
    }
    rewriter.replaceOpWithNewOp<CastOp>(subViewOp, subViewOp.getType(),
                                        subViewOp.source());
    return success();
  }
};
} // namespace

/// Return the canonical type of the result of a subview.
struct SubViewReturnTypeCanonicalizer {
  MemRefType operator()(SubViewOp op, ArrayRef<OpFoldResult> mixedOffsets,
                        ArrayRef<OpFoldResult> mixedSizes,
                        ArrayRef<OpFoldResult> mixedStrides) {
    return getCanonicalSubViewResultType(op.getType(), op.getSourceType(),
                                         mixedOffsets, mixedSizes,
                                         mixedStrides);
  }
};

/// A canonicalizer wrapper to replace SubViewOps.
struct SubViewCanonicalizer {
  void operator()(PatternRewriter &rewriter, SubViewOp op, SubViewOp newOp) {
    rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(), newOp);
  }
};

void SubViewOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results
      .add<OpWithOffsetSizesAndStridesConstantArgumentFolder<
               SubViewOp, SubViewReturnTypeCanonicalizer, SubViewCanonicalizer>,
           SubViewOpMemRefCastFolder, TrivialSubViewOpFolder>(context);
}

OpFoldResult SubViewOp::fold(ArrayRef<Attribute> operands) {
  auto resultShapedType = getResult().getType().cast<ShapedType>();
  auto sourceShapedType = source().getType().cast<ShapedType>();

  if (resultShapedType.hasStaticShape() &&
      resultShapedType == sourceShapedType) {
    return getViewSource();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

/// Build a strided memref type by applying `permutationMap` tp `memRefType`.
static MemRefType inferTransposeResultType(MemRefType memRefType,
                                           AffineMap permutationMap) {
  auto rank = memRefType.getRank();
  auto originalSizes = memRefType.getShape();
  // Compute permuted sizes.
  SmallVector<int64_t, 4> sizes(rank, 0);
  for (const auto &en : llvm::enumerate(permutationMap.getResults()))
    sizes[en.index()] =
        originalSizes[en.value().cast<AffineDimExpr>().getPosition()];

  // Compute permuted strides.
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto res = getStridesAndOffset(memRefType, strides, offset);
  assert(succeeded(res) && strides.size() == static_cast<unsigned>(rank));
  (void)res;
  auto map =
      makeStridedLinearLayoutMap(strides, offset, memRefType.getContext());
  map = permutationMap ? map.compose(permutationMap) : map;
  return MemRefType::Builder(memRefType)
      .setShape(sizes)
      .setLayout(AffineMapAttr::get(map));
}

void TransposeOp::build(OpBuilder &b, OperationState &result, Value in,
                        AffineMapAttr permutation,
                        ArrayRef<NamedAttribute> attrs) {
  auto permutationMap = permutation.getValue();
  assert(permutationMap);

  auto memRefType = in.getType().cast<MemRefType>();
  // Compute result type.
  MemRefType resultType = inferTransposeResultType(memRefType, permutationMap);

  build(b, result, resultType, in, attrs);
  result.addAttribute(TransposeOp::getPermutationAttrName(), permutation);
}

// transpose $in $permutation attr-dict : type($in) `to` type(results)
void TransposeOp::print(OpAsmPrinter &p) {
  p << " " << in() << " " << permutation();
  p.printOptionalAttrDict((*this)->getAttrs(), {getPermutationAttrName()});
  p << " : " << in().getType() << " to " << getType();
}

ParseResult TransposeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType in;
  AffineMap permutation;
  MemRefType srcType, dstType;
  if (parser.parseOperand(in) || parser.parseAffineMap(permutation) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(in, srcType, result.operands) ||
      parser.parseKeywordType("to", dstType) ||
      parser.addTypeToList(dstType, result.types))
    return failure();

  result.addAttribute(TransposeOp::getPermutationAttrName(),
                      AffineMapAttr::get(permutation));
  return success();
}

LogicalResult TransposeOp::verify() {
  if (!permutation().isPermutation())
    return emitOpError("expected a permutation map");
  if (permutation().getNumDims() != getShapedType().getRank())
    return emitOpError("expected a permutation map of same rank as the input");

  auto srcType = in().getType().cast<MemRefType>();
  auto dstType = getType().cast<MemRefType>();
  auto transposedType = inferTransposeResultType(srcType, permutation());
  if (dstType != transposedType)
    return emitOpError("output type ")
           << dstType << " does not match transposed input type " << srcType
           << ", " << transposedType;
  return success();
}

OpFoldResult TransposeOp::fold(ArrayRef<Attribute>) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}

//===----------------------------------------------------------------------===//
// ViewOp
//===----------------------------------------------------------------------===//

LogicalResult ViewOp::verify() {
  auto baseType = getOperand(0).getType().cast<MemRefType>();
  auto viewType = getType();

  // The base memref should have identity layout map (or none).
  if (!baseType.getLayout().isIdentity())
    return emitError("unsupported map for base memref type ") << baseType;

  // The result memref should have identity layout map (or none).
  if (!viewType.getLayout().isIdentity())
    return emitError("unsupported map for result memref type ") << viewType;

  // The base memref and the view memref should be in the same memory space.
  if (baseType.getMemorySpace() != viewType.getMemorySpace())
    return emitError("different memory spaces specified for base memref "
                     "type ")
           << baseType << " and view memref type " << viewType;

  // Verify that we have the correct number of sizes for the result type.
  unsigned numDynamicDims = viewType.getNumDynamicDims();
  if (sizes().size() != numDynamicDims)
    return emitError("incorrect number of size operands for type ") << viewType;

  return success();
}

Value ViewOp::getViewSource() { return source(); }

namespace {

struct ViewOpShapeFolder : public OpRewritePattern<ViewOp> {
  using OpRewritePattern<ViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ViewOp viewOp,
                                PatternRewriter &rewriter) const override {
    // Return if none of the operands are constants.
    if (llvm::none_of(viewOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    // Get result memref type.
    auto memrefType = viewOp.getType();

    // Get offset from old memref view type 'memRefType'.
    int64_t oldOffset;
    SmallVector<int64_t, 4> oldStrides;
    if (failed(getStridesAndOffset(memrefType, oldStrides, oldOffset)))
      return failure();
    assert(oldOffset == 0 && "Expected 0 offset");

    SmallVector<Value, 4> newOperands;

    // Offset cannot be folded into result type.

    // Fold any dynamic dim operands which are produced by a constant.
    SmallVector<int64_t, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType.getRank());

    unsigned dynamicDimPos = 0;
    unsigned rank = memrefType.getRank();
    for (unsigned dim = 0, e = rank; dim < e; ++dim) {
      int64_t dimSize = memrefType.getDimSize(dim);
      // If this is already static dimension, keep it.
      if (!ShapedType::isDynamic(dimSize)) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto *defOp = viewOp.sizes()[dynamicDimPos].getDefiningOp();
      if (auto constantIndexOp =
              dyn_cast_or_null<arith::ConstantIndexOp>(defOp)) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp.value());
      } else {
        // Dynamic shape dimension not folded; copy operand from old memref.
        newShapeConstants.push_back(dimSize);
        newOperands.push_back(viewOp.sizes()[dynamicDimPos]);
      }
      dynamicDimPos++;
    }

    // Create new memref type with constant folded dims.
    MemRefType newMemRefType =
        MemRefType::Builder(memrefType).setShape(newShapeConstants);
    // Nothing new, don't fold.
    if (newMemRefType == memrefType)
      return failure();

    // Create new ViewOp.
    auto newViewOp = rewriter.create<ViewOp>(viewOp.getLoc(), newMemRefType,
                                             viewOp.getOperand(0),
                                             viewOp.byte_shift(), newOperands);
    // Insert a cast so we have the same type as the old memref type.
    rewriter.replaceOpWithNewOp<CastOp>(viewOp, viewOp.getType(), newViewOp);
    return success();
  }
};

struct ViewOpMemrefCastFolder : public OpRewritePattern<ViewOp> {
  using OpRewritePattern<ViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ViewOp viewOp,
                                PatternRewriter &rewriter) const override {
    Value memrefOperand = viewOp.getOperand(0);
    CastOp memrefCastOp = memrefOperand.getDefiningOp<CastOp>();
    if (!memrefCastOp)
      return failure();
    Value allocOperand = memrefCastOp.getOperand();
    AllocOp allocOp = allocOperand.getDefiningOp<AllocOp>();
    if (!allocOp)
      return failure();
    rewriter.replaceOpWithNewOp<ViewOp>(viewOp, viewOp.getType(), allocOperand,
                                        viewOp.byte_shift(), viewOp.sizes());
    return success();
  }
};

} // namespace

void ViewOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<ViewOpShapeFolder, ViewOpMemrefCastFolder>(context);
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicRMWOp::verify() {
  if (getMemRefType().getRank() != getNumOperands() - 2)
    return emitOpError(
        "expects the number of subscripts to be equal to memref rank");
  switch (kind()) {
  case arith::AtomicRMWKind::addf:
  case arith::AtomicRMWKind::maxf:
  case arith::AtomicRMWKind::minf:
  case arith::AtomicRMWKind::mulf:
    if (!value().getType().isa<FloatType>())
      return emitOpError() << "with kind '"
                           << arith::stringifyAtomicRMWKind(kind())
                           << "' expects a floating-point type";
    break;
  case arith::AtomicRMWKind::addi:
  case arith::AtomicRMWKind::maxs:
  case arith::AtomicRMWKind::maxu:
  case arith::AtomicRMWKind::mins:
  case arith::AtomicRMWKind::minu:
  case arith::AtomicRMWKind::muli:
  case arith::AtomicRMWKind::ori:
  case arith::AtomicRMWKind::andi:
    if (!value().getType().isa<IntegerType>())
      return emitOpError() << "with kind '"
                           << arith::stringifyAtomicRMWKind(kind())
                           << "' expects an integer type";
    break;
  default:
    break;
  }
  return success();
}

OpFoldResult AtomicRMWOp::fold(ArrayRef<Attribute> operands) {
  /// atomicrmw(memrefcast) -> atomicrmw
  if (succeeded(foldMemRefCast(*this, value())))
    return getResult();
  return OpFoldResult();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MemRef/IR/MemRefOps.cpp.inc"
