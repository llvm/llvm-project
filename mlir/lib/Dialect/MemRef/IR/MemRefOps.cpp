//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::memref;

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *MemRefDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return builder.create<mlir::ConstantOp>(loc, type, value);
}

/// Extract int64_t values from the assumed ArrayAttr of IntegerAttr.
static SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(
      llvm::map_range(attr.cast<ArrayAttr>(), [](Attribute a) -> int64_t {
        return a.cast<IntegerAttr>().getInt();
      }));
}

/// Helper function to dispatch an OpFoldResult into either the `dynamicVec` if
/// it is a Value or into `staticVec` if it is an IntegerAttr.
/// In the case of a Value, a copy of the `sentinel` value is also pushed to
/// `staticVec`. This is useful to extract mixed static and dynamic entries that
/// come from an AttrSizedOperandSegments trait.
static void dispatchIndexOpFoldResult(OpFoldResult ofr,
                                      SmallVectorImpl<Value> &dynamicVec,
                                      SmallVectorImpl<int64_t> &staticVec,
                                      int64_t sentinel) {
  if (auto v = ofr.dyn_cast<Value>()) {
    dynamicVec.push_back(v);
    staticVec.push_back(sentinel);
    return;
  }
  APInt apInt = ofr.dyn_cast<Attribute>().cast<IntegerAttr>().getValue();
  staticVec.push_back(apInt.getSExtValue());
}

static void dispatchIndexOpFoldResults(ArrayRef<OpFoldResult> ofrs,
                                       SmallVectorImpl<Value> &dynamicVec,
                                       SmallVectorImpl<int64_t> &staticVec,
                                       int64_t sentinel) {
  for (auto ofr : ofrs)
    dispatchIndexOpFoldResult(ofr, dynamicVec, staticVec, sentinel);
}

//===----------------------------------------------------------------------===//
// Common canonicalization pattern support logic
//===----------------------------------------------------------------------===//

/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref.cast
/// into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto cast = operand.get().getDefiningOp<CastOp>();
    if (cast && !cast.getOperand().getType().isa<UnrankedMemRefType>()) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

//===----------------------------------------------------------------------===//
// Helpers for GlobalOp
//===----------------------------------------------------------------------===//

static Type getTensorTypeFromMemRefType(Type type) {
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
  if (!memRefType.getAffineMaps().empty())
    numSymbols = memRefType.getAffineMaps().front().getNumSymbols();
  if (op.symbolOperands().size() != numSymbols)
    return op.emitOpError(
        "symbol operand count does not equal memref symbol count");

  return success();
}

static LogicalResult verify(AllocOp op) { return verifyAllocLikeOp(op); }

static LogicalResult verify(AllocaOp op) {
  // An alloca op needs to have an ancestor with an allocation scope trait.
  if (!op->getParentWithTrait<OpTrait::AutomaticAllocationScope>())
    return op.emitOpError(
        "requires an ancestor op with AutomaticAllocationScope trait");

  return verifyAllocLikeOp(op);
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
    if (llvm::none_of(alloc.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    auto memrefType = alloc.getType();

    // Ok, we have one or more constant operands.  Collect the non-constant ones
    // and keep track of the resultant memref type to build.
    SmallVector<int64_t, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType.getRank());
    SmallVector<Value, 4> newOperands;

    unsigned dynamicDimPos = 0;
    for (unsigned dim = 0, e = memrefType.getRank(); dim < e; ++dim) {
      int64_t dimSize = memrefType.getDimSize(dim);
      // If this is already static dimension, keep it.
      if (dimSize != -1) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto *defOp = alloc.getOperand(dynamicDimPos).getDefiningOp();
      if (auto constantIndexOp = dyn_cast_or_null<ConstantIndexOp>(defOp)) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp.getValue());
      } else {
        // Dynamic shape dimension not folded; copy operand from old memref.
        newShapeConstants.push_back(-1);
        newOperands.push_back(alloc.getOperand(dynamicDimPos));
      }
      dynamicDimPos++;
    }

    // Create new memref type (which will have fewer dynamic dimensions).
    MemRefType newMemRefType =
        MemRefType::Builder(memrefType).setShape(newShapeConstants);
    assert(static_cast<int64_t>(newOperands.size()) ==
           newMemRefType.getNumDynamicDims());

    // Create and insert the alloc op for the new memref.
    auto newAlloc = rewriter.create<AllocLikeOp>(alloc.getLoc(), newMemRefType,
                                                 newOperands, IntegerAttr());
    // Insert a cast so we have the same type as the old alloc.
    auto resultCast =
        rewriter.create<CastOp>(alloc.getLoc(), newAlloc, alloc.getType());

    rewriter.replaceOp(alloc, {resultCast});
    return success();
  }
};

/// Fold alloc operations with no uses. Alloc has side effects on the heap,
/// but can still be deleted if it has zero uses.
struct SimplifyDeadAlloc : public OpRewritePattern<AllocOp> {
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    if (alloc.use_empty()) {
      rewriter.eraseOp(alloc);
      return success();
    }
    return failure();
  }
};
} // end anonymous namespace.

void AllocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<SimplifyAllocConst<AllocOp>, SimplifyDeadAlloc>(context);
}

void AllocaOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<SimplifyAllocConst<AllocaOp>>(context);
}

//===----------------------------------------------------------------------===//
// AssumeAlignmentOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(AssumeAlignmentOp op) {
  unsigned alignment = op.alignment();
  if (!llvm::isPowerOf2_32(alignment))
    return op.emitOpError("alignment must be power of 2");
  return success();
}

//===----------------------------------------------------------------------===//
// BufferCastOp
//===----------------------------------------------------------------------===//

OpFoldResult BufferCastOp::fold(ArrayRef<Attribute>) {
  if (auto tensorLoad = tensor().getDefiningOp<TensorLoadOp>())
    if (tensorLoad.memref().getType() == getType())
      return tensorLoad.memref();
  return {};
}

namespace {
/// Replace tensor_cast + buffer_cast by buffer_cast + memref_cast.
struct BufferCast : public OpRewritePattern<BufferCastOp> {
  using OpRewritePattern<BufferCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferCastOp bufferCast,
                                PatternRewriter &rewriter) const final {
    auto tensorCastOperand =
        bufferCast.getOperand().getDefiningOp<tensor::CastOp>();
    if (!tensorCastOperand)
      return failure();
    auto srcTensorType =
        tensorCastOperand.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!srcTensorType)
      return failure();
    auto memrefType = MemRefType::get(srcTensorType.getShape(),
                                      srcTensorType.getElementType());
    Value memref = rewriter.create<BufferCastOp>(
        bufferCast.getLoc(), memrefType, tensorCastOperand.getOperand());
    rewriter.replaceOpWithNewOp<CastOp>(bufferCast, bufferCast.getType(),
                                        memref);
    return success();
  }
};

/// Canonicalize memref.tensor_load + memref.buffer_cast to memref.cast when
/// type mismatches prevent `BufferCastOp::fold` to kick in.
struct TensorLoadToMemRef : public OpRewritePattern<BufferCastOp> {
  using OpRewritePattern<BufferCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferCastOp bufferCast,
                                PatternRewriter &rewriter) const final {
    auto tensorLoad = bufferCast.tensor().getDefiningOp<TensorLoadOp>();
    // Bail unless we have a tensor_load + memref.buffer_cast with different
    // types. `BufferCastOp::fold` handles the same type case.
    if (!tensorLoad || tensorLoad.memref().getType() == bufferCast.getType())
      return failure();
    // If types are not cast-compatible, bail.
    if (!CastOp::areCastCompatible(tensorLoad.memref().getType(),
                                   bufferCast.getType()))
      return failure();
    rewriter.replaceOpWithNewOp<CastOp>(bufferCast, bufferCast.getType(),
                                        tensorLoad.memref());
    return success();
  }
};

} // namespace

void BufferCastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<BufferCast, TensorLoadToMemRef>(context);
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
      if (MemRefType::isDynamic(ss) && !MemRefType::isDynamic(st))
        return false;
  }

  // If cast is towards more static offset along any dimension, don't fold.
  if (sourceOffset != resultOffset)
    if (MemRefType::isDynamicStrideOrOffset(sourceOffset) &&
        !MemRefType::isDynamicStrideOrOffset(resultOffset))
      return false;

  // If cast is towards more static strides along any dimension, don't fold.
  for (auto it : llvm::zip(sourceStrides, resultStrides)) {
    auto ss = std::get<0>(it), st = std::get<1>(it);
    if (ss != st)
      if (MemRefType::isDynamicStrideOrOffset(ss) &&
          !MemRefType::isDynamicStrideOrOffset(st))
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
    if (aT.getAffineMaps() != bT.getAffineMaps()) {
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
      for (auto aStride : enumerate(aStrides))
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
    if (aMemSpace != bMemSpace)
      return false;

    return true;
  }

  return false;
}

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  return succeeded(foldMemRefCast(*this)) ? getResult() : Value();
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//
namespace {
/// Fold Dealloc operations that are deallocating an AllocOp that is only used
/// by other Dealloc operations.
struct SimplifyDeadDealloc : public OpRewritePattern<DeallocOp> {
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp dealloc,
                                PatternRewriter &rewriter) const override {
    // Check that the memref operand's defining operation is an AllocOp.
    Value memref = dealloc.memref();
    if (!isa_and_nonnull<AllocOp>(memref.getDefiningOp()))
      return failure();

    // Check that all of the uses of the AllocOp are other DeallocOps.
    for (auto *user : memref.getUsers())
      if (!isa<DeallocOp>(user))
        return failure();

    // Erase the dealloc operation.
    rewriter.eraseOp(dealloc);
    return success();
  }
};
} // end anonymous namespace.

static LogicalResult verify(DeallocOp op) {
  if (!op.memref().getType().isa<MemRefType>())
    return op.emitOpError("operand must be a memref");
  return success();
}

void DeallocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<SimplifyDeadDealloc>(context);
}

LogicalResult DeallocOp::fold(ArrayRef<Attribute> cstOperands,
                              SmallVectorImpl<OpFoldResult> &results) {
  /// dealloc(memrefcast) -> dealloc
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

void DimOp::build(OpBuilder &builder, OperationState &result, Value memref,
                  int64_t index) {
  auto loc = result.location;
  Value indexValue = builder.create<ConstantIndexOp>(loc, index);
  build(builder, result, memref, indexValue);
}

void DimOp::build(OpBuilder &builder, OperationState &result, Value memref,
                  Value index) {
  auto indexTy = builder.getIndexType();
  build(builder, result, indexTy, memref, index);
}

Optional<int64_t> DimOp::getConstantIndex() {
  if (auto constantOp = index().getDefiningOp<ConstantOp>())
    return constantOp.getValue().cast<IntegerAttr>().getInt();
  return {};
}

static LogicalResult verify(DimOp op) {
  // Assume unknown index to be in range.
  Optional<int64_t> index = op.getConstantIndex();
  if (!index.hasValue())
    return success();

  // Check that constant index is not knowingly out of range.
  auto type = op.memrefOrTensor().getType();
  if (auto memrefType = type.dyn_cast<MemRefType>()) {
    if (index.getValue() >= memrefType.getRank())
      return op.emitOpError("index is out of range");
  } else if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    if (index.getValue() >= tensorType.getRank())
      return op.emitOpError("index is out of range");
  } else if (type.isa<UnrankedMemRefType>() || type.isa<UnrankedTensorType>()) {
    // Assume index to be in range.
  } else {
    llvm_unreachable("expected operand with memref type");
  }
  return success();
}

OpFoldResult DimOp::fold(ArrayRef<Attribute> operands) {
  auto index = operands[1].dyn_cast_or_null<IntegerAttr>();

  // All forms of folding require a known index.
  if (!index)
    return {};

  auto argTy = memrefOrTensor().getType();
  // Fold if the shape extent along the given index is known.
  if (auto shapedTy = argTy.dyn_cast<ShapedType>()) {
    // Folding for unranked types (UnrankedMemRefType) is not supported.
    if (!shapedTy.hasRank())
      return {};
    if (!shapedTy.isDynamicDim(index.getInt())) {
      Builder builder(getContext());
      return builder.getIndexAttr(shapedTy.getShape()[index.getInt()]);
    }
  }

  Operation *definingOp = memrefOrTensor().getDefiningOp();

  // dim(memref.tensor_load(memref)) -> dim(memref)
  if (auto tensorLoadOp = dyn_cast_or_null<TensorLoadOp>(definingOp)) {
    setOperand(0, tensorLoadOp.memref());
    return getResult();
  }

  // Fold dim to the operand of tensor.generate.
  if (auto fromElements = dyn_cast_or_null<tensor::GenerateOp>(definingOp)) {
    auto resultType =
        fromElements.getResult().getType().cast<RankedTensorType>();
    // The case where the type encodes the size of the dimension is handled
    // above.
    assert(resultType.getShape()[index.getInt()] ==
           RankedTensorType::kDynamicSize);

    // Find the operand of the fromElements that corresponds to this index.
    auto dynExtents = fromElements.dynamicExtents().begin();
    for (auto dim : resultType.getShape().take_front(index.getInt()))
      if (dim == RankedTensorType::kDynamicSize)
        dynExtents++;

    return Value{*dynExtents};
  }

  // The size at the given index is now known to be a dynamic size.
  unsigned unsignedIndex = index.getValue().getZExtValue();

  if (auto subtensor = dyn_cast_or_null<mlir::SubTensorOp>(definingOp)) {
    assert(subtensor.isDynamicSize(unsignedIndex) &&
           "Expected dynamic subtensor size");
    return subtensor.getDynamicSize(unsignedIndex);
  }

  // Fold dim to the size argument for an `AllocOp`, `ViewOp`, or `SubViewOp`.
  auto memrefType = argTy.dyn_cast<MemRefType>();
  if (!memrefType)
    return {};

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
    assert(subview.isDynamicSize(unsignedIndex) &&
           "Expected dynamic subview size");
    return subview.getDynamicSize(unsignedIndex);
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
    auto reshape = dim.memrefOrTensor().getDefiningOp<ReshapeOp>();

    if (!reshape)
      return failure();

    // Place the load directly after the reshape to ensure that the shape memref
    // was not mutated.
    rewriter.setInsertionPointAfter(reshape);
    rewriter.replaceOpWithNewOp<LoadOp>(dim, reshape.shape(),
                                        llvm::makeArrayRef({dim.index()}));
    return success();
  }
};

/// Fold dim of a dim of a cast into the dim of the source of the tensor cast.
template <typename CastOpTy>
struct DimOfCastOp : public OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = dimOp.memrefOrTensor().getDefiningOp<CastOpTy>();
    if (!castOp)
      return failure();
    Value newSource = castOp.getOperand();
    rewriter.replaceOpWithNewOp<DimOp>(dimOp, newSource, dimOp.index());
    return success();
  }
};
} // end anonymous namespace.

void DimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<DimOfMemRefReshape, DimOfCastOp<BufferCastOp>,
              DimOfCastOp<tensor::CastOp>>(context);
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
  p << getOperationName() << " " << getSrcMemRef() << '[' << getSrcIndices()
    << "], " << getDstMemRef() << '[' << getDstIndices() << "], "
    << getNumElements() << ", " << getTagMemRef() << '[' << getTagIndices()
    << ']';
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

  // DMAs from different memory spaces supported.
  if (getSrcMemorySpace() == getDstMemorySpace())
    return emitOpError("DMA should be between different memory spaces");

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

void DmaWaitOp::build(OpBuilder &builder, OperationState &result,
                      Value tagMemRef, ValueRange tagIndices,
                      Value numElements) {
  result.addOperands(tagMemRef);
  result.addOperands(tagIndices);
  result.addOperands(numElements);
}

void DmaWaitOp::print(OpAsmPrinter &p) {
  p << getOperationName() << " " << getTagMemRef() << '[' << getTagIndices()
    << "], " << getNumElements();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getTagMemRef().getType();
}

// Parse DmaWaitOp.
// Eg:
//   dma_wait %tag[%index], %num_elements : memref<1 x i32, (d0) -> (d0), 4>
//
ParseResult DmaWaitOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 2> tagIndexInfos;
  Type type;
  auto indexType = parser.getBuilder().getIndexType();
  OpAsmParser::OperandType numElementsInfo;

  // Parse tag memref, its indices, and dma size.
  if (parser.parseOperand(tagMemrefInfo) ||
      parser.parseOperandList(tagIndexInfos, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(numElementsInfo) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(tagMemrefInfo, type, result.operands) ||
      parser.resolveOperands(tagIndexInfos, indexType, result.operands) ||
      parser.resolveOperand(numElementsInfo, indexType, result.operands))
    return failure();

  return success();
}

LogicalResult DmaWaitOp::fold(ArrayRef<Attribute> cstOperands,
                              SmallVectorImpl<OpFoldResult> &results) {
  /// dma_wait(memrefcast) -> dma_wait
  return foldMemRefCast(*this);
}

LogicalResult DmaWaitOp::verify() {
  // Mandatory non-variadic operands are tag and the number of elements.
  if (getNumOperands() < 2)
    return emitOpError() << "expected at least 2 operands";

  // Check types of operands. The order of these calls is important: the later
  // calls rely on some type properties to compute the operand position.
  if (!getTagMemRef().getType().isa<MemRefType>())
    return emitOpError() << "expected tag to be of memref type";

  if (getNumOperands() != 2 + getTagMemRefRank())
    return emitOpError() << "expected " << 2 + getTagMemRefRank()
                         << " operands";

  if (!getTagIndices().empty() &&
      !llvm::all_of(getTagIndices().getTypes(),
                    [](Type t) { return t.isIndex(); }))
    return emitOpError() << "expected tag indices to be of index type";

  if (!getNumElements().getType().isIndex())
    return emitOpError()
           << "expected the number of elements to be of index type";

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
    initialValue = UnitAttr::get(parser.getBuilder().getContext());
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

static LogicalResult verify(GlobalOp op) {
  auto memrefType = op.type().dyn_cast<MemRefType>();
  if (!memrefType || !memrefType.hasStaticShape())
    return op.emitOpError("type should be static shaped memref, but got ")
           << op.type();

  // Verify that the initial value, if present, is either a unit attribute or
  // an elements attribute.
  if (op.initial_value().hasValue()) {
    Attribute initValue = op.initial_value().getValue();
    if (!initValue.isa<UnitAttr>() && !initValue.isa<ElementsAttr>())
      return op.emitOpError("initial value should be a unit or elements "
                            "attribute, but got ")
             << initValue;

    // Check that the type of the initial value is compatible with the type of
    // the global variable.
    if (initValue.isa<ElementsAttr>()) {
      Type initType = initValue.getType();
      Type tensorType = getTensorTypeFromMemRefType(memrefType);
      if (initType != tensorType)
        return op.emitOpError("initial value expected to be of type ")
               << tensorType << ", but was of type " << initType;
    }
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

static LogicalResult verify(LoadOp op) {
  if (op.getNumOperands() != 1 + op.getMemRefType().getRank())
    return op.emitOpError("incorrect number of indices for load");
  return success();
}

OpFoldResult LoadOp::fold(ArrayRef<Attribute> cstOperands) {
  /// load(memrefcast) -> load
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return OpFoldResult();
}

namespace {
/// Fold a load on a buffer_cast operation into an tensor.extract on the
/// corresponding tensor.
struct LoadOfBufferCast : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp load,
                                PatternRewriter &rewriter) const override {
    auto buffercast = load.memref().getDefiningOp<BufferCastOp>();
    if (!buffercast)
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(load, buffercast.tensor(),
                                                   load.indices());
    return success();
  }
};
} // end anonymous namespace.

void LoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<LoadOfBufferCast>(context);
}

//===----------------------------------------------------------------------===//
// PrefetchOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, PrefetchOp op) {
  p << PrefetchOp::getOperationName() << " " << op.memref() << '[';
  p.printOperands(op.indices());
  p << ']' << ", " << (op.isWrite() ? "write" : "read");
  p << ", locality<" << op.localityHint();
  p << ">, " << (op.isDataCache() ? "data" : "instr");
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"localityHint", "isWrite", "isDataCache"});
  p << " : " << op.getMemRefType();
}

static ParseResult parsePrefetchOp(OpAsmParser &parser,
                                   OperationState &result) {
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

static LogicalResult verify(PrefetchOp op) {
  if (op.getNumOperands() != 1 + op.getMemRefType().getRank())
    return op.emitOpError("too few indices");

  return success();
}

LogicalResult PrefetchOp::fold(ArrayRef<Attribute> cstOperands,
                               SmallVectorImpl<OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return foldMemRefCast(*this);
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
// completed automatically, like we have for subview and subtensor.
static LogicalResult verify(ReinterpretCastOp op) {
  // The source and result memrefs should be in the same memory space.
  auto srcType = op.source().getType().cast<BaseMemRefType>();
  auto resultType = op.getType().cast<MemRefType>();
  if (srcType.getMemorySpace() != resultType.getMemorySpace())
    return op.emitError("different memory spaces specified for source type ")
           << srcType << " and result memref type " << resultType;
  if (srcType.getElementType() != resultType.getElementType())
    return op.emitError("different element types specified for source type ")
           << srcType << " and result memref type " << resultType;

  // Match sizes in result memref type and in static_sizes attribute.
  for (auto &en :
       llvm::enumerate(llvm::zip(resultType.getShape(),
                                 extractFromI64ArrayAttr(op.static_sizes())))) {
    int64_t resultSize = std::get<0>(en.value());
    int64_t expectedSize = std::get<1>(en.value());
    if (resultSize != expectedSize)
      return op.emitError("expected result type with size = ")
             << expectedSize << " instead of " << resultSize
             << " in dim = " << en.index();
  }

  // Match offset and strides in static_offset and static_strides attributes if
  // result memref type has an affine map specified.
  if (!resultType.getAffineMaps().empty()) {
    int64_t resultOffset;
    SmallVector<int64_t, 4> resultStrides;
    if (failed(getStridesAndOffset(resultType, resultStrides, resultOffset)))
      return failure();

    // Match offset in result memref type and in static_offsets attribute.
    int64_t expectedOffset =
        extractFromI64ArrayAttr(op.static_offsets()).front();
    if (resultOffset != expectedOffset)
      return op.emitError("expected result type with offset = ")
             << resultOffset << " instead of " << expectedOffset;

    // Match strides in result memref type and in static_strides attribute.
    for (auto &en : llvm::enumerate(llvm::zip(
             resultStrides, extractFromI64ArrayAttr(op.static_strides())))) {
      int64_t resultStride = std::get<0>(en.value());
      int64_t expectedStride = std::get<1>(en.value());
      if (resultStride != expectedStride)
        return op.emitError("expected result type with stride = ")
               << expectedStride << " instead of " << resultStride
               << " in dim = " << en.index();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReshapeOp op) {
  Type operandType = op.source().getType();
  Type resultType = op.result().getType();

  Type operandElementType = operandType.cast<ShapedType>().getElementType();
  Type resultElementType = resultType.cast<ShapedType>().getElementType();
  if (operandElementType != resultElementType)
    return op.emitOpError("element types of source and destination memref "
                          "types should be the same");

  if (auto operandMemRefType = operandType.dyn_cast<MemRefType>())
    if (!operandMemRefType.getAffineMaps().empty())
      return op.emitOpError(
          "source memref type should have identity affine map");

  int64_t shapeSize = op.shape().getType().cast<MemRefType>().getDimSize(0);
  auto resultMemRefType = resultType.dyn_cast<MemRefType>();
  if (resultMemRefType) {
    if (!resultMemRefType.getAffineMaps().empty())
      return op.emitOpError(
          "result memref type should have identity affine map");
    if (shapeSize == ShapedType::kDynamicSize)
      return op.emitOpError("cannot use shape operand with dynamic length to "
                            "reshape to statically-ranked memref type");
    if (shapeSize != resultMemRefType.getRank())
      return op.emitOpError(
          "length of shape operand differs from the result's memref rank");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(StoreOp op) {
  if (op.getNumOperands() != 2 + op.getMemRefType().getRank())
    return op.emitOpError("store index operand count not equal to memref rank");

  return success();
}

LogicalResult StoreOp::fold(ArrayRef<Attribute> cstOperands,
                            SmallVectorImpl<OpFoldResult> &results) {
  /// store(memrefcast) -> store
  return foldMemRefCast(*this);
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
} // end namespace saturated_arith
} // end namespace

/// A subview result type can be fully inferred from the source type and the
/// static representation of offsets, sizes and strides. Special sentinels
/// encode the dynamic case.
Type SubViewOp::inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<int64_t> leadingStaticOffsets,
                                ArrayRef<int64_t> leadingStaticSizes,
                                ArrayRef<int64_t> leadingStaticStrides) {
  // A subview may specify only a leading subset of offset/sizes/strides in
  // which case we complete with offset=0, sizes from memref type and strides=1.
  unsigned rank = sourceMemRefType.getRank();
  assert(leadingStaticOffsets.size() <= rank &&
         "unexpected leadingStaticOffsets overflow");
  assert(leadingStaticSizes.size() <= rank &&
         "unexpected leadingStaticSizes overflow");
  assert(leadingStaticStrides.size() <= rank &&
         "unexpected leadingStaticStrides overflow");
  auto staticOffsets = llvm::to_vector<4>(leadingStaticOffsets);
  auto staticSizes = llvm::to_vector<4>(leadingStaticSizes);
  auto staticStrides = llvm::to_vector<4>(leadingStaticStrides);
  unsigned numTrailingOffsets = rank - staticOffsets.size();
  unsigned numTrailingSizes = rank - staticSizes.size();
  unsigned numTrailingStrides = rank - staticStrides.size();
  staticOffsets.append(numTrailingOffsets, 0);
  llvm::append_range(staticSizes,
                     sourceMemRefType.getShape().take_back(numTrailingSizes));
  staticStrides.append(numTrailingStrides, 1);

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
                                ArrayRef<OpFoldResult> leadingStaticOffsets,
                                ArrayRef<OpFoldResult> leadingStaticSizes,
                                ArrayRef<OpFoldResult> leadingStaticStrides) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(leadingStaticOffsets, dynamicOffsets,
                             staticOffsets, ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(leadingStaticSizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(leadingStaticStrides, dynamicStrides,
                             staticStrides, ShapedType::kDynamicStrideOrOffset);
  return SubViewOp::inferResultType(sourceMemRefType, staticOffsets,
                                    staticSizes, staticStrides)
      .cast<MemRefType>();
}

Type SubViewOp::inferRankReducedResultType(
    unsigned resultRank, MemRefType sourceRankedTensorType,
    ArrayRef<int64_t> leadingStaticOffsets,
    ArrayRef<int64_t> leadingStaticSizes,
    ArrayRef<int64_t> leadingStaticStrides) {
  auto inferredType =
      inferResultType(sourceRankedTensorType, leadingStaticOffsets,
                      leadingStaticSizes, leadingStaticStrides)
          .cast<MemRefType>();
  assert(inferredType.getRank() >= resultRank && "expected ");
  int rankDiff = inferredType.getRank() - resultRank;
  if (rankDiff > 0) {
    auto shape = inferredType.getShape();
    llvm::SmallDenseSet<unsigned> dimsToProject;
    mlir::getPositionsOfShapeOne(rankDiff, shape, dimsToProject);
    SmallVector<int64_t> projectedShape;
    for (unsigned pos = 0, e = shape.size(); pos < e; ++pos)
      if (!dimsToProject.contains(pos))
        projectedShape.push_back(shape[pos]);

    AffineMap map;
    auto maps = inferredType.getAffineMaps();
    if (!maps.empty() && maps.front())
      map = getProjectedMap(maps.front(), dimsToProject);
    inferredType =
        MemRefType::get(projectedShape, inferredType.getElementType(), map,
                        inferredType.getMemorySpace());
  }
  return inferredType;
}

Type SubViewOp::inferRankReducedResultType(
    unsigned resultRank, MemRefType sourceRankedTensorType,
    ArrayRef<OpFoldResult> leadingStaticOffsets,
    ArrayRef<OpFoldResult> leadingStaticSizes,
    ArrayRef<OpFoldResult> leadingStaticStrides) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(leadingStaticOffsets, dynamicOffsets,
                             staticOffsets, ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(leadingStaticSizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(leadingStaticStrides, dynamicStrides,
                             staticStrides, ShapedType::kDynamicStrideOrOffset);
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

enum SubViewVerificationResult {
  Success,
  RankTooLarge,
  SizeMismatch,
  ElemTypeMismatch,
  MemSpaceMismatch,
  AffineMapMismatch
};

/// Checks if `original` Type type can be rank reduced to `reduced` type.
/// This function is slight variant of `is subsequence` algorithm where
/// not matching dimension must be 1.
static SubViewVerificationResult
isRankReducedType(Type originalType, Type candidateReducedType,
                  std::string *errMsg = nullptr) {
  if (originalType == candidateReducedType)
    return SubViewVerificationResult::Success;
  if (!originalType.isa<MemRefType>())
    return SubViewVerificationResult::Success;
  if (originalType.isa<MemRefType>() && !candidateReducedType.isa<MemRefType>())
    return SubViewVerificationResult::Success;

  ShapedType originalShapedType = originalType.cast<ShapedType>();
  ShapedType candidateReducedShapedType =
      candidateReducedType.cast<ShapedType>();

  // Rank and size logic is valid for all ShapedTypes.
  ArrayRef<int64_t> originalShape = originalShapedType.getShape();
  ArrayRef<int64_t> candidateReducedShape =
      candidateReducedShapedType.getShape();
  unsigned originalRank = originalShape.size(),
           candidateReducedRank = candidateReducedShape.size();
  if (candidateReducedRank > originalRank)
    return SubViewVerificationResult::RankTooLarge;

  auto optionalUnusedDimsMask =
      computeRankReductionMask(originalShape, candidateReducedShape);

  // Sizes cannot be matched in case empty vector is returned.
  if (!optionalUnusedDimsMask.hasValue())
    return SubViewVerificationResult::SizeMismatch;

  if (originalShapedType.getElementType() !=
      candidateReducedShapedType.getElementType())
    return SubViewVerificationResult::ElemTypeMismatch;

  // Strided layout logic is relevant for MemRefType only.
  MemRefType original = originalType.cast<MemRefType>();
  MemRefType candidateReduced = candidateReducedType.cast<MemRefType>();
  if (original.getMemorySpace() != candidateReduced.getMemorySpace())
    return SubViewVerificationResult::MemSpaceMismatch;

  llvm::SmallDenseSet<unsigned> unusedDims = optionalUnusedDimsMask.getValue();
  auto inferredType =
      getProjectedMap(getStridedLinearLayoutMap(original), unusedDims);
  AffineMap candidateLayout;
  if (candidateReduced.getAffineMaps().empty())
    candidateLayout = getStridedLinearLayoutMap(candidateReduced);
  else
    candidateLayout = candidateReduced.getAffineMaps().front();
  assert(inferredType.getNumResults() == 1 &&
         candidateLayout.getNumResults() == 1);
  if (inferredType.getNumSymbols() != candidateLayout.getNumSymbols() ||
      inferredType.getNumDims() != candidateLayout.getNumDims()) {
    if (errMsg) {
      llvm::raw_string_ostream os(*errMsg);
      os << "inferred type: " << inferredType;
    }
    return SubViewVerificationResult::AffineMapMismatch;
  }
  // Check that the difference of the affine maps simplifies to 0.
  AffineExpr diffExpr =
      inferredType.getResult(0) - candidateLayout.getResult(0);
  diffExpr = simplifyAffineExpr(diffExpr, inferredType.getNumDims(),
                                inferredType.getNumSymbols());
  auto cst = diffExpr.dyn_cast<AffineConstantExpr>();
  if (!(cst && cst.getValue() == 0)) {
    if (errMsg) {
      llvm::raw_string_ostream os(*errMsg);
      os << "inferred type: " << inferredType;
    }
    return SubViewVerificationResult::AffineMapMismatch;
  }
  return SubViewVerificationResult::Success;
}

template <typename OpTy>
static LogicalResult produceSubViewErrorMsg(SubViewVerificationResult result,
                                            OpTy op, Type expectedType,
                                            StringRef errMsg = "") {
  auto memrefType = expectedType.cast<ShapedType>();
  switch (result) {
  case SubViewVerificationResult::Success:
    return success();
  case SubViewVerificationResult::RankTooLarge:
    return op.emitError("expected result rank to be smaller or equal to ")
           << "the source rank. " << errMsg;
  case SubViewVerificationResult::SizeMismatch:
    return op.emitError("expected result type to be ")
           << expectedType
           << " or a rank-reduced version. (mismatch of result sizes) "
           << errMsg;
  case SubViewVerificationResult::ElemTypeMismatch:
    return op.emitError("expected result element type to be ")
           << memrefType.getElementType() << errMsg;
  case SubViewVerificationResult::MemSpaceMismatch:
    return op.emitError("expected result and source memory spaces to match.")
           << errMsg;
  case SubViewVerificationResult::AffineMapMismatch:
    return op.emitError("expected result type to be ")
           << expectedType
           << " or a rank-reduced version. (mismatch of result affine map) "
           << errMsg;
  }
  llvm_unreachable("unexpected subview verification result");
}

/// Verifier for SubViewOp.
static LogicalResult verify(SubViewOp op) {
  MemRefType baseType = op.getSourceType();
  MemRefType subViewType = op.getType();

  // The base memref and the view memref should be in the same memory space.
  if (baseType.getMemorySpace() != subViewType.getMemorySpace())
    return op.emitError("different memory spaces specified for base memref "
                        "type ")
           << baseType << " and subview memref type " << subViewType;

  // Verify that the base memref type has a strided layout map.
  if (!isStrided(baseType))
    return op.emitError("base type ") << baseType << " is not strided";

  // Verify result type against inferred type.
  auto expectedType = SubViewOp::inferResultType(
      baseType, extractFromI64ArrayAttr(op.static_offsets()),
      extractFromI64ArrayAttr(op.static_sizes()),
      extractFromI64ArrayAttr(op.static_strides()));

  std::string errMsg;
  auto result = isRankReducedType(expectedType, subViewType, &errMsg);
  return produceSubViewErrorMsg(result, op, expectedType, errMsg);
}

raw_ostream &mlir::operator<<(raw_ostream &os, Range &range) {
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
            : b.create<ConstantIndexOp>(loc, op.getStaticOffset(idx));
    Value size = op.isDynamicSize(idx)
                     ? op.getDynamicSize(idx)
                     : b.create<ConstantIndexOp>(loc, op.getStaticSize(idx));
    Value stride =
        op.isDynamicStride(idx)
            ? op.getDynamicStride(idx)
            : b.create<ConstantIndexOp>(loc, op.getStaticStride(idx));
    res.emplace_back(Range{offset, size, stride});
  }
  return res;
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

    /// Deduce the resultType of the SubViewOp using `inferSubViewResultType` on
    /// the cast source operand type and the SubViewOp static information. This
    /// is the resulting type if the MemRefCastOp were folded.
    auto resultType = SubViewOp::inferRankReducedResultType(
        subViewOp.getType().getRank(),
        castOp.source().getType().cast<MemRefType>(),
        subViewOp.getMixedOffsets(), subViewOp.getMixedSizes(),
        subViewOp.getMixedStrides());
    Value newSubView = rewriter.create<SubViewOp>(
        subViewOp.getLoc(), resultType, castOp.source(), subViewOp.offsets(),
        subViewOp.sizes(), subViewOp.strides(), subViewOp.static_offsets(),
        subViewOp.static_sizes(), subViewOp.static_strides());
    rewriter.replaceOpWithNewOp<CastOp>(subViewOp, subViewOp.getType(),
                                        newSubView);
    return success();
  }
};
} // namespace

/// A canonicalizer wrapper to replace SubViewOps.
struct SubViewCanonicalizer {
  void operator()(PatternRewriter &rewriter, SubViewOp op, SubViewOp newOp) {
    rewriter.replaceOpWithNewOp<CastOp>(op, newOp, op.getType());
  }
};

void SubViewOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<OpWithOffsetSizesAndStridesConstantArgumentFolder<
                  SubViewOp, SubViewCanonicalizer>,
              SubViewOpMemRefCastFolder>(context);
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
// TensorLoadOp
//===----------------------------------------------------------------------===//

OpFoldResult TensorLoadOp::fold(ArrayRef<Attribute>) {
  if (auto bufferCast = memref().getDefiningOp<BufferCastOp>())
    // Approximate alias analysis by conservatively folding only when no there
    // is no interleaved operation.
    if (bufferCast->getBlock() == this->getOperation()->getBlock() &&
        bufferCast->getNextNode() == this->getOperation())
      return bufferCast.tensor();
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
  for (auto en : llvm::enumerate(permutationMap.getResults()))
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
  return MemRefType::Builder(memRefType).setShape(sizes).setAffineMaps(map);
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
static void print(OpAsmPrinter &p, TransposeOp op) {
  p << "memref.transpose " << op.in() << " " << op.permutation();
  p.printOptionalAttrDict(op->getAttrs(),
                          {TransposeOp::getPermutationAttrName()});
  p << " : " << op.in().getType() << " to " << op.getType();
}

static ParseResult parseTransposeOp(OpAsmParser &parser,
                                    OperationState &result) {
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

static LogicalResult verify(TransposeOp op) {
  if (!op.permutation().isPermutation())
    return op.emitOpError("expected a permutation map");
  if (op.permutation().getNumDims() != op.getShapedType().getRank())
    return op.emitOpError(
        "expected a permutation map of same rank as the input");

  auto srcType = op.in().getType().cast<MemRefType>();
  auto dstType = op.getType().cast<MemRefType>();
  auto transposedType = inferTransposeResultType(srcType, op.permutation());
  if (dstType != transposedType)
    return op.emitOpError("output type ")
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

static ParseResult parseViewOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType srcInfo;
  SmallVector<OpAsmParser::OperandType, 1> offsetInfo;
  SmallVector<OpAsmParser::OperandType, 4> sizesInfo;
  auto indexType = parser.getBuilder().getIndexType();
  Type srcType, dstType;
  llvm::SMLoc offsetLoc;
  if (parser.parseOperand(srcInfo) || parser.getCurrentLocation(&offsetLoc) ||
      parser.parseOperandList(offsetInfo, OpAsmParser::Delimiter::Square))
    return failure();

  if (offsetInfo.size() != 1)
    return parser.emitError(offsetLoc) << "expects 1 offset operand";

  return failure(
      parser.parseOperandList(sizesInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(srcInfo, srcType, result.operands) ||
      parser.resolveOperands(offsetInfo, indexType, result.operands) ||
      parser.resolveOperands(sizesInfo, indexType, result.operands) ||
      parser.parseKeywordType("to", dstType) ||
      parser.addTypeToList(dstType, result.types));
}

static void print(OpAsmPrinter &p, ViewOp op) {
  p << op.getOperationName() << ' ' << op.getOperand(0) << '[';
  p.printOperand(op.byte_shift());
  p << "][" << op.sizes() << ']';
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op.getOperand(0).getType() << " to " << op.getType();
}

static LogicalResult verify(ViewOp op) {
  auto baseType = op.getOperand(0).getType().cast<MemRefType>();
  auto viewType = op.getType();

  // The base memref should have identity layout map (or none).
  if (baseType.getAffineMaps().size() > 1 ||
      (baseType.getAffineMaps().size() == 1 &&
       !baseType.getAffineMaps()[0].isIdentity()))
    return op.emitError("unsupported map for base memref type ") << baseType;

  // The result memref should have identity layout map (or none).
  if (viewType.getAffineMaps().size() > 1 ||
      (viewType.getAffineMaps().size() == 1 &&
       !viewType.getAffineMaps()[0].isIdentity()))
    return op.emitError("unsupported map for result memref type ") << viewType;

  // The base memref and the view memref should be in the same memory space.
  if (baseType.getMemorySpace() != viewType.getMemorySpace())
    return op.emitError("different memory spaces specified for base memref "
                        "type ")
           << baseType << " and view memref type " << viewType;

  // Verify that we have the correct number of sizes for the result type.
  unsigned numDynamicDims = viewType.getNumDynamicDims();
  if (op.sizes().size() != numDynamicDims)
    return op.emitError("incorrect number of size operands for type ")
           << viewType;

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
      if (auto constantIndexOp = dyn_cast_or_null<ConstantIndexOp>(defOp)) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp.getValue());
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
    rewriter.replaceOpWithNewOp<CastOp>(viewOp, newViewOp, viewOp.getType());
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

} // end anonymous namespace

void ViewOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<ViewOpShapeFolder, ViewOpMemrefCastFolder>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MemRef/IR/MemRefOps.cpp.inc"
