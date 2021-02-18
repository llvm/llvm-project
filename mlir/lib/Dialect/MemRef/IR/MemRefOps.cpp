//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::memref;

/// Matches a ConstantIndexOp.
/// TODO: This should probably just be a general matcher that uses m_Constant
/// and checks the operation for an index type.
static detail::op_matcher<ConstantIndexOp> m_ConstantIndex() {
  return detail::op_matcher<ConstantIndexOp>();
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
// Helpers for Tensor[Load|Store]Op, TensorToMemrefOp, and GlobalOp
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
          return matchPattern(operand, m_ConstantIndex());
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

void AllocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<SimplifyAllocConst<AllocOp>, SimplifyDeadAlloc>(context);
}

void AllocaOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SimplifyAllocConst<AllocaOp>>(context);
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

void DeallocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<SimplifyDeadDealloc>(context);
}

LogicalResult DeallocOp::fold(ArrayRef<Attribute> cstOperands,
                              SmallVectorImpl<OpFoldResult> &results) {
  /// dealloc(memrefcast) -> dealloc
  return foldMemRefCast(*this);
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
// PrefetchOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, PrefetchOp op) {
  p << PrefetchOp::getOperationName() << " " << op.memref() << '[';
  p.printOperands(op.indices());
  p << ']' << ", " << (op.isWrite() ? "write" : "read");
  p << ", locality<" << op.localityHint();
  p << ">, " << (op.isDataCache() ? "data" : "instr");
  p.printOptionalAttrDict(
      op.getAttrs(),
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
  p << "transpose " << op.in() << " " << op.permutation();
  p.printOptionalAttrDict(op.getAttrs(),
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
  p.printOptionalAttrDict(op.getAttrs());
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
          return matchPattern(operand, m_ConstantIndex());
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

void ViewOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<ViewOpShapeFolder, ViewOpMemrefCastFolder>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MemRef/IR/MemRefOps.cpp.inc"
