//===- LinalgOps.cpp - Implementation of the linalg operations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

/// Forward declarations.
template <typename NamedStructuredOpType>
static void buildNamedStructuredOpRegionAndAttributes(
    Builder &builder, OperationState &result, TypeRange operandTypes,
    TypeRange tensorResultTypes);
template <typename NamedStructuredOpType>
static void printNamedStructuredOp(OpAsmPrinter &p, NamedStructuredOpType op);
template <typename NamedStructuredOpType>
static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result);
template <typename NamedStructuredOpType>
static LogicalResult verifyNamedStructuredOp(NamedStructuredOpType op);

/// This is a common class used for patterns of the form
/// ```
///    someop(memrefcast) -> someop
/// ```
/// It folds the source of the memref_cast into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<MemRefCastOp>();
    if (castOp && canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

///////////////////// Operations defined with Tablegen /////////////////////////
// For such operations that do not correspond to library calls (i.e. defined in
// LinalgOps.td), we define an overloaded `print` function and a
// parse`className` function.

//===----------------------------------------------------------------------===//
// GenericOps
//===----------------------------------------------------------------------===//

template <typename GenericOpType>
static void printGenericOp(OpAsmPrinter &p, GenericOpType op) {
  auto attrNames = op.linalgTraitAttrNames();
  llvm::StringSet<> linalgTraitAttrsSet;
  linalgTraitAttrsSet.insert(attrNames.begin(), attrNames.end());
  SmallVector<NamedAttribute, 8> attrs;
  for (auto attr : op.getAttrs())
    if (linalgTraitAttrsSet.count(attr.first.strref()) > 0)
      attrs.push_back(attr);

  auto dictAttr = DictionaryAttr::get(attrs, op.getContext());
  p << op.getOperationName() << " " << dictAttr;
  p.printOptionalAttrDict(op.getAttrs(), attrNames);
  p << " " << op.getOperands();
  if (!op.region().empty())
    p.printRegion(op.region());
  p << ": " << op.getOperandTypes();
  auto outputTensorTypes = op.getResultTypes();
  if (!outputTensorTypes.empty())
    p << " -> " << outputTensorTypes;
}

static void print(OpAsmPrinter &p, GenericOp op) { printGenericOp(p, op); }

static void print(OpAsmPrinter &p, IndexedGenericOp op) {
  printGenericOp(p, op);
}

static ParseResult parseGenericOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> operandsInfo, regionOperandsInfo;
  DictionaryAttr dictAttr;
  // Parse the core linalg traits that must check into a dictAttr.
  // The name is unimportant as we will overwrite result.attributes.
  // The core linalg traits must contain the information necessary to pass the
  // verifier.
  if (parser.parseAttribute(dictAttr, "_", result.attributes))
    return failure();
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());

  // Optional attributes may be added.
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOperandList(operandsInfo))
    return failure();

  Region &region = *result.addRegion();
  SmallVector<Type, 8> operandTypes, regionTypes;
  if (parser.parseRegion(region, regionOperandsInfo, regionTypes))
    return failure();
  if (parser.parseColonTypeList(operandTypes))
    return failure();
  // Generic ops may specify that a subset of its outputs are tensors. Such
  // outputs are specified in the result type.
  SmallVector<Type, 8> tensorResultTypes;
  if (parser.parseOptionalArrowTypeList(tensorResultTypes))
    return failure();
  if (!tensorResultTypes.empty())
    result.addTypes(tensorResultTypes);
  return parser.resolveOperands(operandsInfo, operandTypes,
                                parser.getCurrentLocation(), result.operands);
}

namespace {
template <typename GenericOpType>
struct BlockArgsVerifier {
  static LogicalResult verify(GenericOpType op, Block &block);
};

template <typename GenericOpType>
LogicalResult BlockArgsVerifier<GenericOpType>::verify(GenericOpType op,
                                                       Block &block) {
  auto nOperands = op.getNumOperands();
  if (block.getNumArguments() != nOperands)
    return op.emitOpError("expected number of block arguments to match number "
                          "of operands");

  // Note: the number and type of yield values are checked in the YieldOp.
  auto nInputViews = op.getNumInputs();
  for (unsigned i = 0; i < nOperands; ++i) {
    auto viewType = op.getShapedType(i);
    if (viewType.getElementType() != block.getArgument(i).getType())
      return op.emitOpError("expected block argument ")
             << (i + 1) << " of the same type as elemental type of "
             << ((i < nInputViews) ? "input " : "output ")
             << "operand: " << viewType;
  }
  return success();
}

template <>
LogicalResult BlockArgsVerifier<IndexedGenericOp>::verify(IndexedGenericOp op,
                                                          Block &block) {
  auto nInputViews = op.getNumInputs();
  auto nLoops = op.getNumLoops();
  auto nOperands = op.getNumOperands();
  if (block.getNumArguments() != nOperands + nLoops)
    return op.emitOpError(
        "expected number of block arguments to match number of operands + "
        "number of loops");

  // Note: the number and type of yield values are checked in the YieldOp.
  for (unsigned i = 0; i < nLoops; ++i)
    if (!block.getArgument(i).getType().isIndex())
      return op.emitOpError("expected block argument ")
             << (i + 1) << " to be an index";

  for (unsigned i = 0; i < nOperands; ++i) {
    unsigned memrefArgIndex = i + nLoops;
    auto viewType = op.getShapedType(i);
    if (viewType.getElementType() !=
        block.getArgument(memrefArgIndex).getType())
      return op.emitOpError("expected block argument ")
             << (memrefArgIndex + 1)
             << " of the same type as elemental type of "
             << ((i < nInputViews) ? "input " : "output ")
             << "operand: " << viewType;
  }
  return success();
}
} // namespace

template <typename GenericOpType>
static LogicalResult verifyGenericOp(GenericOpType op) {
  auto nInputViews = op.getNumInputs();
  auto nLoops = op.getNumLoops();
  auto nInputsAndOutputBuffers = op.getNumInputsAndOutputBuffers();
  if (nInputsAndOutputBuffers != llvm::size(op.views()))
    return op.emitOpError("expected exactly ")
           << nInputsAndOutputBuffers
           << " inputs (tensor or buffer) and output buffer operands";

  auto &region = op.region();
  if (region.getBlocks().size() != 1)
    return op.emitOpError("expected region with 1 block");
  if (failed(BlockArgsVerifier<GenericOpType>::verify(op, region.front())))
    return failure();

  SmallVector<AffineMap, 4> indexingMaps;
  indexingMaps.reserve(op.indexing_maps().size());
  for (auto en : llvm::enumerate(op.indexing_maps())) {
    auto idx = en.index();
    auto m = en.value().template cast<AffineMapAttr>().getValue();
    indexingMaps.push_back(m); // Save reference to map for further checks.
    auto view = (idx < nInputViews) ? op.getInputShapedType(idx)
                                    : op.getOutputShapedType(idx - nInputViews);

    if (m.getNumSymbols() != 0)
      return op.emitOpError("expected indexing_map #")
             << idx << " to have no symbols";

    if (m.getNumDims() != nLoops)
      return op.emitOpError("expected indexing_map #")
             << idx << " to have " << nLoops
             << " dim(s) to match the number of loops";

    if (m.getNumResults() != view.getRank())
      return op.emitOpError("expected indexing_map #")
             << idx << " results to match view rank: " << view;
  }

  auto concatMap = concatAffineMaps(indexingMaps);
  auto aggregateMap = inversePermutation(concatMap);
  if (!aggregateMap)
    return op.emitOpError("expected the concatenation of maps in indexing_map "
                          "to be invertible");

  return success();
}

static LogicalResult verify(GenericOp op) { return verifyGenericOp(op); }
static LogicalResult verify(IndexedGenericOp op) { return verifyGenericOp(op); }

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

/// Collapse reassociation maps that are used in pair of reshape ops where one
/// is a producer and other is the consumer. Only valid to use this method when
/// both the producer and consumer are collapsing dimensions or both are
/// expanding dimensions.
///
/// For example,
///   mapsProducer = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
///                   affine_map<(d0, d1, d2, d3, d4) -> (d2)>,
///                   affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>]
///   mapsConsumer = [affine_map<(d0, d1, d2) -> (d0, d1)>,
///                   affine_map<(d0, d1, d2) -> (d2)>]
///
/// is folded into
///
///   result = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
///             affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>]
static ArrayAttr collapseReassociationMaps(ArrayRef<AffineMap> mapsProducer,
                                           ArrayRef<AffineMap> mapsConsumer,
                                           MLIRContext *context) {
  if (mapsProducer.empty() || mapsConsumer.empty() ||
      mapsProducer[0].getNumDims() < mapsConsumer[0].getNumDims() ||
      mapsProducer.size() != mapsConsumer[0].getNumDims())
    return nullptr;
  unsigned numLhsDims = mapsProducer[0].getNumDims();
  unsigned currDim = 0;
  SmallVector<AffineExpr, 4> reassociations;
  SmallVector<Attribute, 4> reassociationMaps;
  for (AffineMap rhs : mapsConsumer) {
    for (AffineExpr rhsExpr : rhs.getResults()) {
      AffineDimExpr dimExpr = rhsExpr.cast<AffineDimExpr>();
      for (int i = 0, e = mapsProducer[dimExpr.getPosition()].getNumResults();
           i < e; ++i) {
        reassociations.push_back(getAffineDimExpr(currDim++, context));
      }
    }
    reassociationMaps.push_back(AffineMapAttr::get(AffineMap::get(
        numLhsDims, /*numSymbols =*/0, reassociations, context)));
    reassociations.clear();
  }
  return ArrayAttr::get(reassociationMaps, context);
}

namespace {
/// Pattern to collapse producer/consumer reshape ops that are both collapsing
/// dimensions or are both expanding dimensions.
template <typename ReshapeOpTy>
struct CollapseReshapeOps : public OpRewritePattern<ReshapeOpTy> {
  using OpRewritePattern<ReshapeOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOpTy reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto srcReshapeOp = reshapeOp.src().template getDefiningOp<ReshapeOpTy>();
    if (!srcReshapeOp)
      return failure();

    auto areReshapeOpsFoldable = [](ShapedType largerType,
                                    ShapedType intermediateType,
                                    ShapedType smallerType) -> bool {
      return largerType.getRank() > intermediateType.getRank() &&
             intermediateType.getRank() > smallerType.getRank() &&
             smallerType.getRank() > 0;
    };
    // Check if producer and consumer are both expanding dims.
    if (areReshapeOpsFoldable(reshapeOp.getResultType(), reshapeOp.getSrcType(),
                              srcReshapeOp.getSrcType())) {
      rewriter.replaceOpWithNewOp<ReshapeOpTy>(
          reshapeOp, reshapeOp.getResultType(), srcReshapeOp.src(),
          collapseReassociationMaps(reshapeOp.getReassociationMaps(),
                                    srcReshapeOp.getReassociationMaps(),
                                    rewriter.getContext()));
      return success();
    }
    // Check if producer and consumer are both collapsing dims.
    else if (areReshapeOpsFoldable(srcReshapeOp.getSrcType(),
                                   reshapeOp.getSrcType(),
                                   reshapeOp.getResultType())) {
      rewriter.replaceOpWithNewOp<ReshapeOpTy>(
          reshapeOp, reshapeOp.getResultType(), srcReshapeOp.src(),
          collapseReassociationMaps(srcReshapeOp.getReassociationMaps(),
                                    reshapeOp.getReassociationMaps(),
                                    rewriter.getContext()));
      return success();
    }
    return failure();
  }
};
} // namespace

template <typename ReshapeOpTy>
static OpFoldResult foldReshapeOp(ReshapeOpTy reshapeOp) {
  // Fold producer-consumer reshape ops that where the operand type of the
  // producer is same as the return type of the consumer. This can only be
  // verified if the shapes in question are static.
  ReshapeOpTy reshapeSrcOp =
      reshapeOp.src().template getDefiningOp<ReshapeOpTy>();
  if (reshapeSrcOp && reshapeSrcOp.getSrcType().hasStaticShape() &&
      reshapeOp.getResultType().hasStaticShape() &&
      reshapeSrcOp.getSrcType() == reshapeOp.getResultType())
    return reshapeSrcOp.src();
  return nullptr;
}

/// Return true if the reassociation specification is valid, false otherwise.
/// When false, the `invalidIndex` integer pointer is optionally filled with the
/// index of the offending reassociation map.
static bool isReassociationValid(ArrayRef<AffineMap> reassociation,
                                 int *invalidIndex = nullptr) {
  if (reassociation.empty())
    return true;
  unsigned nDims = reassociation[0].getNumDims();
  unsigned nextExpectedDim = 0;
  for (auto it : llvm::enumerate(reassociation)) {
    auto m = it.value();
    if (m.getNumDims() != nDims || m.getNumSymbols() != 0) {
      if (invalidIndex)
        *invalidIndex = it.index();
      return false;
    }
    for (auto e : m.getResults()) {
      auto d = e.dyn_cast<AffineDimExpr>();
      if (!d || d.getPosition() != nextExpectedDim++) {
        if (invalidIndex)
          *invalidIndex = it.index();
        return false;
      }
    }
  }
  if (nextExpectedDim != nDims) {
    if (invalidIndex)
      *invalidIndex = reassociation.size() - 1;
    return false;
  }
  return true;
}

/// Detect whether memref dims [dim, dim + extent) can be reshaped without
/// copies.
static bool isReshapableDimBand(unsigned dim, unsigned extent,
                                ArrayRef<int64_t> sizes,
                                ArrayRef<AffineExpr> strides) {
  assert(sizes.size() == strides.size() && "mismatched ranks");
  // off by 1 indexing to avoid out of bounds
  //                       V
  for (auto idx = dim, e = dim + extent; idx + 1 < e; ++idx) {
    // Only bands of static shapes are reshapable. This is due to the fact that
    // there is no relation between dynamic sizes and dynamic strides: we do not
    // have enough information to know whether a "-1" size corresponds to the
    // proper symbol in the AffineExpr of a stride.
    if (ShapedType::isDynamic(sizes[dim + 1]))
      return false;
    // TODO(ntv) Refine this by passing the proper nDims and nSymbols so we can
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
    if (!isReshapableDimBand(currentDim, dim, sizes, strides)) {
      size = ShapedType::kDynamicSize;
      stride = AffineExpr();
    } else {
      for (unsigned d = 0; d < dim; ++d)
        size *= sizes[currentDim + d];
    }
    newSizes.push_back(size);
    newStrides.push_back(stride);
    currentDim += dim;
  }

  // Early-exit: if `type` is contiguous, the result must be contiguous.
  if (canonicalizeStridedLayout(type).getAffineMaps().empty())
    return MemRefType::Builder(type).setShape(newSizes).setAffineMaps({});

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
      MemRefType::Builder(type).setShape(newSizes).setAffineMaps({layout}));
}

/// Helper functions assert Attribute of the proper type in attr and returns the
/// corresponding vector.
/// TODO(rridle,ntv) this should be evolved into a generic
/// `getRangeOfType<AffineMap>(ArrayAttr attrs)` that does not copy.
static SmallVector<AffineMap, 4> getAffineMaps(ArrayAttr attrs) {
  return llvm::to_vector<8>(llvm::map_range(
      attrs, [](Attribute a) { return a.cast<AffineMapAttr>().getValue(); }));
}

template <typename AffineExprTy>
unsigned getMaxPosOfType(ArrayRef<ArrayRef<AffineExpr>> exprArrays) {
  unsigned pos = 0;
  for (auto exprs : exprArrays) {
    for (auto expr : exprs) {
      expr.walk([&pos](AffineExpr e) {
        if (auto d = e.dyn_cast<AffineExprTy>())
          pos = std::max(pos, d.getPosition());
      });
    }
  }
  return pos;
}

static SmallVector<AffineMap, 4>
getSymbolLessAffineMaps(ArrayRef<ArrayRef<AffineExpr>> reassociation) {
  unsigned maxDim = getMaxPosOfType<AffineDimExpr>(reassociation);
  assert(getMaxPosOfType<AffineSymbolExpr>(reassociation) == 0 &&
         "Expected symbol-less expressions");
  SmallVector<AffineMap, 4> maps;
  maps.reserve(reassociation.size());
  for (auto exprs : reassociation) {
    assert(exprs.size() != 0);
    maps.push_back(AffineMap::get(maxDim + 1, 0, exprs, exprs[0].getContext()));
  }
  return maps;
}

void mlir::linalg::ReshapeOp::build(
    OpBuilder &b, OperationState &result, Value src,
    ArrayRef<ArrayRef<AffineExpr>> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  auto memRefType = src.getType().cast<MemRefType>();
  auto resultType = computeReshapeCollapsedType(memRefType, maps);
  build(b, result, resultType, src, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      b.getAffineMapArrayAttr(maps));
}

void mlir::linalg::ReshapeOp::build(
    OpBuilder &b, OperationState &result, Type resultType, Value src,
    ArrayRef<ArrayRef<AffineExpr>> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  build(b, result, resultType, src, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      b.getAffineMapArrayAttr(maps));
}

// Common verifier for reshape-like types. Fills `expandedType` and
// `collapsedType` with the proper `src` or `result` type.
template <typename Op, typename T>
static LogicalResult verifyReshapeLikeTypes(Op op, T &expandedType,
                                            T &collapsedType) {
  expandedType = op.getSrcType();
  collapsedType = op.getResultType();
  unsigned expandedRank = expandedType.getRank();
  unsigned collapsedRank = collapsedType.getRank();
  bool isCollapse = expandedRank > collapsedRank;
  if (!isCollapse) {
    std::swap(expandedRank, collapsedRank);
    std::swap(expandedType, collapsedType);
  }
  if (expandedRank == 0)
    return op.emitOpError("expected non-zero memref ranks");
  if (expandedRank == collapsedRank)
    return op.emitOpError("expected to collapse or expand dims");

  if (collapsedRank == 0) {
    // If collapsed rank is 0, then expanded type must be static shaped and of
    // sizes 1.
    if (llvm::any_of(expandedType.getShape(),
                     [](int64_t dim) -> bool { return dim != 1; }))
      return op.emitOpError(
          "invalid to reshape tensor/memref with non-unit extent dimensions to "
          "zero-rank tensor/memref");
    return success();
  }
  if (collapsedRank != op.reassociation().size())
    return op.emitOpError("expected rank of the collapsed type(")
           << collapsedRank << ") to be the number of reassociation maps("
           << op.reassociation().size() << ")";
  auto maps = getAffineMaps(op.reassociation());
  for (auto it : llvm::enumerate(maps))
    if (it.value().getNumDims() != expandedRank)
      return op.emitOpError("expected reassociation map #")
             << it.index() << " of same rank as expanded memref("
             << expandedRank << "), but got " << it.value().getNumDims();
  int invalidIdx = 0;
  if (!isReassociationValid(maps, &invalidIdx))
    return op.emitOpError("expected reassociation map #")
           << invalidIdx << " to be valid and contiguous";
  return success();
}

static LogicalResult verify(ReshapeOp op) {
  MemRefType expandedType, collapsedType;
  if (failed(verifyReshapeLikeTypes(op, expandedType, collapsedType)))
    return failure();
  auto maps = getAffineMaps(op.reassociation());
  MemRefType expectedType = computeReshapeCollapsedType(expandedType, maps);
  if (collapsedType != expectedType)
    return op.emitOpError("expected collapsed type to be ")
           << expectedType << ", but got " << collapsedType;
  return success();
}

void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<CollapseReshapeOps<ReshapeOp>>(context);
}

//===----------------------------------------------------------------------===//
// TensorReshapeOp
//===----------------------------------------------------------------------===//

/// Compute the RankedTensorType obtained by applying `reassociation` to `type`.
static RankedTensorType
computeTensorReshapeCollapsedType(RankedTensorType type,
                                  ArrayRef<AffineMap> reassociation) {
  auto shape = type.getShape();
  SmallVector<int64_t, 4> newShape;
  newShape.reserve(reassociation.size());

  // Use the fact that reassociation is valid to simplify the logic: only use
  // each map's rank.
  assert(isReassociationValid(reassociation) && "invalid reassociation");
  unsigned currentDim = 0;
  for (AffineMap m : reassociation) {
    unsigned dim = m.getNumResults();
    auto band = shape.slice(currentDim, dim);
    int64_t size = 1;
    if (llvm::is_contained(band, ShapedType::kDynamicSize))
      size = ShapedType::kDynamicSize;
    else
      for (unsigned d = 0; d < dim; ++d)
        size *= shape[currentDim + d];
    newShape.push_back(size);
    currentDim += dim;
  }

  return RankedTensorType::get(newShape, type.getElementType());
}

void mlir::linalg::TensorReshapeOp::build(
    OpBuilder &b, OperationState &result, Value src,
    ArrayRef<ArrayRef<AffineExpr>> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  auto resultType = computeTensorReshapeCollapsedType(
      src.getType().cast<RankedTensorType>(), maps);
  build(b, result, resultType, src, attrs);
  result.addAttribute(TensorReshapeOp::getReassociationAttrName(),
                      b.getAffineMapArrayAttr(maps));
}

void mlir::linalg::TensorReshapeOp::build(
    OpBuilder &b, OperationState &result, Type resultType, Value src,
    ArrayRef<ArrayRef<AffineExpr>> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  build(b, result, resultType, src, attrs);
  result.addAttribute(TensorReshapeOp::getReassociationAttrName(),
                      b.getAffineMapArrayAttr(maps));
}

static LogicalResult verify(TensorReshapeOp op) {
  RankedTensorType expandedType, collapsedType;
  if (failed(verifyReshapeLikeTypes(op, expandedType, collapsedType)))
    return failure();
  auto maps = getAffineMaps(op.reassociation());
  // TODO(ntv): expanding a ? with a non-constant is under-specified. Error
  // out.
  RankedTensorType expectedType =
      computeTensorReshapeCollapsedType(expandedType, maps);
  if (collapsedType != expectedType)
    return op.emitOpError("expected collapsed type to be ")
           << expectedType << ", but got " << collapsedType;
  return success();
}

void TensorReshapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CollapseReshapeOps<TensorReshapeOp>>(context);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//
void mlir::linalg::SliceOp::build(OpBuilder &b, OperationState &result,
                                  Value base, ValueRange indexings) {
  result.addOperands(base);
  result.addOperands(indexings);

  auto memRefType = base.getType().cast<MemRefType>();
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto res = getStridesAndOffset(memRefType, strides, offset);
  assert(succeeded(res) && strides.size() == indexings.size());
  (void)res;

  unsigned rank = memRefType.getRank();
  // TODO(ntv): propagate static size and stride information when available.
  SmallVector<int64_t, 4> sizes(rank, -1); // -1 encodes dynamic size.
  result.addTypes({MemRefType::Builder(memRefType)
                       .setShape(sizes)
                       .setAffineMaps(makeStridedLinearLayoutMap(
                           strides, offset, b.getContext()))});
}

static void print(OpAsmPrinter &p, SliceOp op) {
  auto indexings = op.indexings();
  p << SliceOp::getOperationName() << " " << op.view() << "[" << indexings
    << "] ";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getBaseViewType();
  if (!indexings.empty())
    p << ", " << op.indexings().getTypes();
  p << ", " << op.getType();
}

static ParseResult parseSliceOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType baseInfo;
  SmallVector<OpAsmParser::OperandType, 8> operands;
  SmallVector<Type, 8> types;
  if (parser.parseOperand(baseInfo) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types))
    return failure();

  if (types.size() < 2)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected at least input and result view types");

  ArrayRef<Type> indexingTypes = ArrayRef<Type>(types).drop_front().drop_back();
  return failure(
      parser.resolveOperand(baseInfo, types.front(), result.operands) ||
      (!operands.empty() &&
       parser.resolveOperands(operands, indexingTypes,
                              operands.front().location, result.operands)) ||
      parser.addTypeToList(types.back(), result.types));
}

static LogicalResult verify(SliceOp op) {
  unsigned rank = op.getBaseViewRank();
  if (rank != llvm::size(op.indexings()))
    return op.emitOpError("expected ")
           << rank << " indexings, got " << llvm::size(op.indexings());
  unsigned index = 0;
  for (auto indexing : op.indexings()) {
    if (indexing.getType().isa<IndexType>())
      --rank;
    ++index;
  }
  if (op.getRank() != rank)
    return op.emitOpError() << "expected rank of the view(" << op.getRank()
                            << ") to be the number of ranges(" << rank << ")";
  return success();
}

Value SliceOp::getViewSource() { return view(); }

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
void mlir::linalg::TransposeOp::build(OpBuilder &b, OperationState &result,
                                      Value view, AffineMapAttr permutation,
                                      ArrayRef<NamedAttribute> attrs) {
  auto permutationMap = permutation.getValue();
  assert(permutationMap);

  auto memRefType = view.getType().cast<MemRefType>();
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
  auto map = makeStridedLinearLayoutMap(strides, offset, b.getContext());
  map = permutationMap ? map.compose(permutationMap) : map;
  // Compute result type.
  MemRefType resultType =
      MemRefType::Builder(memRefType).setShape(sizes).setAffineMaps(map);

  build(b, result, resultType, view, attrs);
  result.addAttribute(TransposeOp::getPermutationAttrName(), permutation);
}

static void print(OpAsmPrinter &p, TransposeOp op) {
  p << op.getOperationName() << " " << op.view() << " " << op.permutation();
  p.printOptionalAttrDict(op.getAttrs(),
                          {TransposeOp::getPermutationAttrName()});
  p << " : " << op.view().getType();
}

static ParseResult parseTransposeOp(OpAsmParser &parser,
                                    OperationState &result) {
  OpAsmParser::OperandType view;
  AffineMap permutation;
  MemRefType type;
  if (parser.parseOperand(view) || parser.parseAffineMap(permutation) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(view, type, result.operands) ||
      parser.addTypeToList(type, result.types))
    return failure();

  result.addAttribute(TransposeOp::getPermutationAttrName(),
                      AffineMapAttr::get(permutation));
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, YieldOp op) {
  p << op.getOperationName();
  if (op.getNumOperands() > 0)
    p << ' ' << op.getOperands();
  p.printOptionalAttrDict(op.getAttrs());
  if (op.getNumOperands() > 0)
    p << " : " << op.getOperandTypes();
}

static ParseResult parseYieldOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

// Check the operand number and types must match the element types of the
// LinalgOp interface's shaped operands.
static LogicalResult verifyYield(YieldOp op, LinalgOp linalgOpInterface) {
  auto nOutputs = linalgOpInterface.getNumOutputs();
  if (op.getNumOperands() != nOutputs)
    return op.emitOpError("expected number of yield values (")
           << nOutputs << ") to match the number of operands of the enclosing "
           << "LinalgOp (" << op.getNumOperands() << ")";

  for (unsigned i = 0; i != nOutputs; ++i) {
    auto elementType =
        linalgOpInterface.getOutputShapedType(i).getElementType();
    if (op.getOperand(i).getType() != elementType)
      return op.emitOpError("type of yield operand ")
             << (i + 1) << " (" << op.getOperand(i).getType()
             << ") doesn't match "
             << "the element type of the enclosing linalg.generic op ("
             << elementType << ")";
  }
  return success();
}

static LogicalResult verify(YieldOp op) {
  auto *parentOp = op.getParentOp();
  if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty())
    return op.emitOpError("expected single non-empty parent region");

  if (auto linalgOp = dyn_cast<LinalgOp>(parentOp))
    return verifyYield(op, cast<LinalgOp>(parentOp));

  return op.emitOpError("expected parent op with LinalgOp interface");
}

/////// Operations corresponding to library calls defined with Tablegen ////////

static LogicalResult verify(FillOp op) {
  auto viewType = op.getOutputShapedType(0);
  auto fillType = op.value().getType();
  if (viewType.getElementType() != fillType)
    return op.emitOpError("expects fill type to match view elemental type");
  return success();
}

static LogicalResult verify(CopyOp op) {
  auto outputViewType = op.getOutputShapedType(0);
  auto inputViewType = op.getInputShapedType(0);
  if (inputViewType.getElementType() != outputViewType.getElementType())
    return op.emitOpError("expects views of the same type");
  if (inputViewType.getRank() != outputViewType.getRank())
    return op.emitOpError("expects views of the same rank");
  auto rank = op.getNumParallelLoops();
  auto inputPermutationMap = op.inputPermutation();
  if (inputPermutationMap) {
    if (inputPermutationMap->getNumInputs() != rank)
      return op.emitOpError("expects optional input_permutation map of rank ")
             << rank;
    if (!inputPermutationMap->isPermutation())
      return op.emitOpError(
          "expects optional input_permutation map to be a permutation");
  }
  auto outputPermutationMap = op.outputPermutation();
  if (outputPermutationMap) {
    if (outputPermutationMap->getNumInputs() != rank)
      return op.emitOpError("expects optional output_permutation map of rank ")
             << rank;
    if (!outputPermutationMap->isPermutation())
      return op.emitOpError(
          "expects optional output_permutation map to be a permutation");
  }
  if (rank == 0 && inputPermutationMap)
    return op.emitOpError("expected no input permutation when rank == 0");
  if (rank == 0 && outputPermutationMap)
    return op.emitOpError("expected no output permutation when rank == 0");
  return success();
}

template <typename LinalgPoolingOp>
static LogicalResult verifyStrideOrDilation(LinalgPoolingOp op,
                                            ArrayRef<Attribute> attrs,
                                            bool isStride) {
  auto strideOrDilation = isStride ? "stride" : "dilation";
  if (attrs.size() != op.getNumWindowLoops())
    return op.emitOpError("expects num ")
           << strideOrDilation
           << "s equal to number of window dimensions: " << attrs.size()
           << " vs " << op.getNumWindowLoops();
  return success();
}

static LogicalResult verify(ConvOp op) {
  auto oType = op.output().getType().cast<MemRefType>();
  auto fType = op.filter().getType().cast<MemRefType>();
  auto iType = op.input().getType().cast<MemRefType>();
  if (oType.getElementType() != iType.getElementType() ||
      oType.getElementType() != fType.getElementType())
    return op.emitOpError("expects memref elemental types to match");
  if (oType.getRank() != iType.getRank() || oType.getRank() != fType.getRank())
    return op.emitOpError("expects memref ranks to match");
  if (auto strides = op.strides()) {
    if (failed(
            verifyStrideOrDilation(op, strides->getValue(), /*isStride=*/true)))
      return failure();
  }
  if (auto dilations = op.dilations()) {
    if (failed(verifyStrideOrDilation(op, dilations->getValue(),
                                      /*isStride=*/false)))
      return failure();
  }
  return success();
}

template <typename PoolingOp>
static LogicalResult verifySingleInputPoolingOp(PoolingOp op) {
  auto inputType = op.input().getType().template cast<MemRefType>();
  auto outputType = op.output().getType().template cast<MemRefType>();
  if (outputType.getElementType() != inputType.getElementType())
    return op.emitOpError("expects memref elemental types to match");

  auto windowDimsType = op.windowDims().getType().template cast<MemRefType>();
  if (outputType.getRank() != inputType.getRank() ||
      outputType.getRank() != windowDimsType.getRank())
    return op.emitOpError("expects memref ranks to match");

  if (auto strides = op.strides()) {
    if (failed(
            verifyStrideOrDilation(op, strides->getValue(), /*isStride=*/true)))
      return failure();
  }
  if (auto dilations = op.dilations()) {
    if (failed(verifyStrideOrDilation(op, dilations->getValue(),
                                      /*isStride=*/false)))
      return failure();
  }
  return success();
}

static LogicalResult verify(PoolingMaxOp op) {
  return verifySingleInputPoolingOp(op);
}
static LogicalResult verify(PoolingMinOp op) {
  return verifySingleInputPoolingOp(op);
}
static LogicalResult verify(PoolingSumOp op) {
  return verifySingleInputPoolingOp(op);
}

namespace mlir {
namespace linalg {

#include "mlir/Dialect/Linalg/IR/LinalgStructuredOpsInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"

} // namespace linalg
} // namespace mlir

AffineMap mlir::linalg::extractOrIdentityMap(Optional<AffineMap> maybeMap,
                                             unsigned rank,
                                             MLIRContext *context) {
  if (maybeMap)
    return maybeMap.getValue();
  if (rank == 0)
    return AffineMap::get(context);
  return AffineMap::getMultiDimIdentityMap(rank, context);
}

SmallVector<AffineExpr, 4>
mlir::linalg::makeAffineDimExprs(unsigned num, unsigned &startIdx,
                                 MLIRContext *context) {
  SmallVector<AffineExpr, 4> res;
  res.reserve(num);
  for (unsigned i = 0; i < num; ++i)
    res.push_back(getAffineDimExpr(startIdx++, context));
  return res;
}

template <typename PoolingOp>
SmallVector<AffineExpr, 4>
mlir::linalg::weightedPoolingInputIndex(PoolingOp op,
                                        ArrayRef<AffineExpr> outputDims,
                                        ArrayRef<AffineExpr> windowDims) {
  assert(outputDims.size() == windowDims.size());
  SmallVector<AffineExpr, 4> res;
  res.reserve(outputDims.size());
  for (unsigned i = 0, e = outputDims.size(); i < e; ++i) {
    // TODO(ntv): add a level of indirection to linalg.generic.
    auto expr = op.getStride(i) * outputDims[i] +
                op.getDilation(i) * windowDims[i] - op.getLowPad(i);
    res.push_back(expr);
  }
  return res;
}

#define INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(OP_TYPE)                      \
  template SmallVector<AffineExpr, 4>                                          \
  mlir::linalg::weightedPoolingInputIndex<OP_TYPE>(                            \
      OP_TYPE op, ArrayRef<AffineExpr> outputDims,                             \
      ArrayRef<AffineExpr> windowDims);

INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(ConvOp)
INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(PoolingMaxOp)
INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(PoolingMinOp)
INSTANTIATE_WEIGHTED_POOLING_INPUT_INDEX(PoolingSumOp)

SmallVector<AffineExpr, 4> mlir::linalg::concat(ArrayRef<AffineExpr> a,
                                                ArrayRef<AffineExpr> b) {
  auto rangeA = llvm::make_range(a.begin(), a.end());
  auto rangeB = llvm::make_range(b.begin(), b.end());
  auto concatRanges = llvm::concat<const AffineExpr>(rangeA, rangeB);
  return llvm::to_vector<4>(concatRanges);
}

static void appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto memref = t.dyn_cast<MemRefType>()) {
    ss << "view";
    for (auto size : memref.getShape())
      if (size < 0)
        ss << "sx";
      else
        ss << size << "x";
    appendMangledType(ss, memref.getElementType());
  } else if (auto vec = t.dyn_cast<VectorType>()) {
    ss << "vector";
    llvm::interleave(
        vec.getShape(), [&](int64_t i) { ss << i; }, [&]() { ss << "x"; });
    appendMangledType(ss, vec.getElementType());
  } else if (t.isSignlessIntOrIndexOrFloat()) {
    ss << t;
  } else {
    llvm_unreachable("Invalid type for linalg library name mangling");
  }
}

std::string mlir::linalg::generateLibraryCallName(Operation *op) {
  assert(isa<LinalgOp>(op));
  std::string name(op->getName().getStringRef().str());
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);
  ss << "_";
  auto types = op->getOperandTypes();
  llvm::interleave(
      types.begin(), types.end(), [&](Type t) { appendMangledType(ss, t); },
      [&]() { ss << "_"; });
  return ss.str();
}

// TODO(ntv, rriddle): Consider making all this boilerplate easy to autogenerate
// with Tablegen. This seems a desirable property in the context of OpInterfaces
// where a Linalg "named" op **isa** LinalgOp.
LogicalResult ConvOp::fold(ArrayRef<Attribute>,
                           SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult PoolingMaxOp::fold(ArrayRef<Attribute>,
                                 SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult PoolingMinOp::fold(ArrayRef<Attribute>,
                                 SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult PoolingSumOp::fold(ArrayRef<Attribute>,
                                 SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult CopyOp::fold(ArrayRef<Attribute>,
                           SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult DotOp::fold(ArrayRef<Attribute>,
                          SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult FillOp::fold(ArrayRef<Attribute>,
                           SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult GenericOp::fold(ArrayRef<Attribute>,
                              SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult IndexedGenericOp::fold(ArrayRef<Attribute>,
                                     SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult MatvecOp::fold(ArrayRef<Attribute>,
                             SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
LogicalResult MatmulOp::fold(ArrayRef<Attribute>,
                             SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
OpFoldResult ReshapeOp::fold(ArrayRef<Attribute>) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return foldReshapeOp(*this);
}
OpFoldResult SliceOp::fold(ArrayRef<Attribute>) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}
OpFoldResult TensorReshapeOp::fold(ArrayRef<Attribute>) {
  return foldReshapeOp(*this);
}
OpFoldResult TransposeOp::fold(ArrayRef<Attribute>) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}

//===----------------------------------------------------------------------===//
// Auto-generated Linalg named ops.
//===----------------------------------------------------------------------===//

template <typename NamedStructuredOpType>
void buildNamedStructuredOpRegionAndAttributes(Builder &builder,
                                               OperationState &result,
                                               TypeRange operandTypes,
                                               TypeRange tensorResultTypes) {
  Region &region = *result.addRegion();
  Block *body = new Block();
  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  for (auto t : operandTypes)
    body->addArgument(getElementTypeOrSelf(t));
  for (auto t : tensorResultTypes)
    body->addArgument(getElementTypeOrSelf(t));
  region.push_back(body);

  OpBuilder opBuilder(builder.getContext());
  opBuilder.setInsertionPointToStart(&region.front());
  mlir::edsc::ScopedContext scope(opBuilder, builder.getUnknownLoc());
  NamedStructuredOpType::regionBuilder(*body);

  auto indexingMaps = builder.getAffineMapArrayAttr(
      NamedStructuredOpType::referenceIndexingMaps(operandTypes,
                                                   tensorResultTypes));
  result.addAttribute(getIndexingMapsAttrName(), indexingMaps);

  auto iterators =
      builder.getStrArrayAttr(NamedStructuredOpType::referenceIterators(
          operandTypes, tensorResultTypes));
  result.addAttribute(getIteratorTypesAttrName(), iterators);
}

template <typename NamedStructuredOpType>
static void printNamedStructuredOp(OpAsmPrinter &p, NamedStructuredOpType op) {
  std::array<StringRef, 2> silentAttrNames{getIndexingMapsAttrName(),
                                           getIteratorTypesAttrName()};
  p << op.getOperationName() << ' ';
  p.printOptionalAttrDict(op.getAttrs(), silentAttrNames);
  p << ' ' << op.getOperands();
  p << ": (" << op.getOperandTypes() << ")";
  auto outputTensorTypes = op.getResultTypes();
  if (!outputTensorTypes.empty())
    p << " -> (" << outputTensorTypes << ")";
}

template <typename NamedStructuredOpType>
static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> operandsInfo;

  // Optional attributes may be added.
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOperandList(operandsInfo))
    return failure();

  SmallVector<Type, 8> operandTypes;
  if (parser.parseColon() || parser.parseLParen() ||
      parser.parseTypeList(operandTypes) || parser.parseRParen())
    return failure();

  // Generic ops may specify that a subset of its outputs are tensors. Such
  // outputs are specified in the result type.
  SmallVector<Type, 8> tensorResultTypes;
  if (parser.parseOptionalArrowTypeList(tensorResultTypes))
    return failure();

  if (!tensorResultTypes.empty())
    result.addTypes(tensorResultTypes);

  buildNamedStructuredOpRegionAndAttributes<NamedStructuredOpType>(
      parser.getBuilder(), result, operandTypes, tensorResultTypes);

  return parser.resolveOperands(operandsInfo, operandTypes,
                                parser.getCurrentLocation(), result.operands);
}

template <typename NamedStructuredOpType>
static LogicalResult verifyNamedStructuredOp(NamedStructuredOpType op) {
  return verifyGenericOp<NamedStructuredOpType>(op);
}

#include "mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.cpp.inc"

// TODO: Determine whether we can generate the folders and verifiers.
LogicalResult BatchMatmulOp::fold(ArrayRef<Attribute>,
                                  SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}
