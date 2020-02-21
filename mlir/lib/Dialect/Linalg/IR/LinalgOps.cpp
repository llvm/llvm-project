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
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

/// Determines whether it is possible to fold it away in the parent Linalg op:
///
/// ```mlir
///   %1 = memref_cast %0 : memref<8x16xf32> to memref<?x?xf32>
///   %2 = linalg.slice %1 ... : memref<?x?xf32> ...
///   // or
///   %1 = memref_cast %0 : memref<8x16xf32, affine_map<(i, j)->(16 * i + j)>>
///          to memref<?x?xf32>
///   linalg.generic(%1 ...) : memref<?x?xf32> ...
/// ```
///
/// into
///
/// ```mlir
///   %2 = linalg.slice %0 ... : memref<8x16xf32> ...
///   // or
///   linalg.generic(%0 ... : memref<8x16xf32, affine_map<(i, j)->(16 * i + j)>>
/// ```
///
static bool canFold(MemRefCastOp castOp) {
  MemRefType sourceType = castOp.source().getType().dyn_cast<MemRefType>();
  MemRefType resultType = castOp.getType().dyn_cast<MemRefType>();

  // If we don't have MemRefType as source and destination, bail out.
  if (!sourceType || !resultType)
    return false;

  // If resultType has a map, it needs to be the same as the source type to
  // canonicalize.
  if (!resultType.getAffineMaps().empty() &&
      sourceType.getAffineMaps() != resultType.getAffineMaps())
    return false;

  // Ensure that:
  //   1. source is static
  //   2. source and target have the same rank (will be extended when needed)
  //   3. if result is partially static, ensure sizes match.
  if (!sourceType.hasStaticShape() ||
      sourceType.getRank() != resultType.getRank())
    return false;

  for (auto it : llvm::zip(sourceType.getShape(), resultType.getShape())) {
    auto sourceSize = std::get<0>(it);
    auto resultSize = std::get<1>(it);
    if (ShapedType::isDynamic(resultSize))
      continue;
    if (sourceSize != resultSize)
      return false;
  }

  // If source has a map, it can only canonicalize if it is the canonical
  // strided layout map.
  if (sourceType.getAffineMaps().empty())
    return true;

  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto res = getStridesAndOffset(sourceType, strides, offset);
  (void)res;
  assert(succeeded(res));
  auto stridedMap =
      makeStridedLinearLayoutMap(strides, offset, castOp.getContext());
  AffineMap sourceMap = sourceType.getAffineMaps().front();
  return sourceMap == stridedMap;
}

/// This is a common class used for patterns of the form
/// ```
///    someop(memrefcast) -> someop
/// ```
/// It folds the source of any memref_cast into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = dyn_cast_or_null<MemRefCastOp>(operand.get().getDefiningOp());
    if (castOp && canFold(castOp)) {
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
  p << op.getOperationName() << " " << dictAttr << " " << op.getOperands();
  if (!op.region().empty())
    p.printRegion(op.region());
  p.printOptionalAttrDict(op.getAttrs(), attrNames);
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
  if (parser.parseAttribute(dictAttr, "_", result.attributes) ||
      parser.parseOperandList(operandsInfo))
    return failure();
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());

  Region &region = *result.addRegion();
  SmallVector<Type, 8> operandTypes, regionTypes;
  // Optional attributes may be added.
  // Either Optional getFunAttrName() attribute or region must be specified.
  if (!dictAttr.get(getFunAttrName()) &&
      parser.parseOptionalRegion(region, regionOperandsInfo, regionTypes))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(operandTypes))
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

template <typename GenericOpType>
static LogicalResult verifyBlockArgs(GenericOpType op, Block &block);

template <> LogicalResult verifyBlockArgs(GenericOp op, Block &block) {
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

template <> LogicalResult verifyBlockArgs(IndexedGenericOp op, Block &block) {
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

template <typename GenericOpType>
static LogicalResult verifyFuncArgs(GenericOpType op, FunctionType funType);

template <typename GenericOpType>
static LogicalResult verifyFuncArgsGeneric(GenericOpType op,
                                           FunctionType funType) {
  auto res = verifyFuncArgs(op, funType);
  if (failed(res))
    return res;

  auto nInputs = op.getNumInputs();
  auto nOutputs = op.getNumOutputs();
  // linalg.generic output element types are exactly the function results.
  for (unsigned idx = 0; idx < nOutputs; ++idx) {
    ShapedType shapedType = op.getShapedType(nInputs + idx);
    if (funType.getResult(idx) != shapedType.getElementType())
      return op.emitOpError("expected function result ")
             << (idx + 1) << " of the same type as elemental type "
             << shapedType.getElementType() << " of output " << (idx + 1);
  }
  return success();
}

template <> LogicalResult verifyFuncArgs(GenericOp op, FunctionType funType) {
  auto nOperands = op.getNumOperands();
  if (funType.getNumInputs() != nOperands)
    return op.emitOpError(
        "expected function arguments to match number of operands");
  if (funType.getNumResults() != op.getNumOutputs())
    return op.emitOpError("expected function results(")
           << funType.getNumResults() << ") to match number of outputs("
           << op.getNumOutputs() << ")";

  // linalg.generic operands element types are exactly the first function
  // arguments.
  for (unsigned idx = 0; idx < nOperands; ++idx) {
    ShapedType shapedType = op.getShapedType(idx);
    if (funType.getInput(idx) != shapedType.getElementType())
      return op.emitOpError("expected function argument ")
             << (idx + 1) << " of the same type as elemental type "
             << shapedType.getElementType() << " of operand " << (idx + 1);
  }

  return success();
}

template <>
LogicalResult verifyFuncArgs(IndexedGenericOp op, FunctionType funType) {
  auto nLoops = op.getNumLoops();
  auto nOutputs = op.getNumOutputs();
  auto nOperands = op.getNumOperands();
  if (funType.getNumInputs() != nOperands + nLoops)
    return op.emitOpError("expected function arguments to match number of "
                          "loops + number of operands");
  if (funType.getNumResults() != nOutputs)
    return op.emitOpError(
        "expected function results to match number of outputs");
  for (unsigned i = 0; i < nLoops; ++i)
    if (!funType.getInput(i).isIndex())
      return op.emitOpError("expected function argument ")
             << (i + 1) << " to be an index";

  // linalg.generic operands element types are exactly the first function
  // arguments.
  for (unsigned idx = 0; idx < nOperands; ++idx) {
    ShapedType shapedType = op.getShapedType(idx);
    if (funType.getInput(idx + nLoops) != shapedType.getElementType())
      return op.emitOpError("expected function argument ")
             << (idx + nLoops + 1) << " of the same type as elemental type "
             << shapedType.getElementType() << " of input " << (idx + 1);
  }

  return success();
}

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
  auto funOp = op.getFunction();
  auto funType = funOp ? funOp.getType() : FunctionType();
  if (!region.empty()) {
    if (region.getBlocks().size() != 1)
      return op.emitOpError("expected region with 1 block");
    if (failed(verifyBlockArgs(op, region.getBlocks().front())))
      return failure();
  } else {
    if (!funOp || !funOp.getType())
      return op.emitOpError(
          "expected function attribute to refer to a defined symbol");
    if (failed(verifyFuncArgsGeneric(op, funType)))
      return failure();
  }

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

    if (m.getNumResults() == 1 && view.getRank() == 0) {
      auto cst = m.getResult(0).template dyn_cast<AffineConstantExpr>();
      if (!cst || cst.getValue() != 0)
        return op.emitOpError("expected indexing_map #")
               << idx << " to be 0 to match 0-D view: " << view;
    } else if (m.getNumResults() != view.getRank()) {
      return op.emitOpError("expected indexing_map #")
             << idx << " results to match view rank: " << view;
    }
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
  return functional::map(
      [](Attribute a) { return a.cast<AffineMapAttr>().getValue(); }, attrs);
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
  for (auto exprs : reassociation)
    maps.push_back(AffineMap::get(maxDim + 1, 0, exprs));
  return maps;
}

void mlir::linalg::ReshapeOp::build(
    Builder *b, OperationState &result, Value view,
    ArrayRef<ArrayRef<AffineExpr>> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  auto memRefType = view.getType().cast<MemRefType>();
  auto resultType = computeReshapeCollapsedType(memRefType, maps);
  build(b, result, resultType, view, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      b->getAffineMapArrayAttr(maps));
}

void mlir::linalg::ReshapeOp::build(
    Builder *b, OperationState &result, Type resultType, Value view,
    ArrayRef<ArrayRef<AffineExpr>> reassociation,
    ArrayRef<NamedAttribute> attrs) {
  auto maps = getSymbolLessAffineMaps(reassociation);
  build(b, result, resultType, view, attrs);
  result.addAttribute(ReshapeOp::getReassociationAttrName(),
                      b->getAffineMapArrayAttr(maps));
}

static LogicalResult verify(ReshapeOp op) {
  MemRefType expandedType = op.getViewType();
  MemRefType collapsedType = op.getResult().getType().cast<MemRefType>();
  unsigned expandedRank = expandedType.getRank();
  unsigned collapsedRank = collapsedType.getRank();
  bool isCollapse = expandedRank > collapsedRank;
  if (!isCollapse) {
    std::swap(expandedRank, collapsedRank);
    std::swap(expandedType, collapsedType);
  }
  if (expandedRank == 0 || collapsedRank == 0)
    return op.emitOpError("expected non-zero memref ranks");
  if (expandedRank == collapsedRank)
    return op.emitOpError("expected to collapse or expand dims");

  if (collapsedRank != op.reassociation().size())
    return op.emitOpError("expected rank of the collapsed view(")
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
  MemRefType expectedType = computeReshapeCollapsedType(expandedType, maps);
  if (collapsedType != expectedType)
    return op.emitOpError("expected collapsed type to be ")
           << expectedType << ", but got " << collapsedType;
  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//
void mlir::linalg::SliceOp::build(Builder *b, OperationState &result,
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
                           strides, offset, b->getContext()))});
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

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
void mlir::linalg::TransposeOp::build(Builder *b, OperationState &result,
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
  auto map = makeStridedLinearLayoutMap(strides, offset, b->getContext());
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

template <typename GenericOpType>
static LogicalResult verifyYield(YieldOp op, GenericOpType genericOp) {
  // The operand number and types must match the view element types.
  auto nOutputs = genericOp.getNumOutputs();
  if (op.getNumOperands() != nOutputs)
    return op.emitOpError("expected number of yield values (")
           << nOutputs << ") to match the number of operands of the enclosing "
           << "linalg.generic op (" << op.getNumOperands() << ")";

  for (unsigned i = 0; i != nOutputs; ++i) {
    auto elementType = genericOp.getOutputShapedType(i).getElementType();
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

  auto genericOp = dyn_cast<GenericOp>(parentOp);
  if (genericOp)
    return verifyYield(op, genericOp);

  auto indexedGenericOp = dyn_cast<IndexedGenericOp>(parentOp);
  if (indexedGenericOp)
    return verifyYield(op, indexedGenericOp);

  return op.emitOpError("expected '")
         << GenericOp::getOperationName() << "' or '"
         << IndexedGenericOp::getOperationName() << "' parent op";
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

static LogicalResult
verifyStrideOrDilation(ConvOp op, ArrayRef<Attribute> attrs, bool isStride) {
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

static AffineMap extractOrIdentityMap(Optional<AffineMap> maybeMap,
                                      unsigned rank, MLIRContext *context) {
  if (maybeMap)
    return maybeMap.getValue();
  if (rank == 0)
    return AffineMap();
  return AffineMap::getMultiDimIdentityMap(rank, context);
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

// Returns `num` AffineDimExpr dimensions at positions [curIdx, curIdx + num)
// and increments `curIdx` to `curIdx + num`.
static SmallVector<AffineExpr, 4>
makeAffineDimExprs(unsigned num, unsigned &curIdx, MLIRContext *context) {
  SmallVector<AffineExpr, 4> res;
  res.reserve(num);
  for (unsigned i = 0; i < num; ++i)
    res.push_back(getAffineDimExpr(curIdx++, context));
  return res;
}

static SmallVector<AffineExpr, 4>
weightedConvInputIndex(ConvOp op, ArrayRef<AffineExpr> a,
                       ArrayRef<AffineExpr> b) {
  assert(a.size() == b.size());
  SmallVector<AffineExpr, 4> res;
  res.reserve(a.size());
  for (unsigned i = 0, e = a.size(); i < e; ++i) {
    res.push_back(op.getStride(i) * a[i] + op.getDilation(i) * b[i]);
  }
  return res;
}

static SmallVector<AffineExpr, 4> concat(ArrayRef<AffineExpr> a,
                                         ArrayRef<AffineExpr> b) {
  SmallVector<AffineExpr, 4> res;
  res.reserve(a.size() + b.size());
  res.assign(a.begin(), a.end());
  res.append(b.begin(), b.end());
  return res;
}

// Note: both functions below would completely disappear with a simple tensor
// kernel language.
//
// Ideally this should all be Tablegen'd but there is no good story for
// AffineMap for now.
SmallVector<AffineMap, 4> mlir::linalg::loopToOperandRangesMaps(Operation *op) {
  MLIRContext *context = op->getContext();
  if (auto copyOp = dyn_cast<CopyOp>(op)) {
    // I(input_perm(ivs)) -> O(output_perm(ivs))
    auto maybeInputMap = copyOp.inputPermutation();
    auto maybeOutputMap = copyOp.outputPermutation();
    unsigned inputRank = copyOp.getInputShapedType(0).getRank();
    unsigned outputRank = copyOp.getOutputShapedType(0).getRank();
    return SmallVector<AffineMap, 4>{
        extractOrIdentityMap(maybeInputMap, inputRank, context),
        extractOrIdentityMap(maybeOutputMap, outputRank, context)};
  }
  if (auto fillOp = dyn_cast<FillOp>(op)) {
    // filling_value -> O(ivs)
    unsigned rank = fillOp.getNumParallelLoops();
    return SmallVector<AffineMap, 4>{
        extractOrIdentityMap(llvm::None, rank, context)};
  }
  auto i = getAffineDimExpr(0, context);
  auto j = getAffineDimExpr(1, context);
  auto k = getAffineDimExpr(2, context);
  if (isa<DotOp>(op))
    // A(r_i) * B(r_i) -> C()
    return SmallVector<AffineMap, 4>{AffineMap::get(1, 0, {i}),
                                     AffineMap::get(1, 0, {i}), AffineMap()};
  if (isa<MatvecOp>(op))
    //   A(i, r_j) * B(r_j) -> C(i)
    return SmallVector<AffineMap, 4>{AffineMap::get(2, 0, {i, j}),
                                     AffineMap::get(2, 0, {j}),
                                     AffineMap::get(2, 0, {i})};
  if (isa<MatmulOp>(op))
    //   A(i, r_k) * B(r_k, j) -> C(i, j)
    return SmallVector<AffineMap, 4>{AffineMap::get(3, 0, {i, k}),
                                     AffineMap::get(3, 0, {k, j}),
                                     AffineMap::get(3, 0, {i, j})};
  if (auto convOp = dyn_cast<ConvOp>(op)) {
    //   F(z0, ..., zN-1, q, k) * I(b, x0 + z0, ..., xN-1 + zN-1, q) ->
    //     O(b, x0, ..., xN-1, k)
    // for N equal to `nWindow`.
    auto nWin = convOp.getNumWindowLoops();
    assert(nWin > 0 && "expected at least one window dimension");
    unsigned idx = 0;
    // In the following, AffineDimExprs are indexed in loop order:
    //   [ b, xs, k,           q,                     zs]
    //    parallels     non-window reductions     windows
    //
    // Parallel dims are exactly the dimensions indexing `output`:
    //     output[b, x[0], ..., x[N-1], k]; i.e.
    //  * batch dimensions (bs with #bs = 1 for now)
    //  * "image" dimensions (xs with #xs = #zs = output_rank - #bs - #ks)
    //  * output filter dimensions (ks with #ks = 1 for now)
    auto bs = makeAffineDimExprs(convOp.getNumBatchDimensions(), idx, context);
    auto xs = makeAffineDimExprs(nWin, idx, context);
    auto ks = makeAffineDimExprs(convOp.getNumOutputFeatureDimensions(), idx,
                                 context);
    // Non-window reduction dim: sum_{z[0], ..., z[N-1], q}
    auto qs =
        makeAffineDimExprs(convOp.getNumInputFeatureDimensions(), idx, context);
    // Window reduction dims: sum_{z[0], ..., z[N-1], q}
    auto zs = makeAffineDimExprs(nWin, idx, context);
    // Construct the weighedSum expression.
    auto ws = weightedConvInputIndex(convOp, xs, zs);
    return SmallVector<AffineMap, 4>{
        // filter[z[0], ..., z[N-1], q, k]
        AffineMap::get(idx, 0, concat(concat(zs, qs), ks)),
        // input[b,
        //       x[0]*s[0] + d[0]*z[0], ..., x[N-1]*s[N-1] + d[N-1]*z[N-1],
        //       q]
        AffineMap::get(idx, 0, concat(concat(bs, ws), qs)),
        // output[b, x[0], ..., x[N-1], k]
        AffineMap::get(idx, 0, concat(concat(bs, xs), ks))};
  }
  SmallVector<AffineMap, 4> res;
  auto linalgOp = cast<LinalgOp>(op);
  unsigned nViews = linalgOp.getNumInputsAndOutputs();
  res.reserve(nViews);
  for (unsigned i = 0, e = nViews; i < e; ++i)
    res.push_back(linalgOp.getIndexingMap(i));
  assert(nViews == linalgOp.indexing_maps().size());
  return res;
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
    interleave(
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
  interleave(
      types.begin(), types.end(), [&](Type t) { appendMangledType(ss, t); },
      [&]() { ss << "_"; });
  return ss.str();
}

static ArrayAttr getIndexingMaps(Operation *op) {
  LinalgOp linalgOp = cast<LinalgOp>(op);
  SmallVector<Attribute, 4> maps;
  maps.reserve(linalgOp.getNumInputsAndOutputs());
  for (AffineMap map : loopToOperandRangesMaps(op))
    maps.push_back(AffineMapAttr::get(map));
  return ArrayAttr::get(maps, op->getContext());
}
ArrayAttr mlir::linalg::ConvOp::indexing_maps() {
  return getIndexingMaps(getOperation());
}
ArrayAttr mlir::linalg::CopyOp::indexing_maps() {
  return getIndexingMaps(getOperation());
}
ArrayAttr mlir::linalg::DotOp::indexing_maps() {
  return getIndexingMaps(getOperation());
}
ArrayAttr mlir::linalg::FillOp::indexing_maps() {
  return getIndexingMaps(getOperation());
}
ArrayAttr mlir::linalg::MatmulOp::indexing_maps() {
  return getIndexingMaps(getOperation());
}
ArrayAttr mlir::linalg::MatvecOp::indexing_maps() {
  return getIndexingMaps(getOperation());
}

// TODO(ntv, rriddle): Consider making all this boilerplate easy to autogenerate
// with Tablegen. This seems a desirable property in the context of OpInterfaces
// where a Linalg "named" op **isa** LinalgOp.
LogicalResult ConvOp::fold(ArrayRef<Attribute>,
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
  return {};
}
OpFoldResult SliceOp::fold(ArrayRef<Attribute>) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}
OpFoldResult TransposeOp::fold(ArrayRef<Attribute>) {
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}
