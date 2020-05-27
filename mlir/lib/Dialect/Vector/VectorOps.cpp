//===- VectorOps.cpp - MLIR Vector Dialect Operations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements convenience types for working with super-vectorization
// operations, in particular super-vector loads and stores.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/StringSet.h"
#include <numeric>

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
// VectorDialect
//===----------------------------------------------------------------------===//

VectorDialect::VectorDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Vector/VectorOps.cpp.inc"
      >();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *VectorDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return builder.create<ConstantOp>(loc, type, value);
}

IntegerType vector::getVectorSubscriptType(Builder &builder) {
  return builder.getIntegerType(64);
}

ArrayAttr vector::getVectorSubscriptAttr(Builder &builder,
                                         ArrayRef<int64_t> values) {
  return builder.getI64ArrayAttr(values);
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReductionOp op) {
  // Verify for 1-D vector.
  int64_t rank = op.getVectorType().getRank();
  if (rank != 1)
    return op.emitOpError("unsupported reduction rank: ") << rank;

  // Verify supported reduction kind.
  auto kind = op.kind();
  Type eltType = op.dest().getType();
  if (kind == "add" || kind == "mul" || kind == "min" || kind == "max") {
    if (!eltType.isF32() && !eltType.isF64() &&
        !eltType.isSignlessInteger(32) && !eltType.isSignlessInteger(64))
      return op.emitOpError("unsupported reduction type");
  } else if (kind == "and" || kind == "or" || kind == "xor") {
    if (!eltType.isSignlessInteger(32) && !eltType.isSignlessInteger(64))
      return op.emitOpError("unsupported reduction type");
  } else {
    return op.emitOpError("unknown reduction kind: ") << kind;
  }

  // Verify optional accumulator.
  if (!op.acc().empty()) {
    if (kind != "add" && kind != "mul")
      return op.emitOpError("no accumulator for reduction kind: ") << kind;
    if (!eltType.isF32() && !eltType.isF64())
      return op.emitOpError("no accumulator for type: ") << eltType;
  }

  return success();
}

static ParseResult parseReductionOp(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> operandsInfo;
  Type redType;
  Type resType;
  Attribute attr;
  if (parser.parseAttribute(attr, "kind", result.attributes) ||
      parser.parseComma() || parser.parseOperandList(operandsInfo) ||
      parser.parseColonType(redType) ||
      parser.parseKeywordType("into", resType) ||
      (operandsInfo.size() > 0 &&
       parser.resolveOperand(operandsInfo[0], redType, result.operands)) ||
      (operandsInfo.size() > 1 &&
       parser.resolveOperand(operandsInfo[1], resType, result.operands)) ||
      parser.addTypeToList(resType, result.types))
    return failure();
  if (operandsInfo.size() < 1 || operandsInfo.size() > 2)
    return parser.emitError(parser.getNameLoc(),
                            "unsupported number of operands");
  return success();
}

static void print(OpAsmPrinter &p, ReductionOp op) {
  p << op.getOperationName() << " \"" << op.kind() << "\", " << op.vector();
  if (!op.acc().empty())
    p << ", " << op.acc();
  p << " : " << op.vector().getType() << " into " << op.dest().getType();
}

//===----------------------------------------------------------------------===//
// ContractionOp
//===----------------------------------------------------------------------===//

void vector::ContractionOp::build(OpBuilder &builder, OperationState &result,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayRef<ArrayRef<AffineExpr>> indexingExprs,
                                  ArrayRef<StringRef> iteratorTypes) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(getIndexingMapsAttrName(),
                      builder.getAffineMapArrayAttr(
                          AffineMap::inferFromExprList(indexingExprs)));
  result.addAttribute(getIteratorTypesAttrName(),
                      builder.getStrArrayAttr(iteratorTypes));
}

void vector::ContractionOp::build(OpBuilder &builder, OperationState &result,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayAttr indexingMaps,
                                  ArrayAttr iteratorTypes) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(getIndexingMapsAttrName(), indexingMaps);
  result.addAttribute(getIteratorTypesAttrName(), iteratorTypes);
}

static ParseResult parseContractionOp(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::OperandType lhsInfo;
  OpAsmParser::OperandType rhsInfo;
  OpAsmParser::OperandType accInfo;
  SmallVector<OpAsmParser::OperandType, 2> masksInfo;
  SmallVector<Type, 2> types;
  Type resultType;
  auto loc = parser.getCurrentLocation();
  DictionaryAttr dictAttr;
  // TODO(andydavis, ntv) Unify linalg op attribute parsing.
  if (parser.parseAttribute(dictAttr, "_", result.attributes) ||
      parser.parseOperand(lhsInfo) || parser.parseComma() ||
      parser.parseOperand(rhsInfo) || parser.parseComma() ||
      parser.parseOperand(accInfo) ||
      parser.parseTrailingOperandList(masksInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.parseKeywordType("into", resultType) ||
      parser.resolveOperand(lhsInfo, types[0], result.operands) ||
      parser.resolveOperand(rhsInfo, types[1], result.operands) ||
      parser.resolveOperand(accInfo, resultType, result.operands) ||
      parser.addTypeToList(resultType, result.types))
    return failure();
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());
  if (masksInfo.empty())
    return success();
  if (masksInfo.size() != 2)
    return parser.emitError(parser.getNameLoc(),
                            "expected zero or exactly 2 vector mask operands");
  auto lhsType = types[0].cast<VectorType>();
  auto rhsType = types[1].cast<VectorType>();
  auto maskElementType = parser.getBuilder().getI1Type();
  SmallVector<Type, 2> maskTypes;
  maskTypes.push_back(VectorType::get(lhsType.getShape(), maskElementType));
  maskTypes.push_back(VectorType::get(rhsType.getShape(), maskElementType));
  if (parser.resolveOperands(masksInfo, maskTypes, loc, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, ContractionOp op) {
  // TODO(andydavis, ntv) Unify printing code with linalg ops.
  auto attrNames = op.getTraitAttrNames();
  llvm::StringSet<> traitAttrsSet;
  traitAttrsSet.insert(attrNames.begin(), attrNames.end());
  SmallVector<NamedAttribute, 8> attrs;
  for (auto attr : op.getAttrs())
    if (traitAttrsSet.count(attr.first.strref()) > 0)
      attrs.push_back(attr);

  auto dictAttr = DictionaryAttr::get(attrs, op.getContext());
  p << op.getOperationName() << " " << dictAttr << " " << op.lhs() << ", ";
  p << op.rhs() << ", " << op.acc();
  if (op.masks().size() == 2)
    p << ", " << op.masks();

  p.printOptionalAttrDict(op.getAttrs(), attrNames);
  p << " : " << op.lhs().getType() << ", " << op.rhs().getType() << " into "
    << op.getResultType();
}

static bool verifyDimMap(VectorType lhsType, VectorType rhsType,
                         const std::vector<std::pair<int64_t, int64_t>> &map) {
  for (auto &dimPair : map) {
    if (dimPair.first < 0 || dimPair.first >= lhsType.getRank() ||
        dimPair.second < 0 || dimPair.second >= rhsType.getRank() ||
        lhsType.getDimSize(dimPair.first) != rhsType.getDimSize(dimPair.second))
      return false;
  }
  return true;
}

static bool verifyOutputShape(
    VectorType lhsType, VectorType rhsType, Type accType, Type resType,
    const std::vector<std::pair<int64_t, int64_t>> &contractingDimMap,
    const std::vector<std::pair<int64_t, int64_t>> &batchDimMap) {
  DenseSet<int64_t> lhsContractingDimSet;
  DenseSet<int64_t> rhsContractingDimSet;
  for (auto &dimPair : contractingDimMap) {
    lhsContractingDimSet.insert(dimPair.first);
    rhsContractingDimSet.insert(dimPair.second);
  }
  DenseSet<int64_t> rhsBatchDimSet;
  for (auto &dimPair : batchDimMap)
    rhsBatchDimSet.insert(dimPair.second);

  // Add free and batch dimensions from 'lhsType' to 'expectedResultDims'.
  SmallVector<int64_t, 4> expectedResultDims;
  for (int64_t i = 0, e = lhsType.getRank(); i < e; ++i) {
    if (lhsContractingDimSet.count(i) > 0)
      continue;
    expectedResultDims.push_back(lhsType.getDimSize(i));
  }

  // Add free dimensions from 'rhsType' to 'expectedResultDims'.
  for (int64_t i = 0, e = rhsType.getRank(); i < e; ++i) {
    if (rhsContractingDimSet.count(i) > 0 || rhsBatchDimSet.count(i) > 0)
      continue;
    expectedResultDims.push_back(rhsType.getDimSize(i));
  }

  // Verify 'expectedResultDims'.
  if (expectedResultDims.size() == 0) {
    // No batch or free dimension implies a scalar result.
    if (resType.isa<VectorType>() || accType.isa<VectorType>())
      return false;

  } else {
    // At least one batch or free dimension implies a vector result.
    auto resVectorType = resType.dyn_cast<VectorType>();
    auto accVectorType = accType.dyn_cast<VectorType>();
    if (!resVectorType || !accVectorType)
      return false;

    // Verify dimension from 'resType' against 'expectedResultDims'.
    if (resVectorType.getShape().size() != expectedResultDims.size() ||
        accVectorType.getShape().size() != expectedResultDims.size())
      return false;
    for (int64_t i = 0, e = resVectorType.getRank(); i < e; ++i) {
      if (resVectorType.getDimSize(i) != expectedResultDims[i] ||
          accVectorType.getDimSize(i) != expectedResultDims[i])
        return false;
    }
  }
  return true;
}

static LogicalResult verify(ContractionOp op) {
  auto lhsType = op.getLhsType();
  auto rhsType = op.getRhsType();
  auto accType = op.getAccType();
  auto resType = op.getResultType();

  // Verify that an indexing map was specified for each vector operand.
  if (op.indexing_maps().size() != 3)
    return op.emitOpError("expected an indexing map for each vector operand");

  // Verify that each index map has 'numIterators' inputs, no symbols, and
  // that the number of map outputs equals the rank of its associated
  // vector operand.
  unsigned numIterators = op.iterator_types().getValue().size();
  for (auto it : llvm::enumerate(op.indexing_maps())) {
    auto index = it.index();
    auto map = it.value().cast<AffineMapAttr>().getValue();
    if (map.getNumSymbols() != 0)
      return op.emitOpError("expected indexing map ")
             << index << " to have no symbols";
    auto vectorType = op.getOperand(index).getType().dyn_cast<VectorType>();
    unsigned rank = vectorType ? vectorType.getShape().size() : 0;
    // Verify that the map has the right number of inputs, outputs, and indices.
    // This also correctly accounts for (..) -> () for rank-0 results.
    if (map.getNumDims() != numIterators)
      return op.emitOpError("expected indexing map ")
             << index << " to have " << numIterators << " number of inputs";
    if (map.getNumResults() != rank)
      return op.emitOpError("expected indexing map ")
             << index << " to have " << rank << " number of outputs";
    if (!map.isProjectedPermutation())
      return op.emitOpError("expected indexing map ")
             << index << " to be a projected permutation of its inputs";
  }

  auto contractingDimMap = op.getContractingDimMap();
  auto batchDimMap = op.getBatchDimMap();

  // Verify at least one contracting dimension pair was specified.
  if (contractingDimMap.empty())
    return op.emitOpError("expected at least one contracting dimension pair");

  // Verify contracting dimension map was properly constructed.
  if (!verifyDimMap(lhsType, rhsType, contractingDimMap))
    return op.emitOpError("invalid contracting dimension map");

  // Verify batch dimension map was properly constructed.
  if (!verifyDimMap(lhsType, rhsType, batchDimMap))
    return op.emitOpError("invalid batch dimension map");

  // Verify 'accType' and 'resType' shape.
  if (!verifyOutputShape(lhsType, rhsType, accType, resType, contractingDimMap,
                         batchDimMap))
    return op.emitOpError("invalid accumulator/result vector shape");

  // Verify that either two vector masks are set or none are set.
  auto lhsMaskType = op.getLHSVectorMaskType();
  auto rhsMaskType = op.getRHSVectorMaskType();
  if ((lhsMaskType && !rhsMaskType) || (!lhsMaskType && rhsMaskType))
    return op.emitOpError("invalid number of vector masks specified");
  if (lhsMaskType && rhsMaskType) {
    // Verify mask rank == argument rank.
    if (lhsMaskType.getShape().size() != lhsType.getShape().size() ||
        rhsMaskType.getShape().size() != rhsType.getShape().size())
      return op.emitOpError("invalid vector mask rank");
  }
  return success();
}

ArrayRef<StringRef> ContractionOp::getTraitAttrNames() {
  static constexpr StringRef names[2] = {getIndexingMapsAttrName(),
                                         getIteratorTypesAttrName()};
  return llvm::makeArrayRef(names);
}

static int64_t getResultIndex(AffineMap map, AffineExpr targetExpr) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i)
    if (targetExpr == map.getResult(i))
      return i;
  return -1;
}

static std::vector<std::pair<int64_t, int64_t>>
getDimMap(ArrayRef<AffineMap> indexingMaps, ArrayAttr iteratorTypes,
          StringRef targetIteratorTypeName, MLIRContext *context) {
  std::vector<std::pair<int64_t, int64_t>> dimMap;
  for (auto it : llvm::enumerate(iteratorTypes)) {
    auto iteratorTypeName = it.value().cast<StringAttr>().getValue();
    if (iteratorTypeName != targetIteratorTypeName)
      continue;
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), context);
    int64_t lhsDim = getResultIndex(indexingMaps[0], targetExpr);
    int64_t rhsDim = getResultIndex(indexingMaps[1], targetExpr);
    if (lhsDim >= 0 && rhsDim >= 0)
      dimMap.push_back({lhsDim, rhsDim});
  }
  return dimMap;
}

void ContractionOp::getIterationBounds(
    SmallVectorImpl<int64_t> &iterationBounds) {
  auto lhsShape = getLhsType().getShape();
  auto resVectorType = getResultType().dyn_cast<VectorType>();
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMaps());
  SmallVector<int64_t, 2> iterationShape;
  for (auto it : llvm::enumerate(iterator_types())) {
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), getContext());
    auto iteratorTypeName = it.value().cast<StringAttr>().getValue();
    if (iteratorTypeName == getReductionIteratorTypeName()) {
      // Get reduction dim size from lhs shape (same size in rhsShape).
      int64_t lhsDimIndex = getResultIndex(indexingMaps[0], targetExpr);
      assert(lhsDimIndex >= 0);
      iterationBounds.push_back(lhsShape[lhsDimIndex]);
      continue;
    }
    // Get parallel dimension size from result shape.
    int64_t resDimIndex = getResultIndex(indexingMaps[2], targetExpr);
    assert(resDimIndex >= 0);
    assert(resVectorType != nullptr);
    iterationBounds.push_back(resVectorType.getShape()[resDimIndex]);
  }
}

void ContractionOp::getIterationIndexMap(
    std::vector<DenseMap<int64_t, int64_t>> &iterationIndexMap) {
  unsigned numMaps = indexing_maps().getValue().size();
  iterationIndexMap.resize(numMaps);
  for (auto it : llvm::enumerate(indexing_maps())) {
    auto index = it.index();
    auto map = it.value().cast<AffineMapAttr>().getValue();
    for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
      auto dim = map.getResult(i).cast<AffineDimExpr>();
      iterationIndexMap[index][dim.getPosition()] = i;
    }
  }
}

std::vector<std::pair<int64_t, int64_t>> ContractionOp::getContractingDimMap() {
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMaps());
  return getDimMap(indexingMaps, iterator_types(),
                   getReductionIteratorTypeName(), getContext());
}

std::vector<std::pair<int64_t, int64_t>> ContractionOp::getBatchDimMap() {
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMaps());
  return getDimMap(indexingMaps, iterator_types(),
                   getParallelIteratorTypeName(), getContext());
}

SmallVector<AffineMap, 4> ContractionOp::getIndexingMaps() {
  SmallVector<AffineMap, 4> res;
  auto mapAttrs = indexing_maps().getValue();
  res.reserve(mapAttrs.size());
  for (auto mapAttr : mapAttrs)
    res.push_back(mapAttr.cast<AffineMapAttr>().getValue());
  return res;
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(vector::ExtractElementOp op) {
  VectorType vectorType = op.getVectorType();
  if (vectorType.getRank() != 1)
    return op.emitOpError("expected 1-D vector");
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

static Type inferExtractOpResultType(VectorType vectorType,
                                     ArrayAttr position) {
  if (static_cast<int64_t>(position.size()) == vectorType.getRank())
    return vectorType.getElementType();
  return VectorType::get(vectorType.getShape().drop_front(position.size()),
                         vectorType.getElementType());
}

void vector::ExtractOp::build(OpBuilder &builder, OperationState &result,
                              Value source, ArrayRef<int64_t> position) {
  result.addOperands(source);
  auto positionAttr = getVectorSubscriptAttr(builder, position);
  result.addTypes(inferExtractOpResultType(source.getType().cast<VectorType>(),
                                           positionAttr));
  result.addAttribute(getPositionAttrName(), positionAttr);
}

// Convenience builder which assumes the values are constant indices.
void vector::ExtractOp::build(OpBuilder &builder, OperationState &result,
                              Value source, ValueRange position) {
  SmallVector<int64_t, 4> positionConstants =
      llvm::to_vector<4>(llvm::map_range(position, [](Value pos) {
        return pos.getDefiningOp<ConstantIndexOp>().getValue();
      }));
  build(builder, result, source, positionConstants);
}

static void print(OpAsmPrinter &p, vector::ExtractOp op) {
  p << op.getOperationName() << " " << op.vector() << op.position();
  p.printOptionalAttrDict(op.getAttrs(), {"position"});
  p << " : " << op.vector().getType();
}

static ParseResult parseExtractOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc attributeLoc, typeLoc;
  NamedAttrList attrs;
  OpAsmParser::OperandType vector;
  Type type;
  Attribute attr;
  if (parser.parseOperand(vector) || parser.getCurrentLocation(&attributeLoc) ||
      parser.parseAttribute(attr, "position", attrs) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.getCurrentLocation(&typeLoc) || parser.parseColonType(type))
    return failure();

  auto vectorType = type.dyn_cast<VectorType>();
  if (!vectorType)
    return parser.emitError(typeLoc, "expected vector type");

  auto positionAttr = attr.dyn_cast<ArrayAttr>();
  if (!positionAttr ||
      static_cast<int64_t>(positionAttr.size()) > vectorType.getRank())
    return parser.emitError(
        attributeLoc,
        "expected position attribute of rank smaller than vector rank");

  Type resType = inferExtractOpResultType(vectorType, positionAttr);
  result.attributes = attrs;
  return failure(parser.resolveOperand(vector, type, result.operands) ||
                 parser.addTypeToList(resType, result.types));
}

static LogicalResult verify(vector::ExtractOp op) {
  auto positionAttr = op.position().getValue();
  if (positionAttr.empty())
    return op.emitOpError("expected non-empty position attribute");
  if (positionAttr.size() > static_cast<unsigned>(op.getVectorType().getRank()))
    return op.emitOpError(
        "expected position attribute of rank smaller than vector rank");
  for (auto en : llvm::enumerate(positionAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 ||
        attr.getInt() >= op.getVectorType().getDimSize(en.index()))
      return op.emitOpError("expected position attribute #")
             << (en.index() + 1)
             << " to be a non-negative integer smaller than the corresponding "
                "vector dimension";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractSlicesOp
//===----------------------------------------------------------------------===//

void ExtractSlicesOp::build(OpBuilder &builder, OperationState &result,
                            TupleType tupleType, Value vector,
                            ArrayRef<int64_t> sizes,
                            ArrayRef<int64_t> strides) {
  result.addOperands(vector);
  auto sizesAttr = getVectorSubscriptAttr(builder, sizes);
  auto stridesAttr = getVectorSubscriptAttr(builder, strides);
  result.addTypes(tupleType);
  result.addAttribute(getSizesAttrName(), sizesAttr);
  result.addAttribute(getStridesAttrName(), stridesAttr);
}

static LogicalResult
isValidExtractOrInsertSlicesType(Operation *op, VectorType vectorType,
                                 TupleType tupleType, ArrayRef<int64_t> sizes,
                                 ArrayRef<int64_t> strides) {
  // Check for non-unit strides.
  // TODO(b/144845578) Support non-1 strides.
  if (llvm::any_of(strides, [](int64_t s) { return s != 1; }))
    return op->emitError("requires unit strides");
  // Check that 'vectorType' rank matches rank of tuple element vectors.
  unsigned rank = vectorType.getRank();
  auto is_vector_type_of_rank = [&](Type t) {
    return t.isa<VectorType>() && t.cast<VectorType>().getRank() == rank;
  };
  if (!llvm::all_of(tupleType.getTypes(), is_vector_type_of_rank))
    return op->emitError("requires vector tuple elements of rank ") << rank;
  // Check that 'sizes' and 'strides' are of size == 'rank'.
  if (sizes.size() != rank || strides.size() != rank)
    return op->emitError("requires sizes and strides of rank ") << rank;

  // Generate each slice shape based on 'sizes', 'strides' and 'vectorType',
  // and verify that the same matches the corresponding tuple element 'i'.
  auto shape = vectorType.getShape();
  auto sliceStrides = computeStrides(shape, sizes);
  for (int64_t i = 0, e = tupleType.size(); i < e; ++i) {
    auto vectorOffsets = delinearize(sliceStrides, i);
    auto elementOffsets =
        computeElementOffsetsFromVectorSliceOffsets(sizes, vectorOffsets);
    auto sliceSizes = computeSliceSizes(shape, sizes, elementOffsets);
    // Create slice VectorType type.
    auto sliceVectorType =
        VectorType::get(sliceSizes, vectorType.getElementType());
    // Verify that 'sliceVectorType' matches tupleType.getTypes(i)
    if (sliceVectorType != tupleType.getType(i))
      return op->emitError("invalid tuple element type ") << sliceVectorType;
  }
  return success();
}

static LogicalResult verify(ExtractSlicesOp op) {
  SmallVector<int64_t, 4> sizes;
  op.getSizes(sizes);
  SmallVector<int64_t, 4> strides;
  op.getStrides(strides);
  return isValidExtractOrInsertSlicesType(
      op.getOperation(), op.getSourceVectorType(), op.getResultTupleType(),
      sizes, strides);
}

static void populateFromInt64AttrArray(ArrayAttr arrayAttr,
                                       SmallVectorImpl<int64_t> &results) {
  for (auto attr : arrayAttr)
    results.push_back(attr.cast<IntegerAttr>().getInt());
}

void ExtractSlicesOp::getSizes(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(sizes(), results);
}

void ExtractSlicesOp::getStrides(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(strides(), results);
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(BroadcastOp op) {
  VectorType srcVectorType = op.getSourceType().dyn_cast<VectorType>();
  VectorType dstVectorType = op.getVectorType();
  // Scalar to vector broadcast is always valid. A vector
  // to vector broadcast needs some additional checking.
  if (srcVectorType) {
    int64_t srcRank = srcVectorType.getRank();
    int64_t dstRank = dstVectorType.getRank();
    if (srcRank > dstRank)
      return op.emitOpError("source rank higher than destination rank");
    // Source has an exact match or singleton value for all trailing dimensions
    // (all leading dimensions are simply duplicated).
    int64_t lead = dstRank - srcRank;
    for (int64_t r = 0; r < srcRank; ++r) {
      int64_t srcDim = srcVectorType.getDimSize(r);
      int64_t dstDim = dstVectorType.getDimSize(lead + r);
      if (srcDim != 1 && srcDim != dstDim)
        return op.emitOpError("dimension mismatch (")
               << srcDim << " vs. " << dstDim << ")";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

void ShuffleOp::build(OpBuilder &builder, OperationState &result, Value v1,
                      Value v2, ArrayRef<int64_t> mask) {
  result.addOperands({v1, v2});
  auto maskAttr = getVectorSubscriptAttr(builder, mask);
  result.addTypes(v1.getType());
  result.addAttribute(getMaskAttrName(), maskAttr);
}

static void print(OpAsmPrinter &p, ShuffleOp op) {
  p << op.getOperationName() << " " << op.v1() << ", " << op.v2() << " "
    << op.mask();
  p.printOptionalAttrDict(op.getAttrs(), {ShuffleOp::getMaskAttrName()});
  p << " : " << op.v1().getType() << ", " << op.v2().getType();
}

static LogicalResult verify(ShuffleOp op) {
  VectorType resultType = op.getVectorType();
  VectorType v1Type = op.getV1VectorType();
  VectorType v2Type = op.getV2VectorType();
  // Verify ranks.
  int64_t resRank = resultType.getRank();
  int64_t v1Rank = v1Type.getRank();
  int64_t v2Rank = v2Type.getRank();
  if (resRank != v1Rank || v1Rank != v2Rank)
    return op.emitOpError("rank mismatch");
  // Verify all but leading dimension sizes.
  for (int64_t r = 1; r < v1Rank; ++r) {
    int64_t resDim = resultType.getDimSize(r);
    int64_t v1Dim = v1Type.getDimSize(r);
    int64_t v2Dim = v2Type.getDimSize(r);
    if (resDim != v1Dim || v1Dim != v2Dim)
      return op.emitOpError("dimension mismatch");
  }
  // Verify mask length.
  auto maskAttr = op.mask().getValue();
  int64_t maskLength = maskAttr.size();
  if (maskLength != resultType.getDimSize(0))
    return op.emitOpError("mask length mismatch");
  // Verify all indices.
  int64_t indexSize = v1Type.getDimSize(0) + v2Type.getDimSize(0);
  for (auto en : llvm::enumerate(maskAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 || attr.getInt() >= indexSize)
      return op.emitOpError("mask index #")
             << (en.index() + 1) << " out of range";
  }
  return success();
}

static ParseResult parseShuffleOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType v1, v2;
  Attribute attr;
  VectorType v1Type, v2Type;
  if (parser.parseOperand(v1) || parser.parseComma() ||
      parser.parseOperand(v2) ||
      parser.parseAttribute(attr, ShuffleOp::getMaskAttrName(),
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(v1Type) || parser.parseComma() ||
      parser.parseType(v2Type) ||
      parser.resolveOperand(v1, v1Type, result.operands) ||
      parser.resolveOperand(v2, v2Type, result.operands))
    return failure();
  // Construct resulting type: leading dimension matches mask length,
  // all trailing dimensions match the operands.
  auto maskAttr = attr.dyn_cast<ArrayAttr>();
  if (!maskAttr)
    return parser.emitError(parser.getNameLoc(), "missing mask attribute");
  int64_t maskLength = maskAttr.size();
  if (maskLength <= 0)
    return parser.emitError(parser.getNameLoc(), "invalid mask length");
  int64_t v1Rank = v1Type.getRank();
  SmallVector<int64_t, 4> shape;
  shape.reserve(v1Rank);
  shape.push_back(maskLength);
  for (int64_t r = 1; r < v1Rank; ++r)
    shape.push_back(v1Type.getDimSize(r));
  VectorType resType = VectorType::get(shape, v1Type.getElementType());
  parser.addTypeToList(resType, result.types);
  return success();
}

//===----------------------------------------------------------------------===//
// InsertElementOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(InsertElementOp op) {
  auto dstVectorType = op.getDestVectorType();
  if (dstVectorType.getRank() != 1)
    return op.emitOpError("expected 1-D vector");
  return success();
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

void InsertOp::build(OpBuilder &builder, OperationState &result, Value source,
                     Value dest, ArrayRef<int64_t> position) {
  result.addOperands({source, dest});
  auto positionAttr = getVectorSubscriptAttr(builder, position);
  result.addTypes(dest.getType());
  result.addAttribute(getPositionAttrName(), positionAttr);
}

// Convenience builder which assumes the values are constant indices.
void InsertOp::build(OpBuilder &builder, OperationState &result, Value source,
                     Value dest, ValueRange position) {
  SmallVector<int64_t, 4> positionConstants =
      llvm::to_vector<4>(llvm::map_range(position, [](Value pos) {
        return pos.getDefiningOp<ConstantIndexOp>().getValue();
      }));
  build(builder, result, source, dest, positionConstants);
}

static LogicalResult verify(InsertOp op) {
  auto positionAttr = op.position().getValue();
  if (positionAttr.empty())
    return op.emitOpError("expected non-empty position attribute");
  auto destVectorType = op.getDestVectorType();
  if (positionAttr.size() > static_cast<unsigned>(destVectorType.getRank()))
    return op.emitOpError(
        "expected position attribute of rank smaller than dest vector rank");
  auto srcVectorType = op.getSourceType().dyn_cast<VectorType>();
  if (srcVectorType &&
      (static_cast<unsigned>(srcVectorType.getRank()) + positionAttr.size() !=
       static_cast<unsigned>(destVectorType.getRank())))
    return op.emitOpError("expected position attribute rank + source rank to "
                          "match dest vector rank");
  else if (!srcVectorType && (positionAttr.size() !=
                              static_cast<unsigned>(destVectorType.getRank())))
    return op.emitOpError(
        "expected position attribute rank to match the dest vector rank");
  for (auto en : llvm::enumerate(positionAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (!attr || attr.getInt() < 0 ||
        attr.getInt() >= destVectorType.getDimSize(en.index()))
      return op.emitOpError("expected position attribute #")
             << (en.index() + 1)
             << " to be a non-negative integer smaller than the corresponding "
                "dest vector dimension";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InsertSlicesOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(InsertSlicesOp op) {
  SmallVector<int64_t, 4> sizes;
  op.getSizes(sizes);
  SmallVector<int64_t, 4> strides;
  op.getStrides(strides);
  return isValidExtractOrInsertSlicesType(
      op.getOperation(), op.getResultVectorType(), op.getSourceTupleType(),
      sizes, strides);
}

void InsertSlicesOp::getSizes(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(sizes(), results);
}

void InsertSlicesOp::getStrides(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(strides(), results);
}

//===----------------------------------------------------------------------===//
// InsertStridedSliceOp
//===----------------------------------------------------------------------===//

void InsertStridedSliceOp::build(OpBuilder &builder, OperationState &result,
                                 Value source, Value dest,
                                 ArrayRef<int64_t> offsets,
                                 ArrayRef<int64_t> strides) {
  result.addOperands({source, dest});
  auto offsetsAttr = getVectorSubscriptAttr(builder, offsets);
  auto stridesAttr = getVectorSubscriptAttr(builder, strides);
  result.addTypes(dest.getType());
  result.addAttribute(getOffsetsAttrName(), offsetsAttr);
  result.addAttribute(getStridesAttrName(), stridesAttr);
}

// TODO(ntv) Should be moved to Tablegen Confined attributes.
template <typename OpType>
static LogicalResult isIntegerArrayAttrSmallerThanShape(OpType op,
                                                        ArrayAttr arrayAttr,
                                                        ArrayRef<int64_t> shape,
                                                        StringRef attrName) {
  if (arrayAttr.size() > shape.size())
    return op.emitOpError("expected ")
           << attrName << " attribute of rank smaller than vector rank";
  return success();
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult
isIntegerArrayAttrConfinedToRange(OpType op, ArrayAttr arrayAttr, int64_t min,
                                  int64_t max, StringRef attrName,
                                  bool halfOpen = true) {
  for (auto attr : arrayAttr) {
    auto val = attr.cast<IntegerAttr>().getInt();
    auto upper = max;
    if (!halfOpen)
      upper += 1;
    if (val < min || val >= upper)
      return op.emitOpError("expected ") << attrName << " to be confined to ["
                                         << min << ", " << upper << ")";
  }
  return success();
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult
isIntegerArrayAttrConfinedToShape(OpType op, ArrayAttr arrayAttr,
                                  ArrayRef<int64_t> shape, StringRef attrName,
                                  bool halfOpen = true, int64_t min = 0) {
  assert(arrayAttr.size() <= shape.size());
  unsigned index = 0;
  for (auto it : llvm::zip(arrayAttr, shape)) {
    auto val = std::get<0>(it).cast<IntegerAttr>().getInt();
    auto max = std::get<1>(it);
    if (!halfOpen)
      max += 1;
    if (val < min || val >= max)
      return op.emitOpError("expected ")
             << attrName << " dimension " << index << " to be confined to ["
             << min << ", " << max << ")";
    ++index;
  }
  return success();
}

// Returns true if all integers in `arrayAttr` are in the interval [min, max}.
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult isSumOfIntegerArrayAttrConfinedToShape(
    OpType op, ArrayAttr arrayAttr1, ArrayAttr arrayAttr2,
    ArrayRef<int64_t> shape, StringRef attrName1, StringRef attrName2,
    bool halfOpen = true, int64_t min = 1) {
  assert(arrayAttr1.size() <= shape.size());
  assert(arrayAttr2.size() <= shape.size());
  unsigned index = 0;
  for (auto it : llvm::zip(arrayAttr1, arrayAttr2, shape)) {
    auto val1 = std::get<0>(it).cast<IntegerAttr>().getInt();
    auto val2 = std::get<1>(it).cast<IntegerAttr>().getInt();
    auto max = std::get<2>(it);
    if (!halfOpen)
      max += 1;
    if (val1 + val2 < 0 || val1 + val2 >= max)
      return op.emitOpError("expected sum(")
             << attrName1 << ", " << attrName2 << ") dimension " << index
             << " to be confined to [" << min << ", " << max << ")";
    ++index;
  }
  return success();
}

static ArrayAttr makeI64ArrayAttr(ArrayRef<int64_t> values,
                                  MLIRContext *context) {
  auto attrs = llvm::map_range(values, [context](int64_t v) -> Attribute {
    return IntegerAttr::get(IntegerType::get(64, context), APInt(64, v));
  });
  return ArrayAttr::get(llvm::to_vector<8>(attrs), context);
}

static LogicalResult verify(InsertStridedSliceOp op) {
  auto sourceVectorType = op.getSourceVectorType();
  auto destVectorType = op.getDestVectorType();
  auto offsets = op.offsets();
  auto strides = op.strides();
  if (offsets.size() != static_cast<unsigned>(destVectorType.getRank()))
    return op.emitOpError(
        "expected offsets of same size as destination vector rank");
  if (strides.size() != static_cast<unsigned>(sourceVectorType.getRank()))
    return op.emitOpError(
        "expected strides of same size as source vector rank");
  if (sourceVectorType.getRank() > destVectorType.getRank())
    return op.emitOpError(
        "expected source rank to be smaller than destination rank");

  auto sourceShape = sourceVectorType.getShape();
  auto destShape = destVectorType.getShape();
  SmallVector<int64_t, 4> sourceShapeAsDestShape(
      destShape.size() - sourceShape.size(), 0);
  sourceShapeAsDestShape.append(sourceShape.begin(), sourceShape.end());
  auto offName = InsertStridedSliceOp::getOffsetsAttrName();
  auto stridesName = InsertStridedSliceOp::getStridesAttrName();
  if (failed(
          isIntegerArrayAttrConfinedToShape(op, offsets, destShape, offName)) ||
      failed(isIntegerArrayAttrConfinedToRange(op, strides, 1, 1, stridesName,
                                               /*halfOpen=*/false)) ||
      failed(isSumOfIntegerArrayAttrConfinedToShape(
          op, offsets,
          makeI64ArrayAttr(sourceShapeAsDestShape, op.getContext()), destShape,
          offName, "source vector shape",
          /*halfOpen=*/false, /*min=*/1)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// OuterProductOp
//===----------------------------------------------------------------------===//

/// Build an op without mask, use the type of `acc` as the return type.
void OuterProductOp::build(OpBuilder &builder, OperationState &result,
                           Value lhs, Value rhs, Value acc) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
}

static void print(OpAsmPrinter &p, OuterProductOp op) {
  p << op.getOperationName() << " " << op.lhs() << ", " << op.rhs();
  if (!op.acc().empty())
    p << ", " << op.acc();
  p << " : " << op.lhs().getType() << ", " << op.rhs().getType();
}

static ParseResult parseOuterProductOp(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> operandsInfo;
  Type tLHS, tRHS;
  if (parser.parseOperandList(operandsInfo) || parser.parseColonType(tLHS) ||
      parser.parseComma() || parser.parseType(tRHS))
    return failure();
  if (operandsInfo.size() < 2)
    return parser.emitError(parser.getNameLoc(),
                            "expected at least 2 operands");
  VectorType vLHS = tLHS.dyn_cast<VectorType>();
  VectorType vRHS = tRHS.dyn_cast<VectorType>();
  if (!vLHS || !vRHS)
    return parser.emitError(parser.getNameLoc(), "expected 2 vector types");
  VectorType resType = VectorType::get({vLHS.getDimSize(0), vRHS.getDimSize(0)},
                                       vLHS.getElementType());
  return failure(
      parser.resolveOperand(operandsInfo[0], tLHS, result.operands) ||
      parser.resolveOperand(operandsInfo[1], tRHS, result.operands) ||
      (operandsInfo.size() > 2 &&
       parser.resolveOperand(operandsInfo[2], resType, result.operands)) ||
      parser.addTypeToList(resType, result.types));
}

static LogicalResult verify(OuterProductOp op) {
  VectorType vLHS = op.getOperandVectorTypeLHS(),
             vRHS = op.getOperandVectorTypeRHS(),
             vACC = op.getOperandVectorTypeACC(), vRES = op.getVectorType();
  if (vLHS.getRank() != 1)
    return op.emitOpError("expected 1-d vector for operand #1");
  if (vRHS.getRank() != 1)
    return op.emitOpError("expected 1-d vector for operand #2");
  if (vRES.getRank() != 2)
    return op.emitOpError("expected 2-d vector result");
  if (vLHS.getDimSize(0) != vRES.getDimSize(0))
    return op.emitOpError("expected #1 operand dim to match result dim #1");
  if (vRHS.getDimSize(0) != vRES.getDimSize(1))
    return op.emitOpError("expected #2 operand dim to match result dim #2");
  if (vACC && vACC != vRES)
    return op.emitOpError("expected operand #3 of same type as result type");
  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReshapeOp op) {
  // Verify that rank(numInputs/outputs) + numFixedVec dim matches vec rank.
  auto inputVectorType = op.getInputVectorType();
  auto outputVectorType = op.getOutputVectorType();
  int64_t inputShapeRank = op.getNumInputShapeSizes();
  int64_t outputShapeRank = op.getNumOutputShapeSizes();
  SmallVector<int64_t, 4> fixedVectorSizes;
  op.getFixedVectorSizes(fixedVectorSizes);
  int64_t numFixedVectorSizes = fixedVectorSizes.size();

  if (inputVectorType.getRank() != inputShapeRank + numFixedVectorSizes)
    return op.emitError("invalid input shape for vector type ")
           << inputVectorType;

  if (outputVectorType.getRank() != outputShapeRank + numFixedVectorSizes)
    return op.emitError("invalid output shape for vector type ")
           << outputVectorType;

  // Verify that the 'fixedVectorSizes' match an input/output vector shape
  // suffix.
  unsigned inputVectorRank = inputVectorType.getRank();
  for (unsigned i = 0; i < numFixedVectorSizes; ++i) {
    unsigned index = inputVectorRank - numFixedVectorSizes - i;
    if (fixedVectorSizes[i] != inputVectorType.getShape()[index])
      return op.emitError("fixed vector size must match input vector for dim ")
             << i;
  }

  unsigned outputVectorRank = outputVectorType.getRank();
  for (unsigned i = 0; i < numFixedVectorSizes; ++i) {
    unsigned index = outputVectorRank - numFixedVectorSizes - i;
    if (fixedVectorSizes[i] != outputVectorType.getShape()[index])
      return op.emitError("fixed vector size must match output vector for dim ")
             << i;
  }

  // If all shape operands are produced by constant ops, verify that product
  // of dimensions for input/output shape match.
  auto isDefByConstant = [](Value operand) {
    return isa_and_nonnull<ConstantIndexOp>(operand.getDefiningOp());
  };
  if (llvm::all_of(op.input_shape(), isDefByConstant) &&
      llvm::all_of(op.output_shape(), isDefByConstant)) {
    int64_t numInputElements = 1;
    for (auto operand : op.input_shape())
      numInputElements *=
          cast<ConstantIndexOp>(operand.getDefiningOp()).getValue();
    int64_t numOutputElements = 1;
    for (auto operand : op.output_shape())
      numOutputElements *=
          cast<ConstantIndexOp>(operand.getDefiningOp()).getValue();
    if (numInputElements != numOutputElements)
      return op.emitError("product of input and output shape sizes must match");
  }
  return success();
}

void ReshapeOp::getFixedVectorSizes(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(fixed_vector_sizes(), results);
}

//===----------------------------------------------------------------------===//
// ExtractStridedSliceOp
//===----------------------------------------------------------------------===//

// Inference works as follows:
//   1. Add 'sizes' from prefix of dims in 'offsets'.
//   2. Add sizes from 'vectorType' for remaining dims.
static Type inferStridedSliceOpResultType(VectorType vectorType,
                                          ArrayAttr offsets, ArrayAttr sizes,
                                          ArrayAttr strides) {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  SmallVector<int64_t, 4> shape;
  shape.reserve(vectorType.getRank());
  unsigned idx = 0;
  for (unsigned e = offsets.size(); idx < e; ++idx)
    shape.push_back(sizes[idx].cast<IntegerAttr>().getInt());
  for (unsigned e = vectorType.getShape().size(); idx < e; ++idx)
    shape.push_back(vectorType.getShape()[idx]);

  return VectorType::get(shape, vectorType.getElementType());
}

void ExtractStridedSliceOp::build(OpBuilder &builder, OperationState &result,
                                  Value source, ArrayRef<int64_t> offsets,
                                  ArrayRef<int64_t> sizes,
                                  ArrayRef<int64_t> strides) {
  result.addOperands(source);
  auto offsetsAttr = getVectorSubscriptAttr(builder, offsets);
  auto sizesAttr = getVectorSubscriptAttr(builder, sizes);
  auto stridesAttr = getVectorSubscriptAttr(builder, strides);
  result.addTypes(
      inferStridedSliceOpResultType(source.getType().cast<VectorType>(),
                                    offsetsAttr, sizesAttr, stridesAttr));
  result.addAttribute(getOffsetsAttrName(), offsetsAttr);
  result.addAttribute(getSizesAttrName(), sizesAttr);
  result.addAttribute(getStridesAttrName(), stridesAttr);
}

static LogicalResult verify(ExtractStridedSliceOp op) {
  auto type = op.getVectorType();
  auto offsets = op.offsets();
  auto sizes = op.sizes();
  auto strides = op.strides();
  if (offsets.size() != sizes.size() || offsets.size() != strides.size()) {
    op.emitOpError(
        "expected offsets, sizes and strides attributes of same size");
    return failure();
  }

  auto shape = type.getShape();
  auto offName = ExtractStridedSliceOp::getOffsetsAttrName();
  auto sizesName = ExtractStridedSliceOp::getSizesAttrName();
  auto stridesName = ExtractStridedSliceOp::getStridesAttrName();
  if (failed(isIntegerArrayAttrSmallerThanShape(op, offsets, shape, offName)) ||
      failed(isIntegerArrayAttrSmallerThanShape(op, sizes, shape, sizesName)) ||
      failed(isIntegerArrayAttrSmallerThanShape(op, strides, shape,
                                                stridesName)) ||
      failed(isIntegerArrayAttrConfinedToShape(op, offsets, shape, offName)) ||
      failed(isIntegerArrayAttrConfinedToShape(op, sizes, shape, sizesName,
                                               /*halfOpen=*/false,
                                               /*min=*/1)) ||
      failed(isIntegerArrayAttrConfinedToRange(op, strides, 1, 1, stridesName,
                                               /*halfOpen=*/false)) ||
      failed(isSumOfIntegerArrayAttrConfinedToShape(op, offsets, sizes, shape,
                                                    offName, sizesName,
                                                    /*halfOpen=*/false)))
    return failure();

  auto resultType = inferStridedSliceOpResultType(
      op.getVectorType(), op.offsets(), op.sizes(), op.strides());
  if (op.getResult().getType() != resultType) {
    op.emitOpError("expected result type to be ") << resultType;
    return failure();
  }

  return success();
}

void ExtractStridedSliceOp::getOffsets(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(offsets(), results);
}

namespace {

// Pattern to rewrite a ExtractStridedSliceOp(ConstantMaskOp) -> ConstantMaskOp.
class StridedSliceConstantMaskFolder final
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern<ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp extractStridedSliceOp,
                                PatternRewriter &rewriter) const override {
    // Return if 'extractStridedSliceOp' operand is not defined by a
    // ConstantMaskOp.
    auto defOp = extractStridedSliceOp.vector().getDefiningOp();
    auto constantMaskOp = dyn_cast_or_null<ConstantMaskOp>(defOp);
    if (!constantMaskOp)
      return failure();
    // Return if 'extractStridedSliceOp' has non-unit strides.
    if (llvm::any_of(extractStridedSliceOp.strides(), [](Attribute attr) {
          return attr.cast<IntegerAttr>().getInt() != 1;
        }))
      return failure();
    // Gather constant mask dimension sizes.
    SmallVector<int64_t, 4> maskDimSizes;
    populateFromInt64AttrArray(constantMaskOp.mask_dim_sizes(), maskDimSizes);
    // Gather strided slice offsets and sizes.
    SmallVector<int64_t, 4> sliceOffsets;
    populateFromInt64AttrArray(extractStridedSliceOp.offsets(), sliceOffsets);
    SmallVector<int64_t, 4> sliceSizes;
    populateFromInt64AttrArray(extractStridedSliceOp.sizes(), sliceSizes);

    // Compute slice of vector mask region.
    SmallVector<int64_t, 4> sliceMaskDimSizes;
    assert(sliceOffsets.size() == maskDimSizes.size());
    for (auto it : llvm::zip(maskDimSizes, sliceOffsets, sliceSizes)) {
      int64_t maskDimSize = std::get<0>(it);
      int64_t sliceOffset = std::get<1>(it);
      int64_t sliceSize = std::get<2>(it);
      int64_t sliceMaskDimSize = std::max(
          static_cast<int64_t>(0),
          std::min(sliceOffset + sliceSize, maskDimSize) - sliceOffset);
      sliceMaskDimSizes.push_back(sliceMaskDimSize);
    }
    // If any of 'sliceMaskDimSizes' are zero, then set all to zero (masked
    // region is a conjunction of mask dim intervals).
    if (llvm::any_of(sliceMaskDimSizes, [](int64_t sz) { return sz == 0; }))
      sliceMaskDimSizes.assign(maskDimSizes.size(), 0);

    // Replace 'extractStridedSliceOp' with ConstantMaskOp with sliced mask
    // region.
    rewriter.replaceOpWithNewOp<ConstantMaskOp>(
        extractStridedSliceOp, extractStridedSliceOp.getResult().getType(),
        vector::getVectorSubscriptAttr(rewriter, sliceMaskDimSizes));
    return success();
  }
};

} // end anonymous namespace

void ExtractStridedSliceOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  // Pattern to rewrite a ExtractStridedSliceOp(ConstantMaskOp) ->
  // ConstantMaskOp.
  results.insert<StridedSliceConstantMaskFolder>(context);
}

//===----------------------------------------------------------------------===//
// TransferReadOp
//===----------------------------------------------------------------------===//

/// Build the default minor identity map suitable for a vector transfer. This
/// also handles the case memref<... x vector<...>> -> vector<...> in which the
/// rank of the identity map must take the vector element type into account.
AffineMap
mlir::vector::impl::getTransferMinorIdentityMap(MemRefType memRefType,
                                                VectorType vectorType) {
  int64_t elementVectorRank = 0;
  VectorType elementVectorType =
      memRefType.getElementType().dyn_cast<VectorType>();
  if (elementVectorType)
    elementVectorRank += elementVectorType.getRank();
  return AffineMap::getMinorIdentityMap(
      memRefType.getRank(), vectorType.getRank() - elementVectorRank,
      memRefType.getContext());
}

template <typename EmitFun>
static LogicalResult verifyPermutationMap(AffineMap permutationMap,
                                          EmitFun emitOpError) {
  SmallVector<bool, 8> seen(permutationMap.getNumInputs(), false);
  for (auto expr : permutationMap.getResults()) {
    auto dim = expr.dyn_cast<AffineDimExpr>();
    auto zero = expr.dyn_cast<AffineConstantExpr>();
    if (zero) {
      if (zero.getValue() != 0) {
        return emitOpError(
            "requires a projected permutation_map (at most one dim or the zero "
            "constant can appear in each result)");
      }
      continue;
    }
    if (!dim) {
      return emitOpError("requires a projected permutation_map (at most one "
                         "dim or the zero constant can appear in each result)");
    }
    if (seen[dim.getPosition()]) {
      return emitOpError(
          "requires a permutation_map that is a permutation (found one dim "
          "used more than once)");
    }
    seen[dim.getPosition()] = true;
  }
  return success();
}

static LogicalResult verifyTransferOp(Operation *op, MemRefType memrefType,
                                      VectorType vectorType,
                                      AffineMap permutationMap,
                                      ArrayAttr optionalMasked) {
  auto memrefElementType = memrefType.getElementType();
  if (auto memrefVectorElementType = memrefElementType.dyn_cast<VectorType>()) {
    // Memref has vector element type.

    // Check that 'memrefVectorElementType' and vector element types match.
    if (memrefVectorElementType.getElementType() != vectorType.getElementType())
      return op->emitOpError(
          "requires memref and vector types of the same elemental type");

    // Check that memref vector type is a suffix of 'vectorType.
    unsigned memrefVecEltRank = memrefVectorElementType.getRank();
    unsigned resultVecRank = vectorType.getRank();
    if (memrefVecEltRank > resultVecRank)
      return op->emitOpError(
          "requires memref vector element and vector result ranks to match.");
    // TODO(b/146516564) Move this to isSuffix in Vector/Utils.h.
    unsigned rankOffset = resultVecRank - memrefVecEltRank;
    auto memrefVecEltShape = memrefVectorElementType.getShape();
    auto resultVecShape = vectorType.getShape();
    for (unsigned i = 0; i < memrefVecEltRank; ++i)
      if (memrefVecEltShape[i] != resultVecShape[rankOffset + i])
        return op->emitOpError(
            "requires memref vector element shape to match suffix of "
            "vector result shape.");
    // Check that permutation map results match 'rankOffset' of vector type.
    if (permutationMap.getNumResults() != rankOffset)
      return op->emitOpError("requires a permutation_map with result dims of "
                             "the same rank as the vector type");
  } else {
    // Memref has scalar element type.

    // Check that memref and vector element types match.
    if (memrefType.getElementType() != vectorType.getElementType())
      return op->emitOpError(
          "requires memref and vector types of the same elemental type");

    // Check that permutation map results match rank of vector type.
    if (permutationMap.getNumResults() != vectorType.getRank())
      return op->emitOpError("requires a permutation_map with result dims of "
                             "the same rank as the vector type");
  }

  if (permutationMap.getNumSymbols() != 0)
    return op->emitOpError("requires permutation_map without symbols");
  if (permutationMap.getNumInputs() != memrefType.getRank())
    return op->emitOpError("requires a permutation_map with input dims of the "
                           "same rank as the memref type");

  if (optionalMasked) {
    if (permutationMap.getNumResults() !=
        static_cast<int64_t>(optionalMasked.size()))
      return op->emitOpError("expects the optional masked attr of same rank as "
                             "permutation_map results: ")
             << AffineMapAttr::get(permutationMap);
  }

  return success();
}

/// Builder that sets padding to zero.
void TransferReadOp::build(OpBuilder &builder, OperationState &result,
                           VectorType vector, Value memref, ValueRange indices,
                           AffineMap permutationMap,
                           ArrayRef<bool> maybeMasked) {
  Type elemType = vector.cast<VectorType>().getElementType();
  Value padding = builder.create<ConstantOp>(result.location, elemType,
                                             builder.getZeroAttr(elemType));
  if (maybeMasked.empty())
    return build(builder, result, vector, memref, indices, permutationMap,
                 padding, ArrayAttr());
  ArrayAttr maskedArrayAttr = builder.getBoolArrayAttr(maybeMasked);
  build(builder, result, vector, memref, indices, permutationMap, padding,
        maskedArrayAttr);
}

/// Builder that sets permutation map (resp. padding) to 'getMinorIdentityMap'
/// (resp. zero).
void TransferReadOp::build(OpBuilder &builder, OperationState &result,
                           VectorType vectorType, Value memref,
                           ValueRange indices, ArrayRef<bool> maybeMasked) {
  auto permMap = getTransferMinorIdentityMap(
      memref.getType().cast<MemRefType>(), vectorType);
  build(builder, result, vectorType, memref, indices, permMap, maybeMasked);
}

template <typename TransferOp>
static void printTransferAttrs(OpAsmPrinter &p, TransferOp op) {
  SmallVector<StringRef, 2> elidedAttrs;
  if (op.permutation_map() == TransferOp::getTransferMinorIdentityMap(
                                  op.getMemRefType(), op.getVectorType()))
    elidedAttrs.push_back(op.getPermutationMapAttrName());
  bool elideMasked = true;
  if (auto maybeMasked = op.masked()) {
    for (auto attr : *maybeMasked) {
      if (!attr.template cast<BoolAttr>().getValue()) {
        elideMasked = false;
        break;
      }
    }
  }
  if (elideMasked)
    elidedAttrs.push_back(op.getMaskedAttrName());
  p.printOptionalAttrDict(op.getAttrs(), elidedAttrs);
}

static void print(OpAsmPrinter &p, TransferReadOp op) {
  p << op.getOperationName() << " " << op.memref() << "[" << op.indices()
    << "], " << op.padding();
  printTransferAttrs(p, op);
  p << " : " << op.getMemRefType() << ", " << op.getVectorType();
}

static ParseResult parseTransferReadOp(OpAsmParser &parser,
                                       OperationState &result) {
  llvm::SMLoc typesLoc;
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexInfo;
  OpAsmParser::OperandType paddingInfo;
  SmallVector<Type, 2> types;
  // Parsing with support for paddingValue.
  if (parser.parseOperand(memrefInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(paddingInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");
  auto indexType = parser.getBuilder().getIndexType();
  MemRefType memRefType = types[0].dyn_cast<MemRefType>();
  if (!memRefType)
    return parser.emitError(typesLoc, "requires memref type");
  VectorType vectorType = types[1].dyn_cast<VectorType>();
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  auto permutationAttrName = TransferReadOp::getPermutationMapAttrName();
  auto attr = result.attributes.get(permutationAttrName);
  if (!attr) {
    auto permMap =
        TransferReadOp::getTransferMinorIdentityMap(memRefType, vectorType);
    result.attributes.set(permutationAttrName, AffineMapAttr::get(permMap));
  }
  return failure(
      parser.resolveOperand(memrefInfo, memRefType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands) ||
      parser.resolveOperand(paddingInfo, memRefType.getElementType(),
                            result.operands) ||
      parser.addTypeToList(vectorType, result.types));
}

static LogicalResult verify(TransferReadOp op) {
  // Consistency of elemental types in memref and vector.
  MemRefType memrefType = op.getMemRefType();
  VectorType vectorType = op.getVectorType();
  auto paddingType = op.padding().getType();
  auto permutationMap = op.permutation_map();
  auto memrefElementType = memrefType.getElementType();

  if (static_cast<int64_t>(op.indices().size()) != memrefType.getRank())
    return op.emitOpError("requires ") << memrefType.getRank() << " indices";

  if (failed(verifyTransferOp(op.getOperation(), memrefType, vectorType,
                              permutationMap,
                              op.masked() ? *op.masked() : ArrayAttr())))
    return failure();

  if (auto memrefVectorElementType = memrefElementType.dyn_cast<VectorType>()) {
    // Memref has vector element type.
    // Check that 'memrefVectorElementType' and 'paddingType' types match.
    if (memrefVectorElementType != paddingType)
      return op.emitOpError(
          "requires memref element type and padding type to match.");

  } else {
    // Check that 'paddingType' is valid to store in a vector type.
    if (!VectorType::isValidElementType(paddingType))
      return op.emitOpError("requires valid padding vector elemental type");

    // Check that padding type and vector element types match.
    if (paddingType != vectorType.getElementType())
      return op.emitOpError(
          "requires formal padding and vector of the same elemental type");
  }

  return verifyPermutationMap(permutationMap,
                              [&op](Twine t) { return op.emitOpError(t); });
}

//===----------------------------------------------------------------------===//
// TransferWriteOp
//===----------------------------------------------------------------------===//

/// Builder that sets permutation map to 'getMinorIdentityMap'.
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value memref, ValueRange indices,
                            ArrayRef<bool> maybeMasked) {
  auto vectorType = vector.getType().cast<VectorType>();
  auto permMap = getTransferMinorIdentityMap(
      memref.getType().cast<MemRefType>(), vectorType);
  if (maybeMasked.empty())
    return build(builder, result, vector, memref, indices, permMap,
                 ArrayAttr());
  ArrayAttr maskedArrayAttr = builder.getBoolArrayAttr(maybeMasked);
  build(builder, result, vector, memref, indices, permMap, maskedArrayAttr);
}

/// Builder that sets permutation map to 'getMinorIdentityMap'.
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value memref, ValueRange indices,
                            AffineMap permutationMap) {
  build(builder, result, vector, memref, indices,
        /*maybeMasked=*/ArrayRef<bool>{});
}

static ParseResult parseTransferWriteOp(OpAsmParser &parser,
                                        OperationState &result) {
  llvm::SMLoc typesLoc;
  OpAsmParser::OperandType vectorInfo, memrefInfo;
  SmallVector<OpAsmParser::OperandType, 8> indexInfo;
  SmallVector<Type, 2> types;
  if (parser.parseOperand(vectorInfo) || parser.parseComma() ||
      parser.parseOperand(memrefInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");
  auto indexType = parser.getBuilder().getIndexType();
  VectorType vectorType = types[0].dyn_cast<VectorType>();
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  MemRefType memRefType = types[1].dyn_cast<MemRefType>();
  if (!memRefType)
    return parser.emitError(typesLoc, "requires memref type");
  auto permutationAttrName = TransferWriteOp::getPermutationMapAttrName();
  auto attr = result.attributes.get(permutationAttrName);
  if (!attr) {
    auto permMap =
        TransferWriteOp::getTransferMinorIdentityMap(memRefType, vectorType);
    result.attributes.set(permutationAttrName, AffineMapAttr::get(permMap));
  }
  return failure(
      parser.resolveOperand(vectorInfo, vectorType, result.operands) ||
      parser.resolveOperand(memrefInfo, memRefType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands));
}

static void print(OpAsmPrinter &p, TransferWriteOp op) {
  p << op.getOperationName() << " " << op.vector() << ", " << op.memref() << "["
    << op.indices() << "]";
  printTransferAttrs(p, op);
  p << " : " << op.getVectorType() << ", " << op.getMemRefType();
}

static LogicalResult verify(TransferWriteOp op) {
  // Consistency of elemental types in memref and vector.
  MemRefType memrefType = op.getMemRefType();
  VectorType vectorType = op.getVectorType();
  auto permutationMap = op.permutation_map();

  if (llvm::size(op.indices()) != memrefType.getRank())
    return op.emitOpError("requires ") << memrefType.getRank() << " indices";

  if (failed(verifyTransferOp(op.getOperation(), memrefType, vectorType,
                              permutationMap,
                              op.masked() ? *op.masked() : ArrayAttr())))
    return failure();

  return verifyPermutationMap(permutationMap,
                              [&op](Twine t) { return op.emitOpError(t); });
}

//===----------------------------------------------------------------------===//
// ShapeCastOp
//===----------------------------------------------------------------------===//

/// Returns true if each element of 'a' is equal to the product of a contiguous
/// sequence of the elements of 'b'. Returns false otherwise.
static bool isValidShapeCast(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  unsigned rankA = a.size();
  unsigned rankB = b.size();
  assert(rankA < rankB);

  unsigned i = 0;
  unsigned j = 0;
  while (i < rankA && j < rankB) {
    int64_t dimA = a[i];
    int64_t dimB = 1;
    while (dimB < dimA && j < rankB)
      dimB *= b[j++];
    if (dimA != dimB)
      break;
    ++i;
  }

  return i == rankA && j == rankB;
}

static LogicalResult verifyVectorShapeCast(Operation *op,
                                           VectorType sourceVectorType,
                                           VectorType resultVectorType) {
  // Check that element type is the same.
  if (sourceVectorType.getElementType() != resultVectorType.getElementType())
    return op->emitOpError("source/result vectors must have same element type");
  auto sourceShape = sourceVectorType.getShape();
  auto resultShape = resultVectorType.getShape();

  // Check that product of source dim sizes matches product of result dim sizes.
  int64_t sourceDimProduct = std::accumulate(
      sourceShape.begin(), sourceShape.end(), 1LL, std::multiplies<int64_t>{});
  int64_t resultDimProduct = std::accumulate(
      resultShape.begin(), resultShape.end(), 1LL, std::multiplies<int64_t>{});
  if (sourceDimProduct != resultDimProduct)
    return op->emitOpError("source/result number of elements must match");

  // Check that expanding/contracting rank cases.
  unsigned sourceRank = sourceVectorType.getRank();
  unsigned resultRank = resultVectorType.getRank();
  if (sourceRank < resultRank) {
    if (!isValidShapeCast(sourceShape, resultShape))
      return op->emitOpError("invalid shape cast");
  } else if (sourceRank > resultRank) {
    if (!isValidShapeCast(resultShape, sourceShape))
      return op->emitOpError("invalid shape cast");
  }
  return success();
}

static LogicalResult verify(ShapeCastOp op) {
  auto sourceVectorType = op.source().getType().dyn_cast_or_null<VectorType>();
  auto resultVectorType = op.result().getType().dyn_cast_or_null<VectorType>();

  // Check if source/result are of vector type.
  if (sourceVectorType && resultVectorType)
    return verifyVectorShapeCast(op, sourceVectorType, resultVectorType);

  // Check if source/result are "tuple of vectors" type.
  auto sourceTupleType = op.source().getType().dyn_cast_or_null<TupleType>();
  auto resultTupleType = op.result().getType().dyn_cast_or_null<TupleType>();
  if (!sourceTupleType || !resultTupleType)
    return op.emitOpError("source/result must be of same type");

  // Check that source/result tuple sizes are the same.
  if (sourceTupleType.size() != resultTupleType.size())
    return op.emitOpError("source/result tuples must be the same size");

  // Check each source/result tuple element pair.
  for (unsigned i = 0, e = sourceTupleType.size(); i < e; ++i)
    if (failed(verifyVectorShapeCast(
            op, sourceTupleType.getType(i).cast<VectorType>(),
            resultTupleType.getType(i).cast<VectorType>())))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TypeCastOp
//===----------------------------------------------------------------------===//

static SmallVector<int64_t, 8> extractShape(MemRefType memRefType) {
  auto vectorType = memRefType.getElementType().dyn_cast<VectorType>();
  SmallVector<int64_t, 8> res(memRefType.getShape().begin(),
                              memRefType.getShape().end());
  if (vectorType) {
    res.reserve(memRefType.getRank() + vectorType.getRank());
    for (auto s : vectorType.getShape())
      res.push_back(s);
  }
  return res;
}

/// Build the canonical memRefType with a single vector.
/// E.g. memref<4 x 5 x vector<6 x f32>> -> memref<vector<4 x 5 x 6 x f32>>.
void TypeCastOp::build(OpBuilder &builder, OperationState &result,
                       Value source) {
  result.addOperands(source);
  MemRefType memRefType = source.getType().cast<MemRefType>();
  VectorType vectorType =
      VectorType::get(extractShape(memRefType),
                      getElementTypeOrSelf(getElementTypeOrSelf(memRefType)));
  result.addTypes(
      MemRefType::get({}, vectorType, {}, memRefType.getMemorySpace()));
}

static LogicalResult verify(TypeCastOp op) {
  MemRefType canonicalType = canonicalizeStridedLayout(op.getMemRefType());
  if (!canonicalType.getAffineMaps().empty())
    return op.emitOpError("expects operand to be a memref with no layout");
  if (!op.getResultMemRefType().getAffineMaps().empty())
    return op.emitOpError("expects result to be a memref with no layout");
  if (op.getResultMemRefType().getMemorySpace() !=
      op.getMemRefType().getMemorySpace())
    return op.emitOpError("expects result in same memory space");

  auto sourceType = op.getMemRefType();
  auto resultType = op.getResultMemRefType();
  if (getElementTypeOrSelf(getElementTypeOrSelf(sourceType)) !=
      getElementTypeOrSelf(getElementTypeOrSelf(resultType)))
    return op.emitOpError(
               "expects result and operand with same underlying scalar type: ")
           << resultType;
  if (extractShape(sourceType) != extractShape(resultType))
    return op.emitOpError(
               "expects concatenated result and operand shapes to be equal: ")
           << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

static ParseResult parseTupleOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operandInfos;
  SmallVector<Type, 4> types;
  auto loc = parser.getCurrentLocation();
  auto *ctx = parser.getBuilder().getContext();
  return failure(
      parser.parseOperandList(operandInfos) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(operandInfos, types, loc, result.operands) ||
      parser.addTypeToList(TupleType::get(types, ctx), result.types));
}

static void print(OpAsmPrinter &p, TupleOp op) {
  p << op.getOperationName() << ' ';
  p.printOperands(op.getOperands());
  p.printOptionalAttrDict(op.getAttrs());
  p << " : ";
  llvm::interleaveComma(op.getOperation()->getOperandTypes(), p);
}

static LogicalResult verify(TupleOp op) { return success(); }

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void vector::TransposeOp::build(OpBuilder &builder, OperationState &result,
                                Value vector, ArrayRef<int64_t> transp) {
  VectorType vt = vector.getType().cast<VectorType>();
  SmallVector<int64_t, 4> transposedShape(vt.getRank());
  for (unsigned i = 0; i < transp.size(); ++i)
    transposedShape[i] = vt.getShape()[transp[i]];

  result.addOperands(vector);
  result.addTypes(VectorType::get(transposedShape, vt.getElementType()));
  result.addAttribute(getTranspAttrName(), builder.getI64ArrayAttr(transp));
}

// Eliminates transpose operations, which produce values identical to their
// input values. This happens when the dimensions of the input vector remain in
// their original order after the transpose operation.
OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  SmallVector<int64_t, 4> transp;
  getTransp(transp);

  // Check if the permutation of the dimensions contains sequential values:
  // {0, 1, 2, ...}.
  for (int64_t i = 0, e = transp.size(); i < e; i++) {
    if (transp[i] != i)
      return {};
  }

  return vector();
}

static LogicalResult verify(TransposeOp op) {
  VectorType vectorType = op.getVectorType();
  VectorType resultType = op.getResultType();
  int64_t rank = resultType.getRank();
  if (vectorType.getRank() != rank)
    return op.emitOpError("vector result rank mismatch: ") << rank;
  // Verify transposition array.
  auto transpAttr = op.transp().getValue();
  int64_t size = transpAttr.size();
  if (rank != size)
    return op.emitOpError("transposition length mismatch: ") << size;
  SmallVector<bool, 8> seen(rank, false);
  for (auto ta : llvm::enumerate(transpAttr)) {
    int64_t i = ta.value().cast<IntegerAttr>().getInt();
    if (i < 0 || i >= rank)
      return op.emitOpError("transposition index out of range: ") << i;
    if (seen[i])
      return op.emitOpError("duplicate position index: ") << i;
    seen[i] = true;
    if (resultType.getDimSize(ta.index()) != vectorType.getDimSize(i))
      return op.emitOpError("dimension size mismatch at: ") << i;
  }
  return success();
}

namespace {

// Rewrites two back-to-back TransposeOp operations into a single TransposeOp.
class TransposeFolder final : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Wrapper around TransposeOp::getTransp() for cleaner code.
    auto getPermutation = [](TransposeOp transpose) {
      SmallVector<int64_t, 4> permutation;
      transpose.getTransp(permutation);
      return permutation;
    };

    // Composes two permutations: result[i] = permutation1[permutation2[i]].
    auto composePermutations = [](ArrayRef<int64_t> permutation1,
                                  ArrayRef<int64_t> permutation2) {
      SmallVector<int64_t, 4> result;
      for (auto index : permutation2)
        result.push_back(permutation1[index]);
      return result;
    };

    // Return if the input of 'transposeOp' is not defined by another transpose.
    TransposeOp parentTransposeOp =
        transposeOp.vector().getDefiningOp<TransposeOp>();
    if (!parentTransposeOp)
      return failure();

    SmallVector<int64_t, 4> permutation = composePermutations(
        getPermutation(parentTransposeOp), getPermutation(transposeOp));
    // Replace 'transposeOp' with a new transpose operation.
    rewriter.replaceOpWithNewOp<TransposeOp>(
        transposeOp, transposeOp.getResult().getType(),
        parentTransposeOp.vector(),
        vector::getVectorSubscriptAttr(rewriter, permutation));
    return success();
  }
};

} // end anonymous namespace

void TransposeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<TransposeFolder>(context);
}

void TransposeOp::getTransp(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(transp(), results);
}

//===----------------------------------------------------------------------===//
// TupleGetOp
//===----------------------------------------------------------------------===//

static ParseResult parseTupleGetOp(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr indexAttr;
  StringRef indexAttrName = TupleGetOp::getIndexAttrName();
  Type indexType = parser.getBuilder().getIndexType();
  TupleType tupleType;
  if (parser.parseOperand(operandInfo) || parser.parseComma() ||
      parser.parseAttribute(indexAttr, indexType, indexAttrName,
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(tupleType) ||
      parser.resolveOperand(operandInfo, tupleType, result.operands))
    return failure();
  if (indexAttr.getInt() < 0 ||
      indexAttr.getInt() >= static_cast<int64_t>(tupleType.size()))
    return failure();
  parser.addTypeToList(tupleType.getType(indexAttr.getInt()), result.types);
  return success();
}

static void print(OpAsmPrinter &p, TupleGetOp op) {
  p << op.getOperationName() << ' ' << op.getOperand() << ", " << op.index();
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{TupleGetOp::getIndexAttrName()});
  p << " : " << op.getOperand().getType();
}

static LogicalResult verify(TupleGetOp op) {
  auto tupleType = op.getOperand().getType().cast<TupleType>();
  if (op.getIndex() < 0 ||
      op.getIndex() >= static_cast<int64_t>(tupleType.size()))
    return op.emitOpError("tuple get index out of range");
  return success();
}

OpFoldResult TupleGetOp::fold(ArrayRef<Attribute> operands) {
  // Rewrite:
  //    %t = vector.tuple .., %e_i, ..
  //    %x = vector.tuple_get %t, i
  // into:
  //    %t = vector.tuple .., %e_i, ..  // one less use
  //    %x = %e_i
  if (auto tupleOp = getOperand().getDefiningOp<TupleOp>())
    return tupleOp.getOperand(getIndex());
  return {};
}

//===----------------------------------------------------------------------===//
// ConstantMaskOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ConstantMaskOp &op) {
  // Verify that array attr size matches the rank of the vector result.
  auto resultType = op.getResult().getType().cast<VectorType>();
  if (static_cast<int64_t>(op.mask_dim_sizes().size()) != resultType.getRank())
    return op.emitOpError(
        "must specify array attr of size equal vector result rank");
  // Verify that each array attr element is in bounds of corresponding vector
  // result dimension size.
  auto resultShape = resultType.getShape();
  SmallVector<int64_t, 4> maskDimSizes;
  for (auto it : llvm::enumerate(op.mask_dim_sizes())) {
    int64_t attrValue = it.value().cast<IntegerAttr>().getInt();
    if (attrValue < 0 || attrValue > resultShape[it.index()])
      return op.emitOpError(
          "array attr of size out of bounds of vector result dimension size");
    maskDimSizes.push_back(attrValue);
  }
  // Verify that if one mask dim size is zero, they all should be zero (because
  // the mask region is a conjunction of each mask dimension interval).
  bool any_zeros = llvm::is_contained(maskDimSizes, 0);
  bool all_zeros = llvm::all_of(maskDimSizes, [](int64_t s) { return s == 0; });
  if (any_zeros && !all_zeros)
    return op.emitOpError("expected all mask dim sizes to be zeros, "
                          "as a result of conjunction with zero mask dim");
  return success();
}

//===----------------------------------------------------------------------===//
// CreateMaskOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(CreateMaskOp op) {
  // Verify that an operand was specified for each result vector each dimension.
  if (op.getNumOperands() !=
      op.getResult().getType().cast<VectorType>().getRank())
    return op.emitOpError(
        "must specify an operand for each result vector dimension");
  return success();
}

namespace {

// Pattern to rewrite a CreateMaskOp with a ConstantMaskOp.
class CreateMaskFolder final : public OpRewritePattern<CreateMaskOp> {
public:
  using OpRewritePattern<CreateMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateMaskOp createMaskOp,
                                PatternRewriter &rewriter) const override {
    // Return if any of 'createMaskOp' operands are not defined by a constant.
    auto is_not_def_by_constant = [](Value operand) {
      return !isa_and_nonnull<ConstantIndexOp>(operand.getDefiningOp());
    };
    if (llvm::any_of(createMaskOp.operands(), is_not_def_by_constant))
      return failure();
    // Gather constant mask dimension sizes.
    SmallVector<int64_t, 4> maskDimSizes;
    for (auto operand : createMaskOp.operands()) {
      auto defOp = operand.getDefiningOp();
      maskDimSizes.push_back(cast<ConstantIndexOp>(defOp).getValue());
    }
    // Replace 'createMaskOp' with ConstantMaskOp.
    rewriter.replaceOpWithNewOp<ConstantMaskOp>(
        createMaskOp, createMaskOp.getResult().getType(),
        vector::getVectorSubscriptAttr(rewriter, maskDimSizes));
    return success();
  }
};

} // end anonymous namespace

void CreateMaskOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CreateMaskFolder>(context);
}

void mlir::vector::populateVectorToVectorCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<CreateMaskFolder, StridedSliceConstantMaskFolder,
                  TransposeFolder>(context);
}

namespace mlir {
namespace vector {

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/VectorOps.cpp.inc"

} // namespace vector
} // namespace mlir
