//===- SparseTensorDialect.cpp - Sparse tensor dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// TensorDialect Attribute Methods.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.cpp.inc"

static bool acceptBitWidth(unsigned bitWidth) {
  switch (bitWidth) {
  case 0:
  case 8:
  case 16:
  case 32:
  case 64:
    return true;
  default:
    return false;
  }
}

Attribute SparseTensorEncodingAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  // Parse the data as a dictionary.
  DictionaryAttr dict;
  if (failed(parser.parseAttribute(dict)))
    return {};
  if (failed(parser.parseGreater()))
    return {};
  // Process the data from the parsed dictionary value into struct-like data.
  SmallVector<SparseTensorEncodingAttr::DimLevelType, 4> dlt;
  AffineMap map = {};
  unsigned ptr = 0;
  unsigned ind = 0;
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "dimLevelType") {
      auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>();
      if (!arrayAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an array for dimension level types");
        return {};
      }
      for (auto i : arrayAttr) {
        auto strAttr = i.dyn_cast<StringAttr>();
        if (!strAttr) {
          parser.emitError(parser.getNameLoc(),
                           "expected a string value in dimension level types");
          return {};
        }
        auto strVal = strAttr.getValue();
        if (strVal == "dense") {
          dlt.push_back(SparseTensorEncodingAttr::DimLevelType::Dense);
        } else if (strVal == "compressed") {
          dlt.push_back(SparseTensorEncodingAttr::DimLevelType::Compressed);
        } else if (strVal == "singleton") {
          dlt.push_back(SparseTensorEncodingAttr::DimLevelType::Singleton);
        } else {
          parser.emitError(parser.getNameLoc(),
                           "unexpected dimension level type: ")
              << strVal;
          return {};
        }
      }
    } else if (attr.getName() == "dimOrdering") {
      auto affineAttr = attr.getValue().dyn_cast<AffineMapAttr>();
      if (!affineAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an affine map for dimension ordering");
        return {};
      }
      map = affineAttr.getValue();
    } else if (attr.getName() == "pointerBitWidth") {
      auto intAttr = attr.getValue().dyn_cast<IntegerAttr>();
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an integral pointer bitwidth");
        return {};
      }
      ptr = intAttr.getInt();
    } else if (attr.getName() == "indexBitWidth") {
      auto intAttr = attr.getValue().dyn_cast<IntegerAttr>();
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an integral index bitwidth");
        return {};
      }
      ind = intAttr.getInt();
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }
  // Construct struct-like storage for attribute.
  return parser.getChecked<SparseTensorEncodingAttr>(parser.getContext(), dlt,
                                                     map, ptr, ind);
}

void SparseTensorEncodingAttr::print(AsmPrinter &printer) const {
  // Print the struct-like storage in dictionary fashion.
  printer << "<{ dimLevelType = [ ";
  for (unsigned i = 0, e = getDimLevelType().size(); i < e; i++) {
    switch (getDimLevelType()[i]) {
    case DimLevelType::Dense:
      printer << "\"dense\"";
      break;
    case DimLevelType::Compressed:
      printer << "\"compressed\"";
      break;
    case DimLevelType::Singleton:
      printer << "\"singleton\"";
      break;
    }
    if (i != e - 1)
      printer << ", ";
  }
  printer << " ]";
  if (getDimOrdering())
    printer << ", dimOrdering = affine_map<" << getDimOrdering() << ">";
  printer << ", pointerBitWidth = " << getPointerBitWidth()
          << ", indexBitWidth = " << getIndexBitWidth() << " }>";
}

LogicalResult SparseTensorEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<DimLevelType> dimLevelType, AffineMap dimOrdering,
    unsigned pointerBitWidth, unsigned indexBitWidth) {
  if (!acceptBitWidth(pointerBitWidth))
    return emitError() << "unexpected pointer bitwidth: " << pointerBitWidth;
  if (!acceptBitWidth(indexBitWidth))
    return emitError() << "unexpected index bitwidth: " << indexBitWidth;
  if (dimOrdering) {
    if (!dimOrdering.isPermutation())
      return emitError()
             << "expected a permutation affine map for dimension ordering";
    if (dimOrdering.getNumResults() != dimLevelType.size())
      return emitError() << "unexpected mismatch in ordering and dimension "
                            "level types size";
  }
  return success();
}

LogicalResult SparseTensorEncodingAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    function_ref<InFlightDiagnostic()> emitError) const {
  // Check structural integrity.
  if (failed(verify(emitError, getDimLevelType(), getDimOrdering(),
                    getPointerBitWidth(), getIndexBitWidth())))
    return failure();
  // Check integrity with tensor type specifics. Dimension ordering is optional,
  // but we always should have dimension level types for the full rank.
  unsigned size = shape.size();
  if (size == 0)
    return emitError() << "expected non-scalar sparse tensor";
  if (getDimOrdering() && getDimOrdering().getNumResults() != size)
    return emitError() << "expected an affine map of size " << size
                       << " for dimension ordering";
  if (getDimLevelType().size() != size)
    return emitError() << "expected an array of size " << size
                       << " for dimension level types";
  return success();
}

SparseTensorEncodingAttr
mlir::sparse_tensor::getSparseTensorEncoding(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<SparseTensorEncodingAttr>();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TensorDialect Operations.
//===----------------------------------------------------------------------===//

static LogicalResult isInBounds(Value dim, Value tensor) {
  IntegerAttr constantAttr;
  if (matchPattern(dim, m_Constant(&constantAttr))) {
    unsigned d = constantAttr.getInt();
    if (d >= tensor.getType().cast<RankedTensorType>().getRank())
      return failure();
  }
  return success(); // in bounds, or symbolic
}

static LogicalResult isMatchingWidth(Value result, unsigned width) {
  Type etp = result.getType().cast<MemRefType>().getElementType();
  if ((width == 0 && etp.isIndex()) || (width > 0 && etp.isInteger(width)))
    return success();
  return failure();
}

LogicalResult NewOp::verify() {
  if (!getSparseTensorEncoding(result().getType()))
    return emitError("expected a sparse tensor result");
  return success();
}

LogicalResult InitOp::verify() {
  if (!getSparseTensorEncoding(result().getType()))
    return emitError("expected a sparse tensor result");
  RankedTensorType ttp = getType().cast<RankedTensorType>();
  unsigned rank = ttp.getRank();
  if (rank != sizes().size())
    return emitError("unexpected mismatch between tensor rank and sizes: ")
           << rank << " vs. " << sizes().size();
  auto shape = ttp.getShape();
  for (unsigned i = 0; i < rank; i++) {
    if (shape[i] == ShapedType::kDynamicSize)
      continue;
    IntegerAttr constantAttr;
    if (!matchPattern(sizes()[i], m_Constant(&constantAttr)) ||
        constantAttr.getInt() != shape[i]) {
      return emitError("unexpected mismatch with static dimension size ")
             << shape[i];
    }
  }
  return success();
}

LogicalResult ConvertOp::verify() {
  if (auto tp1 = source().getType().dyn_cast<RankedTensorType>()) {
    if (auto tp2 = dest().getType().dyn_cast<RankedTensorType>()) {
      if (tp1.getRank() != tp2.getRank())
        return emitError("unexpected conversion mismatch in rank");
      auto shape1 = tp1.getShape();
      auto shape2 = tp2.getShape();
      // Accept size matches between the source and the destination type
      // (e.g. 10 vs. 10, 10 vs. ?, or ? vs. ?), but reject direct mismatches or
      // matches that would need a runtime assert (e.g. 10 vs. 20 or ? vs. 10).
      for (unsigned d = 0, rank = tp1.getRank(); d < rank; d++)
        if (shape1[d] != shape2[d] && shape2[d] != ShapedType::kDynamicSize)
          return emitError("unexpected conversion mismatch in dimension ") << d;
      return success();
    }
  }
  return emitError("unexpected type in convert");
}

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
  if (getType() == source().getType())
    return source();
  return {};
}

LogicalResult ToPointersOp::verify() {
  if (auto e = getSparseTensorEncoding(tensor().getType())) {
    if (failed(isInBounds(dim(), tensor())))
      return emitError("requested pointers dimension out of bounds");
    if (failed(isMatchingWidth(result(), e.getPointerBitWidth())))
      return emitError("unexpected type for pointers");
    return success();
  }
  return emitError("expected a sparse tensor to get pointers");
}

LogicalResult ToIndicesOp::verify() {
  if (auto e = getSparseTensorEncoding(tensor().getType())) {
    if (failed(isInBounds(dim(), tensor())))
      return emitError("requested indices dimension out of bounds");
    if (failed(isMatchingWidth(result(), e.getIndexBitWidth())))
      return emitError("unexpected type for indices");
    return success();
  }
  return emitError("expected a sparse tensor to get indices");
}

LogicalResult ToValuesOp::verify() {
  if (!getSparseTensorEncoding(tensor().getType()))
    return emitError("expected a sparse tensor to get values");
  RankedTensorType ttp = tensor().getType().cast<RankedTensorType>();
  MemRefType mtp = result().getType().cast<MemRefType>();
  if (ttp.getElementType() != mtp.getElementType())
    return emitError("unexpected mismatch in element types");
  return success();
}

//===----------------------------------------------------------------------===//
// TensorDialect Management Operations.
//===----------------------------------------------------------------------===//

LogicalResult LexInsertOp::verify() {
  if (!getSparseTensorEncoding(tensor().getType()))
    return emitError("expected a sparse tensor for insertion");
  return success();
}

LogicalResult ExpandOp::verify() {
  if (!getSparseTensorEncoding(tensor().getType()))
    return emitError("expected a sparse tensor for expansion");
  return success();
}

LogicalResult CompressOp::verify() {
  if (!getSparseTensorEncoding(tensor().getType()))
    return emitError("expected a sparse tensor for compression");
  return success();
}

LogicalResult LoadOp::verify() {
  if (!getSparseTensorEncoding(tensor().getType()))
    return emitError("expected a sparse tensor to materialize");
  return success();
}

LogicalResult ReleaseOp::verify() {
  if (!getSparseTensorEncoding(tensor().getType()))
    return emitError("expected a sparse tensor to release");
  return success();
}

LogicalResult OutOp::verify() {
  if (!getSparseTensorEncoding(tensor().getType()))
    return emitError("expected a sparse tensor for output");
  return success();
}

//===----------------------------------------------------------------------===//
// TensorDialect Methods.
//===----------------------------------------------------------------------===//

void SparseTensorDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SparseTensor/IR/SparseTensorOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorOps.cpp.inc"

#include "mlir/Dialect/SparseTensor/IR/SparseTensorOpsDialect.cpp.inc"
