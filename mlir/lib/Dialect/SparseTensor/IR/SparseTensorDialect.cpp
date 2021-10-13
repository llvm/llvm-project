//===- SparseTensorDialect.cpp - Sparse tensor dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

#include "mlir/Dialect/SparseTensor/IR/SparseTensorOpsDialect.cpp.inc"

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

Attribute SparseTensorEncodingAttr::parse(DialectAsmParser &parser, Type type) {
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
    if (attr.first == "dimLevelType") {
      auto arrayAttr = attr.second.dyn_cast<ArrayAttr>();
      if (!arrayAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an array for dimension level types");
        return {};
      }
      for (unsigned i = 0, e = arrayAttr.size(); i < e; i++) {
        auto strAttr = arrayAttr[i].dyn_cast<StringAttr>();
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
    } else if (attr.first == "dimOrdering") {
      auto affineAttr = attr.second.dyn_cast<AffineMapAttr>();
      if (!affineAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an affine map for dimension ordering");
        return {};
      }
      map = affineAttr.getValue();
    } else if (attr.first == "pointerBitWidth") {
      auto intAttr = attr.second.dyn_cast<IntegerAttr>();
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an integral pointer bitwidth");
        return {};
      }
      ptr = intAttr.getInt();
    } else if (attr.first == "indexBitWidth") {
      auto intAttr = attr.second.dyn_cast<IntegerAttr>();
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an integral index bitwidth");
        return {};
      }
      ind = intAttr.getInt();
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.first.str();
      return {};
    }
  }
  // Construct struct-like storage for attribute.
  return parser.getChecked<SparseTensorEncodingAttr>(parser.getContext(), dlt,
                                                     map, ptr, ind);
}

void SparseTensorEncodingAttr::print(DialectAsmPrinter &printer) const {
  // Print the struct-like storage in dictionary fashion.
  printer << "encoding<{ dimLevelType = [ ";
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
  if (auto constantOp = dim.getDefiningOp<arith::ConstantOp>()) {
    unsigned d = constantOp.value().cast<IntegerAttr>().getInt();
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

static LogicalResult verify(NewOp op) {
  if (!getSparseTensorEncoding(op.result().getType()))
    return op.emitError("expected a sparse tensor result");
  return success();
}

static LogicalResult verify(InitOp op) {
  if (!getSparseTensorEncoding(op.result().getType()))
    return op.emitError("expected a sparse tensor result");
  RankedTensorType ttp = op.getType().cast<RankedTensorType>();
  unsigned rank = ttp.getRank();
  if (rank != op.sizes().size())
    return op.emitError("unexpected mismatch between tensor rank and sizes: ")
           << rank << " vs. " << op.sizes().size();
  auto shape = ttp.getShape();
  for (unsigned i = 0; i < rank; i++) {
    if (shape[i] == ShapedType::kDynamicSize)
      continue;
    auto constantOp = op.sizes()[i].getDefiningOp<arith::ConstantOp>();
    if (!constantOp ||
        constantOp.value().cast<IntegerAttr>().getInt() != shape[i])
      return op.emitError("unexpected mismatch with static dimension size ")
             << shape[i];
  }
  return success();
}

static LogicalResult verify(ConvertOp op) {
  if (auto tp1 = op.source().getType().dyn_cast<RankedTensorType>()) {
    if (auto tp2 = op.dest().getType().dyn_cast<RankedTensorType>()) {
      assert(tp1.getRank() == tp2.getRank());
      auto shape1 = tp1.getShape();
      auto shape2 = tp2.getShape();
      for (unsigned d = 0, rank = tp1.getRank(); d < rank; d++) {
        if (shape1[d] != shape2[d])
          return op.emitError("unexpected conversion mismatch in dimension ")
                 << d;
      }
      return success();
    }
  }
  return op.emitError("unexpected type in convert");
}

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
  if (getType() == source().getType())
    return source();
  return {};
}

static LogicalResult verify(ReleaseOp op) {
  if (!getSparseTensorEncoding(op.tensor().getType()))
    return op.emitError("expected a sparse tensor to release");
  return success();
}

static LogicalResult verify(ToPointersOp op) {
  if (auto e = getSparseTensorEncoding(op.tensor().getType())) {
    if (failed(isInBounds(op.dim(), op.tensor())))
      return op.emitError("requested pointers dimension out of bounds");
    if (failed(isMatchingWidth(op.result(), e.getPointerBitWidth())))
      return op.emitError("unexpected type for pointers");
    return success();
  }
  return op.emitError("expected a sparse tensor to get pointers");
}

static LogicalResult verify(ToIndicesOp op) {
  if (auto e = getSparseTensorEncoding(op.tensor().getType())) {
    if (failed(isInBounds(op.dim(), op.tensor())))
      return op.emitError("requested indices dimension out of bounds");
    if (failed(isMatchingWidth(op.result(), e.getIndexBitWidth())))
      return op.emitError("unexpected type for indices");
    return success();
  }
  return op.emitError("expected a sparse tensor to get indices");
}

static LogicalResult verify(ToValuesOp op) {
  if (!getSparseTensorEncoding(op.tensor().getType()))
    return op.emitError("expected a sparse tensor to get values");
  RankedTensorType ttp = op.tensor().getType().cast<RankedTensorType>();
  MemRefType mtp = op.result().getType().cast<MemRefType>();
  if (ttp.getElementType() != mtp.getElementType())
    return op.emitError("unexpected mismatch in element types");
  return success();
}

static LogicalResult verify(ToTensorOp op) {
  if (!getSparseTensorEncoding(op.result().getType()))
    return op.emitError("expected a sparse tensor result");
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

Attribute SparseTensorDialect::parseAttribute(DialectAsmParser &parser,
                                              Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
  if (parseResult.hasValue())
    return attr;
  parser.emitError(parser.getNameLoc(), "unknown sparse tensor attribute");
  return Attribute();
}

void SparseTensorDialect::printAttribute(Attribute attr,
                                         DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}
