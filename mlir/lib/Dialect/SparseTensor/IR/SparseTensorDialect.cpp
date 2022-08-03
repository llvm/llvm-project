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
#include "llvm/Support/FormatVariadic.h"

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

LogicalResult ConvertOp::verify() {
  if (auto tp1 = getSource().getType().dyn_cast<RankedTensorType>()) {
    if (auto tp2 = getDest().getType().dyn_cast<RankedTensorType>()) {
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
  if (getType() == getSource().getType())
    return getSource();
  return {};
}

LogicalResult ToPointersOp::verify() {
  auto e = getSparseTensorEncoding(getTensor().getType());
  if (failed(isInBounds(getDim(), getTensor())))
    return emitError("requested pointers dimension out of bounds");
  if (failed(isMatchingWidth(getResult(), e.getPointerBitWidth())))
    return emitError("unexpected type for pointers");
  return success();
}

LogicalResult ToIndicesOp::verify() {
  auto e = getSparseTensorEncoding(getTensor().getType());
  if (failed(isInBounds(getDim(), getTensor())))
    return emitError("requested indices dimension out of bounds");
  if (failed(isMatchingWidth(getResult(), e.getIndexBitWidth())))
    return emitError("unexpected type for indices");
  return success();
}

LogicalResult ToValuesOp::verify() {
  RankedTensorType ttp = getTensor().getType().cast<RankedTensorType>();
  MemRefType mtp = getResult().getType().cast<MemRefType>();
  if (ttp.getElementType() != mtp.getElementType())
    return emitError("unexpected mismatch in element types");
  return success();
}

//===----------------------------------------------------------------------===//
// TensorDialect Linalg.Generic Operations.
//===----------------------------------------------------------------------===//

template <class T>
static LogicalResult verifyNumBlockArgs(T *op, Region &region,
                                        const char *regionName,
                                        TypeRange inputTypes, Type outputType) {
  unsigned numArgs = region.getNumArguments();
  unsigned expectedNum = inputTypes.size();
  if (numArgs != expectedNum)
    return op->emitError() << regionName << " region must have exactly "
                           << expectedNum << " arguments";

  for (unsigned i = 0; i < numArgs; i++) {
    Type typ = region.getArgument(i).getType();
    if (typ != inputTypes[i])
      return op->emitError() << regionName << " region argument " << (i + 1)
                             << " type mismatch";
  }
  Operation *term = region.front().getTerminator();
  YieldOp yield = dyn_cast<YieldOp>(term);
  if (!yield)
    return op->emitError() << regionName
                           << " region must end with sparse_tensor.yield";
  if (yield.getOperand().getType() != outputType)
    return op->emitError() << regionName << " region yield type mismatch";

  return success();
}

LogicalResult BinaryOp::verify() {
  NamedAttrList attrs = (*this)->getAttrs();
  Type leftType = getX().getType();
  Type rightType = getY().getType();
  Type outputType = getOutput().getType();
  Region &overlap = getOverlapRegion();
  Region &left = getLeftRegion();
  Region &right = getRightRegion();

  // Check correct number of block arguments and return type for each
  // non-empty region.
  LogicalResult regionResult = success();
  if (!overlap.empty()) {
    regionResult = verifyNumBlockArgs(
        this, overlap, "overlap", TypeRange{leftType, rightType}, outputType);
    if (failed(regionResult))
      return regionResult;
  }
  if (!left.empty()) {
    regionResult =
        verifyNumBlockArgs(this, left, "left", TypeRange{leftType}, outputType);
    if (failed(regionResult))
      return regionResult;
  } else if (getLeftIdentity()) {
    if (leftType != outputType)
      return emitError("left=identity requires first argument to have the same "
                       "type as the output");
  }
  if (!right.empty()) {
    regionResult = verifyNumBlockArgs(this, right, "right",
                                      TypeRange{rightType}, outputType);
    if (failed(regionResult))
      return regionResult;
  } else if (getRightIdentity()) {
    if (rightType != outputType)
      return emitError("right=identity requires second argument to have the "
                       "same type as the output");
  }

  return success();
}

LogicalResult UnaryOp::verify() {
  Type inputType = getX().getType();
  Type outputType = getOutput().getType();
  LogicalResult regionResult = success();

  // Check correct number of block arguments and return type for each
  // non-empty region.
  Region &present = getPresentRegion();
  if (!present.empty()) {
    regionResult = verifyNumBlockArgs(this, present, "present",
                                      TypeRange{inputType}, outputType);
    if (failed(regionResult))
      return regionResult;
  }
  Region &absent = getAbsentRegion();
  if (!absent.empty()) {
    regionResult =
        verifyNumBlockArgs(this, absent, "absent", TypeRange{}, outputType);
    if (failed(regionResult))
      return regionResult;
  }

  return success();
}

LogicalResult ConcatenateOp::verify() {
  auto dstTp = getType().cast<RankedTensorType>();
  uint64_t concatDim = getDimension().getZExtValue();
  unsigned rank = dstTp.getRank();

  if (getInputs().size() <= 1)
    return emitError("Need at least two tensors to concatenate.");

  for (auto type : getInputs().getTypes()) {
    auto shape = type.cast<RankedTensorType>().getShape();
    for (auto dim : shape) {
      if (dim == ShapedType::kDynamicSize)
        return emitError("Only statically-sized input tensors are supported.");
    }
  }

  if (concatDim >= rank)
    return emitError(llvm::formatv(
        "Failed to concatentate tensors with rank={0} on dimension={1}.", rank,
        concatDim));

  for (size_t i = 0; i < getInputs().size(); i++) {
    Value input = getInputs()[i];
    auto inputRank = input.getType().cast<RankedTensorType>().getRank();
    if (inputRank != rank)
      return emitError(
          llvm::formatv("The input tensor ${0} has a different rank (rank={1}) "
                        "from the output tensor (rank={2}).",
                        i, inputRank, rank));
  }

  for (unsigned i = 0; i < rank; i++) {
    auto dstDim = dstTp.getShape()[i];
    if (i == concatDim) {
      if (dstDim != ShapedType::kDynamicSize) {
        unsigned sumDim = 0;
        for (auto src : getInputs()) {
          // If we reach here, all inputs should have static shapes.
          auto d = src.getType().cast<RankedTensorType>().getShape()[i];
          sumDim += d;
        }
        // If all dimension are statically known, the sum of all the input
        // dimensions should be equal to the output dimension.
        if (sumDim != dstDim)
          return emitError(
              "The concatenation dimension of the output tensor should be the "
              "sum of all the concatenation dimensions of the input tensors.");
      }
    } else {
      int prev = dstDim;
      for (auto src : getInputs()) {
        auto d = src.getType().cast<RankedTensorType>().getShape()[i];
        if (prev != ShapedType::kDynamicSize && d != prev)
          return emitError("All dimensions (expect for the concatenating one) "
                           "should be equal.");
        prev = d;
      }
    }
  }

  return success();
}

LogicalResult ReduceOp::verify() {
  Type inputType = getX().getType();
  LogicalResult regionResult = success();

  // Check correct number of block arguments and return type.
  Region &formula = getRegion();
  if (!formula.empty()) {
    regionResult = verifyNumBlockArgs(
        this, formula, "reduce", TypeRange{inputType, inputType}, inputType);
    if (failed(regionResult))
      return regionResult;
  }

  return success();
}

LogicalResult YieldOp::verify() {
  // Check for compatible parent.
  auto *parentOp = (*this)->getParentOp();
  if (isa<BinaryOp>(parentOp) || isa<UnaryOp>(parentOp) ||
      isa<ReduceOp>(parentOp))
    return success();

  return emitOpError(
      "expected parent op to be sparse_tensor unary, binary, or reduce");
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
