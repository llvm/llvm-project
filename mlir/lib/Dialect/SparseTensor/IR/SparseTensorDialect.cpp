//===- SparseTensorDialect.cpp - Sparse tensor dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.cpp.inc"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorTypes.cpp.inc"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// TensorDialect Attribute Methods.
//===----------------------------------------------------------------------===//

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

void SparseTensorDimSliceAttr::print(AsmPrinter &printer) const {
  printer << "(";
  printer << (getStaticOffset() ? std::to_string(*getStaticOffset()) : "?");
  printer << ", ";
  printer << (getStaticSize() ? std::to_string(*getStaticSize()) : "?");
  printer << ", ";
  printer << (getStaticStride() ? std::to_string(*getStaticStride()) : "?");
  printer << ")";
}

static ParseResult parseOptionalStaticSlice(int64_t &result,
                                            AsmParser &parser) {
  auto parseResult = parser.parseOptionalInteger(result);
  if (parseResult.has_value()) {
    if (parseResult.value().succeeded() && result < 0) {
      parser.emitError(
          parser.getCurrentLocation(),
          "expect positive value or ? for slice offset/size/stride");
      return failure();
    }
    return parseResult.value();
  }

  // Else, and '?' which represented dynamic slice
  result = SparseTensorDimSliceAttr::kDynamic;
  return parser.parseQuestion();
}

Attribute SparseTensorDimSliceAttr::parse(AsmParser &parser, Type type) {
  int64_t offset = -1, size = -1, stride = -1;

  if (failed(parser.parseLParen()) ||
      failed(parseOptionalStaticSlice(offset, parser)) ||
      failed(parser.parseComma()) ||
      failed(parseOptionalStaticSlice(size, parser)) ||
      failed(parser.parseComma()) ||
      failed(parseOptionalStaticSlice(stride, parser)) ||
      failed(parser.parseRParen()))
    return {};

  return parser.getChecked<SparseTensorDimSliceAttr>(parser.getContext(),
                                                     offset, size, stride);
}

LogicalResult
SparseTensorDimSliceAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 int64_t offset, int64_t size, int64_t stride) {
  if ((offset == SparseTensorDimSliceAttr::kDynamic || offset >= 0) &&
      (size == SparseTensorDimSliceAttr::kDynamic || size > 0) &&
      (stride == SparseTensorDimSliceAttr::kDynamic || stride > 0)) {
    return success();
  }
  return emitError()
         << "expect positive value or ? for slice offset/size/stride";
}

Type SparseTensorEncodingAttr::getPointerType() const {
  unsigned ptrWidth = getPointerBitWidth();
  Type indexType = IndexType::get(getContext());
  return ptrWidth ? IntegerType::get(getContext(), ptrWidth) : indexType;
}

Type SparseTensorEncodingAttr::getIndexType() const {
  unsigned idxWidth = getIndexBitWidth();
  Type indexType = IndexType::get(getContext());
  return idxWidth ? IntegerType::get(getContext(), idxWidth) : indexType;
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::withoutOrdering() const {
  return SparseTensorEncodingAttr::get(
      getContext(), getDimLevelType(), AffineMap(), AffineMap(),
      getPointerBitWidth(), getIndexBitWidth());
}

bool SparseTensorEncodingAttr::isAllDense() const {
  return llvm::all_of(getDimLevelType(), isDenseDLT);
}

bool SparseTensorEncodingAttr::hasIdDimOrdering() const {
  return !getDimOrdering() || getDimOrdering().isIdentity();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticDimSliceOffset(unsigned dim) const {
  return getDimSlices()[dim].getStaticOffset();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticDimSliceSize(unsigned dim) const {
  return getDimSlices()[dim].getStaticSize();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticDimSliceStride(unsigned dim) const {
  return getDimSlices()[dim].getStaticStride();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticLvlSliceOffset(unsigned lvl) const {
  return getStaticDimSliceOffset(toOrigDim(*this, lvl));
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticLvlSliceSize(unsigned lvl) const {
  return getStaticDimSliceSize(toOrigDim(*this, lvl));
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticLvlSliceStride(unsigned lvl) const {
  return getStaticDimSliceStride(toOrigDim(*this, lvl));
}

Attribute SparseTensorEncodingAttr::parse(AsmParser &parser, Type type) {
#define RETURN_ON_FAIL(stmt)                                                   \
  if (failed(stmt)) {                                                          \
    return {};                                                                 \
  }

  RETURN_ON_FAIL(parser.parseLess())
  RETURN_ON_FAIL(parser.parseLBrace())

  // Process the data from the parsed dictionary value into struct-like data.
  SmallVector<DimLevelType> dlt;
  SmallVector<SparseTensorDimSliceAttr> slices;
  AffineMap dimOrd = {};
  AffineMap higherOrd = {};
  unsigned ptr = 0;
  unsigned ind = 0;

  StringRef attrName;
  // Exactly 6 keys.
  SmallVector<StringRef, 6> keys = {"dimLevelType",   "dimOrdering",
                                    "higherOrdering", "pointerBitWidth",
                                    "indexBitWidth",  "slice"};
  while (succeeded(parser.parseOptionalKeyword(&attrName))) {
    if (!llvm::is_contained(keys, attrName)) {
      parser.emitError(parser.getNameLoc(), "unexpected key: ") << attrName;
      return {};
    }

    // Consume the `=` after keys
    RETURN_ON_FAIL(parser.parseEqual())
    if (attrName == "dimLevelType") {
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr));
      auto arrayAttr = attr.dyn_cast<ArrayAttr>();
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
          dlt.push_back(DimLevelType::Dense);
        } else if (strVal == "compressed") {
          dlt.push_back(DimLevelType::Compressed);
        } else if (strVal == "compressed-nu") {
          dlt.push_back(DimLevelType::CompressedNu);
        } else if (strVal == "compressed-no") {
          dlt.push_back(DimLevelType::CompressedNo);
        } else if (strVal == "compressed-nu-no") {
          dlt.push_back(DimLevelType::CompressedNuNo);
        } else if (strVal == "singleton") {
          dlt.push_back(DimLevelType::Singleton);
        } else if (strVal == "singleton-nu") {
          dlt.push_back(DimLevelType::SingletonNu);
        } else if (strVal == "singleton-no") {
          dlt.push_back(DimLevelType::SingletonNo);
        } else if (strVal == "singleton-nu-no") {
          dlt.push_back(DimLevelType::SingletonNuNo);
        } else {
          parser.emitError(parser.getNameLoc(),
                           "unexpected dimension level type: ")
              << strVal;
          return {};
        }
      }
    } else if (attrName == "dimOrdering") {
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))

      auto affineAttr = attr.dyn_cast<AffineMapAttr>();
      if (!affineAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an affine map for dimension ordering");
        return {};
      }
      dimOrd = affineAttr.getValue();
    } else if (attrName == "higherOrdering") {
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))

      auto affineAttr = attr.dyn_cast<AffineMapAttr>();
      if (!affineAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an affine map for higher ordering");
        return {};
      }
      higherOrd = affineAttr.getValue();
    } else if (attrName == "pointerBitWidth") {
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))

      auto intAttr = attr.dyn_cast<IntegerAttr>();
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an integral pointer bitwidth");
        return {};
      }
      ptr = intAttr.getInt();
    } else if (attrName == "indexBitWidth") {
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))

      auto intAttr = attr.dyn_cast<IntegerAttr>();
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an integral index bitwidth");
        return {};
      }
      ind = intAttr.getInt();
    } else if (attrName == "slice") {
      RETURN_ON_FAIL(parser.parseLSquare())
      // Dispatches to DimSliceAttr to skip mnemonic
      bool finished = false;
      while (auto attr = SparseTensorDimSliceAttr::parse(parser, nullptr)) {
        auto sliceAttr = attr.cast<SparseTensorDimSliceAttr>();
        slices.push_back(sliceAttr);
        if (parser.parseOptionalComma().failed()) {
          finished = true;
          break;
        }
      }
      // Wrong when parsing slices
      if (!finished)
        return {};
      RETURN_ON_FAIL(parser.parseRSquare())
    }

    // Only the last item can omit the comma
    if (parser.parseOptionalComma().failed())
      break;
  }

  RETURN_ON_FAIL(parser.parseRBrace())
  RETURN_ON_FAIL(parser.parseGreater())
#undef RETURN_ON_FAIL

  // Construct struct-like storage for attribute.
  return parser.getChecked<SparseTensorEncodingAttr>(
      parser.getContext(), dlt, dimOrd, higherOrd, ptr, ind, slices);
}

void SparseTensorEncodingAttr::print(AsmPrinter &printer) const {
  // Print the struct-like storage in dictionary fashion.
  printer << "<{ dimLevelType = [ ";
  for (unsigned i = 0, e = getDimLevelType().size(); i < e; i++) {
    printer << "\"" << toMLIRString(getDimLevelType()[i]) << "\"";
    if (i != e - 1)
      printer << ", ";
  }
  printer << " ]";
  // Print remaining members only for non-default values.
  if (!hasIdDimOrdering())
    printer << ", dimOrdering = affine_map<" << getDimOrdering() << ">";
  if (getHigherOrdering())
    printer << ", higherOrdering = affine_map<" << getHigherOrdering() << ">";
  if (getPointerBitWidth())
    printer << ", pointerBitWidth = " << getPointerBitWidth();
  if (getIndexBitWidth())
    printer << ", indexBitWidth = " << getIndexBitWidth();
  if (!getDimSlices().empty()) {
    printer << ", slice = [ ";
    llvm::interleaveComma(getDimSlices(), printer,
                          [&](SparseTensorDimSliceAttr attr) {
                            // Calls SparseTensorDimSliceAttr::print directly to
                            // skip mnemonic.
                            attr.print(printer);
                          });
    printer << " ]";
  }

  printer << " }>";
}

LogicalResult SparseTensorEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<DimLevelType> dimLevelType, AffineMap dimOrdering,
    AffineMap higherOrdering, unsigned pointerBitWidth, unsigned indexBitWidth,
    ArrayRef<SparseTensorDimSliceAttr> dimSlices) {
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
  if (higherOrdering) {
    if (higherOrdering.getNumDims() >= higherOrdering.getNumResults())
      return emitError() << "unexpected higher ordering mapping from "
                         << higherOrdering.getNumDims() << " to "
                         << higherOrdering.getNumResults();
    if (higherOrdering.getNumResults() != dimLevelType.size())
      return emitError() << "unexpected mismatch in higher ordering and "
                            "dimension level types size";
  }
  if (!dimSlices.empty() && dimSlices.size() != dimLevelType.size()) {
    return emitError() << "unexpected mismatch in dimension slices and "
                          "dimension level type size";
  }

  return success();
}

LogicalResult SparseTensorEncodingAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    function_ref<InFlightDiagnostic()> emitError) const {
  // Check structural integrity.
  if (failed(verify(emitError, getDimLevelType(), getDimOrdering(),
                    getHigherOrdering(), getPointerBitWidth(),
                    getIndexBitWidth(), getDimSlices())))
    return failure();
  // Check integrity with tensor type specifics. Dimension ordering is optional,
  // but we always should have dimension level types for the full rank.
  unsigned size = shape.size();
  if (size == 0)
    return emitError() << "expected non-scalar sparse tensor";
  if (getHigherOrdering()) {
    if (getHigherOrdering().getNumDims() != size)
      return emitError() << "expected an affine map of size " << size
                         << " for higher ordering";

    // TODO: verification of higher ordering contents

    size = getHigherOrdering().getNumResults(); // higher-order size!
  }
  if (getDimOrdering() && getDimOrdering().getNumResults() != size)
    return emitError() << "expected an affine map of size " << size
                       << " for dimension ordering";
  if (getDimLevelType().size() != size)
    return emitError() << "expected an array of size " << size
                       << " for dimension level types";
  return success();
}

//===----------------------------------------------------------------------===//
// Convenience Methods.
//===----------------------------------------------------------------------===//

SparseTensorEncodingAttr
mlir::sparse_tensor::getSparseTensorEncoding(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<SparseTensorEncodingAttr>();
  if (auto mdtp = type.dyn_cast<StorageSpecifierType>())
    return mdtp.getEncoding();
  return nullptr;
}

/// Returns true iff the given sparse tensor encoding attribute has a trailing
/// COO region starting at the given dimension.
static bool isCOOType(SparseTensorEncodingAttr enc, uint64_t s, bool isUnique) {
  uint64_t rank = enc.getDimLevelType().size();
  assert(s < rank && "Dimension out of bounds");
  if (!isCompressedDim(enc, s))
    return false;

  for (uint64_t i = s + 1; i < rank; ++i)
    if (!isSingletonDim(enc, i))
      return false;

  // If isUnique is true, then make sure that the last dimension level is
  // unique, that is, rank == 1 (unique the only compressed) and rank > 1
  // (unique on the last singleton).
  return !isUnique || isUniqueDLT(getDimLevelType(enc, rank - 1));
}

bool mlir::sparse_tensor::isUniqueCOOType(RankedTensorType tp) {
  SparseTensorEncodingAttr enc = getSparseTensorEncoding(tp);
  if (!enc)
    return false;

  return isCOOType(enc, 0, /*isUnique=*/true);
}

unsigned mlir::sparse_tensor::getCOOStart(SparseTensorEncodingAttr enc) {
  unsigned rank = enc.getDimLevelType().size();
  if (rank <= 1)
    return rank;

  // We only consider COO region with at least two dimensions for the purpose
  // of AOS storage optimization.
  for (unsigned r = 0; r < rank - 1; r++) {
    if (isCOOType(enc, r, /*isUnique=*/false))
      return r;
  }

  return rank;
}

uint64_t mlir::sparse_tensor::toOrigDim(SparseTensorEncodingAttr enc,
                                        uint64_t d) {
  if (enc) {
    auto order = enc.getDimOrdering();
    if (order) {
      assert(order.isPermutation());
      return order.getDimPosition(d);
    }
  }
  return d;
}

uint64_t mlir::sparse_tensor::toStoredDim(SparseTensorEncodingAttr enc,
                                          uint64_t d) {
  if (enc) {
    auto order = enc.getDimOrdering();
    if (order) {
      assert(order.isPermutation());
      auto maybePos =
          order.getResultPosition(getAffineDimExpr(d, enc.getContext()));
      assert(maybePos.has_value());
      return *maybePos;
    }
  }
  return d;
}

uint64_t mlir::sparse_tensor::toOrigDim(RankedTensorType type, uint64_t d) {
  assert(d < static_cast<uint64_t>(type.getRank()));
  return toOrigDim(getSparseTensorEncoding(type), d);
}

uint64_t mlir::sparse_tensor::toStoredDim(RankedTensorType type, uint64_t d) {
  assert(d < static_cast<uint64_t>(type.getRank()));
  return toStoredDim(getSparseTensorEncoding(type), d);
}

//===----------------------------------------------------------------------===//
// SparseTensorDialect Types.
//===----------------------------------------------------------------------===//

/// We normalized sparse tensor encoding attribute by always using
/// ordered/unique DLT such that "compressed-nu-no" and "compressed-nu" (as well
/// as other variants) lead to the same storage specifier type, and stripping
/// irrelevant fields that does not alter the sparse tensor memory layout.
static SparseTensorEncodingAttr
getNormalizedEncodingForSpecifier(SparseTensorEncodingAttr enc) {
  SmallVector<DimLevelType> dlts;
  for (auto dlt : enc.getDimLevelType())
    dlts.push_back(*getDimLevelType(*getLevelFormat(dlt), true, true));

  return SparseTensorEncodingAttr::get(
      enc.getContext(), dlts,
      AffineMap(), // dimOrdering (irrelavant to storage speicifer)
      AffineMap(), // highLvlOrdering (irrelavant to storage specifer)
      enc.getPointerBitWidth(), enc.getIndexBitWidth());
}

StorageSpecifierType
StorageSpecifierType::get(MLIRContext *ctx, SparseTensorEncodingAttr encoding) {
  return Base::get(ctx, getNormalizedEncodingForSpecifier(encoding));
}

IntegerType StorageSpecifierType::getSizesType() const {
  unsigned idxBitWidth =
      getEncoding().getIndexBitWidth() ? getEncoding().getIndexBitWidth() : 64u;
  unsigned ptrBitWidth =
      getEncoding().getIndexBitWidth() ? getEncoding().getIndexBitWidth() : 64u;

  return IntegerType::get(getContext(), std::max(idxBitWidth, ptrBitWidth));
}

Type StorageSpecifierType::getFieldType(StorageSpecifierKind kind,
                                        std::optional<unsigned> dim) const {
  if (kind != StorageSpecifierKind::ValMemSize)
    assert(dim);

  // Right now, we store every sizes metadata using the same size type.
  // TODO: the field size type can be defined dimensional wise after sparse
  // tensor encoding supports per dimension index/pointer bitwidth.
  return getSizesType();
}

Type StorageSpecifierType::getFieldType(StorageSpecifierKind kind,
                                        std::optional<APInt> dim) const {
  std::optional<unsigned> intDim = std::nullopt;
  if (dim)
    intDim = dim.value().getZExtValue();
  return getFieldType(kind, intDim);
}

//===----------------------------------------------------------------------===//
// SparseTensorDialect Operations.
//===----------------------------------------------------------------------===//

static LogicalResult isInBounds(uint64_t dim, Value tensor) {
  uint64_t rank = tensor.getType().cast<RankedTensorType>().getRank();
  if (dim >= rank)
    return failure();
  return success(); // in bounds
}

static LogicalResult isMatchingWidth(Value result, unsigned width) {
  Type etp = result.getType().cast<MemRefType>().getElementType();
  if ((width == 0 && etp.isIndex()) || (width > 0 && etp.isInteger(width)))
    return success();
  return failure();
}

static LogicalResult verifySparsifierGetterSetter(
    StorageSpecifierKind mdKind, std::optional<APInt> dim,
    TypedValue<StorageSpecifierType> md, Operation *op) {
  if (mdKind == StorageSpecifierKind::ValMemSize && dim) {
    return op->emitError(
        "redundant dimension argument for querying value memory size");
  }

  auto enc = md.getType().getEncoding();
  ArrayRef<DimLevelType> dlts = enc.getDimLevelType();
  unsigned rank = dlts.size();

  if (mdKind != StorageSpecifierKind::ValMemSize) {
    if (!dim)
      return op->emitError("missing dimension argument");

    unsigned d = dim.value().getZExtValue();
    if (d >= rank)
      return op->emitError("requested dimension out of bound");

    if (mdKind == StorageSpecifierKind::PtrMemSize && isSingletonDLT(dlts[d]))
      return op->emitError(
          "requested pointer memory size on a singleton level");
  }
  return success();
}

LogicalResult NewOp::verify() {
  if (getExpandSymmetry() &&
      getResult().getType().cast<RankedTensorType>().getRank() != 2)
    return emitOpError("expand_symmetry can only be used for 2D tensors");
  return success();
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
        if (shape1[d] != shape2[d] && shape2[d] != ShapedType::kDynamic)
          return emitError("unexpected conversion mismatch in dimension ") << d;
      return success();
    }
  }
  return emitError("unexpected type in convert");
}

OpFoldResult ConvertOp::fold(FoldAdaptor adaptor) {
  Type dstType = getType();
  // Fold trivial dense-to-dense convert and leave trivial sparse-to-sparse
  // convert for codegen to remove. This is because we use trivial
  // sparse-to-sparse convert to tell bufferization that the sparse codegen
  // will expand the tensor buffer into sparse tensor storage.
  if (!getSparseTensorEncoding(dstType) && dstType == getSource().getType())
    return getSource();
  return {};
}

LogicalResult ToPointersOp::verify() {
  auto e = getSparseTensorEncoding(getTensor().getType());
  if (failed(isInBounds(getDimension().getZExtValue(), getTensor())))
    return emitError("requested pointers dimension out of bounds");
  if (failed(isMatchingWidth(getResult(), e.getPointerBitWidth())))
    return emitError("unexpected type for pointers");
  return success();
}

LogicalResult ToIndicesOp::verify() {
  auto e = getSparseTensorEncoding(getTensor().getType());
  if (failed(isInBounds(getDimension().getZExtValue(), getTensor())))
    return emitError("requested indices dimension out of bounds");
  if (failed(isMatchingWidth(getResult(), e.getIndexBitWidth())))
    return emitError("unexpected type for indices");
  return success();
}

LogicalResult ToIndicesBufferOp::verify() {
  auto e = getSparseTensorEncoding(getTensor().getType());
  if (getCOOStart(e) >= e.getDimLevelType().size())
    return emitError("expected sparse tensor with a COO region");
  return success();
}

LogicalResult ToValuesOp::verify() {
  RankedTensorType ttp = getTensor().getType().cast<RankedTensorType>();
  MemRefType mtp = getResult().getType().cast<MemRefType>();
  if (ttp.getElementType() != mtp.getElementType())
    return emitError("unexpected mismatch in element types");
  return success();
}

LogicalResult GetStorageSpecifierOp::verify() {
  if (failed(verifySparsifierGetterSetter(getSpecifierKind(), getDim(),
                                          getSpecifier(), getOperation()))) {
    return failure();
  }

  // Checks the result type
  if (getSpecifier().getType().getFieldType(getSpecifierKind(), getDim()) !=
      getResult().getType()) {
    return emitError(
        "type mismatch between requested specifier field and result value");
  }
  return success();
}

template <typename SpecifierOp>
static SetStorageSpecifierOp getSpecifierSetDef(SpecifierOp op) {
  return op.getSpecifier().template getDefiningOp<SetStorageSpecifierOp>();
}

OpFoldResult GetStorageSpecifierOp::fold(FoldAdaptor adaptor) {
  StorageSpecifierKind kind = getSpecifierKind();
  std::optional<APInt> dim = getDim();
  for (auto op = getSpecifierSetDef(*this); op; op = getSpecifierSetDef(op))
    if (kind == op.getSpecifierKind() && dim == op.getDim())
      return op.getValue();
  return {};
}

LogicalResult SetStorageSpecifierOp::verify() {
  if (failed(verifySparsifierGetterSetter(getSpecifierKind(), getDim(),
                                          getSpecifier(), getOperation()))) {
    return failure();
  }

  // Checks the input type
  if (getSpecifier().getType().getFieldType(getSpecifierKind(), getDim()) !=
      getValue().getType()) {
    return emitError(
        "type mismatch between requested specifier field and input value");
  }
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
  if (!yield.getResult() || yield.getResult().getType() != outputType)
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
      if (ShapedType::isDynamic(dim))
        return emitError("Only statically-sized input tensors are supported.");
    }
  }

  if (concatDim >= rank)
    return emitError(llvm::formatv(
        "Failed to concatentate tensors with rank={0} on dimension={1}.", rank,
        concatDim));

  for (size_t i = 0, e = getInputs().size(); i < e; i++) {
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
      if (!ShapedType::isDynamic(dstDim)) {
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
      int64_t prev = dstDim;
      for (auto src : getInputs()) {
        auto d = src.getType().cast<RankedTensorType>().getShape()[i];
        if (!ShapedType::isDynamic(prev) && d != prev)
          return emitError("All dimensions (expect for the concatenating one) "
                           "should be equal.");
        prev = d;
      }
    }
  }

  return success();
}

LogicalResult InsertOp::verify() {
  RankedTensorType ttp = getTensor().getType().cast<RankedTensorType>();
  if (ttp.getRank() != static_cast<int64_t>(getIndices().size()))
    return emitOpError("incorrect number of indices");
  return success();
}

void PushBackOp::build(OpBuilder &builder, OperationState &result,
                       Value curSize, Value inBuffer, Value value) {
  build(builder, result, curSize, inBuffer, value, Value());
}

LogicalResult PushBackOp::verify() {
  Value n = getN();
  if (n) {
    auto nValue = dyn_cast_or_null<arith::ConstantIndexOp>(n.getDefiningOp());
    if (nValue && nValue.value() < 1)
      return emitOpError("n must be not less than 1");
  }
  return success();
}

LogicalResult CompressOp::verify() {
  RankedTensorType ttp = getTensor().getType().cast<RankedTensorType>();
  if (ttp.getRank() != 1 + static_cast<int64_t>(getIndices().size()))
    return emitOpError("incorrect number of indices");
  return success();
}

void ForeachOp::build(
    OpBuilder &builder, OperationState &result, Value tensor,
    function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)>
        bodyBuilder) {
  build(builder, result, tensor, std::nullopt, bodyBuilder);
}

void ForeachOp::build(
    OpBuilder &builder, OperationState &result, Value tensor,
    ValueRange initArgs,
    function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)>
        bodyBuilder) {
  build(builder, result, initArgs.getTypes(), tensor, initArgs);
  // Builds foreach body.
  if (!bodyBuilder)
    return;
  auto rtp = tensor.getType().cast<RankedTensorType>();
  int64_t rank = rtp.getRank();

  SmallVector<Type> blockArgTypes;
  // Starts with n index.
  std::fill_n(std::back_inserter(blockArgTypes), rank, builder.getIndexType());
  // Followed by one value.
  blockArgTypes.push_back(rtp.getElementType());
  // Followed by reduction variable.
  blockArgTypes.append(initArgs.getTypes().begin(), initArgs.getTypes().end());

  SmallVector<Location> blockArgLocs;
  std::fill_n(std::back_inserter(blockArgLocs), blockArgTypes.size(),
              tensor.getLoc());

  OpBuilder::InsertionGuard guard(builder);
  auto &region = *result.regions.front();
  Block *bodyBlock =
      builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
  bodyBuilder(builder, result.location,
              bodyBlock->getArguments().slice(0, rank),
              bodyBlock->getArguments()[rank],
              bodyBlock->getArguments().drop_front(rank + 1));
}

LogicalResult ForeachOp::verify() {
  auto t = getTensor().getType().cast<RankedTensorType>();
  auto args = getBody()->getArguments();

  if (static_cast<size_t>(t.getRank()) + 1 + getInitArgs().size() !=
      args.size())
    return emitError("Unmatched number of arguments in the block");

  if (getNumResults() != getInitArgs().size())
    return emitError("Mismatch in number of init arguments and results");

  if (getResultTypes() != getInitArgs().getTypes())
    return emitError("Mismatch in types of init arguments and results");

  auto yield = cast<YieldOp>(getBody()->getTerminator());
  if (yield.getNumOperands() != getNumResults() ||
      yield.getOperands().getTypes() != getResultTypes())
    return emitError("Mismatch in types of yield values and results");

  for (int64_t i = 0, e = t.getRank(); i < e; i++)
    if (args[i].getType() != IndexType::get(getContext()))
      emitError(
          llvm::formatv("Expecting Index type for argument at index {0}", i));

  auto elemTp = t.getElementType();
  auto valueTp = args[t.getRank()].getType();
  if (elemTp != valueTp)
    emitError(llvm::formatv("Unmatched element type between input tensor and "
                            "block argument, expected:{0}, got: {1}",
                            elemTp, valueTp));
  return success();
}

LogicalResult ReduceOp::verify() {
  Type inputType = getX().getType();
  LogicalResult regionResult = success();

  // Check correct number of block arguments and return type.
  Region &formula = getRegion();
  regionResult = verifyNumBlockArgs(this, formula, "reduce",
                                    TypeRange{inputType, inputType}, inputType);
  if (failed(regionResult))
    return regionResult;

  return success();
}

LogicalResult SelectOp::verify() {
  Builder b(getContext());

  Type inputType = getX().getType();
  Type boolType = b.getI1Type();
  LogicalResult regionResult = success();

  // Check correct number of block arguments and return type.
  Region &formula = getRegion();
  regionResult = verifyNumBlockArgs(this, formula, "select",
                                    TypeRange{inputType}, boolType);
  if (failed(regionResult))
    return regionResult;

  return success();
}

LogicalResult SortOp::verify() {
  if (getXs().empty())
    return emitError("need at least one xs buffer.");

  auto n = getN().getDefiningOp<arith::ConstantIndexOp>();

  Type xtp = getXs().front().getType().cast<MemRefType>().getElementType();
  auto checkTypes = [&](ValueRange operands,
                        bool checkEleType = true) -> LogicalResult {
    for (Value opnd : operands) {
      MemRefType mtp = opnd.getType().cast<MemRefType>();
      int64_t dim = mtp.getShape()[0];
      // We can't check the size of dynamic dimension at compile-time, but all
      // xs and ys should have a dimension not less than n at runtime.
      if (n && !ShapedType::isDynamic(dim) && dim < n.value())
        return emitError(llvm::formatv("xs and ys need to have a dimension >= n"
                                       ": {0} < {1}",
                                       dim, n.value()));

      if (checkEleType && xtp != mtp.getElementType())
        return emitError("mismatch xs element types");
    }
    return success();
  };

  LogicalResult result = checkTypes(getXs());
  if (failed(result))
    return result;

  if (n)
    return checkTypes(getYs(), false);

  return success();
}

LogicalResult SortCooOp::verify() {
  auto cn = getN().getDefiningOp<arith::ConstantIndexOp>();
  // We can't check the size of the buffers when n or buffer dimensions aren't
  // compile-time constants.
  if (!cn)
    return success();

  uint64_t n = cn.value();
  uint64_t nx = 1;
  if (auto nxAttr = getNxAttr()) {
    nx = nxAttr.getInt();
    if (nx < 1)
      emitError(llvm::formatv("Expected nx > 1, got {0}", nx));
  }
  uint64_t ny = 0;
  if (auto nyAttr = getNyAttr()) {
    ny = nyAttr.getInt();
  }

  auto checkDim = [&](Value v, uint64_t min, const char *message) {
    MemRefType tp = v.getType().cast<MemRefType>();
    int64_t dim = tp.getShape()[0];
    if (!ShapedType::isDynamic(dim) && dim < (int64_t)min) {
      emitError(llvm::formatv("{0} got {1} < {2}", message, dim, min));
    }
  };

  checkDim(getXy(), n * (nx + ny), "Expected dimension(xy) >= n * (nx + ny)");

  for (Value opnd : getYs()) {
    checkDim(opnd, n, "Expected dimension(y) >= n");
  }

  return success();
}

LogicalResult YieldOp::verify() {
  // Check for compatible parent.
  auto *parentOp = (*this)->getParentOp();
  if (isa<BinaryOp>(parentOp) || isa<UnaryOp>(parentOp) ||
      isa<ReduceOp>(parentOp) || isa<SelectOp>(parentOp) ||
      isa<ForeachOp>(parentOp))
    return success();

  return emitOpError("expected parent op to be sparse_tensor unary, binary, "
                     "reduce, select or foreach");
}

//===----------------------------------------------------------------------===//
// TensorDialect Methods.
//===----------------------------------------------------------------------===//

void SparseTensorDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/SparseTensor/IR/SparseTensorAttrDefs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/SparseTensor/IR/SparseTensorTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SparseTensor/IR/SparseTensorOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/SparseTensor/IR/SparseTensorOps.cpp.inc"

#include "mlir/Dialect/SparseTensor/IR/SparseTensorOpsDialect.cpp.inc"
