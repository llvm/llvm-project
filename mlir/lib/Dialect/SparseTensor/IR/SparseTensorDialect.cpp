//===- SparseTensorDialect.cpp - Sparse tensor dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
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
// Additional convenience methods.
//===----------------------------------------------------------------------===//

/// Gets the dimension-rank of the type of some `T`.  (In particular
/// this is only used for `Value` and `TypedValue<RankedTensorType>`.)
template <typename T>
static inline Dimension getDimRank(T t) {
  return getRankedTensorType(t).getRank();
}

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

Type mlir::sparse_tensor::detail::getIntegerOrIndexType(MLIRContext *ctx,
                                                        unsigned bitwidth) {
  if (bitwidth)
    return IntegerType::get(ctx, bitwidth);
  return IndexType::get(ctx);
}

Type SparseTensorEncodingAttr::getPointerType() const {
  return detail::getIntegerOrIndexType(getContext(), getPointerBitWidth());
}

Type SparseTensorEncodingAttr::getIndexType() const {
  return detail::getIntegerOrIndexType(getContext(), getIndexBitWidth());
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::withoutOrdering() const {
  return SparseTensorEncodingAttr::get(
      getContext(), getDimLevelType(), AffineMap(), AffineMap(),
      getPointerBitWidth(), getIndexBitWidth());
}

bool SparseTensorEncodingAttr::isAllDense() const {
  return !getImpl() || llvm::all_of(getDimLevelType(), isDenseDLT);
}

bool SparseTensorEncodingAttr::isAllOrdered() const {
  return !getImpl() || llvm::all_of(getDimLevelType(), isOrderedDLT);
}

bool SparseTensorEncodingAttr::hasIdDimOrdering() const {
  return !getImpl() || !getDimOrdering() || getDimOrdering().isIdentity();
}

Level SparseTensorEncodingAttr::getLvlRank() const {
  assert(getImpl() && "Uninitialized SparseTensorEncodingAttr");
  return getDimLevelType().size();
}

DimLevelType SparseTensorEncodingAttr::getLvlType(Level l) const {
  if (!getImpl())
    return DimLevelType::Dense;
  assert(l < getLvlRank() && "Level is out of bounds");
  return getDimLevelType()[l];
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticDimSliceOffset(Dimension dim) const {
  return getDimSlices()[dim].getStaticOffset();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticDimSliceSize(Dimension dim) const {
  return getDimSlices()[dim].getStaticSize();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticDimSliceStride(Dimension dim) const {
  return getDimSlices()[dim].getStaticStride();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticLvlSliceOffset(Level lvl) const {
  // FIXME: `toOrigDim` is deprecated.
  return getStaticDimSliceOffset(toOrigDim(*this, lvl));
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticLvlSliceSize(Level lvl) const {
  // FIXME: `toOrigDim` is deprecated.
  return getStaticDimSliceSize(toOrigDim(*this, lvl));
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticLvlSliceStride(Level lvl) const {
  // FIXME: `toOrigDim` is deprecated.
  return getStaticDimSliceStride(toOrigDim(*this, lvl));
}

const static DimLevelType validDLTs[] = {
    DimLevelType::Dense,          DimLevelType::Compressed,
    DimLevelType::CompressedNu,   DimLevelType::CompressedNo,
    DimLevelType::CompressedNuNo, DimLevelType::Singleton,
    DimLevelType::SingletonNu,    DimLevelType::SingletonNo,
    DimLevelType::SingletonNuNo};

static std::optional<DimLevelType> parseDLT(StringRef str) {
  for (DimLevelType dlt : validDLTs)
    if (str == toMLIRString(dlt))
      return dlt;
  return std::nullopt;
}

Attribute SparseTensorEncodingAttr::parse(AsmParser &parser, Type type) {
#define RETURN_ON_FAIL(stmt)                                                   \
  if (failed(stmt)) {                                                          \
    return {};                                                                 \
  }
#define ERROR_IF(COND, MSG)                                                    \
  if (COND) {                                                                  \
    parser.emitError(parser.getNameLoc(), MSG);                                \
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
      ERROR_IF(!arrayAttr, "expected an array for dimension level types")
      for (auto i : arrayAttr) {
        auto strAttr = i.dyn_cast<StringAttr>();
        ERROR_IF(!strAttr, "expected a string value in dimension level types")
        auto strVal = strAttr.getValue();
        if (auto optDLT = parseDLT(strVal)) {
          dlt.push_back(optDLT.value());
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
      ERROR_IF(!affineAttr, "expected an affine map for dimension ordering")
      dimOrd = affineAttr.getValue();
    } else if (attrName == "higherOrdering") {
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))
      auto affineAttr = attr.dyn_cast<AffineMapAttr>();
      ERROR_IF(!affineAttr, "expected an affine map for higher ordering")
      higherOrd = affineAttr.getValue();
    } else if (attrName == "pointerBitWidth") {
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))
      auto intAttr = attr.dyn_cast<IntegerAttr>();
      ERROR_IF(!intAttr, "expected an integral pointer bitwidth")
      ptr = intAttr.getInt();
    } else if (attrName == "indexBitWidth") {
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))
      auto intAttr = attr.dyn_cast<IntegerAttr>();
      ERROR_IF(!intAttr, "expected an integral index bitwidth")
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
#undef ERROR_IF
#undef RETURN_ON_FAIL

  // Construct struct-like storage for attribute.
  return parser.getChecked<SparseTensorEncodingAttr>(
      parser.getContext(), dlt, dimOrd, higherOrd, ptr, ind, slices);
}

void SparseTensorEncodingAttr::print(AsmPrinter &printer) const {
  // Print the struct-like storage in dictionary fashion.
  printer << "<{ dimLevelType = [ ";
  llvm::interleaveComma(getDimLevelType(), printer, [&](DimLevelType dlt) {
    printer << "\"" << toMLIRString(dlt) << "\"";
  });
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
  // Before we can check that the level-rank is consistent/coherent
  // across all fields, we need to define it.  The source-of-truth for
  // the `getLvlRank` method is the length of the level-types array,
  // since it must always be provided and have full rank; therefore we
  // use that same source-of-truth here.
  const Level lvlRank = dimLevelType.size();
  if (lvlRank == 0)
    return emitError() << "expected a non-empty array for level types";
  if (dimOrdering) {
    if (!dimOrdering.isPermutation())
      return emitError()
             << "expected a permutation affine map for dimension ordering";
    if (dimOrdering.getNumResults() != lvlRank)
      return emitError() << "unexpected mismatch in ordering and dimension "
                            "level types size";
  }
  if (higherOrdering) {
    if (higherOrdering.getNumDims() >= higherOrdering.getNumResults())
      return emitError() << "unexpected higher ordering mapping from "
                         << higherOrdering.getNumDims() << " to "
                         << higherOrdering.getNumResults();
    if (higherOrdering.getNumResults() != lvlRank)
      return emitError() << "unexpected mismatch in higher ordering and "
                            "dimension level types size";
  }
  if (!dimSlices.empty() && dimSlices.size() != lvlRank) {
    return emitError() << "unexpected mismatch in dimension slices and "
                          "dimension level type size";
  }
  return success();
}

#define RETURN_FAILURE_IF_FAILED(X)                                            \
  if (failed(X)) {                                                             \
    return failure();                                                          \
  }

LogicalResult SparseTensorEncodingAttr::verifyEncoding(
    ArrayRef<DynSize> dimShape, Type elementType,
    function_ref<InFlightDiagnostic()> emitError) const {
  // Check structural integrity.  In particular, this ensures that the
  // level-rank is coherent across all the fields.
  RETURN_FAILURE_IF_FAILED(verify(
      emitError, getDimLevelType(), getDimOrdering(), getHigherOrdering(),
      getPointerBitWidth(), getIndexBitWidth(), getDimSlices()))
  // Check integrity with tensor type specifics.  In particular, we
  // need only check that the dimension-rank of the tensor agrees with
  // the dimension-rank of the encoding.
  const Dimension dimRank = dimShape.size();
  if (dimRank == 0)
    return emitError() << "expected non-scalar sparse tensor";
  if (const auto higherOrdering = getHigherOrdering()) {
    if (higherOrdering.getNumDims() != dimRank)
      return emitError() << "expected an affine map with " << dimRank
                         << " dimensions for higher ordering";
    // TODO: verification of higher ordering contents
  } else if (dimRank != getLvlRank()) {
    return emitError() << "expected an array of size " << dimRank
                       << " for dimension level types";
  }
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
/// COO region starting at the given level.
static bool isCOOType(SparseTensorEncodingAttr enc, Level startLvl,
                      bool isUnique) {
  if (!enc || !enc.isCompressedLvl(startLvl))
    return false;
  const Level lvlRank = enc.getLvlRank();
  for (Level l = startLvl + 1; l < lvlRank; ++l)
    if (!enc.isSingletonLvl(l))
      return false;
  // If isUnique is true, then make sure that the last level is unique,
  // that is, lvlRank == 1 (unique the only compressed) and lvlRank > 1
  // (unique on the last singleton).
  return !isUnique || enc.isUniqueLvl(lvlRank - 1);
}

bool mlir::sparse_tensor::isUniqueCOOType(TensorType tp) {
  return isCOOType(getSparseTensorEncoding(tp), 0, /*isUnique=*/true);
}

Level mlir::sparse_tensor::getCOOStart(SparseTensorEncodingAttr enc) {
  // We only consider COO region with at least two levels for the purpose
  // of AOS storage optimization.
  const Level lvlRank = enc.getLvlRank();
  if (lvlRank > 1)
    for (Level l = 0; l < lvlRank - 1; l++)
      if (isCOOType(enc, l, /*isUnique=*/false))
        return l;
  return lvlRank;
}

// Helpers to setup a COO type.
RankedTensorType sparse_tensor::getCOOFromTypeWithOrdering(RankedTensorType rtt,
                                                           AffineMap lvlPerm,
                                                           bool ordered) {
  const SparseTensorType src(rtt);
  // The dim-rank of the source `RankedTensorType` is used as the lvl-rank
  // of the result `RankedTensorType`.  This follows from the fact that the
  // result's encoding has the default higher-ordering (hence the result's
  // lvl-rank equals its dim-rank).  We don't need to assert that `lvlRank`
  // agrees with the size of `lvlPerm` because that will be verified by
  // `STEA::get`.
  const Level lvlRank = src.getDimRank();
  SmallVector<DimLevelType> lvlTypes;

  // An unordered and non-unique compressed level at beginning.
  // If this is also the last level, then it is unique.
  lvlTypes.push_back(
      *getDimLevelType(LevelFormat::Compressed, ordered, lvlRank == 1));
  if (lvlRank > 1) {
    // TODO: it is actually ordered at the level for ordered input.
    // Followed by unordered non-unique n-2 singleton levels.
    std::fill_n(std::back_inserter(lvlTypes), lvlRank - 2,
                *getDimLevelType(LevelFormat::Singleton, ordered, false));
    // Ends by a unique singleton level unless the lvlRank is 1.
    lvlTypes.push_back(*getDimLevelType(LevelFormat::Singleton, ordered, true));
  }

  // TODO: Maybe pick the bitwidth based on input/output tensors (probably the
  // largest one among them) in the original operation instead of using the
  // default value.
  unsigned pointerBitWidth = src.getPointerBitWidth();
  unsigned indexBitWidth = src.getIndexBitWidth();
  auto enc = SparseTensorEncodingAttr::get(src.getContext(), lvlTypes, lvlPerm,
                                           AffineMap(), pointerBitWidth,
                                           indexBitWidth);
  return RankedTensorType::get(src.getDimShape(), src.getElementType(), enc);
}

RankedTensorType sparse_tensor::getCOOFromType(RankedTensorType src,
                                               bool ordered) {
  return getCOOFromTypeWithOrdering(
      src, AffineMap::getMultiDimIdentityMap(src.getRank(), src.getContext()),
      ordered);
}

// TODO: Remove this definition once all use-sites have been fixed to
// properly handle non-permutations.
Dimension mlir::sparse_tensor::toOrigDim(SparseTensorEncodingAttr enc,
                                         Level l) {
  if (enc) {
    auto order = enc.getDimOrdering();
    if (order) {
      assert(order.isPermutation());
      return order.getDimPosition(l);
    }
  }
  return l;
}

// TODO: Remove this definition once all use-sites have been fixed to
// properly handle non-permutations.
Level mlir::sparse_tensor::toStoredDim(SparseTensorEncodingAttr enc,
                                       Dimension d) {
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

// TODO: Remove this definition once all use-sites have been fixed to
// properly handle non-permutations.
Dimension mlir::sparse_tensor::toOrigDim(RankedTensorType type, Level l) {
  const auto enc = getSparseTensorEncoding(type);
  assert(l < enc.getLvlRank());
  return toOrigDim(enc, l);
}

// TODO: Remove this definition once all use-sites have been fixed to
// properly handle non-permutations.
Level mlir::sparse_tensor::toStoredDim(RankedTensorType type, Dimension d) {
  assert(d < static_cast<Dimension>(type.getRank()));
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
      enc.getPointerBitWidth(), enc.getIndexBitWidth(),
      // FIXME: we should keep the slice information, for now it is okay as only
      // constant can be used for slice
      ArrayRef<SparseTensorDimSliceAttr>{} /*enc.getDimSlices()*/);
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

// FIXME: see note [CLARIFY_DIM_LVL] in
// "lib/Dialect/SparseTensor/Transforms/SparseTensorStorageLayout.h"
Type StorageSpecifierType::getFieldType(StorageSpecifierKind kind,
                                        std::optional<unsigned> dim) const {
  if (kind != StorageSpecifierKind::ValMemSize)
    assert(dim);

  // Right now, we store every sizes metadata using the same size type.
  // TODO: the field size type can be defined dimensional wise after sparse
  // tensor encoding supports per dimension index/pointer bitwidth.
  return getSizesType();
}

// FIXME: see note [CLARIFY_DIM_LVL] in
// "lib/Dialect/SparseTensor/Transforms/SparseTensorStorageLayout.h"
Type StorageSpecifierType::getFieldType(StorageSpecifierKind kind,
                                        std::optional<APInt> dim) const {
  return getFieldType(kind, dim ? std::optional(dim.value().getZExtValue())
                                : std::nullopt);
}

//===----------------------------------------------------------------------===//
// SparseTensorDialect Operations.
//===----------------------------------------------------------------------===//

static LogicalResult dimIsInBounds(Dimension dim, Value tensor) {
  return success(dim < getDimRank(tensor));
}

static LogicalResult isMatchingWidth(Value result, unsigned width) {
  const Type etp = getMemRefType(result).getElementType();
  return success(width == 0 ? etp.isIndex() : etp.isInteger(width));
}

static LogicalResult verifySparsifierGetterSetter(
    StorageSpecifierKind mdKind, std::optional<APInt> lvl,
    TypedValue<StorageSpecifierType> md, Operation *op) {
  if (mdKind == StorageSpecifierKind::ValMemSize && lvl) {
    return op->emitError(
        "redundant level argument for querying value memory size");
  }

  const auto enc = md.getType().getEncoding();
  const Level lvlRank = enc.getLvlRank();

  if (mdKind != StorageSpecifierKind::ValMemSize) {
    if (!lvl)
      return op->emitError("missing level argument");

    const Level l = lvl.value().getZExtValue();
    if (l >= lvlRank)
      return op->emitError("requested level out of bound");

    if (mdKind == StorageSpecifierKind::PtrMemSize && enc.isSingletonLvl(l))
      return op->emitError(
          "requested pointer memory size on a singleton level");
  }
  return success();
}

LogicalResult NewOp::verify() {
  if (getExpandSymmetry() && getDimRank(getResult()) != 2)
    return emitOpError("expand_symmetry can only be used for 2D tensors");
  return success();
}

static LogicalResult verifyPackUnPack(Operation *op, TensorType cooTp,
                                      TensorType dataTp, TensorType idxTp) {
  if (!isUniqueCOOType(cooTp))
    return op->emitError("must operate on a COO tensor");

  auto enc = getSparseTensorEncoding(cooTp);
  if (idxTp.getElementType() != enc.getIndexType() ||
      dataTp.getElementType() != cooTp.getElementType())
    return op->emitError("unmatched type between input and output");

  auto dNOE = dataTp.getShape()[0];
  auto iNOE = idxTp.getShape()[0];
  if (!ShapedType::isDynamic(dNOE) && !ShapedType::isDynamic(iNOE) &&
      dNOE != iNOE)
    return op->emitError("unmatched number of elements in data and indices");

  // A tensor<?xNxi32> for indices means the input COO is rank N
  auto inRank = idxTp.getShape()[1];
  auto ouRank = cooTp.getRank();
  if (!ShapedType::isDynamic(inRank) && inRank != ouRank)
    return op->emitError("unmatched rank between input and output");

  return success();
}

LogicalResult PackOp::verify() {
  TensorType dataTp = getData().getType(), idxTp = getIndices().getType();
  TensorType retTp = getResult().getType();

  if (!retTp.hasStaticShape() || !dataTp.hasStaticShape() ||
      !idxTp.hasStaticShape())
    return emitError("all input types must be statically shaped");

  return verifyPackUnPack(*this, retTp, dataTp, idxTp);
}

LogicalResult UnpackOp::verify() {
  TensorType dataTp = getData().getType(), idxTp = getIndices().getType();
  TensorType srcTp = getTensor().getType();

  return verifyPackUnPack(*this, srcTp, dataTp, idxTp);
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
      for (Dimension d = 0, dimRank = tp1.getRank(); d < dimRank; d++)
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
  // FIXME: there seems to be some dim/lvl confusion here.
  if (failed(dimIsInBounds(getDimension().getZExtValue(), getTensor())))
    return emitError("requested pointers dimension out of bounds");
  if (failed(isMatchingWidth(getResult(), e.getPointerBitWidth())))
    return emitError("unexpected type for pointers");
  return success();
}

LogicalResult ToIndicesOp::verify() {
  auto e = getSparseTensorEncoding(getTensor().getType());
  // FIXME: there seems to be some dim/lvl confusion here.
  if (failed(dimIsInBounds(getDimension().getZExtValue(), getTensor())))
    return emitError("requested indices dimension out of bounds");
  if (failed(isMatchingWidth(getResult(), e.getIndexBitWidth())))
    return emitError("unexpected type for indices");
  return success();
}

LogicalResult ToIndicesBufferOp::verify() {
  auto e = getSparseTensorEncoding(getTensor().getType());
  if (getCOOStart(e) >= e.getLvlRank())
    return emitError("expected sparse tensor with a COO region");
  return success();
}

LogicalResult ToValuesOp::verify() {
  auto ttp = getRankedTensorType(getTensor());
  auto mtp = getMemRefType(getResult());
  if (ttp.getElementType() != mtp.getElementType())
    return emitError("unexpected mismatch in element types");
  return success();
}

LogicalResult GetStorageSpecifierOp::verify() {
  RETURN_FAILURE_IF_FAILED(verifySparsifierGetterSetter(
      getSpecifierKind(), getDim(), getSpecifier(), getOperation()))
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
  RETURN_FAILURE_IF_FAILED(verifySparsifierGetterSetter(
      getSpecifierKind(), getDim(), getSpecifier(), getOperation()))
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
  if (!overlap.empty()) {
    RETURN_FAILURE_IF_FAILED(verifyNumBlockArgs(
        this, overlap, "overlap", TypeRange{leftType, rightType}, outputType))
  }
  if (!left.empty()) {
    RETURN_FAILURE_IF_FAILED(
        verifyNumBlockArgs(this, left, "left", TypeRange{leftType}, outputType))
  } else if (getLeftIdentity()) {
    if (leftType != outputType)
      return emitError("left=identity requires first argument to have the same "
                       "type as the output");
  }
  if (!right.empty()) {
    RETURN_FAILURE_IF_FAILED(verifyNumBlockArgs(
        this, right, "right", TypeRange{rightType}, outputType))
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

  // Check correct number of block arguments and return type for each
  // non-empty region.
  Region &present = getPresentRegion();
  if (!present.empty()) {
    RETURN_FAILURE_IF_FAILED(verifyNumBlockArgs(
        this, present, "present", TypeRange{inputType}, outputType))
  }
  Region &absent = getAbsentRegion();
  if (!absent.empty()) {
    RETURN_FAILURE_IF_FAILED(
        verifyNumBlockArgs(this, absent, "absent", TypeRange{}, outputType))
  }
  return success();
}

LogicalResult ConcatenateOp::verify() {
  const auto dstTp = getSparseTensorType(*this);
  const Dimension concatDim = getDimension().getZExtValue();
  const Dimension dimRank = dstTp.getDimRank();

  if (getInputs().size() <= 1)
    return emitError("Need at least two tensors to concatenate.");

  if (concatDim >= dimRank)
    return emitError(llvm::formatv(
        "Concat-dimension is out of bounds for dimension-rank ({0} >= {1})",
        concatDim, dimRank));

  for (const auto &it : llvm::enumerate(getInputs())) {
    const auto i = it.index();
    const auto srcTp = getSparseTensorType(it.value());
    if (srcTp.hasDynamicDimShape())
      return emitError(llvm::formatv("Input tensor ${0} has dynamic shape", i));
    const Dimension srcDimRank = srcTp.getDimRank();
    if (srcDimRank != dimRank)
      return emitError(
          llvm::formatv("Input tensor ${0} has a different rank (rank={1}) "
                        "from the output tensor (rank={2}).",
                        i, srcDimRank, dimRank));
  }

  for (Dimension d = 0; d < dimRank; d++) {
    const DynSize dstSh = dstTp.getDimShape()[d];
    if (d == concatDim) {
      if (!ShapedType::isDynamic(dstSh)) {
        // If we reach here, then all inputs have static shapes.  So we
        // can use `getDimShape()[d]` instead of `*getDynamicDimSize(d)`
        // to avoid redundant assertions in the loop.
        StaticSize sumSz = 0;
        for (const auto src : getInputs())
          sumSz += getSparseTensorType(src).getDimShape()[d];
        // If all dimension are statically known, the sum of all the input
        // dimensions should be equal to the output dimension.
        if (sumSz != dstSh)
          return emitError(
              "The concatenation dimension of the output tensor should be the "
              "sum of all the concatenation dimensions of the input tensors.");
      }
    } else {
      DynSize prev = dstSh;
      for (const auto src : getInputs()) {
        const auto sh = getSparseTensorType(src).getDimShape()[d];
        if (!ShapedType::isDynamic(prev) && sh != prev)
          return emitError("All dimensions (expect for the concatenating one) "
                           "should be equal.");
        prev = sh;
      }
    }
  }

  return success();
}

LogicalResult InsertOp::verify() {
  if (getDimRank(getTensor()) != static_cast<Dimension>(getIndices().size()))
    return emitOpError("incorrect number of indices");
  return success();
}

void PushBackOp::build(OpBuilder &builder, OperationState &result,
                       Value curSize, Value inBuffer, Value value) {
  build(builder, result, curSize, inBuffer, value, Value());
}

LogicalResult PushBackOp::verify() {
  if (Value n = getN()) {
    auto nValue = dyn_cast_or_null<arith::ConstantIndexOp>(n.getDefiningOp());
    if (nValue && nValue.value() < 1)
      return emitOpError("n must be not less than 1");
  }
  return success();
}

LogicalResult CompressOp::verify() {
  if (getDimRank(getTensor()) !=
      1 + static_cast<Dimension>(getIndices().size()))
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
  const auto stt = getSparseTensorType(tensor);
  const Dimension dimRank = stt.getDimRank();

  // Starts with `dimRank`-many indices.
  SmallVector<Type> blockArgTypes(dimRank, builder.getIndexType());
  // Followed by one value.
  blockArgTypes.push_back(stt.getElementType());
  // Followed by the reduction variables.
  blockArgTypes.append(initArgs.getTypes().begin(), initArgs.getTypes().end());

  SmallVector<Location> blockArgLocs(blockArgTypes.size(), tensor.getLoc());

  OpBuilder::InsertionGuard guard(builder);
  auto &region = *result.regions.front();
  Block *bodyBlock =
      builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
  bodyBuilder(builder, result.location,
              bodyBlock->getArguments().slice(0, dimRank),
              bodyBlock->getArguments()[dimRank],
              bodyBlock->getArguments().drop_front(dimRank + 1));
}

LogicalResult ForeachOp::verify() {
  const auto t = getSparseTensorType(getTensor());
  const Dimension dimRank = t.getDimRank();
  const auto args = getBody()->getArguments();

  if (static_cast<size_t>(dimRank) + 1 + getInitArgs().size() != args.size())
    return emitError("Unmatched number of arguments in the block");

  if (getNumResults() != getInitArgs().size())
    return emitError("Mismatch in number of init arguments and results");

  if (getResultTypes() != getInitArgs().getTypes())
    return emitError("Mismatch in types of init arguments and results");

  // Cannot mark this const, because the getters aren't.
  auto yield = cast<YieldOp>(getBody()->getTerminator());
  if (yield.getNumOperands() != getNumResults() ||
      yield.getOperands().getTypes() != getResultTypes())
    return emitError("Mismatch in types of yield values and results");

  const auto iTp = IndexType::get(getContext());
  for (Dimension d = 0; d < dimRank; d++)
    if (args[d].getType() != iTp)
      emitError(
          llvm::formatv("Expecting Index type for argument at index {0}", d));

  const auto elemTp = t.getElementType();
  const auto valueTp = args[dimRank].getType();
  if (elemTp != valueTp)
    emitError(llvm::formatv("Unmatched element type between input tensor and "
                            "block argument, expected:{0}, got: {1}",
                            elemTp, valueTp));
  return success();
}

LogicalResult ReduceOp::verify() {
  Type inputType = getX().getType();
  // Check correct number of block arguments and return type.
  Region &formula = getRegion();
  RETURN_FAILURE_IF_FAILED(verifyNumBlockArgs(
      this, formula, "reduce", TypeRange{inputType, inputType}, inputType))
  return success();
}

LogicalResult SelectOp::verify() {
  Builder b(getContext());
  Type inputType = getX().getType();
  Type boolType = b.getI1Type();
  // Check correct number of block arguments and return type.
  Region &formula = getRegion();
  RETURN_FAILURE_IF_FAILED(verifyNumBlockArgs(this, formula, "select",
                                              TypeRange{inputType}, boolType))
  return success();
}

LogicalResult SortOp::verify() {
  if (getXs().empty())
    return emitError("need at least one xs buffer.");

  auto n = getN().getDefiningOp<arith::ConstantIndexOp>();

  Type xtp = getMemRefType(getXs().front()).getElementType();
  auto checkTypes = [&](ValueRange operands,
                        bool checkEleType = true) -> LogicalResult {
    for (Value opnd : operands) {
      auto mtp = getMemRefType(opnd);
      const DynSize sh = mtp.getShape()[0];
      // We can't check the size of dynamic dimension at compile-time, but all
      // xs and ys should have a dimension not less than n at runtime.
      if (n && !ShapedType::isDynamic(sh) && sh < n.value())
        return emitError(llvm::formatv("xs and ys need to have a dimension >= n"
                                       ": {0} < {1}",
                                       sh, n.value()));

      if (checkEleType && xtp != mtp.getElementType())
        return emitError("mismatch xs element types");
    }
    return success();
  };
  RETURN_FAILURE_IF_FAILED(checkTypes(getXs()))
  return n ? checkTypes(getYs(), false) : success();
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

  // FIXME: update the types of variables used in expressions bassed as
  // the `minSize` argument, to avoid implicit casting at the callsites
  // of this lambda.
  const auto checkDim = [&](Value v, StaticSize minSize, const char *message) {
    const DynSize sh = getMemRefType(v).getShape()[0];
    if (!ShapedType::isDynamic(sh) && sh < minSize)
      emitError(llvm::formatv("{0} got {1} < {2}", message, sh, minSize));
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

#undef RETURN_FAILURE_IF_FAILED

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
