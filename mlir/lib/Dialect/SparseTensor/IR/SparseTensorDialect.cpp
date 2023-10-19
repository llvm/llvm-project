//===- SparseTensorDialect.cpp - Sparse tensor dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "Detail/DimLvlMapParser.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorStorageLayout.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

static constexpr bool acceptBitWidth(unsigned bitWidth) {
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

//===----------------------------------------------------------------------===//
// StorageLayout
//===----------------------------------------------------------------------===//

static constexpr Level kInvalidLevel = -1u;
static constexpr Level kInvalidFieldIndex = -1u;
static constexpr FieldIndex kDataFieldStartingIdx = 0;

void StorageLayout::foreachField(
    llvm::function_ref<bool(FieldIndex, SparseTensorFieldKind, Level,
                            DimLevelType)>
        callback) const {
#define RETURN_ON_FALSE(fidx, kind, lvl, dlt)                                  \
  if (!(callback(fidx, kind, lvl, dlt)))                                       \
    return;

  const auto lvlTypes = enc.getLvlTypes();
  const Level lvlRank = enc.getLvlRank();
  const Level cooStart = getCOOStart(enc);
  const Level end = cooStart == lvlRank ? cooStart : cooStart + 1;
  FieldIndex fieldIdx = kDataFieldStartingIdx;
  // Per-level storage.
  for (Level l = 0; l < end; l++) {
    const auto dlt = lvlTypes[l];
    if (isDLTWithPos(dlt)) {
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::PosMemRef, l, dlt);
    }
    if (isDLTWithCrd(dlt)) {
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::CrdMemRef, l, dlt);
    }
  }
  // The values array.
  RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::ValMemRef, kInvalidLevel,
                  DimLevelType::Undef);

  // Put metadata at the end.
  RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::StorageSpec, kInvalidLevel,
                  DimLevelType::Undef);

#undef RETURN_ON_FALSE
}

void sparse_tensor::foreachFieldAndTypeInSparseTensor(
    SparseTensorType stt,
    llvm::function_ref<bool(Type, FieldIndex, SparseTensorFieldKind, Level,
                            DimLevelType)>
        callback) {
  assert(stt.hasEncoding());
  // Construct the basic types.
  const Type crdType = stt.getCrdType();
  const Type posType = stt.getPosType();
  const Type eltType = stt.getElementType();

  const Type specType = StorageSpecifierType::get(stt.getEncoding());
  // memref<? x pos>  positions
  const Type posMemType = MemRefType::get({ShapedType::kDynamic}, posType);
  // memref<? x crd>  coordinates
  const Type crdMemType = MemRefType::get({ShapedType::kDynamic}, crdType);
  // memref<? x eltType> values
  const Type valMemType = MemRefType::get({ShapedType::kDynamic}, eltType);

  StorageLayout(stt).foreachField(
      [specType, posMemType, crdMemType, valMemType,
       callback](FieldIndex fieldIdx, SparseTensorFieldKind fieldKind,
                 Level lvl, DimLevelType dlt) -> bool {
        switch (fieldKind) {
        case SparseTensorFieldKind::StorageSpec:
          return callback(specType, fieldIdx, fieldKind, lvl, dlt);
        case SparseTensorFieldKind::PosMemRef:
          return callback(posMemType, fieldIdx, fieldKind, lvl, dlt);
        case SparseTensorFieldKind::CrdMemRef:
          return callback(crdMemType, fieldIdx, fieldKind, lvl, dlt);
        case SparseTensorFieldKind::ValMemRef:
          return callback(valMemType, fieldIdx, fieldKind, lvl, dlt);
        };
        llvm_unreachable("unrecognized field kind");
      });
}

unsigned StorageLayout::getNumFields() const {
  unsigned numFields = 0;
  foreachField([&numFields](FieldIndex, SparseTensorFieldKind, Level,
                            DimLevelType) -> bool {
    numFields++;
    return true;
  });
  return numFields;
}

unsigned StorageLayout::getNumDataFields() const {
  unsigned numFields = 0; // one value memref
  foreachField([&numFields](FieldIndex fidx, SparseTensorFieldKind, Level,
                            DimLevelType) -> bool {
    if (fidx >= kDataFieldStartingIdx)
      numFields++;
    return true;
  });
  numFields -= 1; // the last field is StorageSpecifier
  assert(numFields == getNumFields() - kDataFieldStartingIdx - 1);
  return numFields;
}

std::pair<FieldIndex, unsigned>
StorageLayout::getFieldIndexAndStride(SparseTensorFieldKind kind,
                                      std::optional<Level> lvl) const {
  FieldIndex fieldIdx = kInvalidFieldIndex;
  unsigned stride = 1;
  if (kind == SparseTensorFieldKind::CrdMemRef) {
    assert(lvl.has_value());
    const Level cooStart = getCOOStart(enc);
    const Level lvlRank = enc.getLvlRank();
    if (lvl.value() >= cooStart && lvl.value() < lvlRank) {
      lvl = cooStart;
      stride = lvlRank - cooStart;
    }
  }
  foreachField([lvl, kind, &fieldIdx](FieldIndex fIdx,
                                      SparseTensorFieldKind fKind, Level fLvl,
                                      DimLevelType dlt) -> bool {
    if ((lvl && fLvl == lvl.value() && kind == fKind) ||
        (kind == fKind && fKind == SparseTensorFieldKind::ValMemRef)) {
      fieldIdx = fIdx;
      // Returns false to break the iteration.
      return false;
    }
    return true;
  });
  assert(fieldIdx != kInvalidFieldIndex);
  return std::pair<FieldIndex, unsigned>(fieldIdx, stride);
}

//===----------------------------------------------------------------------===//
// TensorDialect Attribute Methods.
//===----------------------------------------------------------------------===//

std::optional<uint64_t> SparseTensorDimSliceAttr::getStatic(int64_t v) {
  return isDynamic(v) ? std::nullopt
                      : std::make_optional(static_cast<uint64_t>(v));
}

std::optional<uint64_t> SparseTensorDimSliceAttr::getStaticOffset() const {
  return getStatic(getOffset());
}

std::optional<uint64_t> SparseTensorDimSliceAttr::getStaticStride() const {
  return getStatic(getStride());
}

std::optional<uint64_t> SparseTensorDimSliceAttr::getStaticSize() const {
  return getStatic(getSize());
}

bool SparseTensorDimSliceAttr::isCompletelyDynamic() const {
  return isDynamic(getOffset()) && isDynamic(getStride()) &&
         isDynamic(getSize());
}

std::string SparseTensorDimSliceAttr::getStaticString(int64_t v) {
  return isDynamic(v) ? "?" : std::to_string(v);
}

void SparseTensorDimSliceAttr::print(llvm::raw_ostream &os) const {
  assert(getImpl() && "Uninitialized SparseTensorDimSliceAttr");
  os << '(';
  os << getStaticString(getOffset());
  os << ", ";
  os << getStaticString(getSize());
  os << ", ";
  os << getStaticString(getStride());
  os << ')';
}

void SparseTensorDimSliceAttr::print(AsmPrinter &printer) const {
  print(printer.getStream());
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
  int64_t offset = kDynamic, size = kDynamic, stride = kDynamic;

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
  if (!isDynamic(offset) && offset < 0)
    return emitError() << "expect non-negative value or ? for slice offset";
  if (!isDynamic(size) && size <= 0)
    return emitError() << "expect positive value or ? for slice size";
  if (!isDynamic(stride) && stride <= 0)
    return emitError() << "expect positive value or ? for slice stride";
  return success();
}

Type mlir::sparse_tensor::detail::getIntegerOrIndexType(MLIRContext *ctx,
                                                        unsigned bitwidth) {
  if (bitwidth)
    return IntegerType::get(ctx, bitwidth);
  return IndexType::get(ctx);
}

Type SparseTensorEncodingAttr::getPosType() const {
  assert(getImpl() && "Uninitialized SparseTensorEncodingAttr");
  return detail::getIntegerOrIndexType(getContext(), getPosWidth());
}

Type SparseTensorEncodingAttr::getCrdType() const {
  assert(getImpl() && "Uninitialized SparseTensorEncodingAttr");
  return detail::getIntegerOrIndexType(getContext(), getCrdWidth());
}

SparseTensorEncodingAttr
SparseTensorEncodingAttr::withDimToLvl(AffineMap dimToLvl) const {
  assert(getImpl() && "Uninitialized SparseTensorEncodingAttr");
  return SparseTensorEncodingAttr::get(getContext(), getLvlTypes(), dimToLvl,
                                       getLvlToDim(), getPosWidth(),
                                       getCrdWidth());
}

SparseTensorEncodingAttr
SparseTensorEncodingAttr::withDimToLvl(SparseTensorEncodingAttr enc) const {
  return withDimToLvl(enc ? enc.getDimToLvl() : AffineMap());
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::withoutDimToLvl() const {
  return withDimToLvl(AffineMap());
}

SparseTensorEncodingAttr
SparseTensorEncodingAttr::withBitWidths(unsigned posWidth,
                                        unsigned crdWidth) const {
  assert(getImpl() && "Uninitialized SparseTensorEncodingAttr");
  return SparseTensorEncodingAttr::get(getContext(), getLvlTypes(),
                                       getDimToLvl(), getLvlToDim(), posWidth,
                                       crdWidth);
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::withoutBitWidths() const {
  return withBitWidths(0, 0);
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::withDimSlices(
    ArrayRef<SparseTensorDimSliceAttr> dimSlices) const {
  return SparseTensorEncodingAttr::get(getContext(), getLvlTypes(),
                                       getDimToLvl(), getLvlToDim(),
                                       getPosWidth(), getCrdWidth(), dimSlices);
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::withoutDimSlices() const {
  return withDimSlices(ArrayRef<SparseTensorDimSliceAttr>{});
}

bool SparseTensorEncodingAttr::isAllDense() const {
  return !getImpl() || llvm::all_of(getLvlTypes(), isDenseDLT);
}

bool SparseTensorEncodingAttr::isCOO() const {
  return getImpl() && isCOOType(*this, 0, true);
}

bool SparseTensorEncodingAttr::isAllOrdered() const {
  return !getImpl() || llvm::all_of(getLvlTypes(), isOrderedDLT);
}

bool SparseTensorEncodingAttr::isIdentity() const {
  return !getImpl() || !getDimToLvl() || getDimToLvl().isIdentity();
}

bool SparseTensorEncodingAttr::isPermutation() const {
  return !getImpl() || !getDimToLvl() || getDimToLvl().isPermutation();
}

Dimension SparseTensorEncodingAttr::getDimRank() const {
  assert(getImpl() && "Uninitialized SparseTensorEncodingAttr");
  const auto dimToLvl = getDimToLvl();
  return dimToLvl ? dimToLvl.getNumDims() : getLvlRank();
}

Level SparseTensorEncodingAttr::getLvlRank() const {
  assert(getImpl() && "Uninitialized SparseTensorEncodingAttr");
  return getLvlTypes().size();
}

DimLevelType SparseTensorEncodingAttr::getLvlType(Level l) const {
  if (!getImpl())
    return DimLevelType::Dense;
  assert(l < getLvlRank() && "Level is out of bounds");
  return getLvlTypes()[l];
}

bool SparseTensorEncodingAttr::isSlice() const {
  assert(getImpl() && "Uninitialized SparseTensorEncodingAttr");
  return !getDimSlices().empty();
}

SparseTensorDimSliceAttr
SparseTensorEncodingAttr::getDimSlice(Dimension dim) const {
  assert(isSlice() && "Is not a slice");
  const auto dimSlices = getDimSlices();
  assert(dim < dimSlices.size() && "Dimension is out of bounds");
  return dimSlices[dim];
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticDimSliceOffset(Dimension dim) const {
  return getDimSlice(dim).getStaticOffset();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticDimSliceSize(Dimension dim) const {
  return getDimSlice(dim).getStaticSize();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticDimSliceStride(Dimension dim) const {
  return getDimSlice(dim).getStaticStride();
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

const static DimLevelType validDLTs[] = {DimLevelType::Dense,
                                         DimLevelType::TwoOutOfFour,
                                         DimLevelType::Compressed,
                                         DimLevelType::CompressedNu,
                                         DimLevelType::CompressedNo,
                                         DimLevelType::CompressedNuNo,
                                         DimLevelType::Singleton,
                                         DimLevelType::SingletonNu,
                                         DimLevelType::SingletonNo,
                                         DimLevelType::SingletonNuNo,
                                         DimLevelType::LooseCompressed,
                                         DimLevelType::LooseCompressedNu,
                                         DimLevelType::LooseCompressedNo,
                                         DimLevelType::LooseCompressedNuNo};

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
  SmallVector<DimLevelType> lvlTypes;
  SmallVector<SparseTensorDimSliceAttr> dimSlices;
  AffineMap dimToLvl = {};
  unsigned posWidth = 0;
  unsigned crdWidth = 0;
  StringRef attrName;
  SmallVector<StringRef, 6> keys = {"lvlTypes", "dimToLvl",  "posWidth",
                                    "crdWidth", "dimSlices", "map"};
  while (succeeded(parser.parseOptionalKeyword(&attrName))) {
    // Detect admissible keyword.
    auto *it = find(keys, attrName);
    if (it == keys.end()) {
      parser.emitError(parser.getNameLoc(), "unexpected key: ") << attrName;
      return {};
    }
    unsigned keyWordIndex = it - keys.begin();
    // Consume the `=` after keys
    RETURN_ON_FAIL(parser.parseEqual())
    // Dispatch on keyword.
    switch (keyWordIndex) {
    case 0: { // lvlTypes
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr));
      auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr);
      ERROR_IF(!arrayAttr, "expected an array for lvlTypes")
      for (auto i : arrayAttr) {
        auto strAttr = llvm::dyn_cast<StringAttr>(i);
        ERROR_IF(!strAttr, "expected a string value in lvlTypes")
        auto strVal = strAttr.getValue();
        if (auto optDLT = parseDLT(strVal)) {
          lvlTypes.push_back(optDLT.value());
        } else {
          parser.emitError(parser.getNameLoc(), "unexpected level-type: ")
              << strVal;
          return {};
        }
      }
      break;
    }
    case 1: { // dimToLvl
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))
      auto affineAttr = llvm::dyn_cast<AffineMapAttr>(attr);
      ERROR_IF(!affineAttr, "expected an affine map for dimToLvl")
      dimToLvl = affineAttr.getValue();
      break;
    }
    case 2: { // posWidth
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))
      auto intAttr = llvm::dyn_cast<IntegerAttr>(attr);
      ERROR_IF(!intAttr, "expected an integral position bitwidth")
      posWidth = intAttr.getInt();
      break;
    }
    case 3: { // crdWidth
      Attribute attr;
      RETURN_ON_FAIL(parser.parseAttribute(attr))
      auto intAttr = llvm::dyn_cast<IntegerAttr>(attr);
      ERROR_IF(!intAttr, "expected an integral index bitwidth")
      crdWidth = intAttr.getInt();
      break;
    }
    case 4: { // dimSlices
      RETURN_ON_FAIL(parser.parseLSquare())
      // Dispatches to DimSliceAttr to skip mnemonic
      bool finished = false;
      while (auto attr = SparseTensorDimSliceAttr::parse(parser, nullptr)) {
        auto sliceAttr = llvm::cast<SparseTensorDimSliceAttr>(attr);
        dimSlices.push_back(sliceAttr);
        if (parser.parseOptionalComma().failed()) {
          finished = true;
          break;
        }
      }
      // Wrong when parsing slices
      if (!finished)
        return {};
      RETURN_ON_FAIL(parser.parseRSquare())
      break;
    }
    case 5: { // map (new STEA surface syntax)
      ir_detail::DimLvlMapParser cParser(parser);
      auto res = cParser.parseDimLvlMap();
      RETURN_ON_FAIL(res);
      // TODO: use DimLvlMap directly as storage representation, rather
      // than converting things over.
      const auto &dlm = *res;

      ERROR_IF(!lvlTypes.empty(), "Cannot mix `lvlTypes` with `map`")
      const Level lvlRank = dlm.getLvlRank();
      for (Level lvl = 0; lvl < lvlRank; lvl++)
        lvlTypes.push_back(dlm.getLvlType(lvl));

      ERROR_IF(!dimSlices.empty(), "Cannot mix `dimSlices` with `map`")
      const Dimension dimRank = dlm.getDimRank();
      for (Dimension dim = 0; dim < dimRank; dim++)
        dimSlices.push_back(dlm.getDimSlice(dim));
      // NOTE: the old syntax requires an all-or-nothing approach to
      // `dimSlices`; therefore, if any slice actually exists then we need
      // to convert null-DSA into default/nop DSA.
      const auto isDefined = [](SparseTensorDimSliceAttr slice) {
        return static_cast<bool>(slice.getImpl());
      };
      if (llvm::any_of(dimSlices, isDefined)) {
        const auto defaultSlice =
            SparseTensorDimSliceAttr::get(parser.getContext());
        for (Dimension dim = 0; dim < dimRank; dim++)
          if (!isDefined(dimSlices[dim]))
            dimSlices[dim] = defaultSlice;
      } else {
        dimSlices.clear();
      }

      ERROR_IF(dimToLvl, "Cannot mix `dimToLvl` with `map`")
      dimToLvl = dlm.getDimToLvlMap(parser.getContext());
      break;
    }
    } // switch
    // Only last item can omit the comma.
    if (parser.parseOptionalComma().failed())
      break;
  }

  RETURN_ON_FAIL(parser.parseRBrace())
  RETURN_ON_FAIL(parser.parseGreater())
#undef ERROR_IF
#undef RETURN_ON_FAIL

  // Construct struct-like storage for attribute.
  // TODO: Fetch lvlToDim if user provides one
  AffineMap lvlToDim = inferLvlToDim(dimToLvl, parser.getContext());
  return parser.getChecked<SparseTensorEncodingAttr>(
      parser.getContext(), lvlTypes, dimToLvl, lvlToDim, posWidth, crdWidth,
      dimSlices);
}

void SparseTensorEncodingAttr::print(AsmPrinter &printer) const {
  auto map = static_cast<AffineMap>(getDimToLvl());
  // Empty affine map indicates identity map
  if (!map)
    map = AffineMap::getMultiDimIdentityMap(getLvlTypes().size(), getContext());
  printer << "<{ map = ";
  printSymbols(map, printer);
  printer << '(';
  printDimensions(map, printer, getDimSlices());
  printer << ") -> (";
  printLevels(map, printer, getLvlTypes());
  printer << ')';
  // Print remaining members only for non-default values.
  if (getPosWidth())
    printer << ", posWidth = " << getPosWidth();
  if (getCrdWidth())
    printer << ", crdWidth = " << getCrdWidth();
  printer << " }>";
}

void SparseTensorEncodingAttr::printSymbols(AffineMap &map,
                                            AsmPrinter &printer) const {
  if (map.getNumSymbols() == 0)
    return;
  printer << '[';
  for (unsigned i = 0, n = map.getNumSymbols() - 1; i < n; i++)
    printer << 's' << i << ", ";
  if (map.getNumSymbols() >= 1)
    printer << 's' << map.getNumSymbols() - 1;
  printer << ']';
}

void SparseTensorEncodingAttr::printDimensions(
    AffineMap &map, AsmPrinter &printer,
    ArrayRef<SparseTensorDimSliceAttr> dimSlices) const {
  if (!dimSlices.empty()) {
    for (unsigned i = 0, n = map.getNumDims() - 1; i < n; i++)
      printer << 'd' << i << " : " << dimSlices[i] << ", ";
    if (map.getNumDims() >= 1) {
      printer << 'd' << map.getNumDims() - 1 << " : "
              << dimSlices[map.getNumDims() - 1];
    }
  } else {
    for (unsigned i = 0, n = map.getNumDims() - 1; i < n; i++)
      printer << 'd' << i << ", ";
    if (map.getNumDims() >= 1)
      printer << 'd' << map.getNumDims() - 1;
  }
}

void SparseTensorEncodingAttr::printLevels(
    AffineMap &map, AsmPrinter &printer,
    ArrayRef<DimLevelType> lvlTypes) const {
  for (unsigned i = 0, n = map.getNumResults() - 1; i < n; i++) {
    map.getResult(i).print(printer.getStream());
    printer << " : " << toMLIRString(lvlTypes[i]) << ", ";
  }
  if (map.getNumResults() >= 1) {
    auto lastIndex = map.getNumResults() - 1;
    map.getResult(lastIndex).print(printer.getStream());
    printer << " : " << toMLIRString(lvlTypes[lastIndex]);
  }
}

LogicalResult
SparseTensorEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<DimLevelType> lvlTypes,
                                 AffineMap dimToLvl, AffineMap lvlToDim,
                                 unsigned posWidth, unsigned crdWidth,
                                 ArrayRef<SparseTensorDimSliceAttr> dimSlices) {
  if (!acceptBitWidth(posWidth))
    return emitError() << "unexpected position bitwidth: " << posWidth;
  if (!acceptBitWidth(crdWidth))
    return emitError() << "unexpected coordinate bitwidth: " << crdWidth;
  // Before we can check that the level-rank is consistent/coherent
  // across all fields, we need to define it.  The source-of-truth for
  // the `getLvlRank` method is the length of the level-types array,
  // since it must always be provided and have full rank; therefore we
  // use that same source-of-truth here.
  const Level lvlRank = lvlTypes.size();
  if (lvlRank == 0)
    return emitError() << "expected a non-empty array for lvlTypes";
  // We save `dimRank` here because we'll also need it to verify `dimSlices`.
  const Dimension dimRank = dimToLvl ? dimToLvl.getNumDims() : lvlRank;
  if (dimToLvl) {
    if (dimToLvl.getNumResults() != lvlRank)
      return emitError()
             << "level-rank mismatch between dimToLvl and lvlTypes: "
             << dimToLvl.getNumResults() << " != " << lvlRank;
    // TODO:  The following is attempting to match the old error-conditions
    // from prior to merging dimOrdering and higherOrdering into dimToLvl.
    // That is, we currently require `dimToLvl` to be either a permutation
    // (as when higherOrdering is the identity) or expansive (as per the
    // constraints on higherOrdering).  However, those constraints do
    // not match the intended semantics of `dimToLvl`.  As we improve the
    // compiler to actually handle non-permutations, we need to update these
    // checks to match what is actually supported.  In particular, this is
    // where we'll have to check that when `lvlToDim` is provided then it
    // is indeed an inverse of `dimToLvl`, and when it isn't provided then
    // it can be automatically inferred.
    if (dimRank == lvlRank && !dimToLvl.isPermutation())
      return emitError() << "expected a permutation affine map for dimToLvl";
    if (dimRank > lvlRank)
      return emitError() << "unexpected dimToLvl mapping from " << dimRank
                         << " to " << lvlRank;
  }
  if (!dimSlices.empty()) {
    if (dimSlices.size() != dimRank)
      return emitError()
             << "dimension-rank mismatch between dimSlices and dimToLvl: "
             << dimSlices.size() << " != " << dimRank;
    // Compiler support for `dimSlices` currently requires that the two
    // ranks agree.  (However, it does allow `dimToLvl` to be a permutation.)
    if (dimRank != lvlRank)
      return emitError()
             << "dimSlices expected dimension-rank to match level-rank: "
             << dimRank << " != " << lvlRank;
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
  RETURN_FAILURE_IF_FAILED(verify(emitError, getLvlTypes(), getDimToLvl(),
                                  getLvlToDim(), getPosWidth(), getCrdWidth(),
                                  getDimSlices()))
  // Check integrity with tensor type specifics.  In particular, we
  // need only check that the dimension-rank of the tensor agrees with
  // the dimension-rank of the encoding.
  const Dimension dimRank = dimShape.size();
  if (dimRank == 0)
    return emitError() << "expected non-scalar sparse tensor";
  if (getDimRank() != dimRank)
    return emitError()
           << "dimension-rank mismatch between encoding and tensor shape: "
           << getDimRank() << " != " << dimRank;
  return success();
}

//===----------------------------------------------------------------------===//
// Convenience Methods.
//===----------------------------------------------------------------------===//

SparseTensorEncodingAttr
mlir::sparse_tensor::getSparseTensorEncoding(Type type) {
  if (auto ttp = llvm::dyn_cast<RankedTensorType>(type))
    return llvm::dyn_cast_or_null<SparseTensorEncodingAttr>(ttp.getEncoding());
  if (auto mdtp = llvm::dyn_cast<StorageSpecifierType>(type))
    return mdtp.getEncoding();
  return nullptr;
}

AffineMap mlir::sparse_tensor::inferLvlToDim(AffineMap dimToLvl,
                                             MLIRContext *context) {
  auto map = static_cast<AffineMap>(dimToLvl);
  AffineMap lvlToDim;
  // Return an empty lvlToDim when inference is not successful.
  if (!map || map.getNumSymbols() != 0) {
    lvlToDim = AffineMap();
  } else if (map.isPermutation()) {
    lvlToDim = inversePermutation(map);
  } else {
    // TODO: check if it's block sparsity
    lvlToDim = inverseBlockSparsity(map, context);
  }
  return lvlToDim;
}

AffineMap mlir::sparse_tensor::inverseBlockSparsity(AffineMap dimToLvl,
                                                    MLIRContext *context) {
  SmallVector<AffineExpr> lvlExprs;
  auto numLvls = dimToLvl.getNumResults();
  lvlExprs.reserve(numLvls);
  // lvlExprComponents stores information of the floordiv and mod operations
  // applied to the same dimension, so as to build the lvlToDim map.
  std::map<unsigned, SmallVector<AffineExpr, 3>> lvlExprComponents;
  for (unsigned i = 0, n = numLvls; i < n; i++) {
    auto result = dimToLvl.getResult(i);
    if (auto binOp = result.dyn_cast<AffineBinaryOpExpr>()) {
      if (result.getKind() == AffineExprKind::FloorDiv) {
        // Position of the dimension in dimToLvl.
        auto pos = binOp.getLHS().dyn_cast<AffineDimExpr>().getPosition();
        assert(lvlExprComponents.find(pos) == lvlExprComponents.end() &&
               "expected only one floordiv for each dimension");
        SmallVector<AffineExpr, 3> components;
        // Level variable for floordiv.
        components.push_back(getAffineDimExpr(i, context));
        // Multiplier.
        components.push_back(binOp.getRHS());
        // Map key is the position of the dimension.
        lvlExprComponents[pos] = components;
      } else if (result.getKind() == AffineExprKind::Mod) {
        auto pos = binOp.getLHS().dyn_cast<AffineDimExpr>().getPosition();
        assert(lvlExprComponents.find(pos) != lvlExprComponents.end() &&
               "expected floordiv before mod");
        // Add level variable for mod to the same vector
        // of the corresponding floordiv.
        lvlExprComponents[pos].push_back(getAffineDimExpr(i, context));
      } else {
        assert(false && "expected floordiv or mod");
      }
    } else {
      lvlExprs.push_back(getAffineDimExpr(i, context));
    }
  }
  // Build lvlExprs from lvlExprComponents.
  // For example, for il = i floordiv 2 and ii = i mod 2, the components
  // would be [il, 2, ii]. It could be used to build the AffineExpr
  // i = il * 2 + ii in lvlToDim.
  for (auto &components : lvlExprComponents) {
    assert(components.second.size() == 3 &&
           "expected 3 components to build lvlExprs");
    auto mulOp = getAffineBinaryOpExpr(
        AffineExprKind::Mul, components.second[0], components.second[1]);
    auto addOp =
        getAffineBinaryOpExpr(AffineExprKind::Add, mulOp, components.second[2]);
    lvlExprs.push_back(addOp);
  }
  return dimToLvl.get(dimToLvl.getNumResults(), 0, lvlExprs, context);
}

bool mlir::sparse_tensor::isCOOType(SparseTensorEncodingAttr enc,
                                    Level startLvl, bool isUnique) {
  if (!enc ||
      !(enc.isCompressedLvl(startLvl) || enc.isLooseCompressedLvl(startLvl)))
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

bool mlir::sparse_tensor::isUniqueCOOType(Type tp) {
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
  // TODO: This assertion is to match the behavior from before we merged
  // dimOrdering and higherOrdering into dimToLvl.  However, there's no
  // in-principle reason to require this.  (wrengr has a commit in the
  // wings to fix this.)
  assert(src.isPermutation());
  const Level lvlRank = src.getLvlRank();
  SmallVector<DimLevelType> lvlTypes;
  lvlTypes.reserve(lvlRank);

  // An unordered and non-unique compressed level at beginning.
  // If this is also the last level, then it is unique.
  lvlTypes.push_back(
      *buildLevelType(LevelFormat::Compressed, ordered, lvlRank == 1));
  if (lvlRank > 1) {
    // TODO: it is actually ordered at the level for ordered input.
    // Followed by unordered non-unique n-2 singleton levels.
    std::fill_n(std::back_inserter(lvlTypes), lvlRank - 2,
                *buildLevelType(LevelFormat::Singleton, ordered, false));
    // Ends by a unique singleton level unless the lvlRank is 1.
    lvlTypes.push_back(*buildLevelType(LevelFormat::Singleton, ordered, true));
  }

  // TODO: Maybe pick the bitwidth based on input/output tensors (probably the
  // largest one among them) in the original operation instead of using the
  // default value.
  unsigned posWidth = src.getPosWidth();
  unsigned crdWidth = src.getCrdWidth();
  AffineMap invPerm = src.getLvlToDim();
  auto enc = SparseTensorEncodingAttr::get(src.getContext(), lvlTypes, lvlPerm,
                                           invPerm, posWidth, crdWidth);
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
    if (const auto dimToLvl = enc.getDimToLvl()) {
      assert(enc.isPermutation());
      return dimToLvl.getDimPosition(l);
    }
  }
  return l;
}

// TODO: Remove this definition once all use-sites have been fixed to
// properly handle non-permutations.
Level mlir::sparse_tensor::toStoredDim(SparseTensorEncodingAttr enc,
                                       Dimension d) {
  if (enc) {
    if (const auto dimToLvl = enc.getDimToLvl()) {
      assert(enc.isPermutation());
      auto maybePos =
          dimToLvl.getResultPosition(getAffineDimExpr(d, enc.getContext()));
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
/// ordered/unique DLT such that "compressed_nu_no" and "compressed_nu" (as well
/// as other variants) lead to the same storage specifier type, and stripping
/// irrelevant fields that do not alter the sparse tensor memory layout.
static SparseTensorEncodingAttr
getNormalizedEncodingForSpecifier(SparseTensorEncodingAttr enc) {
  SmallVector<DimLevelType> dlts;
  for (auto dlt : enc.getLvlTypes())
    dlts.push_back(*buildLevelType(*getLevelFormat(dlt), true, true));

  return SparseTensorEncodingAttr::get(
      enc.getContext(), dlts,
      AffineMap(), // dimToLvl (irrelevant to storage specifier)
      AffineMap(), // lvlToDim (irrelevant to storage specifier)
      // Always use `index` for memSize and lvlSize instead of reusing
      // `getPosWidth` and `getCrdWidth`. It allows us to reuse the same SSA
      // value for different bitwidth, it also avoids casting between index and
      // integer (returned by DimOp)
      0, 0, enc.getDimSlices());
}

StorageSpecifierType
StorageSpecifierType::get(MLIRContext *ctx, SparseTensorEncodingAttr encoding) {
  return Base::get(ctx, getNormalizedEncodingForSpecifier(encoding));
}

//===----------------------------------------------------------------------===//
// SparseTensorDialect Operations.
//===----------------------------------------------------------------------===//

static LogicalResult lvlIsInBounds(Level lvl, Value tensor) {
  return success(lvl < getSparseTensorType(tensor).getLvlRank());
}

static LogicalResult isMatchingWidth(Value mem, unsigned width) {
  const Type etp = getMemRefType(mem).getElementType();
  return success(width == 0 ? etp.isIndex() : etp.isInteger(width));
}

static LogicalResult verifySparsifierGetterSetter(
    StorageSpecifierKind mdKind, std::optional<Level> lvl,
    TypedValue<StorageSpecifierType> md, Operation *op) {
  if (mdKind == StorageSpecifierKind::ValMemSize && lvl) {
    return op->emitError(
        "redundant level argument for querying value memory size");
  }

  const auto enc = md.getType().getEncoding();
  const Level lvlRank = enc.getLvlRank();

  if (mdKind == StorageSpecifierKind::DimOffset ||
      mdKind == StorageSpecifierKind::DimStride)
    if (!enc.isSlice())
      return op->emitError("requested slice data on non-slice tensor");

  if (mdKind != StorageSpecifierKind::ValMemSize) {
    if (!lvl)
      return op->emitError("missing level argument");

    const Level l = lvl.value();
    if (l >= lvlRank)
      return op->emitError("requested level is out of bounds");

    if (mdKind == StorageSpecifierKind::PosMemSize && enc.isSingletonLvl(l))
      return op->emitError(
          "requested position memory size on a singleton level");
  }
  return success();
}

static Type getFieldElemType(SparseTensorType stt, SparseTensorFieldKind kind) {
  switch (kind) {
  case SparseTensorFieldKind::CrdMemRef:
    return stt.getCrdType();
  case SparseTensorFieldKind::PosMemRef:
    return stt.getPosType();
  case SparseTensorFieldKind::ValMemRef:
    return stt.getElementType();
  case SparseTensorFieldKind::StorageSpec:
    return nullptr;
  }
  llvm_unreachable("Unrecognizable FieldKind");
}

static LogicalResult verifyPackUnPack(Operation *op, bool requiresStaticShape,
                                      SparseTensorType stt,
                                      RankedTensorType valTp,
                                      TypeRange lvlTps) {
  if (requiresStaticShape && !stt.hasStaticDimShape())
    return op->emitError("the sparse-tensor must have static shape");
  if (!stt.hasEncoding())
    return op->emitError("the sparse-tensor must have an encoding attribute");
  if (!stt.isIdentity())
    return op->emitError("the sparse-tensor must have the identity mapping");

  // Verifies the trailing COO.
  Level cooStartLvl = getCOOStart(stt.getEncoding());
  if (cooStartLvl < stt.getLvlRank()) {
    // We only supports trailing COO for now, must be the last input.
    auto cooTp = llvm::cast<ShapedType>(lvlTps.back());
    // The coordinates should be in shape of <? x rank>
    unsigned expCOORank = stt.getLvlRank() - cooStartLvl;
    if (cooTp.getRank() != 2 || expCOORank != cooTp.getShape().back()) {
      op->emitError("input/output trailing COO level-ranks don't match");
    }
  }

  // Verifies that all types match.
  StorageLayout layout(stt.getEncoding());
  if (layout.getNumDataFields() != lvlTps.size() + 1) // plus one value memref
    return op->emitError("inconsistent number of fields between input/output");

  unsigned idx = 0;
  bool misMatch = false;
  layout.foreachField([&idx, &misMatch, stt, valTp,
                       lvlTps](FieldIndex fid, SparseTensorFieldKind fKind,
                               Level lvl, DimLevelType dlt) -> bool {
    if (fKind == SparseTensorFieldKind::StorageSpec)
      return true;

    Type inputTp = nullptr;
    if (fKind == SparseTensorFieldKind::ValMemRef) {
      inputTp = valTp;
    } else {
      assert(fid == idx && stt.getLvlType(lvl) == dlt);
      inputTp = lvlTps[idx++];
    }
    // The input element type and expected element type should match.
    Type inpElemTp = llvm::cast<TensorType>(inputTp).getElementType();
    Type expElemTp = getFieldElemType(stt, fKind);
    if (inpElemTp != expElemTp) {
      misMatch = true;
      return false; // to terminate the iteration
    }
    return true;
  });

  if (misMatch)
    return op->emitError("input/output element-types don't match");
  return success();
}

LogicalResult AssembleOp::verify() {
  const auto valuesTp = getRankedTensorType(getValues());
  const auto lvlsTp = getLevels().getTypes();
  const auto resTp = getSparseTensorType(getResult());
  return verifyPackUnPack(*this, true, resTp, valuesTp, lvlsTp);
}

LogicalResult DisassembleOp::verify() {
  if (getOutValues().getType() != getRetValues().getType())
    return emitError("output values and return value type mismatch");

  for (auto [ot, rt] : llvm::zip_equal(getOutLevels(), getRetLevels()))
    if (ot.getType() != rt.getType())
      return emitError("output levels and return levels type mismatch");

  const auto valuesTp = getRankedTensorType(getRetValues());
  const auto lvlsTp = getRetLevels().getTypes();
  const auto srcTp = getSparseTensorType(getTensor());
  return verifyPackUnPack(*this, false, srcTp, valuesTp, lvlsTp);
}

LogicalResult ConvertOp::verify() {
  if (auto tp1 = llvm::dyn_cast<RankedTensorType>(getSource().getType())) {
    if (auto tp2 = llvm::dyn_cast<RankedTensorType>(getDest().getType())) {
      if (tp1.getRank() != tp2.getRank())
        return emitError("unexpected conversion mismatch in rank");
      auto dstEnc =
          llvm::dyn_cast_or_null<SparseTensorEncodingAttr>(tp2.getEncoding());
      if (dstEnc && dstEnc.isSlice())
        return emitError("cannot convert to a sparse tensor slice");

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
  if (getType() == getSource().getType())
    return getSource();
  return {};
}

bool ConvertOp::needsExtraSort() {
  SparseTensorType srcStt = getSparseTensorType(getSource());
  SparseTensorType dstStt = getSparseTensorType(getDest());

  // We do not need an extra sort when returning unordered sparse tensors or
  // dense tensor since dense tensor support random access.
  if (dstStt.isAllDense() || !dstStt.isAllOrdered())
    return false;

  if (srcStt.isAllOrdered() && dstStt.isAllOrdered() &&
      srcStt.hasSameDimToLvl(dstStt)) {
    return false;
  }

  // Source and dest tensors are ordered in different ways. We only do direct
  // dense to sparse conversion when the dense input is defined by a sparse
  // constant. Note that we can theoretically always directly convert from dense
  // inputs by rotating dense loops but it leads to bad cache locality and hurt
  // performance.
  if (auto constOp = getSource().getDefiningOp<arith::ConstantOp>())
    if (isa<SparseElementsAttr>(constOp.getValue()))
      return false;

  return true;
}

LogicalResult CrdTranslateOp::verify() {
  uint64_t inRank = getEncoder().getLvlRank();
  uint64_t outRank = getEncoder().getDimRank();

  if (getDirection() == CrdTransDirectionKind::dim2lvl)
    std::swap(inRank, outRank);

  if (inRank != getInCrds().size() || outRank != getOutCrds().size())
    return emitError("Coordinate rank mismatch with encoding");

  return success();
}

LogicalResult CrdTranslateOp::fold(FoldAdaptor adaptor,
                                   SmallVectorImpl<OpFoldResult> &results) {
  if (getEncoder().isPermutation()) {
    AffineMap perm = getDirection() == CrdTransDirectionKind::dim2lvl
                         ? getEncoder().getDimToLvl()
                         : getEncoder().getLvlToDim();
    for (AffineExpr exp : perm.getResults())
      results.push_back(getInCrds()[exp.cast<AffineDimExpr>().getPosition()]);
    return success();
  }

  // Fuse dim2lvl/lvl2dim pairs.
  auto def = getInCrds()[0].getDefiningOp<CrdTranslateOp>();
  bool sameDef = def && llvm::all_of(getInCrds(), [def](Value v) {
                   return v.getDefiningOp() == def;
                 });
  if (!sameDef)
    return failure();

  bool oppositeDir = def.getDirection() != getDirection();
  bool sameOracle =
      def.getEncoder().getDimToLvl() == getEncoder().getDimToLvl();
  bool sameCount = def.getNumResults() == getInCrds().size();
  if (!oppositeDir || !sameOracle || !sameCount)
    return failure();

  // The definition produces the coordinates in the same order as the input
  // coordinates.
  bool sameOrder = llvm::all_of(llvm::zip_equal(def.getOutCrds(), getInCrds()),
                                [](auto valuePair) {
                                  auto [lhs, rhs] = valuePair;
                                  return lhs == rhs;
                                });

  if (!sameOrder)
    return failure();
  // l1 = dim2lvl (lvl2dim l0)
  // ==> l0
  results.append(def.getInCrds().begin(), def.getInCrds().end());
  return success();
}

LogicalResult ToPositionsOp::verify() {
  auto e = getSparseTensorEncoding(getTensor().getType());
  if (failed(lvlIsInBounds(getLevel(), getTensor())))
    return emitError("requested level is out of bounds");
  if (failed(isMatchingWidth(getResult(), e.getPosWidth())))
    return emitError("unexpected type for positions");
  return success();
}

LogicalResult ToCoordinatesOp::verify() {
  auto e = getSparseTensorEncoding(getTensor().getType());
  if (failed(lvlIsInBounds(getLevel(), getTensor())))
    return emitError("requested level is out of bounds");
  if (failed(isMatchingWidth(getResult(), e.getCrdWidth())))
    return emitError("unexpected type for coordinates");
  return success();
}

LogicalResult ToCoordinatesBufferOp::verify() {
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

LogicalResult ToSliceOffsetOp::verify() {
  auto rank = getRankedTensorType(getSlice()).getRank();
  if (rank <= getDim().getSExtValue() || getDim().getSExtValue() < 0)
    return emitError("requested dimension out of bound");
  return success();
}

LogicalResult ToSliceStrideOp::verify() {
  auto rank = getRankedTensorType(getSlice()).getRank();
  if (rank <= getDim().getSExtValue() || getDim().getSExtValue() < 0)
    return emitError("requested dimension out of bound");
  return success();
}

LogicalResult GetStorageSpecifierOp::verify() {
  RETURN_FAILURE_IF_FAILED(verifySparsifierGetterSetter(
      getSpecifierKind(), getLevel(), getSpecifier(), getOperation()))
  return success();
}

template <typename SpecifierOp>
static SetStorageSpecifierOp getSpecifierSetDef(SpecifierOp op) {
  return op.getSpecifier().template getDefiningOp<SetStorageSpecifierOp>();
}

OpFoldResult GetStorageSpecifierOp::fold(FoldAdaptor adaptor) {
  const StorageSpecifierKind kind = getSpecifierKind();
  const auto lvl = getLevel();
  for (auto op = getSpecifierSetDef(*this); op; op = getSpecifierSetDef(op))
    if (kind == op.getSpecifierKind() && lvl == op.getLevel())
      return op.getValue();
  return {};
}

LogicalResult SetStorageSpecifierOp::verify() {
  RETURN_FAILURE_IF_FAILED(verifySparsifierGetterSetter(
      getSpecifierKind(), getLevel(), getSpecifier(), getOperation()))
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

bool ConcatenateOp::needsExtraSort() {
  SparseTensorType dstStt = getSparseTensorType(*this);
  if (dstStt.isAllDense() || !dstStt.isAllOrdered())
    return false;

  bool allSameOrdered = llvm::all_of(getInputs(), [dstStt](Value op) {
    return getSparseTensorType(op).hasSameDimToLvl(dstStt);
  });
  // TODO: When conDim != 0, as long as conDim corresponding to the first level
  // in all input/output buffers, and all input/output buffers have the same
  // dimToLvl, the tmp COO buffer is still unnecessary (e.g, concatenate
  // CSC matrices along column).
  bool directLowerable =
      allSameOrdered && getDimension() == 0 && dstStt.isIdentity();
  return !directLowerable;
}

LogicalResult ConcatenateOp::verify() {
  const auto dstTp = getSparseTensorType(*this);
  const Dimension concatDim = getDimension();
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
  const auto stt = getSparseTensorType(getTensor());
  if (stt.getLvlRank() != static_cast<Level>(getLvlCoords().size()))
    return emitOpError("incorrect number of coordinates");
  return success();
}

void PushBackOp::build(OpBuilder &builder, OperationState &result,
                       Value curSize, Value inBuffer, Value value) {
  build(builder, result, curSize, inBuffer, value, Value());
}

LogicalResult PushBackOp::verify() {
  if (Value n = getN()) {
    std::optional<int64_t> nValue = getConstantIntValue(n);
    if (nValue && nValue.value() < 1)
      return emitOpError("n must be not less than 1");
  }
  return success();
}

LogicalResult CompressOp::verify() {
  const auto stt = getSparseTensorType(getTensor());
  if (stt.getLvlRank() != 1 + static_cast<Level>(getLvlCoords().size()))
    return emitOpError("incorrect number of coordinates");
  return success();
}

void ForeachOp::build(
    OpBuilder &builder, OperationState &result, Value tensor,
    ValueRange initArgs, AffineMapAttr order,
    function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)>
        bodyBuilder) {
  build(builder, result, initArgs.getTypes(), tensor, initArgs, order);
  // Builds foreach body.
  if (!bodyBuilder)
    return;
  const auto stt = getSparseTensorType(tensor);
  const Dimension dimRank = stt.getDimRank();

  // Starts with `dimRank`-many coordinates.
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

  if (getOrder().has_value() &&
      (t.getEncoding() || !getOrder()->isPermutation()))
    return emitError("Only support permuted order on non encoded dense tensor");

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

OpFoldResult ReorderCOOOp::fold(FoldAdaptor adaptor) {
  if (getSparseTensorEncoding(getInputCoo().getType()) ==
      getSparseTensorEncoding(getResultCoo().getType()))
    return getInputCoo();

  return {};
}

LogicalResult ReorderCOOOp::verify() {
  SparseTensorType srcStt = getSparseTensorType(getInputCoo());
  SparseTensorType dstStt = getSparseTensorType(getResultCoo());

  if (!srcStt.hasSameDimToLvl(dstStt))
    emitError("Unmatched dim2lvl map between input and result COO");

  if (srcStt.getPosType() != dstStt.getPosType() ||
      srcStt.getCrdType() != dstStt.getCrdType() ||
      srcStt.getElementType() != dstStt.getElementType()) {
    emitError("Unmatched storage format between input and result COO");
  }
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
  AffineMap xPerm = getPermMap();
  uint64_t nx = xPerm.getNumDims();
  if (nx < 1)
    emitError(llvm::formatv("Expected rank(perm_map) > 1, got {0}", nx));

  if (!xPerm.isPermutation())
    emitError(llvm::formatv("Expected a permutation map, got {0}", xPerm));

  std::optional<int64_t> cn = getConstantIntValue(getN());
  // We can't check the size of the buffers when n or buffer dimensions aren't
  // compile-time constants.
  if (!cn)
    return success();

  uint64_t n = cn.value();
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

  checkDim(getXy(), n * (nx + ny),
           "Expected dimension(xy) >= n * (rank(perm_map) + ny)");

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
