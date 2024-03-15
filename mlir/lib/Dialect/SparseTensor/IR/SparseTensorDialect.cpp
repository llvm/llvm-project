//===- SparseTensorDialect.cpp - Sparse tensor dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "Detail/DimLvlMapParser.h"

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
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

// Support hashing LevelType such that SparseTensorEncodingAttr can be hashed as
// well.
namespace mlir::sparse_tensor {
llvm::hash_code hash_value(LevelType lt) {
  return llvm::hash_value(static_cast<uint64_t>(lt));
}
} // namespace mlir::sparse_tensor

//===----------------------------------------------------------------------===//
// Local Convenience Methods.
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
// SparseTensorDialect StorageLayout.
//===----------------------------------------------------------------------===//

static constexpr Level kInvalidLevel = -1u;
static constexpr Level kInvalidFieldIndex = -1u;
static constexpr FieldIndex kDataFieldStartingIdx = 0;

void StorageLayout::foreachField(
    llvm::function_ref<bool(FieldIndex, SparseTensorFieldKind, Level,
                            LevelType)>
        callback) const {
  const auto lvlTypes = enc.getLvlTypes();
  const Level lvlRank = enc.getLvlRank();
  SmallVector<COOSegment> cooSegs = SparseTensorType(enc).getCOOSegments();
  FieldIndex fieldIdx = kDataFieldStartingIdx;

  ArrayRef cooSegsRef = cooSegs;
  // Per-level storage.
  for (Level l = 0; l < lvlRank; /*l += 1 or l += AoSCooLen*/) {
    const auto lt = lvlTypes[l];
    if (isWithPosLT(lt)) {
      if (!(callback(fieldIdx++, SparseTensorFieldKind::PosMemRef, l, lt)))
        return;
    }
    if (isWithCrdLT(lt)) {
      if (!(callback(fieldIdx++, SparseTensorFieldKind::CrdMemRef, l, lt)))
        return;
    }
    if (!cooSegsRef.empty() && cooSegsRef.front().isSegmentStart(l)) {
      if (!cooSegsRef.front().isSoA) {
        // AoS COO, all singletons are fused into one memrefs. Skips the entire
        // COO segement.
        l = cooSegsRef.front().lvlRange.second;
      } else {
        // SoA COO, each singleton level has one memref.
        l++;
      }
      // Expire handled COO segment.
      cooSegsRef = cooSegsRef.drop_front();
    } else {
      // Non COO levels.
      l++;
    }
  }
  // The values array.
  if (!(callback(fieldIdx++, SparseTensorFieldKind::ValMemRef, kInvalidLevel,
                 LevelFormat::Undef)))
    return;
  // Put metadata at the end.
  if (!(callback(fieldIdx++, SparseTensorFieldKind::StorageSpec, kInvalidLevel,
                 LevelFormat::Undef)))
    return;
}

void sparse_tensor::foreachFieldAndTypeInSparseTensor(
    SparseTensorType stt,
    llvm::function_ref<bool(Type, FieldIndex, SparseTensorFieldKind, Level,
                            LevelType)>
        callback) {
  assert(stt.hasEncoding());
  // Construct the basic types.
  const Type crdType = stt.getCrdType();
  const Type posType = stt.getPosType();
  const Type eltType = stt.getElementType();

  SmallVector<int64_t> memrefShape = stt.getBatchLvlShape();
  memrefShape.push_back(ShapedType::kDynamic);

  const Type specType = StorageSpecifierType::get(stt.getEncoding());
  // memref<[batch] x ? x pos>  positions
  const Type posMemType = MemRefType::get(memrefShape, posType);
  // memref<[batch] x ? x crd>  coordinates
  const Type crdMemType = MemRefType::get(memrefShape, crdType);
  // memref<[batch] x ? x eltType> values
  const Type valMemType = MemRefType::get(memrefShape, eltType);

  StorageLayout(stt).foreachField([specType, posMemType, crdMemType, valMemType,
                                   callback](FieldIndex fieldIdx,
                                             SparseTensorFieldKind fieldKind,
                                             Level lvl, LevelType lt) -> bool {
    switch (fieldKind) {
    case SparseTensorFieldKind::StorageSpec:
      return callback(specType, fieldIdx, fieldKind, lvl, lt);
    case SparseTensorFieldKind::PosMemRef:
      return callback(posMemType, fieldIdx, fieldKind, lvl, lt);
    case SparseTensorFieldKind::CrdMemRef:
      return callback(crdMemType, fieldIdx, fieldKind, lvl, lt);
    case SparseTensorFieldKind::ValMemRef:
      return callback(valMemType, fieldIdx, fieldKind, lvl, lt);
    };
    llvm_unreachable("unrecognized field kind");
  });
}

unsigned StorageLayout::getNumFields() const {
  unsigned numFields = 0;
  foreachField([&numFields](FieldIndex, SparseTensorFieldKind, Level,
                            LevelType) -> bool {
    numFields++;
    return true;
  });
  return numFields;
}

unsigned StorageLayout::getNumDataFields() const {
  unsigned numFields = 0; // one value memref
  foreachField([&numFields](FieldIndex fidx, SparseTensorFieldKind, Level,
                            LevelType) -> bool {
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
    const Level cooStart = SparseTensorType(enc).getAoSCOOStart();
    const Level lvlRank = enc.getLvlRank();
    if (lvl.value() >= cooStart && lvl.value() < lvlRank) {
      lvl = cooStart;
      stride = lvlRank - cooStart;
    }
  }
  foreachField([lvl, kind, &fieldIdx](FieldIndex fIdx,
                                      SparseTensorFieldKind fKind, Level fLvl,
                                      LevelType lt) -> bool {
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
// SparseTensorDialect Attribute Methods.
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

SparseTensorEncodingAttr
SparseTensorEncodingAttr::withDimToLvl(AffineMap dimToLvl) const {
  assert(getImpl() && "Uninitialized SparseTensorEncodingAttr");
  return SparseTensorEncodingAttr::get(getContext(), getLvlTypes(), dimToLvl,
                                       AffineMap(), getPosWidth(),
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

uint64_t SparseTensorEncodingAttr::getBatchLvlRank() const {
  ArrayRef<LevelType> lvlTypes = getLvlTypes();
  auto lastBatch = std::find_if(lvlTypes.rbegin(), lvlTypes.rend(), isBatchLT);
  return std::distance(lastBatch, lvlTypes.rend());
}

bool SparseTensorEncodingAttr::isAllDense() const {
  return !getImpl() || llvm::all_of(getLvlTypes(), isDenseLT);
}

bool SparseTensorEncodingAttr::isAllOrdered() const {
  return !getImpl() || llvm::all_of(getLvlTypes(), isOrderedLT);
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

LevelType SparseTensorEncodingAttr::getLvlType(Level l) const {
  if (!getImpl())
    return LevelFormat::Batch;
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
SparseTensorEncodingAttr::getStaticDimSliceStride(Dimension dim) const {
  return getDimSlice(dim).getStaticStride();
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticLvlSliceOffset(Level lvl) const {
  return getStaticDimSliceOffset(toDim(*this, lvl));
}

std::optional<uint64_t>
SparseTensorEncodingAttr::getStaticLvlSliceStride(Level lvl) const {
  return getStaticDimSliceStride(toDim(*this, lvl));
}

SmallVector<int64_t>
SparseTensorEncodingAttr::translateShape(ArrayRef<int64_t> srcShape,
                                         CrdTransDirectionKind dir) const {
  if (isIdentity())
    return SmallVector<int64_t>(srcShape);

  SmallVector<int64_t> ret;
  unsigned rank =
      dir == CrdTransDirectionKind::dim2lvl ? getLvlRank() : getDimRank();
  ret.reserve(rank);

  if (isPermutation()) {
    for (unsigned r = 0; r < rank; r++) {
      unsigned trans = dir == CrdTransDirectionKind::dim2lvl ? toDim(*this, r)
                                                             : toLvl(*this, r);
      ret.push_back(srcShape[trans]);
    }
    return ret;
  }

  // Handle non-permutation maps.
  AffineMap transMap =
      dir == CrdTransDirectionKind::dim2lvl ? getDimToLvl() : getLvlToDim();

  SmallVector<AffineExpr> dimRep;
  dimRep.reserve(srcShape.size());
  for (int64_t sz : srcShape) {
    if (!ShapedType::isDynamic(sz)) {
      // Push back the max coordinate for the given dimension/level size.
      dimRep.push_back(getAffineConstantExpr(sz - 1, getContext()));
    } else {
      // A dynamic size, use a AffineDimExpr to symbolize the value.
      dimRep.push_back(getAffineDimExpr(dimRep.size(), getContext()));
    }
  };

  for (AffineExpr exp : transMap.getResults()) {
    // Do constant propagation on the affine map.
    AffineExpr evalExp =
        simplifyAffineExpr(exp.replaceDims(dimRep), srcShape.size(), 0);
    // use llvm namespace here to avoid ambiguity
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(evalExp)) {
      ret.push_back(c.getValue() + 1);
    } else {
      if (auto mod = llvm::dyn_cast<AffineBinaryOpExpr>(evalExp);
          mod && mod.getKind() == AffineExprKind::Mod) {
        // We can still infer a static bound for expressions in form
        // "d % constant" since d % constant \in [0, constant).
        if (auto bound = llvm::dyn_cast<AffineConstantExpr>(mod.getRHS())) {
          ret.push_back(bound.getValue());
          continue;
        }
      }
      ret.push_back(ShapedType::kDynamic);
    }
  }
  assert(ret.size() == rank);
  return ret;
}

ValueRange
SparseTensorEncodingAttr::translateCrds(OpBuilder &builder, Location loc,
                                        ValueRange crds,
                                        CrdTransDirectionKind dir) const {
  if (!getImpl())
    return crds;

  SmallVector<Type> retType(
      dir == CrdTransDirectionKind::lvl2dim ? getDimRank() : getLvlRank(),
      builder.getIndexType());
  auto transOp = builder.create<CrdTranslateOp>(loc, retType, crds, dir, *this);
  return transOp.getOutCrds();
}

Attribute SparseTensorEncodingAttr::parse(AsmParser &parser, Type type) {
  // Open "<{" part.
  if (failed(parser.parseLess()))
    return {};
  if (failed(parser.parseLBrace()))
    return {};

  // Process the data from the parsed dictionary value into struct-like data.
  SmallVector<LevelType> lvlTypes;
  SmallVector<SparseTensorDimSliceAttr> dimSlices;
  AffineMap dimToLvl = {};
  AffineMap lvlToDim = {};
  unsigned posWidth = 0;
  unsigned crdWidth = 0;
  StringRef attrName;
  SmallVector<StringRef, 3> keys = {"map", "posWidth", "crdWidth"};
  while (succeeded(parser.parseOptionalKeyword(&attrName))) {
    // Detect admissible keyword.
    auto *it = find(keys, attrName);
    if (it == keys.end()) {
      parser.emitError(parser.getNameLoc(), "unexpected key: ") << attrName;
      return {};
    }
    unsigned keyWordIndex = it - keys.begin();
    // Consume the `=` after keys
    if (failed(parser.parseEqual()))
      return {};
    // Dispatch on keyword.
    switch (keyWordIndex) {
    case 0: { // map
      ir_detail::DimLvlMapParser cParser(parser);
      auto res = cParser.parseDimLvlMap();
      if (failed(res))
        return {};
      const auto &dlm = *res;

      const Level lvlRank = dlm.getLvlRank();
      for (Level lvl = 0; lvl < lvlRank; lvl++)
        lvlTypes.push_back(dlm.getLvlType(lvl));

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

      dimToLvl = dlm.getDimToLvlMap(parser.getContext());
      lvlToDim = dlm.getLvlToDimMap(parser.getContext());
      break;
    }
    case 1: { // posWidth
      Attribute attr;
      if (failed(parser.parseAttribute(attr)))
        return {};
      auto intAttr = llvm::dyn_cast<IntegerAttr>(attr);
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an integral position bitwidth");
        return {};
      }
      posWidth = intAttr.getInt();
      break;
    }
    case 2: { // crdWidth
      Attribute attr;
      if (failed(parser.parseAttribute(attr)))
        return {};
      auto intAttr = llvm::dyn_cast<IntegerAttr>(attr);
      if (!intAttr) {
        parser.emitError(parser.getNameLoc(),
                         "expected an integral index bitwidth");
        return {};
      }
      crdWidth = intAttr.getInt();
      break;
    }
    } // switch
    // Only last item can omit the comma.
    if (parser.parseOptionalComma().failed())
      break;
  }

  // Close "}>" part.
  if (failed(parser.parseRBrace()))
    return {};
  if (failed(parser.parseGreater()))
    return {};

  // Construct struct-like storage for attribute.
  if (!lvlToDim || lvlToDim.isEmpty()) {
    lvlToDim = inferLvlToDim(dimToLvl, parser.getContext());
  }
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

void SparseTensorEncodingAttr::printLevels(AffineMap &map, AsmPrinter &printer,
                                           ArrayRef<LevelType> lvlTypes) const {
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

LogicalResult SparseTensorEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<LevelType> lvlTypes,
    AffineMap dimToLvl, AffineMap lvlToDim, unsigned posWidth,
    unsigned crdWidth, ArrayRef<SparseTensorDimSliceAttr> dimSlices) {
  if (!acceptBitWidth(posWidth))
    return emitError() << "unexpected position bitwidth: " << posWidth;
  if (!acceptBitWidth(crdWidth))
    return emitError() << "unexpected coordinate bitwidth: " << crdWidth;
  if (auto it = std::find_if(lvlTypes.begin(), lvlTypes.end(), isSingletonLT);
      it != std::end(lvlTypes)) {
    if (it == lvlTypes.begin() ||
        (!isCompressedLT(*(it - 1)) && !isLooseCompressedLT(*(it - 1))))
      return emitError() << "expected compressed or loose_compressed level "
                            "before singleton level";
    if (!std::all_of(it, lvlTypes.end(),
                     [](LevelType i) { return isSingletonLT(i); }))
      return emitError() << "expected all singleton lvlTypes "
                            "following a singleton level";
    // We can potentially support mixed SoA/AoS singleton levels.
    if (!std::all_of(it, lvlTypes.end(), [it](LevelType i) {
          return it->isa<LevelPropNonDefault::SoA>() ==
                 i.isa<LevelPropNonDefault::SoA>();
        })) {
      return emitError() << "expected all singleton lvlTypes stored in the "
                            "same memory layout (SoA vs AoS).";
    }
  }

  auto lastBatch = std::find_if(lvlTypes.rbegin(), lvlTypes.rend(), isBatchLT);
  if (!std::all_of(lastBatch, lvlTypes.rend(), isBatchLT))
    return emitError() << "Batch lvlType can only be leading levels.";

  // SoA property can only be applied on singleton level.
  auto soaLvls = llvm::make_filter_range(lvlTypes, [](LevelType lt) {
    return lt.isa<LevelPropNonDefault::SoA>();
  });
  if (llvm::any_of(soaLvls, [](LevelType lt) {
        return !lt.isa<LevelFormat::Singleton>();
      })) {
    return emitError() << "SoA is only applicable to singleton lvlTypes.";
  }

  // TODO: audit formats that actually are supported by backend.
  if (auto it = std::find_if(lvlTypes.begin(), lvlTypes.end(), isNOutOfMLT);
      it != std::end(lvlTypes)) {
    if (it != lvlTypes.end() - 1)
      return emitError() << "expected n_out_of_m to be the last level type";
    if (!std::all_of(lvlTypes.begin(), it,
                     [](LevelType i) { return isDenseLT(i); }))
      return emitError() << "expected all dense lvlTypes "
                            "before a n_out_of_m level";
    if (dimToLvl && (dimToLvl.getNumDims() != dimToLvl.getNumResults())) {
      if (!isBlockSparsity(dimToLvl)) {
        return emitError()
               << "expected 1xm block structure for n_out_of_m level";
      }
      auto sizes = getBlockSize(dimToLvl);
      unsigned coefficient = 0;
      for (const auto &elem : sizes) {
        if (elem != 0) {
          if (elem != coefficient && coefficient != 0) {
            return emitError() << "expected only one blocked level "
                                  "with the same coefficients";
          }
          coefficient = elem;
        }
      }
      if (coefficient != getM(*it)) {
        return emitError() << "expected coeffiencts of Affine expressions "
                              "to be equal to m of n_out_of_m level";
      }
    }
  }
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
    auto inferRes = inferLvlToDim(dimToLvl, dimToLvl.getContext());
    // Symbols can't be inferred but are acceptable.
    if (!inferRes && dimToLvl.getNumSymbols() == 0)
      return emitError() << "failed to infer lvlToDim from dimToLvl";
    if (lvlToDim && (inferRes != lvlToDim))
      return emitError() << "expected lvlToDim to be an inverse of dimToLvl";
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

LogicalResult SparseTensorEncodingAttr::verifyEncoding(
    ArrayRef<Size> dimShape, Type elementType,
    function_ref<InFlightDiagnostic()> emitError) const {
  // Check structural integrity.  In particular, this ensures that the
  // level-rank is coherent across all the fields.
  if (failed(verify(emitError, getLvlTypes(), getDimToLvl(), getLvlToDim(),
                    getPosWidth(), getCrdWidth(), getDimSlices())))
    return failure();
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
// SparseTensorType Methods.
//===----------------------------------------------------------------------===//

bool mlir::sparse_tensor::SparseTensorType::isCOOType(Level startLvl,
                                                      bool isUnique) const {
  if (!hasEncoding())
    return false;
  if (!isCompressedLvl(startLvl) && !isLooseCompressedLvl(startLvl))
    return false;
  for (Level l = startLvl + 1; l < lvlRank; ++l)
    if (!isSingletonLvl(l))
      return false;
  // If isUnique is true, then make sure that the last level is unique,
  // that is, when lvlRank == 1, the only compressed level is unique,
  // and when lvlRank > 1, the last singleton is unique.
  return !isUnique || isUniqueLvl(lvlRank - 1);
}

Level mlir::sparse_tensor::SparseTensorType::getAoSCOOStart() const {
  SmallVector<COOSegment> coo = getCOOSegments();
  assert(coo.size() == 1 || coo.empty());
  if (!coo.empty() && coo.front().isAoS()) {
    return coo.front().lvlRange.first;
  }
  return lvlRank;
}

SmallVector<COOSegment>
mlir::sparse_tensor::SparseTensorType::getCOOSegments() const {
  SmallVector<COOSegment> ret;
  if (!hasEncoding() || lvlRank <= 1)
    return ret;

  ArrayRef<LevelType> lts = getLvlTypes();
  Level l = 0;
  while (l < lvlRank) {
    auto lt = lts[l];
    if (lt.isa<LevelFormat::Compressed, LevelFormat::LooseCompressed>()) {
      auto cur = lts.begin() + l;
      auto end = std::find_if(cur + 1, lts.end(), [](LevelType lt) {
        return !lt.isa<LevelFormat::Singleton>();
      });
      unsigned cooLen = std::distance(cur, end);
      if (cooLen > 1) {
        // To support mixed SoA/AoS COO, we should break the segment when the
        // storage scheme changes, for now we faithfully assume that all
        // consecutive singleton levels have the same storage format as verified
        // STEA.
        ret.push_back(COOSegment{std::make_pair(l, l + cooLen),
                                 lts[l + 1].isa<LevelPropNonDefault::SoA>()});
      }
      l += cooLen;
    } else {
      l++;
    }
  }
  return ret;
}

RankedTensorType
mlir::sparse_tensor::SparseTensorType::getCOOType(bool ordered) const {
  SmallVector<LevelType> lvlTypes;
  lvlTypes.reserve(lvlRank);
  // A non-unique compressed level at beginning (unless this is
  // also the last level, then it is unique).
  lvlTypes.push_back(
      *buildLevelType(LevelFormat::Compressed, ordered, lvlRank == 1));
  if (lvlRank > 1) {
    // Followed by n-2 non-unique singleton levels.
    std::fill_n(std::back_inserter(lvlTypes), lvlRank - 2,
                *buildLevelType(LevelFormat::Singleton, ordered, false));
    // Ends by a unique singleton level.
    lvlTypes.push_back(*buildLevelType(LevelFormat::Singleton, ordered, true));
  }
  auto enc = SparseTensorEncodingAttr::get(getContext(), lvlTypes,
                                           getDimToLvl(), getLvlToDim(),
                                           getPosWidth(), getCrdWidth());
  return RankedTensorType::get(getDimShape(), getElementType(), enc);
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
  } else if (isBlockSparsity(map)) {
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
    if (auto binOp = dyn_cast<AffineBinaryOpExpr>(result)) {
      if (result.getKind() == AffineExprKind::FloorDiv) {
        // Position of the dimension in dimToLvl.
        auto pos = dyn_cast<AffineDimExpr>(binOp.getLHS()).getPosition();
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
        auto pos = dyn_cast<AffineDimExpr>(binOp.getLHS()).getPosition();
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

SmallVector<unsigned> mlir::sparse_tensor::getBlockSize(AffineMap dimToLvl) {
  assert(isBlockSparsity(dimToLvl) &&
         "expected dimToLvl to be block sparsity for calling getBlockSize");
  SmallVector<unsigned> blockSize;
  for (auto result : dimToLvl.getResults()) {
    if (auto binOp = dyn_cast<AffineBinaryOpExpr>(result)) {
      if (result.getKind() == AffineExprKind::Mod) {
        blockSize.push_back(
            dyn_cast<AffineConstantExpr>(binOp.getRHS()).getValue());
      }
    } else {
      blockSize.push_back(0);
    }
  }
  return blockSize;
}

bool mlir::sparse_tensor::isBlockSparsity(AffineMap dimToLvl) {
  if (!dimToLvl)
    return false;
  std::map<unsigned, int64_t> coeffientMap;
  bool hasBlock = false;
  for (auto result : dimToLvl.getResults()) {
    if (auto binOp = dyn_cast<AffineBinaryOpExpr>(result)) {
      // Check for "dim op const".
      auto dimOp = dyn_cast<AffineDimExpr>(binOp.getLHS());
      auto conOp = dyn_cast<AffineConstantExpr>(binOp.getRHS());
      if (!dimOp || !conOp || conOp.getValue() <= 0)
        return false;
      // Inspect "dim / const" or "dim % const".
      auto pos = dimOp.getPosition();
      if (binOp.getKind() == AffineExprKind::FloorDiv) {
        // Expect only one floordiv for each dimension.
        if (coeffientMap.find(pos) != coeffientMap.end())
          return false;
        // Record coefficient of the floordiv.
        coeffientMap[pos] = conOp.getValue();
      } else if (binOp.getKind() == AffineExprKind::Mod) {
        // Expect floordiv before mod.
        if (coeffientMap.find(pos) == coeffientMap.end())
          return false;
        // Expect mod to have the same coefficient as floordiv.
        if (conOp.getValue() != coeffientMap[pos])
          return false;
        hasBlock = true;
      } else {
        return false;
      }
    } else if (auto dimOp = dyn_cast<AffineDimExpr>(result)) {
      auto pos = dimOp.getPosition();
      // Expect dim to be unset.
      if (coeffientMap.find(pos) != coeffientMap.end())
        return false;
      coeffientMap[pos] = 0;
    } else {
      return false;
    }
  }
  return hasBlock;
}

bool mlir::sparse_tensor::hasAnyNonIdentityOperandsOrResults(Operation *op) {
  auto hasNonIdentityMap = [](Value v) {
    auto stt = tryGetSparseTensorType(v);
    return stt && !stt->isIdentity();
  };

  return llvm::any_of(op->getOperands(), hasNonIdentityMap) ||
         llvm::any_of(op->getResults(), hasNonIdentityMap);
}

Dimension mlir::sparse_tensor::toDim(SparseTensorEncodingAttr enc, Level l) {
  if (enc) {
    assert(enc.isPermutation() && "Non permutation map not supported");
    if (const auto dimToLvl = enc.getDimToLvl())
      return dimToLvl.getDimPosition(l);
  }
  return l;
}

Level mlir::sparse_tensor::toLvl(SparseTensorEncodingAttr enc, Dimension d) {
  if (enc) {
    assert(enc.isPermutation() && "Non permutation map not supported");
    if (const auto lvlToDim = enc.getLvlToDim())
      return lvlToDim.getDimPosition(d);
  }
  return d;
}

/// We normalized sparse tensor encoding attribute by always using
/// ordered/unique LT such that "compressed_nu_no" and "compressed_nu" (as well
/// as other variants) lead to the same storage specifier type, and stripping
/// irrelevant fields that do not alter the sparse tensor memory layout.
static SparseTensorEncodingAttr
getNormalizedEncodingForSpecifier(SparseTensorEncodingAttr enc) {
  SmallVector<LevelType> lts;
  for (auto lt : enc.getLvlTypes())
    lts.push_back(lt.stripStorageIrrelevantProperties());

  return SparseTensorEncodingAttr::get(
      enc.getContext(), lts,
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

  // Verifies the trailing COO.
  Level cooStartLvl = stt.getAoSCOOStart();
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
                               Level lvl, LevelType lt) -> bool {
    if (fKind == SparseTensorFieldKind::StorageSpec)
      return true;

    Type inputTp = nullptr;
    if (fKind == SparseTensorFieldKind::ValMemRef) {
      inputTp = valTp;
    } else {
      assert(fid == idx && stt.getLvlType(lvl) == lt);
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
  if (getEncoder().isIdentity()) {
    results.assign(getInCrds().begin(), getInCrds().end());
    return success();
  }
  if (getEncoder().isPermutation()) {
    AffineMap perm = getDirection() == CrdTransDirectionKind::dim2lvl
                         ? getEncoder().getDimToLvl()
                         : getEncoder().getLvlToDim();
    for (AffineExpr exp : perm.getResults())
      results.push_back(getInCrds()[cast<AffineDimExpr>(exp).getPosition()]);
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

void LvlOp::build(OpBuilder &builder, OperationState &state, Value source,
                  int64_t index) {
  Value val = builder.create<arith::ConstantIndexOp>(state.location, index);
  return build(builder, state, source, val);
}

LogicalResult LvlOp::verify() {
  if (std::optional<uint64_t> lvl = getConstantLvlIndex()) {
    auto stt = getSparseTensorType(getSource());
    if (static_cast<uint64_t>(lvl.value()) >= stt.getLvlRank())
      emitError("Level index exceeds the rank of the input sparse tensor");
  }
  return success();
}

std::optional<uint64_t> LvlOp::getConstantLvlIndex() {
  return getConstantIntValue(getIndex());
}

Speculation::Speculatability LvlOp::getSpeculatability() {
  auto constantIndex = getConstantLvlIndex();
  if (!constantIndex)
    return Speculation::NotSpeculatable;

  assert(constantIndex <
         cast<RankedTensorType>(getSource().getType()).getRank());
  return Speculation::Speculatable;
}

OpFoldResult LvlOp::fold(FoldAdaptor adaptor) {
  auto lvlIndex = llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getIndex());
  if (!lvlIndex)
    return {};

  Level lvl = lvlIndex.getAPSInt().getZExtValue();
  auto stt = getSparseTensorType(getSource());
  if (lvl >= stt.getLvlRank()) {
    // Follows the same convention used by tensor.dim operation. Out of bound
    // indices produce undefined behavior but are still valid IR. Don't choke on
    // them.
    return {};
  }

  // Helper lambda to build an IndexAttr.
  auto getIndexAttr = [this](int64_t lvlSz) {
    return IntegerAttr::get(IndexType::get(getContext()), APInt(64, lvlSz));
  };

  SmallVector<Size> lvlShape = stt.getLvlShape();
  if (!ShapedType::isDynamic(lvlShape[lvl]))
    return getIndexAttr(lvlShape[lvl]);

  return {};
}

void ReinterpretMapOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             SparseTensorEncodingAttr dstEnc, Value source) {
  auto srcStt = getSparseTensorType(source);
  SmallVector<int64_t> srcLvlShape = srcStt.getLvlShape();
  SmallVector<int64_t> dstDimShape =
      dstEnc.translateShape(srcLvlShape, CrdTransDirectionKind::lvl2dim);
  auto dstTp =
      RankedTensorType::get(dstDimShape, srcStt.getElementType(), dstEnc);
  return build(odsBuilder, odsState, dstTp, source);
}

LogicalResult ReinterpretMapOp::verify() {
  auto srcStt = getSparseTensorType(getSource());
  auto dstStt = getSparseTensorType(getDest());
  ArrayRef<LevelType> srcLvlTps = srcStt.getLvlTypes();
  ArrayRef<LevelType> dstLvlTps = dstStt.getLvlTypes();

  if (srcLvlTps.size() != dstLvlTps.size())
    return emitError("Level rank mismatch between source/dest tensors");

  for (auto [srcLvlTp, dstLvlTp] : llvm::zip(srcLvlTps, dstLvlTps))
    if (srcLvlTp != dstLvlTp)
      return emitError("Level type mismatch between source/dest tensors");

  if (srcStt.getPosWidth() != dstStt.getPosWidth() ||
      srcStt.getCrdWidth() != dstStt.getCrdWidth()) {
    return emitError("Crd/Pos width mismatch between source/dest tensors");
  }

  if (srcStt.getElementType() != dstStt.getElementType())
    return emitError("Element type mismatch between source/dest tensors");

  SmallVector<Size> srcLvlShape = srcStt.getLvlShape();
  SmallVector<Size> dstLvlShape = dstStt.getLvlShape();
  for (auto [srcLvlSz, dstLvlSz] : llvm::zip(srcLvlShape, dstLvlShape)) {
    if (srcLvlSz != dstLvlSz) {
      // Should we allow one side to be dynamic size, e.g., <?x?> should be
      // compatible to <3x4>? For now, we require all the level sizes to be
      // *exactly* matched for simplicity.
      return emitError("Level size mismatch between source/dest tensors");
    }
  }

  return success();
}

OpFoldResult ReinterpretMapOp::fold(FoldAdaptor adaptor) {
  if (getSource().getType() == getDest().getType())
    return getSource();

  if (auto def = getSource().getDefiningOp<ReinterpretMapOp>()) {
    // A -> B, B -> A ==> A
    if (def.getSource().getType() == getDest().getType())
      return def.getSource();
  }
  return {};
}

template <typename ToBufferOp>
static LogicalResult inferSparseBufferType(ValueRange ops, DictionaryAttr attr,
                                           OpaqueProperties prop,
                                           RegionRange region,
                                           SmallVectorImpl<mlir::Type> &ret) {
  typename ToBufferOp::Adaptor adaptor(ops, attr, prop, region);
  SparseTensorType stt = getSparseTensorType(adaptor.getTensor());
  Type elemTp = nullptr;
  bool withStride = false;
  if constexpr (std::is_same_v<ToBufferOp, ToPositionsOp>) {
    elemTp = stt.getPosType();
  } else if constexpr (std::is_same_v<ToBufferOp, ToCoordinatesOp> ||
                       std::is_same_v<ToBufferOp, ToCoordinatesBufferOp>) {
    elemTp = stt.getCrdType();
    if constexpr (std::is_same_v<ToBufferOp, ToCoordinatesOp>)
      withStride = stt.getAoSCOOStart() <= adaptor.getLevel();
  } else if constexpr (std::is_same_v<ToBufferOp, ToValuesOp>) {
    elemTp = stt.getElementType();
  }

  assert(elemTp && "unhandled operation.");
  SmallVector<int64_t> bufShape = stt.getBatchLvlShape();
  bufShape.push_back(ShapedType::kDynamic);

  auto layout = withStride ? StridedLayoutAttr::StridedLayoutAttr::get(
                                 stt.getContext(), ShapedType::kDynamic,
                                 {ShapedType::kDynamic})
                           : StridedLayoutAttr();
  ret.emplace_back(MemRefType::get(bufShape, elemTp, layout));
  return success();
}

LogicalResult ToPositionsOp::verify() {
  auto stt = getSparseTensorType(getTensor());
  if (failed(lvlIsInBounds(getLevel(), getTensor())))
    return emitError("requested level is out of bounds");
  if (failed(isMatchingWidth(getResult(), stt.getPosWidth())))
    return emitError("unexpected type for positions");
  return success();
}

LogicalResult
ToPositionsOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location> loc,
                                ValueRange ops, DictionaryAttr attr,
                                OpaqueProperties prop, RegionRange region,
                                SmallVectorImpl<mlir::Type> &ret) {
  return inferSparseBufferType<ToPositionsOp>(ops, attr, prop, region, ret);
}

LogicalResult ToCoordinatesOp::verify() {
  auto stt = getSparseTensorType(getTensor());
  if (failed(lvlIsInBounds(getLevel(), getTensor())))
    return emitError("requested level is out of bounds");
  if (failed(isMatchingWidth(getResult(), stt.getCrdWidth())))
    return emitError("unexpected type for coordinates");
  return success();
}

LogicalResult
ToCoordinatesOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location> loc,
                                  ValueRange ops, DictionaryAttr attr,
                                  OpaqueProperties prop, RegionRange region,
                                  SmallVectorImpl<mlir::Type> &ret) {
  return inferSparseBufferType<ToCoordinatesOp>(ops, attr, prop, region, ret);
}

LogicalResult ToCoordinatesBufferOp::verify() {
  auto stt = getSparseTensorType(getTensor());
  if (stt.getAoSCOOStart() >= stt.getLvlRank())
    return emitError("expected sparse tensor with a COO region");
  return success();
}

LogicalResult ToCoordinatesBufferOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange ops,
    DictionaryAttr attr, OpaqueProperties prop, RegionRange region,
    SmallVectorImpl<mlir::Type> &ret) {
  return inferSparseBufferType<ToCoordinatesBufferOp>(ops, attr, prop, region,
                                                      ret);
}

LogicalResult ToValuesOp::verify() {
  auto stt = getSparseTensorType(getTensor());
  auto mtp = getMemRefType(getResult());
  if (stt.getElementType() != mtp.getElementType())
    return emitError("unexpected mismatch in element types");
  return success();
}

LogicalResult ToValuesOp::inferReturnTypes(MLIRContext *ctx,
                                           std::optional<Location> loc,
                                           ValueRange ops, DictionaryAttr attr,
                                           OpaqueProperties prop,
                                           RegionRange region,
                                           SmallVectorImpl<mlir::Type> &ret) {
  return inferSparseBufferType<ToValuesOp>(ops, attr, prop, region, ret);
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
  return verifySparsifierGetterSetter(getSpecifierKind(), getLevel(),
                                      getSpecifier(), getOperation());
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
  return verifySparsifierGetterSetter(getSpecifierKind(), getLevel(),
                                      getSpecifier(), getOperation());
}

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
    if (failed(verifyNumBlockArgs(this, overlap, "overlap",
                                  TypeRange{leftType, rightType}, outputType)))
      return failure();
  }
  if (!left.empty()) {
    if (failed(verifyNumBlockArgs(this, left, "left", TypeRange{leftType},
                                  outputType)))
      return failure();
  } else if (getLeftIdentity()) {
    if (leftType != outputType)
      return emitError("left=identity requires first argument to have the same "
                       "type as the output");
  }
  if (!right.empty()) {
    if (failed(verifyNumBlockArgs(this, right, "right", TypeRange{rightType},
                                  outputType)))
      return failure();
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
    if (failed(verifyNumBlockArgs(this, present, "present",
                                  TypeRange{inputType}, outputType)))
      return failure();
  }
  Region &absent = getAbsentRegion();
  if (!absent.empty()) {
    if (failed(verifyNumBlockArgs(this, absent, "absent", TypeRange{},
                                  outputType)))
      return failure();
    // Absent branch can only yield invariant values.
    Block *absentBlock = &absent.front();
    Block *parent = getOperation()->getBlock();
    Value absentVal = cast<YieldOp>(absentBlock->getTerminator()).getResult();
    if (auto arg = dyn_cast<BlockArgument>(absentVal)) {
      if (arg.getOwner() == parent)
        return emitError("absent region cannot yield linalg argument");
    } else if (Operation *def = absentVal.getDefiningOp()) {
      if (!isa<arith::ConstantOp>(def) &&
          (def->getBlock() == absentBlock || def->getBlock() == parent))
        return emitError("absent region cannot yield locally computed value");
    }
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
    const Size dstSh = dstTp.getDimShape()[d];
    if (d == concatDim) {
      if (!ShapedType::isDynamic(dstSh)) {
        // If we reach here, then all inputs have static shapes.  So we
        // can use `getDimShape()[d]` instead of `*getDynamicDimSize(d)`
        // to avoid redundant assertions in the loop.
        Size sumSz = 0;
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
      Size prev = dstSh;
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

  if (getOrder().has_value() && getOrder()->getNumDims() != t.getLvlRank())
    return emitError("Level traverse order does not match tensor's level rank");

  if (dimRank + 1 + getInitArgs().size() != args.size())
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

  if (!srcStt.isCOOType() || !dstStt.isCOOType())
    emitError("Expected COO sparse tensors only");

  if (!srcStt.hasSameDimToLvl(dstStt))
    emitError("Unmatched dim2lvl map between input and result COO");

  if (srcStt.getPosType() != dstStt.getPosType() ||
      srcStt.getCrdType() != dstStt.getCrdType() ||
      srcStt.getElementType() != dstStt.getElementType())
    emitError("Unmatched storage format between input and result COO");

  return success();
}

LogicalResult ReduceOp::verify() {
  Type inputType = getX().getType();
  Region &formula = getRegion();
  return verifyNumBlockArgs(this, formula, "reduce",
                            TypeRange{inputType, inputType}, inputType);
}

LogicalResult SelectOp::verify() {
  Builder b(getContext());
  Type inputType = getX().getType();
  Type boolType = b.getI1Type();
  Region &formula = getRegion();
  return verifyNumBlockArgs(this, formula, "select", TypeRange{inputType},
                            boolType);
}

LogicalResult SortOp::verify() {
  AffineMap xPerm = getPermMap();
  uint64_t nx = xPerm.getNumDims();
  if (nx < 1)
    emitError(llvm::formatv("Expected rank(perm_map) > 1, got {0}", nx));

  if (!xPerm.isPermutation())
    emitError(llvm::formatv("Expected a permutation map, got {0}", xPerm));

  // We can't check the size of the buffers when n or buffer dimensions aren't
  // compile-time constants.
  std::optional<int64_t> cn = getConstantIntValue(getN());
  if (!cn)
    return success();

  // Verify dimensions.
  const auto checkDim = [&](Value v, Size minSize, const char *message) {
    const Size sh = getMemRefType(v).getShape()[0];
    if (!ShapedType::isDynamic(sh) && sh < minSize)
      emitError(llvm::formatv("{0} got {1} < {2}", message, sh, minSize));
  };
  uint64_t n = cn.value();
  uint64_t ny = 0;
  if (auto nyAttr = getNyAttr())
    ny = nyAttr.getInt();
  checkDim(getXy(), n * (nx + ny),
           "Expected dimension(xy) >= n * (rank(perm_map) + ny)");
  for (Value opnd : getYs())
    checkDim(opnd, n, "Expected dimension(y) >= n");

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

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *SparseTensorDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  if (auto op = arith::ConstantOp::materialize(builder, value, type, loc))
    return op;
  return nullptr;
}

namespace {
struct SparseTensorAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (attr.isa<SparseTensorEncodingAttr>()) {
      os << "sparse";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void SparseTensorDialect::initialize() {
  addInterface<SparseTensorAsmDialectInterface>();
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
