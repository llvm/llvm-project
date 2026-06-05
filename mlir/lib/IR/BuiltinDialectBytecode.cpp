//===- BuiltinDialectBytecode.cpp - Builtin Bytecode Implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BuiltinDialectBytecode.h"
#include "AttributeDetail.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cstdint>

using namespace mlir;

//===----------------------------------------------------------------------===//
// BuiltinDialectBytecodeInterface
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// TODO: Move these to separate file.

// Returns the bitwidth if known, else return std::nullopt.
static std::optional<unsigned> getIntegerBitWidth(DialectBytecodeReader &reader,
                                                  Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (llvm::isa<IndexType>(type))
    return IndexType::kInternalStorageBitWidth;
  reader.emitError()
      << "expected integer or index type for IntegerAttr, but got: " << type;
  return std::nullopt;
}

static LogicalResult readAPIntWithKnownWidth(DialectBytecodeReader &reader,
                                             Type type, FailureOr<APInt> &val) {
  std::optional<unsigned> bitWidth = getIntegerBitWidth(reader, type);
  // getIntegerBitWidth returns std::nullopt and emits an error for unsupported
  // types. Bail out early to avoid creating a zero-width APInt with a non-zero
  // value.
  if (!bitWidth)
    return failure();
  val = reader.readAPIntWithKnownWidth(*bitWidth);
  return val;
}

static LogicalResult
readAPFloatWithKnownSemantics(DialectBytecodeReader &reader, Type type,
                              FailureOr<APFloat> &val) {
  auto ftype = dyn_cast<FloatType>(type);
  if (!ftype)
    return failure();
  val = reader.readAPFloatWithKnownSemantics(ftype.getFloatSemantics());
  return success();
}

LogicalResult
readPotentiallySplatString(DialectBytecodeReader &reader, ShapedType type,
                           bool isSplat,
                           SmallVectorImpl<StringRef> &rawStringData) {
  rawStringData.resize(isSplat ? 1 : type.getNumElements());
  for (StringRef &value : rawStringData)
    if (failed(reader.readString(value)))
      return failure();
  return success();
}

static void writePotentiallySplatString(DialectBytecodeWriter &writer,
                                        DenseStringElementsAttr attr) {
  bool isSplat = attr.isSplat();
  if (isSplat)
    return writer.writeOwnedString(attr.getRawStringData().front());

  for (StringRef str : attr.getRawStringData())
    writer.writeOwnedString(str);
}

//===----------------------------------------------------------------------===//
// AffineExpr / AffineMap bytecode helpers
//===----------------------------------------------------------------------===//

// AffineExpr kind encoding:
// Extra kinds may be appended here but the existing ones and their ordering
// should not be changed.
enum class AffineExprBytecodeKind : uint64_t {
  DimId = 0,
  SymbolId = 1,
  Constant = 2,
  Add = 3,
  Mul = 4,
  Mod = 5,
  FloorDiv = 6,
  CeilDiv = 7
};

// AffineMap kind encoding. These are packed into the low 2 bits of the header
// varint and fixed.
enum class AffineMapBytecodeKind : unsigned {
  Identity = 0,
  Permutation = 1,
  ProjectedPermutation = 2,
  General = 3
};

/// Convert a binary AffineExprKind to its bytecode wire encoding.
static AffineExprBytecodeKind toBytecodeKind(AffineExprKind k) {
  switch (k) {
  case AffineExprKind::Add:
    return AffineExprBytecodeKind::Add;
  case AffineExprKind::Mul:
    return AffineExprBytecodeKind::Mul;
  case AffineExprKind::Mod:
    return AffineExprBytecodeKind::Mod;
  case AffineExprKind::FloorDiv:
    return AffineExprBytecodeKind::FloorDiv;
  case AffineExprKind::CeilDiv:
    return AffineExprBytecodeKind::CeilDiv;
  default:
    llvm_unreachable("not a binary AffineExprKind");
  }
}

/// Convert a bytecode wire value back to a binary AffineExprKind.
/// Caller must guarantee `kind` is one of the binary operator values.
static AffineExprKind fromBytecodeKind(uint64_t kind) {
  switch (kind) {
  case static_cast<uint64_t>(AffineExprBytecodeKind::Add):
    return AffineExprKind::Add;
  case static_cast<uint64_t>(AffineExprBytecodeKind::Mul):
    return AffineExprKind::Mul;
  case static_cast<uint64_t>(AffineExprBytecodeKind::Mod):
    return AffineExprKind::Mod;
  case static_cast<uint64_t>(AffineExprBytecodeKind::FloorDiv):
    return AffineExprKind::FloorDiv;
  case static_cast<uint64_t>(AffineExprBytecodeKind::CeilDiv):
    return AffineExprKind::CeilDiv;
  }
  llvm_unreachable("not a binary AffineExprBytecodeKind");
}

/// Read a single AffineExpr using iterative prefix decoding. The wire format
/// is prefix order (operator and then children), which is self-delimiting.
/// Instead of C++ recursion the reader uses an explicit work stack, bounding
/// memory to O(depth) and eliminating stack-overflow risk on malicious input.
static FailureOr<AffineExpr> readAffineExpr(DialectBytecodeReader &reader,
                                            MLIRContext *context) {
  // A work-stack item is either ReadOperand (0) or a combine marker whose
  // payload is an AffineExprKind.
  struct WorkItem {
    bool isCombine;
    AffineExprKind combineKind; // only valid when isCombine == true
    static WorkItem read() { return {false, {}}; }
    static WorkItem combine(AffineExprKind k) { return {true, k}; }
  };

  SmallVector<WorkItem, 16> work;
  SmallVector<AffineExpr, 8> operands;
  work.push_back(WorkItem::read());

  while (!work.empty()) {
    // Bound total iterations to catch malformed input.
    if (work.size() > 128)
      return reader.emitError("AffineExpr work stack overflow"), failure();

    WorkItem item = work.pop_back_val();

    if (item.isCombine) {
      // Pop two operands and combine.
      if (operands.size() < 2)
        return reader.emitError("malformed AffineExpr: operand underflow"),
               failure();
      AffineExpr rhs = operands.pop_back_val();
      AffineExpr lhs = operands.pop_back_val();
      operands.push_back(getAffineBinaryOpExpr(item.combineKind, lhs, rhs));
      continue;
    }

    // ReadOperand: read the next token.
    uint64_t kind;
    if (failed(reader.readVarInt(kind)))
      return failure();

    // Switch on the raw uint64_t to keep the default case valid for
    // unknown/future wire values without triggering -Wcovered-switch-default.
    switch (kind) {
    case static_cast<uint64_t>(AffineExprBytecodeKind::DimId): {
      uint64_t position;
      if (failed(reader.readVarInt(position)))
        return failure();
      operands.push_back(getAffineDimExpr(position, context));
      break;
    }
    case static_cast<uint64_t>(AffineExprBytecodeKind::SymbolId): {
      uint64_t position;
      if (failed(reader.readVarInt(position)))
        return failure();
      operands.push_back(getAffineSymbolExpr(position, context));
      break;
    }
    case static_cast<uint64_t>(AffineExprBytecodeKind::Constant): {
      int64_t value;
      if (failed(reader.readSignedVarInt(value)))
        return failure();
      operands.push_back(getAffineConstantExpr(value, context));
      break;
    }
    case static_cast<uint64_t>(AffineExprBytecodeKind::Add):
    case static_cast<uint64_t>(AffineExprBytecodeKind::Mul):
    case static_cast<uint64_t>(AffineExprBytecodeKind::Mod):
    case static_cast<uint64_t>(AffineExprBytecodeKind::FloorDiv):
    case static_cast<uint64_t>(AffineExprBytecodeKind::CeilDiv): {
      // Schedule: read RHS, read LHS, then combine.
      // Work stack is LIFO, so push in reverse order.
      work.push_back(WorkItem::combine(fromBytecodeKind(kind)));
      work.push_back(WorkItem::read()); // RHS
      work.push_back(WorkItem::read()); // LHS
      break;
    }
    default:
      return reader.emitError("unknown AffineExpr kind: ") << kind, failure();
    }
  }

  if (operands.size() != 1)
    return reader.emitError("malformed AffineExpr: expected single result"),
           failure();

  return operands.front();
}

/// Write an AffineExpr in prefix order (operator first, then children).
static void writeAffineExpr(DialectBytecodeWriter &writer, AffineExpr expr) {
  switch (expr.getKind()) {
  case AffineExprKind::DimId:
    writer.writeVarInt(static_cast<uint64_t>(AffineExprBytecodeKind::DimId));
    writer.writeVarInt(cast<AffineDimExpr>(expr).getPosition());
    break;
  case AffineExprKind::SymbolId:
    writer.writeVarInt(static_cast<uint64_t>(AffineExprBytecodeKind::SymbolId));
    writer.writeVarInt(cast<AffineSymbolExpr>(expr).getPosition());
    break;
  case AffineExprKind::Constant:
    writer.writeVarInt(static_cast<uint64_t>(AffineExprBytecodeKind::Constant));
    writer.writeSignedVarInt(cast<AffineConstantExpr>(expr).getValue());
    break;
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::Mod:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    // Write operator first (prefix order).
    writer.writeVarInt(static_cast<uint64_t>(toBytecodeKind(expr.getKind())));
    auto binExpr = cast<AffineBinaryOpExpr>(expr);
    writeAffineExpr(writer, binExpr.getLHS());
    writeAffineExpr(writer, binExpr.getRHS());
    break;
  }
  }
}

/// Read an AffineMap with packed kind header.
///
/// AffineMap :=
///   header(varint)        // (numDims << 2) | mapKind
///   payload               // depends on mapKind
///
/// The header is there for concise encoding of the most common occuring cases.
///
/// mapKind = header & 0x3:
///   Identity(0): no further data
///   Permutation(1): positions(varint*)
///   ProjectedPermutation(2): numResults(varint), positions(varint*)
///   General(3): numSymbols(varint), numResults(varint), results(AffineExpr*)
static LogicalResult readAffineMap(DialectBytecodeReader &reader,
                                   MLIRContext *context, AffineMap &map) {
  uint64_t header;
  if (failed(reader.readVarInt(header)))
    return failure();

  // Keep as unsigned to avoid -Wcovered-switch-default below.
  unsigned mapKind = header & 0x3;
  unsigned numDims = header >> 2;

  switch (mapKind) {
  case static_cast<unsigned>(AffineMapBytecodeKind::Identity):
    map = AffineMap::getMultiDimIdentityMap(numDims, context);
    return success();

  case static_cast<unsigned>(AffineMapBytecodeKind::Permutation): {
    SmallVector<unsigned> perm(numDims);
    for (unsigned i = 0; i < numDims; ++i) {
      uint64_t pos;
      if (failed(reader.readVarInt(pos)))
        return failure();
      perm[i] = pos;
    }
    map = AffineMap::getPermutationMap(perm, context);
    return success();
  }

  case static_cast<unsigned>(AffineMapBytecodeKind::ProjectedPermutation): {
    uint64_t numResults;
    if (failed(reader.readVarInt(numResults)))
      return failure();
    SmallVector<AffineExpr> results;
    results.reserve(numResults);
    for (uint64_t i = 0; i < numResults; ++i) {
      uint64_t pos;
      if (failed(reader.readVarInt(pos)))
        return failure();
      results.push_back(getAffineDimExpr(pos, context));
    }
    map = AffineMap::get(numDims, /*numSymbols=*/0, results, context);
    return success();
  }

  case static_cast<unsigned>(AffineMapBytecodeKind::General): {
    uint64_t numSymbols, numResults;
    if (failed(reader.readVarInt(numSymbols)) ||
        failed(reader.readVarInt(numResults)))
      return failure();
    SmallVector<AffineExpr> results;
    results.reserve(numResults);
    for (uint64_t i = 0; i < numResults; ++i) {
      auto expr = readAffineExpr(reader, context);
      if (failed(expr))
        return failure();
      results.push_back(*expr);
    }
    map = AffineMap::get(numDims, numSymbols, results, context);
    return success();
  }

  default:
    return reader.emitError("unknown AffineMap kind: ")
               << static_cast<unsigned>(mapKind),
           failure();
  }
}

/// Write an AffineMap with packed kind header (see readAffineMap for format).
static void writeAffineMap(DialectBytecodeWriter &writer, AffineMapAttr attr) {
  AffineMap map = attr.getValue();
  unsigned numDims = map.getNumDims();

  // Identity maps: (d0, d1, ..., d_{n-1}) -> (d0, d1, ..., d_{n-1})
  // Note: isIdentity() does not check numSymbols, so guard explicitly.
  if (map.getNumSymbols() == 0 && map.isIdentity()) {
    writer.writeVarInt((numDims << 2) |
                       static_cast<unsigned>(AffineMapBytecodeKind::Identity));
    return;
  }

  // Permutation maps: numResults == numDims, each result is a unique dim
  if (map.isPermutation()) {
    writer.writeVarInt(
        (numDims << 2) |
        static_cast<unsigned>(AffineMapBytecodeKind::Permutation));
    for (unsigned i = 0; i < map.getNumResults(); ++i)
      writer.writeVarInt(map.getDimPosition(i));
    return;
  }

  // Projected permutation maps (symbol-less): subset of dims
  if (map.getNumSymbols() == 0 && map.isProjectedPermutation()) {
    writer.writeVarInt(
        (numDims << 2) |
        static_cast<unsigned>(AffineMapBytecodeKind::ProjectedPermutation));
    writer.writeVarInt(map.getNumResults());
    for (unsigned i = 0; i < map.getNumResults(); ++i)
      writer.writeVarInt(map.getDimPosition(i));
    return;
  }

  // General case
  writer.writeVarInt((numDims << 2) |
                     static_cast<unsigned>(AffineMapBytecodeKind::General));
  writer.writeVarInt(map.getNumSymbols());
  writer.writeVarInt(map.getNumResults());
  for (AffineExpr expr : map.getResults())
    writeAffineExpr(writer, expr);
}

static FileLineColRange getFileLineColRange(MLIRContext *context,
                                            StringAttr filename,
                                            ArrayRef<uint64_t> lineCols) {
  switch (lineCols.size()) {
  case 0:
    return FileLineColRange::get(filename);
  case 1:
    return FileLineColRange::get(filename, lineCols[0]);
  case 2:
    return FileLineColRange::get(filename, lineCols[0], lineCols[1]);
  case 3:
    return FileLineColRange::get(filename, lineCols[0], lineCols[1],
                                 lineCols[2]);
  case 4:
    return FileLineColRange::get(filename, lineCols[0], lineCols[1],
                                 lineCols[2], lineCols[3]);
  default:
    return {};
  }
}

static LogicalResult
readFileLineColRangeLocs(DialectBytecodeReader &reader,
                         SmallVectorImpl<uint64_t> &lineCols) {
  return reader.readList(
      lineCols, [&reader](uint64_t &val) { return reader.readVarInt(val); });
}

static void writeFileLineColRangeLocs(DialectBytecodeWriter &writer,
                                      FileLineColRange range) {
  if (range.getStartLine() == 0 && range.getStartColumn() == 0 &&
      range.getEndLine() == 0 && range.getEndColumn() == 0) {
    writer.writeVarInt(0);
    return;
  }
  if (range.getStartColumn() == 0 &&
      range.getStartLine() == range.getEndLine()) {
    writer.writeVarInt(1);
    writer.writeVarInt(range.getStartLine());
    return;
  }
  // The single file:line:col is handled by other writer, but checked here for
  // completeness.
  if (range.getEndColumn() == range.getStartColumn() &&
      range.getStartLine() == range.getEndLine()) {
    writer.writeVarInt(2);
    writer.writeVarInt(range.getStartLine());
    writer.writeVarInt(range.getStartColumn());
    return;
  }
  if (range.getStartLine() == range.getEndLine()) {
    writer.writeVarInt(3);
    writer.writeVarInt(range.getStartLine());
    writer.writeVarInt(range.getStartColumn());
    writer.writeVarInt(range.getEndColumn());
    return;
  }
  writer.writeVarInt(4);
  writer.writeVarInt(range.getStartLine());
  writer.writeVarInt(range.getStartColumn());
  writer.writeVarInt(range.getEndLine());
  writer.writeVarInt(range.getEndColumn());
}

static LogicalResult
readDenseTypedElementsAttr(DialectBytecodeReader &reader, ShapedType type,
                           SmallVectorImpl<char> &rawData) {
  // Validate that the element type implements DenseElementTypeInterface.
  // Without this check, downstream code unconditionally calls
  // getDenseElementBitWidth() which asserts on unsupported types.
  if (!llvm::isa<DenseElementType>(type.getElementType())) {
    reader.emitError() << "DenseTypedElementsAttr element type must implement "
                          "DenseElementTypeInterface, but got: "
                       << type.getElementType();
    return failure();
  }

  ArrayRef<char> blob;
  if (failed(reader.readBlob(blob)))
    return failure();

  // If the type is not i1, just copy the blob.
  if (!type.getElementType().isInteger(1)) {
    rawData.append(blob.begin(), blob.end());
    return success();
  }

  // Check to see if this is using the packed format.
  // Note: this could be asserted instead as this should be the case. But we
  // did have period where the unpacked was being serialized, this enables
  // consuming those still and the check for which case we are in is pretty
  // cheap.
  size_t numElements = type.getNumElements();
  size_t packedSize = llvm::divideCeil(numElements, 8);

  // Unpack splats to single element 0x01 to match unpacked splat format.
  if (blob.size() == 1 && blob[0] == static_cast<char>(~0x00)) {
    rawData.resize(1);
    rawData[0] = 0x01;
    return success();
  }

  // Unpack the blob if it's packed.
  // Splat and blob.size() == packedSize for all N<=8 elements are ambiguous,
  // non 0xFF means not splat so must be unpacked.
  if (blob.size() == packedSize && blob.size() != numElements) {
    rawData.resize(numElements);
    for (size_t i = 0; i < numElements; ++i)
      rawData[i] = (blob[i / 8] & (1 << (i % 8))) ? 1 : 0;
    return success();
  }
  // Otherwise, fallback to the default behavior.
  rawData.append(blob.begin(), blob.end());
  return success();
}

static void writeDenseTypedElementsAttr(DialectBytecodeWriter &writer,
                                        DenseTypedElementsAttr attr) {
  // Check to see if this is an i1 dense attribute.
  if (attr.getElementType().isInteger(1)) {
    // Pack the data.
    SmallVector<char> data;
    ArrayRef<char> rawData = attr.getRawData();

    // If the attribute is a splat, we can just splat the value directly.
    // Use 0xFF to avoid ambiguity with packed format of <=8 elements,
    // written ~0x00 to ensure proper compilation with signed chars.
    if (attr.isSplat()) {
      data.resize(1);
      data[0] = rawData[0] ? ~0x00 : 0x00;
      writer.writeUnownedBlob(data);
      return;
    }

    size_t numElements = attr.getNumElements();
    data.resize(llvm::divideCeil(numElements, 8));
    // Otherwise, pack the data manually.
    for (size_t i = 0; i < numElements; ++i)
      if (rawData[i])
        data[i / 8] |= (1 << (i % 8));
    writer.writeUnownedBlob(data);
    return;
  }

  writer.writeOwnedBlob(attr.getRawData());
}

#include "mlir/IR/BuiltinDialectBytecode.cpp.inc"

/// This class implements the bytecode interface for the builtin dialect.
struct BuiltinDialectBytecodeInterface : public BytecodeDialectInterface {
  BuiltinDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }

  //===--------------------------------------------------------------------===//
  // Version

  void writeVersion(DialectBytecodeWriter &writer) const override {
    auto configVersion = writer.getDialectVersion(getDialect()->getNamespace());
    // Write version set in config.
    if (succeeded(configVersion)) {
      auto *version =
          static_cast<const BuiltinDialectVersion *>(*configVersion);
      writer.writeVarInt(static_cast<uint64_t>(version->getVersion()));
      return;
    }
    // Else, write current set version version if not 0.
    if (auto version = cast<BuiltinDialect>(getDialect())->getVersion();
        version && version->getVersion() > 0) {
      writer.writeVarInt(static_cast<uint64_t>(version->getVersion()));
    }
  }

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const override {
    uint64_t version;
    if (failed(reader.readVarInt(version)))
      return nullptr;

    auto dialectVersion = std::make_unique<BuiltinDialectVersion>(version);
    if (BuiltinDialectVersion::getCurrentVersion() < *dialectVersion) {
      reader.emitError()
          << "reading newer builtin dialect version than supported";
      return nullptr;
    }

    return dialectVersion;
  }
};
} // namespace

void builtin_dialect_detail::addBytecodeInterface(BuiltinDialect *dialect) {
  dialect->addInterfaces<BuiltinDialectBytecodeInterface>();
}
