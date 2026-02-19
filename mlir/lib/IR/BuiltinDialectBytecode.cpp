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

// Returns the bitwidth if known, else return 0.
static unsigned getIntegerBitWidth(DialectBytecodeReader &reader, Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth();
  }
  if (llvm::isa<IndexType>(type)) {
    return IndexType::kInternalStorageBitWidth;
  }
  reader.emitError()
      << "expected integer or index type for IntegerAttr, but got: " << type;
  return 0;
}

static LogicalResult readAPIntWithKnownWidth(DialectBytecodeReader &reader,
                                             Type type, FailureOr<APInt> &val) {
  unsigned bitWidth = getIntegerBitWidth(reader, type);
  val = reader.readAPIntWithKnownWidth(bitWidth);
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
readDenseIntOrFPElementsAttr(DialectBytecodeReader &reader, ShapedType type,
                             SmallVectorImpl<char> &rawData) {
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
  if (blob.size() == packedSize && blob.size() != numElements &&
      blob.size() != 1) {
    // Unpack the blob.
    rawData.resize(numElements);
    for (size_t i = 0; i < numElements; ++i)
      rawData[i] = (blob[i / 8] & (1 << (i % 8))) ? 0xFF : 0x00;
    return success();
  }
  // Otherwise, fallback to the default behavior.
  rawData.append(blob.begin(), blob.end());
  return success();
}

static void writeDenseIntOrFPElementsAttr(DialectBytecodeWriter &writer,
                                          DenseIntOrFPElementsAttr attr) {
  // Check to see if this is an i1 dense attribute.
  if (attr.getElementType().isInteger(1)) {
    // Pack the data.
    SmallVector<char> data;
    ArrayRef<char> rawData = attr.getRawData();

    // If the attribute is a splat, we can just splat the value directly.
    if (attr.isSplat()) {
      data.resize(1);
      data[0] = rawData[0] ? 0xFF : 0x00;
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
};
} // namespace

void builtin_dialect_detail::addBytecodeInterface(BuiltinDialect *dialect) {
  dialect->addInterfaces<BuiltinDialectBytecodeInterface>();
}
