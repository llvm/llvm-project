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
