//===- AMXDialect.cpp - MLIR AMX ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMX dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "mlir/Dialect/AMX/AMXDialect.cpp.inc"

void amx::AMXDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/AMX/AMXTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/AMX/AMX.cpp.inc"
      >();
}

/// Verify that AMX supports the implied tile shape.
static LogicalResult verifyTileSize(Operation *op, amx::TileType tp) {
  const unsigned kMaxRows = 16;
  const unsigned kBitsPerRow = 64 * 8;
  unsigned col = tp.getDimSize(1) * tp.getElementType().getIntOrFloatBitWidth();
  if (tp.getDimSize(0) > kMaxRows)
    return op->emitOpError("bad row height: ") << tp.getDimSize(0);
  if (col > kBitsPerRow || col & 0x1f)
    return op->emitOpError("bad column width: ") << (col >> 3);
  return success();
}

/// Verify that AMX supports the multiplication.
static LogicalResult verifyMultShape(Operation *op, amx::TileType atp,
                                     amx::TileType btp, amx::TileType ctp,
                                     unsigned scale) {
  unsigned am = atp.getDimSize(0), ak = atp.getDimSize(1) >> scale;
  unsigned bk = btp.getDimSize(0), bn = btp.getDimSize(1) >> scale;
  unsigned cm = ctp.getDimSize(0), cn = ctp.getDimSize(1);
  if (cm != am || cn != bn || ak != bk)
    return op->emitOpError("bad mult shape: ")
           << cm << " x " << cn << " x " << ak;
  return success();
}

LogicalResult amx::TileZeroOp::verify() {
  return verifyTileSize(*this, getTileType());
}

LogicalResult amx::TileLoadOp::verify() {
  unsigned rank = getMemRefType().getRank();
  if (getIndices().size() != rank)
    return emitOpError("requires ") << rank << " indices";
  return verifyTileSize(*this, getTileType());
}

LogicalResult amx::TileStoreOp::verify() {
  unsigned rank = getMemRefType().getRank();
  if (getIndices().size() != rank)
    return emitOpError("requires ") << rank << " indices";
  return verifyTileSize(*this, getTileType());
}

LogicalResult amx::TileMulFOp::verify() {
  amx::TileType aType = getLhsTileType();
  amx::TileType bType = getRhsTileType();
  amx::TileType cType = getTileType();
  if (failed(verifyTileSize(*this, aType)) ||
      failed(verifyTileSize(*this, bType)) ||
      failed(verifyTileSize(*this, cType)) ||
      failed(verifyMultShape(*this, aType, bType, cType, 1)))
    return failure();
  Type ta = aType.getElementType();
  Type tb = bType.getElementType();
  Type tc = cType.getElementType();
  if ((!ta.isBF16() && !ta.isF16()) || (ta != tb) || !tc.isF32())
    return emitOpError("unsupported type combination");
  return success();
}

LogicalResult amx::TileMulIOp::verify() {
  amx::TileType aType = getLhsTileType();
  amx::TileType bType = getRhsTileType();
  amx::TileType cType = getTileType();
  if (failed(verifyTileSize(*this, aType)) ||
      failed(verifyTileSize(*this, bType)) ||
      failed(verifyTileSize(*this, cType)) ||
      failed(verifyMultShape(*this, aType, bType, cType, 2)))
    return failure();
  Type ta = aType.getElementType();
  Type tb = bType.getElementType();
  Type tc = cType.getElementType();
  if (!ta.isInteger(8) || !tb.isInteger(8) || !tc.isInteger(32))
    return emitOpError("unsupported type combination");
  return success();
}

Type amx::TileType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return nullptr;

  SmallVector<int64_t, 2> shape;
  if (parser.parseDimensionList(shape, false, true))
    return nullptr;

  Type elementType;
  if (parser.parseType(elementType))
    return nullptr;

  if (parser.parseGreater())
    return nullptr;

  return TileType::get(shape, elementType);
}

void amx::TileType::print(AsmPrinter &os) const {
  os << "<";
  os.printDimensionList(getShape());
  os << 'x';
  os.printType(getElementType());
  os << '>';
}

#define GET_OP_CLASSES
#include "mlir/Dialect/AMX/AMX.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/AMX/AMXTypes.cpp.inc"
