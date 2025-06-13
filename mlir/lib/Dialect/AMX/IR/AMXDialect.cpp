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
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "mlir/Dialect/AMX/AMXInterfaces.cpp.inc"

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

/// Maps the 2-dim vector shape to the two 16-bit tile sizes. The first
/// dimension directly translates into the number of rows of the tiles.
/// The second dimensions needs to be scaled by the number of bytes.
static SmallVector<Value> getTileSizes(Location loc, amx::TileType tType,
                                       RewriterBase &rewriter) {
  Type llvmInt16Type = rewriter.getIntegerType(16);
  unsigned width = tType.getElementType().getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_64(width) && width >= 8);
  unsigned bytes = width >> 3;
  auto mattr = rewriter.getI16IntegerAttr(tType.getDimSize(0));
  auto nattr = rewriter.getI16IntegerAttr(tType.getDimSize(1) * bytes);
  return SmallVector<Value>{
      rewriter.create<LLVM::ConstantOp>(loc, llvmInt16Type, mattr),
      rewriter.create<LLVM::ConstantOp>(loc, llvmInt16Type, nattr)};
}

/// Maps the 2-dim memref shape to the 64-bit stride. Note that the buffer
/// shape may "envelop" the actual tile shape, and may be dynamically sized.
static Value getStride(Location loc, MemRefType mType, Value base,
                       RewriterBase &rewriter) {
  assert(mType.getRank() >= 2 && "Invalid shape for AMX strides");
  int64_t preLast = mType.getRank() - 2;
  Type llvmInt64Type = rewriter.getIntegerType(64);
  unsigned width = mType.getElementType().getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_64(width) && width >= 8);
  unsigned bytes = width >> 3;
  auto [strides, offset] = mType.getStridesAndOffset();
  if (strides[preLast] == ShapedType::kDynamic) {
    // Dynamic stride needs code to compute the stride at runtime.
    MemRefDescriptor memrefDescriptor(base);
    auto attr = rewriter.getI64IntegerAttr(bytes);
    Value scale = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, attr);
    return rewriter
        .create<LLVM::MulOp>(loc, llvmInt64Type, scale,
                             memrefDescriptor.stride(rewriter, loc, preLast))
        .getResult();
  }
  // Use direct constant for static stride.
  auto attr = rewriter.getI64IntegerAttr(strides[preLast] * bytes);
  return rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, attr)
      .getResult();
}

LogicalResult amx::TileZeroOp::verify() {
  return verifyTileSize(*this, getTileType());
}

SmallVector<Value>
amx::TileZeroOp::getIntrinsicOperands(ArrayRef<Value> operands,
                                      const LLVMTypeConverter &typeConverter,
                                      RewriterBase &rewriter) {
  return getTileSizes(getLoc(), getTileType(), rewriter);
}

LogicalResult amx::TileLoadOp::verify() {
  MemRefType memrefTy = getMemRefType();
  unsigned rank = memrefTy.getRank();
  if (rank < 2)
    return emitOpError("requires at least 2D memref");
  if (getIndices().size() != rank)
    return emitOpError("requires ") << rank << " indices";
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(memrefTy.getStridesAndOffset(strides, offset)) ||
      strides.back() != 1)
    return emitOpError("requires memref with unit innermost stride");
  return verifyTileSize(*this, getTileType());
}

SmallVector<Value>
amx::TileLoadOp::getIntrinsicOperands(ArrayRef<Value> operands,
                                      const LLVMTypeConverter &typeConverter,
                                      RewriterBase &rewriter) {
  auto loc = getLoc();
  Adaptor adaptor(operands, *this);

  SmallVector<Value> intrinsicOperands;
  intrinsicOperands.append(getTileSizes(loc, getTileType(), rewriter));
  intrinsicOperands.push_back(
      LLVM::getStridedElementPtr(rewriter, loc, typeConverter, getMemRefType(),
                                 adaptor.getBase(), adaptor.getIndices()));
  intrinsicOperands.push_back(
      getStride(loc, getMemRefType(), adaptor.getBase(), rewriter));

  return intrinsicOperands;
}

LogicalResult amx::TileStoreOp::verify() {
  MemRefType memrefTy = getMemRefType();
  unsigned rank = memrefTy.getRank();
  if (rank < 2)
    return emitOpError("requires at least 2D memref");
  if (getIndices().size() != rank)
    return emitOpError("requires ") << rank << " indices";
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(memrefTy.getStridesAndOffset(strides, offset)) ||
      strides.back() != 1)
    return emitOpError("requires memref with unit innermost stride");
  return verifyTileSize(*this, getTileType());
}

SmallVector<Value>
amx::TileStoreOp::getIntrinsicOperands(ArrayRef<Value> operands,
                                       const LLVMTypeConverter &typeConverter,
                                       RewriterBase &rewriter) {
  auto loc = getLoc();
  Adaptor adaptor(operands, *this);

  SmallVector<Value> intrinsicOperands;
  intrinsicOperands.append(getTileSizes(loc, getTileType(), rewriter));
  intrinsicOperands.push_back(
      LLVM::getStridedElementPtr(rewriter, loc, typeConverter, getMemRefType(),
                                 adaptor.getBase(), adaptor.getIndices()));
  intrinsicOperands.push_back(
      getStride(loc, getMemRefType(), adaptor.getBase(), rewriter));
  intrinsicOperands.push_back(adaptor.getVal());

  return intrinsicOperands;
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

SmallVector<Value>
amx::TileMulFOp::getIntrinsicOperands(ArrayRef<Value> operands,
                                      const LLVMTypeConverter &typeConverter,
                                      RewriterBase &rewriter) {
  auto loc = getLoc();
  Adaptor adaptor(operands, *this);

  amx::TileType aType = getLhsTileType();
  amx::TileType bType = getRhsTileType();
  SmallVector<Value> tsza = getTileSizes(loc, aType, rewriter);
  SmallVector<Value> tszb = getTileSizes(loc, bType, rewriter);

  SmallVector<Value> intrinsicOperands = {tsza[0],          tszb[1],
                                          tsza[1],          adaptor.getAcc(),
                                          adaptor.getLhs(), adaptor.getRhs()};

  return intrinsicOperands;
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

SmallVector<Value>
amx::TileMulIOp::getIntrinsicOperands(ArrayRef<Value> operands,
                                      const LLVMTypeConverter &typeConverter,
                                      RewriterBase &rewriter) {
  auto loc = getLoc();
  Adaptor adaptor(operands, *this);

  amx::TileType aType = getLhsTileType();
  amx::TileType bType = getRhsTileType();
  SmallVector<Value> tsza = getTileSizes(loc, aType, rewriter);
  SmallVector<Value> tszb = getTileSizes(loc, bType, rewriter);

  SmallVector<Value> intrinsicOperands = {tsza[0],          tszb[1],
                                          tsza[1],          adaptor.getAcc(),
                                          adaptor.getLhs(), adaptor.getRhs()};

  return intrinsicOperands;
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
