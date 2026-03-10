//===-- XeVMDialect.cpp - XeVM dialect registration -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::xevm;

#include "mlir/Dialect/LLVMIR/XeVMOpsDialect.cpp.inc"
#include "mlir/Dialect/LLVMIR/XeVMOpsEnums.cpp.inc"

namespace {
static constexpr uint32_t subgroupSize = 16;

template <typename Op>
LogicalResult verifyMatrixInput(Op op) {
  static_assert(llvm::is_one_of<Op, BlockLoad2dOp, BlockStore2dOp,
                                BlockPrefetch2dOp>::value,
                "Unexpected template parameter");

  std::optional<int64_t> width = getConstantIntValue(op.getBaseWidth());
  std::optional<int64_t> pitch = getConstantIntValue(op.getBasePitch());
  if (pitch && width && *pitch < *width)
    return op->emitOpError(
        "4th operand (base pitch) should be >= 2nd operand (base width)");

  uint32_t elemSize = op.getElemSizeInBits();
  if (elemSize < 8 || !llvm::isPowerOf2_32(elemSize) || elemSize > 32)
    return op->emitOpError("expecting 'elem_size_in_bits' to be 8, 16, or 32");

  uint32_t tileHeight = op.getTileHeight();
  if (tileHeight > 32 || !llvm::isPowerOf2_32(tileHeight))
    return op->emitOpError("expecting tile_height to be 1, 2, 4, 8, 16, or 32");

  uint32_t vBlocks = op.getVBlocks();
  if (vBlocks > 8 || !llvm::isPowerOf2_32(vBlocks))
    return op->emitOpError("expecting v_blocks to be 1, 2, 4, or 8");

  return success();
}

LogicalResult verify2DBlockLoadRestriction(BlockLoad2dOp op) {
  VectorType resTy = op.getRes().getType();
  if (!resTy.getElementType().isIntOrFloat())
    return op.emitOpError()
           << "expecting result element type to be int or float";
  unsigned resElemTySize = resTy.getElementType().getIntOrFloatBitWidth();
  unsigned resSize = resTy.getNumElements() * resElemTySize;
  unsigned expectedSize = op.getElemSizeInBits() * op.getTileHeight() *
                          op.getTileWidth() * op.getVBlocks() / subgroupSize;
  if (resSize != expectedSize)
    return op.emitOpError() << "result size of " << resSize
                            << " bits does not match the expected size of "
                            << expectedSize << " bits";

  if (op.getTranspose() && op.getPackRegister())
    return op.emitOpError("transpose and pack_register are mutually exclusive");

  if (!op.getTranspose() && !op.getPackRegister()) {
    uint32_t tileHeight = op.getTileHeight();
    if (tileHeight < 1 || tileHeight > 32)
      return op.emitOpError("expecting tile_height to be between 1 and 32");

    uint32_t tileWidth = op.getTileWidth();
    uint32_t vBlocks = op.getVBlocks();
    switch (op.getElemSizeInBits()) {
    case 8:
      if (tileWidth < 4 || tileWidth > 64)
        return op.emitOpError("expecting tile_width to be between 4 and 64");
      if (vBlocks != 1 && vBlocks != 2 && vBlocks != 4)
        return op.emitOpError("expecting v_blocks to be 1, 2, or 4");
      if (tileWidth * vBlocks > 64)
        return op.emitOpError(
            "tile_width * v_blocks should be less than or equal "
            "to 64 for 8 bit elements");
      break;
    case 16:
      if (tileWidth < 2 || tileWidth > 32)
        return op.emitOpError("expecting tile_width to be between 2 and 32");
      if (vBlocks != 1 && vBlocks != 2 && vBlocks != 4)
        return op.emitOpError("expecting v_blocks to be 1, 2, or 4");
      if (tileWidth * vBlocks > 32)
        return op.emitOpError(
            "tile_width * v_blocks should be less than or equal "
            "to 32 for 16 bit elements");
      break;
    case 32:
      if (tileWidth < 1 || tileWidth > 16)
        return op.emitOpError("expecting tile_width to be between 1 and 16");
      if (vBlocks != 1 && vBlocks != 2)
        return op.emitOpError("expecting v_blocks to be 1 or 2");
      if (tileWidth * vBlocks > 16)
        return op.emitOpError(
            "tile_width * v_blocks should be less than or equal "
            "to 16 for 32 bit elements");
      break;
    case 64:
      if (tileWidth < 1 || tileWidth > 8)
        return op.emitOpError("expecting tile_width to be between 1 and 8");
      if (vBlocks != 1)
        return op.emitOpError("expecting v_blocks to be 1");
      break;
    default:
      return op.emitOpError(
          "expecting elem_size_in_bits to be 8, 16, 32, or 64");
    }

    return success();
  }

  if (op.getTranspose()) {
    assert(!op.getPackRegister() && "Expecting pack_register should be false");

    uint32_t vBlocks = op.getVBlocks();
    if (vBlocks != 1)
      return op.emitOpError("expecting v_blocks to be 1");

    uint32_t tileHeight = op.getTileHeight();
    uint32_t tileWidth = op.getTileWidth();
    switch (op.getElemSizeInBits()) {
    case 32:
      if (tileHeight < 1 || tileHeight > 32)
        return op.emitOpError("expecting tile_height to be between 1 and 32");
      if (tileWidth < 1 || tileWidth > 8)
        return op.emitOpError("expecting tile_width to be between 1 and 8");
      break;
    case 64:
      if (tileHeight != 8)
        return op.emitOpError(
            "expecting tile_height to be 8 for 64 bit elements");
      if (tileWidth != 1 && tileWidth != 2 && tileWidth != 4)
        return op.emitOpError("expecting tile_width to be 1, 2, or 4");
      break;
    default:
      return op.emitOpError("transpose is only supported for 32 and 64 bit "
                            "elements");
    }

    return success();
  }

  assert(op.getPackRegister() && !op.getTranspose() &&
         "Expecting pack_register should be true and transpose should be "
         "false");

  uint32_t vBlocks = op.getVBlocks();
  if (vBlocks != 1 && vBlocks != 2 && vBlocks != 4)
    return op.emitOpError("expecting v_blocks to be 1, 2, or 4");

  uint32_t tileHeight = op.getTileHeight();
  uint32_t tileWidth = op.getTileWidth();
  switch (op.getElemSizeInBits()) {
  case 8:
    if (tileHeight < 4 || tileHeight > 32)
      return op.emitOpError("expecting tile_height to be between 4 and 32");
    if (tileWidth < 4 || tileWidth > 16)
      return op.emitOpError("expecting tile_width to be between 4 and 16");
    break;
  case 16:
    if (tileHeight < 2 || tileHeight > 32)
      return op.emitOpError("expecting tile_height to be between 2 and 32");
    if (tileWidth < 2 || tileWidth > 16)
      return op.emitOpError("expecting tile_width to be between 2 and 16");
    if (tileWidth * vBlocks > 32)
      return op.emitOpError(
          "tile_width * v_blocks should be less than or equal "
          "to 32 for 16 bit elements");
    break;
  default:
    return op.emitOpError("pack_register is only supported for 8 and 16 bit "
                          "elements");
  }

  return success();
}

static LogicalResult verify2DBlockStoreRestriction(BlockStore2dOp op) {
  uint32_t tileHeight = op.getTileHeight();
  if (tileHeight < 1 || tileHeight > 8)
    return op.emitOpError("expecting tile_height to be between 1 and 8");

  uint32_t tileWidth = op.getTileWidth();
  switch (op.getElemSizeInBits()) {
  case 8:
    if (tileWidth < 4 || tileWidth > 64)
      return op.emitOpError("expecting tile_width to be between 4 and 64");
    break;
  case 16:
    if (tileWidth < 2 || tileWidth > 32)
      return op.emitOpError("expecting tile_width to be between 2 and 32");
    break;
  case 32:
    if (tileWidth < 1 || tileWidth > 16)
      return op.emitOpError("expecting tile_width to be between 1 and 16");
    break;
  case 64:
    if (tileWidth < 1 || tileWidth > 8)
      return op.emitOpError("expecting tile_width to be between 1 and 8");
    break;
  default:
    return op.emitOpError("expecting elem_size_in_bits to be 8, 16, 32, or 64");
  }

  uint32_t vBlocks = op.getVBlocks();
  if (vBlocks != 1)
    return op.emitOpError("expecting v_blocks to be 1");
  return success();
}

} // namespace

LogicalResult BlockLoad2dOp::verify() {
  if (verify2DBlockLoadRestriction(*this).failed())
    return failure();

  if (verifyMatrixInput(*this).failed())
    return failure();

  VectorType resTy = getRes().getType();
  if (!resTy.getElementType().isIntOrFloat())
    return emitOpError() << "expecting result element type to be int of float";
  unsigned resElemTySize = resTy.getElementType().getIntOrFloatBitWidth();
  if (getElemSizeInBits() == 32 || getPackRegister()) {
    if (resElemTySize != 32)
      return emitOpError() << "expecting result element type to be 32 bits";
  }

  uint32_t tileWidth = getTileWidth();
  if (getPackRegister()) {
    if (tileWidth != 16)
      return emitOpError(
          "tile_width when pack_register is true should be equal "
          "to subgroup size (16 elements)");
    return success();
  }

  return success();
}

LogicalResult BlockStore2dOp::verify() {
  if (verify2DBlockStoreRestriction(*this).failed())
    return failure();

  if (verifyMatrixInput(*this).failed())
    return failure();

  uint32_t tileWidth = getTileWidth();
  switch (getElemSizeInBits()) {
  case 8:
    if (tileWidth != 16 && tileWidth != 32)
      return emitOpError("tile_width for 8 bit elements should be equal to "
                         "16 or 32");
    break;
  case 16:
    if (tileWidth != 16)
      return emitOpError("tile_width for 16 bit elements should be equal "
                         "to 16");
    break;
  case 32:
    if (tileWidth != 16)
      return emitOpError("tile_width for 32 bit elements should be equal "
                         "to 16");
    break;
  default:
    llvm_unreachable("unexpected element size");
  }

  return success();
}

LogicalResult BlockPrefetch2dOp::verify() {
  if (verifyMatrixInput(*this).failed())
    return failure();

  uint32_t tileWidth = getTileWidth();
  switch (getElemSizeInBits()) {
  case 8:
    if (tileWidth != 16 && tileWidth != 32)
      return emitOpError("tile_width for 8 bit elements should be equal to "
                         "16 or 32");
    break;
  case 16:
    if (tileWidth != 16)
      return emitOpError("tile_width for 16 bit elements should be equal "
                         "to 16");
    break;
  case 32:
    if (tileWidth != 8 && tileWidth != 16)
      return emitOpError(
          "tile_width for 32 bit elements should be equal to 8 or 16");
    break;
  default:
    llvm_unreachable("unexpected element size");
  }

  return success();
}

template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                               OpType, BlockLoadOp, BlockStoreOp>::value>>
LogicalResult verify1DBlockArg(OpType op) {
  Type srcOrDstTy;
  if constexpr (std::is_same_v<OpType, BlockLoadOp>)
    srcOrDstTy = op.getResult().getType();
  else
    srcOrDstTy = op.getVal().getType();
  VectorType vTy = dyn_cast<VectorType>(srcOrDstTy);
  // scalar case is always valid
  if (!vTy)
    return success();
  int elemTySize = vTy.getElementType().getIntOrFloatBitWidth() / 8;
  if (elemTySize == 1) {
    llvm::SmallSet<int, 4> validSizes{2, 4, 8, 16};
    if (validSizes.contains(vTy.getNumElements()))
      return success();
    else
      return op.emitOpError(
          "vector size must be 2, 4, 8 or 16 for 8-bit element type");
  } else {
    llvm::SmallSet<int, 3> validSizes{2, 4, 8};
    if (validSizes.contains(vTy.getNumElements()))
      return success();
    else
      return op.emitOpError(
          "vector size must be 2, 4 or 8 for element type > 8 bits");
  }
}

LogicalResult BlockLoadOp::verify() { return verify1DBlockArg(*this); }

LogicalResult BlockStoreOp::verify() { return verify1DBlockArg(*this); }

LogicalResult MMAOp::verify() {
  if (getC()) {
    if (getResult().getType() != getC().getType())
      return emitOpError("type of C operand must match result type");
  }
  return success();
}

LogicalResult
XeVMTargetAttr::verify(function_ref<InFlightDiagnostic()> emitError, int O,
                       StringRef triple, StringRef chip, DictionaryAttr flags,
                       ArrayAttr linkFiles) {
  if (O < 0 || O > 3) {
    return emitError()
           << "The optimization level must be a number between 0 and 3.";
  }
  if (triple.empty()) {
    return emitError() << "The target triple cannot be empty.";
  }
  if (chip.empty()) {
    return emitError() << "The target chip cannot be empty.";
  }
  if (linkFiles) {
    for (Attribute fileAttr : linkFiles) {
      if (auto fileStrAttr = llvm::dyn_cast<StringAttr>(fileAttr)) {
        StringRef filePath = fileStrAttr.getValue();
        if (filePath.empty()) {
          return emitError() << "File paths in linkFiles cannot be empty.";
        }
        if (!llvm::sys::fs::exists(filePath)) {
          return emitError() << "File '" << filePath << "' does not exist.";
        }
      }
    }
  }
  return success();
}

void XeVMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/XeVMOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/XeVMOpsAttributes.cpp.inc"
      >();
  declarePromisedInterface<mlir::gpu::TargetAttrInterface,
                           mlir::xevm::XeVMTargetAttr>();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/XeVMOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/XeVMOpsAttributes.cpp.inc"
