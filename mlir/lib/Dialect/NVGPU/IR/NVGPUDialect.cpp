//===- NVGPUDialect.cpp - MLIR NVGPU ops implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the NVGPU dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nvgpu;

#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.cpp.inc"

void nvgpu::NVGPUDialect::initialize() {
  addTypes<DeviceAsyncTokenType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/NVGPU/IR/NVGPU.cpp.inc"
      >();
}

Type NVGPUDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();
  // Handle 'device async token' types.
  if (keyword == "device.async.token")
    return DeviceAsyncTokenType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown nvgpu type: " + keyword);
  return Type();
}

void NVGPUDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<DeviceAsyncTokenType>([&](Type) { os << "device.async.token"; })
      .Default([](Type) { llvm_unreachable("unexpected 'nvgpu' type kind"); });
}
//===----------------------------------------------------------------------===//
// NVGPU_DeviceAsyncCopyOp
//===----------------------------------------------------------------------===//

/// Return true if the last dimension of the MemRefType has unit stride. Also
/// return true for memrefs with no strides.
static bool isLastMemrefDimUnitStride(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(type, strides, offset))) {
    return false;
  }
  return strides.back() == 1;
}

LogicalResult DeviceAsyncCopyOp::verify() {
  auto srcMemref = getSrc().getType().cast<MemRefType>();
  auto dstMemref = getDst().getType().cast<MemRefType>();
  unsigned workgroupAddressSpace = gpu::GPUDialect::getWorkgroupAddressSpace();
  if (!isLastMemrefDimUnitStride(srcMemref))
    return emitError("source memref most minor dim must have unit stride");
  if (!isLastMemrefDimUnitStride(dstMemref))
    return emitError("destination memref most minor dim must have unit stride");
  if (dstMemref.getMemorySpaceAsInt() != workgroupAddressSpace)
    return emitError("destination memref must have memory space ")
           << workgroupAddressSpace;
  if (dstMemref.getElementType() != srcMemref.getElementType())
    return emitError("source and destination must have the same element type");
  if (size_t(srcMemref.getRank()) != getSrcIndices().size())
    return emitOpError() << "expected " << srcMemref.getRank()
                         << " source indices, got " << getSrcIndices().size();
  if (size_t(dstMemref.getRank()) != getDstIndices().size())
    return emitOpError() << "expected " << dstMemref.getRank()
                         << " destination indices, got "
                         << getDstIndices().size();
  return success();
}

//===----------------------------------------------------------------------===//
// NVGPU_MmaSyncOp
//===----------------------------------------------------------------------===//
void MmaSyncOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, Value matrixA,
                      Value matrixB, Value matrixC, ArrayAttr mmaShape) {
  build(odsBuilder, odsState, matrixC.getType(), matrixA, matrixB, matrixC,
        mmaShape, UnitAttr());
}

LogicalResult MmaSyncOp::verify() {

  // Fundamental tensor core mma.sync op
  // For F32 (TF32), F16, S8, and S4 data types fundamental tensor core
  // operation is of shape: 8-by-8-by-128b. F64 is an exception. The
  // verification for mma.sync covering various shapes and data types is based
  // on the fundamental tensor core operionation.
  constexpr int kThreads = 32; // 32 threads per warp
  int64_t shapeM = 8;
  int64_t shapeN = 8;
  int64_t shapeK; // set based on data type (128b for all data types except F64)

  // Number of elements A, B, and C per thread per fundamental tensor core tile
  int64_t numElementA;    // set based on data type (32b except F64)
  int64_t numElementB;    // set based on data type (32b except F64)
  int64_t numElementC{2}; // two accumulator elements per fundamental tile

  // nvgpu.mma.sync vector operands (per thread)
  auto aVector = getMatrixA().getType().cast<VectorType>();
  auto bVector = getMatrixB().getType().cast<VectorType>();
  auto cVector = getMatrixC().getType().cast<VectorType>();

  // vector shapes
  ArrayRef<int64_t> aShape = aVector.getShape();
  ArrayRef<int64_t> bShape = bVector.getShape();
  ArrayRef<int64_t> cShape = cVector.getShape();

  // vector element type
  Type aType = aVector.getElementType();

  // tensor float32 (TF32) enabled
  bool tf32Enabled = getOperation()->hasAttr(getTf32EnabledAttrName());

  // nvgpu.mma.sync shape (per 32 threads or per warp)
  int64_t m = getMmaShape()[0].cast<IntegerAttr>().getInt();
  int64_t n = getMmaShape()[1].cast<IntegerAttr>().getInt();
  int64_t k = getMmaShape()[2].cast<IntegerAttr>().getInt();

  if (aType.isF64()) {
    // exception to 8-by-8-128b fundamental tensor core tile size
    shapeK = 4;
    numElementA = 1;
    numElementB = 1;
  } else if (aType.isF32() || aType.isBF16() || aType.isF16() ||
             aType.isInteger(8) || aType.isInteger(4)) {
    // 8-by-8-128b fundamental tensor core tile size
    int operandBitwidth = aType.getIntOrFloatBitWidth();
    shapeK = 128 / operandBitwidth;     // 128b wide shapeK
    numElementA = 32 / operandBitwidth; // 32b wide operand A
    numElementB = 32 / operandBitwidth; // 32b wide operand B
  } else {
    return emitError() << "expected input data type (i4,i8,f16,bf16,tf32,f64) "
                          "supported by nvgpu.mma.sync";
  }

  //
  // Basic verification
  //

  // verify warp-wide size for vector a
  if (aShape[0] * aShape[1] * kThreads != m * k)
    return emitOpError() << "expected " << m * k
                         << " warp-wide matrix A elements";

  // verify warp-wide size for vector b
  if (bShape[0] * bShape[1] * kThreads != k * n)
    return emitOpError() << "expected " << k * n
                         << " warp-wide matrix B elements";

  // verify warp-wide size for vector c
  if (cShape[0] * cShape[1] * kThreads != m * n)
    return emitOpError() << "expected " << m * n
                         << " warp-wide matrix C elements";

  // verify tf32 tensor cores are enabled for only F32 datatype
  if (tf32Enabled && !(aType.isF32()))
    return emitOpError() << "expected tf32 tensor cores only for F32 operands";

  //
  // Extended verification
  //

  // tiles of fundamental tensor core operations
  int64_t mTile = m / shapeM;
  int64_t nTile = n / shapeN;
  int64_t kTile = k / shapeK;

  // verify shape of aVector
  if (!((aShape[0] == mTile * kTile) && (aShape[1] == numElementA)))
    return emitOpError() << "expected matrix A to be shaped (" << mTile * kTile
                         << " x " << numElementA << ")";

  // verify shape of bVector
  if (!((bShape[0] == kTile * nTile) && (bShape[1] == numElementB)))
    return emitOpError() << "expected matrix B to be shaped (" << kTile * nTile
                         << " x " << numElementB << ")";

  // verify shape of cVector
  if (!((cShape[0] == mTile * nTile) && (cShape[1] == numElementC)))
    return emitOpError() << "expected matrix C to be shaped (" << mTile * nTile
                         << " x " << numElementC << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// NVGPU_LdMatrixOp
//===----------------------------------------------------------------------===//
LogicalResult LdMatrixOp::verify() {

  // ldmatrix reads data from source in shared memory
  auto srcMemref = getSrcMemref().getType().cast<MemRefType>();

  // ldmatrix writes data to result/destination in vector registers
  auto resVector = getRes().getType().cast<VectorType>();

  // vector register shape, element type, and bitwidth
  ArrayRef<int64_t> resShape = resVector.getShape();
  Type resType = resVector.getElementType();
  int64_t elementBitWidth = resType.getIntOrFloatBitWidth();

  // ldmatrix loads 32 bits into vector registers per 8-by-8 tile per thread
  int64_t numElementsPer32b = 32 / elementBitWidth;

  // number of 8-by-8 tiles
  int64_t numTiles = getNumTiles();

  // transpose elements in vector registers at 16b granularity when true
  bool isTranspose = getTranspose();

  // address space id for shared memory
  unsigned smemAddressSpace = gpu::GPUDialect::getWorkgroupAddressSpace();

  //
  // verification
  //

  if (!(srcMemref.getMemorySpaceAsInt() == smemAddressSpace))
    return emitError()
           << "expected nvgpu.ldmatrix srcMemref must have memory space "
           << smemAddressSpace;
  if (elementBitWidth > 32)
    return emitError() << "nvgpu.ldmatrix works for 32b or lower";
  if (isTranspose && !(elementBitWidth == 16))
    return emitError()
           << "nvgpu.ldmatrix transpose works only at 16b granularity";
  if (!(resShape[1] == numElementsPer32b))
    return emitError() << "expected vector register shape[1] = "
                       << numElementsPer32b;
  if (!(resShape[0] == numTiles))
    return emitError()
           << "expected vector register shape[0] and numTiles to match";

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPU.cpp.inc"
