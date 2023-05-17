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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nvgpu;

void nvgpu::NVGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/NVGPU/IR/NVGPUTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/NVGPU/IR/NVGPU.cpp.inc"
      >();
}

bool nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  if (!memorySpace)
    return false;
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(memorySpace))
    return intAttr.getInt() == NVGPUDialect::kSharedMemoryAddressSpace;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuAttr.getValue() == gpu::AddressSpace::Workgroup;
  return false;
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
  auto srcMemref = llvm::cast<MemRefType>(getSrc().getType());
  auto dstMemref = llvm::cast<MemRefType>(getDst().getType());

  if (!isLastMemrefDimUnitStride(srcMemref))
    return emitError("source memref most minor dim must have unit stride");
  if (!isLastMemrefDimUnitStride(dstMemref))
    return emitError("destination memref most minor dim must have unit stride");
  if (!NVGPUDialect::hasSharedMemoryAddressSpace(dstMemref))
    return emitError()
           << "destination memref must have a memory space attribute of "
              "IntegerAttr("
           << NVGPUDialect::kSharedMemoryAddressSpace
           << ") or gpu::AddressSpaceAttr(Workgroup)";
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

/// Performs verification for MmaSyncOp and MmaSparseSyncOp.
static LogicalResult verifyMmaSyncOp(Operation *op,
                                     TypedValue<VectorType> matrixA,
                                     TypedValue<VectorType> matrixB,
                                     TypedValue<VectorType> matrixC,
                                     const std::array<int64_t, 3> &mmaShape,
                                     bool tf32Enabled, bool sparse = false) {

  // The verification for mma.sync covering various shapes and data types is
  // based on the fundamental tensor core shape.

  // "Fundamental" tensor core shapes:
  //  - For F32 (TF32), F16, S8, and S4 data
  //    types the fundamental tensor core operation is of shape 8-by-8-by-128b.
  //  - F64 is an exception and is of shape 8-by-8-by-256b.
  constexpr int kThreads = 32; // 32 threads per warp
  int64_t shapeM = 8;
  int64_t shapeN = 8;
  int64_t shapeK; // set based on data type (128b for all data types except F64)

  // Number of elements A, B, and C per thread per fundamental tensor core tile
  int64_t numElementA;    // set based on data type (32b except F64)
  int64_t numElementB;    // set based on data type (32b except F64)
  int64_t numElementC{2}; // two accumulator elements per fundamental tile

  // nvgpu.mma.sync vector operands (per thread)
  auto aVector = matrixA.getType();
  auto bVector = matrixB.getType();
  auto cVector = matrixC.getType();

  // vector shapes
  ArrayRef<int64_t> aShape = aVector.getShape();
  ArrayRef<int64_t> bShape = bVector.getShape();
  ArrayRef<int64_t> cShape = cVector.getShape();

  // vector element type
  Type aType = aVector.getElementType();

  // Certain data types are not allowed in sparse mode.
  if (sparse && aType.isF64())
    return op->emitError() << "f64 is not supported for sparse mode";

  if (aType.isF64()) {
    // exception to 8-by-8-128b fundamental tensor core tile size
    shapeK = 4;
    numElementA = 1;
    numElementB = 1;
  } else if (aType.isF32() || aType.isBF16() || aType.isF16() ||
             aType.isInteger(8) || aType.isInteger(4)) {
    // 8-by-8-128b fundamental tensor core tile size
    int operandBitwidth = aType.getIntOrFloatBitWidth();
    shapeK = 128 / operandBitwidth; // 128b wide shapeK

    numElementA = 32 / operandBitwidth; // 32b wide operand A
    numElementB = 32 / operandBitwidth; // 32b wide operand B
  } else {
    return op->emitError()
           << "expected input data type (i4,i8,f16,bf16,tf32,f64) "
              "supported by "
           << op->getName();
  }

  //
  // Basic verification
  //

  auto [m, n, k] = mmaShape;

  // verify warp-wide size for vector a
  int64_t sparseFactor = sparse ? 2 : 1;
  if (aShape[0] * aShape[1] * kThreads != m * k / sparseFactor)
    return op->emitOpError()
           << "expected " << m * k << " warp-wide matrix A elements";

  // verify warp-wide size for vector b
  if (bShape[0] * bShape[1] * kThreads != k * n)
    return op->emitOpError()
           << "expected " << k * n << " warp-wide matrix B elements";

  // verify warp-wide size for vector c
  if (cShape[0] * cShape[1] * kThreads != m * n)
    return op->emitOpError()
           << "expected " << m * n << " warp-wide matrix C elements";

  // verify tf32 tensor cores are enabled for only F32 datatype
  if (tf32Enabled && !(aType.isF32()))
    return op->emitOpError()
           << "expected tf32 tensor cores only for F32 operands";

  //
  // Extended verification
  //

  // tiles of fundamental tensor core operations
  int64_t mTile = m / shapeM;
  int64_t nTile = n / shapeN;
  int64_t kTile = k / shapeK;

  // verify shape of aVector
  if ((aShape[0] != mTile * kTile / (sparse ? 2 : 1)) ||
      (aShape[1] != numElementA))
    return op->emitOpError() << "expected matrix A to be shaped ("
                             << mTile * kTile << " x " << numElementA << ")";

  // verify shape of bVector
  if ((bShape[0] != kTile * nTile) || (bShape[1] != numElementB))
    return op->emitOpError() << "expected matrix B to be shaped ("
                             << kTile * nTile << " x " << numElementB << ")";

  // verify shape of cVector
  if ((cShape[0] != mTile * nTile) || (cShape[1] != numElementC))
    return op->emitOpError() << "expected matrix C to be shaped ("
                             << mTile * nTile << " x " << numElementC << ")";

  return success();
}

LogicalResult MmaSyncOp::verify() {
  return verifyMmaSyncOp(this->getOperation(), getMatrixA(), getMatrixB(),
                         getMatrixC(), getMmaShapeAsArray(),
                         getOperation()->hasAttr(getTf32EnabledAttrName()));
}

//===----------------------------------------------------------------------===//
// NVGPU_MmaSparseSyncOp
//===----------------------------------------------------------------------===//
void MmaSparseSyncOp::build(::mlir::OpBuilder &odsBuilder,
                            ::mlir::OperationState &odsState, Value matrixA,
                            Value matrixB, Value matrixC, Value sparseMetadata,
                            ArrayRef<int64_t> mmaShape) {
  build(odsBuilder, odsState, matrixC.getType(), matrixA, matrixB, matrixC,
        sparseMetadata, odsBuilder.getI64ArrayAttr(mmaShape), 0, UnitAttr());
}

LogicalResult MmaSparseSyncOp::verify() {
  unsigned sparsitySelector = getSparsitySelector();
  if (sparsitySelector > 1)
    return emitOpError() << "sparsity selector should be 0 or 1";
  return verifyMmaSyncOp(this->getOperation(), getMatrixA(), getMatrixB(),
                         getMatrixC(), getMmaShapeAsArray(),
                         getOperation()->hasAttr(getTf32EnabledAttrName()),
                         true);
}

//===----------------------------------------------------------------------===//
// NVGPU_LdMatrixOp
//===----------------------------------------------------------------------===//
LogicalResult LdMatrixOp::verify() {

  // ldmatrix reads data from source in shared memory
  auto srcMemref = llvm::cast<MemRefType>(getSrcMemref().getType());

  // ldmatrix writes data to result/destination in vector registers
  auto resVector = llvm::cast<VectorType>(getRes().getType());

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

  //
  // verification
  //

  if (!NVGPUDialect::hasSharedMemoryAddressSpace(srcMemref))
    return emitError()
           << "expected nvgpu.ldmatrix srcMemref must have a memory space "
              "attribute of IntegerAttr("
           << NVGPUDialect::kSharedMemoryAddressSpace
           << ") or gpu::AddressSpaceAttr(Workgroup)";
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

//===----------------------------------------------------------------------===//
// TableGen'd dialect, type, and op definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPU.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPUTypes.cpp.inc"
