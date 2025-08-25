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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nvgpu;

#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.cpp.inc"

void nvgpu::NVGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/NVGPU/IR/NVGPUTypeDefs.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/NVGPU/IR/NVGPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/NVGPU/IR/NVGPUOps.cpp.inc"
      >();
}

bool nvgpu::NVGPUDialect::isSharedMemoryAddressSpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(memorySpace))
    return intAttr.getInt() == NVGPUDialect::kSharedMemoryAddressSpace;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuAttr.getValue() == gpu::AddressSpace::Workgroup;
  return false;
}

bool nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  return isSharedMemoryAddressSpace(memorySpace);
}

//===----------------------------------------------------------------------===//
// NVGPU_DeviceAsyncCopyOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceAsyncCopyOp::verify() {
  auto srcMemref = llvm::cast<MemRefType>(getSrc().getType());
  auto dstMemref = llvm::cast<MemRefType>(getDst().getType());

  if (!srcMemref.isLastDimUnitStride())
    return emitError("source memref most minor dim must have unit stride");
  if (!dstMemref.isLastDimUnitStride())
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
  int64_t dstElements = getDstElements().getZExtValue();
  int64_t sizeInBytes = (dstMemref.getElementTypeBitWidth() * dstElements) / 8;
  if (sizeInBytes != 4 && sizeInBytes != 8 && sizeInBytes != 16) {
    unsigned dstWidth = dstMemref.getElementTypeBitWidth();
    InFlightDiagnostic diag = emitError();
    diag << "Requested copy elements is " << dstElements << " with width "
         << dstMemref.getElementTypeBitWidth()
         << ". But copy elements could be one of ";
    if ((32 / dstWidth) > 0)
      diag << (32 / dstWidth) << ", ";
    if ((64 / dstWidth) > 0)
      diag << (64 / dstWidth) << ", ";
    if ((128 / dstWidth) > 0)
      diag << (128 / dstWidth) << ".";
    return diag;
  }
  if (getBypassL1().has_value()) {
    int64_t req = 16 * 8 / dstMemref.getElementTypeBitWidth();
    if (getBypassL1().value() && sizeInBytes != 16) {
      return emitOpError() << "bypassL1 does not satify alignment for "
                           << dstMemref << " with destination element "
                           << dstElements
                           << ". Unset bypassL1, or set "
                              "destination element to "
                           << req;
    }
  }
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

void MmaSyncOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, Value matrixA,
                      Value matrixB, Value matrixC, ArrayRef<int64_t> mmaShape,
                      bool tf32Enabled) {
  build(odsBuilder, odsState, matrixC.getType(), matrixA, matrixB, matrixC,
        odsBuilder.getI64ArrayAttr(mmaShape),
        tf32Enabled ? odsBuilder.getUnitAttr() : UnitAttr());
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

  if (aShape.size() != 2) {
    return op->emitError() << "matrixA must be 2 dimensional vector";
  }

  if (bShape.size() != 2) {
    return op->emitError() << "matrixB must be 2 dimensional vector";
  }

  if (cShape.size() != 2) {
    return op->emitError() << "matrixC must be 2 dimensional vector";
  }

  auto [m, n, k] = mmaShape;

  // verify warp-wide size for vector a
  int64_t sparseFactor = sparse ? 2 : 1;
  if (aShape[0] * aShape[1] * kWarpSize != m * k / sparseFactor)
    return op->emitOpError()
           << "expected " << m * k << " warp-wide matrix A elements";

  // verify warp-wide size for vector b
  if (bShape[0] * bShape[1] * kWarpSize != k * n)
    return op->emitOpError()
           << "expected " << k * n << " warp-wide matrix B elements";

  // verify warp-wide size for vector c
  if (cShape[0] * cShape[1] * kWarpSize != m * n)
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
  if (resShape.size() != 2) {
    return emitError() << "results must be 2 dimensional vector";
  }
  if (!(resShape[1] == numElementsPer32b))
    return emitError() << "expected vector register shape[1] = "
                       << numElementsPer32b;
  if (!(resShape[0] == numTiles))
    return emitError()
           << "expected vector register shape[0] and numTiles to match";

  return success();
}

//===----------------------------------------------------------------------===//
// NVGPU_TmaAsyncLoadOp
//===----------------------------------------------------------------------===//

unsigned getSwizzleBytes(TensorMapSwizzleKind kind) {
  switch (kind) {
  case TensorMapSwizzleKind::SWIZZLE_32B:
    return 32;
  case TensorMapSwizzleKind::SWIZZLE_64B:
    return 64;
  case TensorMapSwizzleKind::SWIZZLE_128B:
    return 128;
  default:
    return 0;
  }
}

std::optional<InFlightDiagnostic> verifyTmaDescriptorWithMemref(
    Operation *op, nvgpu::TensorMapDescriptorType descType,
    std::optional<MemRefType> memrefType = std::nullopt) {
  MemRefType descMemref = descType.getTensor();
  // Limitation
  if (descType.getInterleave() != TensorMapInterleaveKind::INTERLEAVE_NONE)
    return op->emitError() << "Interleave options are not supported yet.";

  // Address space check for shared memory check
  if (!NVGPUDialect::hasSharedMemoryAddressSpace(descMemref)) {
    return op->emitError() << "the tensor map descriptor has incorrect address "
                              "space, it must be shared memory address space.";
  }
  // Support only static shape for the time being
  if (!descMemref.hasStaticShape())
    return op->emitError() << "the tensor map descriptor must be static shaped";

  for (auto dim : descMemref.getShape()) {
    if (dim <= 0 || dim > kMaxTMADimension) {
      return op->emitError() << "the tensor map descriptor must have "
                                "dimensions between 1 and "
                             << kMaxTMADimension << " but it is " << dim;
    }
  }
  if (descMemref.getRank() > 1 &&
      descType.getSwizzle() != TensorMapSwizzleKind::SWIZZLE_NONE) {
    unsigned lastDimensionByte =
        descMemref.getElementTypeBitWidth() * descMemref.getShape().back() / 8;
    unsigned expectByte = getSwizzleBytes(descType.getSwizzle());
    if (lastDimensionByte != expectByte)
      return op->emitError() << "the tensormap descriptor must have last "
                                "dimension of "
                             << expectByte << " bytes but it is "
                             << lastDimensionByte << " bytes";
  }

  // No verification if memref type is not provided
  if (!memrefType.has_value())
    return std::nullopt;

  MemRefType dstMemref = memrefType.value();

  // Check element type
  if (descMemref.getElementType() != dstMemref.getElementType()) {
    return op->emitError() << "the element type of tensor map descriptor and "
                              "memref must be same";
  }

  if (!NVGPUDialect::hasSharedMemoryAddressSpace(dstMemref)) {
    return op->emitError() << "the destination memref has incorrect address "
                              "space, it must be shared memory address space.";
  }
  if (!dstMemref.hasStaticShape())
    return op->emitError() << "the destination memref must be static shaped";

  if (dstMemref.getRank() != descMemref.getRank()) {
    return op->emitError() << "the shape of tensor map descriptor and "
                              "memref must have same rank";
  }
  if (!descMemref.getShape().equals(dstMemref.getShape())) {
    return op->emitError() << "memref and tensor map shapes mismatch "
                           << descMemref << " != " << dstMemref;
  }

  int lastDimBytes =
      descMemref.getShape().back() * descMemref.getElementTypeBitWidth() / 8;
  if (lastDimBytes % 16 != 0) {
    return op->emitError() << "the bytes in the last dimension of the tensor "
                              "map must be a multiple of 16";
  }
  return std::nullopt;
}

LogicalResult TmaAsyncLoadOp::verify() {
  std::optional<InFlightDiagnostic> error = verifyTmaDescriptorWithMemref(
      *this, getTensorMapDescriptor().getType(), getDst().getType());
  if (error.has_value())
    return error.value();

  if (getCoordinates().size() > kMaxTMATensorDimension) {
    return emitError() << "Maximum " << kMaxTMATensorDimension
                       << " coordinates are supported.";
  }
  if (getCoordinates().size() !=
      size_t(getTensorMapDescriptor().getType().getTensor().getRank())) {
    return emitError() << "number of coordinates do not match with the rank of "
                          "tensor descriptor map.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// NVGPU_TmaAsyncStoreOp
//===----------------------------------------------------------------------===//

LogicalResult TmaAsyncStoreOp::verify() {
  std::optional<InFlightDiagnostic> error = verifyTmaDescriptorWithMemref(
      *this, getTensorMapDescriptor().getType(), getSrc().getType());
  if (error.has_value())
    return error.value();

  if (getCoordinates().size() > kMaxTMATensorDimension) {
    return emitError() << "Maximum " << kMaxTMATensorDimension
                       << " coordinates are supported.";
  }
  if (getCoordinates().size() !=
      size_t(getTensorMapDescriptor().getType().getTensor().getRank())) {
    return emitError() << "number of coordinates do not match with the rank of "
                          "tensor descriptor map.";
  }

  return success();
}

LogicalResult TmaCreateDescriptorOp::verify() {
  if (getBoxDimensions().size() > kMaxTMATensorDimension) {
    return emitError() << "Maximum " << kMaxTMATensorDimension
                       << " coordinates are supported.";
  }

  std::optional<InFlightDiagnostic> error =
      verifyTmaDescriptorWithMemref(*this, getTensorMap().getType());
  if (error.has_value())
    return error.value();

  return success();
}

//===----------------------------------------------------------------------===//
// NVGPU_WarpgroupGenerateDescriptorOp
//===----------------------------------------------------------------------===//

LogicalResult WarpgroupGenerateDescriptorOp::verify() {
  std::optional<InFlightDiagnostic> error =
      verifyTmaDescriptorWithMemref(*this, getTensorMap().getType());
  if (error.has_value())
    return error.value();

  if (getTensorMap().getType().getSwizzle() !=
      TensorMapSwizzleKind::SWIZZLE_128B) {
    return emitError() << "supports only "
                       << stringifyTensorMapSwizzleKind(
                              TensorMapSwizzleKind::SWIZZLE_128B)
                       << " is supported for the time being";
  }

  if (getTensorMap().getType().getInterleave() !=
      TensorMapInterleaveKind::INTERLEAVE_NONE) {
    return emitError() << "supports only "
                       << stringifyTensorMapInterleaveKind(
                              TensorMapInterleaveKind::INTERLEAVE_NONE)
                       << " is supported for the time being";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// WarpgroupMmaOp
//===----------------------------------------------------------------------===//

LogicalResult isAllowedWGMMADataType(Type typeD, Type typeA, Type typeB) {
  // F32 += F16 + F16
  // F16 += F16 + F16
  if (typeA.isF16() && typeB.isF16() && (typeD.isF32() || typeD.isF16()))
    return success();
  // F32 += TF32 + TF32
  if (typeA.isTF32() && typeD.isF32() && typeB.isTF32())
    return success();
  // s32 += i8 + i8
  if (typeA.isInteger(16) && typeB.isInteger(16) && typeD.isInteger(32))
    return success();
  // s32 += i1 + i1
  if (typeA.isInteger(1) && typeB.isInteger(1) && typeD.isInteger(32))
    return success();
  // F32 += BF16 + BF16
  // F16 += BF16 + BF16
  if (typeA.isBF16() && typeB.isBF16() && (typeD.isF32() || typeD.isF16()))
    return success();
  // F16 += f8 + f8
  // F32 += f8 + f8
  if (isa<Float8E5M2Type, Float8E4M3FNType>(typeA) &&
      isa<Float8E5M2Type, Float8E4M3FNType>(typeB) &&
      (typeD.isF32() || typeD.isF16()))
    return success();

  return failure();
}

LogicalResult isAllowedSizeM(int sizeM) {
  if (sizeM % kWgmmaSizeM)
    return failure();
  return success();
}

LogicalResult isAllowedSizeN(int sizeN, Type typeA) {
  SmallVector<int> allowedN = {8,   16,  24,  32,  40,  48,  56,  64,
                               72,  80,  88,  96,  104, 112, 120, 128,
                               136, 144, 152, 160, 168, 176, 184, 192,
                               200, 208, 216, 224, 232, 240, 248, 256};
  SmallVector<int> allowedNshort = {8,   16,  24,  32,  48,  64,
                                    80,  96,  112, 128, 144, 160,
                                    176, 192, 208, 224, 240, 256};
  if (typeA.isBF16() || typeA.isF16() || typeA.isF32() || typeA.isTF32() ||
      isa<Float8E5M2Type, Float8E4M3FNType>(typeA))
    if (llvm::is_contained(allowedN, sizeN))
      return success();

  if (typeA.isInteger(8) || typeA.isInteger(1))
    if (llvm::is_contained(allowedNshort, sizeN))
      return success();
  return failure();
}

LogicalResult WarpgroupMmaOp::verify() {
  if (getTransposeA() && !getTransposeB())
    return emitOpError()
           << "supports non-transpose A (Row Major) "
              "and transpose B (Column Major) for the time being ";
  MemRefType matrixA = getDescriptorA().getType().getTensor();
  MemRefType matrixB = getDescriptorB().getType().getTensor();
  VectorType matrixC = getMatrixC().getType().getFragmented();
  VectorType matrixD = getMatrixD().getType().getFragmented();

  if (matrixC != matrixD)
    return emitOpError() << "type of matrix C and matrix D must be the same";

  if (matrixA.getRank() != 2 || matrixB.getRank() != 2 ||
      matrixC.getRank() != 2 || matrixD.getRank() != 2) {
    return emitOpError()
           << "has matrices A, B, C and D, they must be 2 dimensional";
  }

  if (matrixA.getShape()[1] != matrixB.getShape()[0])
    return emitOpError() << "2nd dim matrix-A (" << matrixA.getShape()[1]
                         << ")!= 1st dim matrix-B (" << matrixB.getShape()[0]
                         << " )";
  if (matrixA.getShape()[0] != matrixC.getShape()[0])
    return emitOpError() << "1st dim matrix-A ( " << matrixA.getShape()[0]
                         << " )!= 1st dim matrix-C ( " << matrixC.getShape()[0]
                         << " )";
  if (matrixB.getShape()[1] != matrixC.getShape()[1])
    return emitOpError() << "2nd dim matrix-B ( " << matrixB.getShape()[1]
                         << " ) != 2nd dim matrix-C ( " << matrixC.getShape()[1]
                         << " )";

  if (failed(isAllowedWGMMADataType(matrixC.getElementType(),
                                    matrixA.getElementType(),
                                    matrixB.getElementType())))
    return emitOpError() << matrixC.getElementType()
                         << " += " << matrixA.getElementType() << " * "
                         << matrixB.getElementType()
                         << ", it is not supported.";
  // Check N
  if (failed(isAllowedSizeN(matrixB.getDimSize(1), matrixA.getElementType()))) {
    return emitOpError() << "has input type " << matrixB << " n is set to "
                         << matrixB.getDimSize(1) << ", it is not supported";
  }

  // Currently, f16/bf16 supported
  if (!matrixC.getElementType().isF32() && !matrixA.getElementType().isF16() &&
      !matrixA.getElementType().isBF16()) {
    return emitOpError() << "hit a limitation: " << matrixC.getElementType()
                         << " += " << matrixA.getElementType() << " * "
                         << matrixB.getElementType()
                         << ", it is not supported yet";
  }

  return success();
}

LogicalResult WarpgroupMmaStoreOp::verify() {
  MemRefType dstMemrefType = getDstMemref().getType();
  VectorType vtype = getMatrixD().getType().getFragmented();

  // Limitation
  if (!vtype.getElementType().isF32()) {
    return emitOpError()
           << "hit a limitation: only f32 results for the time being";
  }
  if (vtype.getDimSize(0) != dstMemrefType.getDimSize(0) ||
      vtype.getDimSize(1) != dstMemrefType.getDimSize(1)) {
    return emitOpError() << "results [" << vtype << "][" << vtype.getDimSize(1)
                         << "] values. However, destination memref["
                         << dstMemrefType.getDimSize(0) << "]["
                         << dstMemrefType.getDimSize(1)
                         << "]  does not have same size as results";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// WarpgroupMmaInitAccumulatorOp
//===----------------------------------------------------------------------===//

LogicalResult WarpgroupMmaInitAccumulatorOp::verify() {

  nvgpu::WarpgroupAccumulatorType accType = getMatrixC().getType();
  int64_t sizeM = accType.getFragmented().getDimSize(0);
  int64_t sizeN = accType.getFragmented().getDimSize(1);
  Type elemType = accType.getFragmented().getElementType();

  if (failed(isAllowedSizeM(sizeM)) ||
      failed(isAllowedSizeN(sizeN, elemType))) {
    return emitOpError() << "has type " << accType.getFragmented()
                         << ". It does not fit into warp-group "
                            "level (wgmma) matrix multiplication instruction "
                            "(or not supported yet)";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RcpOp
//===----------------------------------------------------------------------===//

LogicalResult RcpOp::verify() {
  RcpRoundingModeAttr rounding = getRoundingAttr();
  bool ftz = getFtz();
  // Currently, only `rcp_approx` and `ftz` is supported.
  if (rounding.getValue() != RcpRoundingMode::APPROX || !ftz) {
    return emitOpError() << "has a limitation. " << rounding
                         << " or non-ftz is not supported yet.";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd dialect, type, and op definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPUAttrDefs.cpp.inc"

#include "mlir/Dialect/NVGPU/IR/NVGPUEnums.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPUOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPUTypeDefs.cpp.inc"
