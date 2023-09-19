//===--------- WmmaOpsToROCDL.cpp - GPU WMMA ops to ROCDL lowering --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of patterns to lower GPU Subgroup MMA ops to
// ROCDL Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/GPUToAMDGPU/GPUToAMDGPUPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace {

/// Checks if all the operands of the op being lowered are of LLVM Types. The
/// types are expected to be converted by the `LLVMTypeConverter` before the op
/// is actually lowered. If the type of an operands is not already converted it
/// hints a missing typeConversion and failure is returned in that case.
static LogicalResult areAllLLVMTypes(Operation *op, ValueRange operands,
                                     ConversionPatternRewriter &rewriter) {
  if (!llvm::all_of(operands, [](Value value) {
        return LLVM::isCompatibleType(value.getType());
      })) {
    return rewriter.notifyMatchFailure(
        op, "cannot convert if operands aren't of LLVM type.");
  }

  return success();
}

/// Return the WMMA operand corresponding to `operandName`.
static ROCDL::ROCDLWMMAFrag convertOperand(StringRef operandName) {
  if (operandName.equals("AOp"))
    return ROCDL::ROCDLWMMAFrag::a;
  if (operandName.equals("BOp"))
    return ROCDL::ROCDLWMMAFrag::b;
  if (operandName.equals("COp"))
    return ROCDL::ROCDLWMMAFrag::c;
  llvm_unreachable("Unknown operand name");
}

/// Generate load ops for `AOp` or `BOp`. `dataPtr` is the base address starting
/// from which values will be loaded. `laneId` lane ID of the thread loading the
/// values. `vecType` is the vector type of the values that will be loaded. The
/// loaded values are returned in `loadedValues`. The address for loading the
/// values is generated in the following manner:
///
/// wrappedLaneId = laneId % 16
/// for i in vectorSize {
///   loadedValues[i] = dataPtr + ((wrappedLaneId * leadingDim) + i);
/// }
static void generateAbLoadOpsVecFirst(Location loc, Value dataPtr, Value laneId,
                                      Value leadingDim, VectorType vecType,
                                      PatternRewriter &rewriter,
                                      Value &loadedValues) {
  // We wrap the laneId to 16 because of matrix replication in RDNA 3.
  Value wrapSize = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                     /*value=*/16);
  mlir::TypedAttr x;
  Value wrappedLaneId = rewriter.create<LLVM::SRemOp>(loc, laneId, wrapSize);
  loadedValues = rewriter.create<LLVM::UndefOp>(loc, vecType);
  Value laneIdLdm =
      rewriter.create<LLVM::MulOp>(loc, wrappedLaneId, leadingDim);
  for (unsigned i = 0; i < vecType.getNumElements(); ++i) {
    Value iter = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                   /*value=*/i);
    Value curInx = rewriter.create<LLVM::AddOp>(loc, laneIdLdm, iter);
    Value curAddress = rewriter.create<LLVM::GEPOp>(
        loc, dataPtr.getType(), vecType.getElementType(), dataPtr, curInx);
    // Load the value from the current index.
    Value loaded = rewriter.create<LLVM::LoadOp>(loc, vecType.getElementType(),
                                                 curAddress);
    loadedValues = rewriter.create<LLVM::InsertElementOp>(
        loc, vecType, loadedValues, loaded, iter);
  }
}

/// Generate load ops for `AOp` or `BOp`. `dataPtr` is the base address starting
/// from which values will be loaded. `laneId` is the lane ID of the thread
/// loading the values. `vecType` is the vector type of the values that will be
/// loaded. The loaded values are returned in `loadedValues`. The address for
/// loading the values is generated in the following manner:
///
/// wrappedLaneId = laneId % 16
/// for i in vectorSize {
///   loadedValues[i] = dataPtr + ((i * leadingDim) + wrappedLaneId);
/// }
static void generateAbLoadOpsLaneFirst(Location loc, Value dataPtr,
                                       Value laneId, Value leadingDim,
                                       VectorType vecType,
                                       PatternRewriter &rewriter,
                                       Value &loadedValues) {
  // We wrap the laneId to 16 because of matrix replication in RDNA 3.
  Value wrapSize = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                     /*value=*/16);
  Value wrappedLaneId = rewriter.create<LLVM::SRemOp>(loc, laneId, wrapSize);
  loadedValues = rewriter.create<LLVM::UndefOp>(loc, vecType);
  for (unsigned i = 0; i < vecType.getNumElements(); ++i) {
    Value iter = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                   /*value=*/i);
    Value iterLdm = rewriter.create<LLVM::MulOp>(loc, iter, leadingDim);
    Value curInx = rewriter.create<LLVM::AddOp>(loc, iterLdm, wrappedLaneId);
    Value curAddress = rewriter.create<LLVM::GEPOp>(
        loc, dataPtr.getType(), vecType.getElementType(), dataPtr, curInx);
    // Load the value from the current index.
    Value loaded = rewriter.create<LLVM::LoadOp>(loc, vecType.getElementType(),
                                                 curAddress);
    loadedValues = rewriter.create<LLVM::InsertElementOp>(
        loc, vecType, loadedValues, loaded, iter);
  }
}

/// Generate load ops for `COp`. `dataPtr` is the base address starting
/// from which values will be loaded. `laneId` is the lane ID  of the
/// thread loading the values. `vecType` is the vector type of the values that
/// will be loaded. The loaded values are returned in `loadedValues`. The
/// address for loading the values is generated in the following manner:
///
/// wrappedLaneId = laneId % 16
/// for i in vectorSize {
///   row = i * 2 + (laneId / 16)
///   if opSelect
///     loadedValues[i * 2 + 1] = dataPtr + ((row * leadingDim) +
///     wrappedLaneId);
///   else
///     loadedValues[i * 2] = dataPtr + ((row * leadingDim) + wrappedLaneId);
/// }
static void generateCLoadOpsLaneFirst(bool opSelect, Location loc,
                                      Value dataPtr, Value laneId,
                                      Value leadingDim, VectorType vecType,
                                      PatternRewriter &rewriter,
                                      Value &loadedValues) {
  // We wrap the laneId to 16 because of matrix replication in RDNA 3.
  Value wrapSize = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                     /*value=*/16);
  Value wrappedLaneId = rewriter.create<LLVM::SRemOp>(loc, laneId, wrapSize);
  loadedValues = rewriter.create<LLVM::UndefOp>(loc, vecType);
  Value constTwo = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                     /*value=*/2);
  Value sixteen = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                    /*value=*/16);
  Value laneIdHalf = rewriter.create<LLVM::SDivOp>(loc, laneId, sixteen);
  for (unsigned i = 0; i < vecType.getNumElements(); ++i) {
    Value iter = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                   /*value=*/i);
    Value iterTwo = rewriter.create<LLVM::MulOp>(loc, iter, constTwo);
    Value row = rewriter.create<LLVM::AddOp>(loc, iterTwo, laneIdHalf);
    Value rowLdm = rewriter.create<LLVM::MulOp>(loc, row, leadingDim);
    Value curInx = rewriter.create<LLVM::AddOp>(loc, rowLdm, wrappedLaneId);
    Value curAddress = rewriter.create<LLVM::GEPOp>(
        loc, dataPtr.getType(), vecType.getElementType(), dataPtr, curInx);
    // Load the value from the current index.
    Value loaded = rewriter.create<LLVM::LoadOp>(loc, vecType.getElementType(),
                                                 curAddress);
    // We have to skip every second element if opselect is true.
    Value inx = iter;
    if (vecType.getElementType().isF16()) {
      if (opSelect) {
        Value constOne =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                              /*value=*/1);
        inx = rewriter.create<LLVM::AddOp>(loc, iterTwo, constOne);
      } else {
        inx = iterTwo;
      }
    }
    loadedValues = rewriter.create<LLVM::InsertElementOp>(
        loc, vecType, loadedValues, loaded, inx);
  }
}

/// Generate load ops for `AOp`, `BOp`, or `COp`. `opSelect` is the opSelect bit
/// governing how to store/load half precision `COp` values. `transpose` tells
/// if the matrix has to be loaded in a transposed manner. `frag` is the type of
/// the WMMA operand being loaded. `dataPtr` is the base address starting from
/// which values will be loaded. `vecType` is the vector type of the values that
/// will be loaded. The loaded values are returned in `loadedValues`.
static LogicalResult generateLoadOps(bool opSelect, bool transpose,
                                     Location loc, ROCDL::ROCDLWMMAFrag frag,
                                     unsigned indexBitwidth, Value dataPtr,
                                     Value leadingDim, VectorType vecType,
                                     PatternRewriter &rewriter,
                                     Value &loadedValues) {
  Value laneId = amd::getLaneId(rewriter, loc, indexBitwidth);
  Type eltType = vecType.getElementType();
  if (frag == ROCDL::ROCDLWMMAFrag::a && !transpose && eltType.isF16()) {
    generateAbLoadOpsVecFirst(loc, dataPtr, laneId, leadingDim, vecType,
                              rewriter, loadedValues);
    return success();
  }
  if (frag == ROCDL::ROCDLWMMAFrag::a && transpose && eltType.isF16()) {
    generateAbLoadOpsLaneFirst(loc, dataPtr, laneId, leadingDim, vecType,
                               rewriter, loadedValues);
    return success();
  }
  if (frag == ROCDL::ROCDLWMMAFrag::b && transpose && eltType.isF16()) {
    generateAbLoadOpsVecFirst(loc, dataPtr, laneId, leadingDim, vecType,
                              rewriter, loadedValues);
    return success();
  }
  if (frag == ROCDL::ROCDLWMMAFrag::b && !transpose && eltType.isF16()) {
    generateAbLoadOpsLaneFirst(loc, dataPtr, laneId, leadingDim, vecType,
                               rewriter, loadedValues);
    return success();
  }
  if (frag == ROCDL::ROCDLWMMAFrag::c && !transpose &&
      (eltType.isF32() || eltType.isF16())) {
    generateCLoadOpsLaneFirst(opSelect, loc, dataPtr, laneId, leadingDim,
                              vecType, rewriter, loadedValues);
    return success();
  }

  return failure();
}

/// This class implements the conversion of GPU MMA loadOp to wmma.load op
/// in the ROCDL dialect. The conversion not only emits the ROCDL op but also
/// emits code that is necessary to store the data in the destination memref
/// after it has been loaded.
struct WmmaLoadOpToROCDLLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaLoadMatrixOp> {
  WmmaLoadOpToROCDLLowering(LLVMTypeConverter &typeConverter, StringRef chip,
                            unsigned indexBitwidth, unsigned warpSize)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaLoadMatrixOp>(typeConverter),
        indexBitwidth(indexBitwidth), warpSize(warpSize), chip(chip){};

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp subgroupMmaLoadMatrixOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(areAllLLVMTypes(subgroupMmaLoadMatrixOp.getOperation(),
                               adaptor.getOperands(), rewriter)))
      return failure();

    std::size_t firstPos = chip.find("gfx11");
    std::size_t lastPos = chip.rfind("gfx11");
    if (firstPos != 0 || (firstPos != lastPos))
      return subgroupMmaLoadMatrixOp->emitError(
          "wmma lowering is supported for gfx11 series only");

    if (warpSize != amd::kWaveFrontSize32)
      return subgroupMmaLoadMatrixOp->emitError(
          "only size 32 wavefronts are supported");

    auto transpose = subgroupMmaLoadMatrixOp.getTranspose();
    gpu::MMAMatrixType retType =
        subgroupMmaLoadMatrixOp.getRes().getType().cast<gpu::MMAMatrixType>();
    SmallVector<int64_t> retTypeShape(retType.getShape());

    if (!llvm::all_of(retTypeShape, [](int dim) { return dim == 16; }))
      return subgroupMmaLoadMatrixOp->emitError(
          "wmma ops of shape 16x16x16 are only supported.");

    auto srcMemrefType =
        subgroupMmaLoadMatrixOp.getSrcMemref().getType().cast<MemRefType>();

    if (srcMemrefType.getElementType() != retType.getElementType())
      return subgroupMmaLoadMatrixOp->emitError(
          "src memref type and mma matrix element type must be same");

    // Get the LLVM type of corresponding to the result MMAMatrixType.
    Type llvmRetType = amd::convertWMMAToROCDLLLVMType(retType);

    // We need to declare a vector type and then emit instructions to load the
    // elements into the vector type.
    Location loc = subgroupMmaLoadMatrixOp.getLoc();
    Value dataPtr =
        getStridedElementPtr(loc, srcMemrefType, adaptor.getSrcMemref(),
                             adaptor.getIndices(), rewriter);

    Value leadingDim = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        subgroupMmaLoadMatrixOp.getLeadDimensionAttr());

    Value loadedValues;
    ROCDL::ROCDLWMMAFrag operand = convertOperand(retType.getOperand());
    if (auto vecType = dyn_cast<VectorType>(llvmRetType)) {
      bool opSelect = subgroupMmaLoadMatrixOp->hasAttrOfType<UnitAttr>(
          amd::kAMDGpuOpselectAttrName);
      if (failed(generateLoadOps(opSelect,
                                 transpose.has_value() && transpose.value(),
                                 loc, operand, indexBitwidth, dataPtr,
                                 leadingDim, vecType, rewriter, loadedValues)))
        return rewriter.notifyMatchFailure(subgroupMmaLoadMatrixOp,
                                           "unsupported load op variant.");
      rewriter.replaceOp(subgroupMmaLoadMatrixOp, loadedValues);
      return success();
    }
    return rewriter.notifyMatchFailure(subgroupMmaLoadMatrixOp,
                                       "unsupported load op variant.");
  }

  /// Index bitwidth to use in any index calculation.
  unsigned indexBitwidth;

  /// `warpSize` is the warp size to use when generating WMMA intrinsics.
  unsigned warpSize;

  /// The target chip for which to generate the lowering.
  std::string chip;
};

/// Generate store ops for `COp`. `dataPtr` is the base address starting
/// to which the values will be stored. `laneId` is the lane ID  of the
/// thread loading the values. `vecType` is the vector type of the values that
/// are being stored. The values to be stored are supplied in `toStore`. The
/// address for storing the values is generated in the following manner:
///
/// wrappedLaneId = laneId % 16
/// for i in vectorSize {
///   row = i * 2 + (laneId / 16)
///   if opSelect
///     store toStore[i * 2 + 1], dataPtr + ((row * leadingDim) + wrappedLaneId)
///   else
///     store toStore[i * 2], dataPtr + ((row * leadingDim) + wrappedLaneId)
/// }
static void generateCStoreOpsLaneFirst(bool opSelect, Location loc,
                                       Value dataPtr, Value laneId,
                                       Value leadingDim, VectorType vecType,
                                       Value toStore,
                                       PatternRewriter &rewriter) {
  // We wrap the laneId to 16 because of matrix replication in RDNA 3.
  Value wrapSize = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                     /*value=*/16);
  Value wrappedLaneId = rewriter.create<LLVM::SRemOp>(loc, laneId, wrapSize);
  Value constSixteen =
      rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                        /*value=*/16);
  Value constTwo = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                     /*value=*/2);
  Value laneIdHalf = rewriter.create<LLVM::SDivOp>(loc, laneId, constSixteen);
  for (int i = 0; i < vecType.getNumElements(); ++i) {
    Value inx = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                  /*value=*/i);
    Value inxTimesTwo = rewriter.create<LLVM::MulOp>(loc, inx, constTwo);
    Value row = rewriter.create<LLVM::AddOp>(loc, laneIdHalf, inxTimesTwo);
    Value rowLdm = rewriter.create<LLVM::MulOp>(loc, row, leadingDim);
    Value offset = rewriter.create<LLVM::AddOp>(loc, rowLdm, wrappedLaneId);
    Value storeAddress = rewriter.create<LLVM::GEPOp>(
        loc, dataPtr.getType(), vecType.getElementType(), dataPtr, offset);
    Value toStoreAtInx;
    if (vecType.getElementType().isF16()) {
      if (!opSelect) {
        toStoreAtInx = rewriter.create<LLVM::ExtractElementOp>(
            loc, vecType.getElementType(), toStore, inxTimesTwo);

      } else {
        Value constOne =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                              /*value=*/1);
        Value inxTimesTwoAddOne =
            rewriter.create<LLVM::AddOp>(loc, inxTimesTwo, constOne);
        toStoreAtInx = rewriter.create<LLVM::ExtractElementOp>(
            loc, vecType.getElementType(), toStore, inxTimesTwoAddOne);
      }
    } else if (vecType.getElementType().isF32()) {
      toStoreAtInx = rewriter.create<LLVM::ExtractElementOp>(
          loc, vecType.getElementType(), toStore, inx);
    }
    rewriter.create<LLVM::StoreOp>(loc, toStoreAtInx, storeAddress);
  }
}

/// Generate store ops for `COp`. `opSelect` is the opSelect bit governing how
/// to store half precision `COp` values. `frag` is the type of the WMMA
/// operand being stored. `dataPtr` is the base address starting from which
/// starting from which the values will be stored. `vecType` is the vector type
/// of the values being stored. `toStore` contains the values to be stored.
static LogicalResult generateStoreOps(bool opSelect, Location loc,
                                      ROCDL::ROCDLWMMAFrag frag, Value dataPtr,
                                      unsigned indexBitwidth, Value leadingDim,
                                      VectorType vecType, Value toStore,
                                      PatternRewriter &rewriter) {
  // Store ops can only be generated for C operands.
  if (frag != ROCDL::ROCDLWMMAFrag::c)
    return emitError(toStore.getLoc(), "only COp can be stored");

  // Get the laneID.
  Value laneId = amd::getLaneId(rewriter, loc, indexBitwidth);
  Type eltType = vecType.getElementType();
  if (eltType.isF16() || eltType.isF32()) {
    generateCStoreOpsLaneFirst(opSelect, loc, dataPtr, laneId, leadingDim,
                               vecType, toStore, rewriter);
    return success();
  }

  return failure();
}

/// This class implements the conversion of GPU MMA storeOp to wmma.store op
/// in the ROCDL dialect.
struct WmmaStoreOpToROCDLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaStoreMatrixOp> {
  WmmaStoreOpToROCDLowering(LLVMTypeConverter &typeConverter, StringRef chip,
                            unsigned indexBitwidth, unsigned warpSize)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaStoreMatrixOp>(typeConverter),
        indexBitwidth(indexBitwidth), warpSize(warpSize), chip(chip){};

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaStoreMatrixOp subgroupMmaStoreMatrixOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(areAllLLVMTypes(subgroupMmaStoreMatrixOp.getOperation(),
                               adaptor.getOperands(), rewriter)))
      return failure();

    std::size_t firstPos = chip.find("gfx11");
    std::size_t lastPos = chip.rfind("gfx11");
    if (firstPos != 0 || (firstPos != lastPos))
      return subgroupMmaStoreMatrixOp->emitError(
          "wmma lowering is supported for gfx11 series only");

    if (warpSize != amd::kWaveFrontSize32)
      return subgroupMmaStoreMatrixOp->emitError(
          "wavefront of size 32 only supported");

    Location loc = subgroupMmaStoreMatrixOp->getLoc();

    auto transpose = subgroupMmaStoreMatrixOp.getTranspose();
    if (transpose.has_value() && transpose.value())
      return subgroupMmaStoreMatrixOp->emitError(
          "lowering with transpose is not supported.");

    gpu::MMAMatrixType retType =
        subgroupMmaStoreMatrixOp.getSrc().getType().cast<gpu::MMAMatrixType>();
    SmallVector<int64_t> retTypeShape(retType.getShape());

    if (!llvm::all_of(retTypeShape, [](int dim) { return dim == 16; }))
      return subgroupMmaStoreMatrixOp->emitError(
          "wmma ops of shape 16x16x16 are only supported.");

    auto dstMemrefType =
        subgroupMmaStoreMatrixOp.getDstMemref().getType().cast<MemRefType>();

    if (dstMemrefType.getElementType() != retType.getElementType())
      return subgroupMmaStoreMatrixOp->emitError(
          "dst memref type and mma matrix element type must be same");

    Value dataPtr = getStridedElementPtr(
        loc,
        subgroupMmaStoreMatrixOp.getDstMemref().getType().cast<MemRefType>(),
        adaptor.getDstMemref(), adaptor.getIndices(), rewriter);
    Value leadingDim = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        subgroupMmaStoreMatrixOp.getLeadDimensionAttr());

    // Get the LLVM type of corresponding to the result MMAMatrixType.
    Type llvmRetType = amd::convertWMMAToROCDLLLVMType(retType);

    Value toStore = adaptor.getSrc();

    bool opSelect = subgroupMmaStoreMatrixOp->hasAttrOfType<UnitAttr>(
        amd::kAMDGpuOpselectAttrName);
    if (auto vecType = dyn_cast<VectorType>(llvmRetType)) {
      if (failed(generateStoreOps(
              opSelect, loc, convertOperand(retType.getOperand()), dataPtr,
              indexBitwidth, leadingDim, vecType, toStore, rewriter)))
        return rewriter.notifyMatchFailure(subgroupMmaStoreMatrixOp,
                                           "unsupported store op variant.");
    }
    rewriter.eraseOp(subgroupMmaStoreMatrixOp);
    return success();
  }

  /// Index bitwidth to use in any index calculation.
  unsigned indexBitwidth;

  /// `warpSize` is the warp size to use when generating WMMA intrinsics.
  unsigned warpSize;

  /// The target chip for which to generate the lowering.
  std::string chip;
};
} // namespace

// Convert the MMAMatrix type to LLVM types based of the elemental type of
// MMAMatrixType.
Type mlir::amd::convertWMMAToROCDLLLVMType(
    mlir::gpu::MMAMatrixType matrixType) {
  Type eltType = matrixType.getElementType();
  ROCDL::ROCDLWMMAFrag frag = convertOperand(matrixType.getOperand());
  if (eltType.isF16() &&
      (frag == ROCDL::ROCDLWMMAFrag::a || frag == ROCDL::ROCDLWMMAFrag::b ||
       frag == ROCDL::ROCDLWMMAFrag::c))
    return VectorType::get({16}, eltType);
  if (eltType.isF32() && frag == ROCDL::ROCDLWMMAFrag::c)
    return VectorType::get({8}, eltType);

  llvm_unreachable("Unsupported data type");
}

void mlir::populateGpuWMMAToROCDLConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, StringRef chip,
    unsigned indexBitwidth, unsigned warpSize) {
  patterns.add<WmmaLoadOpToROCDLLowering, WmmaStoreOpToROCDLowering>(
      converter, chip, indexBitwidth, warpSize);
}
