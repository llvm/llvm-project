//===------ WmmaOpsToNVVM.cpp - WMMA LD/ST/Compute to NVVM lowering -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of patterns to lower GPU Subgroup MMA ops to
// NVVM Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;

namespace {

/// Contains all the common LLVM types which are used across the lowerings of
/// GPU subgroup ops to NVVM dialect.
struct CommonLLVMAndBuiltInMLIRTypes {
public:
  CommonLLVMAndBuiltInMLIRTypes(MLIRContext *context) {
    numHalfsInOpFrags.resize(4);
    numHalfsInOpFrags[A] = 8;
    numHalfsInOpFrags[B] = 8;
    numHalfsInOpFrags[C] = 4;
    i32Ty = IntegerType::get(context, 32);
    f16Ty = FloatType::getF16(context);
    f32Ty = FloatType::getF32(context);
    f16x2Ty = VectorType::get(2, f16Ty);
    fragArrayABTy = LLVM::LLVMStructType::getLiteral(
        context, SmallVector<Type>(8, f16x2Ty));
    fragArrayCDTy = LLVM::LLVMStructType::getLiteral(
        context, SmallVector<Type>(4, f16x2Ty));
    fragArrayCDF32Ty =
        LLVM::LLVMStructType::getLiteral(context, SmallVector<Type>(8, f32Ty));
  };

  Type i32Ty;
  Type f16Ty;
  Type f32Ty;
  Type f16x2Ty;
  /// Type for the fragment of A and B operands that a single thread holds for
  /// fp16 data type in a WMMA operation of the form D = (alpha*(A*B)) +
  /// (beta*C).
  Type fragArrayABTy;
  /// Type for the fragment of C and D operands that a single thread holds for
  /// fp16 data type in a WMMA operation of the form D = (alpha*(A*B)) +
  /// (beta*C).
  Type fragArrayCDTy;
  /// Type for the fragment of C and D operands that a single thread holds for
  /// fp32 data type in a WMMA operation of the form D = (alpha*(A*B)) +
  /// (beta*C).
  Type fragArrayCDF32Ty;
  /// Represents the number of f16 elements a single thread holds in a WMMA
  /// operation of the form D = (alpha*(A*B)) + (beta*C) .
  SmallVector<unsigned, 4> numHalfsInOpFrags;
  /// Represents the operands of a MMA operation of the form D = (alpha*(A*B)) +
  /// (beta*C).
  enum OperandMap { A, B, C };
};

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

/// Error string to emit when unimplemented WMMA variant is encountered.
static constexpr StringRef kInvalidCaseStr =
    "Unimplemented WMMA variant, Only M16N16K16 version implemented.";

/// This class implements the conversion of GPU MMA loadOp to wmma.load op
/// in the NVVM dialect. The conversion not only emits the NVVM op but also
/// emits code that is necessary to store the data in the destination memref
/// after it has been loaded.
struct WmmaLoadOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaLoadMatrixOp>,
      private CommonLLVMAndBuiltInMLIRTypes {
public:
  explicit WmmaLoadOpToNVVMLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaLoadMatrixOp>(typeConverter),
        CommonLLVMAndBuiltInMLIRTypes(&this->getTypeConverter()->getContext()) {
  }

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp subgroupMmaLoadMatrixOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaLoadMatrixOp.getOperation();
    if (failed(areAllLLVMTypes(op, operands, rewriter)))
      return failure();

    unsigned indexTypeBitwidth =
        this->getTypeConverter()->getIndexTypeBitwidth();

    // The corresponding intrinsics expects leadDimension to be a 32-bit
    // integer, so all the calculations of linearizing the load address
    // must also follow this restriction.
    if (indexTypeBitwidth != 32)
      return rewriter.notifyMatchFailure(
          op, "Expected indices to the memref to be 32-bit wide.");

    // Source memref of the original op.
    MemRefType srcMemrefType =
        subgroupMmaLoadMatrixOp.srcMemref().getType().cast<MemRefType>();
    Location loc = op->getLoc();

    auto leadDimension = subgroupMmaLoadMatrixOp.leadDimensionAttr();

    // MemRefDescriptor to extract alignedPtr and offset.
    MemRefDescriptor promotedSrcOp(
        gpu::SubgroupMmaLoadMatrixOpAdaptor(operands).srcMemref());

    // Emit ops which compute the load offset using `srcOffsetI`,
    // `srcOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr +
    // ((leadDimension * srcOffsetI) + srcOffsetJ)). The memrefs here are
    // assumed to be normalized and hence the simple conversion works.
    SmallVector<Value> indices(subgroupMmaLoadMatrixOp.indices());
    Value srcOffsetIVal = indices[0];
    Value srcOffsetJVal = indices[1];
    Value leadingDim32 =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, leadDimension);
    Value numElemsLeadDim =
        rewriter.create<LLVM::MulOp>(loc, i32Ty, leadingDim32, srcOffsetIVal);
    Value loadOffset = rewriter.create<LLVM::AddOp>(loc, i32Ty, numElemsLeadDim,
                                                    srcOffsetJVal);

    Value promotedSrcOpToUse;
    promotedSrcOpToUse = promotedSrcOp.offset(rewriter, loc);
    Value actualOffset = rewriter.create<LLVM::AddOp>(loc, i32Ty, loadOffset,
                                                      promotedSrcOpToUse);
    Value loadAddress = rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(f16Ty, srcMemrefType.getMemorySpaceAsInt()),
        promotedSrcOp.alignedPtr(rewriter, loc), ArrayRef<Value>{actualOffset});

    // Bitcast the base address pointer of the destination memref, So that
    // values can be stored in chunks of 32-bits and semantics match with the
    // intrinsic exposed by NVPTX backend.
    Value loadAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(i32Ty, srcMemrefType.getMemorySpaceAsInt()),
        loadAddress);

    // Get the shape of the MMAMatrix type being returned. The shape will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType retType =
        subgroupMmaLoadMatrixOp.res().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> retTypeShape = retType.getShape();

    Type resType;
    StringRef operandStr = retType.getOperand();
    if (operandStr.equals("AOp") || operandStr.equals("BOp")) {
      resType = fragArrayABTy;
    } else {
      if (srcMemrefType.getElementType().isF16())
        resType = fragArrayCDTy;
      else if (srcMemrefType.getElementType().isF32())
        resType = fragArrayCDF32Ty;
      else
        return failure();
    }

    // Create nvvm.mma_load op according to the operand types.
    SmallVector<Value, 2> loadOpOperands({loadAddressCasted, leadingDim32});
    if (operandStr.equals("AOp")) {
      if (retTypeShape[0] == 16 && retTypeShape[1] == 16) {
        NVVM::WMMALoadAM16N16K16Op wmmaLoadAOp =
            rewriter.create<NVVM::WMMALoadAM16N16K16Op>(loc, resType,
                                                        loadOpOperands);
        rewriter.replaceOp(op, wmmaLoadAOp.getResult());
      } else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
    } else if (operandStr.equals("BOp")) {
      if (retTypeShape[0] == 16 && retTypeShape[1] == 16) {
        NVVM::WMMALoadBM16N16K16Op wmmaLoadBOp =
            rewriter.create<NVVM::WMMALoadBM16N16K16Op>(loc, resType,
                                                        loadOpOperands);
        rewriter.replaceOp(op, wmmaLoadBOp.getResult());
      } else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
    } else {
      if (retTypeShape[0] == 16 && retTypeShape[1] == 16) {
        if (srcMemrefType.getElementType().isF16()) {
          NVVM::WMMALoadCF16M16N16K16Op wmmaLoadCOp =
              rewriter.create<NVVM::WMMALoadCF16M16N16K16Op>(loc, resType,
                                                             loadOpOperands);
          rewriter.replaceOp(op, wmmaLoadCOp.getResult());
        } else if (srcMemrefType.getElementType().isF32()) {
          NVVM::WMMALoadCF32M16N16K16Op wmmaLoadCOp =
              rewriter.create<NVVM::WMMALoadCF32M16N16K16Op>(loc, resType,
                                                             loadOpOperands);
          rewriter.replaceOp(op, wmmaLoadCOp.getResult());
        }
      } else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
    }
    return success();
  }
};

/// This class implements the conversion of GPU MMA storeOp to wmma.store op
/// in the NVVM dialect. The conversion not only emits the NVVM op but also
/// emits code that is necessary to unpack the data in the source and
/// convert the data in the format that is needed by the NVVM op.
struct WmmaStoreOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaStoreMatrixOp>,
      private CommonLLVMAndBuiltInMLIRTypes {
public:
  explicit WmmaStoreOpToNVVMLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaStoreMatrixOp>(typeConverter),
        CommonLLVMAndBuiltInMLIRTypes(&this->getTypeConverter()->getContext()) {
  }

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaStoreMatrixOp subgroupMmaStoreMatrixOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaStoreMatrixOp.getOperation();
    if (failed(areAllLLVMTypes(op, operands, rewriter)))
      return failure();

    unsigned indexTypeBitwidth =
        this->getTypeConverter()->getIndexTypeBitwidth();
    // The corresponding intrinsics expects leadDimension to be a 32-bit
    // integer, so all the calculations of linearizing the store address
    // must also follow this restriction.
    if (indexTypeBitwidth != 32)
      return rewriter.notifyMatchFailure(
          op, "expected indices to the memref to be 32-bit wide.");

    Location loc = op->getLoc();

    // Destination memref of the original op.
    MemRefType dstMemrefType =
        subgroupMmaStoreMatrixOp.dstMemref().getType().cast<MemRefType>();

    // MemRefDescriptor to extract alignedPtr and offset.
    MemRefDescriptor promotedDstOp(
        gpu::SubgroupMmaStoreMatrixOpAdaptor(operands).dstMemref());

    auto leadDimension = subgroupMmaStoreMatrixOp.leadDimensionAttr();

    // Emit ops which compute the store offset using `dstOffsetI`,
    // `dstOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr +
    // ((leadDimension * dstOffsetI) + dstOffsetJ)).
    SmallVector<Value> indices(subgroupMmaStoreMatrixOp.indices());
    Value dstOffsetIVal = indices[0];
    Value dstOffsetJVal = indices[1];
    Value leadingDim32 =
        rewriter.create<LLVM::ConstantOp>(loc, i32Ty, leadDimension);
    Value numElemsLeadDim =
        rewriter.create<LLVM::MulOp>(loc, i32Ty, leadingDim32, dstOffsetIVal);
    Value loadOffset = rewriter.create<LLVM::AddOp>(loc, i32Ty, numElemsLeadDim,
                                                    dstOffsetJVal);

    Value promotedDstOpToUse;
    promotedDstOpToUse = promotedDstOp.offset(rewriter, loc);
    Value actualOffset = rewriter.create<LLVM::AddOp>(loc, i32Ty, loadOffset,
                                                      promotedDstOpToUse);
    Value storeAddress = rewriter.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(f16Ty, dstMemrefType.getMemorySpaceAsInt()),
        promotedDstOp.alignedPtr(rewriter, loc), ArrayRef<Value>{actualOffset});

    // Bitcast the base address pointer of the destination memref, So that
    // values can be stored in chunks of 32-bits and semantics match with the
    // intrinsic exposed by NVPTX backend.
    Value storeAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(i32Ty, dstMemrefType.getMemorySpaceAsInt()),
        storeAddress);

    SmallVector<Value, 4> storeOpOperands;
    storeOpOperands.push_back(storeAddressCasted);

    // Get the shape of the MMAMatrix type being stored. The shape will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType srcType =
        subgroupMmaStoreMatrixOp.src().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> srcTypeShape = srcType.getShape();

    // Unpack the results from the source.
    if (subgroupMmaStoreMatrixOp.src()
            .getType()
            .cast<gpu::MMAMatrixType>()
            .getElementType() == f16Ty) {
      for (unsigned i = 0, e = numHalfsInOpFrags[C]; i < e; ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, f16x2Ty, operands[0], rewriter.getI32ArrayAttr(i));
        storeOpOperands.push_back(toUse);
      }
      storeOpOperands.push_back(leadingDim32);

      // Create nvvm.mma_store op.
      if (srcTypeShape[0] == 16 && srcTypeShape[1] == 16) {
        rewriter.create<NVVM::WMMAStoreF16M16N16K16Op>(loc, storeOpOperands);
      } else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
      rewriter.eraseOp(op);
      return success();
    } else if (subgroupMmaStoreMatrixOp.src()
                   .getType()
                   .cast<gpu::MMAMatrixType>()
                   .getElementType() == f32Ty) {
      for (unsigned i = 0, e = 8; i < e; ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, f32Ty, operands[0], rewriter.getI32ArrayAttr(i));
        storeOpOperands.push_back(toUse);
      }
      storeOpOperands.push_back(leadingDim32);

      // Create nvvm.mma_store op.
      if (srcTypeShape[0] == 16 && srcTypeShape[1] == 16)
        rewriter.create<NVVM::WMMAStoreF32M16N16K16Op>(loc, storeOpOperands);
      else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

/// This class implements the conversion of GPU MMA computeOp to wmma.mma op
/// in the NVVM dialect.
struct WmmaMmaOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaComputeOp>,
      private CommonLLVMAndBuiltInMLIRTypes {
  explicit WmmaMmaOpToNVVMLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<gpu::SubgroupMmaComputeOp>(typeConverter),
        CommonLLVMAndBuiltInMLIRTypes(&this->getTypeConverter()->getContext()) {
  }

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaComputeOp subgroupMmaComputeOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaComputeOp.getOperation();
    if (failed(areAllLLVMTypes(op, operands, rewriter)))
      return failure();

    Location loc = op->getLoc();

    // The wmma.mma intrinsic in llvm requires the operands as individual
    // values. So individual elements from the memrefs need to be extracted and
    // then passed on to the intrinsic call. Emit llvm ops to extract individual
    // values form lowered memrefs.
    SmallVector<Value> unpackedOps;

    auto unpackOp = [&](CommonLLVMAndBuiltInMLIRTypes::OperandMap op,
                        Value operand, unsigned numElems, Type elemType) {
      for (unsigned i = 0; i < numElems; ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, elemType, operand, rewriter.getI32ArrayAttr(i));
        unpackedOps.push_back(toUse);
      }
    };

    // Get the shapes of the MMAMatrix type being used. The shapes will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType aType =
        subgroupMmaComputeOp.opA().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> aTypeShape = aType.getShape();
    gpu::MMAMatrixType bType =
        subgroupMmaComputeOp.opA().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> bTypeShape = bType.getShape();
    gpu::MMAMatrixType cType =
        subgroupMmaComputeOp.opA().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> cTypeShape = cType.getShape();

    gpu::SubgroupMmaComputeOpAdaptor transformedOperands(operands);
    if (subgroupMmaComputeOp.opC()
            .getType()
            .cast<gpu::MMAMatrixType>()
            .getElementType() == f16Ty) {
      unpackOp(A, transformedOperands.opA(), numHalfsInOpFrags[A], f16x2Ty);
      unpackOp(B, transformedOperands.opB(), numHalfsInOpFrags[B], f16x2Ty);
      unpackOp(C, transformedOperands.opC(), numHalfsInOpFrags[C], f16x2Ty);

      if (aTypeShape[0] == 16 && aTypeShape[1] == 16 && bTypeShape[0] == 16 &&
          bTypeShape[1] == 16 && cTypeShape[0] == 16 && cTypeShape[1] == 16) {
        // Create nvvm.wmma.mma op.
        NVVM::WMMAMmaF16F16M16N16K16Op wmmaMmaOp =
            rewriter.create<NVVM::WMMAMmaF16F16M16N16K16Op>(loc, fragArrayCDTy,
                                                            unpackedOps);

        rewriter.replaceOp(op, wmmaMmaOp.getResult());
        return success();
      } else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
    } else if (subgroupMmaComputeOp.opC()
                   .getType()
                   .cast<gpu::MMAMatrixType>()
                   .getElementType() == f32Ty) {
      unpackOp(A, transformedOperands.opA(), numHalfsInOpFrags[A], f16x2Ty);
      unpackOp(B, transformedOperands.opB(), numHalfsInOpFrags[B], f16x2Ty);
      unpackOp(C, transformedOperands.opC(), 8, f32Ty);

      if (aTypeShape[0] == 16 && aTypeShape[1] == 16 && bTypeShape[0] == 16 &&
          bTypeShape[1] == 16 && cTypeShape[0] == 16 && cTypeShape[1] == 16) {
        // Create nvvm.wmma.mma op.
        NVVM::WMMAMmaF32F32M16N16K16Op wmmaMmaOp =
            rewriter.create<NVVM::WMMAMmaF32F32M16N16K16Op>(
                loc, fragArrayCDF32Ty, unpackedOps);

        rewriter.replaceOp(op, wmmaMmaOp.getResult());
        return success();
      } else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
    }

    return failure();
  }
};

} // anonymous namespace

namespace mlir {
void populateGpuWMMAToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns) {
  patterns.insert<WmmaLoadOpToNVVMLowering>(converter);
  patterns.insert<WmmaMmaOpToNVVMLowering>(converter);
  patterns.insert<WmmaStoreOpToNVVMLowering>(converter);
}
} // namespace mlir
