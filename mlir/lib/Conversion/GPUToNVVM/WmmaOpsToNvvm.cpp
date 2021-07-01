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

/// Return the LLVMStructureType corresponding to the MMAMatrixType `type`.
static LLVM::LLVMStructType convertMMAToLLVMType(gpu::MMAMatrixType type) {
  StringRef operandStr = type.getOperand();
  assert(type.getElementType().isa<FloatType>());
  Type baseType = type.getElementType().isF16()
                      ? VectorType::get(2, type.getElementType())
                      : type.getElementType();
  auto getLLVMType = [&](int64_t numElements) {
    return LLVM::LLVMStructType::getLiteral(
        type.getContext(), SmallVector<Type, 8>(numElements, baseType));
  };
  if (operandStr.equals("AOp") || operandStr.equals("BOp"))
    return getLLVMType(8);
  if (type.getElementType().isF16())
    return getLLVMType(4);
  return getLLVMType(8);
}

/// This class implements the conversion of GPU MMA loadOp to wmma.load op
/// in the NVVM dialect. The conversion not only emits the NVVM op but also
/// emits code that is necessary to store the data in the destination memref
/// after it has been loaded.
struct WmmaLoadOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaLoadMatrixOp> {
  using ConvertOpToLLVMPattern<
      gpu::SubgroupMmaLoadMatrixOp>::ConvertOpToLLVMPattern;

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
    Location loc = op->getLoc();

    auto leadDimension = subgroupMmaLoadMatrixOp.leadDimensionAttr();

    gpu::SubgroupMmaLoadMatrixOpAdaptor adaptor(operands);
    // MemRefDescriptor to extract alignedPtr and offset.
    MemRefDescriptor promotedSrcOp(adaptor.srcMemref());

    // Emit ops which compute the load offset using `srcOffsetI`,
    // `srcOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr +
    // ((leadDimension * srcOffsetI) + srcOffsetJ)). The memrefs here are
    // assumed to be normalized and hence the simple conversion works.
    SmallVector<Value> indices(adaptor.indices());
    Value srcOffsetIVal = indices[0];
    Value srcOffsetJVal = indices[1];
    Type i32Ty = rewriter.getI32Type();
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
        loc, promotedSrcOp.getElementPtrType(),
        promotedSrcOp.alignedPtr(rewriter, loc), ArrayRef<Value>{actualOffset});

    // Bitcast the base address pointer of the destination memref, So that
    // values can be stored in chunks of 32-bits and semantics match with the
    // intrinsic exposed by NVPTX backend.
    Value loadAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(
            i32Ty, promotedSrcOp.getElementPtrType().getAddressSpace()),
        loadAddress);

    // Get the shape of the MMAMatrix type being returned. The shape will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType retType =
        subgroupMmaLoadMatrixOp.res().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> retTypeShape = retType.getShape();

    Type resType = convertMMAToLLVMType(retType);
    StringRef operandStr = retType.getOperand();

    // Create nvvm.mma_load op according to the operand types.
    SmallVector<Value, 2> loadOpOperands({loadAddressCasted, leadingDim32});
    if (operandStr.equals("AOp")) {
      if (retTypeShape[0] == 16 && retTypeShape[1] == 16) {
        rewriter.replaceOpWithNewOp<NVVM::WMMALoadAM16N16K16Op>(op, resType,
                                                                loadOpOperands);
      } else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
    } else if (operandStr.equals("BOp")) {
      if (retTypeShape[0] == 16 && retTypeShape[1] == 16) {
        rewriter.replaceOpWithNewOp<NVVM::WMMALoadBM16N16K16Op>(op, resType,
                                                                loadOpOperands);
      } else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
    } else {
      if (retTypeShape[0] == 16 && retTypeShape[1] == 16) {
        if (retType.getElementType().isF16()) {
          rewriter.replaceOpWithNewOp<NVVM::WMMALoadCF16M16N16K16Op>(
              op, resType, loadOpOperands);
        } else if (retType.getElementType().isF32()) {
          rewriter.replaceOpWithNewOp<NVVM::WMMALoadCF32M16N16K16Op>(
              op, resType, loadOpOperands);
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
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaStoreMatrixOp> {
  using ConvertOpToLLVMPattern<
      gpu::SubgroupMmaStoreMatrixOp>::ConvertOpToLLVMPattern;

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

    gpu::SubgroupMmaStoreMatrixOpAdaptor adaptor(operands);
    // MemRefDescriptor to extract alignedPtr and offset.
    MemRefDescriptor promotedDstOp(adaptor.dstMemref());

    auto leadDimension = subgroupMmaStoreMatrixOp.leadDimensionAttr();

    // Emit ops which compute the store offset using `dstOffsetI`,
    // `dstOffsetJ`. The actualOffset is (memrefOffset + (alignedPtr +
    // ((leadDimension * dstOffsetI) + dstOffsetJ)).
    SmallVector<Value> indices(adaptor.indices());
    Value dstOffsetIVal = indices[0];
    Value dstOffsetJVal = indices[1];
    Type i32Ty = rewriter.getI32Type();
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
        loc, promotedDstOp.getElementPtrType(),
        promotedDstOp.alignedPtr(rewriter, loc), ArrayRef<Value>{actualOffset});

    // Bitcast the base address pointer of the destination memref, So that
    // values can be stored in chunks of 32-bits and semantics match with the
    // intrinsic exposed by NVPTX backend.
    Value storeAddressCasted = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(
            i32Ty, promotedDstOp.getElementPtrType().getAddressSpace()),
        storeAddress);

    SmallVector<Value, 4> storeOpOperands;
    storeOpOperands.push_back(storeAddressCasted);

    // Get the shape of the MMAMatrix type being stored. The shape will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType srcType =
        subgroupMmaStoreMatrixOp.src().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> srcTypeShape = srcType.getShape();

    auto matrixType = adaptor.src().getType().cast<LLVM::LLVMStructType>();
    for (unsigned i = 0, e = matrixType.getBody().size(); i < e; ++i) {
      Value toUse = rewriter.create<LLVM::ExtractValueOp>(
          loc, matrixType.getBody()[i], adaptor.src(),
          rewriter.getI32ArrayAttr(i));
      storeOpOperands.push_back(toUse);
    }
    storeOpOperands.push_back(leadingDim32);
    // Unpack the results from the source.
    if (srcType.getElementType().isF16()) {
      // Create nvvm.mma_store op.
      if (srcTypeShape[0] == 16 && srcTypeShape[1] == 16) {
        rewriter.create<NVVM::WMMAStoreF16M16N16K16Op>(loc, storeOpOperands);
      } else {
        return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
      }
      rewriter.eraseOp(op);
      return success();
    }
    if (srcType.getElementType().isF32()) {
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
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaComputeOp> {
  using ConvertOpToLLVMPattern<
      gpu::SubgroupMmaComputeOp>::ConvertOpToLLVMPattern;

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

    auto unpackOp = [&](Value operand) {
      auto structType = operand.getType().cast<LLVM::LLVMStructType>();
      for (size_t i = 0, e = structType.getBody().size(); i < e; ++i) {
        Value toUse = rewriter.create<LLVM::ExtractValueOp>(
            loc, structType.getBody()[i], operand, rewriter.getI32ArrayAttr(i));
        unpackedOps.push_back(toUse);
      }
    };

    // Get the shapes of the MMAMatrix type being used. The shapes will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType aType =
        subgroupMmaComputeOp.opA().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> aTypeShape = aType.getShape();
    gpu::MMAMatrixType bType =
        subgroupMmaComputeOp.opB().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> bTypeShape = bType.getShape();
    gpu::MMAMatrixType cType =
        subgroupMmaComputeOp.opC().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> cTypeShape = cType.getShape();

    gpu::SubgroupMmaComputeOpAdaptor transformedOperands(operands);
    unpackOp(transformedOperands.opA());
    unpackOp(transformedOperands.opB());
    unpackOp(transformedOperands.opC());

    if (cType.getElementType().isF16()) {
      if (aTypeShape[0] == 16 && aTypeShape[1] == 16 && bTypeShape[0] == 16 &&
          bTypeShape[1] == 16 && cTypeShape[0] == 16 && cTypeShape[1] == 16) {
        // Create nvvm.wmma.mma op.
        rewriter.replaceOpWithNewOp<NVVM::WMMAMmaF16F16M16N16K16Op>(
            op, transformedOperands.opC().getType(), unpackedOps);

        return success();
      }
      return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
    }
    if (cType.getElementType().isF32()) {
      if (aTypeShape[0] == 16 && aTypeShape[1] == 16 && bTypeShape[0] == 16 &&
          bTypeShape[1] == 16 && cTypeShape[0] == 16 && cTypeShape[1] == 16) {
        // Create nvvm.wmma.mma op.
        rewriter.replaceOpWithNewOp<NVVM::WMMAMmaF32F32M16N16K16Op>(
            op, transformedOperands.opC().getType(), unpackedOps);

        return success();
      }
      return rewriter.notifyMatchFailure(op, kInvalidCaseStr);
    }
    return failure();
  }
};

/// Convert GPU MMA ConstantMatrixOp to a chain of InsertValueOp.
struct WmmaConstantOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaConstantMatrixOp> {
  using ConvertOpToLLVMPattern<
      gpu::SubgroupMmaConstantMatrixOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaConstantMatrixOp subgroupMmaConstantOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(areAllLLVMTypes(subgroupMmaConstantOp.getOperation(), operands,
                               rewriter)))
      return failure();
    Location loc = subgroupMmaConstantOp.getLoc();
    Value cst = operands[0];
    LLVM::LLVMStructType type = convertMMAToLLVMType(
        subgroupMmaConstantOp.getType().cast<gpu::MMAMatrixType>());
    // If the element type is a vector create a vector from the operand.
    if (auto vecType = type.getBody()[0].dyn_cast<VectorType>()) {
      Value vecCst = rewriter.create<LLVM::UndefOp>(loc, vecType);
      for (int64_t vecEl = 0; vecEl < vecType.getNumElements(); vecEl++) {
        Value idx = rewriter.create<LLVM::ConstantOp>(
            loc, typeConverter->convertType(rewriter.getIntegerType(32)),
            rewriter.getI32IntegerAttr(vecEl));
        vecCst = rewriter.create<LLVM::InsertElementOp>(loc, vecType, vecCst,
                                                        cst, idx);
      }
      cst = vecCst;
    }
    Value matrixStruct = rewriter.create<LLVM::UndefOp>(loc, type);
    for (size_t i : llvm::seq(size_t(0), type.getBody().size())) {
      matrixStruct = rewriter.create<LLVM::InsertValueOp>(
          loc, matrixStruct, cst, rewriter.getI32ArrayAttr(i));
    }
    rewriter.replaceOp(subgroupMmaConstantOp, matrixStruct);
    return success();
  }
};

} // anonymous namespace

namespace mlir {
void populateGpuWMMAToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns) {
  patterns.insert<WmmaLoadOpToNVVMLowering, WmmaMmaOpToNVVMLowering,
                  WmmaStoreOpToNVVMLowering, WmmaConstantOpToNVVMLowering>(
      converter);
}
} // namespace mlir
