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

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/TypeUtilities.h"

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

/// Error string to emit when an unimplemented WMMA variant is encountered.
static constexpr StringRef kInvalidCaseStr = "Unsupported WMMA variant.";

static NVVM::MMAFrag convertOperand(StringRef operandName) {
  if (operandName.equals("AOp"))
    return NVVM::MMAFrag::a;
  if (operandName.equals("BOp"))
    return NVVM::MMAFrag::b;
  if (operandName.equals("COp"))
    return NVVM::MMAFrag::c;
  llvm_unreachable("Unknown operand name");
}

static NVVM::MMATypes getElementType(gpu::MMAMatrixType type) {
  if (type.getElementType().isF16())
    return NVVM::MMATypes::f16;
  if (type.getElementType().isF32())
    return type.getOperand().equals("COp") ? NVVM::MMATypes::f32
                                           : NVVM::MMATypes::tf32;
  llvm_unreachable("Unsupported type");
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
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaLoadMatrixOp.getOperation();
    if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)))
      return failure();

    // Get the shape of the MMAMatrix type being returned. The shape will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType retType =
        subgroupMmaLoadMatrixOp.res().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> retTypeShape = retType.getShape();
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    NVVM::MMATypes eltype = getElementType(retType);
    // NVVM intrinsics require to give mxnxk dimensions, infer the missing
    // dimension based on the valid intrinsics available.
    if (retType.getOperand().equals("AOp")) {
      m = retTypeShape[0];
      k = retTypeShape[1];
      n = NVVM::WMMALoadOp::inferNDimension(m, k, eltype);
    } else if (retType.getOperand().equals("BOp")) {
      k = retTypeShape[0];
      n = retTypeShape[1];
      m = NVVM::WMMALoadOp::inferMDimension(k, n, eltype);
    } else if (retType.getOperand().equals("COp")) {
      m = retTypeShape[0];
      n = retTypeShape[1];
      k = NVVM::WMMALoadOp::inferKDimension(m, n, eltype);
    }
    NVVM::MMALayout layout = NVVM::MMALayout::row;
    NVVM::MMAFrag frag = convertOperand(retType.getOperand());
    // Check that there is an exisiting instruction for the combination we need.
    if (NVVM::WMMALoadOp::getIntrinsicID(m, n, k, layout, eltype, frag) == 0)
      return rewriter.notifyMatchFailure(op, kInvalidCaseStr);

    Type resType = convertMMAToLLVMType(retType);
    Location loc = op->getLoc();

    // Create nvvm.mma_load op according to the operand types.
    Value dataPtr = getStridedElementPtr(
        loc, subgroupMmaLoadMatrixOp.srcMemref().getType().cast<MemRefType>(),
        adaptor.srcMemref(), adaptor.indices(), rewriter);

    Value leadingDim = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        subgroupMmaLoadMatrixOp.leadDimensionAttr());
    rewriter.replaceOpWithNewOp<NVVM::WMMALoadOp>(
        op, resType, dataPtr, leadingDim, m, n, k, layout, eltype, frag);
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
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaStoreMatrixOp.getOperation();
    if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)))
      return failure();

    Location loc = op->getLoc();

    SmallVector<Value, 4> storeOpOperands;
    // Get the shape of the MMAMatrix type being stored. The shape will
    // choose which intrinsic this op will be lowered to.
    gpu::MMAMatrixType srcType =
        subgroupMmaStoreMatrixOp.src().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> srcTypeShape = srcType.getShape();
    NVVM::MMALayout layout = NVVM::MMALayout::row;
    NVVM::MMATypes eltype = getElementType(srcType);
    int64_t m = srcTypeShape[0];
    int64_t n = srcTypeShape[1];
    int64_t k = NVVM::WMMAStoreOp::inferKDimension(m, n, eltype);
    if (NVVM::WMMAStoreOp::getIntrinsicID(m, n, k, layout, eltype) == 0)
      return rewriter.notifyMatchFailure(op, kInvalidCaseStr);

    auto matrixType = adaptor.src().getType().cast<LLVM::LLVMStructType>();
    for (unsigned i = 0, e = matrixType.getBody().size(); i < e; ++i) {
      Value toUse = rewriter.create<LLVM::ExtractValueOp>(
          loc, matrixType.getBody()[i], adaptor.src(),
          rewriter.getI32ArrayAttr(i));
      storeOpOperands.push_back(toUse);
    }

    Value dataPtr = getStridedElementPtr(
        loc, subgroupMmaStoreMatrixOp.dstMemref().getType().cast<MemRefType>(),
        adaptor.dstMemref(), adaptor.indices(), rewriter);
    Value leadingDim = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        subgroupMmaStoreMatrixOp.leadDimensionAttr());
    rewriter.replaceOpWithNewOp<NVVM::WMMAStoreOp>(
        op, dataPtr, m, n, k, layout, eltype, storeOpOperands, leadingDim);
    return success();
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
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *op = subgroupMmaComputeOp.getOperation();
    if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter)))
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
    gpu::MMAMatrixType cType =
        subgroupMmaComputeOp.opC().getType().cast<gpu::MMAMatrixType>();
    ArrayRef<int64_t> cTypeShape = cType.getShape();
    int64_t m = cTypeShape[0];
    int64_t n = cTypeShape[1];
    int64_t k = aTypeShape[1];
    NVVM::MMALayout layout = NVVM::MMALayout::row;
    NVVM::MMATypes sourceType = getElementType(aType);
    NVVM::MMATypes destType = getElementType(cType);
    if (NVVM::WMMAMmaOp::getIntrinsicID(m, n, k, layout, layout, sourceType,
                                        destType) == 0)
      return rewriter.notifyMatchFailure(op, kInvalidCaseStr);

    unpackOp(adaptor.opA());
    unpackOp(adaptor.opB());
    unpackOp(adaptor.opC());

    rewriter.replaceOpWithNewOp<NVVM::WMMAMmaOp>(
        op, adaptor.opC().getType(), m, n, k, layout, layout, sourceType,
        destType, unpackedOps);
    return success();
  }
};

/// Convert GPU MMA ConstantMatrixOp to a chain of InsertValueOp.
struct WmmaConstantOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaConstantMatrixOp> {
  using ConvertOpToLLVMPattern<
      gpu::SubgroupMmaConstantMatrixOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaConstantMatrixOp subgroupMmaConstantOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(areAllLLVMTypes(subgroupMmaConstantOp.getOperation(),
                               adaptor.getOperands(), rewriter)))
      return failure();
    Location loc = subgroupMmaConstantOp.getLoc();
    Value cst = adaptor.getOperands()[0];
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

static Value createMinMaxF(OpBuilder &builder, Location loc, Value lhs,
                           Value rhs, bool isMin) {
  auto floatType = getElementTypeOrSelf(lhs.getType()).cast<FloatType>();
  Type i1Type = builder.getI1Type();
  if (auto vecType = lhs.getType().dyn_cast<VectorType>())
    i1Type = VectorType::get(vecType.getShape(), i1Type);
  Value cmp = builder.create<LLVM::FCmpOp>(
      loc, i1Type, isMin ? LLVM::FCmpPredicate::olt : LLVM::FCmpPredicate::ogt,
      lhs, rhs);
  Value sel = builder.create<LLVM::SelectOp>(loc, cmp, lhs, rhs);
  Value isNan = builder.create<LLVM::FCmpOp>(
      loc, i1Type, LLVM::FCmpPredicate::uno, lhs, rhs);
  Value nan = builder.create<LLVM::ConstantOp>(
      loc, lhs.getType(),
      builder.getFloatAttr(floatType,
                           APFloat::getQNaN(floatType.getFloatSemantics())));
  return builder.create<LLVM::SelectOp>(loc, isNan, sel, nan);
}

static Value createScalarOp(OpBuilder &builder, Location loc,
                            gpu::MMAElementwiseOp op,
                            ArrayRef<Value> operands) {
  switch (op) {
  case gpu::MMAElementwiseOp::ADDF:
    return builder.create<LLVM::FAddOp>(loc, operands[0].getType(), operands);
  case gpu::MMAElementwiseOp::MULF:
    return builder.create<LLVM::FMulOp>(loc, operands[0].getType(), operands);
  case gpu::MMAElementwiseOp::DIVF:
    return builder.create<LLVM::FDivOp>(loc, operands[0].getType(), operands);
  case gpu::MMAElementwiseOp::MAXF:
    return createMinMaxF(builder, loc, operands[0], operands[1],
                         /*isMin=*/false);
  case gpu::MMAElementwiseOp::MINF:
    return createMinMaxF(builder, loc, operands[0], operands[1],
                         /*isMin=*/true);
  }
  llvm_unreachable("unknown op");
}

/// Convert GPU MMA elementwise ops to extract + op + insert.
struct WmmaElementwiseOpToNVVMLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupMmaElementwiseOp> {
  using ConvertOpToLLVMPattern<
      gpu::SubgroupMmaElementwiseOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaElementwiseOp subgroupMmaElementwiseOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(areAllLLVMTypes(subgroupMmaElementwiseOp.getOperation(),
                               adaptor.getOperands(), rewriter)))
      return failure();
    Location loc = subgroupMmaElementwiseOp.getLoc();
    size_t numOperands = adaptor.getOperands().size();
    LLVM::LLVMStructType destType = convertMMAToLLVMType(
        subgroupMmaElementwiseOp.getType().cast<gpu::MMAMatrixType>());
    Value matrixStruct = rewriter.create<LLVM::UndefOp>(loc, destType);
    for (size_t i = 0, e = destType.getBody().size(); i < e; ++i) {
      SmallVector<Value> extractedOperands;
      for (size_t opIdx = 0; opIdx < numOperands; opIdx++) {
        Type elementType = adaptor.getOperands()[opIdx]
                               .getType()
                               .cast<LLVM::LLVMStructType>()
                               .getBody()[i];
        extractedOperands.push_back(rewriter.create<LLVM::ExtractValueOp>(
            loc, elementType, adaptor.getOperands()[opIdx],
            rewriter.getI32ArrayAttr(i)));
      }
      Value element =
          createScalarOp(rewriter, loc, subgroupMmaElementwiseOp.operation(),
                         extractedOperands);
      matrixStruct = rewriter.create<LLVM::InsertValueOp>(
          loc, matrixStruct, element, rewriter.getI32ArrayAttr(i));
    }
    rewriter.replaceOp(subgroupMmaElementwiseOp, matrixStruct);
    return success();
  }
};

} // namespace

/// Return the LLVMStructureType corresponding to the MMAMatrixType `type`.
LLVM::LLVMStructType mlir::convertMMAToLLVMType(gpu::MMAMatrixType type) {
  NVVM::MMAFrag frag = convertOperand(type.getOperand());
  NVVM::MMATypes eltType = getElementType(type);
  std::pair<Type, unsigned> typeInfo =
      NVVM::inferMMAType(eltType, frag, type.getContext());
  return LLVM::LLVMStructType::getLiteral(
      type.getContext(), SmallVector<Type, 8>(typeInfo.second, typeInfo.first));
}

void mlir::populateGpuWMMAToNVVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<WmmaLoadOpToNVVMLowering, WmmaMmaOpToNVVMLowering,
               WmmaStoreOpToNVVMLowering, WmmaConstantOpToNVVMLowering,
               WmmaElementwiseOpToNVVMLowering>(converter);
}
