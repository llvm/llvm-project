//===- NVGPUToNVVM.cpp - NVGPU to NVVM dialect conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/NVGPU/NVGPUDialect.h"

using namespace mlir;

/// Returns the type for the intrinsic given the vectorResultType of the
/// `gpu.mma.sync` operation.
static Type inferIntrinsicResultType(Type vectorResultType) {
  MLIRContext *ctx = vectorResultType.getContext();
  auto a = vectorResultType.cast<LLVM::LLVMArrayType>();
  auto f16x2Ty = LLVM::getFixedVectorType(Float16Type::get(ctx), 2);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i32x2Ty = LLVM::getFixedVectorType(i32Ty, 2);
  Type f64Ty = Float64Type::get(ctx);
  Type f64x2Ty = LLVM::getFixedVectorType(f64Ty, 2);
  Type f32Ty = Float32Type::get(ctx);
  Type f32x2Ty = LLVM::getFixedVectorType(f32Ty, 2);
  if (a.getElementType() == f16x2Ty) {
    return LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(a.getNumElements(), f16x2Ty));
  }
  if (a.getElementType() == i32x2Ty) {
    return LLVM::LLVMStructType::getLiteral(
        ctx,
        SmallVector<Type>(static_cast<size_t>(a.getNumElements()) * 2, i32Ty));
  }
  if (a.getElementType() == f64x2Ty) {
    return LLVM::LLVMStructType::getLiteral(ctx, {f64Ty, f64Ty});
  }
  if (a.getElementType() == f32x2Ty) {
    return LLVM::LLVMStructType::getLiteral(
        ctx,
        SmallVector<Type>(static_cast<size_t>(a.getNumElements()) * 2, f32Ty));
  }
  if (a.getElementType() == LLVM::getFixedVectorType(f32Ty, 1)) {
    return LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(static_cast<size_t>(a.getNumElements()), f32Ty));
  }
  return vectorResultType;
}

/// Convert the SSA result of the NVVM intrinsic `nvvm.mma.sync` (which is
/// always an LLVM struct) into a fragment that is compatible with the vector
/// type of this operation. This involves extracting elements from the struct
/// and inserting them into an LLVM array. These extra data-movement
/// operations should be canonicalized away by the LLVM backend.
static Value convertIntrinsicResult(Location loc, Type intrinsicResultType,
                                    Type resultType, Value intrinsicResult,
                                    RewriterBase &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  auto structType = intrinsicResultType.dyn_cast<LLVM::LLVMStructType>();
  auto arrayType = resultType.dyn_cast<LLVM::LLVMArrayType>();
  Type i32Ty = rewriter.getI32Type();
  Type f32Ty = rewriter.getF32Type();
  Type f64Ty = rewriter.getF64Type();
  Type f16x2Ty = LLVM::getFixedVectorType(rewriter.getF16Type(), 2);
  Type i32x2Ty = LLVM::getFixedVectorType(i32Ty, 2);
  Type f64x2Ty = LLVM::getFixedVectorType(f64Ty, 2);
  Type f32x2Ty = LLVM::getFixedVectorType(f32Ty, 2);
  Type f32x1Ty = LLVM::getFixedVectorType(f32Ty, 1);

  auto makeConst = [&](int32_t index) -> Value {
    return rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 32),
                                             rewriter.getI32IntegerAttr(index));
  };

  if (arrayType) {
    SmallVector<Value, 4> elements;

    // The intrinsic returns 32-bit wide elements in a form which can be
    // directly bitcasted and inserted into the result vector.
    if (arrayType.getElementType() == f16x2Ty ||
        arrayType.getElementType() == f32x1Ty) {
      for (unsigned i = 0; i < structType.getBody().size(); i++) {
        Value el = rewriter.create<LLVM::ExtractValueOp>(
            loc, structType.getBody()[i], intrinsicResult,
            rewriter.getI64ArrayAttr(i));
        el = rewriter.createOrFold<LLVM::BitcastOp>(
            loc, arrayType.getElementType(), el);
        elements.push_back(el);
      }
    }

    // The intrinsic returns i32, f64, and f32 values as individual scalars,
    // even when the result is notionally a 64-bit wide element (e.g. f32x2). We
    // need to extract them from the struct and pack them into the 64-bit wide
    // rows of the vector result.
    if (arrayType.getElementType() == i32x2Ty ||
        arrayType.getElementType() == f64x2Ty ||
        arrayType.getElementType() == f32x2Ty) {

      for (unsigned i = 0, e = structType.getBody().size() / 2; i < e; i++) {
        Value vec =
            rewriter.create<LLVM::UndefOp>(loc, arrayType.getElementType());
        Value x1 = rewriter.create<LLVM::ExtractValueOp>(
            loc, structType.getBody()[i * 2], intrinsicResult,
            rewriter.getI64ArrayAttr(i * 2));
        Value x2 = rewriter.create<LLVM::ExtractValueOp>(
            loc, structType.getBody()[i * 2 + 1], intrinsicResult,
            rewriter.getI64ArrayAttr(i * 2 + 1));
        vec = rewriter.create<LLVM::InsertElementOp>(loc, vec.getType(), vec,
                                                     x1, makeConst(0));
        vec = rewriter.create<LLVM::InsertElementOp>(loc, vec.getType(), vec,
                                                     x2, makeConst(1));
        elements.push_back(vec);
      }
    }

    // Create the final vectorized result.
    Value result = rewriter.create<LLVM::UndefOp>(loc, arrayType);
    for (const auto &el : llvm::enumerate(elements)) {
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, arrayType, result, el.value(),
          rewriter.getI64ArrayAttr(el.index()));
    }
    return result;
  }

  return intrinsicResult;
}

/// The `gpu.mma.sync` converter below expects matrix fragment operands to be
/// given as 2D `vectors` where the rows are 32b or 64b wide. The
/// `nvvm.mma.sync` op expects these argments to be a given in a long list of
/// scalars of certain types. This function helps unpack the `vector` arguments
/// and cast them to the types expected by `nvvm.mma.sync`.
static SmallVector<Value> unpackOperandVector(RewriterBase &rewriter,
                                              Location loc, Value operand,
                                              NVVM::MMATypes operandPtxType) {
  SmallVector<Value> result;
  Type i32Ty = rewriter.getI32Type();
  Type f64Ty = rewriter.getF64Type();
  Type f32Ty = rewriter.getF32Type();
  Type i8Ty = rewriter.getI8Type();
  Type i8x4Ty = LLVM::getFixedVectorType(i8Ty, 4);
  Type f32x1Ty = LLVM::getFixedVectorType(f32Ty, 1);
  auto arrayTy = operand.getType().cast<LLVM::LLVMArrayType>();

  for (unsigned i = 0, e = arrayTy.getNumElements(); i < e; ++i) {
    Value toUse = rewriter.create<LLVM::ExtractValueOp>(
        loc, arrayTy.getElementType(), operand, rewriter.getI64ArrayAttr(i));

    // For 4xi8 vectors, the intrinsic expects these to be provided as i32
    // scalar types.
    if (arrayTy.getElementType() == i8x4Ty ||
        (arrayTy.getElementType() == f32x1Ty &&
         operandPtxType == NVVM::MMATypes::tf32)) {
      result.push_back(
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), toUse));
      continue;
    }

    // For some element types (i32, f32, f64), we need to unpack the inner
    // vector/array type as well because the intrinsic expects individual
    // scalars to be provided.
    VectorType innerArrayTy = arrayTy.getElementType().dyn_cast<VectorType>();
    if (innerArrayTy && (innerArrayTy.getElementType() == i32Ty ||
                         innerArrayTy.getElementType() == f64Ty ||
                         innerArrayTy.getElementType() == f32Ty)) {
      for (unsigned idx = 0, innerSize = innerArrayTy.getNumElements();
           idx < innerSize; idx++) {
        result.push_back(rewriter.create<LLVM::ExtractElementOp>(
            loc, toUse,
            rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(idx))));
      }
      continue;
    }
    result.push_back(toUse);
  }
  return result;
}

namespace {

struct MmaLdMatrixOpToNVVM : public ConvertOpToLLVMPattern<nvgpu::LdMatrixOp> {
  using ConvertOpToLLVMPattern<nvgpu::LdMatrixOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::LdMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = getContext();
    Location loc = op->getLoc();

    // The result type of ldmatrix will always be a struct of 32bit integer
    // registers if more than one 32bit value is returned. Otherwise, the result
    // is a single i32. The result type of the GPU operation is always a vector
    // of shape (NumRegisters, VectorRegister) where VectorRegister is the
    // vector type of the result and always 32 bits long. We bitcast the result
    // of the NVVM::LdMatrix to this vector type.
    auto vectorResultType = op->getResultTypes()[0].dyn_cast<VectorType>();
    if (!vectorResultType) {
      return failure();
    }
    Type innerVectorType = LLVM::getFixedVectorType(
        vectorResultType.getElementType(), vectorResultType.getDimSize(1));

    int64_t num32BitRegs = vectorResultType.getDimSize(0);

    Type ldMatrixResultType;
    if (num32BitRegs > 1) {
      ldMatrixResultType = LLVM::LLVMStructType::getLiteral(
          ctx, SmallVector<Type>(num32BitRegs, rewriter.getI32Type()));
    } else {
      ldMatrixResultType = rewriter.getI32Type();
    }

    auto srcMemrefType = op.srcMemref().getType().cast<MemRefType>();
    Value srcPtr = getStridedElementPtr(loc, srcMemrefType, adaptor.srcMemref(),
                                        adaptor.indices(), rewriter);
    Value ldMatrixResult = rewriter.create<NVVM::LdMatrixOp>(
        loc, ldMatrixResultType, srcPtr,
        /*num=*/op.numTiles(),
        /*layout=*/op.transpose() ? NVVM::MMALayout::col
                                  : NVVM::MMALayout::row);

    // The ldmatrix operation returns either a single i32 value or a struct of
    // i32 values. Here we unpack those values and cast them back to their
    // actual vector type (still of width 32b) and repack them into a result
    // struct.
    Type finalResultType = typeConverter->convertType(vectorResultType);
    Value result = rewriter.create<LLVM::UndefOp>(loc, finalResultType);
    for (int64_t i = 0, e = vectorResultType.getDimSize(0); i < e; i++) {
      Value i32Register = num32BitRegs > 1
                              ? rewriter.create<LLVM::ExtractValueOp>(
                                    loc, rewriter.getI32Type(), ldMatrixResult,
                                    rewriter.getI64ArrayAttr(i))
                              : ldMatrixResult;
      Value casted =
          rewriter.create<LLVM::BitcastOp>(loc, innerVectorType, i32Register);
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, finalResultType, result, casted, rewriter.getI64ArrayAttr(i));
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct MmaSyncOptoNVVM : public ConvertOpToLLVMPattern<nvgpu::MmaSyncOp> {
  using ConvertOpToLLVMPattern<nvgpu::MmaSyncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::MmaSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // Get the shapes of the MMAMatrix type being used. The shapes will
    // choose which intrinsic this op will be lowered to.
    auto aType = op.matrixA().getType().cast<VectorType>();
    auto cType = op.matrixC().getType().cast<VectorType>();

    int64_t m = op.mmaShape()[0].cast<IntegerAttr>().getInt();
    int64_t n = op.mmaShape()[1].cast<IntegerAttr>().getInt();
    int64_t k = op.mmaShape()[2].cast<IntegerAttr>().getInt();
    std::array<int64_t, 3> gemmShape{m, n, k};

    NVVM::MMATypes ptxTypeA;
    NVVM::MMATypes ptxTypeB;
    Optional<NVVM::MMATypes> ptxTypeC = NVVM::MmaOp::inferOperandMMAType(
        cType.getElementType(), /*isAccumulator=*/true);
    if (!ptxTypeC) {
      return op->emitError(
          "could not infer the PTX type for the accumulator/result");
    }

    Optional<NVVM::MMAIntOverflow> overflow(llvm::None);
    if (aType.getElementType().isInteger(8)) {
      ptxTypeA = NVVM::MMATypes::s8;
      ptxTypeB = NVVM::MMATypes::s8;
      overflow = NVVM::MMAIntOverflow::satfinite;
    } else if (aType.getElementType().isF16()) {
      ptxTypeA = NVVM::MMATypes::f16;
      ptxTypeB = NVVM::MMATypes::f16;
    } else if (aType.getElementType().isF64()) {
      ptxTypeA = NVVM::MMATypes::f64;
      ptxTypeB = NVVM::MMATypes::f64;
    } else if (aType.getElementType().isF32()) {
      ptxTypeA = NVVM::MMATypes::tf32;
      ptxTypeB = NVVM::MMATypes::tf32;
    } else {
      return op->emitError("could not deduce operand PTX types");
    }

    SmallVector<Value> matA =
        unpackOperandVector(rewriter, loc, adaptor.matrixA(), ptxTypeA);
    SmallVector<Value> matB =
        unpackOperandVector(rewriter, loc, adaptor.matrixB(), ptxTypeB);
    SmallVector<Value> matC =
        unpackOperandVector(rewriter, loc, adaptor.matrixC(), *ptxTypeC);

    Type desiredRetTy = typeConverter->convertType(op->getResultTypes()[0]);
    Type intrinsicResTy = inferIntrinsicResultType(
        typeConverter->convertType(op->getResultTypes()[0]));
    Value intrinsicResult = rewriter.create<NVVM::MmaOp>(
        op.getLoc(), intrinsicResTy, matA, matB, matC,
        /*shape=*/gemmShape,
        /*b1Op=*/llvm::None,
        /*intOverflow=*/overflow,
        /*multiplicandPtxTypes=*/
        std::array<NVVM::MMATypes, 2>{ptxTypeA, ptxTypeB},
        /*multiplicandLayouts=*/
        std::array<NVVM::MMALayout, 2>{NVVM::MMALayout::row,
                                       NVVM::MMALayout::col});
    rewriter.replaceOp(op, convertIntrinsicResult(op.getLoc(), intrinsicResTy,
                                                  desiredRetTy, intrinsicResult,
                                                  rewriter));
    return success();
  }
};

struct ConvertNVGPUToNVVMPass
    : public ConvertNVGPUToNVVMBase<ConvertNVGPUToNVVMPass> {
  ConvertNVGPUToNVVMPass() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());
    /// device-side async tokens cannot be materialized in nvvm. We just convert
    /// them to a dummy i32 type in order to easily drop them during conversion.
    converter.addConversion([&](nvgpu::DeviceAsyncTokenType type) -> Type {
      return converter.convertType(IntegerType::get(type.getContext(), 32));
    });
    populateNVGPUToNVVMConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

struct NVGPUAsyncCopyLowering
    : public ConvertOpToLLVMPattern<nvgpu::DeviceAsyncCopyOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::DeviceAsyncCopyOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::DeviceAsyncCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto dstMemrefType = op.dst().getType().cast<MemRefType>();
    Value dstPtr = getStridedElementPtr(loc, dstMemrefType, adaptor.dst(),
                                        adaptor.dstIndices(), rewriter);
    auto i8Ty = IntegerType::get(op.getContext(), 8);
    auto dstPointerType =
        LLVM::LLVMPointerType::get(i8Ty, dstMemrefType.getMemorySpaceAsInt());
    dstPtr = rewriter.create<LLVM::BitcastOp>(loc, dstPointerType, dstPtr);

    auto srcMemrefType = op.src().getType().cast<MemRefType>();

    Value scrPtr = getStridedElementPtr(loc, srcMemrefType, adaptor.src(),
                                        adaptor.srcIndices(), rewriter);
    auto srcPointerType =
        LLVM::LLVMPointerType::get(i8Ty, srcMemrefType.getMemorySpaceAsInt());
    scrPtr = rewriter.create<LLVM::BitcastOp>(loc, srcPointerType, scrPtr);
    // Intrinsics takes a global pointer so we need an address space cast.
    auto srcPointerGlobalType = LLVM::LLVMPointerType::get(
        i8Ty, NVVM::NVVMMemorySpace::kGlobalMemorySpace);
    scrPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, srcPointerGlobalType,
                                                    scrPtr);
    int64_t numElements = adaptor.numElements().getZExtValue();
    int64_t sizeInBytes =
        (dstMemrefType.getElementTypeBitWidth() / 8) * numElements;
    // bypass L1 is only supported for byte sizes of 16, we drop the hint
    // otherwise.
    UnitAttr bypassL1 = sizeInBytes == 16 ? adaptor.bypassL1Attr() : UnitAttr();
    rewriter.create<NVVM::CpAsyncOp>(
        loc, dstPtr, scrPtr, rewriter.getI32IntegerAttr(sizeInBytes), bypassL1);

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct NVGPUAsyncCreateGroupLowering
    : public ConvertOpToLLVMPattern<nvgpu::DeviceAsyncCreateGroupOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::DeviceAsyncCreateGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::DeviceAsyncCreateGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.create<NVVM::CpAsyncCommitGroupOp>(op.getLoc());
    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct NVGPUAsyncWaitLowering
    : public ConvertOpToLLVMPattern<nvgpu::DeviceAsyncWaitOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::DeviceAsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::DeviceAsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If numGroup is not present pick 0 as a conservative correct value.
    int32_t numGroups = adaptor.numGroups() ? *adaptor.numGroups() : 0;
    rewriter.create<NVVM::CpAsyncWaitGroupOp>(op.getLoc(), numGroups);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::populateNVGPUToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns) {
  patterns.add<MmaSyncOptoNVVM, MmaLdMatrixOpToNVVM, NVGPUAsyncCopyLowering,
               NVGPUAsyncCreateGroupLowering, NVGPUAsyncWaitLowering>(
      converter);
}

std::unique_ptr<Pass> mlir::createConvertNVGPUToNVVMPass() {
  return std::make_unique<ConvertNVGPUToNVVMPass>();
}
