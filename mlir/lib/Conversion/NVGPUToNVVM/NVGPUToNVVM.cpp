//===- NVGPUToNVVM.cpp - NVGPU to NVVM dialect conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "nvgpu-to-nvvm"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBGSE() (llvm::dbgs())

namespace mlir {
#define GEN_PASS_DEF_CONVERTNVGPUTONVVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// GPU has 32 bit registers, this function truncates values when larger width
/// is not needed.
static Value truncToI32(ConversionPatternRewriter &rewriter, Location loc,
                        Value value) {
  Type type = value.getType();
  assert(llvm::isa<IntegerType>(type) && "expected an integer Value");
  if (type.getIntOrFloatBitWidth() <= 32)
    return value;
  return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), value);
}

/// Returns the type for the intrinsic given the vectorResultType of the
/// `gpu.mma.sync` operation.
static Type inferIntrinsicResultType(Type vectorResultType) {
  MLIRContext *ctx = vectorResultType.getContext();
  auto a = cast<LLVM::LLVMArrayType>(vectorResultType);
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
  auto structType = dyn_cast<LLVM::LLVMStructType>(intrinsicResultType);
  auto arrayType = dyn_cast<LLVM::LLVMArrayType>(resultType);
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
        Value el =
            rewriter.create<LLVM::ExtractValueOp>(loc, intrinsicResult, i);
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
        Value x1 =
            rewriter.create<LLVM::ExtractValueOp>(loc, intrinsicResult, i * 2);
        Value x2 = rewriter.create<LLVM::ExtractValueOp>(loc, intrinsicResult,
                                                         i * 2 + 1);
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
      result = rewriter.create<LLVM::InsertValueOp>(loc, result, el.value(),
                                                    el.index());
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
  Type i4Ty = rewriter.getIntegerType(4);
  Type i8x4Ty = LLVM::getFixedVectorType(i8Ty, 4);
  Type i4x8Ty = LLVM::getFixedVectorType(i4Ty, 8);
  Type f32x1Ty = LLVM::getFixedVectorType(f32Ty, 1);
  auto arrayTy = cast<LLVM::LLVMArrayType>(operand.getType());

  for (unsigned i = 0, e = arrayTy.getNumElements(); i < e; ++i) {
    Value toUse = rewriter.create<LLVM::ExtractValueOp>(loc, operand, i);

    // For 4xi8 vectors, the intrinsic expects these to be provided as i32
    // scalar types.
    if (arrayTy.getElementType() == i8x4Ty ||
        arrayTy.getElementType() == i4x8Ty ||
        (arrayTy.getElementType() == f32x1Ty &&
         operandPtxType == NVVM::MMATypes::tf32)) {
      result.push_back(
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), toUse));
      continue;
    }

    // For some element types (i32, f32, f64), we need to unpack the inner
    // vector/array type as well because the intrinsic expects individual
    // scalars to be provided.
    VectorType innerArrayTy = dyn_cast<VectorType>(arrayTy.getElementType());
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

/// Returns whether mbarrier object has shared memory address space.
static bool isMbarrierShared(nvgpu::MBarrierType barrierType) {
  return (mlir::nvgpu::NVGPUDialect::isSharedMemoryAddressSpace(
      barrierType.getMemorySpace()));
}

/// Returns the memory space attribute of the mbarrier object.
Attribute nvgpu::getMbarrierMemorySpace(MLIRContext *context,
                                        nvgpu::MBarrierType barrierType) {
  Attribute memorySpace = {};
  if (isMbarrierShared(barrierType)) {
    memorySpace =
        IntegerAttr::get(IntegerType::get(context, 64),
                         nvgpu::NVGPUDialect::kSharedMemoryAddressSpace);
  }
  return memorySpace;
}

/// Returns memref type of the mbarrier object. The type is defined in the
/// MBarrierType.
MemRefType nvgpu::getMBarrierMemrefType(MLIRContext *context,
                                        nvgpu::MBarrierType barrierType) {
  Attribute memorySpace = nvgpu::getMbarrierMemorySpace(context, barrierType);
  MemRefLayoutAttrInterface layout;
  return MemRefType::get({1}, IntegerType::get(context, 64), layout,
                         memorySpace);
}

/// Returns the base pointer of the mbarrier object.
static Value getMbarrierPtr(ConversionPatternRewriter &rewriter,
                            const LLVMTypeConverter &typeConverter,
                            TypedValue<nvgpu::MBarrierType> barrier,
                            Value barrierMemref) {
  MemRefType memrefType =
      nvgpu::getMBarrierMemrefType(rewriter.getContext(), barrier.getType());
  MemRefDescriptor memRefDescriptor(barrierMemref);
  return memRefDescriptor.bufferPtr(rewriter, barrier.getLoc(), typeConverter,
                                    memrefType);
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
    auto vectorResultType = dyn_cast<VectorType>(op->getResultTypes()[0]);
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

    auto srcMemrefType = cast<MemRefType>(op.getSrcMemref().getType());
    Value srcPtr =
        getStridedElementPtr(loc, srcMemrefType, adaptor.getSrcMemref(),
                             adaptor.getIndices(), rewriter);
    Value ldMatrixResult = rewriter.create<NVVM::LdMatrixOp>(
        loc, ldMatrixResultType, srcPtr,
        /*num=*/op.getNumTiles(),
        /*layout=*/op.getTranspose() ? NVVM::MMALayout::col
                                     : NVVM::MMALayout::row);

    // The ldmatrix operation returns either a single i32 value or a struct of
    // i32 values. Here we unpack those values and cast them back to their
    // actual vector type (still of width 32b) and repack them into a result
    // struct.
    Type finalResultType = typeConverter->convertType(vectorResultType);
    Value result = rewriter.create<LLVM::UndefOp>(loc, finalResultType);
    for (int64_t i = 0, e = vectorResultType.getDimSize(0); i < e; i++) {
      Value i32Register =
          num32BitRegs > 1
              ? rewriter.create<LLVM::ExtractValueOp>(loc, ldMatrixResult, i)
              : ldMatrixResult;
      Value casted =
          rewriter.create<LLVM::BitcastOp>(loc, innerVectorType, i32Register);
      result = rewriter.create<LLVM::InsertValueOp>(loc, result, casted, i);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert the given type into the corresponding PTX type (NVVM::MMATypes
/// enum).
static FailureOr<NVVM::MMATypes> getNvvmMmaType(Type t) {
  Type elType = getElementTypeOrSelf(t);
  if (elType.isInteger(8))
    return NVVM::MMATypes::s8;
  if (elType.isInteger(4))
    return NVVM::MMATypes::s4;
  if (elType.isF16())
    return NVVM::MMATypes::f16;
  if (elType.isF64())
    return NVVM::MMATypes::f64;
  if (elType.isF32())
    return NVVM::MMATypes::tf32;
  return failure();
}

struct MmaSyncOptoNVVM : public ConvertOpToLLVMPattern<nvgpu::MmaSyncOp> {
  using ConvertOpToLLVMPattern<nvgpu::MmaSyncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::MmaSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // Get the shapes of the MMAMatrix type being used. The shapes will
    // choose which intrinsic this op will be lowered to.
    VectorType aType = op.getMatrixA().getType();
    VectorType bType = op.getMatrixA().getType();
    VectorType cType = op.getMatrixC().getType();

    std::array<int64_t, 3> gemmShape = op.getMmaShapeAsArray();

    // Tensor Cores (mma.sync) on F32 works only with TensorFloat32 (TF32).
    bool tf32Enabled = op->hasAttr(op.getTf32EnabledAttrName());
    if (aType.getElementType().isF32() && !tf32Enabled)
      return failure();

    FailureOr<NVVM::MMATypes> ptxTypeA = getNvvmMmaType(aType);
    if (failed(ptxTypeA))
      return op->emitOpError("failed to deduce operand PTX types");
    FailureOr<NVVM::MMATypes> ptxTypeB = getNvvmMmaType(bType);
    if (failed(ptxTypeB))
      return op->emitOpError("failed to deduce operand PTX types");
    std::optional<NVVM::MMATypes> ptxTypeC =
        NVVM::MmaOp::inferOperandMMAType(cType.getElementType(),
                                         /*isAccumulator=*/true);
    if (!ptxTypeC)
      return op->emitError(
          "could not infer the PTX type for the accumulator/result");

    // TODO: add an attribute to the op to customize this behavior.
    std::optional<NVVM::MMAIntOverflow> overflow(std::nullopt);
    if (isa<IntegerType>(aType.getElementType()))
      overflow = NVVM::MMAIntOverflow::satfinite;

    SmallVector<Value> matA =
        unpackOperandVector(rewriter, loc, adaptor.getMatrixA(), *ptxTypeA);
    SmallVector<Value> matB =
        unpackOperandVector(rewriter, loc, adaptor.getMatrixB(), *ptxTypeB);
    SmallVector<Value> matC =
        unpackOperandVector(rewriter, loc, adaptor.getMatrixC(), *ptxTypeC);

    Type desiredRetTy = typeConverter->convertType(op->getResultTypes()[0]);
    Type intrinsicResTy = inferIntrinsicResultType(
        typeConverter->convertType(op->getResultTypes()[0]));
    Value intrinsicResult = rewriter.create<NVVM::MmaOp>(
        op.getLoc(), intrinsicResTy, matA, matB, matC,
        /*shape=*/gemmShape,
        /*b1Op=*/std::nullopt,
        /*intOverflow=*/overflow,
        /*multiplicandPtxTypes=*/
        std::array<NVVM::MMATypes, 2>{*ptxTypeA, *ptxTypeB},
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
    : public impl::ConvertNVGPUToNVVMPassBase<ConvertNVGPUToNVVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    LowerToLLVMOptions options(&getContext());
    options.useOpaquePointers = useOpaquePointers;
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext(), options);
    IRRewriter rewriter(&getContext());
    /// device-side async tokens cannot be materialized in nvvm. We just
    /// convert them to a dummy i32 type in order to easily drop them during
    /// conversion.
    converter.addConversion([&](nvgpu::DeviceAsyncTokenType type) -> Type {
      return converter.convertType(IntegerType::get(type.getContext(), 32));
    });
    converter.addConversion([&](nvgpu::MBarrierTokenType type) -> Type {
      return converter.convertType(IntegerType::get(type.getContext(), 64));
    });
    converter.addConversion(
        [&](nvgpu::WarpgroupMatrixDescriptorType type) -> Type {
          return converter.convertType(IntegerType::get(type.getContext(), 64));
        });
    converter.addConversion([&](nvgpu::MBarrierType type) -> Type {
      return converter.convertType(
          nvgpu::getMBarrierMemrefType(rewriter.getContext(), type));
    });
    converter.addConversion([&](nvgpu::TensorMapDescriptorType type) -> Type {
      return converter.getPointerType(type.getTensor().getElementType());
    });
    populateNVGPUToNVVMConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<::mlir::memref::MemRefDialect>();
    target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

/// Returns the constraints for the sparse MMA inline assembly instruction.
static std::string buildMmaSparseAsmConstraintString(unsigned matASize,
                                                     unsigned matBSize,
                                                     unsigned matCSize) {
  std::string str;
  llvm::raw_string_ostream ss(str);
  for (unsigned i = 0; i < matCSize; i++)
    ss << "=r,";
  for (unsigned i = 0; i < matASize + matBSize + matCSize; i++)
    ss << "r,";
  // The final operand is for the sparsity metadata.
  // The sparsity selector appears as direct literal.
  ss << "r";
  ss.flush();
  return str;
}

/// Returns the string for the `mma.sp.sync` instruction that corresponds to
/// the given parameters. Note that this function doesn't do any validation,
/// it's expected that the provided parameters correspond to a valid
/// instruction.
static std::string buildMmaSparseAsmString(
    const std::array<int64_t, 3> &shape, unsigned matASize, unsigned matBSize,
    unsigned matCSize, NVVM::MMATypes ptxTypeA, NVVM::MMATypes ptxTypeB,
    NVVM::MMATypes ptxTypeC, NVVM::MMATypes ptxTypeD,
    std::optional<NVVM::MMAIntOverflow> overflow, unsigned metaDataSelector) {
  auto ptxTypeStr = [](NVVM::MMATypes ptxType) {
    return NVVM::stringifyMMATypes(ptxType);
  };

  std::string asmStr;
  llvm::raw_string_ostream ss(asmStr);
  ss << "mma.sp.sync.aligned.m" << shape[0] << "n" << shape[1] << "k"
     << shape[2] << ".row.col.";

  if (overflow)
    ss << NVVM::stringifyMMAIntOverflow(*overflow) << ".";

  ss << ptxTypeStr(ptxTypeD) << "." << ptxTypeStr(ptxTypeA) << "."
     << ptxTypeStr(ptxTypeB) << "." << ptxTypeStr(ptxTypeC) << " ";
  unsigned asmArgIdx = 0;

  // The operand string is structured into sections `{matC elements...},
  // {matA elements...}, {matB elements...}, {matC elements}`.
  for (const auto arrSize : {matCSize, matASize, matBSize, matCSize}) {
    ss << "{";
    for (unsigned i = 0; i < arrSize; i++)
      ss << "$" << asmArgIdx++ << (i < arrSize - 1 ? "," : "");
    ss << "},";
  }
  ss << "$" << asmArgIdx++ << ",";
  assert(metaDataSelector <= 1);
  ss << "0x" << metaDataSelector << ";";
  ss.flush();
  return asmStr;
}

/// Builds an inline assembly operation corresponding to the specified MMA
/// sparse sync operation.
static FailureOr<LLVM::InlineAsmOp> emitMmaSparseSyncOpAsm(
    Location loc, NVVM::MMATypes ptxTypeA, NVVM::MMATypes ptxTypeB,
    NVVM::MMATypes ptxTypeC, NVVM::MMATypes ptxTypeD,
    std::optional<NVVM::MMAIntOverflow> overflow, ArrayRef<Value> unpackedAData,
    ArrayRef<Value> unpackedB, ArrayRef<Value> unpackedC, Value indexData,
    int64_t metadataSelector, const std::array<int64_t, 3> &shape,
    Type intrinsicResultType, ConversionPatternRewriter &rewriter) {
  auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                  LLVM::AsmDialect::AD_ATT);

  const unsigned matASize = unpackedAData.size();
  const unsigned matBSize = unpackedB.size();
  const unsigned matCSize = unpackedC.size();

  std::string asmStr = buildMmaSparseAsmString(
      shape, matASize, matBSize, matCSize, ptxTypeA, ptxTypeB, ptxTypeC,
      ptxTypeD, overflow, metadataSelector);
  std::string constraintStr =
      buildMmaSparseAsmConstraintString(matASize, matBSize, matCSize);

  SmallVector<Value> asmVals;
  asmVals.reserve(matASize + matBSize + matCSize + 1);
  for (ArrayRef<Value> args : {unpackedAData, unpackedB, unpackedC})
    llvm::append_range(asmVals, args);
  asmVals.push_back(indexData);

  return rewriter.create<LLVM::InlineAsmOp>(loc,
                                            /*resultTypes=*/intrinsicResultType,
                                            /*operands=*/asmVals,
                                            /*asm_string=*/asmStr,
                                            /*constraints=*/constraintStr,
                                            /*has_side_effects=*/true,
                                            /*is_align_stack=*/false,
                                            /*asm_dialect=*/asmDialectAttr,
                                            /*operand_attrs=*/ArrayAttr());
}

/// Lowers `nvgpu.mma.sp.sync` to inline assembly.
struct NVGPUMmaSparseSyncLowering
    : public ConvertOpToLLVMPattern<nvgpu::MmaSparseSyncOp> {
  using ConvertOpToLLVMPattern<nvgpu::MmaSparseSyncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::MmaSparseSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // Get the shapes of the MMAMatrix type being used. The shapes will
    // choose which intrinsic this op will be lowered to.
    VectorType aType = op.getMatrixA().getType();
    VectorType bType = op.getMatrixB().getType();
    VectorType cType = op.getMatrixC().getType();

    FailureOr<NVVM::MMATypes> ptxTypeA = getNvvmMmaType(aType);
    if (failed(ptxTypeA))
      return op->emitOpError("failed to deduce operand PTX types");
    FailureOr<NVVM::MMATypes> ptxTypeB = getNvvmMmaType(bType);
    if (failed(ptxTypeB))
      return op->emitOpError("failed to deduce operand PTX types");
    std::optional<NVVM::MMATypes> ptxTypeC =
        NVVM::MmaOp::inferOperandMMAType(cType.getElementType(),
                                         /*isAccumulator=*/true);
    if (!ptxTypeC)
      return op->emitError(
          "could not infer the PTX type for the accumulator/result");

    // Same as `mma.sync`, F32 works only with TensorFloat32 (TF32).
    bool tf32Enabled = op->hasAttr(op.getTf32EnabledAttrName());
    if (aType.getElementType().isF32() && !tf32Enabled)
      return failure();

    // TODO: add an attribute to the op to customize this behavior.
    std::optional<NVVM::MMAIntOverflow> overflow(std::nullopt);
    if (isa<IntegerType>(aType.getElementType()))
      overflow = NVVM::MMAIntOverflow::satfinite;

    SmallVector<Value> matA =
        unpackOperandVector(rewriter, loc, adaptor.getMatrixA(), *ptxTypeA);
    SmallVector<Value> matB =
        unpackOperandVector(rewriter, loc, adaptor.getMatrixB(), *ptxTypeB);
    SmallVector<Value> matC =
        unpackOperandVector(rewriter, loc, adaptor.getMatrixC(), *ptxTypeC);

    Type desiredRetTy = typeConverter->convertType(op->getResultTypes()[0]);
    Type intrinsicResTy = inferIntrinsicResultType(
        typeConverter->convertType(op->getResultTypes()[0]));

    // Bitcast the sparse metadata from vector<2xf16> to an i32.
    Value sparseMetadata = adaptor.getSparseMetadata();
    if (sparseMetadata.getType() !=
        LLVM::getFixedVectorType(rewriter.getI16Type(), 2))
      return op->emitOpError() << "Expected metadata type to be LLVM "
                                  "VectorType of 2 i16 elements";
    sparseMetadata = rewriter.create<LLVM::BitcastOp>(
        loc, rewriter.getI32Type(), sparseMetadata);

    FailureOr<LLVM::InlineAsmOp> intrinsicResult = emitMmaSparseSyncOpAsm(
        loc, *ptxTypeA, *ptxTypeB, *ptxTypeC, *ptxTypeC, overflow, matA, matB,
        matC, sparseMetadata, op.getSparsitySelector(), op.getMmaShapeAsArray(),
        intrinsicResTy, rewriter);
    if (failed(intrinsicResult))
      return failure();

    assert((*intrinsicResult).getNumResults() == 1 &&
           "expected inline asm op returns a single LLVM struct type");
    rewriter.replaceOp(
        op, convertIntrinsicResult(op.getLoc(), intrinsicResTy, desiredRetTy,
                                   (*intrinsicResult)->getResult(0), rewriter));
    return success();
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
    auto dstMemrefType = cast<MemRefType>(op.getDst().getType());
    Value dstPtr = getStridedElementPtr(loc, dstMemrefType, adaptor.getDst(),
                                        adaptor.getDstIndices(), rewriter);
    auto i8Ty = IntegerType::get(op.getContext(), 8);
    FailureOr<unsigned> dstAddressSpace =
        getTypeConverter()->getMemRefAddressSpace(dstMemrefType);
    if (failed(dstAddressSpace))
      return rewriter.notifyMatchFailure(
          loc, "destination memref address space not convertible to integer");
    auto dstPointerType =
        getTypeConverter()->getPointerType(i8Ty, *dstAddressSpace);
    if (!getTypeConverter()->useOpaquePointers())
      dstPtr = rewriter.create<LLVM::BitcastOp>(loc, dstPointerType, dstPtr);

    auto srcMemrefType = cast<MemRefType>(op.getSrc().getType());
    FailureOr<unsigned> srcAddressSpace =
        getTypeConverter()->getMemRefAddressSpace(srcMemrefType);
    if (failed(srcAddressSpace))
      return rewriter.notifyMatchFailure(
          loc, "source memref address space not convertible to integer");

    Value scrPtr = getStridedElementPtr(loc, srcMemrefType, adaptor.getSrc(),
                                        adaptor.getSrcIndices(), rewriter);
    auto srcPointerType =
        getTypeConverter()->getPointerType(i8Ty, *srcAddressSpace);
    if (!getTypeConverter()->useOpaquePointers())
      scrPtr = rewriter.create<LLVM::BitcastOp>(loc, srcPointerType, scrPtr);
    // Intrinsics takes a global pointer so we need an address space cast.
    auto srcPointerGlobalType = getTypeConverter()->getPointerType(
        i8Ty, NVVM::NVVMMemorySpace::kGlobalMemorySpace);
    scrPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, srcPointerGlobalType,
                                                    scrPtr);
    int64_t dstElements = adaptor.getDstElements().getZExtValue();
    int64_t sizeInBytes =
        (dstMemrefType.getElementTypeBitWidth() * dstElements) / 8;
    // When the optional SrcElements argument is *not* present, the regular
    // CpAsyncOp is generated. CopyAsyncOp reads bytes from source (global
    // memory) to fill DstElements number of elements in the destination
    // (shared memory).
    Value srcBytes = adaptor.getSrcElements();
    if (srcBytes) {
      // When the optional SrcElements argument is present, the source (global
      // memory) of CpAsyncOp is read only for SrcElements number of elements.
      // The rest of the DstElements in the destination (shared memory) are
      // filled with zeros.
      Value c3I32 = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(3));
      Value bitwidth = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(srcMemrefType.getElementTypeBitWidth()));
      Value srcElementsI32 =
          rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), srcBytes);
      srcBytes = rewriter.create<LLVM::LShrOp>(
          loc, rewriter.create<LLVM::MulOp>(loc, bitwidth, srcElementsI32),
          c3I32);
    }
    // Cache global (.cg) for 16 dst bytes, Cache all (.ca) for sizes other than
    // 16 dst bytes.
    NVVM::LoadCacheModifierKind cacheModifier =
        (op.getBypassL1().value_or(false) && sizeInBytes == 16)
            ? NVVM::LoadCacheModifierKind::CG
            : NVVM::LoadCacheModifierKind::CA;

    rewriter.create<NVVM::CpAsyncOp>(
        loc, dstPtr, scrPtr, rewriter.getI32IntegerAttr(sizeInBytes),
        NVVM::LoadCacheModifierKindAttr::get(op->getContext(), cacheModifier),
        srcBytes);

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
    int32_t numGroups = adaptor.getNumGroups().value_or(0);
    rewriter.create<NVVM::CpAsyncWaitGroupOp>(op.getLoc(), numGroups);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Creates mbarrier object in shared memory
struct NVGPUMBarrierCreateLowering
    : public ConvertOpToLLVMPattern<nvgpu::MBarrierCreateOp> {
  using ConvertOpToLLVMPattern<nvgpu::MBarrierCreateOp>::ConvertOpToLLVMPattern;

  template <typename moduleT>
  memref::GlobalOp generateGlobalBarrier(ConversionPatternRewriter &rewriter,
                                         Operation *funcOp, moduleT moduleOp,
                                         MemRefType barrierType) const {
    SymbolTable symbolTable(moduleOp);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&moduleOp.front());
    auto global = rewriter.create<memref::GlobalOp>(
        funcOp->getLoc(), "__mbarrier",
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/barrierType,
        /*initial_value=*/ElementsAttr(),
        /*constant=*/false,
        /*alignment=*/rewriter.getI64IntegerAttr(8));
    symbolTable.insert(global);
    return global;
  }

  LogicalResult
  matchAndRewrite(nvgpu::MBarrierCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *funcOp = op->getParentOp();
    MemRefType barrierType = nvgpu::getMBarrierMemrefType(
        rewriter.getContext(), op.getBarrier().getType());

    memref::GlobalOp global;
    if (auto moduleOp = funcOp->getParentOfType<gpu::GPUModuleOp>())
      global = generateGlobalBarrier(rewriter, funcOp, moduleOp, barrierType);
    else if (auto moduleOp = funcOp->getParentOfType<ModuleOp>())
      global = generateGlobalBarrier(rewriter, funcOp, moduleOp, barrierType);

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, barrierType,
                                                     global.getName());
    return success();
  }
};

/// Lowers `nvgpu.mbarrier.init` to `nvvm.mbarrier.init`
struct NVGPUMBarrierInitLowering
    : public ConvertOpToLLVMPattern<nvgpu::MBarrierInitOp> {
  using ConvertOpToLLVMPattern<nvgpu::MBarrierInitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::MBarrierInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    Value barrier = getMbarrierPtr(rewriter, *getTypeConverter(),
                                   op.getBarrier(), adaptor.getBarrier());

    Value count = truncToI32(rewriter, op->getLoc(), adaptor.getCount());

    if (isMbarrierShared(op.getBarrier().getType())) {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierInitSharedOp>(op, barrier,
                                                              count);
    } else {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierInitOp>(op, barrier, count);
    }
    return success();
  }
};

/// Lowers `nvgpu.mbarrier.arrive` to `nvvm.mbarrier.arrive`
struct NVGPUMBarrierArriveLowering
    : public ConvertOpToLLVMPattern<nvgpu::MBarrierArriveOp> {
  using ConvertOpToLLVMPattern<nvgpu::MBarrierArriveOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(nvgpu::MBarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value barrier = getMbarrierPtr(rewriter, *getTypeConverter(),
                                   op.getBarrier(), adaptor.getBarrier());
    Type tokenType = getTypeConverter()->convertType(
        nvgpu::MBarrierTokenType::get(op->getContext()));
    if (isMbarrierShared(op.getBarrier().getType())) {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierArriveSharedOp>(op, tokenType,
                                                                barrier);
    } else {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierArriveOp>(op, tokenType,
                                                          barrier);
    }
    return success();
  }
};

/// Lowers `nvgpu.mbarrier.arrive.nocomplete` to
/// `nvvm.mbarrier.arrive.nocomplete`
struct NVGPUMBarrierArriveNoCompleteLowering
    : public ConvertOpToLLVMPattern<nvgpu::MBarrierArriveNoCompleteOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::MBarrierArriveNoCompleteOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::MBarrierArriveNoCompleteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value barrier = getMbarrierPtr(rewriter, *getTypeConverter(),
                                   op.getBarrier(), adaptor.getBarrier());
    Type tokenType = getTypeConverter()->convertType(
        nvgpu::MBarrierTokenType::get(op->getContext()));
    Value count = truncToI32(rewriter, op->getLoc(), adaptor.getCount());
    if (isMbarrierShared(op.getBarrier().getType())) {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierArriveNocompleteSharedOp>(
          op, tokenType, barrier, count);
    } else {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierArriveNocompleteOp>(
          op, tokenType, barrier, count);
    }
    return success();
  }
};

/// Lowers `nvgpu.mbarrier.test.wait` to `nvvm.mbarrier.test.wait`
struct NVGPUMBarrierTestWaitLowering
    : public ConvertOpToLLVMPattern<nvgpu::MBarrierTestWaitOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::MBarrierTestWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::MBarrierTestWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value barrier = getMbarrierPtr(rewriter, *getTypeConverter(),
                                   op.getBarrier(), adaptor.getBarrier());
    Type retType = rewriter.getI1Type();
    if (isMbarrierShared(op.getBarrier().getType())) {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierTestWaitSharedOp>(
          op, retType, barrier, adaptor.getToken());
    } else {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierTestWaitOp>(
          op, retType, barrier, adaptor.getToken());
    }
    return success();
  }
};

struct NVGPUMBarrierArriveExpectTxLowering
    : public ConvertOpToLLVMPattern<nvgpu::MBarrierArriveExpectTxOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::MBarrierArriveExpectTxOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::MBarrierArriveExpectTxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value barrier = getMbarrierPtr(rewriter, *getTypeConverter(),
                                   op.getBarrier(), adaptor.getBarrier());
    Value txcount = truncToI32(rewriter, op->getLoc(), adaptor.getTxcount());

    if (isMbarrierShared(op.getBarrier().getType())) {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierArriveExpectTxSharedOp>(
          op, barrier, txcount);
      return success();
    }

    rewriter.replaceOpWithNewOp<NVVM::MBarrierArriveExpectTxOp>(op, barrier,
                                                                txcount);
    return success();
  }
};

struct NVGPUMBarrierTryWaitParityLowering
    : public ConvertOpToLLVMPattern<nvgpu::MBarrierTryWaitParityOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::MBarrierTryWaitParityOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::MBarrierTryWaitParityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value barrier = getMbarrierPtr(rewriter, *getTypeConverter(),
                                   op.getBarrier(), adaptor.getBarrier());
    Value ticks = truncToI32(rewriter, op->getLoc(), adaptor.getTicks());
    Value phase = truncToI32(rewriter, op->getLoc(), adaptor.getPhase());

    if (isMbarrierShared(op.getBarrier().getType())) {
      rewriter.replaceOpWithNewOp<NVVM::MBarrierTryWaitParitySharedOp>(
          op, barrier, phase, ticks);
      return success();
    }

    rewriter.replaceOpWithNewOp<NVVM::MBarrierTryWaitParityOp>(op, barrier,
                                                               phase, ticks);
    return success();
  }
};

struct NVGPUTmaAsyncLoadOpLowering
    : public ConvertOpToLLVMPattern<nvgpu::TmaAsyncLoadOp> {
  using ConvertOpToLLVMPattern<nvgpu::TmaAsyncLoadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(nvgpu::TmaAsyncLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcMemrefType = cast<MemRefType>(op.getDst().getType());
    Value dest = getStridedElementPtr(op->getLoc(), srcMemrefType,
                                      adaptor.getDst(), {}, rewriter);
    Value barrier = getMbarrierPtr(rewriter, *getTypeConverter(),
                                   op.getBarrier(), adaptor.getBarrier());

    SmallVector<Value> coords = adaptor.getCoordinates();
    for (auto [index, value] : llvm::enumerate(coords)) {
      coords[index] = truncToI32(rewriter, op->getLoc(), value);
    }

    rewriter.replaceOpWithNewOp<NVVM::CpAsyncBulkTensorGlobalToSharedClusterOp>(
        op, dest, adaptor.getTensorMapDescriptor(), barrier, coords);
    return success();
  }
};
struct NVGPUGenerateGmmaDescriptorLowering
    : public ConvertOpToLLVMPattern<nvgpu::GenerateGmmaDescriptorOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::GenerateGmmaDescriptorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::GenerateGmmaDescriptorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();

    nvgpu::TensorMapSwizzleKind swizzleKind =
        op.getTensorMap().getType().getSwizzle();

    unsigned layout =
        (swizzleKind == nvgpu::TensorMapSwizzleKind::SWIZZLE_128B)  ? 128
        : (swizzleKind == nvgpu::TensorMapSwizzleKind::SWIZZLE_64B) ? 64
        : (swizzleKind == nvgpu::TensorMapSwizzleKind::SWIZZLE_32B) ? 32
                                                                    : 1;
    unsigned swizzle =
        (swizzleKind == nvgpu::TensorMapSwizzleKind::SWIZZLE_128B)  ? 1
        : (swizzleKind == nvgpu::TensorMapSwizzleKind::SWIZZLE_64B) ? 2
        : (swizzleKind == nvgpu::TensorMapSwizzleKind::SWIZZLE_32B) ? 3
                                                                    : 0;

    auto ti64 = rewriter.getIntegerType(64);
    auto makeConst = [&](uint64_t index) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, ti64, rewriter.getI64IntegerAttr(index));
    };
    auto shiftLeft = [&](Value value, unsigned shift) -> Value {
      return rewriter.create<LLVM::ShlOp>(loc, ti64, value, makeConst(shift));
    };
    auto shiftRight = [&](Value value, unsigned shift) -> Value {
      return rewriter.create<LLVM::LShrOp>(loc, ti64, value, makeConst(shift));
    };
    auto insertBit = [&](Value desc, Value val, int startBit) {
      return rewriter.create<LLVM::OrOp>(loc, ti64, desc,
                                         shiftLeft(val, startBit));
    };

    int ex4LSB = 4;
    int64_t sizeN = op.getTensorMap().getType().getTensor().getDimSize(0);
    uint64_t strideDimVal = (layout << 3) >> ex4LSB;
    uint64_t leadDimVal = (sizeN * layout) >> ex4LSB;
    uint64_t offsetVal = 0;

    Value strideDim = makeConst(strideDimVal);
    Value leadDim = makeConst(leadDimVal);

    Value baseAddr = getStridedElementPtr(
        op->getLoc(), cast<MemRefType>(op.getTensor().getType()),
        adaptor.getTensor(), {}, rewriter);
    Value basePtr = rewriter.create<LLVM::PtrToIntOp>(loc, ti64, baseAddr);
    // Just use 14 bits for base address
    Value basePtr14bit = shiftRight(shiftLeft(basePtr, 46), 50);

    int startSwizzleBit = 62, startOffsetBit = 49, startStrideBit = 32,
        startLeadBit = 16, startBaseAddrBit = 0;
    Value dsc = makeConst(0);
    // // [62,64)  swizzle type
    dsc = insertBit(dsc, makeConst(swizzle), startSwizzleBit);
    // // [49,52)  base_offset
    dsc = insertBit(dsc, makeConst(offsetVal), startOffsetBit);
    // // [32,46)  stride
    dsc = insertBit(dsc, strideDim, startStrideBit);
    // // [16,30)  leading dimension
    dsc = insertBit(dsc, leadDim, startLeadBit);
    // // [0,14)   start_address
    dsc = insertBit(dsc, basePtr14bit, startBaseAddrBit);

    LLVM_DEBUG(DBGS() << "Generating wgmma.descriptor: "
                      << "leading_off:" << leadDimVal << "\t"
                      << "stride_off :" << strideDimVal << "\t"
                      << "base_offset:" << offsetVal << "\t"
                      << "layout_type:" << swizzle << " ("
                      << nvgpu::stringifyTensorMapSwizzleKind(swizzleKind)
                      << ")\n start_addr :  " << baseAddr << "\n");

    rewriter.replaceOp(op, dsc);
    return success();
  }
};

static Value makeI64Const(RewriterBase &rewriter, Operation *op,
                          int32_t index) {
  return rewriter.create<LLVM::ConstantOp>(op->getLoc(),
                                           rewriter.getIntegerType(64),
                                           rewriter.getI32IntegerAttr(index));
}

/// Returns a Value that holds data type enum that is expected by CUDA driver.
static Value elementTypeAsLLVMConstant(RewriterBase &rewriter, Operation *op,
                                       Type type) {
  // Enum is from CUDA driver API
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
  enum CUtensorMapDataTypeEnum {
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16,
    CU_TENSOR_MAP_DATA_TYPE_UINT32,
    CU_TENSOR_MAP_DATA_TYPE_INT32,
    CU_TENSOR_MAP_DATA_TYPE_UINT64,
    CU_TENSOR_MAP_DATA_TYPE_INT64,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ
  };

  if (type.isUnsignedInteger(8))
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_UINT8);
  if (type.isUnsignedInteger(16))
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_UINT16);
  if (type.isUnsignedInteger(32))
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_UINT32);
  if (type.isUnsignedInteger(64))
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_UINT64);
  if (type.isSignlessInteger(32))
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_INT32);
  if (type.isSignlessInteger(64))
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_INT64);
  if (type.isF16())
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
  if (type.isF32())
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_FLOAT32);
  if (type.isF64())
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_FLOAT64);
  if (type.isBF16())
    return makeI64Const(rewriter, op, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);

  llvm_unreachable("Not supported data type");
}

struct NVGPUTmaCreateDescriptorOpLowering
    : public ConvertOpToLLVMPattern<nvgpu::TmaCreateDescriptorOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::TmaCreateDescriptorOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(nvgpu::TmaCreateDescriptorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    LLVM::LLVMPointerType llvmPointerType = getTypeConverter()->getPointerType(
        IntegerType::get(op->getContext(), 8));
    Type llvmInt64Type = IntegerType::get(op->getContext(), 64);

    Value tensorElementType = elementTypeAsLLVMConstant(
        rewriter, op, op.getTensor().getType().getElementType());
    auto promotedOperands = getTypeConverter()->promoteOperands(
        loc, op->getOperands(), adaptor.getOperands(), rewriter);

    Value boxArrayPtr = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPointerType, llvmInt64Type, makeI64Const(rewriter, op, 5));
    for (auto [index, value] : llvm::enumerate(adaptor.getBoxDimensions())) {
      Value gep = rewriter.create<LLVM::GEPOp>(
          loc, llvmPointerType, llvmPointerType, boxArrayPtr,
          makeI64Const(rewriter, op, index));
      rewriter.create<LLVM::StoreOp>(loc, value, gep);
    }

    nvgpu::TensorMapDescriptorType desc = op.getTensorMap().getType();
    // Set Arguments for the function call
    SmallVector<Value> arguments;
    arguments.push_back(promotedOperands[0]); // rank
    arguments.push_back(promotedOperands[1]); // descriptor
    arguments.push_back(tensorElementType);   // data type
    arguments.push_back(
        makeI64Const(rewriter, op, (int)desc.getInterleave())); // interleave
    arguments.push_back(
        makeI64Const(rewriter, op, (int)desc.getSwizzle())); // swizzle
    arguments.push_back(
        makeI64Const(rewriter, op, (int)desc.getL2promo())); // l2promo
    arguments.push_back(makeI64Const(rewriter, op, (int)desc.getOob())); // oob
    arguments.push_back(boxArrayPtr); // box dimensions

    // Set data types of the arguments
    SmallVector<Type> argTypes = {
        llvmInt64Type,   /* int64_t tensorRank */
        llvmPointerType, /* ptr */
        llvmInt64Type,   /* int64_t */
        llvmInt64Type,   /* int64_t */
        llvmInt64Type,   /* int64_t */
        llvmInt64Type,   /* int64_t */
        llvmInt64Type,   /* int64_t */
        llvmPointerType  /* ptr  */
    };
    FunctionCallBuilder hostRegisterCallBuilder = {
        "mgpuTensorMapEncodeTiledMemref", llvmPointerType, argTypes};
    Value tensorMap =
        hostRegisterCallBuilder.create(loc, rewriter, arguments).getResult();

    rewriter.replaceOp(op, tensorMap);
    return success();
  }
};

} // namespace

void mlir::populateNVGPUToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns) {
  patterns.add<
      NVGPUMBarrierCreateLowering,           // nvgpu.mbarrier.create
      NVGPUMBarrierInitLowering,             // nvgpu.mbarrier.init
      NVGPUMBarrierArriveLowering,           // nvgpu.mbarrier.arrive
      NVGPUMBarrierArriveNoCompleteLowering, // nvgpu.mbarrier.arrive.no_complete
      NVGPUMBarrierTestWaitLowering,         // nvgpu.mbarrier.test_wait_parity
      NVGPUMBarrierTryWaitParityLowering,    // nvgpu.mbarrier.try_wait_parity
      NVGPUTmaAsyncLoadOpLowering,           // nvgpu.tma.async.load
      NVGPUTmaCreateDescriptorOpLowering,    // nvgpu.tma.create.descriptor
      NVGPUMBarrierArriveExpectTxLowering,   // nvgpu.mbarrier.arrive.expect_tx
      NVGPUGenerateGmmaDescriptorLowering,   // nvgpu.wgmma.generate.descriptor
      MmaSyncOptoNVVM, MmaLdMatrixOpToNVVM, NVGPUAsyncCopyLowering,
      NVGPUAsyncCreateGroupLowering, NVGPUAsyncWaitLowering,
      NVGPUMmaSparseSyncLowering>(converter);
}
