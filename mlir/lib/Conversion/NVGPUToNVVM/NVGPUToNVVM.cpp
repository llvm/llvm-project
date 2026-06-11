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
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "nvgpu-to-nvvm"

namespace mlir {
#define GEN_PASS_DEF_CONVERTNVGPUTONVVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// Number of bits that needs to be excluded when building matrix descriptor for
/// wgmma operations.
constexpr int exclude4LSB = 4;

/// GPU has 32 bit registers, this function truncates values when larger width
/// is not needed.
static Value truncToI32(ImplicitLocOpBuilder &b, Value value) {
  Type type = value.getType();
  assert(llvm::isa<IntegerType>(type) && "expected an integer Value");
  if (type.getIntOrFloatBitWidth() <= 32)
    return value;
  return LLVM::TruncOp::create(b, b.getI32Type(), value);
}

/// Returns the type for the intrinsic given the vectorResultType of the
/// `gpu.mma.sync` operation.
static Type inferIntrinsicResultType(Type vectorResultType) {
  MLIRContext *ctx = vectorResultType.getContext();
  auto a = cast<LLVM::LLVMArrayType>(vectorResultType);
  auto f16x2Ty = VectorType::get(2, Float16Type::get(ctx));
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i32x2Ty = VectorType::get(2, i32Ty);
  Type f64Ty = Float64Type::get(ctx);
  Type f64x2Ty = VectorType::get(2, f64Ty);
  Type f32Ty = Float32Type::get(ctx);
  Type f32x2Ty = VectorType::get(2, f32Ty);
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
  if (a.getElementType() == VectorType::get(1, f32Ty)) {
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
  Type f16x2Ty = VectorType::get(2, rewriter.getF16Type());
  Type i32x2Ty = VectorType::get(2, i32Ty);
  Type f64x2Ty = VectorType::get(2, f64Ty);
  Type f32x2Ty = VectorType::get(2, f32Ty);
  Type f32x1Ty = VectorType::get(1, f32Ty);

  auto makeConst = [&](int32_t index) -> Value {
    return LLVM::ConstantOp::create(rewriter, loc, IntegerType::get(ctx, 32),
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
            LLVM::ExtractValueOp::create(rewriter, loc, intrinsicResult, i);
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
            LLVM::PoisonOp::create(rewriter, loc, arrayType.getElementType());
        Value x1 =
            LLVM::ExtractValueOp::create(rewriter, loc, intrinsicResult, i * 2);
        Value x2 = LLVM::ExtractValueOp::create(rewriter, loc, intrinsicResult,
                                                i * 2 + 1);
        vec = LLVM::InsertElementOp::create(rewriter, loc, vec.getType(), vec,
                                            x1, makeConst(0));
        vec = LLVM::InsertElementOp::create(rewriter, loc, vec.getType(), vec,
                                            x2, makeConst(1));
        elements.push_back(vec);
      }
    }

    // Create the final vectorized result.
    Value result = LLVM::PoisonOp::create(rewriter, loc, arrayType);
    for (const auto &el : llvm::enumerate(elements)) {
      result = LLVM::InsertValueOp::create(rewriter, loc, result, el.value(),
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
static SmallVector<Value> unpackOperandVector(ImplicitLocOpBuilder &b,
                                              Value operand,
                                              NVVM::MMATypes operandPtxType) {
  SmallVector<Value> result;
  Type i32Ty = b.getI32Type();
  Type f64Ty = b.getF64Type();
  Type f32Ty = b.getF32Type();
  Type i64Ty = b.getI64Type();
  Type bf16x2Ty = VectorType::get(2, b.getBF16Type());
  Type i8x4Ty = VectorType::get(4, b.getI8Type());
  Type i4x8Ty = VectorType::get(8, b.getIntegerType(4));
  Type f32x1Ty = VectorType::get(1, f32Ty);
  auto arrayTy = cast<LLVM::LLVMArrayType>(operand.getType());

  for (unsigned i = 0, e = arrayTy.getNumElements(); i < e; ++i) {
    Value toUse = LLVM::ExtractValueOp::create(b, operand, i);

    // For 4xi8 vectors, the intrinsic expects these to be provided as i32
    // scalar types.
    if (arrayTy.getElementType() == i8x4Ty ||
        arrayTy.getElementType() == i4x8Ty ||
        (arrayTy.getElementType() == bf16x2Ty &&
         operandPtxType == NVVM::MMATypes::bf16) ||
        (arrayTy.getElementType() == f32x1Ty &&
         operandPtxType == NVVM::MMATypes::tf32)) {
      result.push_back(LLVM::BitcastOp::create(b, i32Ty, toUse));
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
        result.push_back(LLVM::ExtractElementOp::create(
            b, toUse,
            LLVM::ConstantOp::create(b, i64Ty, b.getI64IntegerAttr(idx))));
      }
      continue;
    }
    result.push_back(toUse);
  }
  return result;
}

/// Returns whether mbarrier object has shared memory address space.
static bool isMbarrierShared(nvgpu::MBarrierGroupType barrierType) {
  return (mlir::nvgpu::NVGPUDialect::isSharedMemoryAddressSpace(
      barrierType.getMemorySpace()));
}

/// Returns the memory space attribute of the mbarrier object.
Attribute nvgpu::getMbarrierMemorySpace(MLIRContext *context,
                                        nvgpu::MBarrierGroupType barrierType) {
  Attribute memorySpace = {};
  if (isMbarrierShared(barrierType)) {
    memorySpace =
        IntegerAttr::get(IntegerType::get(context, 64),
                         nvgpu::NVGPUDialect::kSharedMemoryAddressSpace);
  }
  return memorySpace;
}

/// Returns memref type of the mbarrier object. The type is defined in the
/// MBarrierGroupType.
MemRefType nvgpu::getMBarrierMemrefType(MLIRContext *context,
                                        nvgpu::MBarrierGroupType barrierType) {
  Attribute memorySpace = nvgpu::getMbarrierMemorySpace(context, barrierType);
  MemRefLayoutAttrInterface layout;
  return MemRefType::get({barrierType.getNumBarriers()},
                         IntegerType::get(context, 64), layout, memorySpace);
}

namespace {

struct MmaLdMatrixOpToNVVM : public ConvertOpToLLVMPattern<nvgpu::LdMatrixOp> {
  using ConvertOpToLLVMPattern<nvgpu::LdMatrixOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::LdMatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = getContext();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

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
    Type innerVectorType = VectorType::get(vectorResultType.getDimSize(1),
                                           vectorResultType.getElementType());

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
        getStridedElementPtr(rewriter, b.getLoc(), srcMemrefType,
                             adaptor.getSrcMemref(), adaptor.getIndices());
    auto shape = NVVM::LdStMatrixShapeAttr::get(rewriter.getContext(), 8, 8);
    Value ldMatrixResult = NVVM::LdMatrixOp::create(
        b, ldMatrixResultType, srcPtr,
        /*num=*/op.getNumTiles(),
        /*layout=*/op.getTranspose() ? NVVM::MMALayout::col
                                     : NVVM::MMALayout::row,
        /*shape=*/shape, /*eltType=*/NVVM::LdStMatrixEltType::B16);

    // The ldmatrix operation returns either a single i32 value or a struct of
    // i32 values. Here we unpack those values and cast them back to their
    // actual vector type (still of width 32b) and repack them into a result
    // struct.
    Type finalResultType = typeConverter->convertType(vectorResultType);
    Value result = LLVM::PoisonOp::create(b, finalResultType);
    for (int64_t i = 0, e = vectorResultType.getDimSize(0); i < e; i++) {
      Value i32Register =
          num32BitRegs > 1 ? LLVM::ExtractValueOp::create(b, ldMatrixResult, i)
                           : ldMatrixResult;
      Value casted = LLVM::BitcastOp::create(b, innerVectorType, i32Register);
      result = LLVM::InsertValueOp::create(b, result, casted, i);
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
  if (elType.isBF16())
    return NVVM::MMATypes::bf16;
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
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
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
        unpackOperandVector(b, adaptor.getMatrixA(), *ptxTypeA);
    SmallVector<Value> matB =
        unpackOperandVector(b, adaptor.getMatrixB(), *ptxTypeB);
    SmallVector<Value> matC =
        unpackOperandVector(b, adaptor.getMatrixC(), *ptxTypeC);

    Type desiredRetTy = typeConverter->convertType(op->getResultTypes()[0]);
    Type intrinsicResTy = inferIntrinsicResultType(
        typeConverter->convertType(op->getResultTypes()[0]));
    Value intrinsicResult =
        NVVM::MmaOp::create(b, intrinsicResTy, matA, matB, matC,
                            /*shape=*/gemmShape,
                            /*b1Op=*/std::nullopt,
                            /*intOverflow=*/overflow,
                            /*multiplicandPtxTypes=*/
                            std::array<NVVM::MMATypes, 2>{*ptxTypeA, *ptxTypeB},
                            /*multiplicandLayouts=*/
                            std::array<NVVM::MMALayout, 2>{
                                NVVM::MMALayout::row, NVVM::MMALayout::col});
    rewriter.replaceOp(op, convertIntrinsicResult(op.getLoc(), intrinsicResTy,
                                                  desiredRetTy, intrinsicResult,
                                                  rewriter));
    return success();
  }
};

struct ConvertNVGPUToNVVMPass
    : public impl::ConvertNVGPUToNVVMPassBase<ConvertNVGPUToNVVMPass> {
  using Base::Base;

  void runOnOperation() override {
    LowerToLLVMOptions options(&getContext());
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext(), options);
    IRRewriter rewriter(&getContext());
    nvgpu::populateCommonGPUTypeAndAttributeConversions(converter);

    /// device-side async tokens cannot be materialized in nvvm. We just
    /// convert them to a dummy i32 type in order to easily drop them during
    /// conversion.
    converter.addConversion([&](nvgpu::DeviceAsyncTokenType type) -> Type {
      return converter.convertType(IntegerType::get(type.getContext(), 32));
    });
    converter.addConversion([&](nvgpu::WarpgroupAccumulatorType type) -> Type {
      Type elemType = type.getFragmented().getElementType();
      int64_t sizeM = type.getFragmented().getDimSize(0);
      int64_t sizeN = type.getFragmented().getDimSize(1);

      unsigned numMembers;
      if (elemType.isF32() || elemType.isInteger(32))
        numMembers = sizeN / 2;
      else if (elemType.isF16())
        numMembers = sizeN / 4;
      else
        llvm_unreachable("unsupported type for warpgroup accumulator");

      SmallVector<Type> innerStructBody;
      for (unsigned i = 0; i < numMembers; i++)
        innerStructBody.push_back(elemType);
      auto innerStructType =
          LLVM::LLVMStructType::getLiteral(type.getContext(), innerStructBody);

      SmallVector<Type> structBody;
      for (int i = 0; i < sizeM; i += kWgmmaSizeM)
        structBody.push_back(innerStructType);

      auto convertedType =
          LLVM::LLVMStructType::getLiteral(type.getContext(), structBody);
      return converter.convertType(convertedType);
    });
    converter.addConversion([&](nvgpu::MBarrierTokenType type) -> Type {
      return converter.convertType(IntegerType::get(type.getContext(), 64));
    });
    converter.addConversion(
        [&](nvgpu::WarpgroupMatrixDescriptorType type) -> Type {
          return converter.convertType(IntegerType::get(type.getContext(), 64));
        });
    converter.addConversion([&](nvgpu::MBarrierGroupType type) -> Type {
      return converter.convertType(
          nvgpu::getMBarrierMemrefType(rewriter.getContext(), type));
    });
    converter.addConversion([&](nvgpu::TensorMapDescriptorType type) -> Type {
      return LLVM::LLVMPointerType::get(type.getContext());
    });
    populateNVGPUToNVVMConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<::mlir::arith::ArithDialect>();
    target.addLegalDialect<::mlir::memref::MemRefDialect>();
    target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
    target.addLegalDialect<::mlir::vector::VectorDialect>();
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, patterns, target);
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
  return asmStr;
}

/// Builds an inline assembly operation corresponding to the specified MMA
/// sparse sync operation.
static FailureOr<LLVM::InlineAsmOp> emitMmaSparseSyncOpAsm(
    ImplicitLocOpBuilder &b, NVVM::MMATypes ptxTypeA, NVVM::MMATypes ptxTypeB,
    NVVM::MMATypes ptxTypeC, NVVM::MMATypes ptxTypeD,
    std::optional<NVVM::MMAIntOverflow> overflow, ArrayRef<Value> unpackedAData,
    ArrayRef<Value> unpackedB, ArrayRef<Value> unpackedC, Value indexData,
    int64_t metadataSelector, const std::array<int64_t, 3> &shape,
    Type intrinsicResultType) {
  auto asmDialectAttr =
      LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_ATT);

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

  return LLVM::InlineAsmOp::create(b,
                                   /*resultTypes=*/intrinsicResultType,
                                   /*operands=*/asmVals,
                                   /*asm_string=*/asmStr,
                                   /*constraints=*/constraintStr,
                                   /*has_side_effects=*/true,
                                   /*is_align_stack=*/false,
                                   LLVM::TailCallKind::None,
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
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
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
        unpackOperandVector(b, adaptor.getMatrixA(), *ptxTypeA);
    SmallVector<Value> matB =
        unpackOperandVector(b, adaptor.getMatrixB(), *ptxTypeB);
    SmallVector<Value> matC =
        unpackOperandVector(b, adaptor.getMatrixC(), *ptxTypeC);

    Type desiredRetTy = typeConverter->convertType(op->getResultTypes()[0]);
    Type intrinsicResTy = inferIntrinsicResultType(
        typeConverter->convertType(op->getResultTypes()[0]));

    // Bitcast the sparse metadata from vector<2xf16> to an i32.
    Value sparseMetadata = adaptor.getSparseMetadata();
    if (sparseMetadata.getType() != VectorType::get(2, rewriter.getI16Type()))
      return op->emitOpError() << "Expected metadata type to be LLVM "
                                  "VectorType of 2 i16 elements";
    sparseMetadata =
        LLVM::BitcastOp::create(b, rewriter.getI32Type(), sparseMetadata);

    FailureOr<LLVM::InlineAsmOp> intrinsicResult = emitMmaSparseSyncOpAsm(
        b, *ptxTypeA, *ptxTypeB, *ptxTypeC, *ptxTypeC, overflow, matA, matB,
        matC, sparseMetadata, op.getSparsitySelector(), op.getMmaShapeAsArray(),
        intrinsicResTy);
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
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Location loc = op.getLoc();
    auto dstMemrefType = cast<MemRefType>(op.getDst().getType());
    Value dstPtr =
        getStridedElementPtr(rewriter, b.getLoc(), dstMemrefType,
                             adaptor.getDst(), adaptor.getDstIndices());
    FailureOr<unsigned> dstAddressSpace =
        getTypeConverter()->getMemRefAddressSpace(dstMemrefType);
    if (failed(dstAddressSpace))
      return rewriter.notifyMatchFailure(
          loc, "destination memref address space not convertible to integer");

    auto srcMemrefType = cast<MemRefType>(op.getSrc().getType());
    FailureOr<unsigned> srcAddressSpace =
        getTypeConverter()->getMemRefAddressSpace(srcMemrefType);
    if (failed(srcAddressSpace))
      return rewriter.notifyMatchFailure(
          loc, "source memref address space not convertible to integer");

    Value scrPtr =
        getStridedElementPtr(rewriter, loc, srcMemrefType, adaptor.getSrc(),
                             adaptor.getSrcIndices());
    // Intrinsics takes a global pointer so we need an address space cast.
    auto srcPointerGlobalType = LLVM::LLVMPointerType::get(
        op->getContext(), static_cast<unsigned>(NVVM::NVVMMemorySpace::Global));
    scrPtr = LLVM::AddrSpaceCastOp::create(b, srcPointerGlobalType, scrPtr);
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
      Value c3I32 =
          LLVM::ConstantOp::create(b, b.getI32Type(), b.getI32IntegerAttr(3));
      Value bitwidth = LLVM::ConstantOp::create(
          b, b.getI32Type(),
          b.getI32IntegerAttr(srcMemrefType.getElementTypeBitWidth()));
      Value srcElementsI32 = LLVM::TruncOp::create(b, b.getI32Type(), srcBytes);
      srcBytes = LLVM::LShrOp::create(
          b, LLVM::MulOp::create(b, bitwidth, srcElementsI32), c3I32);
    }
    // Cache global (.cg) for 16 dst bytes, Cache all (.ca) for sizes other than
    // 16 dst bytes.
    NVVM::LoadCacheModifierKind cacheModifier =
        (op.getBypassL1().value_or(false) && sizeInBytes == 16)
            ? NVVM::LoadCacheModifierKind::CG
            : NVVM::LoadCacheModifierKind::CA;

    NVVM::CpAsyncOp::create(
        b, dstPtr, scrPtr, rewriter.getI32IntegerAttr(sizeInBytes),
        NVVM::LoadCacheModifierKindAttr::get(op->getContext(), cacheModifier),
        srcBytes);

    // Drop the result token.
    Value zero =
        LLVM::ConstantOp::create(b, IntegerType::get(op.getContext(), 32),
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
    NVVM::CpAsyncCommitGroupOp::create(rewriter, op.getLoc());
    // Drop the result token.
    Value zero = LLVM::ConstantOp::create(rewriter, op->getLoc(),
                                          IntegerType::get(op.getContext(), 32),
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
    NVVM::CpAsyncWaitGroupOp::create(rewriter, op.getLoc(), numGroups);
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
    auto global = memref::GlobalOp::create(
        rewriter, funcOp->getLoc(), "__mbarrier",
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
        rewriter.getContext(), op.getBarriers().getType());

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

/// Base class for lowering mbarrier operations to nvvm intrinsics.
template <typename SourceOp>
struct MBarrierBasePattern : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  /// Returns the base pointer of the mbarrier object.
  Value getMbarrierPtr(ImplicitLocOpBuilder &b,
                       nvgpu::MBarrierGroupType mbarType, Value memrefDesc,
                       Value mbarId,
                       ConversionPatternRewriter &rewriter) const {
    MemRefType mbarrierMemrefType =
        nvgpu::getMBarrierMemrefType(rewriter.getContext(), mbarType);
    return ConvertToLLVMPattern::getStridedElementPtr(
        rewriter, b.getLoc(), mbarrierMemrefType, memrefDesc, {mbarId});
  }
};

struct NVGPUMBarrierGetLowering
    : public MBarrierBasePattern<nvgpu::MBarrierGetOp> {
  using MBarrierBasePattern<nvgpu::MBarrierGetOp>::MBarrierBasePattern;

  LogicalResult
  matchAndRewrite(nvgpu::MBarrierGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    nvgpu::MBarrierGroupType mbarrierType = op.getBarriers().getType();
    rewriter.setInsertionPoint(op);
    Value barrier = getMbarrierPtr(b, mbarrierType, adaptor.getBarriers(),
                                   adaptor.getMbarId(), rewriter);
    Type resType = op.getMbarrierPointer().getType();
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resType, barrier);
    return success();
  }
};

/// Lowers `nvgpu.mbarrier.init` to `nvvm.mbarrier.init`
struct NVGPUMBarrierInitLowering
    : public MBarrierBasePattern<nvgpu::MBarrierInitOp> {
  using MBarrierBasePattern<nvgpu::MBarrierInitOp>::MBarrierBasePattern;

  LogicalResult
  matchAndRewrite(nvgpu::MBarrierInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    nvgpu::MBarrierGroupType mbarrierType = op.getBarriers().getType();
    rewriter.setInsertionPoint(op);
    Value barrier = getMbarrierPtr(b, mbarrierType, adaptor.getBarriers(),
                                   adaptor.getMbarId(), rewriter);
    Value count = truncToI32(b, adaptor.getCount());
    rewriter.replaceOpWithNewOp<NVVM::MBarrierInitOp>(op, barrier, count,
                                                      adaptor.getPredicate());
    return success();
  }
};

/// Lowers `nvgpu.mbarrier.arrive` to `nvvm.mbarrier.arrive`
struct NVGPUMBarrierArriveLowering
    : public MBarrierBasePattern<nvgpu::MBarrierArriveOp> {
  using MBarrierBasePattern<nvgpu::MBarrierArriveOp>::MBarrierBasePattern;
  LogicalResult
  matchAndRewrite(nvgpu::MBarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    Value barrier =
        getMbarrierPtr(b, op.getBarriers().getType(), adaptor.getBarriers(),
                       adaptor.getMbarId(), rewriter);
    rewriter.replaceOpWithNewOp<NVVM::MBarrierArriveOp>(op, barrier);
    return success();
  }
};

/// Lowers `nvgpu.mbarrier.arrive.nocomplete` to
/// `nvvm.mbarrier.arrive.nocomplete`
struct NVGPUMBarrierArriveNoCompleteLowering
    : public MBarrierBasePattern<nvgpu::MBarrierArriveNoCompleteOp> {
  using MBarrierBasePattern<
      nvgpu::MBarrierArriveNoCompleteOp>::MBarrierBasePattern;
  LogicalResult
  matchAndRewrite(nvgpu::MBarrierArriveNoCompleteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    Value barrier =
        getMbarrierPtr(b, op.getBarriers().getType(), adaptor.getBarriers(),
                       adaptor.getMbarId(), rewriter);
    Type tokenType = getTypeConverter()->convertType(
        nvgpu::MBarrierTokenType::get(op->getContext()));
    Value count = truncToI32(b, adaptor.getCount());
    rewriter.replaceOpWithNewOp<NVVM::MBarrierArriveNocompleteOp>(
        op, tokenType, barrier, count);
    return success();
  }
};

/// Lowers `nvgpu.mbarrier.test.wait` to `nvvm.mbarrier.test.wait`
struct NVGPUMBarrierTestWaitLowering
    : public MBarrierBasePattern<nvgpu::MBarrierTestWaitOp> {
  using MBarrierBasePattern<nvgpu::MBarrierTestWaitOp>::MBarrierBasePattern;
  LogicalResult
  matchAndRewrite(nvgpu::MBarrierTestWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    Value barrier =
        getMbarrierPtr(b, op.getBarriers().getType(), adaptor.getBarriers(),
                       adaptor.getMbarId(), rewriter);
    Type retType = rewriter.getI1Type();
    rewriter.replaceOpWithNewOp<NVVM::MBarrierTestWaitOp>(op, retType, barrier,
                                                          adaptor.getToken());
    return success();
  }
};

struct NVGPUMBarrierArriveExpectTxLowering
    : public MBarrierBasePattern<nvgpu::MBarrierArriveExpectTxOp> {
  using MBarrierBasePattern<
      nvgpu::MBarrierArriveExpectTxOp>::MBarrierBasePattern;
  LogicalResult
  matchAndRewrite(nvgpu::MBarrierArriveExpectTxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    Value barrier =
        getMbarrierPtr(b, op.getBarriers().getType(), adaptor.getBarriers(),
                       adaptor.getMbarId(), rewriter);
    Value txcount = truncToI32(b, adaptor.getTxcount());
    NVVM::MBarrierArriveExpectTxOp::create(
        rewriter, op->getLoc(), barrier, txcount, // barrier and txcount
        NVVM::MemScopeKind::CTA,                  // default scope is CTA
        false,                                    // relaxed-semantics is false
        adaptor.getPredicate());
    rewriter.eraseOp(op);
    return success();
  }
};

struct NVGPUMBarrierTryWaitParityLowering
    : public MBarrierBasePattern<nvgpu::MBarrierTryWaitParityOp> {
  using MBarrierBasePattern<
      nvgpu::MBarrierTryWaitParityOp>::MBarrierBasePattern;
  LogicalResult
  matchAndRewrite(nvgpu::MBarrierTryWaitParityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    Value barrier =
        getMbarrierPtr(b, op.getBarriers().getType(), adaptor.getBarriers(),
                       adaptor.getMbarId(), rewriter);
    Value ticks = truncToI32(b, adaptor.getTicks());
    Value phase =
        LLVM::ZExtOp::create(b, b.getI32Type(), adaptor.getPhaseParity());
    rewriter.replaceOpWithNewOp<NVVM::MBarrierTryWaitParityOp>(op, barrier,
                                                               phase, ticks);
    return success();
  }
};

struct NVGPUTmaAsyncLoadOpLowering
    : public MBarrierBasePattern<nvgpu::TmaAsyncLoadOp> {
  using MBarrierBasePattern<nvgpu::TmaAsyncLoadOp>::MBarrierBasePattern;
  LogicalResult
  matchAndRewrite(nvgpu::TmaAsyncLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    auto srcMemrefType = cast<MemRefType>(op.getDst().getType());
    Value dest = getStridedElementPtr(rewriter, op->getLoc(), srcMemrefType,
                                      adaptor.getDst(), {});
    // Intrinsics takes a shared-cluster pointer so we need an
    // address space cast from 3 to 7.
    // TODO: Introduce AS(7) in NVGPU.
    auto ptrSharedClusterType = LLVM::LLVMPointerType::get(
        op->getContext(),
        static_cast<unsigned>(NVVM::NVVMMemorySpace::SharedCluster));
    dest = LLVM::AddrSpaceCastOp::create(b, ptrSharedClusterType, dest);

    Value barrier =
        getMbarrierPtr(b, op.getBarriers().getType(), adaptor.getBarriers(),
                       adaptor.getMbarId(), rewriter);

    SmallVector<Value> coords = adaptor.getCoordinates();
    for (auto [index, value] : llvm::enumerate(coords)) {
      coords[index] = truncToI32(b, value);
    }

    // TODO: Enhance the NVGPU Op for other modes too
    rewriter.replaceOpWithNewOp<NVVM::CpAsyncBulkTensorGlobalToSharedClusterOp>(
        op, dest, adaptor.getTensorMapDescriptor(), coords, barrier,
        ValueRange{}, adaptor.getMulticastMask(), Value{},
        NVVM::TMALoadMode::TILE, // default is TILE mode
        false,                   // default is cluster-scope
        nullptr,                 // default is no cta-group
        adaptor.getPredicate());
    return success();
  }
};

struct NVGPUTmaAsyncStoreOpLowering
    : public MBarrierBasePattern<nvgpu::TmaAsyncStoreOp> {
  using MBarrierBasePattern<nvgpu::TmaAsyncStoreOp>::MBarrierBasePattern;
  LogicalResult
  matchAndRewrite(nvgpu::TmaAsyncStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    auto srcMemrefType = cast<MemRefType>(op.getSrc().getType());
    Value dest = getStridedElementPtr(rewriter, op->getLoc(), srcMemrefType,
                                      adaptor.getSrc(), {});
    SmallVector<Value> coords = adaptor.getCoordinates();
    for (auto [index, value] : llvm::enumerate(coords)) {
      coords[index] = truncToI32(b, value);
    }

    // TODO: Enhance the NVGPU Op for other modes too
    rewriter.replaceOpWithNewOp<NVVM::CpAsyncBulkTensorSharedCTAToGlobalOp>(
        op, adaptor.getTensorMapDescriptor(), dest, coords, Value{},
        NVVM::TMAStoreMode::TILE, // default is TILE mode
        adaptor.getPredicate());
    return success();
  }
};

struct NVGPUGenerateWarpgroupDescriptorLowering
    : public ConvertOpToLLVMPattern<nvgpu::WarpgroupGenerateDescriptorOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::WarpgroupGenerateDescriptorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::WarpgroupGenerateDescriptorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

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

    auto ti64 = b.getIntegerType(64);
    auto makeConst = [&](uint64_t index) -> Value {
      return LLVM::ConstantOp::create(b, ti64, b.getI64IntegerAttr(index));
    };
    auto shiftLeft = [&](Value value, unsigned shift) -> Value {
      return LLVM::ShlOp::create(b, ti64, value, makeConst(shift));
    };
    auto shiftRight = [&](Value value, unsigned shift) -> Value {
      return LLVM::LShrOp::create(b, ti64, value, makeConst(shift));
    };
    auto insertBit = [&](Value desc, Value val, int startBit) {
      return LLVM::OrOp::create(b, ti64, desc, shiftLeft(val, startBit));
    };

    int64_t sizeN = op.getTensorMap().getType().getTensor().getDimSize(0);
    uint64_t strideDimVal = (layout << 3) >> exclude4LSB;
    uint64_t leadDimVal = (sizeN * layout) >> exclude4LSB;
    uint64_t offsetVal = 0;

    Value strideDim = makeConst(strideDimVal);
    Value leadDim = makeConst(leadDimVal);

    Value baseAddr = getStridedElementPtr(
        rewriter, op->getLoc(), cast<MemRefType>(op.getTensor().getType()),
        adaptor.getTensor(), {});
    Value basePtr = LLVM::PtrToIntOp::create(b, ti64, baseAddr);
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

    LDBG() << "Generating warpgroup.descriptor: " << "leading_off:"
           << leadDimVal << "\t" << "stride_off :" << strideDimVal << "\t"
           << "base_offset:" << offsetVal << "\t" << "layout_type:" << swizzle
           << " (" << nvgpu::stringifyTensorMapSwizzleKind(swizzleKind)
           << ")\n start_addr :  " << baseAddr;

    rewriter.replaceOp(op, dsc);
    return success();
  }
};

static Value makeI64Const(ImplicitLocOpBuilder &b, int32_t index) {
  return LLVM::ConstantOp::create(b, b.getIntegerType(64),
                                  b.getI32IntegerAttr(index));
}

/// Returns a Value that holds data type enum that is expected by CUDA driver.
static Value elementTypeAsLLVMConstant(ImplicitLocOpBuilder &b, Type type) {
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
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_UINT8);
  if (type.isUnsignedInteger(16))
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_UINT16);
  if (type.isUnsignedInteger(32))
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_UINT32);
  if (type.isUnsignedInteger(64))
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_UINT64);
  if (type.isSignlessInteger(32))
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_INT32);
  if (type.isSignlessInteger(64))
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_INT64);
  if (type.isF16())
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
  if (type.isF32())
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_FLOAT32);
  if (type.isF64())
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_FLOAT64);
  if (type.isBF16())
    return makeI64Const(b, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);

  llvm_unreachable("Not supported data type");
}

struct NVGPUTmaCreateDescriptorOpLowering
    : public ConvertOpToLLVMPattern<nvgpu::TmaCreateDescriptorOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::TmaCreateDescriptorOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(nvgpu::TmaCreateDescriptorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    auto llvmPointerType = LLVM::LLVMPointerType::get(op->getContext());
    Type llvmInt64Type = IntegerType::get(op->getContext(), 64);

    Value tensorElementType =
        elementTypeAsLLVMConstant(b, op.getTensor().getType().getElementType());
    auto promotedOperands = getTypeConverter()->promoteOperands(
        b.getLoc(), op->getOperands(), adaptor.getOperands(), b);

    Value boxArrayPtr = LLVM::AllocaOp::create(
        b, llvmPointerType, llvmInt64Type, makeI64Const(b, 5));
    for (auto [index, value] : llvm::enumerate(adaptor.getBoxDimensions())) {
      Value gep = LLVM::GEPOp::create(b, llvmPointerType, llvmPointerType,
                                      boxArrayPtr, makeI64Const(b, index));
      LLVM::StoreOp::create(b, value, gep);
    }

    nvgpu::TensorMapDescriptorType desc = op.getTensorMap().getType();
    // Set Arguments for the function call
    SmallVector<Value> arguments;
    arguments.push_back(promotedOperands[0]); // rank
    arguments.push_back(promotedOperands[1]); // descriptor
    arguments.push_back(tensorElementType);   // data type
    arguments.push_back(
        makeI64Const(b, (int)desc.getInterleave()));              // interleave
    arguments.push_back(makeI64Const(b, (int)desc.getSwizzle())); // swizzle
    arguments.push_back(makeI64Const(b, (int)desc.getL2promo())); // l2promo
    arguments.push_back(makeI64Const(b, (int)desc.getOob()));     // oob
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
        hostRegisterCallBuilder.create(b.getLoc(), b, arguments).getResult();

    rewriter.replaceOp(op, tensorMap);
    return success();
  }
};

struct NVGPUWarpgroupMmaOpLowering
    : public ConvertOpToLLVMPattern<nvgpu::WarpgroupMmaOp> {
  using ConvertOpToLLVMPattern<nvgpu::WarpgroupMmaOp>::ConvertOpToLLVMPattern;

  /// This is a helper class to generate required NVVM Ops for warp-group level
  /// matrix multiplication.
  /// When the given GEMM shape is larger than the shape of
  /// a wgmma instrution in PTX, it can generate multiple NVVM::WgmmaMmaAsyncOp
  /// Op(s), group and execute them asynchronously. The class also handles
  /// waiting for completion and iterates through WarpgroupMatrixDescriptor to
  /// create descriptors for each instruction.
  ///
  /// For example this is the case when the shape of GEMM is 128x128x128
  ///
  ///    nvvm.wgmma.fence.aligned
  ///
  ///    nvvm.wgmma.mma.async descA, descB
  ///    iterate(descA, descB)
  ///    nvvm.wgmma.mma.async descA, descB
  ///    [6x times more]
  ///
  ///    nvvm.wgmma.group.sync.aligned
  ///    nvvm.wgmma.wait.group.sync [groupId]
  ///
  class WarpgroupGemm {
    nvgpu::WarpgroupMmaOp op;
    ImplicitLocOpBuilder b;
    OpAdaptor adaptor;

    // Entire shape of the given Op
    int64_t totalM, totalN, totalK;

    // Shape of one wgmma instruction
    int wgmmaM = 0, wgmmaN = 0, wgmmaK = 0;

    // Iteration counts for GEMM
    int iterationM = 0, iterationN = 0, iterationK = 0;

    /// The function returns the shape of wgmma instruction that is defined in
    /// PTX programming guide.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape
    void findWgmmaShape(int64_t sizeM, int64_t sizeN, Type inputElemType) {
      wgmmaM = 64;
      wgmmaN = sizeN;
      if (inputElemType.isTF32()) {
        wgmmaK = 8;
      } else if (inputElemType.isF16() || inputElemType.isBF16()) {
        wgmmaK = 16;
      } else if (isa<Float8E4M3FNType, Float8E5M2Type>(inputElemType) ||
                 inputElemType.isInteger(16)) {
        wgmmaK = 32;
      } else if (inputElemType.isInteger(1)) {
        wgmmaK = 256;
      } else {
        llvm_unreachable("msg: not supported K shape");
      }
      LDBG() << "Generating WgmmaMmaAsyncOp shape[m = " << wgmmaM
             << ", n = " << wgmmaN << ", k = " << wgmmaK << "]";
    }

    /// Generates WGMMATypesAttr from MLIR Type
    NVVM::WGMMATypesAttr generateWgmmaType(Type type,
                                           bool useF32 = false) const {
      auto getWgmmaType = [=](Type elemType) {
        if (elemType.isF32() || elemType.isTF32())
          return useF32 ? NVVM::WGMMATypes::f32 : NVVM::WGMMATypes::tf32;
        if (elemType.isF16())
          return NVVM::WGMMATypes::f16;
        if (elemType.isBF16())
          return NVVM::WGMMATypes::bf16;
        if (isa<Float8E4M3FNType>(elemType))
          return NVVM::WGMMATypes::e4m3;
        if (isa<Float8E5M2Type>(elemType))
          return NVVM::WGMMATypes::e5m2;
        if (elemType.isInteger(1))
          return NVVM::WGMMATypes::b1;
        if (elemType.isInteger(8))
          return NVVM::WGMMATypes::s8;
        if (elemType.isUnsignedInteger(8))
          return NVVM::WGMMATypes::u8;
        if (elemType.isInteger(32))
          return NVVM::WGMMATypes::s32;
        llvm_unreachable("unsupported type");
      };
      return NVVM::WGMMATypesAttr::get(op->getContext(), getWgmmaType(type));
    }

    /// Generates layout attribute for the input matrix for wgmma instruction
    NVVM::MMALayoutAttr
    generateWgmmaLayout(std::optional<bool> transpose) const {
      if (transpose.value_or(false))
        return NVVM::MMALayoutAttr::get(op->getContext(), NVVM::MMALayout::col);
      return NVVM::MMALayoutAttr::get(op->getContext(), NVVM::MMALayout::row);
    }

    /// Generates shape attribute for wgmma instruction
    NVVM::MMAShapeAttr generateWgmmaShape() const {
      return NVVM::MMAShapeAttr::get(op->getContext(), wgmmaM, wgmmaN, wgmmaK);
    }

    /// Generates scale attributes of output matrix for wgmma instruction
    NVVM::WGMMAScaleOutAttr generateScaleOut() const {
      return NVVM::WGMMAScaleOutAttr::get(op->getContext(),
                                          NVVM::WGMMAScaleOut::one);
    }
    /// Generates scale attributes of input matrix for wgmma instruction
    NVVM::WGMMAScaleInAttr generateScaleIn() const {
      return NVVM::WGMMAScaleInAttr::get(op->getContext(),
                                         NVVM::WGMMAScaleIn::one);
    }

    /// Basic function to generate Add
    Value makeAdd(Value lhs, Value rhs) {
      return LLVM::AddOp::create(b, lhs.getType(), lhs, rhs);
    };

    /// Moves the descriptor pointer of matrix-A for the next wgmma instruction.
    /// Currently, it only handles row-major.
    ///
    /// It moves the pointer like below for [128][64] size:
    ///                 +2 +4 +6
    ///                  ↓  ↓  ↓
    /// descA    ---> +--+--+--+--+
    ///               |->|->|->|->|
    ///               |  |  |  |  |
    ///               |  |  |  |  |
    ///               |  |  |  |  |
    /// descA+512---> +-----------+
    ///               |  |  |  |  |
    ///               |  |  |  |  |
    ///               |  |  |  |  |
    ///               |  |  |  |  |
    ///               +-----------+
    ///
    Value iterateDescriptorA(Value desc, int i, int j, int k) {
      MemRefType matrixTypeA = op.getDescriptorA().getType().getTensor();
      Type elemA = matrixTypeA.getElementType();
      int byte = elemA.getIntOrFloatBitWidth() / 8;
      int tileShapeA = matrixTypeA.getDimSize(1);
      int incrementVal = ((wgmmaK * k) + (totalK * tileShapeA * i)) * byte;
      incrementVal = incrementVal >> exclude4LSB;
      LDBG() << "\t\t[m: " << i << " n: " << j << " k: " << k
             << "] [wgmma descriptors] Descriptor A + " << incrementVal
             << " | \t ";
      if (!incrementVal)
        return desc;
      return makeAdd(desc, makeI64Const(b, incrementVal));
    }

    /// Moves the descriptor pointer of matrix-B for the next wgmma instruction.
    /// Currently, it only handles column-major.
    ///
    /// It moves the pointer like below for [128][64] size:
    /// descB     ---> +--+--+--+--+--+--+--+--+
    ///                |↓ |  |  |  |  |  |  |  |
    ///                |↓ |  |  |  |  |  |  |  |
    ///                |↓ |  |  |  |  |  |  |  |
    ///                |↓ |  |  |  |  |  |  |  |
    ///                +--+--+--+--+--+--+--+--+
    ///
    Value iterateDescriptorB(Value desc, int i, int j, int k) {
      MemRefType matrixTypeB = op.getDescriptorB().getType().getTensor();
      Type elemB = matrixTypeB.getElementType();
      int byte = elemB.getIntOrFloatBitWidth() / 8;
      int incrementVal = matrixTypeB.getDimSize(0) * wgmmaK * k * byte;
      incrementVal = incrementVal >> exclude4LSB;
      LDBG() << "Descriptor B + " << incrementVal;
      if (!incrementVal)
        return desc;
      return makeAdd(desc, makeI64Const(b, incrementVal));
    }

    /// This function generates a WgmmaMmaAsyncOp using provided GMMA matrix
    /// descriptors and arranges them based on induction variables: i, j, and k.
    Value generateWgmma(int i, int j, int k, Value matrixC) {
      LDBG() << "\t wgmma." << "m" << wgmmaM << "n" << wgmmaN << "k" << wgmmaK
             << "(A[" << (iterationM * wgmmaM) << ":"
             << (iterationM * wgmmaM) + wgmmaM << "][" << (iterationK * wgmmaK)
             << ":" << (iterationK * wgmmaK + wgmmaK) << "] * " << " B["
             << (iterationK * wgmmaK) << ":" << (iterationK * wgmmaK + wgmmaK)
             << "][" << 0 << ":" << wgmmaN << "])";

      Value descriptorA = iterateDescriptorA(adaptor.getDescriptorA(), i, j, k);
      Value descriptorB = iterateDescriptorB(adaptor.getDescriptorB(), i, j, k);

      Type elemA = op.getDescriptorA().getType().getTensor().getElementType();
      NVVM::WGMMATypesAttr itypeA = generateWgmmaType(elemA);

      Type elemB = op.getDescriptorB().getType().getTensor().getElementType();
      NVVM::WGMMATypesAttr itypeB = generateWgmmaType(elemB);

      Type elemD = op.getMatrixC().getType().getFragmented().getElementType();
      NVVM::WGMMATypesAttr itypeD = generateWgmmaType(elemD, true);

      NVVM::MMAShapeAttr shape = generateWgmmaShape();
      NVVM::WGMMAScaleOutAttr scaleOut = generateScaleOut();
      NVVM::WGMMAScaleInAttr scaleIn = generateScaleIn();
      NVVM::MMALayoutAttr layoutA = generateWgmmaLayout(op.getTransposeA());
      NVVM::MMALayoutAttr layoutB = generateWgmmaLayout(!op.getTransposeB());

      auto overflow = NVVM::MMAIntOverflowAttr::get(
          op->getContext(), NVVM::MMAIntOverflow::wrapped);

      return NVVM::WgmmaMmaAsyncOp::create(
          b, matrixC.getType(), matrixC, descriptorA, descriptorB, shape,
          itypeA, itypeB, itypeD, scaleOut, scaleIn, scaleIn, layoutA, layoutB,
          overflow);
    }

    /// Generates multiple wgmma instructions to complete the given GEMM shape
    Value generateWgmmaGroup() {
      Value wgmmaResult =
          LLVM::PoisonOp::create(b, adaptor.getMatrixC().getType());

      // Perform GEMM
      SmallVector<Value> wgmmaResults;
      for (int i = 0; i < iterationM; ++i) {
        Value matrixC =
            LLVM::ExtractValueOp::create(b, adaptor.getMatrixC(), i);
        for (int j = 0; j < iterationN; ++j)
          for (int k = 0; k < iterationK; ++k)
            matrixC = generateWgmma(i, j, k, matrixC);
        wgmmaResults.push_back(matrixC);
      }
      for (auto [idx, matrix] : llvm::enumerate(wgmmaResults)) {
        wgmmaResult = LLVM::InsertValueOp::create(b, wgmmaResult.getType(),
                                                  wgmmaResult, matrix, idx);
      }
      return wgmmaResult;
    }

  public:
    WarpgroupGemm(nvgpu::WarpgroupMmaOp op, ImplicitLocOpBuilder &b,
                  OpAdaptor adaptor)
        : op(op), b(b), adaptor(adaptor) {
      // Find the entire GEMM Shape
      totalM = op.getDescriptorA().getType().getTensor().getDimSize(0);
      totalN = op.getDescriptorB().getType().getTensor().getDimSize(1);
      totalK = op.getDescriptorA().getType().getTensor().getDimSize(1);
      LDBG() << "===--- GEMM D[" << totalM << "][" << totalN << "] += A["
             << totalM << "][" << totalK << "] * B[" << totalK << "][" << totalN
             << "] ---===";

      // Find the shape for one wgmma instruction
      findWgmmaShape(
          totalM, totalN,
          op.getDescriptorA().getType().getTensor().getElementType());

      // Iterations counts to complete the given shape with wgmma shape
      iterationM = totalM / wgmmaM;
      iterationN = totalN / wgmmaN;
      iterationK = totalK / wgmmaK;
    }

    /// Generates WgmmaMmaAsync Ops to complete the specified GEMM  shape. It
    /// includes generating a fence Op (WgmmaFenceAlignedOp) before the
    /// instructions and group synchronization, as well as waiting
    /// (WgmmaGroupSyncAlignedOp) for group synchronization
    /// (WgmmaWaitGroupSyncOp) after the instructions.
    Value generateWarpgroupMma() {
      NVVM::WgmmaFenceAlignedOp::create(b);
      Value wgmmaResult = generateWgmmaGroup();
      NVVM::WgmmaGroupSyncAlignedOp::create(b);
      NVVM::WgmmaWaitGroupSyncOp::create(b, op.getWaitGroup());
      return wgmmaResult;
    }
  };
  LogicalResult
  matchAndRewrite(nvgpu::WarpgroupMmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    // Step 1. Build a helper class
    WarpgroupGemm warpgroupGemm(op, b, adaptor);

    // Step 2. Get the entire GEMM Shape
    Value wgmmaResult = warpgroupGemm.generateWarpgroupMma();

    // Step 3. Replace fragmented result struct with the op results
    rewriter.replaceOp(op, wgmmaResult);
    return success();
  }
};

struct NVGPUWarpgroupMmaStoreOpLowering
    : public ConvertOpToLLVMPattern<nvgpu::WarpgroupMmaStoreOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::WarpgroupMmaStoreOp>::ConvertOpToLLVMPattern;

  /// This function stores a fragmented register matrix owned by a warp group
  /// (128 threads) into a memref. Each thread has 64 registers, each the size
  /// of a struct.
  /// Here is what each threads (T) holds, each `d` is struct value with a
  /// number.
  ///
  /// Threads in warp-group (128 threads) and what they owns in the matrixD:
  /// 0-31 	  Warp-0  -> MatrixD[0:15 ][0:N]
  /// 32-63 	Warp-1  -> MatrixD[16:31][0:N]
  /// 64-95 	Warp-2  -> MatrixD[32:47][0:N]
  /// 96-127 	Warp-3  -> MatrixD[48:64][0:N]
  ///
  /// Matrix-D:
  ///   +______________________________________________________________________+
  ///   |     0-1  |    2-3  |    4-5  |    6-7  |   8-9  |   10-11|..|N-8,N-7 |
  /// 0 | T0:d0-d1 |T1:d0-d1 |T2:d0-d1 |T3:d0-d1 |T0:d4-d5| T1:d4-d5..|T0:dX-dY|
  /// 1 | T4:d0-d1 |T5:d0-d1 |T6:d0-d1 |T7:d0-d1 |T4:d4-d5| T5:d4-d5..|T4:dX-dY|
  /// ..| .........|.........|.........|.........|........|...........|........|
  /// 8 | T0:d2-d3 |T1:d2-d3 |T2:d2-d3 |T3:d2-d3 |T0:d6-d7|T1:d6-d7,..|T0:dZ-dW|
  /// 9 | T4:d2-d3 |T5:d2-d3 |T6:d2-d3 |T7:d2-d3 |T4:d6-d7| T5:d6-d7..|T4:dZ-dW|
  /// ..| .........|.........|.........|.........|........|...........|........|
  /// 15| T28:d2-d3|T29:d2-d3|T30:d2-d3|T31:d2-d3|........|...........|........|
  /// 16| T32:d2-d3|T33:d2-d3|T34:d2-d3|T35:d2-d3|........|...........|........|
  /// ..| .........|.........|.........|.........|........|...........|........|
  /// 32| T64:d2-d3|T65:d2-d3|T66:d2-d3|T67:d2-d3|........|...........|........|
  /// ..| .........|.........|.........|.........|........|...........|........|
  /// 48| T96:d2-d3|T97:d2-d3|T98:d2-d3|T99:d2-d3|........|...........|........|
  /// ..| .........|.........|.........|.........|........|...........|........|
  ///   +______________________________________________________________________+
  ///
  /// \param rewriter: The pattern rewriter.
  /// \param matrixD: Result of the warp-group MMA operation (fragmented
  /// matrix). It is holded by a thread and a struct with 64 elements.
  /// \param dstMemref: The memref where the registers will be stored.
  /// \param offset: the offset within the memref where the registers will be
  /// stored.
  void storeFragmentedMatrix(ImplicitLocOpBuilder &b, Value matrixD,
                             TypedValue<MemRefType> dstMemref,
                             int offset) const {
    Type i32 = b.getI32Type();

    auto makeConst = [&](int32_t index) -> Value {
      return LLVM::ConstantOp::create(b, i32, b.getI32IntegerAttr(index));
    };
    Value c1 = makeConst(1);
    Value c2 = makeConst(2);
    Value c4 = makeConst(4);
    Value c8 = makeConst(8);
    Value c16 = makeConst(16);
    Value warpSize = makeConst(kWarpSize);

    auto makeMul = [&](Value lhs, Value rhs) -> Value {
      return LLVM::MulOp::create(b, lhs.getType(), lhs, rhs);
    };
    auto makeAdd = [&](Value lhs, Value rhs) -> Value {
      return LLVM::AddOp::create(b, lhs.getType(), lhs, rhs);
    };

    auto makeExtractAndStore = [&](int i, Value wgmmaResult, Value x, Value y,
                                   TypedValue<::mlir::MemRefType> memref) {
      Type it = b.getIndexType();
      Value idx = arith::IndexCastOp::create(b, it, x);
      Value idy0 = arith::IndexCastOp::create(b, it, y);
      Value idy1 = arith::IndexCastOp::create(b, it, makeAdd(y, c1));
      Value d0 = LLVM::ExtractValueOp::create(b, wgmmaResult, i);
      Value d1 = LLVM::ExtractValueOp::create(b, wgmmaResult, i + 1);
      memref::StoreOp::create(b, d0, memref, ValueRange{idx, idy0});
      memref::StoreOp::create(b, d1, memref, ValueRange{idx, idy1});
    };

    Value tidx = NVVM::ThreadIdXOp::create(b, i32);
    Value laneId = LLVM::URemOp::create(b, i32, tidx, warpSize);
    Value warpId = LLVM::UDivOp::create(b, i32, tidx, warpSize);
    Value lane4Id = LLVM::UDivOp::create(b, i32, laneId, c4);
    Value lane4modId = LLVM::URemOp::create(b, i32, laneId, c4);

    Value tj = makeMul(lane4modId, c2);
    Value ti = makeAdd(lane4Id, makeMul(warpId, c16));
    if (offset)
      ti = makeAdd(ti, makeConst(offset));

    auto structType = cast<LLVM::LLVMStructType>(matrixD.getType());

    // Number of 32-bit registers owns per thread
    constexpr unsigned numAdjacentRegisters = 2;
    // Number of 8x8 matrices one below another per warp
    constexpr unsigned numStackedMatrices = 2;

    size_t storeCount = (structType.getBody().size() /
                         (numStackedMatrices * numAdjacentRegisters));

    for (size_t i = 0; i < numStackedMatrices; ++i) {
      Value idx = makeAdd(ti, makeMul(makeConst(i), c8));
      for (size_t j = 0; j < storeCount; ++j) {
        Value idy = makeAdd(tj, makeMul(makeConst(j), c8));
        size_t structIndex = (i * numAdjacentRegisters) +
                             (j * (numStackedMatrices * numAdjacentRegisters));
        makeExtractAndStore(structIndex, matrixD, idx, idy, dstMemref);
      }
    }
  }

  LogicalResult
  matchAndRewrite(nvgpu::WarpgroupMmaStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int offset = 0;
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    Value matriDValue = adaptor.getMatrixD();
    auto stype = cast<LLVM::LLVMStructType>(matriDValue.getType());
    for (auto [idx, matrixD] : llvm::enumerate(stype.getBody())) {
      auto structType = cast<LLVM::LLVMStructType>(matrixD);
      Value innerStructValue =
          LLVM::ExtractValueOp::create(b, matriDValue, idx);
      storeFragmentedMatrix(b, innerStructValue, op.getDstMemref(), offset);
      offset += structType.getBody().size();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct NVGPUWarpgroupMmaInitAccumulatorOpLowering
    : public ConvertOpToLLVMPattern<nvgpu::WarpgroupMmaInitAccumulatorOp> {
  using ConvertOpToLLVMPattern<
      nvgpu::WarpgroupMmaInitAccumulatorOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(nvgpu::WarpgroupMmaInitAccumulatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    LLVM::LLVMStructType packStructType = cast<LLVM::LLVMStructType>(
        getTypeConverter()->convertType(op.getMatrixC().getType()));
    Type elemType = cast<LLVM::LLVMStructType>(packStructType.getBody().front())
                        .getBody()
                        .front();
    Value zero = LLVM::ConstantOp::create(b, elemType, b.getZeroAttr(elemType));
    Value packStruct = LLVM::PoisonOp::create(b, packStructType);
    SmallVector<Value> innerStructs;
    // Unpack the structs and set all values to zero
    for (auto [idx, s] : llvm::enumerate(packStructType.getBody())) {
      auto structType = cast<LLVM::LLVMStructType>(s);
      Value structValue = LLVM::ExtractValueOp::create(b, packStruct, idx);
      for (unsigned i = 0; i < structType.getBody().size(); ++i) {
        structValue = LLVM::InsertValueOp::create(b, structType, structValue,
                                                  zero, ArrayRef<int64_t>({i}));
      }
      innerStructs.push_back(structValue);
    }
    // Pack the inner structs into a single struct
    for (auto [idx, matrix] : llvm::enumerate(innerStructs)) {
      packStruct = LLVM::InsertValueOp::create(b, packStruct.getType(),
                                               packStruct, matrix, idx);
    }
    rewriter.replaceOp(op, packStruct);
    return success();
  }
};

struct NVGPUTmaFenceOpLowering
    : public ConvertOpToLLVMPattern<nvgpu::TmaFenceOp> {
  using ConvertOpToLLVMPattern<nvgpu::TmaFenceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(nvgpu::TmaFenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    auto i32Ty = b.getI32Type();
    Value tensormapSize =
        LLVM::ConstantOp::create(b, i32Ty, rewriter.getI32IntegerAttr(128));

    auto memscope =
        NVVM::MemScopeKindAttr::get(ctx, ::mlir::NVVM::MemScopeKind::SYS);

    rewriter.replaceOpWithNewOp<NVVM::FenceProxyAcquireOp>(
        op, memscope, adaptor.getTensorMapDescriptor(), tensormapSize);

    return success();
  }
};

struct NVGPUTmaPrefetchOpLowering
    : public ConvertOpToLLVMPattern<nvgpu::TmaPrefetchOp> {
  using ConvertOpToLLVMPattern<nvgpu::TmaPrefetchOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(nvgpu::TmaPrefetchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NVVM::PrefetchOp>(
        op, /* CacheLevel */ nullptr, /* Cache Eviction Priority */ nullptr,
        adaptor.getTensorMapDescriptor(), adaptor.getPredicate(),
        /* Tensormap UnitAttr */ mlir::UnitAttr::get(op.getContext()));
    return success();
  }
};

struct NVGPURcpOpLowering : public ConvertOpToLLVMPattern<nvgpu::RcpOp> {
  using ConvertOpToLLVMPattern<nvgpu::RcpOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(nvgpu::RcpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    auto i64Ty = b.getI64Type();
    auto f32Ty = b.getF32Type();
    VectorType inTy = op.getIn().getType();
    // apply rcp.approx.ftz.f on each element in vector.
    auto convert1DVec = [&](Type llvm1DVectorTy, Value inVec) {
      Value ret1DVec = LLVM::PoisonOp::create(b, llvm1DVectorTy);
      int numElems = llvm::cast<VectorType>(llvm1DVectorTy).getNumElements();
      for (int i = 0; i < numElems; i++) {
        Value idx = LLVM::ConstantOp::create(b, i64Ty, b.getI64IntegerAttr(i));
        Value elem = LLVM::ExtractElementOp::create(b, inVec, idx);
        Value dst = NVVM::RcpApproxFtzF32Op::create(b, f32Ty, elem);
        ret1DVec = LLVM::InsertElementOp::create(b, ret1DVec, dst, idx);
      }
      return ret1DVec;
    };
    if (inTy.getRank() == 1) {
      rewriter.replaceOp(op, convert1DVec(inTy, adaptor.getIn()));
      return success();
    }
    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), adaptor.getOperands(), *(this->getTypeConverter()),
        [&](Type llvm1DVectorTy, ValueRange operands) -> Value {
          OpAdaptor adaptor(operands);
          return convert1DVec(llvm1DVectorTy, adaptor.getIn());
        },
        rewriter);
  }
};

//===----------------------------------------------------------------------===//
// NVGPUConvertFloatOp Lowering (truncation)
//===----------------------------------------------------------------------===//

enum class FPKind { F32, BF16, F16, F8, F6, F4 };

static int getEffectiveBitWidth(int bitWidth) {
  // f6 types are 6-bit but NVVM Ops expect 8-bit (i8) containers.
  return bitWidth == 6 ? 8 : bitWidth;
}

static std::optional<FPKind> classifyFPType(Type t) {
  static constexpr auto isConvertibleF8Type = [](Type t) {
    return isa<Float8E4M3FNType, Float8E5M2Type, Float8E8M0FNUType>(t);
  };
  static constexpr auto isConvertibleF6Type = [](Type t) {
    return isa<Float6E2M3FNType, Float6E3M2FNType>(t);
  };
  static constexpr auto isConvertibleF4Type = [](Type t) {
    return isa<Float4E2M1FNType>(t);
  };

  if (t.isF32())
    return FPKind::F32;
  if (t.isBF16())
    return FPKind::BF16;
  if (t.isF16())
    return FPKind::F16;
  if (isConvertibleF8Type(t))
    return FPKind::F8;
  if (isConvertibleF6Type(t))
    return FPKind::F6;
  if (isConvertibleF4Type(t))
    return FPKind::F4;
  return std::nullopt;
}

/// Number of source-side i32 register slots consumed by each NVVM convert Op.
static int getNumSrcI32PerConv(FPKind src) {
  return src == FPKind::F32 ? 2 : 1;
}

/// Conversion op identifier for nvgpu.convert.float truncation dispatch table.
enum class FPTruncConvOp {
  F32x2_TO_F16x2,
  F32x2_TO_BF16x2,
  F32x2_TO_F8x2,
  F32x2_TO_F6x2,
  F32x2_TO_F4x2,
  F16x2_TO_F8x2,
  F16x2_TO_F6x2,
  F16x2_TO_F4x2,
  BF16x2_TO_F8x2,
  BF16x2_TO_F6x2,
  BF16x2_TO_F4x2,
};

struct FPTruncTableEntry {
  FPKind src;
  FPKind dst;
  FPTruncConvOp convOp;
};

static constexpr FPTruncTableEntry kFPTruncTable[] = {
    // f32 source
    {FPKind::F32, FPKind::F16, FPTruncConvOp::F32x2_TO_F16x2},
    {FPKind::F32, FPKind::BF16, FPTruncConvOp::F32x2_TO_BF16x2},
    {FPKind::F32, FPKind::F8, FPTruncConvOp::F32x2_TO_F8x2},
    {FPKind::F32, FPKind::F6, FPTruncConvOp::F32x2_TO_F6x2},
    {FPKind::F32, FPKind::F4, FPTruncConvOp::F32x2_TO_F4x2},
    // f16 source
    {FPKind::F16, FPKind::F8, FPTruncConvOp::F16x2_TO_F8x2},
    {FPKind::F16, FPKind::F6, FPTruncConvOp::F16x2_TO_F6x2},
    {FPKind::F16, FPKind::F4, FPTruncConvOp::F16x2_TO_F4x2},
    // bf16 source
    {FPKind::BF16, FPKind::F8, FPTruncConvOp::BF16x2_TO_F8x2},
    {FPKind::BF16, FPKind::F6, FPTruncConvOp::BF16x2_TO_F6x2},
    {FPKind::BF16, FPKind::F4, FPTruncConvOp::BF16x2_TO_F4x2},
};

static std::optional<FPTruncTableEntry> lookupTruncConvOp(Type srcElemType,
                                                          Type dstElemType) {
  auto srcKind = classifyFPType(srcElemType);
  auto dstKind = classifyFPType(dstElemType);
  if (!srcKind || !dstKind)
    return std::nullopt;
  for (const auto &entry : kFPTruncTable) {
    if (entry.src == *srcKind && entry.dst == *dstKind)
      return entry;
  }
  return std::nullopt;
}

/// Extract a single element from a vector.
static Value extractElement(ImplicitLocOpBuilder &b, Value srcVec, int idx) {
  auto vecTy = cast<VectorType>(srcVec.getType());
  assert(idx >= 0 && idx < vecTy.getNumElements() &&
         "extractElement: index out of bounds");
  IntegerType i64Ty = b.getI64Type();
  return b.create<LLVM::ExtractElementOp>(
      srcVec, b.create<LLVM::ConstantOp>(i64Ty, b.getI64IntegerAttr(idx)));
}

/// Extract a pair of f32 values from an i32 vector at the given base index.
static std::pair<Value, Value> extractF32Pair(ImplicitLocOpBuilder &b,
                                              Value srcI32Vec, int baseIdx) {
  FloatType f32Ty = b.getF32Type();
  Value elem0 = extractElement(b, srcI32Vec, baseIdx);
  Value elem1 = extractElement(b, srcI32Vec, baseIdx + 1);
  return {b.create<LLVM::BitcastOp>(f32Ty, elem0),
          b.create<LLVM::BitcastOp>(f32Ty, elem1)};
}

/// Extract a vector of elements of size i32 from an i32 vector and bitcast to
/// the specified vector type.
static Value extractAndBitcast(ImplicitLocOpBuilder &b, Value srcI32Vec,
                               int idx, VectorType vecTy) {
  Value elem = extractElement(b, srcI32Vec, idx);
  return b.create<LLVM::BitcastOp>(vecTy, elem);
}

/// Create a sub-byte conversion from an f32 pair source and return the native
/// result.
template <typename ConvertOp, typename... Args>
static Value convertFromF32Pair(ImplicitLocOpBuilder &b, Value srcI32Vec,
                                int srcBaseIdx, Type resultTy, Args &&...args) {
  auto [lo, hi] = extractF32Pair(b, srcI32Vec, srcBaseIdx);
  return b.create<ConvertOp>(resultTy, hi, lo, std::forward<Args>(args)...);
}

/// Create a sub-byte conversion from a packed f16x2/bf16x2 source and return
/// the native result.
template <typename ConvertOp, typename... Args>
static Value convertFromPacked(ImplicitLocOpBuilder &b, Value srcI32Vec,
                               int srcBaseIdx, Type srcElemTy, Type resultTy,
                               Args &&...args) {
  Value src = extractAndBitcast(b, srcI32Vec, srcBaseIdx,
                                VectorType::get(2, srcElemTy));
  return b.create<ConvertOp>(resultTy, src, std::forward<Args>(args)...);
}

/// Create a typed NVVM truncation conversion.
static Value createTruncConversion(
    ImplicitLocOpBuilder &b, MLIRContext *ctx, FPTruncConvOp convOp,
    Value srcI32Vec, int srcBaseIdx, NVVM::FPRoundingModeAttr rndAttr,
    NVVM::SaturationModeAttr satAttr, BoolAttr reluAttr, Type dstElemType,
    Type actualDstFloatType, Value randomBits = Value()) {
  IntegerType i8Ty = b.getI8Type();
  IntegerType i16Ty = b.getI16Type();
  IntegerType i32Ty = b.getI32Type();
  auto dstTyAttr = TypeAttr::get(dstElemType);
  auto actualDstTyAttr = TypeAttr::get(actualDstFloatType);

  switch (convOp) {
  case FPTruncConvOp::F32x2_TO_F16x2: {
    auto [lo, hi] = extractF32Pair(b, srcI32Vec, srcBaseIdx);
    Value r = b.create<NVVM::ConvertF32x2ToF16x2Op>(
        VectorType::get(2, b.getF16Type()), hi, lo, randomBits, rndAttr,
        satAttr, reluAttr);
    return b.create<LLVM::BitcastOp>(i32Ty, r);
  }
  case FPTruncConvOp::F32x2_TO_BF16x2: {
    auto [lo, hi] = extractF32Pair(b, srcI32Vec, srcBaseIdx);
    Value r = b.create<NVVM::ConvertF32x2ToBF16x2Op>(
        VectorType::get(2, b.getBF16Type()), hi, lo, randomBits, rndAttr,
        satAttr, reluAttr);
    return b.create<LLVM::BitcastOp>(i32Ty, r);
  }
  case FPTruncConvOp::F32x2_TO_F8x2:
    return convertFromF32Pair<NVVM::ConvertF32x2ToF8x2Op>(
        b, srcI32Vec, srcBaseIdx, i16Ty, rndAttr, satAttr, reluAttr, dstTyAttr);
  case FPTruncConvOp::F32x2_TO_F6x2:
    return convertFromF32Pair<NVVM::ConvertF32x2ToF6x2Op>(
        b, srcI32Vec, srcBaseIdx, i16Ty, reluAttr, actualDstTyAttr);
  case FPTruncConvOp::F32x2_TO_F4x2:
    return convertFromF32Pair<NVVM::ConvertF32x2ToF4x2Op>(
        b, srcI32Vec, srcBaseIdx, i8Ty, reluAttr, dstTyAttr);
  case FPTruncConvOp::F16x2_TO_F8x2:
    return convertFromPacked<NVVM::ConvertF16x2ToF8x2Op>(
        b, srcI32Vec, srcBaseIdx, b.getF16Type(), i16Ty, reluAttr, dstTyAttr);
  case FPTruncConvOp::F16x2_TO_F6x2:
    return convertFromPacked<NVVM::ConvertF16x2ToF6x2Op>(
        b, srcI32Vec, srcBaseIdx, b.getF16Type(), i16Ty, reluAttr,
        actualDstTyAttr);
  case FPTruncConvOp::F16x2_TO_F4x2:
    return convertFromPacked<NVVM::ConvertF16x2ToF4x2Op>(
        b, srcI32Vec, srcBaseIdx, b.getF16Type(), i8Ty, reluAttr,
        actualDstTyAttr);
  case FPTruncConvOp::BF16x2_TO_F8x2:
    return convertFromPacked<NVVM::ConvertBF16x2ToF8x2Op>(
        b, srcI32Vec, srcBaseIdx, b.getBF16Type(), i16Ty, rndAttr, satAttr,
        reluAttr, dstTyAttr);
  case FPTruncConvOp::BF16x2_TO_F6x2:
    return convertFromPacked<NVVM::ConvertBF16x2ToF6x2Op>(
        b, srcI32Vec, srcBaseIdx, b.getBF16Type(), i16Ty, reluAttr,
        actualDstTyAttr);
  case FPTruncConvOp::BF16x2_TO_F4x2:
    return convertFromPacked<NVVM::ConvertBF16x2ToF4x2Op>(
        b, srcI32Vec, srcBaseIdx, b.getBF16Type(), i8Ty, reluAttr,
        actualDstTyAttr);
  }
  llvm_unreachable("unhandled FPTruncConvOp");
}

static LogicalResult lowerFPTrunc(nvgpu::ConvertFloatOp op,
                                  nvgpu::ConvertFloatOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  const LLVMTypeConverter *typeConverter) {
  MLIRContext *ctx = op.getContext();
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  IntegerType i32Ty = b.getI32Type();
  IntegerType i64Ty = b.getI64Type();
  static constexpr int regBits = 32;

  auto srcType = llvm::dyn_cast<VectorType>(op.getIn().getType());
  auto dstType = llvm::dyn_cast<VectorType>(op.getOut().getType());
  if (!srcType || srcType.getRank() != 1 || !dstType || dstType.getRank() != 1)
    return rewriter.notifyMatchFailure(
        op, "expected 1-D vector; canonicalize pattern handles other shapes");

  auto srcElemType = srcType.getElementType();
  auto dstElemType = dstType.getElementType();
  int srcBW = srcType.getElementTypeBitWidth();
  int dstBW = dstType.getElementTypeBitWidth();
  int numElems = srcType.getNumElements();

  NVVM::FPRoundingModeAttr rndModeAttr = op.getRndAttr();
  NVVM::SaturationModeAttr satModeAttr = op.getSatAttr();
  auto reluBoolAttr = op.getReluAttr();
  Value randomBits = adaptor.getRandomBits();
  Type actualDstFloatType = dstElemType;

  // STEP 1: bitcast input vector to i32 register vector.
  // f64 -> f32/f16/bf16 lowers to a single direct LLVM fptrunc
  // f64 -> f8/f6/f4 first truncates to f32 and then reuses the narrow
  //        conversion path below.
  Value input = adaptor.getIn();
  if (srcBW == 64) {
    if (dstBW >= 16) {
      Type convertedType = typeConverter->convertType(dstType);
      assert(convertedType && "failed to convert type");
      Value result = b.create<LLVM::FPTruncOp>(convertedType, input);
      rewriter.replaceOp(op, result);
      return success();
    }
    auto f32VecTy = VectorType::get(srcType.getShape(), b.getF32Type());
    input = b.create<LLVM::FPTruncOp>(f32VecTy, input);
    srcType = f32VecTy;
    srcElemType = b.getF32Type();
    srcBW = 32;
  }

  // f6 types are 6-bit in MLIR but NVVM uses 8-bit containers.
  int effectiveDstBW = getEffectiveBitWidth(dstBW);

  int srcI32Elems = numElems * srcBW / regBits;
  int dstI32Elems = numElems * effectiveDstBW / regBits;
  Value srcI32Vec =
      b.create<LLVM::BitcastOp>(VectorType::get(srcI32Elems, i32Ty), input);
  Value dstI32Vec =
      b.create<LLVM::UndefOp>(VectorType::get(dstI32Elems, i32Ty));

  // STEP 2: look up the conversion op from the (srcType, dstType) table.
  auto convEntry = lookupTruncConvOp(srcElemType, dstElemType);
  if (!convEntry)
    return rewriter.notifyMatchFailure(
        op, "unsupported type combination for truncation");
  FPTruncConvOp convOp = convEntry->convOp;
  int numSrcI32PerConv = getNumSrcI32PerConv(convEntry->src);

  // STEP 3: pack conversion results into destination i32 vector.
  const int srcStep = srcBW / effectiveDstBW;
  const int resultBW =
      effectiveDstBW * 2; // each conversion produces 2 (packed) elements
  const int numConvsPerI32 = regBits / resultBW;

  for (int srcIdx = 0, dstIdx = 0; dstIdx < dstI32Elems;
       srcIdx += srcStep, dstIdx++) {
    Value dstIdxConst =
        b.create<LLVM::ConstantOp>(i64Ty, b.getI64IntegerAttr(dstIdx));
    Value dstValue;

    if (numConvsPerI32 == 1) {
      // f16/bf16 destinations
      dstValue = createTruncConversion(
          b, ctx, convOp, srcI32Vec, srcIdx, rndModeAttr, satModeAttr,
          reluBoolAttr, dstElemType, actualDstFloatType, randomBits);
    } else {
      // f8/f6/f4 destinations: pack sub-results via vector insert + bitcast.
      auto subResultType = IntegerType::get(ctx, resultBW);
      auto subVecTy = VectorType::get(numConvsPerI32, subResultType);
      Value subVec = b.create<LLVM::UndefOp>(subVecTy);

      int insertIdx = numConvsPerI32 - 1;
      int curStep = srcStep;
      while (curStep > 0) {
        curStep -= numSrcI32PerConv;
        Value subResult = createTruncConversion(
            b, ctx, convOp, srcI32Vec, srcIdx + curStep, rndModeAttr,
            satModeAttr, reluBoolAttr, dstElemType, actualDstFloatType,
            /*randomBits=*/Value());
        subVec = b.create<LLVM::InsertElementOp>(
            subVec, subResult,
            b.create<LLVM::ConstantOp>(i64Ty, b.getI64IntegerAttr(insertIdx)));
        insertIdx--;
      }

      dstValue = b.create<LLVM::BitcastOp>(i32Ty, subVec);
    }

    dstI32Vec =
        b.create<LLVM::InsertElementOp>(dstI32Vec, dstValue, dstIdxConst);
  }

  // STEP 4: produce final result.
  Type convertedType = typeConverter->convertType(dstType);
  assert(convertedType && "failed to convert type");
  if (convEntry->dst == FPKind::F6) {
    IntegerType i8Ty = b.getI8Type();
    auto i8VecTy = VectorType::get(numElems, i8Ty);
    Value i8Vec = b.create<LLVM::BitcastOp>(i8VecTy, dstI32Vec);
    Value truncVec = b.create<LLVM::TruncOp>(convertedType, i8Vec);
    rewriter.replaceOp(op, truncVec);
  } else {
    auto dstVec = b.create<LLVM::BitcastOp>(convertedType, dstI32Vec);
    rewriter.replaceOp(op, dstVec);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NVGPUConvertFloatOp Lowering (extension)
//===----------------------------------------------------------------------===//

/// Conversion op identifier for nvgpu.convert.float extension dispatch table.
enum class FPExtConvOp {
  F8x2_TO_F16x2,
  F8x2_TO_BF16x2,
  F6x2_TO_F16x2,
  F6x2_TO_BF16x2,
  F4x2_TO_F16x2,
  F4x2_TO_BF16x2,
};

struct FPExtTableEntry {
  FPKind src;
  FPKind dst;
  FPExtConvOp convOp;
};

static constexpr FPExtTableEntry kFPExtTable[] = {
    {FPKind::F8, FPKind::F16, FPExtConvOp::F8x2_TO_F16x2},
    {FPKind::F8, FPKind::BF16, FPExtConvOp::F8x2_TO_BF16x2},
    {FPKind::F6, FPKind::F16, FPExtConvOp::F6x2_TO_F16x2},
    {FPKind::F6, FPKind::BF16, FPExtConvOp::F6x2_TO_BF16x2},
    {FPKind::F4, FPKind::F16, FPExtConvOp::F4x2_TO_F16x2},
    {FPKind::F4, FPKind::BF16, FPExtConvOp::F4x2_TO_BF16x2},
};

static std::optional<FPExtTableEntry> lookupExtConvOp(Type srcElemType,
                                                      Type dstElemType) {
  auto srcKind = classifyFPType(srcElemType);
  auto dstKind = classifyFPType(dstElemType);
  if (!srcKind || !dstKind)
    return std::nullopt;
  for (const auto &entry : kFPExtTable) {
    if (entry.src == *srcKind && entry.dst == *dstKind)
      return entry;
  }
  return std::nullopt;
}

/// Create a typed NVVM extension conversion.
/// For f8/f6: src is vector<2xi8>.  For f4: src is i8.
/// Returns i32 (bitcast from vector<2xf16> or vector<2xbf16>).
static Value createExtConversion(ImplicitLocOpBuilder &b, MLIRContext *ctx,
                                 FPExtConvOp convOp, Value src,
                                 BoolAttr reluAttr, Type actualSrcFloatType,
                                 Value extScaleFactor = Value()) {
  IntegerType i32Ty = b.getI32Type();
  auto srcTyAttr = TypeAttr::get(actualSrcFloatType);

  switch (convOp) {
  case FPExtConvOp::F8x2_TO_F16x2: {
    Value r = NVVM::ConvertF8x2ToF16x2Op::create(
        b, VectorType::get(2, b.getF16Type()), src, srcTyAttr, reluAttr);
    return b.create<LLVM::BitcastOp>(i32Ty, r);
  }
  case FPExtConvOp::F8x2_TO_BF16x2: {
    Value r = NVVM::ConvertF8x2ToBF16x2Op::create(
        b, VectorType::get(2, b.getBF16Type()), src, extScaleFactor, srcTyAttr);
    return b.create<LLVM::BitcastOp>(i32Ty, r);
  }
  case FPExtConvOp::F6x2_TO_F16x2: {
    Value r = NVVM::ConvertF6x2ToF16x2Op::create(
        b, VectorType::get(2, b.getF16Type()), src, srcTyAttr, reluAttr);
    return b.create<LLVM::BitcastOp>(i32Ty, r);
  }
  case FPExtConvOp::F6x2_TO_BF16x2: {
    Value r = NVVM::ConvertF6x2ToBF16x2Op::create(
        b, VectorType::get(2, b.getBF16Type()), src, extScaleFactor, srcTyAttr);
    return b.create<LLVM::BitcastOp>(i32Ty, r);
  }
  case FPExtConvOp::F4x2_TO_F16x2: {
    Value r = NVVM::ConvertF4x2ToF16x2Op::create(
        b, VectorType::get(2, b.getF16Type()), src, srcTyAttr, reluAttr);
    return b.create<LLVM::BitcastOp>(i32Ty, r);
  }
  case FPExtConvOp::F4x2_TO_BF16x2: {
    Value r = NVVM::ConvertF4x2ToBF16x2Op::create(
        b, VectorType::get(2, b.getBF16Type()), src, extScaleFactor, srcTyAttr);
    return b.create<LLVM::BitcastOp>(i32Ty, r);
  }
  }
  llvm_unreachable("unhandled FPExtConvOp");
}

static LogicalResult lowerFPExt(nvgpu::ConvertFloatOp op,
                                nvgpu::ConvertFloatOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter,
                                const LLVMTypeConverter *typeConverter) {
  MLIRContext *ctx = op.getContext();
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  IntegerType i8Ty = b.getI8Type();
  IntegerType i16Ty = b.getI16Type();
  IntegerType i32Ty = b.getI32Type();
  IntegerType i64Ty = b.getI64Type();

  static constexpr int regBits = 32;
  auto srcType = llvm::dyn_cast<VectorType>(op.getIn().getType());
  auto dstType = llvm::dyn_cast<VectorType>(op.getOut().getType());
  if (!srcType || srcType.getRank() != 1 || !dstType || dstType.getRank() != 1)
    return rewriter.notifyMatchFailure(
        op, "expected 1-D vector; canonicalize pattern handles other shapes");

  auto srcElemType = srcType.getElementType();
  auto dstElemType = dstType.getElementType();
  int srcBW = srcType.getElementTypeBitWidth();
  int dstBW = dstType.getElementTypeBitWidth();
  int numElems = srcType.getNumElements();

  auto reluBoolAttr = op.getReluAttr();
  Type actualSrcFloatType = srcElemType;

  assert(dstBW == 16 || dstBW == 32 || dstBW == 64);

  // Wide source (f16/bf16/f32) to wide destination (f32/f64): single FPExt.
  if (srcBW >= 16 && dstBW >= 32) {
    Value result = adaptor.getIn();
    if (srcElemType != dstElemType) {
      Type convertedType = typeConverter->convertType(dstType);
      assert(convertedType && "failed to convert type");
      result = b.create<LLVM::FPExtOp>(convertedType, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

  // Narrow source (f8/f6/f4): NVVM typed op produces f16/bf16; optionally
  // followed by FPExt to the final f32/f64 destination.
  bool needsFinalFPExt = (dstBW >= 32);
  Type intermediateDstElem = dstElemType;
  if (needsFinalFPExt && llvm::isa<Float8E8M0FNUType>(srcElemType))
    intermediateDstElem = b.getBF16Type();
  else if (needsFinalFPExt)
    intermediateDstElem = b.getF16Type();
  int intermediateDstBW = needsFinalFPExt ? 16 : dstBW;

  // f6 types are 6-bit in MLIR but NVVM uses 8-bit containers.
  int effectiveSrcBW = getEffectiveBitWidth(srcBW);

  // STEP 1: prepare input as i32 register vector.
  // For f6: zext from vector<Nxi6> to vector<Nxi8>, then bitcast to i32s.
  Value inputVec = adaptor.getIn();
  if (srcBW == 6) {
    auto i8VecTy = VectorType::get(numElems, i8Ty);
    inputVec = b.create<LLVM::ZExtOp>(i8VecTy, inputVec);
  }

  int srcI32Elems = numElems * effectiveSrcBW / regBits;
  int dstI32Elems = numElems * intermediateDstBW / regBits;
  Value srcI32Vec =
      b.create<LLVM::BitcastOp>(VectorType::get(srcI32Elems, i32Ty), inputVec);
  Value dstI32Vec =
      b.create<LLVM::UndefOp>(VectorType::get(dstI32Elems, i32Ty));

  // STEP 2: look up the conversion op from the (srcType, dstType) table.
  auto convEntry = lookupExtConvOp(srcElemType, intermediateDstElem);
  if (!convEntry)
    return rewriter.notifyMatchFailure(
        op, "unsupported type combination for extension");
  FPExtConvOp convOp = convEntry->convOp;
  Value extScaleFactor;

  // STEP 3: iterate over source i32 elements, producing destination i32s.
  for (int srcIdx = 0, dstIdx = 0; srcIdx < srcI32Elems; srcIdx++) {
    Value srcI32 = b.create<LLVM::ExtractElementOp>(
        srcI32Vec,
        b.create<LLVM::ConstantOp>(i64Ty, b.getI64IntegerAttr(srcIdx)));

    if (effectiveSrcBW == 8) {
      // f8/f6: one i32 holds 4 bytes -> split into 2 pairs of i16 -> 2 convs.
      Value i16Vec =
          b.create<LLVM::BitcastOp>(VectorType::get(2, i16Ty), srcI32);
      for (int half = 0; half < 2; half++) {
        Value halfI16 = b.create<LLVM::ExtractElementOp>(
            i16Vec,
            b.create<LLVM::ConstantOp>(i64Ty, b.getI64IntegerAttr(half)));
        Value src =
            b.create<LLVM::BitcastOp>(VectorType::get(2, i8Ty), halfI16);
        Value dstValue =
            createExtConversion(b, ctx, convOp, src, reluBoolAttr,
                                actualSrcFloatType, extScaleFactor);
        Value dstIdxConst =
            b.create<LLVM::ConstantOp>(i64Ty, b.getI64IntegerAttr(dstIdx));
        dstI32Vec =
            b.create<LLVM::InsertElementOp>(dstI32Vec, dstValue, dstIdxConst);
        dstIdx++;
      }
    } else {
      // f4: one i32 holds 4 bytes -> each byte is one conversion input.
      Value i8Vec = b.create<LLVM::BitcastOp>(VectorType::get(4, i8Ty), srcI32);
      for (int byteIdx = 0; byteIdx < 4; byteIdx++) {
        Value src = b.create<LLVM::ExtractElementOp>(
            i8Vec,
            b.create<LLVM::ConstantOp>(i64Ty, b.getI64IntegerAttr(byteIdx)));
        Value dstValue =
            createExtConversion(b, ctx, convOp, src, reluBoolAttr,
                                actualSrcFloatType, extScaleFactor);
        Value dstIdxConst =
            b.create<LLVM::ConstantOp>(i64Ty, b.getI64IntegerAttr(dstIdx));
        dstI32Vec =
            b.create<LLVM::InsertElementOp>(dstI32Vec, dstValue, dstIdxConst);
        dstIdx++;
      }
    }
  }

  // STEP 4: produce final result.
  Type convertedType = typeConverter->convertType(dstType);
  assert(convertedType && "failed to convert type");
  Value result;
  if (needsFinalFPExt) {
    auto intermediateVecTy = VectorType::get(numElems, intermediateDstElem);
    Value intermediateVec =
        b.create<LLVM::BitcastOp>(intermediateVecTy, dstI32Vec);
    result = b.create<LLVM::FPExtOp>(convertedType, intermediateVec);
  } else {
    result = b.create<LLVM::BitcastOp>(convertedType, dstI32Vec);
  }
  rewriter.replaceOp(op, result);
  return success();
}

/// Lowers nvgpu.convert.float by dispatching to the truncation or extension
/// helper based on the source/destination bitwidths.
struct NVGPUConvertFloatOpLowering
    : public ConvertOpToLLVMPattern<nvgpu::ConvertFloatOp> {
  using ConvertOpToLLVMPattern<nvgpu::ConvertFloatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvgpu::ConvertFloatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<RankedTensorType>(op.getIn().getType()))
      return rewriter.notifyMatchFailure(
          op, "tensor inputs not handled; type converter should lower first");

    int srcBW =
        getElementTypeOrSelf(op.getIn().getType()).getIntOrFloatBitWidth();
    int dstBW =
        getElementTypeOrSelf(op.getOut().getType()).getIntOrFloatBitWidth();
    if (srcBW > dstBW)
      return lowerFPTrunc(op, adaptor, rewriter, getTypeConverter());
    return lowerFPExt(op, adaptor, rewriter, getTypeConverter());
  }
};

static int64_t computePaddedElems(int64_t numElems, int srcBW, int dstBW,
                                  int step) {
  static constexpr int regBits = 32;
  int effSrcBW = getEffectiveBitWidth(srcBW);
  int effDstBW = getEffectiveBitWidth(dstBW);
  auto ceilDiv = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
  int64_t padded =
      std::max(ceilDiv(numElems * effSrcBW, regBits) * regBits / effSrcBW,
               ceilDiv(numElems * effDstBW, regBits) * regBits / effDstBW);
  return ceilDiv(padded, step) * step;
}

/// Canonicalization pattern for nvgpu.convert.float: handles scalar inputs,
/// non-32-bit-aligned vectors, and multi-rank vectors. Runs as an
/// OpRewritePattern on MLIR types before LLVM type conversion.
struct NVGPUConvertFloatCanonicalizePattern
    : public OpRewritePattern<nvgpu::ConvertFloatOp> {
  using OpRewritePattern<nvgpu::ConvertFloatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nvgpu::ConvertFloatOp op,
                                PatternRewriter &rewriter) const override {
    Type inType = op.getIn().getType();
    Type outType = op.getOut().getType();

    if (isa<RankedTensorType>(inType))
      return failure();

    Type srcElemTy = getElementTypeOrSelf(inType);
    Type dstElemTy = getElementTypeOrSelf(outType);
    int srcBW = srcElemTy.getIntOrFloatBitWidth();
    int dstBW = dstElemTy.getIntOrFloatBitWidth();
    int effSrcBW = getEffectiveBitWidth(srcBW);
    int effDstBW = getEffectiveBitWidth(dstBW);

    bool isScalar = !isa<VectorType>(inType);
    auto srcVecTy = dyn_cast<VectorType>(inType);
    bool isMultiRank = srcVecTy && srcVecTy.getRank() > 1;
    int64_t numElems = isScalar ? 1 : srcVecTy.getNumElements();
    int step = srcBW > dstBW ? effSrcBW / effDstBW : effDstBW / effSrcBW;
    int64_t paddedElems = computePaddedElems(numElems, srcBW, dstBW, step);
    bool needsPad = (paddedElems != numElems);

    if (!isScalar && !isMultiRank && !needsPad)
      return failure();

    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    Value input = op.getIn();

    if (isScalar)
      input = vector::BroadcastOp::create(b, VectorType::get({1}, srcElemTy),
                                          input);

    if (isMultiRank)
      input = vector::ShapeCastOp::create(
          b, VectorType::get({numElems}, srcElemTy), input);

    if (needsPad) {
      auto paddedTy = VectorType::get({paddedElems}, srcElemTy);
      Value zero = arith::ConstantOp::create(
          b, DenseElementsAttr::get(paddedTy, b.getZeroAttr(srcElemTy)));
      input = vector::InsertStridedSliceOp::create(
          b, input, zero, SmallVector<int64_t>{0}, SmallVector<int64_t>{1});
    }

    auto cvtDstTy =
        VectorType::get({needsPad ? paddedElems : numElems}, dstElemTy);
    Value result = nvgpu::ConvertFloatOp::create(
        b, cvtDstTy, input, op.getRndAttr(), op.getSatAttr(), op.getReluAttr(),
        op.getRandomBits());

    if (needsPad)
      result = vector::ExtractStridedSliceOp::create(
          b, result, SmallVector<int64_t>{0}, SmallVector<int64_t>{numElems},
          SmallVector<int64_t>{1});

    if (isMultiRank)
      result =
          vector::ShapeCastOp::create(b, cast<VectorType>(outType), result);

    if (isScalar)
      result = vector::ExtractOp::create(b, result, SmallVector<int64_t>{0});

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::nvgpu::populateCommonGPUTypeAndAttributeConversions(
    TypeConverter &typeConverter) {
  // NVVM uses alloca in the default address space to represent private
  // memory allocations, so drop private annotations. NVVM uses address
  // space 3 for shared memory. NVVM uses the default address space to
  // represent global memory.
  populateGpuMemorySpaceAttributeConversions(
      typeConverter, [](gpu::AddressSpace space) -> unsigned {
        switch (space) {
        case gpu::AddressSpace::Global:
          return static_cast<unsigned>(NVVM::NVVMMemorySpace::Global);
        case gpu::AddressSpace::Workgroup:
          return static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared);
        case gpu::AddressSpace::Private:
          return 0;
        case gpu::AddressSpace::Constant:
          return static_cast<unsigned>(NVVM::NVVMMemorySpace::Constant);
        }
        llvm_unreachable("unknown address space enum value");
      });
}

void mlir::populateNVGPUToNVVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<
      NVGPUMBarrierCreateLowering,           // nvgpu.mbarrier.create
      NVGPUMBarrierInitLowering,             // nvgpu.mbarrier.init
      NVGPUMBarrierGetLowering,              // nvgpu.mbarrier.get
      NVGPUMBarrierArriveLowering,           // nvgpu.mbarrier.arrive
      NVGPUMBarrierArriveNoCompleteLowering, // nvgpu.mbarrier.arrive.no_complete
      NVGPUMBarrierTestWaitLowering,         // nvgpu.mbarrier.test_wait_parity
      NVGPUMBarrierTryWaitParityLowering,    // nvgpu.mbarrier.try_wait_parity
      NVGPUTmaAsyncLoadOpLowering,           // nvgpu.tma.async.load
      NVGPUTmaAsyncStoreOpLowering,          // nvgpu.tma.async.store
      NVGPUTmaCreateDescriptorOpLowering,    // nvgpu.tma.create.descriptor
      NVGPUTmaPrefetchOpLowering,            // nvgpu.tma.prefetch.descriptor
      NVGPUTmaFenceOpLowering,               // nvgpu.tma.fence.descriptor
      NVGPUMBarrierArriveExpectTxLowering,   // nvgpu.mbarrier.arrive.expect_tx
      NVGPUGenerateWarpgroupDescriptorLowering, // nvgpu.warpgroup.generate.descriptor
      NVGPUWarpgroupMmaOpLowering,              // nvgpu.warpgroup.mma
      NVGPUWarpgroupMmaStoreOpLowering,         // nvgpu.warpgroup.mma.store
      NVGPUWarpgroupMmaInitAccumulatorOpLowering, // nvgpu.warpgroup.mma.init.accumulator
      NVGPUConvertFloatOpLowering,                // nvgpu.convert.float
      MmaSyncOptoNVVM, MmaLdMatrixOpToNVVM, NVGPUAsyncCopyLowering,
      NVGPUAsyncCreateGroupLowering, NVGPUAsyncWaitLowering,
      NVGPUMmaSparseSyncLowering, NVGPURcpOpLowering>(converter);

  patterns.add<NVGPUConvertFloatCanonicalizePattern>(patterns.getContext());
}
