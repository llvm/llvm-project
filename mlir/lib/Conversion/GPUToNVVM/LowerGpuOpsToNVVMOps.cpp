//===- LowerGpuOpsToNVVMOps.cpp - MLIR GPU to NVVM lowering passes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate NVVMIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"
#include "../PassDetail.h"

using namespace mlir;

namespace {

/// NVVM memory space identifiers.
enum NVVMMemorySpace {
  /// Global memory space identifier.
  kGlobalMemorySpace = 1,
  /// Shared memory space identifier.
  kSharedMemorySpace = 3
};

/// Convert gpu dialect shfl mode enum to the equivalent nvvm one.
static NVVM::ShflKind convertShflKind(gpu::ShuffleMode mode) {
  switch (mode) {
  case gpu::ShuffleMode::XOR:
    return NVVM::ShflKind::bfly;
  case gpu::ShuffleMode::UP:
    return NVVM::ShflKind::up;
  case gpu::ShuffleMode::DOWN:
    return NVVM::ShflKind::down;
  case gpu::ShuffleMode::IDX:
    return NVVM::ShflKind::idx;
  }
  llvm_unreachable("unknown shuffle mode");
}

struct GPUShuffleOpLowering : public ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern<gpu::ShuffleOp>::ConvertOpToLLVMPattern;

  /// Lowers a shuffle to the corresponding NVVM op.
  ///
  /// Convert the `width` argument into an activeMask (a bitmask which specifies
  /// which threads participate in the shuffle) and a maskAndClamp (specifying
  /// the highest lane which participates in the shuffle).
  ///
  ///     %one = llvm.constant(1 : i32) : i32
  ///     %minus_one = llvm.constant(-1 : i32) : i32
  ///     %thirty_two = llvm.constant(32 : i32) : i32
  ///     %num_lanes = llvm.sub %thirty_two, %width : i32
  ///     %active_mask = llvm.lshr %minus_one, %num_lanes : i32
  ///     %mask_and_clamp = llvm.sub %width, %one : i32
  ///     %shfl = nvvm.shfl.sync.bfly %active_mask, %value, %offset,
  ///         %mask_and_clamp : !llvm<"{ float, i1 }">
  ///     %shfl_value = llvm.extractvalue %shfl[0 : index] :
  ///         !llvm<"{ float, i1 }">
  ///     %shfl_pred = llvm.extractvalue %shfl[1 : index] :
  ///         !llvm<"{ float, i1 }">
  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto valueTy = adaptor.value().getType();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto predTy = IntegerType::get(rewriter.getContext(), 1);
    auto resultTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                     {valueTy, predTy});

    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(1));
    Value minusOne = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(-1));
    Value thirtyTwo = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(32));
    Value numLeadInactiveLane = rewriter.create<LLVM::SubOp>(
        loc, int32Type, thirtyTwo, adaptor.width());
    // Bit mask of active lanes: `(-1) >> (32 - activeWidth)`.
    Value activeMask = rewriter.create<LLVM::LShrOp>(loc, int32Type, minusOne,
                                                     numLeadInactiveLane);
    Value maskAndClamp;
    if (op.mode() == gpu::ShuffleMode::UP) {
      // Clamp lane: `32 - activeWidth`
      maskAndClamp = numLeadInactiveLane;
    } else {
      // Clamp lane: `activeWidth - 1`
      maskAndClamp =
          rewriter.create<LLVM::SubOp>(loc, int32Type, adaptor.width(), one);
    }

    auto returnValueAndIsValidAttr = rewriter.getUnitAttr();
    Value shfl = rewriter.create<NVVM::ShflOp>(
        loc, resultTy, activeMask, adaptor.value(), adaptor.offset(),
        maskAndClamp, convertShflKind(op.mode()), returnValueAndIsValidAttr);
    Value shflValue = rewriter.create<LLVM::ExtractValueOp>(
        loc, valueTy, shfl, rewriter.getIndexArrayAttr(0));
    Value isActiveSrcLane = rewriter.create<LLVM::ExtractValueOp>(
        loc, predTy, shfl, rewriter.getIndexArrayAttr(1));

    rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    return success();
  }
};

struct GPUAsyncCopyLowering
    : public ConvertOpToLLVMPattern<gpu::DeviceAsyncCopyOp> {
  using ConvertOpToLLVMPattern<gpu::DeviceAsyncCopyOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::DeviceAsyncCopyOp op, OpAdaptor adaptor,
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
    auto srcPointerGlobalType =
        LLVM::LLVMPointerType::get(i8Ty, NVVMMemorySpace::kGlobalMemorySpace);
    scrPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, srcPointerGlobalType,
                                                    scrPtr);
    int64_t numElements = adaptor.numElements().getZExtValue();
    int64_t sizeInBytes =
        (dstMemrefType.getElementTypeBitWidth() / 8) * numElements;
    rewriter.create<NVVM::CpAsyncOp>(loc, dstPtr, scrPtr,
                                     rewriter.getI32IntegerAttr(sizeInBytes));

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct GPUAsyncCreateGroupLowering
    : public ConvertOpToLLVMPattern<gpu::DeviceAsyncCreateGroupOp> {
  using ConvertOpToLLVMPattern<
      gpu::DeviceAsyncCreateGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::DeviceAsyncCreateGroupOp op, OpAdaptor adaptor,
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

struct GPUAsyncWaitLowering
    : public ConvertOpToLLVMPattern<gpu::DeviceAsyncWaitOp> {
  using ConvertOpToLLVMPattern<gpu::DeviceAsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::DeviceAsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If numGroup is not present pick 0 as a conservative correct value.
    int32_t numGroups = adaptor.numGroups() ? *adaptor.numGroups() : 0;
    rewriter.create<NVVM::CpAsyncWaitGroupOp>(op.getLoc(), numGroups);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MmaLdMatrixOpToNVVM : public ConvertOpToLLVMPattern<gpu::MmaLdMatrixOp> {
  using ConvertOpToLLVMPattern<gpu::MmaLdMatrixOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::MmaLdMatrixOp op, OpAdaptor adaptor,
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

/// Checks if all the operands of the op being lowered are of LLVM Types. The
/// types are expected to be converted by the `LLVMTypeConverter` before the
/// op is actually lowered. If the type of an operands is not already
/// converted it hints a missing typeConversion and failure is returned in
/// that case.
LogicalResult areAllLLVMTypes(Operation *op, ValueRange operands,
                              ConversionPatternRewriter &rewriter) {
  if (!llvm::all_of(operands, [](Value value) {
        return LLVM::isCompatibleType(value.getType());
      })) {
    return rewriter.notifyMatchFailure(
        op, "cannot convert if operands aren't of LLVM type.");
  }

  return success();
}

/// Returns the type for the intrinsic given the vectorResultType of the
/// `gpu.mma.sync` operation.
Type inferIntrinsicResultType(Type vectorResultType) {
  MLIRContext *ctx = vectorResultType.getContext();
  auto a = vectorResultType.cast<LLVM::LLVMArrayType>();
  auto f16x2Ty = LLVM::getFixedVectorType(Float16Type::get(ctx), 2);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i32x2Ty = LLVM::getFixedVectorType(i32Ty, 2);
  Type f64Ty = Float64Type::get(ctx);
  Type f64x2Ty = LLVM::getFixedVectorType(f64Ty, 2);
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
  return vectorResultType;
}

/// Convert the SSA result of the NVVM intrinsic `nvvm.mma.sync` (which is
/// always an LLVM struct) into a fragment that is compatible with the vector
/// type of this operation. This involves extracting elements from the struct
/// and inserting them into an LLVM array. These extra data-movement
/// operations should be canonicalized away by the LLVM backend.
Value convertIntrinsicResult(Location loc, Type intrinsicResultType,
                             Type resultType, Value intrinsicResult,
                             RewriterBase &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  auto structType = intrinsicResultType.dyn_cast<LLVM::LLVMStructType>();
  auto arrayType = resultType.dyn_cast<LLVM::LLVMArrayType>();
  Type i32Ty = rewriter.getI32Type();
  Type f64Ty = rewriter.getF64Type();
  Type f16x2Ty = LLVM::getFixedVectorType(rewriter.getF16Type(), 2);
  Type i32x2Ty = LLVM::getFixedVectorType(i32Ty, 2);
  Type f64x2Ty = LLVM::getFixedVectorType(f64Ty, 2);

  auto makeConst = [&](int32_t index) -> Value {
    return rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 32),
                                             rewriter.getI32IntegerAttr(index));
  };

  if (arrayType) {
    SmallVector<Value, 4> elements;

    if (arrayType.getElementType() == f16x2Ty) {
      for (unsigned i = 0; i < structType.getBody().size(); i++) {
        elements.push_back(rewriter.create<LLVM::ExtractValueOp>(
            loc, structType.getBody()[i], intrinsicResult,
            rewriter.getI64ArrayAttr(i)));
      }
    }

    // The intrinsic returns i32 and f64 values as individual scalars. We need
    // to extract them from the struct and pack them into vectors.
    if (arrayType.getElementType() == i32x2Ty ||
        arrayType.getElementType() == f64x2Ty) {
      Value vec =
          rewriter.create<LLVM::UndefOp>(loc, arrayType.getElementType());
      for (unsigned i = 0, e = structType.getBody().size() / 2; i < e; i++) {
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
      }
      elements.push_back(vec);
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
SmallVector<Value> unpackOperandVector(RewriterBase &rewriter, Location loc,
                                       Value operand) {
  SmallVector<Value> result;
  Type i32Ty = rewriter.getI32Type();
  Type f64Ty = rewriter.getF64Type();
  Type i8Ty = rewriter.getI8Type();
  Type i8x4Ty = LLVM::getFixedVectorType(i8Ty, 4);
  auto arrayTy = operand.getType().cast<LLVM::LLVMArrayType>();

  for (unsigned i = 0, e = arrayTy.getNumElements(); i < e; ++i) {
    Value toUse = rewriter.create<LLVM::ExtractValueOp>(
        loc, arrayTy.getElementType(), operand, rewriter.getI64ArrayAttr(i));

    // For 4xi8 vectors, the intrinsic expects these to be provided as i32
    // scalar types.
    if (arrayTy.getElementType() == i8x4Ty) {
      result.push_back(
          rewriter.create<LLVM::BitcastOp>(loc, rewriter.getI32Type(), toUse));
      continue;
    }

    // For some element types (i32, f64), we need to unpack the inner
    // vector/array type as well because the intrinsic expects individual
    // scalars to be provided.
    VectorType innerArrayTy = arrayTy.getElementType().dyn_cast<VectorType>();
    if (innerArrayTy && (innerArrayTy.getElementType() == i32Ty ||
                         innerArrayTy.getElementType() == f64Ty)) {
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

struct MmaSyncOptoNVVM : public ConvertOpToLLVMPattern<gpu::MmaSyncOp> {
  using ConvertOpToLLVMPattern<gpu::MmaSyncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::MmaSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (failed(areAllLLVMTypes(op, adaptor.getOperands(), rewriter))) {
      return failure();
    }

    // Get the shapes of the MMAMatrix type being used. The shapes will
    // choose which intrinsic this op will be lowered to.
    auto aType = op.matrixA().getType().cast<VectorType>();

    int64_t m = op.mmaShape()[0].cast<IntegerAttr>().getInt();
    int64_t n = op.mmaShape()[1].cast<IntegerAttr>().getInt();
    int64_t k = op.mmaShape()[2].cast<IntegerAttr>().getInt();
    std::array<int64_t, 3> gemmShape{m, n, k};

    SmallVector<Value> matA =
        unpackOperandVector(rewriter, loc, adaptor.matrixA());
    SmallVector<Value> matB =
        unpackOperandVector(rewriter, loc, adaptor.matrixB());
    SmallVector<Value> matC =
        unpackOperandVector(rewriter, loc, adaptor.matrixC());

    NVVM::MMATypes ptxTypeA;
    NVVM::MMATypes ptxTypeB;
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
    } else {
      return op->emitError("could not deduce operand PTX types");
    }

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

struct GPULaneIdOpToNVVM : ConvertOpToLLVMPattern<gpu::LaneIdOp> {
  using ConvertOpToLLVMPattern<gpu::LaneIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::LaneIdOp op, gpu::LaneIdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Value newOp = rewriter.create<NVVM::LaneIdOp>(loc, rewriter.getI32Type());
    // Truncate or extend the result depending on the index bitwidth specified
    // by the LLVMTypeConverter options.
    const unsigned indexBitwidth = getTypeConverter()->getIndexTypeBitwidth();
    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    }
    rewriter.replaceOp(op, {newOp});
    return success();
  }
};

/// Import the GPU Ops to NVVM Patterns.
#include "GPUToNVVM.cpp.inc"

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct LowerGpuOpsToNVVMOpsPass
    : public ConvertGpuOpsToNVVMOpsBase<LowerGpuOpsToNVVMOpsPass> {
  LowerGpuOpsToNVVMOpsPass() = default;
  LowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
    this->indexBitwidth = indexBitwidth;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    options.emitCWrappers = true;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    /// MemRef conversion for GPU to NVVM lowering. The GPU dialect uses memory
    /// space 5 for private memory attributions, but NVVM represents private
    /// memory allocations as local `alloca`s in the default address space. This
    /// converter drops the private memory space to support the use case above.
    LLVMTypeConverter converter(m.getContext(), options);
    converter.addConversion([&](MemRefType type) -> Optional<Type> {
      if (type.getMemorySpaceAsInt() !=
          gpu::GPUDialect::getPrivateAddressSpace())
        return llvm::None;
      return converter.convertType(MemRefType::Builder(type).setMemorySpace(0));
    });
    /// device-side async tokens cannot be materialized in nvvm. We just convert
    /// them to a dummy i32 type in order to easily drop them during conversion.
    converter.addConversion([&](gpu::DeviceAsyncTokenType type) -> Type {
      return converter.convertType(IntegerType::get(type.getContext(), 32));
    });
    // Lowering for MMAMatrixType.
    converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      return convertMMAToLLVMType(type);
    });
    RewritePatternSet patterns(m.getContext());
    RewritePatternSet llvmPatterns(m.getContext());

    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    populateGpuRewritePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(m, std::move(patterns));

    arith::populateArithmeticToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
    populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
    LLVMConversionTarget target(getContext());
    configureGpuToNVVMConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::configureGpuToNVVMConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::LogOp, LLVM::Log10Op,
                      LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp, LLVM::SqrtOp>();

  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
}

void mlir::populateGpuToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
  patterns
      .add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                       NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, NVVM::BlockDimXOp,
                                       NVVM::BlockDimYOp, NVVM::BlockDimZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp,
                                       NVVM::BlockIdYOp, NVVM::BlockIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::GridDimOp, NVVM::GridDimXOp,
                                       NVVM::GridDimYOp, NVVM::GridDimZOp>,
           GPULaneIdOpToNVVM, GPUShuffleOpLowering, GPUReturnOpLowering,
           MmaSyncOptoNVVM, MmaLdMatrixOpToNVVM>(converter);

  // Explicitly drop memory space when lowering private memory
  // attributions since NVVM models it as `alloca`s in the default
  // memory space and does not support `alloca`s with addrspace(5).
  patterns.add<GPUFuncOpLowering>(
      converter, /*allocaAddrSpace=*/0,
      StringAttr::get(&converter.getContext(),
                      NVVM::NVVMDialect::getKernelFuncAttrName()));

  patterns.add<OpToFuncCallLowering<math::AbsOp>>(converter, "__nv_fabsf",
                                                  "__nv_fabs");
  patterns.add<OpToFuncCallLowering<math::AtanOp>>(converter, "__nv_atanf",
                                                   "__nv_atan");
  patterns.add<OpToFuncCallLowering<math::Atan2Op>>(converter, "__nv_atan2f",
                                                    "__nv_atan2");
  patterns.add<OpToFuncCallLowering<math::CeilOp>>(converter, "__nv_ceilf",
                                                   "__nv_ceil");
  patterns.add<OpToFuncCallLowering<math::CosOp>>(converter, "__nv_cosf",
                                                  "__nv_cos");
  patterns.add<OpToFuncCallLowering<math::ExpOp>>(converter, "__nv_expf",
                                                  "__nv_exp");
  patterns.add<OpToFuncCallLowering<math::Exp2Op>>(converter, "__nv_exp2f",
                                                   "__nv_exp2");
  patterns.add<OpToFuncCallLowering<math::ExpM1Op>>(converter, "__nv_expm1f",
                                                    "__nv_expm1");
  patterns.add<OpToFuncCallLowering<math::FloorOp>>(converter, "__nv_floorf",
                                                    "__nv_floor");
  patterns.add<OpToFuncCallLowering<math::LogOp>>(converter, "__nv_logf",
                                                  "__nv_log");
  patterns.add<OpToFuncCallLowering<math::Log1pOp>>(converter, "__nv_log1pf",
                                                    "__nv_log1p");
  patterns.add<OpToFuncCallLowering<math::Log10Op>>(converter, "__nv_log10f",
                                                    "__nv_log10");
  patterns.add<OpToFuncCallLowering<math::Log2Op>>(converter, "__nv_log2f",
                                                   "__nv_log2");
  patterns.add<OpToFuncCallLowering<math::PowFOp>>(converter, "__nv_powf",
                                                   "__nv_pow");
  patterns.add<OpToFuncCallLowering<math::RsqrtOp>>(converter, "__nv_rsqrtf",
                                                    "__nv_rsqrt");
  patterns.add<OpToFuncCallLowering<math::SinOp>>(converter, "__nv_sinf",
                                                  "__nv_sin");
  patterns.add<OpToFuncCallLowering<math::SqrtOp>>(converter, "__nv_sqrtf",
                                                   "__nv_sqrt");
  patterns.add<OpToFuncCallLowering<math::TanhOp>>(converter, "__nv_tanhf",
                                                   "__nv_tanh");
  patterns.add<GPUAsyncCopyLowering, GPUAsyncCreateGroupLowering,
               GPUAsyncWaitLowering>(converter);
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass>(indexBitwidth);
}
