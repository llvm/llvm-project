//===- AMDGPUToROCDL.cpp - AMDGPU to ROCDL dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTAMDGPUTOROCDL
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::amdgpu;

static Value createI32Constant(ConversionPatternRewriter &rewriter,
                               Location loc, int32_t value) {
  Type llvmI32 = rewriter.getI32Type();
  return rewriter.create<LLVM::ConstantOp>(loc, llvmI32, value);
}

static Value createI1Constant(ConversionPatternRewriter &rewriter, Location loc,
                              bool value) {
  Type llvmI1 = rewriter.getI1Type();
  return rewriter.create<LLVM::ConstantOp>(loc, llvmI1, value);
}

namespace {
// Define commonly used chipsets versions for convenience.
constexpr Chipset kGfx908 = Chipset(9, 0, 8);
constexpr Chipset kGfx90a = Chipset(9, 0, 0xa);
constexpr Chipset kGfx940 = Chipset(9, 4, 0);

/// Define lowering patterns for raw buffer ops
template <typename GpuOp, typename Intrinsic>
struct RawBufferOpLowering : public ConvertOpToLLVMPattern<GpuOp> {
  RawBufferOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<GpuOp>(converter), chipset(chipset) {}

  Chipset chipset;
  static constexpr uint32_t maxVectorOpWidth = 128;

  LogicalResult
  matchAndRewrite(GpuOp gpuOp, typename GpuOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = gpuOp.getLoc();
    Value memref = adaptor.getMemref();
    Value unconvertedMemref = gpuOp.getMemref();
    MemRefType memrefType = cast<MemRefType>(unconvertedMemref.getType());

    if (chipset.majorVersion < 9)
      return gpuOp.emitOpError("raw buffer ops require GCN or higher");

    Value storeData = adaptor.getODSOperands(0)[0];
    if (storeData == memref) // no write component to this op
      storeData = Value();
    Type wantedDataType;
    if (storeData)
      wantedDataType = storeData.getType();
    else
      wantedDataType = gpuOp.getODSResults(0)[0].getType();

    Value atomicCmpData = Value();
    // Operand index 1 of a load is the indices, trying to read them can crash.
    if (storeData) {
      Value maybeCmpData = adaptor.getODSOperands(1)[0];
      if (maybeCmpData != memref)
        atomicCmpData = maybeCmpData;
    }

    Type llvmWantedDataType = this->typeConverter->convertType(wantedDataType);

    Type i32 = rewriter.getI32Type();
    Type llvmI32 = this->typeConverter->convertType(i32);
    Type llvmI16 = this->typeConverter->convertType(rewriter.getI16Type());

    int64_t elementByteWidth = memrefType.getElementTypeBitWidth() / 8;
    Value byteWidthConst = createI32Constant(rewriter, loc, elementByteWidth);

    // If we want to load a vector<NxT> with total size <= 32
    // bits, use a scalar load and bitcast it. Similarly, if bitsize(T) < 32
    // and the total load size is >= 32, use a vector load of N / (bitsize(T) /
    // 32) x i32 and bitcast. Also, the CAS intrinsic requires integer operands,
    // so bitcast any floats to integers. On top of all this, cast bfloat
    // (vectors) to i16 since the backend doesn't currently support bfloat on
    // these operations.
    Type llvmBufferValType = llvmWantedDataType;
    if (wantedDataType.isBF16())
      llvmBufferValType = rewriter.getI16Type();
    if (auto wantedVecType = dyn_cast<VectorType>(wantedDataType))
      if (wantedVecType.getElementType().isBF16())
        llvmBufferValType = wantedVecType.clone(rewriter.getI16Type());
    if (atomicCmpData) {
      if (auto floatType = dyn_cast<FloatType>(wantedDataType))
        llvmBufferValType = this->getTypeConverter()->convertType(
            rewriter.getIntegerType(floatType.getWidth()));
    }
    if (auto dataVector = dyn_cast<VectorType>(wantedDataType)) {
      uint32_t vecLen = dataVector.getNumElements();
      uint32_t elemBits = dataVector.getElementTypeBitWidth();
      uint32_t totalBits = elemBits * vecLen;
      bool usePackedFp16 =
          isa_and_present<RawBufferAtomicFaddOp>(*gpuOp) && vecLen == 2;
      if (totalBits > maxVectorOpWidth)
        return gpuOp.emitOpError(
            "Total width of loads or stores must be no more than " +
            Twine(maxVectorOpWidth) + " bits, but we call for " +
            Twine(totalBits) +
            " bits. This should've been caught in validation");
      if (!usePackedFp16 && elemBits < 32) {
        if (totalBits > 32) {
          if (totalBits % 32 != 0)
            return gpuOp.emitOpError("Load or store of more than 32-bits that "
                                     "doesn't fit into words. Can't happen\n");
          llvmBufferValType = this->typeConverter->convertType(
              VectorType::get(totalBits / 32, i32));
        } else {
          llvmBufferValType = this->typeConverter->convertType(
              rewriter.getIntegerType(totalBits));
        }
      }
    }

    SmallVector<Value, 6> args;
    if (storeData) {
      if (llvmBufferValType != llvmWantedDataType) {
        Value castForStore =
            rewriter.create<LLVM::BitcastOp>(loc, llvmBufferValType, storeData);
        args.push_back(castForStore);
      } else {
        args.push_back(storeData);
      }
    }

    if (atomicCmpData) {
      if (llvmBufferValType != llvmWantedDataType) {
        Value castForCmp = rewriter.create<LLVM::BitcastOp>(
            loc, llvmBufferValType, atomicCmpData);
        args.push_back(castForCmp);
      } else {
        args.push_back(atomicCmpData);
      }
    }

    // Construct buffer descriptor from memref, attributes
    int64_t offset = 0;
    SmallVector<int64_t, 5> strides;
    if (failed(getStridesAndOffset(memrefType, strides, offset)))
      return gpuOp.emitOpError("Can't lower non-stride-offset memrefs");

    MemRefDescriptor memrefDescriptor(memref);

    Value ptr = memrefDescriptor.alignedPtr(rewriter, loc);
    // The stride value is always 0 for raw buffers. This also disables
    // swizling.
    Value stride = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI16, rewriter.getI16IntegerAttr(0));
    Value numRecords;
    if (memrefType.hasStaticShape()) {
      numRecords = createI32Constant(
          rewriter, loc,
          static_cast<int32_t>(memrefType.getNumElements() * elementByteWidth));
    } else {
      Value maxIndex;
      for (uint32_t i = 0, e = memrefType.getRank(); i < e; ++i) {
        Value size = memrefDescriptor.size(rewriter, loc, i);
        Value stride = memrefDescriptor.stride(rewriter, loc, i);
        stride = rewriter.create<LLVM::MulOp>(loc, stride, byteWidthConst);
        Value maxThisDim = rewriter.create<LLVM::MulOp>(loc, size, stride);
        maxIndex = maxIndex ? rewriter.create<LLVM::MaximumOp>(loc, maxIndex,
                                                               maxThisDim)
                            : maxThisDim;
      }
      numRecords = rewriter.create<LLVM::TruncOp>(loc, llvmI32, maxIndex);
    }

    // Flag word:
    // bits 0-11: dst sel, ignored by these intrinsics
    // bits 12-14: data format (ignored, must be nonzero, 7=float)
    // bits 15-18: data format (ignored, must be nonzero, 4=32bit)
    // bit 19: In nested heap (0 here)
    // bit 20: Behavior on unmap (0 means  "return 0 / ignore")
    // bits 21-22: Index stride for swizzles (N/A)
    // bit 23: Add thread ID (0)
    // bit 24: Reserved to 1 (RDNA) or 0 (CDNA)
    // bits 25-26: Reserved (0)
    // bit 27: Buffer is non-volatile (CDNA only)
    // bits 28-29: Out of bounds select (0 = structured, 1 = check index, 2 =
    //  none, 3 = either swizzles or testing against offset field) RDNA only
    // bits 30-31: Type (must be 0)
    uint32_t flags = (7 << 12) | (4 << 15);
    if (chipset.majorVersion >= 10) {
      flags |= (1 << 24);
      uint32_t oob = adaptor.getBoundsCheck() ? 3 : 2;
      flags |= (oob << 28);
    }
    Value flagsConst = createI32Constant(rewriter, loc, flags);
    Type rsrcType = LLVM::LLVMPointerType::get(rewriter.getContext(), 8);
    Value resource = rewriter.createOrFold<ROCDL::MakeBufferRsrcOp>(
        loc, rsrcType, ptr, stride, numRecords, flagsConst);
    args.push_back(resource);

    // Indexing (voffset)
    Value voffset = createI32Constant(rewriter, loc, 0);
    for (auto pair : llvm::enumerate(adaptor.getIndices())) {
      size_t i = pair.index();
      Value index = pair.value();
      Value strideOp;
      if (ShapedType::isDynamic(strides[i])) {
        strideOp = rewriter.create<LLVM::MulOp>(
            loc, memrefDescriptor.stride(rewriter, loc, i), byteWidthConst);
      } else {
        strideOp =
            createI32Constant(rewriter, loc, strides[i] * elementByteWidth);
      }
      index = rewriter.create<LLVM::MulOp>(loc, index, strideOp);
      voffset = rewriter.create<LLVM::AddOp>(loc, voffset, index);
    }
    if (adaptor.getIndexOffset()) {
      int32_t indexOffset = *gpuOp.getIndexOffset() * elementByteWidth;
      Value extraOffsetConst = createI32Constant(rewriter, loc, indexOffset);
      voffset =
          voffset ? rewriter.create<LLVM::AddOp>(loc, voffset, extraOffsetConst)
                  : extraOffsetConst;
    }
    args.push_back(voffset);

    Value sgprOffset = adaptor.getSgprOffset();
    if (!sgprOffset)
      sgprOffset = createI32Constant(rewriter, loc, 0);
    if (ShapedType::isDynamic(offset))
      sgprOffset = rewriter.create<LLVM::AddOp>(
          loc, memrefDescriptor.offset(rewriter, loc), sgprOffset);
    else if (offset > 0)
      sgprOffset = rewriter.create<LLVM::AddOp>(
          loc, sgprOffset, createI32Constant(rewriter, loc, offset));
    args.push_back(sgprOffset);

    // bit 0: GLC = 0 (atomics drop value, less coherency)
    // bits 1-2: SLC, DLC = 0 (similarly)
    // bit 3: swizzled (0 for raw)
    args.push_back(createI32Constant(rewriter, loc, 0));

    llvm::SmallVector<Type, 1> resultTypes(gpuOp->getNumResults(),
                                           llvmBufferValType);
    Operation *lowered = rewriter.create<Intrinsic>(loc, resultTypes, args,
                                                    ArrayRef<NamedAttribute>());
    if (lowered->getNumResults() == 1) {
      Value replacement = lowered->getResult(0);
      if (llvmBufferValType != llvmWantedDataType) {
        replacement = rewriter.create<LLVM::BitcastOp>(loc, llvmWantedDataType,
                                                       replacement);
      }
      rewriter.replaceOp(gpuOp, replacement);
    } else {
      rewriter.eraseOp(gpuOp);
    }
    return success();
  }
};

struct LDSBarrierOpLowering : public ConvertOpToLLVMPattern<LDSBarrierOp> {
  LDSBarrierOpLowering(LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<LDSBarrierOp>(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(LDSBarrierOp op, LDSBarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool requiresInlineAsm = chipset < kGfx90a || chipset.majorVersion == 11;

    if (requiresInlineAsm) {
      auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                      LLVM::AsmDialect::AD_ATT);
      const char *asmStr =
          ";;;WARNING: BREAKS DEBUG WATCHES\ns_waitcnt lgkmcnt(0)\ns_barrier";
      const char *constraints = "";
      rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
          op,
          /*resultTypes=*/TypeRange(), /*operands=*/ValueRange(),
          /*asm_string=*/asmStr, constraints, /*has_side_effects=*/true,
          /*is_align_stack=*/false, /*asm_dialect=*/asmDialectAttr,
          /*operand_attrs=*/ArrayAttr());
      return success();
    }
    if (chipset.majorVersion < 12) {
      constexpr int32_t ldsOnlyBitsGfx6789 = ~(0x1f << 8);
      constexpr int32_t ldsOnlyBitsGfx10 = ~(0x3f << 8);
      // Left in place in case someone disables the inline ASM path or future
      // chipsets use the same bit pattern.
      constexpr int32_t ldsOnlyBitsGfx11 = ~(0x3f << 4);

      int32_t ldsOnlyBits;
      if (chipset.majorVersion == 11)
        ldsOnlyBits = ldsOnlyBitsGfx11;
      else if (chipset.majorVersion == 10)
        ldsOnlyBits = ldsOnlyBitsGfx10;
      else if (chipset.majorVersion <= 9)
        ldsOnlyBits = ldsOnlyBitsGfx6789;
      else
        return op.emitOpError(
                   "don't know how to lower this for chipset major version")
               << chipset.majorVersion;

      Location loc = op->getLoc();
      rewriter.create<ROCDL::WaitcntOp>(loc, ldsOnlyBits);
      rewriter.replaceOpWithNewOp<ROCDL::SBarrierOp>(op);
    } else {
      Location loc = op->getLoc();
      rewriter.create<ROCDL::WaitDscntOp>(loc, 0);
      rewriter.create<ROCDL::BarrierSignalOp>(loc, -1);
      rewriter.replaceOpWithNewOp<ROCDL::BarrierWaitOp>(op, -1);
    }

    return success();
  }
};

struct SchedBarrierOpLowering : public ConvertOpToLLVMPattern<SchedBarrierOp> {
  SchedBarrierOpLowering(LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<SchedBarrierOp>(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(SchedBarrierOp op, SchedBarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ROCDL::SchedBarrier>(op,
                                                     (uint32_t)op.getOpts());
    return success();
  }
};

} // namespace

/// If `input` is a vector of bytes, concatentate those bytes in little-endian
/// order to form a single integer of size 8 * [vector length]. This works
/// around a wart in the AMDGPU intrinsics where operations that logically take
/// vectors of bytes instead integers. Since we do not want to expose this
/// implementation detail to MLIR, we correct for it here.
///
/// In addition, convert vectors of LLVM bfloats to vectors of i16, since AMDGPU
/// MFMA intrinsics pre-date the bfloat type.
static Value mfmaConcatIfNeeded(ConversionPatternRewriter &rewriter,
                                Location loc, Value input) {
  Type inputType = input.getType();
  if (auto vectorType = dyn_cast<VectorType>(inputType)) {
    if (vectorType.getElementType().isBF16())
      return rewriter.create<LLVM::BitcastOp>(
          loc, vectorType.clone(rewriter.getI16Type()), input);

    if (!vectorType.getElementType().isInteger(8))
      return input;
    int64_t numBytes = vectorType.getNumElements();
    Type destType = rewriter.getIntegerType(numBytes * 8);
    Value result = rewriter.create<LLVM::ConstantOp>(
        loc, destType, rewriter.getIntegerAttr(destType, 0));
    for (int64_t i = 0; i < numBytes; ++i) {
      Value idxConst = createI32Constant(rewriter, loc, i);
      Value element =
          rewriter.create<LLVM::ExtractElementOp>(loc, input, idxConst);
      Value extended = rewriter.create<LLVM::ZExtOp>(loc, destType, element);
      Value shiftConst = rewriter.create<LLVM::ConstantOp>(
          loc, destType, rewriter.getIntegerAttr(destType, i * 8));
      Value shifted = rewriter.create<LLVM::ShlOp>(loc, extended, shiftConst);
      result = rewriter.create<LLVM::OrOp>(loc, result, shifted);
    }
    return result;
  }
  return input;
}

/// Push an input operand. If it is a float type, nothing to do. If it is
/// an integer type, then we need to also push its signdness (1 for signed, 0
/// for unsigned) and we need to pack the input 16xi8 vector into a 4xi32
/// vector. We also need to convert bfloat inputs to i16 to account for the lack
/// of bfloat support in the WMMA intrinsics themselves.
static void wmmaPushInputOperand(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 const TypeConverter *typeConverter,
                                 bool isUnsigned, Value llvmInput,
                                 Value mlirInput,
                                 SmallVector<Value, 4> &operands) {
  Type inputType = llvmInput.getType();
  auto vectorType = dyn_cast<VectorType>(inputType);
  Type elemType = vectorType.getElementType();

  if (elemType.isBF16())
    llvmInput = rewriter.create<LLVM::BitcastOp>(
        loc, vectorType.clone(rewriter.getI16Type()), llvmInput);
  if (!elemType.isInteger(8)) {
    operands.push_back(llvmInput);
    return;
  }

  // We need to check the type of the input before conversion to properly test
  // for int8. This is because, in LLVM, fp8 type is converted to int8, so the
  // fp8/int8 information is lost during the conversion process.
  auto mlirInputType = cast<VectorType>(mlirInput.getType());
  bool isInputInt8 = mlirInputType.getElementType().isInteger(8);
  if (isInputInt8) {
    // if element type is 8-bit signed or unsigned, ignore the isUnsigned flag
    bool localIsUnsigned = isUnsigned;
    if (elemType.isUnsignedInteger(8)) {
      localIsUnsigned = true;
    } else if (elemType.isSignedInteger(8)) {
      localIsUnsigned = false;
    }
    Value sign = createI1Constant(rewriter, loc, !localIsUnsigned);
    operands.push_back(sign);
  }

  int64_t numBytes = vectorType.getNumElements();
  Type i32 = rewriter.getI32Type();
  VectorType vectorType32bits = VectorType::get(numBytes * 8 / 32, i32);
  auto llvmVectorType32bits = typeConverter->convertType(vectorType32bits);
  Value result = rewriter.createOrFold<LLVM::BitcastOp>(
      loc, llvmVectorType32bits, llvmInput);
  operands.push_back(result);
}

/// Push the output operand. For many cases this is only pushing the output in
/// the operand list. But when we have f16 -> f16 or bf16 -> bf16 intrinsics,
/// since the same numbers of VGPRs is used, we need to decide if to store the
/// result in the upper 16 bits of the VGPRs or in the lower part. To store the
/// result in the lower 16 bits, set subwordOffset to 1, otherwise result will
/// be stored it in the upper part
static void wmmaPushOutputOperand(ConversionPatternRewriter &rewriter,
                                  Location loc,
                                  const TypeConverter *typeConverter,
                                  Value output, int32_t subwordOffset,
                                  bool clamp, SmallVector<Value, 4> &operands) {
  Type inputType = output.getType();
  auto vectorType = dyn_cast<VectorType>(inputType);
  Type elemType = vectorType.getElementType();
  if (elemType.isBF16())
    output = rewriter.create<LLVM::BitcastOp>(
        loc, vectorType.clone(rewriter.getI16Type()), output);
  operands.push_back(output);
  if (elemType.isF16() || elemType.isBF16() || elemType.isInteger(16)) {
    operands.push_back(createI1Constant(rewriter, loc, subwordOffset));
  } else if (elemType.isInteger(32)) {
    operands.push_back(createI1Constant(rewriter, loc, clamp));
  }
}

/// Return the `rocdl` intrinsic corresponding to a MFMA operation `mfma`
/// if one exists. This includes checking to ensure the intrinsic is supported
/// on the architecture you are compiling for.
static std::optional<StringRef> mfmaOpToIntrinsic(MFMAOp mfma,
                                                  Chipset chipset) {
  uint32_t m = mfma.getM(), n = mfma.getN(), k = mfma.getK(),
           b = mfma.getBlocks();
  Type sourceElem = mfma.getSourceA().getType();
  if (auto sourceType = dyn_cast<VectorType>(sourceElem))
    sourceElem = sourceType.getElementType();
  Type destElem = mfma.getDestC().getType();
  if (auto destType = dyn_cast<VectorType>(destElem))
    destElem = destType.getElementType();

  if (sourceElem.isF32() && destElem.isF32()) {
    if (mfma.getReducePrecision() && chipset >= kGfx940) {
      if (m == 32 && n == 32 && k == 4 && b == 1)
        return ROCDL::mfma_f32_32x32x4_xf32::getOperationName();
      if (m == 16 && n == 16 && k == 8 && b == 1)
        return ROCDL::mfma_f32_16x16x8_xf32::getOperationName();
    }
    if (m == 32 && n == 32 && k == 1 && b == 2)
      return ROCDL::mfma_f32_32x32x1f32::getOperationName();
    if (m == 16 && n == 16 && k == 1 && b == 4)
      return ROCDL::mfma_f32_16x16x1f32::getOperationName();
    if (m == 4 && n == 4 && k == 1 && b == 16)
      return ROCDL::mfma_f32_4x4x1f32::getOperationName();
    if (m == 32 && n == 32 && k == 2 && b == 1)
      return ROCDL::mfma_f32_32x32x2f32::getOperationName();
    if (m == 16 && n == 16 && k == 4 && b == 1)
      return ROCDL::mfma_f32_16x16x4f32::getOperationName();
  }

  if (sourceElem.isF16() && destElem.isF32()) {
    if (m == 32 && n == 32 && k == 4 && b == 2)
      return ROCDL::mfma_f32_32x32x4f16::getOperationName();
    if (m == 16 && n == 16 && k == 4 && b == 4)
      return ROCDL::mfma_f32_16x16x4f16::getOperationName();
    if (m == 4 && n == 4 && k == 4 && b == 16)
      return ROCDL::mfma_f32_4x4x4f16::getOperationName();
    if (m == 32 && n == 32 && k == 8 && b == 1)
      return ROCDL::mfma_f32_32x32x8f16::getOperationName();
    if (m == 16 && n == 16 && k == 16 && b == 1)
      return ROCDL::mfma_f32_16x16x16f16::getOperationName();
  }

  if (sourceElem.isBF16() && destElem.isF32() && chipset >= kGfx90a) {
    if (m == 32 && n == 32 && k == 4 && b == 2)
      return ROCDL::mfma_f32_32x32x4bf16_1k::getOperationName();
    if (m == 16 && n == 16 && k == 4 && b == 4)
      return ROCDL::mfma_f32_16x16x4bf16_1k::getOperationName();
    if (m == 4 && n == 4 && k == 4 && b == 16)
      return ROCDL::mfma_f32_4x4x4bf16_1k::getOperationName();
    if (m == 32 && n == 32 && k == 8 && b == 1)
      return ROCDL::mfma_f32_32x32x8bf16_1k::getOperationName();
    if (m == 16 && n == 16 && k == 16 && b == 1)
      return ROCDL::mfma_f32_16x16x16bf16_1k::getOperationName();
  }

  if (sourceElem.isBF16() && destElem.isF32()) {
    if (m == 32 && n == 32 && k == 2 && b == 2)
      return ROCDL::mfma_f32_32x32x2bf16::getOperationName();
    if (m == 16 && n == 16 && k == 2 && b == 4)
      return ROCDL::mfma_f32_16x16x2bf16::getOperationName();
    if (m == 4 && n == 4 && k == 2 && b == 16)
      return ROCDL::mfma_f32_4x4x2bf16::getOperationName();
    if (m == 32 && n == 32 && k == 4 && b == 1)
      return ROCDL::mfma_f32_32x32x4bf16::getOperationName();
    if (m == 16 && n == 16 && k == 8 && b == 1)
      return ROCDL::mfma_f32_16x16x8bf16::getOperationName();
  }

  if (isa<IntegerType>(sourceElem) && destElem.isInteger(32)) {
    if (m == 32 && n == 32 && k == 4 && b == 2)
      return ROCDL::mfma_i32_32x32x4i8::getOperationName();
    if (m == 16 && n == 16 && k == 4 && b == 4)
      return ROCDL::mfma_i32_16x16x4i8::getOperationName();
    if (m == 4 && n == 4 && k == 4 && b == 16)
      return ROCDL::mfma_i32_4x4x4i8::getOperationName();
    if (m == 32 && n == 32 && k == 8 && b == 1)
      return ROCDL::mfma_i32_32x32x8i8::getOperationName();
    if (m == 16 && n == 16 && k == 16 && b == 1)
      return ROCDL::mfma_i32_16x16x16i8::getOperationName();
    if (m == 32 && n == 32 && k == 16 && b == 1 && chipset >= kGfx940)
      return ROCDL::mfma_i32_32x32x16_i8::getOperationName();
    if (m == 16 && n == 16 && k == 32 && b == 1 && chipset >= kGfx940)
      return ROCDL::mfma_i32_16x16x32_i8::getOperationName();
  }

  if (sourceElem.isF64() && destElem.isF64() && chipset >= kGfx90a) {
    if (m == 16 && n == 16 && k == 4 && b == 1)
      return ROCDL::mfma_f64_16x16x4f64::getOperationName();
    if (m == 4 && n == 4 && k == 4 && b == 4)
      return ROCDL::mfma_f64_4x4x4f64::getOperationName();
  }

  if (sourceElem.isFloat8E5M2FNUZ() && destElem.isF32() && chipset >= kGfx940) {
    // Known to be correct because there are no scalar f8 instructions and
    // because a length mismatch will have been caught by the verifier.
    Type sourceBElem =
        cast<VectorType>(mfma.getSourceB().getType()).getElementType();
    if (m == 16 && n == 16 && k == 32 && b == 1) {
      if (sourceBElem.isFloat8E5M2FNUZ())
        return ROCDL::mfma_f32_16x16x32_bf8_bf8::getOperationName();
      if (sourceBElem.isFloat8E4M3FNUZ())
        return ROCDL::mfma_f32_16x16x32_bf8_fp8::getOperationName();
    }
    if (m == 32 && n == 32 && k == 16 && b == 1) {
      if (sourceBElem.isFloat8E5M2FNUZ())
        return ROCDL::mfma_f32_32x32x16_bf8_bf8::getOperationName();
      if (sourceBElem.isFloat8E4M3FNUZ())
        return ROCDL::mfma_f32_32x32x16_bf8_fp8::getOperationName();
    }
  }

  if (sourceElem.isFloat8E4M3FNUZ() && destElem.isF32() && chipset >= kGfx940) {
    Type sourceBElem =
        cast<VectorType>(mfma.getSourceB().getType()).getElementType();
    if (m == 16 && n == 16 && k == 32 && b == 1) {
      if (sourceBElem.isFloat8E5M2FNUZ())
        return ROCDL::mfma_f32_16x16x32_fp8_bf8::getOperationName();
      if (sourceBElem.isFloat8E4M3FNUZ())
        return ROCDL::mfma_f32_16x16x32_fp8_fp8::getOperationName();
    }
    if (m == 32 && n == 32 && k == 16 && b == 1) {
      if (sourceBElem.isFloat8E5M2FNUZ())
        return ROCDL::mfma_f32_32x32x16_fp8_bf8::getOperationName();
      if (sourceBElem.isFloat8E4M3FNUZ())
        return ROCDL::mfma_f32_32x32x16_fp8_fp8::getOperationName();
    }
  }

  return std::nullopt;
}

/// Return the `rocdl` intrinsic corresponding to a WMMA operation `wmma`
/// if one exists. This includes checking to ensure the intrinsic is supported
/// on the architecture you are compiling for.
static std::optional<StringRef> wmmaOpToIntrinsic(WMMAOp wmma,
                                                  Chipset chipset) {
  auto sourceVectorType = dyn_cast<VectorType>(wmma.getSourceA().getType());
  auto destVectorType = dyn_cast<VectorType>(wmma.getDestC().getType());
  auto elemSourceType = sourceVectorType.getElementType();
  auto elemDestType = destVectorType.getElementType();

  if (elemSourceType.isF16() && elemDestType.isF32())
    return ROCDL::wmma_f32_16x16x16_f16::getOperationName();
  if (elemSourceType.isBF16() && elemDestType.isF32())
    return ROCDL::wmma_f32_16x16x16_bf16::getOperationName();
  if (elemSourceType.isF16() && elemDestType.isF16())
    return ROCDL::wmma_f16_16x16x16_f16::getOperationName();
  if (elemSourceType.isBF16() && elemDestType.isBF16())
    return ROCDL::wmma_bf16_16x16x16_bf16::getOperationName();
  if (elemSourceType.isInteger(8) && elemDestType.isInteger(32))
    return ROCDL::wmma_i32_16x16x16_iu8::getOperationName();
  if (elemSourceType.isFloat8E4M3FN() && elemDestType.isF32())
    return ROCDL::wmma_f32_16x16x16_fp8::getOperationName();
  if (elemSourceType.isFloat8E5M2() && elemDestType.isF32())
    return ROCDL::wmma_f32_16x16x16_bf8::getOperationName();
  return std::nullopt;
}

namespace {
struct MFMAOpLowering : public ConvertOpToLLVMPattern<MFMAOp> {
  MFMAOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<MFMAOp>(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(MFMAOp op, MFMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outType = typeConverter->convertType(op.getDestD().getType());
    Type intrinsicOutType = outType;
    if (auto outVecType = dyn_cast<VectorType>(outType))
      if (outVecType.getElementType().isBF16())
        intrinsicOutType = outVecType.clone(rewriter.getI16Type());

    if (chipset.majorVersion != 9 || chipset < kGfx908)
      return op->emitOpError("MFMA only supported on gfx908+");
    uint32_t getBlgpField = static_cast<uint32_t>(op.getBlgp());
    if (op.getNegateA() || op.getNegateB() || op.getNegateC()) {
      if (chipset < kGfx940)
        return op.emitOpError("negation unsupported on older than gfx940");
      getBlgpField |=
          op.getNegateA() | (op.getNegateB() << 1) | (op.getNegateC() << 2);
    }
    std::optional<StringRef> maybeIntrinsic = mfmaOpToIntrinsic(op, chipset);
    if (!maybeIntrinsic.has_value())
      return op.emitOpError("no intrinsic matching MFMA size on given chipset");
    OperationState loweredOp(loc, *maybeIntrinsic);
    loweredOp.addTypes(intrinsicOutType);
    loweredOp.addOperands(
        {mfmaConcatIfNeeded(rewriter, loc, adaptor.getSourceA()),
         mfmaConcatIfNeeded(rewriter, loc, adaptor.getSourceB()),
         adaptor.getDestC(), createI32Constant(rewriter, loc, op.getCbsz()),
         createI32Constant(rewriter, loc, op.getAbid()),
         createI32Constant(rewriter, loc, getBlgpField)});
    Value lowered = rewriter.create(loweredOp)->getResult(0);
    if (outType != intrinsicOutType)
      lowered = rewriter.create<LLVM::BitcastOp>(loc, outType, lowered);
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

struct WMMAOpLowering : public ConvertOpToLLVMPattern<WMMAOp> {
  WMMAOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<WMMAOp>(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(WMMAOp op, WMMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outType =
        typeConverter->convertType<VectorType>(op.getDestD().getType());
    if (!outType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    if (chipset.majorVersion != 11 && chipset.majorVersion != 12)
      return op->emitOpError("WMMA only supported on gfx11 and gfx12");

    // The WMMA operations represent vectors of bf16s as vectors of i16s, so we
    // need to bitcast bfloats to i16 and then bitcast them back.
    VectorType rawOutType = outType;
    if (outType.getElementType().isBF16())
      rawOutType = outType.clone(rewriter.getI16Type());

    std::optional<StringRef> maybeIntrinsic = wmmaOpToIntrinsic(op, chipset);

    if (!maybeIntrinsic.has_value())
      return op.emitOpError("no intrinsic matching WMMA on the given chipset");

    OperationState loweredOp(loc, *maybeIntrinsic);
    loweredOp.addTypes(rawOutType);

    SmallVector<Value, 4> operands;
    wmmaPushInputOperand(rewriter, loc, typeConverter, op.getUnsignedA(),
                         adaptor.getSourceA(), op.getSourceA(), operands);
    wmmaPushInputOperand(rewriter, loc, typeConverter, op.getUnsignedB(),
                         adaptor.getSourceB(), op.getSourceB(), operands);
    wmmaPushOutputOperand(rewriter, loc, typeConverter, adaptor.getDestC(),
                          op.getSubwordOffset(), op.getClamp(), operands);

    loweredOp.addOperands(operands);
    Operation *lowered = rewriter.create(loweredOp);

    Operation *maybeCastBack = lowered;
    if (rawOutType != outType)
      maybeCastBack =
          rewriter.create<LLVM::BitcastOp>(loc, outType, lowered->getResult(0));
    rewriter.replaceOp(op, maybeCastBack->getResults());

    return success();
  }
};

namespace {
struct ExtPackedFp8OpLowering final
    : public ConvertOpToLLVMPattern<ExtPackedFp8Op> {
  ExtPackedFp8OpLowering(LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<amdgpu::ExtPackedFp8Op>(converter),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(ExtPackedFp8Op op, ExtPackedFp8OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct PackedTrunc2xFp8OpLowering final
    : public ConvertOpToLLVMPattern<PackedTrunc2xFp8Op> {
  PackedTrunc2xFp8OpLowering(LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<amdgpu::PackedTrunc2xFp8Op>(converter),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(PackedTrunc2xFp8Op op, PackedTrunc2xFp8OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct PackedStochRoundFp8OpLowering final
    : public ConvertOpToLLVMPattern<PackedStochRoundFp8Op> {
  PackedStochRoundFp8OpLowering(LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<amdgpu::PackedStochRoundFp8Op>(converter),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(PackedStochRoundFp8Op op,
                  PackedStochRoundFp8OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // end namespace

LogicalResult ExtPackedFp8OpLowering::matchAndRewrite(
    ExtPackedFp8Op op, ExtPackedFp8OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  if (chipset.majorVersion != 9 || chipset < kGfx940)
    return rewriter.notifyMatchFailure(
        loc, "Fp8 conversion instructions are not available on target "
             "architecture and their emulation is not implemented");
  Type v4i8 =
      getTypeConverter()->convertType(VectorType::get(4, rewriter.getI8Type()));
  Type i32 = getTypeConverter()->convertType(rewriter.getI32Type());
  Type f32 = getTypeConverter()->convertType(op.getResult().getType());

  Value source = adaptor.getSource();
  auto sourceVecType = dyn_cast<VectorType>(op.getSource().getType());
  Type sourceElemType = getElementTypeOrSelf(op.getSource());
  // Extend to a v4i8
  if (!sourceVecType || sourceVecType.getNumElements() < 4) {
    Value longVec = rewriter.create<LLVM::UndefOp>(loc, v4i8);
    if (!sourceVecType) {
      longVec = rewriter.create<LLVM::InsertElementOp>(
          loc, longVec, source, createI32Constant(rewriter, loc, 0));
    } else {
      for (int32_t i = 0, e = sourceVecType.getNumElements(); i < e; ++i) {
        Value idx = createI32Constant(rewriter, loc, i);
        Value elem = rewriter.create<LLVM::ExtractElementOp>(loc, source, idx);
        longVec =
            rewriter.create<LLVM::InsertElementOp>(loc, longVec, elem, idx);
      }
    }
    source = longVec;
  }
  Value i32Source = rewriter.create<LLVM::BitcastOp>(loc, i32, source);
  Value wordSel = createI32Constant(rewriter, loc, op.getIndex());
  if (sourceElemType.isFloat8E5M2FNUZ()) {
    rewriter.replaceOpWithNewOp<ROCDL::CvtF32Bf8Op>(op, f32, i32Source,
                                                    wordSel);
  } else if (sourceElemType.isFloat8E4M3FNUZ()) {
    rewriter.replaceOpWithNewOp<ROCDL::CvtF32Fp8Op>(op, f32, i32Source,
                                                    wordSel);
  }
  return success();
}

LogicalResult PackedTrunc2xFp8OpLowering::matchAndRewrite(
    PackedTrunc2xFp8Op op, PackedTrunc2xFp8OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  if (chipset.majorVersion != 9 || chipset < kGfx940)
    return rewriter.notifyMatchFailure(
        loc, "Fp8 conversion instructions are not available on target "
             "architecture and their emulation is not implemented");
  Type i32 = getTypeConverter()->convertType(rewriter.getI32Type());

  Type resultType = op.getResult().getType();
  Type resultElemType = getElementTypeOrSelf(resultType);

  Value sourceA = adaptor.getSourceA();
  Value sourceB = adaptor.getSourceB();
  if (!sourceB)
    sourceB = rewriter.create<LLVM::UndefOp>(loc, sourceA.getType());
  Value existing = adaptor.getExisting();
  if (existing)
    existing = rewriter.create<LLVM::BitcastOp>(loc, i32, existing);
  else
    existing = rewriter.create<LLVM::UndefOp>(loc, i32);
  Value wordSel = createI1Constant(rewriter, loc, op.getWordIndex());

  Value result;
  if (resultElemType.isFloat8E5M2FNUZ())
    result = rewriter.create<ROCDL::CvtPkBf8F32Op>(loc, i32, sourceA, sourceB,
                                                   existing, wordSel);
  else if (resultElemType.isFloat8E4M3FNUZ())
    result = rewriter.create<ROCDL::CvtPkFp8F32Op>(loc, i32, sourceA, sourceB,
                                                   existing, wordSel);

  result = rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
      op, getTypeConverter()->convertType(resultType), result);
  return success();
}

LogicalResult PackedStochRoundFp8OpLowering::matchAndRewrite(
    PackedStochRoundFp8Op op, PackedStochRoundFp8OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  if (chipset.majorVersion != 9 || chipset < kGfx940)
    return rewriter.notifyMatchFailure(
        loc, "Fp8 conversion instructions are not available on target "
             "architecture and their emulation is not implemented");
  Type i32 = getTypeConverter()->convertType(rewriter.getI32Type());

  Type resultType = op.getResult().getType();
  Type resultElemType = getElementTypeOrSelf(resultType);

  Value source = adaptor.getSource();
  Value stoch = adaptor.getStochiasticParam();
  Value existing = adaptor.getExisting();
  if (existing)
    existing = rewriter.create<LLVM::BitcastOp>(loc, i32, existing);
  else
    existing = rewriter.create<LLVM::UndefOp>(loc, i32);
  Value byteSel = createI32Constant(rewriter, loc, op.getStoreIndex());

  Value result;
  if (resultElemType.isFloat8E5M2FNUZ())
    result = rewriter.create<ROCDL::CvtSrBf8F32Op>(loc, i32, source, stoch,
                                                   existing, byteSel);
  else if (resultElemType.isFloat8E4M3FNUZ())
    result = rewriter.create<ROCDL::CvtSrFp8F32Op>(loc, i32, source, stoch,
                                                   existing, byteSel);

  result = rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
      op, getTypeConverter()->convertType(resultType), result);
  return success();
}

// Implement the AMDGPU_DPPLowering class that will convert the amdgpu.dpp
// operation into the corresponding ROCDL instructions.
struct AMDGPUDPPLowering : public ConvertOpToLLVMPattern<DPPOp> {
  AMDGPUDPPLowering(LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<DPPOp>(converter), chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(DPPOp DppOp, DPPOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Convert the source operand to the corresponding LLVM type
    Location loc = DppOp.getLoc();
    Value src = adaptor.getSrc();
    Value old = adaptor.getOld();
    Type srcType = src.getType();
    Type oldType = old.getType();
    Type llvmType = nullptr;
    if (srcType.getIntOrFloatBitWidth() < 32) {
      llvmType = rewriter.getI32Type();
    } else if (isa<FloatType>(srcType)) {
      llvmType = (srcType.getIntOrFloatBitWidth() == 32)
                     ? rewriter.getF32Type()
                     : rewriter.getF64Type();
    } else if (isa<IntegerType>(srcType)) {
      llvmType = (srcType.getIntOrFloatBitWidth() == 32)
                     ? rewriter.getI32Type()
                     : rewriter.getI64Type();
    }
    auto llvmSrcIntType = typeConverter->convertType(
        rewriter.getIntegerType(srcType.getIntOrFloatBitWidth()));

    // If the source type is less of 32, use bitcast to convert it to i32.
    auto convertOperand = [&](Value operand, Type operandType) {
      if (operandType.getIntOrFloatBitWidth() <= 16) {
        if (llvm::isa<FloatType>(operandType)) {
          operand =
              rewriter.create<LLVM::BitcastOp>(loc, llvmSrcIntType, operand);
        }
        auto llvmVecType = typeConverter->convertType(mlir::VectorType::get(
            32 / operandType.getIntOrFloatBitWidth(), llvmSrcIntType));
        Value undefVec = rewriter.create<LLVM::UndefOp>(loc, llvmVecType);
        operand = rewriter.create<LLVM::InsertElementOp>(
            loc, undefVec, operand, createI32Constant(rewriter, loc, 0));
        operand = rewriter.create<LLVM::BitcastOp>(loc, llvmType, operand);
      }
      return operand;
    };

    src = convertOperand(src, srcType);
    old = convertOperand(old, oldType);

    // This is taken from the following file llvm/lib/Target/AMDGPU/SIDefines.h
    enum DppCtrl : unsigned {
      ROW_SHL0 = 0x100,
      ROW_SHR0 = 0x110,
      ROW_ROR0 = 0x120,
      WAVE_SHL1 = 0x130,
      WAVE_ROL1 = 0x134,
      WAVE_SHR1 = 0x138,
      WAVE_ROR1 = 0x13C,
      ROW_MIRROR = 0x140,
      ROW_HALF_MIRROR = 0x141,
      BCAST15 = 0x142,
      BCAST31 = 0x143,
    };

    auto kind = DppOp.getKind();
    auto permArgument = DppOp.getPermArgument();
    uint32_t DppCtrl = 0;

    switch (kind) {

    case DPPPerm::quad_perm:
      if (auto quadPermAttr = cast<ArrayAttr>(*permArgument)) {
        int32_t i = 0;
        for (auto elem : quadPermAttr.getAsRange<IntegerAttr>()) {
          uint32_t num = elem.getInt();
          DppCtrl |= num << (i * 2);
          i++;
        }
      }
      break;
    case DPPPerm::row_shl:
      if (auto intAttr = cast<IntegerAttr>(*permArgument)) {
        DppCtrl = intAttr.getInt() + DppCtrl::ROW_SHL0;
      }
      break;
    case DPPPerm::row_shr:
      if (auto intAttr = cast<IntegerAttr>(*permArgument)) {
        DppCtrl = intAttr.getInt() + DppCtrl::ROW_SHR0;
      }
      break;
    case DPPPerm::row_ror:
      if (auto intAttr = cast<IntegerAttr>(*permArgument)) {
        DppCtrl = intAttr.getInt() + DppCtrl::ROW_ROR0;
      }
      break;
    case DPPPerm::wave_shl:
      DppCtrl = DppCtrl::WAVE_SHL1;
      break;
    case DPPPerm::wave_shr:
      DppCtrl = DppCtrl::WAVE_SHR1;
      break;
    case DPPPerm::wave_rol:
      DppCtrl = DppCtrl::WAVE_ROL1;
      break;
    case DPPPerm::wave_ror:
      DppCtrl = DppCtrl::WAVE_ROR1;
      break;
    case DPPPerm::row_mirror:
      DppCtrl = DppCtrl::ROW_MIRROR;
      break;
    case DPPPerm::row_half_mirror:
      DppCtrl = DppCtrl::ROW_HALF_MIRROR;
      break;
    case DPPPerm::row_bcast_15:
      DppCtrl = DppCtrl::BCAST15;
      break;
    case DPPPerm::row_bcast_31:
      DppCtrl = DppCtrl::BCAST31;
      break;
    }

    // Check for row_mask, bank_mask, bound_ctrl if they exist and create
    // constants
    auto rowMask = DppOp->getAttrOfType<IntegerAttr>("row_mask").getInt();
    auto bankMask = DppOp->getAttrOfType<IntegerAttr>("bank_mask").getInt();
    bool boundCtrl = DppOp->getAttrOfType<BoolAttr>("bound_ctrl").getValue();

    // create a ROCDL_DPPMovOp instruction with the appropriate attributes
    auto dppMovOp = rewriter.create<ROCDL::DPPUpdateOp>(
        loc, llvmType, old, src, DppCtrl, rowMask, bankMask, boundCtrl);

    Value result = dppMovOp.getRes();
    if (srcType.getIntOrFloatBitWidth() < 32) {
      result = rewriter.create<LLVM::TruncOp>(loc, llvmSrcIntType, result);
      if (!llvm::isa<IntegerType>(srcType)) {
        result = rewriter.create<LLVM::BitcastOp>(loc, srcType, result);
      }
    }

    // We are replacing the AMDGPU_DPPOp instruction with the new
    // ROCDL_DPPMovOp instruction
    rewriter.replaceOp(DppOp, ValueRange(result));
    return success();
  }
};

struct ConvertAMDGPUToROCDLPass
    : public impl::ConvertAMDGPUToROCDLBase<ConvertAMDGPUToROCDLPass> {
  ConvertAMDGPUToROCDLPass() = default;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    FailureOr<Chipset> maybeChipset = Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(ctx), "Invalid chipset name: " + chipset);
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    LLVMTypeConverter converter(ctx);
    populateAMDGPUToROCDLConversionPatterns(converter, patterns, *maybeChipset);
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<::mlir::amdgpu::AMDGPUDialect>();
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<::mlir::ROCDL::ROCDLDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateAMDGPUToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   Chipset chipset) {
  patterns
      .add<RawBufferOpLowering<RawBufferLoadOp, ROCDL::RawPtrBufferLoadOp>,
           RawBufferOpLowering<RawBufferStoreOp, ROCDL::RawPtrBufferStoreOp>,
           RawBufferOpLowering<RawBufferAtomicFaddOp,
                               ROCDL::RawPtrBufferAtomicFaddOp>,
           RawBufferOpLowering<RawBufferAtomicFmaxOp,
                               ROCDL::RawPtrBufferAtomicFmaxOp>,
           RawBufferOpLowering<RawBufferAtomicSmaxOp,
                               ROCDL::RawPtrBufferAtomicSmaxOp>,
           RawBufferOpLowering<RawBufferAtomicUminOp,
                               ROCDL::RawPtrBufferAtomicUminOp>,
           RawBufferOpLowering<RawBufferAtomicCmpswapOp,
                               ROCDL::RawPtrBufferAtomicCmpSwap>,
           AMDGPUDPPLowering, LDSBarrierOpLowering, SchedBarrierOpLowering,
           MFMAOpLowering, WMMAOpLowering, ExtPackedFp8OpLowering,
           PackedTrunc2xFp8OpLowering, PackedStochRoundFp8OpLowering>(converter,
                                                                      chipset);
}

std::unique_ptr<Pass> mlir::createConvertAMDGPUToROCDLPass() {
  return std::make_unique<ConvertAMDGPUToROCDLPass>();
}
