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

#include "../LLVMCommon/MemRefDescriptor.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTAMDGPUTOROCDLPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::amdgpu;

// Define commonly used chipsets versions for convenience.
constexpr Chipset kGfx908 = Chipset(9, 0, 8);
constexpr Chipset kGfx90a = Chipset(9, 0, 0xa);
constexpr Chipset kGfx942 = Chipset(9, 4, 2);
constexpr Chipset kGfx950 = Chipset(9, 5, 0);

/// Convert an unsigned number `val` to i32.
static Value convertUnsignedToI32(ConversionPatternRewriter &rewriter,
                                  Location loc, Value val) {
  IntegerType i32 = rewriter.getI32Type();
  // Force check that `val` is of int type.
  auto valTy = cast<IntegerType>(val.getType());
  if (i32 == valTy)
    return val;
  return valTy.getWidth() > 32
             ? Value(rewriter.create<LLVM::TruncOp>(loc, i32, val))
             : Value(rewriter.create<LLVM::ZExtOp>(loc, i32, val));
}

static Value createI32Constant(ConversionPatternRewriter &rewriter,
                               Location loc, int32_t value) {
  Type i32 = rewriter.getI32Type();
  return rewriter.create<LLVM::ConstantOp>(loc, i32, value);
}

static Value createI1Constant(ConversionPatternRewriter &rewriter, Location loc,
                              bool value) {
  Type llvmI1 = rewriter.getI1Type();
  return rewriter.create<LLVM::ConstantOp>(loc, llvmI1, value);
}

/// Returns the linear index used to access an element in the memref.
static Value getLinearIndexI32(ConversionPatternRewriter &rewriter,
                               Location loc, MemRefDescriptor &memRefDescriptor,
                               ValueRange indices, ArrayRef<int64_t> strides) {
  IntegerType i32 = rewriter.getI32Type();
  Value index;
  for (auto [i, increment, stride] : llvm::enumerate(indices, strides)) {
    if (stride != 1) { // Skip if stride is 1.
      Value strideValue =
          ShapedType::isDynamic(stride)
              ? convertUnsignedToI32(rewriter, loc,
                                     memRefDescriptor.stride(rewriter, loc, i))
              : rewriter.create<LLVM::ConstantOp>(loc, i32, stride);
      increment = rewriter.create<LLVM::MulOp>(loc, increment, strideValue);
    }
    index =
        index ? rewriter.create<LLVM::AddOp>(loc, index, increment) : increment;
  }
  return index ? index : createI32Constant(rewriter, loc, 0);
}

/// Compute the contents of the `num_records` field for a given memref
/// descriptor - that is, the number of bytes that's one element past the
/// greatest possible valid index into the memref.
static Value getNumRecords(ConversionPatternRewriter &rewriter, Location loc,
                           MemRefType memrefType,
                           MemRefDescriptor &memrefDescriptor,
                           ArrayRef<int64_t> strides,
                           uint32_t elementByteWidth) {
  if (memrefType.hasStaticShape() &&
      !llvm::any_of(strides, ShapedType::isDynamic)) {
    int64_t size = memrefType.getRank() == 0 ? 1 : 0;
    ArrayRef<int64_t> shape = memrefType.getShape();
    for (uint32_t i = 0, e = memrefType.getRank(); i < e; ++i)
      size = std::max(shape[i] * strides[i], size);
    size = size * elementByteWidth;
    assert(size < std::numeric_limits<uint32_t>::max() &&
           "the memref buffer is too large");
    return createI32Constant(rewriter, loc, static_cast<int32_t>(size));
  }
  Value maxIndex;
  for (uint32_t i = 0, e = memrefType.getRank(); i < e; ++i) {
    Value size = memrefDescriptor.size(rewriter, loc, i);
    Value stride = memrefDescriptor.stride(rewriter, loc, i);
    Value maxThisDim = rewriter.create<LLVM::MulOp>(loc, size, stride);
    maxIndex = maxIndex
                   ? rewriter.create<LLVM::UMaxOp>(loc, maxIndex, maxThisDim)
                   : maxThisDim;
  }
  Value maxIndexI32 = convertUnsignedToI32(rewriter, loc, maxIndex);
  Value byteWidthConst = createI32Constant(rewriter, loc, elementByteWidth);
  return rewriter.create<LLVM::MulOp>(loc, maxIndexI32, byteWidthConst);
}

static Value makeBufferRsrc(ConversionPatternRewriter &rewriter, Location loc,
                            Value basePointer, Value numRecords,
                            bool boundsCheck, amdgpu::Chipset chipset,
                            Value cacheSwizzleStride = nullptr,
                            unsigned addressSpace = 8) {
  // The stride value is generally 0. However, on MI-300 and onward, you can
  // enable a cache swizzling mode by setting bit 14 of the stride field
  // and setting that stride to a cache stride.
  Type i16 = rewriter.getI16Type();
  Value stride;
  if (chipset.majorVersion == 9 && chipset >= kGfx942 && cacheSwizzleStride) {
    Value cacheStrideZext =
        rewriter.create<LLVM::ZExtOp>(loc, i16, cacheSwizzleStride);
    Value swizzleBit = rewriter.create<LLVM::ConstantOp>(
        loc, i16, rewriter.getI16IntegerAttr(1 << 14));
    stride = rewriter.create<LLVM::OrOp>(loc, cacheStrideZext, swizzleBit,
                                         /*isDisjoint=*/true);
  } else {
    stride = rewriter.create<LLVM::ConstantOp>(loc, i16,
                                               rewriter.getI16IntegerAttr(0));
  }
  // Get the number of elements.
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
    uint32_t oob = boundsCheck ? 3 : 2;
    flags |= (oob << 28);
  }
  Value flagsConst = createI32Constant(rewriter, loc, flags);
  Type rsrcType =
      LLVM::LLVMPointerType::get(rewriter.getContext(), addressSpace);
  Value resource = rewriter.createOrFold<ROCDL::MakeBufferRsrcOp>(
      loc, rsrcType, basePointer, stride, numRecords, flagsConst);
  return resource;
}

namespace {
struct FatRawBufferCastLowering
    : public ConvertOpToLLVMPattern<FatRawBufferCastOp> {
  FatRawBufferCastLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<FatRawBufferCastOp>(converter),
        chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(FatRawBufferCastOp op, FatRawBufferCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value memRef = adaptor.getSource();
    Value unconvertedMemref = op.getSource();
    MemRefType memrefType = cast<MemRefType>(unconvertedMemref.getType());
    MemRefDescriptor descriptor(memRef);

    DataLayout dataLayout = DataLayout::closest(op);
    int64_t elementByteWidth =
        dataLayout.getTypeSizeInBits(memrefType.getElementType()) / 8;

    int64_t unusedOffset = 0;
    SmallVector<int64_t, 5> strideVals;
    if (failed(memrefType.getStridesAndOffset(strideVals, unusedOffset)))
      return op.emitOpError("Can't lower non-stride-offset memrefs");

    Value numRecords = adaptor.getValidBytes();
    if (!numRecords)
      numRecords = getNumRecords(rewriter, loc, memrefType, descriptor,
                                 strideVals, elementByteWidth);

    Value basePointer =
        adaptor.getResetOffset()
            ? descriptor.bufferPtr(rewriter, loc, *getTypeConverter(),
                                   memrefType)
            : descriptor.alignedPtr(rewriter, loc);

    Value offset = adaptor.getResetOffset()
                       ? rewriter.create<LLVM::ConstantOp>(
                             loc, getIndexType(), rewriter.getIndexAttr(0))
                       : descriptor.offset(rewriter, loc);

    bool hasSizes = memrefType.getRank() > 0;
    // No need to unpack() and pack() all the individual sizes and strides,
    // so we'll just extract the arrays.
    Value sizes = hasSizes ? rewriter.create<LLVM::ExtractValueOp>(
                                 loc, descriptor, kSizePosInMemRefDescriptor)
                           : Value{};
    Value strides = hasSizes
                        ? rewriter.create<LLVM::ExtractValueOp>(
                              loc, descriptor, kStridePosInMemRefDescriptor)
                        : Value{};

    Value fatPtr = makeBufferRsrc(
        rewriter, loc, basePointer, numRecords, adaptor.getBoundsCheck(),
        chipset, adaptor.getCacheSwizzleStride(), /*addressSpace=*/7);

    Value result = MemRefDescriptor::poison(
        rewriter, loc,
        getTypeConverter()->convertType(op.getResult().getType()));
    result = rewriter.create<LLVM::InsertValueOp>(
        loc, result, fatPtr, kAllocatedPtrPosInMemRefDescriptor);
    result = rewriter.create<LLVM::InsertValueOp>(
        loc, result, fatPtr, kAlignedPtrPosInMemRefDescriptor);
    result = rewriter.create<LLVM::InsertValueOp>(loc, result, offset,
                                                  kOffsetPosInMemRefDescriptor);
    if (hasSizes) {
      result = rewriter.create<LLVM::InsertValueOp>(loc, result, sizes,
                                                    kSizePosInMemRefDescriptor);
      result = rewriter.create<LLVM::InsertValueOp>(
          loc, result, strides, kStridePosInMemRefDescriptor);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

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

    // Get the type size in bytes.
    DataLayout dataLayout = DataLayout::closest(gpuOp);
    int64_t elementByteWidth =
        dataLayout.getTypeSizeInBits(memrefType.getElementType()) / 8;
    Value byteWidthConst = createI32Constant(rewriter, loc, elementByteWidth);

    // If we want to load a vector<NxT> with total size <= 32
    // bits, use a scalar load and bitcast it. Similarly, if bitsize(T) < 32
    // and the total load size is >= 32, use a vector load of N / (bitsize(T) /
    // 32) x i32 and bitcast. Also, the CAS intrinsic requires integer operands,
    // so bitcast any floats to integers.
    Type llvmBufferValType = llvmWantedDataType;
    if (atomicCmpData) {
      if (auto floatType = dyn_cast<FloatType>(wantedDataType))
        llvmBufferValType = this->getTypeConverter()->convertType(
            rewriter.getIntegerType(floatType.getWidth()));
    }
    if (auto dataVector = dyn_cast<VectorType>(wantedDataType)) {
      uint32_t vecLen = dataVector.getNumElements();
      uint32_t elemBits =
          dataLayout.getTypeSizeInBits(dataVector.getElementType());
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
    if (auto vecType = dyn_cast<VectorType>(llvmBufferValType)) {
      // Buffer intrinsics doesn't support 1-element vectors, cast them to
      // scalars.
      if (vecType.getNumElements() == 1)
        llvmBufferValType = vecType.getElementType();
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
    if (failed(memrefType.getStridesAndOffset(strides, offset)))
      return gpuOp.emitOpError("Can't lower non-stride-offset memrefs");

    MemRefDescriptor memrefDescriptor(memref);

    Value ptr = memrefDescriptor.bufferPtr(
        rewriter, loc, *this->getTypeConverter(), memrefType);
    Value numRecords = getNumRecords(
        rewriter, loc, memrefType, memrefDescriptor, strides, elementByteWidth);
    Value resource = makeBufferRsrc(rewriter, loc, ptr, numRecords,
                                    adaptor.getBoundsCheck(), chipset);
    args.push_back(resource);

    // Indexing (voffset)
    Value voffset = getLinearIndexI32(rewriter, loc, memrefDescriptor,
                                      adaptor.getIndices(), strides);
    if (std::optional<int32_t> indexOffset = adaptor.getIndexOffset();
        indexOffset && *indexOffset > 0) {
      Value extraOffsetConst = createI32Constant(rewriter, loc, *indexOffset);
      voffset =
          voffset ? rewriter.create<LLVM::AddOp>(loc, voffset, extraOffsetConst)
                  : extraOffsetConst;
    }
    voffset = rewriter.create<LLVM::MulOp>(loc, voffset, byteWidthConst);
    args.push_back(voffset);

    // SGPR offset.
    Value sgprOffset = adaptor.getSgprOffset();
    if (!sgprOffset)
      sgprOffset = createI32Constant(rewriter, loc, 0);
    sgprOffset = rewriter.create<LLVM::MulOp>(loc, sgprOffset, byteWidthConst);
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
  LDSBarrierOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
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
          /*is_align_stack=*/false, LLVM::TailCallKind::None,
          /*asm_dialect=*/asmDialectAttr,
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
      rewriter.create<ROCDL::SWaitcntOp>(loc, ldsOnlyBits);
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
  SchedBarrierOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
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

/// Converts a MFMA vector operand from MLIR AMDGPU dialect convention to ROCDL
/// and LLVM AMDGPU intrinsics convention.
///
/// Specifically:
/// 1. If the element type is bfloat16, bitcast it to i16.
/// 2. If instead we have a more than 64-bit quantity, use a <N / 4 x i32>
/// instead, which is what the f8f6f4 intrinsics use.
/// 3. If `input` is a vector of N <= 8 bytes, bitcast it to a (N * 8)-bit
/// integer.
///
/// Note that the type of `input` has already been LLVM type converted:
/// therefore 8-bit and smaller floats are represented as their corresponding
/// `iN` integers.
static Value convertMFMAVectorOperand(ConversionPatternRewriter &rewriter,
                                      Location loc, Value input) {
  Type inputType = input.getType();
  if (auto vectorType = dyn_cast<VectorType>(inputType)) {
    if (vectorType.getElementType().isBF16())
      return rewriter.create<LLVM::BitcastOp>(
          loc, vectorType.clone(rewriter.getI16Type()), input);
    if (vectorType.getElementType().isInteger(8) &&
        vectorType.getNumElements() <= 8)
      return rewriter.create<LLVM::BitcastOp>(
          loc, rewriter.getIntegerType(vectorType.getNumElements() * 8), input);
    if (isa<IntegerType>(vectorType.getElementType()) &&
        vectorType.getElementTypeBitWidth() <= 8) {
      int64_t numWords = llvm::divideCeil(
          vectorType.getNumElements() * vectorType.getElementTypeBitWidth(),
          32);
      return rewriter.create<LLVM::BitcastOp>(
          loc, VectorType::get(numWords, rewriter.getI32Type()), input);
    }
  }
  return input;
}

/// Converts the scaled MFMA operands, `scalesA` and `scalesB`, from MLIR AMDGPU
/// dialect convention to ROCDL and LLVM AMDGPU intrinsics convention.
///
/// Specifically:
/// 1. If `input` is a i8 value, zero extend it to i32
/// 2. If `input` is a vector of length 4 and type i8, cast it to i32
///
/// Note that the type of `input` has already been LLVM type converted:
/// therefore 8-bit and smaller floats are represented as their corresponding
/// `iN` integers.
static Value castMFMAScaleOperand(ConversionPatternRewriter &rewriter,
                                  Location loc, Value input) {
  Type inputType = input.getType();
  Type outputType = rewriter.getI32Type();
  if (auto intType = dyn_cast<IntegerType>(inputType))
    return rewriter.create<LLVM::ZExtOp>(loc, outputType, input);
  return rewriter.create<LLVM::BitcastOp>(loc, outputType, input);
}

/// Push an input operand. If it is a float type, nothing to do. If it is
/// an integer type, then we need to also push its signdness (1 for signed, 0
/// for unsigned) and we need to pack the input 16xi8 vector into a 4xi32
/// vector (or the 8xi8 vector into a 2xi32 one for gfx12+).
/// We also need to convert bfloat inputs to i16 to account for the bfloat
/// intrinsics having been defined before the AMD backend supported bfloat. We
/// similarly need to pack 8-bit float types into integers as if they were i8
/// (which they are for the backend's purposes).
static void wmmaPushInputOperand(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 const TypeConverter *typeConverter,
                                 bool isUnsigned, Value llvmInput,
                                 Value mlirInput,
                                 SmallVector<Value, 4> &operands) {
  Type inputType = llvmInput.getType();
  auto vectorType = dyn_cast<VectorType>(inputType);
  if (!vectorType) {
    operands.push_back(llvmInput);
    return;
  }
  Type elemType = vectorType.getElementType();

  if (elemType.isBF16())
    llvmInput = rewriter.create<LLVM::BitcastOp>(
        loc, vectorType.clone(rewriter.getI16Type()), llvmInput);
  if (elemType.getIntOrFloatBitWidth() > 8) {
    operands.push_back(llvmInput);
    return;
  }

  // We need to check the type of the input before conversion to properly test
  // for int8. This is because, in LLVM, fp8 type is converted to int8, so the
  // fp8/int8 information is lost during the conversion process.
  auto mlirInputType = cast<VectorType>(mlirInput.getType());
  bool isInputInteger = mlirInputType.getElementType().isInteger();
  if (isInputInteger) {
    // if element type is 8-bit signed or unsigned, ignore the isUnsigned flag
    bool localIsUnsigned = isUnsigned;
    if (elemType.isUnsignedInteger()) {
      localIsUnsigned = true;
    } else if (elemType.isSignedInteger()) {
      localIsUnsigned = false;
    }
    Value sign = createI1Constant(rewriter, loc, !localIsUnsigned);
    operands.push_back(sign);
  }

  int64_t numBits =
      vectorType.getNumElements() * elemType.getIntOrFloatBitWidth();
  Type i32 = rewriter.getI32Type();
  Type intrinsicInType = numBits <= 32
                             ? (Type)rewriter.getIntegerType(numBits)
                             : (Type)VectorType::get(numBits / 32, i32);
  auto llvmIntrinsicInType = typeConverter->convertType(intrinsicInType);
  Value castInput = rewriter.createOrFold<LLVM::BitcastOp>(
      loc, llvmIntrinsicInType, llvmInput);
  // The wave64-mode 16x16x16 intrinsics that take 4-bit integers only need
  // (256 / 64) * 4 = 16 bits of input (on gfx12+) but take i32 arguments.
  // Add in the zeros here.
  if (numBits < 32)
    castInput = rewriter.create<LLVM::ZExtOp>(loc, i32, castInput);
  operands.push_back(castInput);
}

/// Push the output operand. For many cases this is only pushing the output in
/// the operand list. But when we have f16 -> f16 or bf16 -> bf16 intrinsics,
/// since the same numbers of VGPRs is used, we need to decide if to store the
/// result in the upper 16 bits of the VGPRs or in the lower part. To store the
/// result in the lower 16 bits, set subwordOffset to 1, otherwise result will
/// be stored it in the upper part. The subwordOffset must not be set for gfx12,
/// as the instructions have been changed to return fewer registers instead.
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

/// Return true if `type` is the E5M2 variant of an 8-bit float that is
/// supported by the `_bf8` instructions on the given `chipset`.
static bool typeIsExpectedBf8ForChipset(Chipset chipset, Type type) {
  return (chipset == kGfx942 && isa<Float8E5M2FNUZType>(type)) ||
         (hasOcpFp8(chipset) && isa<Float8E5M2Type>(type));
}

/// Return true if `type` is the E4M3FN variant of an 8-bit float that is
/// supported by the `_fp8` instructions on the given `chipset`.
static bool typeIsExpectedFp8ForChipset(Chipset chipset, Type type) {
  return (chipset == kGfx942 && isa<Float8E4M3FNUZType>(type)) ||
         (hasOcpFp8(chipset) && isa<Float8E4M3FNType>(type));
}

/// Return the `rocdl` intrinsic corresponding to a MFMA operation `mfma`
/// if one exists. This includes checking to ensure the intrinsic is supported
/// on the architecture you are compiling for.
static std::optional<StringRef> mfmaOpToIntrinsic(MFMAOp mfma,
                                                  Chipset chipset) {
  uint32_t m = mfma.getM(), n = mfma.getN(), k = mfma.getK(),
           b = mfma.getBlocks();
  Type sourceElem = getElementTypeOrSelf(mfma.getSourceA().getType());
  Type destElem = getElementTypeOrSelf(mfma.getDestC().getType());

  if (sourceElem.isF32() && destElem.isF32()) {
    if (mfma.getReducePrecision() && chipset >= kGfx942) {
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
    if (chipset >= kGfx950) {
      if (m == 32 && n == 32 && k == 16 && b == 1)
        return ROCDL::mfma_f32_32x32x16_f16::getOperationName();
      if (m == 16 && n == 16 && k == 32 && b == 1)
        return ROCDL::mfma_f32_16x16x32_f16::getOperationName();
    }
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

  if (sourceElem.isBF16() && destElem.isF32()) {
    if (chipset >= kGfx950) {
      if (m == 32 && n == 32 && k == 16 && b == 1)
        return ROCDL::mfma_f32_32x32x16_bf16::getOperationName();
      if (m == 16 && n == 16 && k == 32 && b == 1)
        return ROCDL::mfma_f32_16x16x32_bf16::getOperationName();
    }
    if (chipset >= kGfx90a) {
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

  if (sourceElem.isInteger(8) && destElem.isInteger(32)) {
    if (chipset >= kGfx950) {
      if (m == 32 && n == 32 && k == 32 && b == 1)
        return ROCDL::mfma_i32_32x32x32_i8::getOperationName();
      if (m == 16 && n == 16 && k == 64 && b == 1)
        return ROCDL::mfma_i32_16x16x64_i8::getOperationName();
    }
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
    if (m == 32 && n == 32 && k == 16 && b == 1 && chipset >= kGfx942)
      return ROCDL::mfma_i32_32x32x16_i8::getOperationName();
    if (m == 16 && n == 16 && k == 32 && b == 1 && chipset >= kGfx942)
      return ROCDL::mfma_i32_16x16x32_i8::getOperationName();
  }

  if (sourceElem.isF64() && destElem.isF64() && chipset >= kGfx90a) {
    if (m == 16 && n == 16 && k == 4 && b == 1)
      return ROCDL::mfma_f64_16x16x4f64::getOperationName();
    if (m == 4 && n == 4 && k == 4 && b == 4)
      return ROCDL::mfma_f64_4x4x4f64::getOperationName();
  }

  if (destElem.isF32() && typeIsExpectedBf8ForChipset(chipset, sourceElem)) {
    // Known to be correct because there are no scalar f8 instructions and
    // because a length mismatch will have been caught by the verifier.
    Type sourceBElem =
        cast<VectorType>(mfma.getSourceB().getType()).getElementType();
    if (m == 16 && n == 16 && k == 32 && b == 1) {
      if (typeIsExpectedBf8ForChipset(chipset, sourceBElem))
        return ROCDL::mfma_f32_16x16x32_bf8_bf8::getOperationName();
      if (typeIsExpectedFp8ForChipset(chipset, sourceBElem))
        return ROCDL::mfma_f32_16x16x32_bf8_fp8::getOperationName();
    }
    if (m == 32 && n == 32 && k == 16 && b == 1) {
      if (typeIsExpectedBf8ForChipset(chipset, sourceBElem))
        return ROCDL::mfma_f32_32x32x16_bf8_bf8::getOperationName();
      if (typeIsExpectedFp8ForChipset(chipset, sourceBElem))
        return ROCDL::mfma_f32_32x32x16_bf8_fp8::getOperationName();
    }
  }

  if (destElem.isF32() && typeIsExpectedFp8ForChipset(chipset, sourceElem)) {
    Type sourceBElem =
        cast<VectorType>(mfma.getSourceB().getType()).getElementType();
    if (m == 16 && n == 16 && k == 32 && b == 1) {
      if (typeIsExpectedBf8ForChipset(chipset, sourceBElem))
        return ROCDL::mfma_f32_16x16x32_fp8_bf8::getOperationName();
      if (typeIsExpectedFp8ForChipset(chipset, sourceBElem))
        return ROCDL::mfma_f32_16x16x32_fp8_fp8::getOperationName();
    }
    if (m == 32 && n == 32 && k == 16 && b == 1) {
      if (typeIsExpectedBf8ForChipset(chipset, sourceBElem))
        return ROCDL::mfma_f32_32x32x16_fp8_bf8::getOperationName();
      if (typeIsExpectedFp8ForChipset(chipset, sourceBElem))
        return ROCDL::mfma_f32_32x32x16_fp8_fp8::getOperationName();
    }
  }

  return std::nullopt;
}

static std::optional<uint32_t> mfmaTypeSelectCode(Type mlirElemType) {
  return llvm::TypeSwitch<Type, std::optional<uint32_t>>(mlirElemType)
      .Case([](Float8E4M3FNType) { return 0u; })
      .Case([](Float8E5M2Type) { return 1u; })
      .Case([](Float6E2M3FNType) { return 2u; })
      .Case([](Float6E3M2FNType) { return 3u; })
      .Case([](Float4E2M1FNType) { return 4u; })
      .Default([](Type) { return std::nullopt; });
}

/// If there is a scaled MFMA instruction for the input element types `aType`
/// and `bType`, output type `destType`, problem size M, N, K, and B (number of
/// blocks) on the given `chipset`, return a tuple consisting of the
/// OperationName of the intrinsic and the type codes that need to be passed to
/// that intrinsic. Note that this is also used to implement some un-scaled
/// MFMAs, since the compiler represents the ordinary instruction as a "scaled"
/// MFMA with a scale of 0.
static std::optional<std::tuple<StringRef, uint32_t, uint32_t>>
mfmaOpToScaledIntrinsic(Type aType, Type bType, Type destType, uint32_t m,
                        uint32_t n, uint32_t k, uint32_t b, Chipset chipset) {
  aType = getElementTypeOrSelf(aType);
  bType = getElementTypeOrSelf(bType);
  destType = getElementTypeOrSelf(destType);

  if (chipset < kGfx950)
    return std::nullopt;
  if (!isa<Float32Type>(destType))
    return std::nullopt;

  std::optional<uint32_t> aTypeCode = mfmaTypeSelectCode(aType);
  std::optional<uint32_t> bTypeCode = mfmaTypeSelectCode(bType);
  if (!aTypeCode || !bTypeCode)
    return std::nullopt;

  if (m == 32 && n == 32 && k == 64 && b == 1)
    return std::tuple{ROCDL::mfma_scale_f32_32x32x64_f8f6f4::getOperationName(),
                      *aTypeCode, *bTypeCode};
  if (m == 16 && n == 16 && k == 128 && b == 1)
    return std::tuple{
        ROCDL::mfma_scale_f32_16x16x128_f8f6f4::getOperationName(), *aTypeCode,
        *bTypeCode};

  return std::nullopt;
}

static std::optional<std::tuple<StringRef, uint32_t, uint32_t>>
mfmaOpToScaledIntrinsic(MFMAOp mfma, Chipset chipset) {
  return mfmaOpToScaledIntrinsic(
      mfma.getSourceA().getType(), mfma.getSourceB().getType(),
      mfma.getDestC().getType(), mfma.getM(), mfma.getN(), mfma.getK(),
      mfma.getBlocks(), chipset);
}

static std::optional<std::tuple<StringRef, uint32_t, uint32_t>>
mfmaOpToScaledIntrinsic(ScaledMFMAOp smfma, Chipset chipset) {
  return mfmaOpToScaledIntrinsic(smfma.getSourceA().getType(),
                                 smfma.getSourceB().getType(),
                                 smfma.getDestC().getType(), smfma.getM(),
                                 smfma.getN(), smfma.getK(), 1u, chipset);
}

/// Return the `rocdl` intrinsic corresponding to a WMMA operation `wmma`
/// if one exists. This includes checking to ensure the intrinsic is supported
/// on the architecture you are compiling for.
static std::optional<StringRef> wmmaOpToIntrinsic(WMMAOp wmma,
                                                  Chipset chipset) {
  auto sourceVectorType = dyn_cast<VectorType>(wmma.getSourceA().getType());
  auto sourceBVectorType = dyn_cast<VectorType>(wmma.getSourceB().getType());
  auto destVectorType = dyn_cast<VectorType>(wmma.getDestC().getType());
  auto elemSourceType = sourceVectorType.getElementType();
  auto elemBSourceType = sourceBVectorType.getElementType();
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
  if (chipset.majorVersion == 11) {
    if (elemSourceType.isInteger(4) && elemDestType.isInteger(32))
      return ROCDL::wmma_i32_16x16x16_iu4::getOperationName();
  }
  if (chipset.majorVersion >= 12) {
    if (isa<Float8E4M3FNType>(elemSourceType) &&
        isa<Float8E4M3FNType>(elemBSourceType) && elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x16_fp8_fp8::getOperationName();
    if (isa<Float8E4M3FNType>(elemSourceType) &&
        isa<Float8E5M2Type>(elemBSourceType) && elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x16_fp8_bf8::getOperationName();
    if (isa<Float8E5M2Type>(elemSourceType) &&
        isa<Float8E5M2Type>(elemBSourceType) && elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x16_bf8_bf8::getOperationName();
    if (isa<Float8E5M2Type>(elemSourceType) &&
        isa<Float8E4M3FNType>(elemBSourceType) && elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x16_bf8_fp8::getOperationName();
    if (elemSourceType.isInteger(4) && elemDestType.isInteger(32)) {
      bool isWave64 = destVectorType.getNumElements() == 4;
      // This is the ambiguous case. 8 inputs to the wave64 version means that
      // we want the 16x16x32 version, but for wave32 they mean the short form.
      bool has8Inputs = sourceVectorType.getNumElements() == 8;
      if ((isWave64 && has8Inputs) || (!isWave64 && !has8Inputs))
        return ROCDL::wmma_i32_16x16x32_iu4::getOperationName();
      return ROCDL::wmma_i32_16x16x16_iu4::getOperationName();
    }
  }
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
      if (chipset < kGfx942)
        return op.emitOpError("negation unsupported on older than gfx942");
      getBlgpField |=
          op.getNegateA() | (op.getNegateB() << 1) | (op.getNegateC() << 2);
    }
    std::optional<StringRef> maybeIntrinsic = mfmaOpToIntrinsic(op, chipset);
    std::optional<std::tuple<StringRef, uint32_t, uint32_t>>
        maybeScaledIntrinsic = mfmaOpToScaledIntrinsic(op, chipset);
    if (!maybeIntrinsic.has_value() && !maybeScaledIntrinsic.has_value())
      return op.emitOpError("no intrinsic matching MFMA size on given chipset");

    bool isScaled =
        !maybeIntrinsic.has_value() && maybeScaledIntrinsic.has_value();
    if (isScaled &&
        (adaptor.getAbid() > 0 || getBlgpField > 0 || op.getCbsz() > 0)) {
      return op.emitOpError(
          "non-default abid, blgp, and cbsz aren't supported on MFMAs that can "
          "be scaled as those fields are used for type information");
    }

    StringRef intrinsicName =
        isScaled ? std::get<0>(*maybeScaledIntrinsic) : *maybeIntrinsic;
    OperationState loweredOp(loc, intrinsicName);
    loweredOp.addTypes(intrinsicOutType);
    loweredOp.addOperands(
        {convertMFMAVectorOperand(rewriter, loc, adaptor.getSourceA()),
         convertMFMAVectorOperand(rewriter, loc, adaptor.getSourceB()),
         adaptor.getDestC()});
    if (isScaled) {
      Value zero = createI32Constant(rewriter, loc, 0);
      auto [_scaledName, aTypeCode, bTypeCode] = *maybeScaledIntrinsic;
      loweredOp.addOperands({createI32Constant(rewriter, loc, aTypeCode),
                             createI32Constant(rewriter, loc, bTypeCode),
                             /*scale A byte=*/zero, /*scale A=*/zero,
                             /*scale B byte=*/zero, /*scale B=*/zero});
    } else {
      loweredOp.addOperands({createI32Constant(rewriter, loc, op.getCbsz()),
                             createI32Constant(rewriter, loc, op.getAbid()),
                             createI32Constant(rewriter, loc, getBlgpField)});
    };
    Value lowered = rewriter.create(loweredOp)->getResult(0);
    if (outType != intrinsicOutType)
      lowered = rewriter.create<LLVM::BitcastOp>(loc, outType, lowered);
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

struct ScaledMFMAOpLowering : public ConvertOpToLLVMPattern<ScaledMFMAOp> {
  ScaledMFMAOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(ScaledMFMAOp op, ScaledMFMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type intrinsicOutType = typeConverter->convertType(op.getDestD().getType());

    if (chipset.majorVersion != 9 || chipset < kGfx950)
      return op->emitOpError("scaled MFMA only supported on gfx908+");
    std::optional<std::tuple<StringRef, uint32_t, uint32_t>>
        maybeScaledIntrinsic = mfmaOpToScaledIntrinsic(op, chipset);
    if (!maybeScaledIntrinsic.has_value())
      return op.emitOpError(
          "no intrinsic matching scaled MFMA size on given chipset");

    auto [intrinsicName, aTypeCode, bTypeCode] = *maybeScaledIntrinsic;
    OperationState loweredOp(loc, intrinsicName);
    loweredOp.addTypes(intrinsicOutType);
    loweredOp.addOperands(
        {convertMFMAVectorOperand(rewriter, loc, adaptor.getSourceA()),
         convertMFMAVectorOperand(rewriter, loc, adaptor.getSourceB()),
         adaptor.getDestC()});
    Value scalesIdxA =
        createI32Constant(rewriter, loc, adaptor.getScalesIdxA());
    Value scalesIdxB =
        createI32Constant(rewriter, loc, adaptor.getScalesIdxB());
    loweredOp.addOperands(
        {createI32Constant(rewriter, loc, aTypeCode),
         createI32Constant(rewriter, loc, bTypeCode),
         /*scales idx A=*/scalesIdxA,
         /*scales A*/
         castMFMAScaleOperand(rewriter, loc, adaptor.getScalesA()),
         /*scales idx B=*/scalesIdxB,
         /*scales B*/
         castMFMAScaleOperand(rewriter, loc, adaptor.getScalesB())});
    Value lowered = rewriter.create(loweredOp)->getResult(0);
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

    if (chipset.majorVersion >= 12 && op.getSubwordOffset() != 0)
      return op.emitOpError("subwordOffset not supported on gfx12+");

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

struct GatherToLDSOpLowering : public ConvertOpToLLVMPattern<GatherToLDSOp> {
  GatherToLDSOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<GatherToLDSOp>(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(GatherToLDSOp op, GatherToLDSOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (chipset.majorVersion < 9 || chipset.majorVersion > 10)
      return op.emitOpError("pre-gfx9 and post-gfx10 not supported");

    Location loc = op.getLoc();

    auto srcMemRefType = cast<MemRefType>(op.getSrc().getType());
    auto dstMemRefType = cast<MemRefType>(op.getDst().getType());

    // TODO: instead of only transfering one element per thread, we could
    // augment it to transfer multiple elements per thread by issuing multiple
    // `global_load_lds` instructions.
    Type transferType = op.getTransferType();
    size_t loadWidth = [&]() -> size_t {
      if (auto transferVectorType = dyn_cast<VectorType>(transferType)) {
        return transferVectorType.getNumElements() *
               (transferVectorType.getElementTypeBitWidth() / 8);
      }
      return transferType.getIntOrFloatBitWidth() / 8;
    }();

    // Currently only 1, 2, and 4 byte loads are supported.
    if (loadWidth != 1 && loadWidth != 2 && loadWidth != 4)
      return op.emitOpError("chipset unsupported element size");

    Value srcPtr =
        getStridedElementPtr(rewriter, loc, srcMemRefType, adaptor.getSrc(),
                             (adaptor.getSrcIndices()));
    Value dstPtr =
        getStridedElementPtr(rewriter, loc, dstMemRefType, adaptor.getDst(),
                             (adaptor.getDstIndices()));

    rewriter.replaceOpWithNewOp<ROCDL::LoadToLDSOp>(
        op, srcPtr, dstPtr, rewriter.getI32IntegerAttr(loadWidth),
        /*offset=*/rewriter.getI32IntegerAttr(0),
        /*aux=*/rewriter.getI32IntegerAttr(0), ArrayAttr{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

namespace {
struct ExtPackedFp8OpLowering final
    : public ConvertOpToLLVMPattern<ExtPackedFp8Op> {
  ExtPackedFp8OpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<amdgpu::ExtPackedFp8Op>(converter),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(ExtPackedFp8Op op, ExtPackedFp8OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct PackedTrunc2xFp8OpLowering final
    : public ConvertOpToLLVMPattern<PackedTrunc2xFp8Op> {
  PackedTrunc2xFp8OpLowering(const LLVMTypeConverter &converter,
                             Chipset chipset)
      : ConvertOpToLLVMPattern<amdgpu::PackedTrunc2xFp8Op>(converter),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(PackedTrunc2xFp8Op op, PackedTrunc2xFp8OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct PackedStochRoundFp8OpLowering final
    : public ConvertOpToLLVMPattern<PackedStochRoundFp8Op> {
  PackedStochRoundFp8OpLowering(const LLVMTypeConverter &converter,
                                Chipset chipset)
      : ConvertOpToLLVMPattern<amdgpu::PackedStochRoundFp8Op>(converter),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(PackedStochRoundFp8Op op,
                  PackedStochRoundFp8OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ScaledExtPackedOpLowering final
    : public ConvertOpToLLVMPattern<ScaledExtPackedOp> {
  ScaledExtPackedOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<amdgpu::ScaledExtPackedOp>(converter),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(ScaledExtPackedOp op, ScaledExtPackedOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct PackedScaledTruncOpLowering final
    : public ConvertOpToLLVMPattern<PackedScaledTruncOp> {
  PackedScaledTruncOpLowering(const LLVMTypeConverter &converter,
                              Chipset chipset)
      : ConvertOpToLLVMPattern<amdgpu::PackedScaledTruncOp>(converter),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(PackedScaledTruncOp op, PackedScaledTruncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // end namespace

LogicalResult ExtPackedFp8OpLowering::matchAndRewrite(
    ExtPackedFp8Op op, ExtPackedFp8OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  if (!(chipset == kGfx942 || hasOcpFp8(chipset)))
    return rewriter.notifyMatchFailure(
        loc, "Fp8 conversion instructions are not available on target "
             "architecture and their emulation is not implemented");
  Type v4i8 =
      getTypeConverter()->convertType(VectorType::get(4, rewriter.getI8Type()));
  Type i32 = getTypeConverter()->convertType(rewriter.getI32Type());
  Type f32 = getTypeConverter()->convertType(op.getResult().getType());

  Value source = adaptor.getSource();
  auto sourceVecType = dyn_cast<VectorType>(op.getSource().getType());
  auto resultVecType = dyn_cast<VectorType>(op.getResult().getType());
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
  if (resultVecType) {
    if (typeIsExpectedBf8ForChipset(chipset, sourceElemType)) {
      rewriter.replaceOpWithNewOp<ROCDL::CvtPkF32Bf8Op>(op, f32, i32Source,
                                                        op.getIndex());
    } else if (typeIsExpectedFp8ForChipset(chipset, sourceElemType)) {
      rewriter.replaceOpWithNewOp<ROCDL::CvtPkF32Fp8Op>(op, f32, i32Source,
                                                        op.getIndex());
    }
  } else {
    if (typeIsExpectedBf8ForChipset(chipset, sourceElemType)) {
      rewriter.replaceOpWithNewOp<ROCDL::CvtF32Bf8Op>(op, f32, i32Source,
                                                      op.getIndex());
    } else if (typeIsExpectedFp8ForChipset(chipset, sourceElemType)) {
      rewriter.replaceOpWithNewOp<ROCDL::CvtF32Fp8Op>(op, f32, i32Source,
                                                      op.getIndex());
    }
  }
  return success();
}

LogicalResult ScaledExtPackedOpLowering::matchAndRewrite(
    ScaledExtPackedOp op, ScaledExtPackedOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  if (chipset != kGfx950)
    return rewriter.notifyMatchFailure(
        loc, "Scaled fp conversion instructions are not available on target "
             "architecture and their emulation is not implemented");
  Type i32 = getTypeConverter()->convertType(rewriter.getI32Type());

  Value source = adaptor.getSource();
  Value scale = adaptor.getScale();

  VectorType sourceVecType = cast<VectorType>(op.getSource().getType());
  Type sourceElemType = sourceVecType.getElementType();
  VectorType destVecType = cast<VectorType>(op.getResult().getType());
  Type destElemType = destVecType.getElementType();

  VectorType packedVecType;
  if (isa<Float8E5M2Type, Float8E4M3FNType>(sourceElemType)) {
    VectorType v4i8 = VectorType::get(4, rewriter.getI8Type());
    packedVecType = cast<VectorType>(getTypeConverter()->convertType(v4i8));
  } else if (isa<Float4E2M1FNType>(sourceElemType)) {
    VectorType v8i4 = VectorType::get(8, rewriter.getI4Type());
    packedVecType = cast<VectorType>(getTypeConverter()->convertType(v8i4));
  } else {
    llvm_unreachable("invalid element type for scaled ext");
  }

  // Extend to a packedVectorType
  if (sourceVecType.getNumElements() < packedVecType.getNumElements()) {
    Value longVec = rewriter.create<LLVM::ZeroOp>(loc, packedVecType);
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

  if (isa<Float8E5M2Type>(sourceElemType) && destElemType.isF32())
    rewriter.replaceOpWithNewOp<ROCDL::CvtScaleF32PkF32Bf8Op>(
        op, destVecType, i32Source, scale, op.getIndex());
  else if (isa<Float8E5M2Type>(sourceElemType) && destElemType.isF16())
    rewriter.replaceOpWithNewOp<ROCDL::CvtScaleF32PkF16Bf8Op>(
        op, destVecType, i32Source, scale, op.getIndex());
  else if (isa<Float8E5M2Type>(sourceElemType) && destElemType.isBF16())
    rewriter.replaceOpWithNewOp<ROCDL::CvtScaleF32PkBf16Bf8Op>(
        op, destVecType, i32Source, scale, op.getIndex());
  else if (isa<Float8E4M3FNType>(sourceElemType) && destElemType.isF32())
    rewriter.replaceOpWithNewOp<ROCDL::CvtScaleF32PkF32Fp8Op>(
        op, destVecType, i32Source, scale, op.getIndex());
  else if (isa<Float8E4M3FNType>(sourceElemType) && destElemType.isF16())
    rewriter.replaceOpWithNewOp<ROCDL::CvtScaleF32PkF16Fp8Op>(
        op, destVecType, i32Source, scale, op.getIndex());
  else if (isa<Float8E4M3FNType>(sourceElemType) && destElemType.isBF16())
    rewriter.replaceOpWithNewOp<ROCDL::CvtScaleF32PkBf16Fp8Op>(
        op, destVecType, i32Source, scale, op.getIndex());
  else if (isa<Float4E2M1FNType>(sourceElemType) && destElemType.isF32())
    rewriter.replaceOpWithNewOp<ROCDL::CvtScaleF32PkF32Fp4Op>(
        op, destVecType, i32Source, scale, op.getIndex());
  else if (isa<Float4E2M1FNType>(sourceElemType) && destElemType.isF16())
    rewriter.replaceOpWithNewOp<ROCDL::CvtScaleF32PkF16Fp4Op>(
        op, destVecType, i32Source, scale, op.getIndex());
  else if (isa<Float4E2M1FNType>(sourceElemType) && destElemType.isBF16())
    rewriter.replaceOpWithNewOp<ROCDL::CvtScaleF32PkBf16Fp4Op>(
        op, destVecType, i32Source, scale, op.getIndex());
  else
    return failure();

  return success();
}

LogicalResult PackedScaledTruncOpLowering::matchAndRewrite(
    PackedScaledTruncOp op, PackedScaledTruncOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  if (chipset != kGfx950)
    return rewriter.notifyMatchFailure(
        loc, "Scaled fp conversion instructions are not available on target "
             "architecture and their emulation is not implemented");
  Type v2i16 = getTypeConverter()->convertType(
      VectorType::get(2, rewriter.getI16Type()));
  Type i32 = getTypeConverter()->convertType(rewriter.getI32Type());

  Type resultType = op.getResult().getType();
  Type resultElemType = getElementTypeOrSelf(resultType);
  VectorType sourceVecType = cast<VectorType>(op.getSource().getType());
  Type sourceElemType = sourceVecType.getElementType();

  Type intResultType = isa<Float4E2M1FNType>(resultElemType) ? i32 : v2i16;

  Value source = adaptor.getSource();
  Value scale = adaptor.getScale();
  Value existing = adaptor.getExisting();
  if (existing)
    existing = rewriter.create<LLVM::BitcastOp>(loc, intResultType, existing);
  else
    existing = rewriter.create<LLVM::ZeroOp>(loc, intResultType);

  if (sourceVecType.getNumElements() < 2) {
    Value c0 = createI32Constant(rewriter, loc, 0);
    Value elem0 = rewriter.create<LLVM::ExtractElementOp>(loc, source, c0);
    VectorType v2 = VectorType::get(2, sourceElemType);
    source = rewriter.create<LLVM::ZeroOp>(loc, v2);
    source = rewriter.create<LLVM::InsertElementOp>(loc, source, elem0, c0);
  }

  Value sourceA, sourceB;
  if (sourceElemType.isF32()) {
    Value c0 = createI32Constant(rewriter, loc, 0);
    Value c1 = createI32Constant(rewriter, loc, 1);
    sourceA = rewriter.create<LLVM::ExtractElementOp>(loc, source, c0);
    sourceB = rewriter.create<LLVM::ExtractElementOp>(loc, source, c1);
  }

  Value result;
  if (sourceElemType.isF32() && isa<Float8E5M2Type>(resultElemType))
    result = rewriter.create<ROCDL::CvtScaleF32PkBf8F32Op>(
        loc, intResultType, existing, sourceA, sourceB, scale, op.getIndex());
  else if (sourceElemType.isF16() && isa<Float8E5M2Type>(resultElemType))
    result = rewriter.create<ROCDL::CvtScaleF32PkBf8F16Op>(
        loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isBF16() && isa<Float8E5M2Type>(resultElemType))
    result = rewriter.create<ROCDL::CvtScaleF32PkBf8Bf16Op>(
        loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isF32() && isa<Float8E4M3FNType>(resultElemType))
    result = rewriter.create<ROCDL::CvtScaleF32PkFp8F32Op>(
        loc, intResultType, existing, sourceA, sourceB, scale, op.getIndex());
  else if (sourceElemType.isF16() && isa<Float8E4M3FNType>(resultElemType))
    result = rewriter.create<ROCDL::CvtScaleF32PkFp8F16Op>(
        loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isBF16() && isa<Float8E4M3FNType>(resultElemType))
    result = rewriter.create<ROCDL::CvtScaleF32PkFp8Bf16Op>(
        loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isF32() && isa<Float4E2M1FNType>(resultElemType))
    result = rewriter.create<ROCDL::CvtScaleF32PkFp4F32Op>(
        loc, intResultType, existing, sourceA, sourceB, scale, op.getIndex());
  else if (sourceElemType.isF16() && isa<Float4E2M1FNType>(resultElemType))
    result = rewriter.create<ROCDL::CvtScaleF32PkFp4F16Op>(
        loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isBF16() && isa<Float4E2M1FNType>(resultElemType))
    result = rewriter.create<ROCDL::CvtScaleF32PkFp4Bf16Op>(
        loc, intResultType, existing, source, scale, op.getIndex());
  else
    return failure();

  result = rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
      op, getTypeConverter()->convertType(resultType), result);
  return success();
}

LogicalResult PackedTrunc2xFp8OpLowering::matchAndRewrite(
    PackedTrunc2xFp8Op op, PackedTrunc2xFp8OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  if (!(chipset == kGfx942 || hasOcpFp8(chipset)))
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

  Value result;
  if (typeIsExpectedBf8ForChipset(chipset, resultElemType))
    result = rewriter.create<ROCDL::CvtPkBf8F32Op>(loc, i32, sourceA, sourceB,
                                                   existing, op.getWordIndex());
  else if (typeIsExpectedFp8ForChipset(chipset, resultElemType))
    result = rewriter.create<ROCDL::CvtPkFp8F32Op>(loc, i32, sourceA, sourceB,
                                                   existing, op.getWordIndex());

  result = rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
      op, getTypeConverter()->convertType(resultType), result);
  return success();
}

LogicalResult PackedStochRoundFp8OpLowering::matchAndRewrite(
    PackedStochRoundFp8Op op, PackedStochRoundFp8OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  if (!(chipset == kGfx942 || hasOcpFp8(chipset)))
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

  Value result;
  if (typeIsExpectedBf8ForChipset(chipset, resultElemType))
    result = rewriter.create<ROCDL::CvtSrBf8F32Op>(
        loc, i32, source, stoch, existing, op.getStoreIndex());
  else if (typeIsExpectedFp8ForChipset(chipset, resultElemType))
    result = rewriter.create<ROCDL::CvtSrFp8F32Op>(
        loc, i32, source, stoch, existing, op.getStoreIndex());

  result = rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(
      op, getTypeConverter()->convertType(resultType), result);
  return success();
}

// Implement the AMDGPU_DPPLowering class that will convert the amdgpu.dpp
// operation into the corresponding ROCDL instructions.
struct AMDGPUDPPLowering : public ConvertOpToLLVMPattern<DPPOp> {
  AMDGPUDPPLowering(const LLVMTypeConverter &converter, Chipset chipset)
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

struct AMDGPUSwizzleBitModeLowering
    : public ConvertOpToLLVMPattern<SwizzleBitModeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SwizzleBitModeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    Value src = adaptor.getSrc();
    SmallVector<Value> decomposed =
        LLVM::decomposeValue(rewriter, loc, src, i32);
    unsigned andMask = op.getAndMask();
    unsigned orMask = op.getOrMask();
    unsigned xorMask = op.getXorMask();

    // bit 15 is 0 for the BitMode swizzle.
    // https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/
    unsigned mask = andMask | (orMask << 5) | (xorMask << 10);
    Value maskValue = createI32Constant(rewriter, loc, mask);
    SmallVector<Value> swizzled;
    for (Value v : decomposed) {
      Value res =
          rewriter.create<ROCDL::DsSwizzleOp>(loc, v.getType(), v, maskValue);
      swizzled.emplace_back(res);
    }

    Value result = LLVM::composeValue(rewriter, loc, swizzled, src.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertAMDGPUToROCDLPass
    : public impl::ConvertAMDGPUToROCDLPassBase<ConvertAMDGPUToROCDLPass> {
  using Base::Base;

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

void mlir::populateAMDGPUMemorySpaceAttributeConversions(
    TypeConverter &typeConverter) {
  typeConverter.addTypeAttributeConversion(
      [](BaseMemRefType type, amdgpu::AddressSpaceAttr as)
          -> TypeConverter::AttributeConversionResult {
        MLIRContext *ctx = as.getContext();
        Type i64 = IntegerType::get(ctx, 64);
        switch (as.getValue()) {
        case amdgpu::AddressSpace::FatRawBuffer:
          return IntegerAttr::get(i64, 7);
        case amdgpu::AddressSpace::BufferRsrc:
          return IntegerAttr::get(i64, 8);
        case amdgpu::AddressSpace::FatStructuredBuffer:
          return IntegerAttr::get(i64, 9);
        }
        return TypeConverter::AttributeConversionResult::abort();
      });
}

void mlir::populateAMDGPUToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   Chipset chipset) {
  populateAMDGPUMemorySpaceAttributeConversions(converter);
  patterns
      .add<FatRawBufferCastLowering,
           RawBufferOpLowering<RawBufferLoadOp, ROCDL::RawPtrBufferLoadOp>,
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
           MFMAOpLowering, ScaledMFMAOpLowering, WMMAOpLowering,
           ExtPackedFp8OpLowering, ScaledExtPackedOpLowering,
           PackedScaledTruncOpLowering, PackedTrunc2xFp8OpLowering,
           PackedStochRoundFp8OpLowering, GatherToLDSOpLowering>(converter,
                                                                 chipset);
  patterns.add<AMDGPUSwizzleBitModeLowering>(converter);
}
