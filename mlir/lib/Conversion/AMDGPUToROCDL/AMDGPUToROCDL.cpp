//===- AMDGPUToROCDL.cpp - AMDGPU to ROCDL dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
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
constexpr Chipset kGfx1250 = Chipset(12, 5, 0);

/// Convert an unsigned number `val` to i32.
static Value convertUnsignedToI32(ConversionPatternRewriter &rewriter,
                                  Location loc, Value val) {
  IntegerType i32 = rewriter.getI32Type();
  // Force check that `val` is of int type.
  auto valTy = cast<IntegerType>(val.getType());
  if (i32 == valTy)
    return val;
  return valTy.getWidth() > 32
             ? Value(LLVM::TruncOp::create(rewriter, loc, i32, val))
             : Value(LLVM::ZExtOp::create(rewriter, loc, i32, val));
}

static Value createI32Constant(ConversionPatternRewriter &rewriter,
                               Location loc, int32_t value) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), value);
}

/// Convert an unsigned number `val` to i64.
static Value convertUnsignedToI64(ConversionPatternRewriter &rewriter,
                                  Location loc, Value val) {
  IntegerType i64 = rewriter.getI64Type();
  // Force check that `val` is of int type.
  auto valTy = cast<IntegerType>(val.getType());
  if (i64 == valTy)
    return val;
  return valTy.getWidth() > 64
             ? Value(LLVM::TruncOp::create(rewriter, loc, i64, val))
             : Value(LLVM::ZExtOp::create(rewriter, loc, i64, val));
}

static Value createI64Constant(ConversionPatternRewriter &rewriter,
                               Location loc, int64_t value) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), value);
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
              : LLVM::ConstantOp::create(rewriter, loc, i32, stride);
      increment = LLVM::MulOp::create(rewriter, loc, increment, strideValue);
    }
    index = index ? LLVM::AddOp::create(rewriter, loc, index, increment)
                  : increment;
  }
  return index ? index : createI32Constant(rewriter, loc, 0);
}

/// Compute the contents of the `num_records` field for a given memref
/// descriptor - that is, the number of bytes that's one element past the
/// greatest possible valid index into the memref.
static Value getNumRecords(ConversionPatternRewriter &rewriter, Location loc,
                           MemRefType memrefType,
                           MemRefDescriptor &memrefDescriptor,
                           ArrayRef<int64_t> strides, int64_t elementByteWidth,
                           amdgpu::Chipset chipset, bool boundsCheck) {
  if (chipset >= kGfx1250 && !boundsCheck) {
    constexpr int64_t first45bits = (1ll << 45) - 1;
    return createI64Constant(rewriter, loc, first45bits);
  }
  if (memrefType.hasStaticShape() &&
      !llvm::any_of(strides, ShapedType::isDynamic)) {
    int64_t size = memrefType.getRank() == 0 ? 1 : 0;
    ArrayRef<int64_t> shape = memrefType.getShape();
    for (uint32_t i = 0, e = memrefType.getRank(); i < e; ++i)
      size = std::max(shape[i] * strides[i], size);
    size = size * elementByteWidth;
    return createI64Constant(rewriter, loc, size);
  }
  Value maxIndex;
  for (uint32_t i = 0, e = memrefType.getRank(); i < e; ++i) {
    Value size = memrefDescriptor.size(rewriter, loc, i);
    Value stride = memrefDescriptor.stride(rewriter, loc, i);
    Value maxThisDim = LLVM::MulOp::create(rewriter, loc, size, stride);
    maxIndex = maxIndex
                   ? LLVM::UMaxOp::create(rewriter, loc, maxIndex, maxThisDim)
                   : maxThisDim;
  }
  Value maxIndexI64 = convertUnsignedToI64(rewriter, loc, maxIndex);
  Value byteWidthConst = createI64Constant(rewriter, loc, elementByteWidth);
  return LLVM::MulOp::create(rewriter, loc, maxIndexI64, byteWidthConst);
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
        LLVM::ZExtOp::create(rewriter, loc, i16, cacheSwizzleStride);
    Value swizzleBit = LLVM::ConstantOp::create(
        rewriter, loc, i16, rewriter.getI16IntegerAttr(1 << 14));
    stride = LLVM::OrOp::create(rewriter, loc, cacheStrideZext, swizzleBit,
                                /*isDisjoint=*/true);
  } else {
    stride = LLVM::ConstantOp::create(rewriter, loc, i16,
                                      rewriter.getI16IntegerAttr(0));
  }

  uint32_t flags = 0;
  if (chipset >= kGfx1250) {
    // Flag word:
    // bit 0: swizzle
    // bit 1: 0 means (total_offset + payload > numRecords)
    //        1 means ((total_offset + payload >) numRecords) || ((offset +
    //        payload) > stride) only applied when swizzle_enable = 0. keep at
    //        zero.
    //        whether oob is done depends on numRecords.
    // bits 2-3: Type (must be 0)
  } else {
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
    flags |= (7 << 12) | (4 << 15);
    if (chipset.majorVersion >= 10) {
      flags |= (1 << 24);
      uint32_t oob = boundsCheck ? 3 : 2;
      flags |= (oob << 28);
    }
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
      numRecords =
          getNumRecords(rewriter, loc, memrefType, descriptor, strideVals,
                        elementByteWidth, chipset, adaptor.getBoundsCheck());

    Value basePointer =
        adaptor.getResetOffset()
            ? descriptor.bufferPtr(rewriter, loc, *getTypeConverter(),
                                   memrefType)
            : descriptor.alignedPtr(rewriter, loc);

    Value offset = adaptor.getResetOffset()
                       ? LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                                  rewriter.getIndexAttr(0))
                       : descriptor.offset(rewriter, loc);

    bool hasSizes = memrefType.getRank() > 0;
    // No need to unpack() and pack() all the individual sizes and strides,
    // so we'll just extract the arrays.
    Value sizes = hasSizes
                      ? LLVM::ExtractValueOp::create(rewriter, loc, descriptor,
                                                     kSizePosInMemRefDescriptor)
                      : Value{};
    Value strides =
        hasSizes ? LLVM::ExtractValueOp::create(rewriter, loc, descriptor,
                                                kStridePosInMemRefDescriptor)
                 : Value{};

    Value fatPtr = makeBufferRsrc(
        rewriter, loc, basePointer, numRecords, adaptor.getBoundsCheck(),
        chipset, adaptor.getCacheSwizzleStride(), /*addressSpace=*/7);

    Value result = MemRefDescriptor::poison(
        rewriter, loc,
        getTypeConverter()->convertType(op.getResult().getType()));
    SmallVector<int64_t> pos{kAllocatedPtrPosInMemRefDescriptor};
    result = LLVM::InsertValueOp::create(rewriter, loc, result, fatPtr, pos);
    result = LLVM::InsertValueOp::create(rewriter, loc, result, fatPtr,
                                         kAlignedPtrPosInMemRefDescriptor);
    result = LLVM::InsertValueOp::create(rewriter, loc, result, offset,
                                         kOffsetPosInMemRefDescriptor);
    if (hasSizes) {
      result = LLVM::InsertValueOp::create(rewriter, loc, result, sizes,
                                           kSizePosInMemRefDescriptor);
      result = LLVM::InsertValueOp::create(rewriter, loc, result, strides,
                                           kStridePosInMemRefDescriptor);
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
        Value castForStore = LLVM::BitcastOp::create(
            rewriter, loc, llvmBufferValType, storeData);
        args.push_back(castForStore);
      } else {
        args.push_back(storeData);
      }
    }

    if (atomicCmpData) {
      if (llvmBufferValType != llvmWantedDataType) {
        Value castForCmp = LLVM::BitcastOp::create(
            rewriter, loc, llvmBufferValType, atomicCmpData);
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
    Value numRecords =
        getNumRecords(rewriter, loc, memrefType, memrefDescriptor, strides,
                      elementByteWidth, chipset, adaptor.getBoundsCheck());
    Value resource = makeBufferRsrc(rewriter, loc, ptr, numRecords,
                                    adaptor.getBoundsCheck(), chipset);
    args.push_back(resource);

    // Indexing (voffset)
    Value voffset = getLinearIndexI32(rewriter, loc, memrefDescriptor,
                                      adaptor.getIndices(), strides);
    if (std::optional<int32_t> indexOffset = adaptor.getIndexOffset();
        indexOffset && *indexOffset > 0) {
      Value extraOffsetConst = createI32Constant(rewriter, loc, *indexOffset);
      voffset = voffset ? LLVM::AddOp::create(rewriter, loc, voffset,
                                              extraOffsetConst)
                        : extraOffsetConst;
    }
    voffset = LLVM::MulOp::create(rewriter, loc, voffset, byteWidthConst);
    args.push_back(voffset);

    // SGPR offset.
    Value sgprOffset = adaptor.getSgprOffset();
    if (!sgprOffset)
      sgprOffset = createI32Constant(rewriter, loc, 0);
    sgprOffset = LLVM::MulOp::create(rewriter, loc, sgprOffset, byteWidthConst);
    args.push_back(sgprOffset);

    // bit 0: GLC = 0 (atomics drop value, less coherency)
    // bits 1-2: SLC, DLC = 0 (similarly)
    // bit 3: swizzled (0 for raw)
    args.push_back(createI32Constant(rewriter, loc, 0));

    llvm::SmallVector<Type, 1> resultTypes(gpuOp->getNumResults(),
                                           llvmBufferValType);
    Operation *lowered = Intrinsic::create(rewriter, loc, resultTypes, args,
                                           ArrayRef<NamedAttribute>());
    if (lowered->getNumResults() == 1) {
      Value replacement = lowered->getResult(0);
      if (llvmBufferValType != llvmWantedDataType) {
        replacement = LLVM::BitcastOp::create(rewriter, loc, llvmWantedDataType,
                                              replacement);
      }
      rewriter.replaceOp(gpuOp, replacement);
    } else {
      rewriter.eraseOp(gpuOp);
    }
    return success();
  }
};

// TODO: AMDGPU backend already have all this bitpacking logic, we should move
// it to some common place.
///  Vmcnt, Expcnt and Lgkmcnt are decoded as follows:
///     Vmcnt = Waitcnt[3:0]        (pre-gfx9)
///     Vmcnt = Waitcnt[15:14,3:0]  (gfx9,10)
///     Vmcnt = Waitcnt[15:10]      (gfx11)
///     Expcnt = Waitcnt[6:4]       (pre-gfx11)
///     Expcnt = Waitcnt[2:0]       (gfx11)
///     Lgkmcnt = Waitcnt[11:8]     (pre-gfx10)
///     Lgkmcnt = Waitcnt[13:8]     (gfx10)
///     Lgkmcnt = Waitcnt[9:4]      (gfx11)
static FailureOr<unsigned> encodeWaitcnt(Chipset chipset, unsigned vmcnt,
                                         unsigned expcnt, unsigned lgkmcnt) {
  if (chipset.majorVersion < 9) {
    vmcnt = std::min(15u, vmcnt);
    expcnt = std::min(7u, expcnt);
    lgkmcnt = std::min(15u, lgkmcnt);
    return vmcnt | (expcnt << 4) | (lgkmcnt << 8);
  }
  if (chipset.majorVersion == 9) {
    vmcnt = std::min(63u, vmcnt);
    expcnt = std::min(7u, expcnt);
    lgkmcnt = std::min(15u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (chipset.majorVersion == 10) {
    vmcnt = std::min(63u, vmcnt);
    expcnt = std::min(7u, expcnt);
    lgkmcnt = std::min(63u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (chipset.majorVersion == 11) {
    vmcnt = std::min(63u, vmcnt);
    expcnt = std::min(7u, expcnt);
    lgkmcnt = std::min(63u, lgkmcnt);
    return (vmcnt << 10) | expcnt | (lgkmcnt << 4);
  }
  return failure();
}

struct MemoryCounterWaitOpLowering
    : public ConvertOpToLLVMPattern<MemoryCounterWaitOp> {
  MemoryCounterWaitOpLowering(const LLVMTypeConverter &converter,
                              Chipset chipset)
      : ConvertOpToLLVMPattern<MemoryCounterWaitOp>(converter),
        chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(MemoryCounterWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (chipset.majorVersion >= 12) {
      Location loc = op.getLoc();
      if (std::optional<int> ds = adaptor.getDs())
        ROCDL::WaitDscntOp::create(rewriter, loc, *ds);

      if (std::optional<int> load = adaptor.getLoad())
        ROCDL::WaitLoadcntOp::create(rewriter, loc, *load);

      if (std::optional<int> store = adaptor.getStore())
        ROCDL::WaitStorecntOp::create(rewriter, loc, *store);

      if (std::optional<int> exp = adaptor.getExp())
        ROCDL::WaitExpcntOp::create(rewriter, loc, *exp);

      if (std::optional<int> tensor = adaptor.getTensor())
        ROCDL::WaitTensorcntOp::create(rewriter, loc, *tensor);

      rewriter.eraseOp(op);
      return success();
    }

    if (adaptor.getTensor())
      return op.emitOpError("unsupported chipset");

    auto getVal = [](Attribute attr) -> unsigned {
      if (attr)
        return cast<IntegerAttr>(attr).getInt();

      // This value will be clamped to the maximum value for the chipset.
      return 1024;
    };
    unsigned ds = getVal(adaptor.getDsAttr());
    unsigned exp = getVal(adaptor.getExpAttr());

    unsigned vmcnt = 1024;
    Attribute load = adaptor.getLoadAttr();
    Attribute store = adaptor.getStoreAttr();
    if (load && store) {
      vmcnt = getVal(load) + getVal(store);
    } else if (load) {
      vmcnt = getVal(load);
    } else if (store) {
      vmcnt = getVal(store);
    }

    FailureOr<unsigned> waitcnt = encodeWaitcnt(chipset, vmcnt, exp, ds);
    if (failed(waitcnt))
      return op.emitOpError("unsupported chipset");

    rewriter.replaceOpWithNewOp<ROCDL::SWaitcntOp>(op, *waitcnt);
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
    Location loc = op.getLoc();
    // This ensures that waits on global memory aren't introduced on
    // chips that don't have the BackOffBarrier feature enabled in LLVM.
    bool requiresInlineAsm = chipset < kGfx90a;

    Attribute mmra =
        rewriter.getAttr<LLVM::MMRATagAttr>("amdgpu-synchronize-as", "local");
    // Note: while there *is* a workgroup-one-as scope, this, when combined with
    // the MMRA, will lead to the fence having no effect. This is because the
    // codepaths for an atomic load or store will observe that a
    // one-address-space atomic to LDS requires no synchronization because
    // operations on LDS are totally ordered with respect to each other, and so
    // will not emit the correct waitcnt operations that these fences are
    // intended to produce. Therefore, we use a broader type of fence and rely
    // on the MMRA to relax it to the semantics we want.
    StringRef scope = "workgroup";

    auto relFence = LLVM::FenceOp::create(rewriter, loc,
                                          LLVM::AtomicOrdering::release, scope);
    relFence->setDiscardableAttr(LLVM::LLVMDialect::getMmraAttrName(), mmra);
    if (requiresInlineAsm) {
      auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                      LLVM::AsmDialect::AD_ATT);
      const char *asmStr = ";;;WARNING: BREAKS DEBUG WATCHES\ns_barrier";
      const char *constraints = "";
      LLVM::InlineAsmOp::create(
          rewriter, loc,
          /*resultTypes=*/TypeRange(), /*operands=*/ValueRange(),
          /*asm_string=*/asmStr, constraints, /*has_side_effects=*/true,
          /*is_align_stack=*/false, LLVM::TailCallKind::None,
          /*asm_dialect=*/asmDialectAttr,
          /*operand_attrs=*/ArrayAttr());
    } else if (chipset.majorVersion < 12) {
      ROCDL::SBarrierOp::create(rewriter, loc);
    } else {
      ROCDL::BarrierSignalOp::create(rewriter, loc, -1);
      ROCDL::BarrierWaitOp::create(rewriter, loc, -1);
    }

    auto acqFence = LLVM::FenceOp::create(rewriter, loc,
                                          LLVM::AtomicOrdering::acquire, scope);
    acqFence->setDiscardableAttr(LLVM::LLVMDialect::getMmraAttrName(), mmra);
    rewriter.replaceOp(op, acqFence);
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

/// Pack small float vector operands (fp4/fp6/fp8/bf16) into the format
/// expected by scaled matrix multiply intrinsics (MFMA/WMMA).
///
/// Specifically:
/// 1. If the element type is bfloat16, bitcast it to i16 unless rocdl intrinsic
/// allows bf16. Newer MFMAs support bf16 types on operand, check
/// IntrinsicsAMDGPU.td file for reference.
/// 2. If instead we have a more than 64-bit quantity, use a <N / 4 x i32>
/// instead, which is what the f8f6f4 intrinsics use.
/// 3. If `input` is a vector of N <= 8 bytes, bitcast it to a (N * 8)-bit
/// integer.
///
/// Note that the type of `input` has already been LLVM type converted:
/// therefore 8-bit and smaller floats are represented as their corresponding
/// `iN` integers.
static Value packSmallFloatVectorOperand(ConversionPatternRewriter &rewriter,
                                         Location loc, Value input,
                                         bool allowBf16 = true) {
  Type inputType = input.getType();
  if (auto vectorType = dyn_cast<VectorType>(inputType)) {
    if (vectorType.getElementType().isBF16() && !allowBf16)
      return LLVM::BitcastOp::create(
          rewriter, loc, vectorType.clone(rewriter.getI16Type()), input);
    if (vectorType.getElementType().isInteger(8) &&
        vectorType.getNumElements() <= 8)
      return LLVM::BitcastOp::create(
          rewriter, loc,
          rewriter.getIntegerType(vectorType.getNumElements() * 8), input);
    if (isa<IntegerType>(vectorType.getElementType()) &&
        vectorType.getElementTypeBitWidth() <= 8) {
      int64_t numWords = llvm::divideCeil(
          vectorType.getNumElements() * vectorType.getElementTypeBitWidth(),
          32);
      return LLVM::BitcastOp::create(
          rewriter, loc, VectorType::get(numWords, rewriter.getI32Type()),
          input);
    }
  }
  return input;
}

/// Converts sparse MFMA (smfmac) operands to the expected ROCDL types.
static Value convertSparseMFMAVectorOperand(ConversionPatternRewriter &rewriter,
                                            Location loc, Value input,
                                            bool allowBf16 = true) {
  Type inputType = input.getType();
  auto vectorType = cast<VectorType>(inputType);
  // bf16 -> i16 when not allowed (pre-gfx950).
  if (vectorType.getElementType().isBF16() && !allowBf16)
    return LLVM::BitcastOp::create(
        rewriter, loc, vectorType.clone(rewriter.getI16Type()), input);
  // i8/fp8 vectors -> vector<Nxi32>.
  if (isa<IntegerType>(vectorType.getElementType()) &&
      vectorType.getElementTypeBitWidth() <= 8) {
    int64_t numWords = llvm::divideCeil(
        vectorType.getNumElements() * vectorType.getElementTypeBitWidth(), 32);
    return LLVM::BitcastOp::create(
        rewriter, loc, VectorType::get(numWords, rewriter.getI32Type()), input);
  }
  return input;
}

/// Converts the scaled MFMA/WMMA operands, `scalesA` and `scalesB`, from MLIR
/// AMDGPU dialect convention to ROCDL and LLVM AMDGPU intrinsics convention.
///
/// Specifically:
/// 1. If `input` is a i8 value, zero extend it to i32
/// 2. If `input` is a vector of length 4 or 8 and type i8, cast it to i32
///
/// Note that the type of `input` has already been LLVM type converted:
/// therefore 8-bit and smaller floats are represented as their corresponding
/// `iN` integers.
static Value castScaleOperand(ConversionPatternRewriter &rewriter, Location loc,
                              Value input) {
  return TypeSwitch<Type, Value>(input.getType())
      .Case([&](IntegerType) {
        // Handle scalar i8: zero extend to i32.
        return LLVM::ZExtOp::create(rewriter, loc, rewriter.getI32Type(),
                                    input);
      })
      .Case([&](VectorType vectorType) {
        // Handle vector<4xi8> -> i32 or vector<8xi8> -> i64.
        int64_t numElements = vectorType.getNumElements();
        assert((numElements == 4 || numElements == 8) &&
               "scale operand must be a vector of length 4 or 8");
        IntegerType outputType =
            (numElements == 4) ? rewriter.getI32Type() : rewriter.getI64Type();
        return LLVM::BitcastOp::create(rewriter, loc, outputType, input);
      })
      .DefaultUnreachable("unexpected input type for scale operand");
}

/// Maps f8 scale element types to WMMA scale format codes.
static std::optional<uint32_t> getWmmaScaleFormat(Type elemType) {
  return TypeSwitch<Type, std::optional<uint32_t>>(elemType)
      .Case([](Float8E8M0FNUType) { return 0; })
      .Case([](Float8E4M3FNType) { return 2; })
      .Default(std::nullopt);
}

/// Determines the ROCDL intrinsic name for scaled WMMA based on dimensions
/// and scale block size (16 or 32).
static std::optional<StringRef>
getScaledWmmaIntrinsicName(int64_t m, int64_t n, int64_t k, bool isScale16) {
  if (m == 16 && n == 16 && k == 128)
    return isScale16
               ? ROCDL::wmma_scale16_f32_16x16x128_f8f6f4::getOperationName()
               : ROCDL::wmma_scale_f32_16x16x128_f8f6f4::getOperationName();

  if (m == 32 && n == 16 && k == 128)
    return isScale16 ? ROCDL::wmma_scale16_f32_32x16x128_f4::getOperationName()
                     : ROCDL::wmma_scale_f32_32x16x128_f4::getOperationName();

  return std::nullopt;
}

/// Push an input operand. If it is a float type, nothing to do. If it is
/// an integer type, then we need to also push its signdness (1 for signed, 0
/// for unsigned) and we need to pack the input 16xi8 vector into a 4xi32
/// vector (or the 8xi8 vector into a 2xi32 one for gfx12+).
/// We also need to convert bfloat inputs to i16 to account for the bfloat
/// intrinsics having been defined before the AMD backend supported bfloat. We
/// similarly need to pack 8-bit float types into integers as if they were i8
/// (which they are for the backend's purposes).
static void wmmaPushInputOperand(
    ConversionPatternRewriter &rewriter, Location loc,
    const TypeConverter *typeConverter, bool isUnsigned, Value llvmInput,
    Value mlirInput, SmallVectorImpl<Value> &operands,
    SmallVectorImpl<NamedAttribute> &attrs, StringRef attrName) {
  Type inputType = llvmInput.getType();
  auto vectorType = dyn_cast<VectorType>(inputType);
  if (!vectorType) {
    operands.push_back(llvmInput);
    return;
  }
  Type elemType = vectorType.getElementType();
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
    attrs.push_back(
        NamedAttribute(attrName, rewriter.getBoolAttr(!localIsUnsigned)));
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
    castInput = LLVM::ZExtOp::create(rewriter, loc, i32, castInput);
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
                                  bool clamp, SmallVectorImpl<Value> &operands,
                                  SmallVectorImpl<NamedAttribute> &attrs) {
  Type inputType = output.getType();
  auto vectorType = dyn_cast<VectorType>(inputType);
  Type elemType = vectorType.getElementType();
  operands.push_back(output);
  if (elemType.isF16() || elemType.isBF16() || elemType.isInteger(16)) {
    attrs.push_back(
        NamedAttribute("opsel", rewriter.getBoolAttr(subwordOffset)));
  } else if (elemType.isInteger(32)) {
    attrs.push_back(NamedAttribute("clamp", rewriter.getBoolAttr(clamp)));
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

static std::optional<uint32_t> smallFloatTypeToFormatCode(Type mlirElemType) {
  return llvm::TypeSwitch<Type, std::optional<uint32_t>>(mlirElemType)
      .Case([](Float8E4M3FNType) { return 0u; })
      .Case([](Float8E5M2Type) { return 1u; })
      .Case([](Float6E2M3FNType) { return 2u; })
      .Case([](Float6E3M2FNType) { return 3u; })
      .Case([](Float4E2M1FNType) { return 4u; })
      .Default(std::nullopt);
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

  std::optional<uint32_t> aTypeCode = smallFloatTypeToFormatCode(aType);
  std::optional<uint32_t> bTypeCode = smallFloatTypeToFormatCode(bType);
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

/// Returns the `rocdl` intrinsic corresponding to a WMMA operation `wmma`
/// for RDNA3/4 architectures.
static std::optional<StringRef>
wmmaOpToIntrinsicRDNA(Type elemSourceType, Type elemBSourceType,
                      Type elemDestType, uint32_t k, bool isRDNA3) {
  using fp8 = Float8E4M3FNType;
  using bf8 = Float8E5M2Type;

  // Handle k == 16 for RDNA3/4.
  if (k == 16) {
    // Common patterns for RDNA3 and RDNA4.
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

    // RDNA3 specific patterns.
    if (isRDNA3) {
      if (elemSourceType.isInteger(4) && elemDestType.isInteger(32))
        return ROCDL::wmma_i32_16x16x16_iu4::getOperationName();
      return std::nullopt;
    }

    // RDNA4 specific patterns (fp8/bf8).
    if (isa<fp8>(elemSourceType) && isa<fp8>(elemBSourceType) &&
        elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x16_fp8_fp8::getOperationName();
    if (isa<fp8>(elemSourceType) && isa<bf8>(elemBSourceType) &&
        elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x16_fp8_bf8::getOperationName();
    if (isa<bf8>(elemSourceType) && isa<bf8>(elemBSourceType) &&
        elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x16_bf8_bf8::getOperationName();
    if (isa<bf8>(elemSourceType) && isa<fp8>(elemBSourceType) &&
        elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x16_bf8_fp8::getOperationName();
    if (elemSourceType.isInteger(4) && elemDestType.isInteger(32))
      return ROCDL::wmma_i32_16x16x16_iu4::getOperationName();

    return std::nullopt;
  }

  // Handle k == 32 for RDNA4.
  if (k == 32 && !isRDNA3) {
    if (elemSourceType.isInteger(4) && elemDestType.isInteger(32))
      return ROCDL::wmma_i32_16x16x32_iu4::getOperationName();
  }

  return std::nullopt;
}

/// Return the `rocdl` intrinsic corresponding to a WMMA operation `wmma`
/// for the gfx1250 architecture.
static std::optional<StringRef> wmmaOpToIntrinsicGfx1250(Type elemSourceType,
                                                         Type elemBSourceType,
                                                         Type elemDestType,
                                                         uint32_t k) {
  using fp8 = Float8E4M3FNType;
  using bf8 = Float8E5M2Type;

  if (k == 4) {
    if (elemSourceType.isF32() && elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x4_f32::getOperationName();

    return std::nullopt;
  }

  if (k == 32) {
    if (elemSourceType.isF16() && elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x32_f16::getOperationName();
    if (elemSourceType.isBF16() && elemDestType.isF32())
      return ROCDL::wmma_f32_16x16x32_bf16::getOperationName();
    if (elemSourceType.isF16() && elemDestType.isF16())
      return ROCDL::wmma_f16_16x16x32_f16::getOperationName();
    if (elemSourceType.isBF16() && elemDestType.isBF16())
      return ROCDL::wmma_bf16_16x16x32_bf16::getOperationName();

    return std::nullopt;
  }

  if (k == 64) {
    if (isa<fp8>(elemSourceType) && isa<fp8>(elemBSourceType)) {
      if (elemDestType.isF32())
        return ROCDL::wmma_f32_16x16x64_fp8_fp8::getOperationName();
      if (elemDestType.isF16())
        return ROCDL::wmma_f16_16x16x64_fp8_fp8::getOperationName();
    }
    if (isa<fp8>(elemSourceType) && isa<bf8>(elemBSourceType)) {
      if (elemDestType.isF32())
        return ROCDL::wmma_f32_16x16x64_fp8_bf8::getOperationName();
      if (elemDestType.isF16())
        return ROCDL::wmma_f16_16x16x64_fp8_bf8::getOperationName();
    }
    if (isa<bf8>(elemSourceType) && isa<bf8>(elemBSourceType)) {
      if (elemDestType.isF32())
        return ROCDL::wmma_f32_16x16x64_bf8_bf8::getOperationName();
      if (elemDestType.isF16())
        return ROCDL::wmma_f16_16x16x64_bf8_bf8::getOperationName();
    }
    if (isa<bf8>(elemSourceType) && isa<fp8>(elemBSourceType)) {
      if (elemDestType.isF32())
        return ROCDL::wmma_f32_16x16x64_bf8_fp8::getOperationName();
      if (elemDestType.isF16())
        return ROCDL::wmma_f16_16x16x64_bf8_fp8::getOperationName();
    }
    if (elemSourceType.isInteger(8) && elemDestType.isInteger(32))
      return ROCDL::wmma_i32_16x16x64_iu8::getOperationName();

    return std::nullopt;
  }

  if (k == 128) {
    if (isa<fp8>(elemSourceType) && isa<fp8>(elemBSourceType)) {
      if (elemDestType.isF32())
        return ROCDL::wmma_f32_16x16x128_fp8_fp8::getOperationName();
      if (elemDestType.isF16())
        return ROCDL::wmma_f16_16x16x128_fp8_fp8::getOperationName();
    }
    if (isa<fp8>(elemSourceType) && isa<bf8>(elemBSourceType)) {
      if (elemDestType.isF32())
        return ROCDL::wmma_f32_16x16x128_fp8_bf8::getOperationName();
      if (elemDestType.isF16())
        return ROCDL::wmma_f16_16x16x128_fp8_bf8::getOperationName();
    }
    if (isa<bf8>(elemSourceType) && isa<bf8>(elemBSourceType)) {
      if (elemDestType.isF32())
        return ROCDL::wmma_f32_16x16x128_bf8_bf8::getOperationName();
      if (elemDestType.isF16())
        return ROCDL::wmma_f16_16x16x128_bf8_bf8::getOperationName();
    }
    if (isa<bf8>(elemSourceType) && isa<fp8>(elemBSourceType)) {
      if (elemDestType.isF32())
        return ROCDL::wmma_f32_16x16x128_bf8_fp8::getOperationName();
      if (elemDestType.isF16())
        return ROCDL::wmma_f16_16x16x128_bf8_fp8::getOperationName();
    }

    return std::nullopt;
  }

  return std::nullopt;
}

/// Returns the `rocdl` intrinsic corresponding to a SparseMFMA (smfmac)
/// operation if one exists. This includes checking to ensure the intrinsic is
/// supported on the architecture you are compiling for.
static std::optional<StringRef> smfmacOpToIntrinsic(SparseMFMAOp op,
                                                    Chipset chipset) {
  bool isGfx950 = chipset >= kGfx950;
  auto isFp8 = [&](Type t) { return typeIsExpectedFp8ForChipset(chipset, t); };
  auto isBf8 = [&](Type t) { return typeIsExpectedBf8ForChipset(chipset, t); };

  uint32_t m = op.getM(), n = op.getN(), k = op.getK();
  Type sourceAElem = getElementTypeOrSelf(op.getSourceA().getType());
  Type sourceBElem = getElementTypeOrSelf(op.getSourceB().getType());
  Type destElem = getElementTypeOrSelf(op.getDestC().getType());

  if (m == 16 && n == 16 && k == 32) {
    if (sourceAElem.isF16() && sourceBElem.isF16() && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x32_f16::getOperationName();
    if (sourceAElem.isBF16() && sourceBElem.isBF16() && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x32_bf16::getOperationName();
  }

  if (m == 16 && n == 16 && k == 64) {
    if (isGfx950) {
      if (sourceAElem.isF16() && sourceBElem.isF16() && destElem.isF32())
        return ROCDL::smfmac_f32_16x16x64_f16::getOperationName();
      if (sourceAElem.isBF16() && sourceBElem.isBF16() && destElem.isF32())
        return ROCDL::smfmac_f32_16x16x64_bf16::getOperationName();
    }
    if (sourceAElem.isInteger(8) && sourceBElem.isInteger(8) &&
        destElem.isInteger(32))
      return ROCDL::smfmac_i32_16x16x64_i8::getOperationName();
    if (isFp8(sourceAElem) && isFp8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x64_fp8_fp8::getOperationName();
    if (isFp8(sourceAElem) && isBf8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x64_fp8_bf8::getOperationName();
    if (isBf8(sourceAElem) && isFp8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x64_bf8_fp8::getOperationName();
    if (isBf8(sourceAElem) && isBf8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x64_bf8_bf8::getOperationName();
  }

  if (m == 16 && n == 16 && k == 128 && isGfx950) {
    if (sourceAElem.isInteger(8) && sourceBElem.isInteger(8) &&
        destElem.isInteger(32))
      return ROCDL::smfmac_i32_16x16x128_i8::getOperationName();
    if (isFp8(sourceAElem) && isFp8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x128_fp8_fp8::getOperationName();
    if (isFp8(sourceAElem) && isBf8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x128_fp8_bf8::getOperationName();
    if (isBf8(sourceAElem) && isFp8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x128_bf8_fp8::getOperationName();
    if (isBf8(sourceAElem) && isBf8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_16x16x128_bf8_bf8::getOperationName();
  }

  if (m == 32 && n == 32 && k == 16) {
    if (sourceAElem.isF16() && sourceBElem.isF16() && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x16_f16::getOperationName();
    if (sourceAElem.isBF16() && sourceBElem.isBF16() && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x16_bf16::getOperationName();
  }

  if (m == 32 && n == 32 && k == 32) {
    if (isGfx950) {
      if (sourceAElem.isF16() && sourceBElem.isF16() && destElem.isF32())
        return ROCDL::smfmac_f32_32x32x32_f16::getOperationName();
      if (sourceAElem.isBF16() && sourceBElem.isBF16() && destElem.isF32())
        return ROCDL::smfmac_f32_32x32x32_bf16::getOperationName();
    }
    if (sourceAElem.isInteger(8) && sourceBElem.isInteger(8) &&
        destElem.isInteger(32))
      return ROCDL::smfmac_i32_32x32x32_i8::getOperationName();
    if (isFp8(sourceAElem) && isFp8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x32_fp8_fp8::getOperationName();
    if (isFp8(sourceAElem) && isBf8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x32_fp8_bf8::getOperationName();
    if (isBf8(sourceAElem) && isFp8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x32_bf8_fp8::getOperationName();
    if (isBf8(sourceAElem) && isBf8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x32_bf8_bf8::getOperationName();
  }

  if (m == 32 && n == 32 && k == 64 && isGfx950) {
    if (sourceAElem.isInteger(8) && sourceBElem.isInteger(8) &&
        destElem.isInteger(32))
      return ROCDL::smfmac_i32_32x32x64_i8::getOperationName();
    if (isFp8(sourceAElem) && isFp8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x64_fp8_fp8::getOperationName();
    if (isFp8(sourceAElem) && isBf8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x64_fp8_bf8::getOperationName();
    if (isBf8(sourceAElem) && isFp8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x64_bf8_fp8::getOperationName();
    if (isBf8(sourceAElem) && isBf8(sourceBElem) && destElem.isF32())
      return ROCDL::smfmac_f32_32x32x64_bf8_bf8::getOperationName();
  }

  return std::nullopt;
}

/// Returns the `rocdl` intrinsic corresponding to a WMMA operation `wmma`
/// if one exists. This includes checking to ensure the intrinsic is supported
/// on the architecture you are compiling for.
static std::optional<StringRef> wmmaOpToIntrinsic(WMMAOp wmma,
                                                  Chipset chipset) {
  auto sourceVectorType = cast<VectorType>(wmma.getSourceA().getType());
  auto sourceBVectorType = cast<VectorType>(wmma.getSourceB().getType());
  auto destVectorType = cast<VectorType>(wmma.getDestC().getType());
  Type elemSourceType = sourceVectorType.getElementType();
  Type elemBSourceType = sourceBVectorType.getElementType();
  Type elemDestType = destVectorType.getElementType();

  const uint32_t k = wmma.getK();
  const bool isRDNA3 = chipset.majorVersion == 11;
  const bool isRDNA4 = chipset.majorVersion == 12 && chipset.minorVersion == 0;

  // Handle RDNA3 and RDNA4.
  if (isRDNA3 || isRDNA4)
    return wmmaOpToIntrinsicRDNA(elemSourceType, elemBSourceType, elemDestType,
                                 k, isRDNA3);

  // Handle gfx1250.
  if (chipset == kGfx1250)
    return wmmaOpToIntrinsicGfx1250(elemSourceType, elemBSourceType,
                                    elemDestType, k);

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
    // Determine if we can use bf16 in the intrinsic. Newer MFMAs in gfx950+
    // allows bf16 as the input. For reference check IntrinsicsAMDGPU.td file.
    bool allowBf16 = [&]() {
      if (chipset < kGfx950)
        return false;
      if (isScaled)
        return true;
      return intrinsicName.contains("16x16x32.bf16") ||
             intrinsicName.contains("32x32x16.bf16");
    }();
    OperationState loweredOp(loc, intrinsicName);
    loweredOp.addTypes(intrinsicOutType);
    loweredOp.addOperands({packSmallFloatVectorOperand(
                               rewriter, loc, adaptor.getSourceA(), allowBf16),
                           packSmallFloatVectorOperand(
                               rewriter, loc, adaptor.getSourceB(), allowBf16),
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
      lowered = LLVM::BitcastOp::create(rewriter, loc, outType, lowered);
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
        {packSmallFloatVectorOperand(rewriter, loc, adaptor.getSourceA()),
         packSmallFloatVectorOperand(rewriter, loc, adaptor.getSourceB()),
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
         castScaleOperand(rewriter, loc, adaptor.getScalesA()),
         /*scales idx B=*/scalesIdxB,
         /*scales B*/
         castScaleOperand(rewriter, loc, adaptor.getScalesB())});
    Value lowered = rewriter.create(loweredOp)->getResult(0);
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

struct SparseMFMAOpLowering : public ConvertOpToLLVMPattern<SparseMFMAOp> {
  SparseMFMAOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<SparseMFMAOp>(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(SparseMFMAOp op, SparseMFMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outType =
        typeConverter->convertType<VectorType>(op.getDestC().getType());
    if (!outType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    // smfmac is supported on gfx942 and gfx950.
    if (chipset.majorVersion != 9 || chipset < kGfx942)
      return op->emitOpError("sparse MFMA (smfmac) only supported on gfx942+");
    bool isGfx950 = chipset >= kGfx950;

    Value a = convertSparseMFMAVectorOperand(rewriter, loc,
                                             adaptor.getSourceA(), isGfx950);
    Value b = convertSparseMFMAVectorOperand(rewriter, loc,
                                             adaptor.getSourceB(), isGfx950);
    Value c = adaptor.getDestC();

    std::optional<StringRef> maybeIntrinsic = smfmacOpToIntrinsic(op, chipset);
    if (!maybeIntrinsic.has_value())
      return op.emitOpError(
          "no intrinsic matching sparse MFMA on the given chipset");

    // Bitcast sparse indices from vector<4xi8> or vector<2xi16> to i32.
    Value sparseIdx = LLVM::BitcastOp::create(
        rewriter, loc, rewriter.getI32Type(), adaptor.getSparseIdx());

    OperationState loweredOp(loc, maybeIntrinsic.value());
    loweredOp.addTypes(outType);
    loweredOp.addOperands({a, b, c, sparseIdx,
                           createI32Constant(rewriter, loc, op.getCbsz()),
                           createI32Constant(rewriter, loc, op.getAbid())});
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

    bool isGFX1250 = chipset >= kGfx1250;

    // The WMMA operations represent vectors of bf16s as vectors of i16s
    // (except on gfx1250), so we need to bitcast bfloats to i16 and then
    // bitcast them back.
    auto aType = cast<VectorType>(adaptor.getSourceA().getType());
    auto bType = cast<VectorType>(adaptor.getSourceB().getType());
    auto destCType = cast<VectorType>(adaptor.getDestC().getType());
    bool castAToI16 = aType.getElementType().isBF16() && !isGFX1250;
    bool castBToI16 = bType.getElementType().isBF16() && !isGFX1250;
    bool castDestCToI16 = destCType.getElementType().isBF16() && !isGFX1250;
    bool castOutToI16 = outType.getElementType().isBF16() && !isGFX1250;
    VectorType rawOutType = outType;
    if (castOutToI16)
      rawOutType = outType.clone(rewriter.getI16Type());
    Value a = adaptor.getSourceA();
    if (castAToI16)
      a = LLVM::BitcastOp::create(rewriter, loc,
                                  aType.clone(rewriter.getI16Type()), a);
    Value b = adaptor.getSourceB();
    if (castBToI16)
      b = LLVM::BitcastOp::create(rewriter, loc,
                                  bType.clone(rewriter.getI16Type()), b);
    Value destC = adaptor.getDestC();
    if (castDestCToI16)
      destC = LLVM::BitcastOp::create(
          rewriter, loc, destCType.clone(rewriter.getI16Type()), destC);

    std::optional<StringRef> maybeIntrinsic = wmmaOpToIntrinsic(op, chipset);

    if (!maybeIntrinsic.has_value())
      return op.emitOpError("no intrinsic matching WMMA on the given chipset");

    if (chipset.majorVersion >= 12 && op.getSubwordOffset() != 0)
      return op.emitOpError("subwordOffset not supported on gfx12+");

    SmallVector<Value, 4> operands;
    SmallVector<NamedAttribute, 4> attrs;
    wmmaPushInputOperand(rewriter, loc, typeConverter, op.getUnsignedA(), a,
                         op.getSourceA(), operands, attrs, "signA");
    wmmaPushInputOperand(rewriter, loc, typeConverter, op.getUnsignedB(), b,
                         op.getSourceB(), operands, attrs, "signB");
    wmmaPushOutputOperand(rewriter, loc, typeConverter, destC,
                          op.getSubwordOffset(), op.getClamp(), operands,
                          attrs);

    OperationState loweredOp(loc, *maybeIntrinsic);
    loweredOp.addTypes(rawOutType);
    loweredOp.addOperands(operands);
    loweredOp.addAttributes(attrs);
    Operation *lowered = rewriter.create(loweredOp);

    Operation *maybeCastBack = lowered;
    if (rawOutType != outType)
      maybeCastBack = LLVM::BitcastOp::create(rewriter, loc, outType,
                                              lowered->getResult(0));
    rewriter.replaceOp(op, maybeCastBack->getResults());

    return success();
  }
};

struct ScaledWMMAOpLowering : public ConvertOpToLLVMPattern<ScaledWMMAOp> {
  ScaledWMMAOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<ScaledWMMAOp>(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(ScaledWMMAOp op, ScaledWMMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outType =
        typeConverter->convertType<VectorType>(op.getDestD().getType());
    if (!outType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    if (chipset < kGfx1250)
      return op->emitOpError("WMMA scale only supported on gfx1250+");

    int64_t m = op.getM();
    int64_t n = op.getN();
    int64_t k = op.getK();

    Type aElemType = getElementTypeOrSelf(op.getSourceA().getType());
    Type bElemType = getElementTypeOrSelf(op.getSourceB().getType());

    std::optional<uint32_t> aFmtCode = smallFloatTypeToFormatCode(aElemType);
    std::optional<uint32_t> bFmtCode = smallFloatTypeToFormatCode(bElemType);

    if (!aFmtCode || !bFmtCode)
      return op.emitOpError("unsupported element types for scaled_wmma");

    // Get scale vector types and determine variant (scale vs scale16).
    auto scaleAVecType = cast<VectorType>(op.getScaleA().getType());
    auto scaleBVecType = cast<VectorType>(op.getScaleB().getType());

    if (scaleAVecType.getNumElements() != scaleBVecType.getNumElements())
      return op.emitOpError("scaleA and scaleB must have equal vector length");

    // Extract scale format from element types.
    Type scaleAElemType = scaleAVecType.getElementType();
    Type scaleBElemType = scaleBVecType.getElementType();

    std::optional<uint32_t> scaleAFmt = getWmmaScaleFormat(scaleAElemType);
    std::optional<uint32_t> scaleBFmt = getWmmaScaleFormat(scaleBElemType);

    if (!scaleAFmt || !scaleBFmt)
      return op.emitOpError("unsupported scale element types");

    // Determine which intrinsic to use based on dimensions.
    bool isScale16 = (scaleAVecType.getNumElements() == 8);
    std::optional<StringRef> intrinsicName =
        getScaledWmmaIntrinsicName(m, n, k, isScale16);
    if (!intrinsicName)
      return op.emitOpError("unsupported scaled_wmma dimensions: ")
             << m << "x" << n << "x" << k;

    SmallVector<NamedAttribute, 8> attrs;

    // The f4 variant does not have fmtA and fmtB attributes.
    bool is32x16 = (m == 32 && n == 16 && k == 128);
    if (!is32x16) {
      attrs.emplace_back("fmtA", rewriter.getI32IntegerAttr(*aFmtCode));
      attrs.emplace_back("fmtB", rewriter.getI32IntegerAttr(*bFmtCode));
    }

    // modC uses default value of 0.
    attrs.emplace_back("modC", rewriter.getI16IntegerAttr(0));

    // Scale attributes. Convert user-facing firstScaleLane (0 or 16) to the
    // half of the wave that is being selected (0 or 1).
    attrs.emplace_back(
        "scaleAType", rewriter.getI32IntegerAttr(op.getAFirstScaleLane() / 16));
    attrs.emplace_back("fmtScaleA", rewriter.getI32IntegerAttr(*scaleAFmt));
    attrs.emplace_back(
        "scaleBType", rewriter.getI32IntegerAttr(op.getBFirstScaleLane() / 16));
    attrs.emplace_back("fmtScaleB", rewriter.getI32IntegerAttr(*scaleBFmt));

    // Reuse flags use default value of false.
    attrs.emplace_back("reuseA", rewriter.getBoolAttr(false));
    attrs.emplace_back("reuseB", rewriter.getBoolAttr(false));

    // Convert typed float vectors to packed format.
    Value sourceA =
        packSmallFloatVectorOperand(rewriter, loc, adaptor.getSourceA());
    Value sourceB =
        packSmallFloatVectorOperand(rewriter, loc, adaptor.getSourceB());

    // Pack scale vectors into i32/i64.
    Value packedScaleA = castScaleOperand(rewriter, loc, adaptor.getScaleA());
    Value packedScaleB = castScaleOperand(rewriter, loc, adaptor.getScaleB());

    // Create the intrinsic call.
    OperationState loweredOp(loc, *intrinsicName);
    loweredOp.addTypes(outType);
    loweredOp.addOperands(
        {sourceA, sourceB, adaptor.getDestC(), packedScaleA, packedScaleB});
    loweredOp.addAttributes(attrs);

    Operation *lowered = rewriter.create(loweredOp);
    rewriter.replaceOp(op, lowered->getResults());

    return success();
  }
};

struct TransposeLoadOpLowering
    : public ConvertOpToLLVMPattern<TransposeLoadOp> {
  TransposeLoadOpLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<TransposeLoadOp>(converter), chipset(chipset) {}

  Chipset chipset;

  LogicalResult
  matchAndRewrite(TransposeLoadOp op, TransposeLoadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (chipset != kGfx950)
      return op.emitOpError("Non-gfx950 chipset not supported");

    Location loc = op.getLoc();
    auto srcMemRefType = cast<MemRefType>(op.getSrc().getType());

    // Elements in subbyte memrefs are stored non-contiguously,
    // reject if source is sub-byte memref. Use emulated memrefs instead.
    size_t srcElementSize =
        srcMemRefType.getElementType().getIntOrFloatBitWidth();
    if (srcElementSize < 8)
      return op.emitOpError("Expect source memref to have at least 8 bits "
                            "element size, got ")
             << srcElementSize;

    auto resultType = cast<VectorType>(op.getResult().getType());
    Value srcPtr =
        getStridedElementPtr(rewriter, loc, srcMemRefType, adaptor.getSrc(),
                             (adaptor.getSrcIndices()));

    size_t numElements = resultType.getNumElements();
    size_t elementTypeSize =
        resultType.getElementType().getIntOrFloatBitWidth();

    // ROCDL transpose load intrinsics return vectors of 32-bit integers, if
    // the element size is smaller than 16 bits.
    Type rocdlResultType = VectorType::get((numElements * elementTypeSize) / 32,
                                           rewriter.getIntegerType(32));
    Type llvmResultType = typeConverter->convertType(resultType);

    switch (elementTypeSize) {
    case 4: {
      assert(numElements == 16);
      auto rocdlOp = ROCDL::ds_read_tr4_b64::create(rewriter, loc,
                                                    rocdlResultType, srcPtr);
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, llvmResultType, rocdlOp);
      break;
    }
    case 6: {
      assert(numElements == 16);
      auto rocdlOp = ROCDL::ds_read_tr6_b96::create(rewriter, loc,
                                                    rocdlResultType, srcPtr);
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, llvmResultType, rocdlOp);
      break;
    }
    case 8: {
      assert(numElements == 8);
      auto rocdlOp = ROCDL::ds_read_tr8_b64::create(rewriter, loc,
                                                    rocdlResultType, srcPtr);
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, llvmResultType, rocdlOp);
      break;
    }
    case 16: {
      assert(numElements == 4);
      rewriter.replaceOpWithNewOp<ROCDL::ds_read_tr16_b64>(op, llvmResultType,
                                                           srcPtr);
      break;
    }
    default:
      return op.emitOpError("Unsupported element size for transpose load");
    }
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
    int loadWidth = [&]() -> int {
      if (auto transferVectorType = dyn_cast<VectorType>(transferType)) {
        return (transferVectorType.getNumElements() *
                transferVectorType.getElementTypeBitWidth()) /
               8;
      }
      return transferType.getIntOrFloatBitWidth() / 8;
    }();

    // Currently only 1, 2, 4, 12 and 16 byte loads are supported.
    if (!llvm::is_contained({1, 2, 4, 12, 16}, loadWidth))
      return op.emitOpError("chipset unsupported element size");

    if (chipset != kGfx950 && llvm::is_contained({12, 16}, loadWidth))
      return op.emitOpError("Gather to LDS instructions with 12-byte and "
                            "16-byte load widths are only supported on gfx950");

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

struct ScaledExtPackedMatrixOpLowering final
    : public ConvertOpToLLVMPattern<ScaledExtPackedMatrixOp> {
  ScaledExtPackedMatrixOpLowering(const LLVMTypeConverter &converter,
                                  Chipset chipset)
      : ConvertOpToLLVMPattern<amdgpu::ScaledExtPackedMatrixOp>(converter),
        chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(ScaledExtPackedMatrixOp op,
                  ScaledExtPackedMatrixOpAdaptor adaptor,
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
    Value longVec = LLVM::UndefOp::create(rewriter, loc, v4i8);
    if (!sourceVecType) {
      longVec = LLVM::InsertElementOp::create(
          rewriter, loc, longVec, source, createI32Constant(rewriter, loc, 0));
    } else {
      for (int32_t i = 0, e = sourceVecType.getNumElements(); i < e; ++i) {
        Value idx = createI32Constant(rewriter, loc, i);
        Value elem = LLVM::ExtractElementOp::create(rewriter, loc, source, idx);
        longVec =
            LLVM::InsertElementOp::create(rewriter, loc, longVec, elem, idx);
      }
    }
    source = longVec;
  }
  Value i32Source = LLVM::BitcastOp::create(rewriter, loc, i32, source);
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

int32_t getScaleSel(int32_t blockSize, unsigned bitWidth, int32_t scaleWaveHalf,
                    int32_t firstScaleByte) {
  // When lowering amdgpu.scaled_ext_packed_matrix to rocdl.cvt.scale.pk*.f*.f*
  // operations, the attributes blockSize, sourceType, scaleWaveHalf, and
  // firstScaleByte are merged into a single attribute scaleSel. This is how
  // those values are merged together. (Note: scaleWaveHalf isn't a high-level
  // attribute but is derifed from firstScaleLane).
  assert(llvm::is_contained({16, 32}, blockSize));
  assert(llvm::is_contained({4u, 6u, 8u}, bitWidth));

  const bool isFp8 = bitWidth == 8;
  const bool isBlock16 = blockSize == 16;

  if (!isFp8) {
    int32_t bit0 = isBlock16;
    assert(llvm::is_contained({0, 1, 2}, firstScaleByte));
    int32_t bit1 = (firstScaleByte == 2) << 1;
    assert(llvm::is_contained({0, 1}, scaleWaveHalf));
    int32_t bit2 = scaleWaveHalf << 2;
    return bit2 | bit1 | bit0;
  }

  int32_t bit0 = isBlock16;
  // firstScaleByte is guaranteed to be defined by two bits.
  assert(llvm::is_contained({0, 1, 2, 3}, firstScaleByte));
  int32_t bits2and1 = firstScaleByte << 1;
  assert(llvm::is_contained({0, 1}, scaleWaveHalf));
  int32_t bit3 = scaleWaveHalf << 3;
  int32_t bits = bit3 | bits2and1 | bit0;
  // These are invalid cases.
  assert(!llvm::is_contained(
      {0b0011, 0b0101, 0b0111, 0b1000, 0b1001, 0b1011, 0b1111}, bits));
  return bits;
}

static std::optional<StringRef>
scaledExtPacked816ToIntrinsic(Type srcElemType, Type destElemType) {
  using fp4 = Float4E2M1FNType;
  using fp8 = Float8E4M3FNType;
  using bf8 = Float8E5M2Type;
  using fp6 = Float6E2M3FNType;
  using bf6 = Float6E3M2FNType;
  if (isa<fp4>(srcElemType)) {
    if (destElemType.isF16())
      return ROCDL::CvtPkScalePk8F16Fp4Op::getOperationName();
    if (destElemType.isBF16())
      return ROCDL::CvtPkScalePk8Bf16Fp4Op::getOperationName();
    if (destElemType.isF32())
      return ROCDL::CvtPkScalePk8F32Fp4Op::getOperationName();
    return std::nullopt;
  }
  if (isa<fp8>(srcElemType)) {
    if (destElemType.isF16())
      return ROCDL::CvtPkScalePk8F16Fp8Op::getOperationName();
    if (destElemType.isBF16())
      return ROCDL::CvtPkScalePk8Bf16Fp8Op::getOperationName();
    if (destElemType.isF32())
      return ROCDL::CvtPkScalePk8F32Fp8Op::getOperationName();
    return std::nullopt;
  }
  if (isa<bf8>(srcElemType)) {
    if (destElemType.isF16())
      return ROCDL::CvtPkScalePk8F16Bf8Op::getOperationName();
    if (destElemType.isBF16())
      return ROCDL::CvtPkScalePk8Bf16Bf8Op::getOperationName();
    if (destElemType.isF32())
      return ROCDL::CvtPkScalePk8F32Bf8Op::getOperationName();
    return std::nullopt;
  }
  if (isa<fp6>(srcElemType)) {
    if (destElemType.isF16())
      return ROCDL::CvtPkScalePk16F16Fp6Op::getOperationName();
    if (destElemType.isBF16())
      return ROCDL::CvtPkScalePk16Bf16Fp6Op::getOperationName();
    if (destElemType.isF32())
      return ROCDL::CvtPkScalePk16F32Fp6Op::getOperationName();
    return std::nullopt;
  }
  if (isa<bf6>(srcElemType)) {
    if (destElemType.isF16())
      return ROCDL::CvtPkScalePk16F16Bf6Op::getOperationName();
    if (destElemType.isBF16())
      return ROCDL::CvtPkScalePk16Bf16Bf6Op::getOperationName();
    if (destElemType.isF32())
      return ROCDL::CvtPkScalePk16F32Bf6Op::getOperationName();
    return std::nullopt;
  }
  llvm_unreachable("invalid combination of element types for packed conversion "
                   "instructions");
}

LogicalResult ScaledExtPackedMatrixOpLowering::matchAndRewrite(
    ScaledExtPackedMatrixOp op, ScaledExtPackedMatrixOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  using fp4 = Float4E2M1FNType;
  using fp8 = Float8E4M3FNType;
  using bf8 = Float8E5M2Type;
  using fp6 = Float6E2M3FNType;
  using bf6 = Float6E3M2FNType;
  Location loc = op.getLoc();
  if (chipset != kGfx1250) {
    return rewriter.notifyMatchFailure(
        loc,
        "Scaled fp packed conversion instructions are not available on target "
        "architecture and their emulation is not implemented");
  }
  // Convert user-facing firstScaleLane (0 or 16) to the half of the wave that
  // is being selected.
  int32_t scaleWaveHalf = op.getFirstScaleLane() / 16;
  int32_t firstScaleByte = op.getFirstScaleByte();
  int32_t blockSize = op.getBlockSize();
  auto sourceType = cast<VectorType>(op.getSource().getType());
  auto srcElemType = cast<FloatType>(sourceType.getElementType());
  unsigned bitWidth = srcElemType.getWidth();

  auto targetType = cast<VectorType>(op.getResult().getType());
  auto destElemType = cast<FloatType>(targetType.getElementType());

  IntegerType i32 = rewriter.getI32Type();
  Value source = adaptor.getSource();
  Type llvmResultType = typeConverter->convertType(op.getResult().getType());
  Type packedType = nullptr;
  if (isa<fp4>(srcElemType)) {
    packedType = i32;
    packedType = getTypeConverter()->convertType(packedType);
  } else if (isa<fp8, bf8>(srcElemType)) {
    packedType = VectorType::get(2, i32);
    packedType = getTypeConverter()->convertType(packedType);
  } else if (isa<fp6, bf6>(srcElemType)) {
    packedType = VectorType::get(3, i32);
    packedType = getTypeConverter()->convertType(packedType);
  } else {
    llvm_unreachable("invalid element type for packed scaled ext");
  }

  if (!packedType || !llvmResultType) {
    return rewriter.notifyMatchFailure(op, "type conversion failed");
  }

  std::optional<StringRef> maybeIntrinsic =
      scaledExtPacked816ToIntrinsic(srcElemType, destElemType);
  if (!maybeIntrinsic.has_value())
    return op.emitOpError(
        "no intrinsic matching packed scaled conversion on the given chipset");

  int32_t scaleSel =
      getScaleSel(blockSize, bitWidth, scaleWaveHalf, firstScaleByte);
  Value castedScale =
      LLVM::BitcastOp::create(rewriter, loc, i32, adaptor.getScale());
  Value castedSource =
      LLVM::BitcastOp::create(rewriter, loc, packedType, source);

  OperationState loweredOp(loc, *maybeIntrinsic);
  loweredOp.addTypes({llvmResultType});
  loweredOp.addOperands({castedSource, castedScale});

  SmallVector<NamedAttribute, 1> attrs;
  attrs.push_back(
      NamedAttribute("scaleSel", rewriter.getI32IntegerAttr(scaleSel)));

  loweredOp.addAttributes(attrs);
  Operation *lowered = rewriter.create(loweredOp);
  rewriter.replaceOp(op, lowered);

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
    Value longVec = LLVM::ZeroOp::create(rewriter, loc, packedVecType);
    if (!sourceVecType) {
      longVec = LLVM::InsertElementOp::create(
          rewriter, loc, longVec, source, createI32Constant(rewriter, loc, 0));
    } else {
      for (int32_t i = 0, e = sourceVecType.getNumElements(); i < e; ++i) {
        Value idx = createI32Constant(rewriter, loc, i);
        Value elem = LLVM::ExtractElementOp::create(rewriter, loc, source, idx);
        longVec =
            LLVM::InsertElementOp::create(rewriter, loc, longVec, elem, idx);
      }
    }
    source = longVec;
  }
  Value i32Source = LLVM::BitcastOp::create(rewriter, loc, i32, source);

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
    existing = LLVM::BitcastOp::create(rewriter, loc, intResultType, existing);
  else
    existing = LLVM::ZeroOp::create(rewriter, loc, intResultType);

  if (sourceVecType.getNumElements() < 2) {
    Value c0 = createI32Constant(rewriter, loc, 0);
    Value elem0 = LLVM::ExtractElementOp::create(rewriter, loc, source, c0);
    VectorType v2 = VectorType::get(2, sourceElemType);
    source = LLVM::ZeroOp::create(rewriter, loc, v2);
    source = LLVM::InsertElementOp::create(rewriter, loc, source, elem0, c0);
  }

  Value sourceA, sourceB;
  if (sourceElemType.isF32()) {
    Value c0 = createI32Constant(rewriter, loc, 0);
    Value c1 = createI32Constant(rewriter, loc, 1);
    sourceA = LLVM::ExtractElementOp::create(rewriter, loc, source, c0);
    sourceB = LLVM::ExtractElementOp::create(rewriter, loc, source, c1);
  }

  Value result;
  if (sourceElemType.isF32() && isa<Float8E5M2Type>(resultElemType))
    result = ROCDL::CvtScaleF32PkBf8F32Op::create(rewriter, loc, intResultType,
                                                  existing, sourceA, sourceB,
                                                  scale, op.getIndex());
  else if (sourceElemType.isF16() && isa<Float8E5M2Type>(resultElemType))
    result = ROCDL::CvtScaleF32PkBf8F16Op::create(
        rewriter, loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isBF16() && isa<Float8E5M2Type>(resultElemType))
    result = ROCDL::CvtScaleF32PkBf8Bf16Op::create(
        rewriter, loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isF32() && isa<Float8E4M3FNType>(resultElemType))
    result = ROCDL::CvtScaleF32PkFp8F32Op::create(rewriter, loc, intResultType,
                                                  existing, sourceA, sourceB,
                                                  scale, op.getIndex());
  else if (sourceElemType.isF16() && isa<Float8E4M3FNType>(resultElemType))
    result = ROCDL::CvtScaleF32PkFp8F16Op::create(
        rewriter, loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isBF16() && isa<Float8E4M3FNType>(resultElemType))
    result = ROCDL::CvtScaleF32PkFp8Bf16Op::create(
        rewriter, loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isF32() && isa<Float4E2M1FNType>(resultElemType))
    result = ROCDL::CvtScaleF32PkFp4F32Op::create(rewriter, loc, intResultType,
                                                  existing, sourceA, sourceB,
                                                  scale, op.getIndex());
  else if (sourceElemType.isF16() && isa<Float4E2M1FNType>(resultElemType))
    result = ROCDL::CvtScaleF32PkFp4F16Op::create(
        rewriter, loc, intResultType, existing, source, scale, op.getIndex());
  else if (sourceElemType.isBF16() && isa<Float4E2M1FNType>(resultElemType))
    result = ROCDL::CvtScaleF32PkFp4Bf16Op::create(
        rewriter, loc, intResultType, existing, source, scale, op.getIndex());
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
    sourceB = LLVM::UndefOp::create(rewriter, loc, sourceA.getType());
  Value existing = adaptor.getExisting();
  if (existing)
    existing = LLVM::BitcastOp::create(rewriter, loc, i32, existing);
  else
    existing = LLVM::UndefOp::create(rewriter, loc, i32);

  Value result;
  if (typeIsExpectedBf8ForChipset(chipset, resultElemType))
    result = ROCDL::CvtPkBf8F32Op::create(rewriter, loc, i32, sourceA, sourceB,
                                          existing, op.getWordIndex());
  else if (typeIsExpectedFp8ForChipset(chipset, resultElemType))
    result = ROCDL::CvtPkFp8F32Op::create(rewriter, loc, i32, sourceA, sourceB,
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
    existing = LLVM::BitcastOp::create(rewriter, loc, i32, existing);
  else
    existing = LLVM::UndefOp::create(rewriter, loc, i32);

  Value result;
  if (typeIsExpectedBf8ForChipset(chipset, resultElemType))
    result = ROCDL::CvtSrBf8F32Op::create(rewriter, loc, i32, source, stoch,
                                          existing, op.getStoreIndex());
  else if (typeIsExpectedFp8ForChipset(chipset, resultElemType))
    result = ROCDL::CvtSrFp8F32Op::create(rewriter, loc, i32, source, stoch,
                                          existing, op.getStoreIndex());

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
              LLVM::BitcastOp::create(rewriter, loc, llvmSrcIntType, operand);
        }
        auto llvmVecType = typeConverter->convertType(mlir::VectorType::get(
            32 / operandType.getIntOrFloatBitWidth(), llvmSrcIntType));
        Value undefVec = LLVM::UndefOp::create(rewriter, loc, llvmVecType);
        operand =
            LLVM::InsertElementOp::create(rewriter, loc, undefVec, operand,
                                          createI32Constant(rewriter, loc, 0));
        operand = LLVM::BitcastOp::create(rewriter, loc, llvmType, operand);
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

    case DPPPerm::quad_perm: {
      auto quadPermAttr = cast<ArrayAttr>(*permArgument);
      int32_t i = 0;
      for (auto elem : quadPermAttr.getAsRange<IntegerAttr>()) {
        uint32_t num = elem.getInt();
        DppCtrl |= num << (i * 2);
        i++;
      }
      break;
    }
    case DPPPerm::row_shl: {
      auto intAttr = cast<IntegerAttr>(*permArgument);
      DppCtrl = intAttr.getInt() + DppCtrl::ROW_SHL0;
      break;
    }
    case DPPPerm::row_shr: {
      auto intAttr = cast<IntegerAttr>(*permArgument);
      DppCtrl = intAttr.getInt() + DppCtrl::ROW_SHR0;
      break;
    }
    case DPPPerm::row_ror: {
      auto intAttr = cast<IntegerAttr>(*permArgument);
      DppCtrl = intAttr.getInt() + DppCtrl::ROW_ROR0;
      break;
    }
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
    auto dppMovOp =
        ROCDL::DPPUpdateOp::create(rewriter, loc, llvmType, old, src, DppCtrl,
                                   rowMask, bankMask, boundCtrl);

    Value result = dppMovOp.getRes();
    if (srcType.getIntOrFloatBitWidth() < 32) {
      result = LLVM::TruncOp::create(rewriter, loc, llvmSrcIntType, result);
      if (!llvm::isa<IntegerType>(srcType)) {
        result = LLVM::BitcastOp::create(rewriter, loc, srcType, result);
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
          ROCDL::DsSwizzleOp::create(rewriter, loc, v.getType(), v, maskValue);
      swizzled.emplace_back(res);
    }

    Value result = LLVM::composeValue(rewriter, loc, swizzled, src.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct AMDGPUPermlaneLowering : public ConvertOpToLLVMPattern<PermlaneSwapOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  AMDGPUPermlaneLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<PermlaneSwapOp>(converter), chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(PermlaneSwapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (chipset < kGfx950)
      return op->emitOpError("permlane_swap is only supported on gfx950+");

    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    Value src = adaptor.getSrc();
    unsigned rowLength = op.getRowLength();
    bool fi = op.getFetchInactive();
    bool boundctrl = op.getBoundCtrl();

    SmallVector<Value> decomposed =
        LLVM::decomposeValue(rewriter, loc, src, i32);

    SmallVector<Value> permuted;
    for (Value v : decomposed) {
      Value res;
      Type i32pair = LLVM::LLVMStructType::getLiteral(
          rewriter.getContext(), {v.getType(), v.getType()});

      if (rowLength == 16)
        res = ROCDL::Permlane16SwapOp::create(rewriter, loc, i32pair, v, v, fi,
                                              boundctrl);
      else if (rowLength == 32)
        res = ROCDL::Permlane32SwapOp::create(rewriter, loc, i32pair, v, v, fi,
                                              boundctrl);
      else
        llvm_unreachable("unsupported row length");

      Value vdst0 = LLVM::ExtractValueOp::create(rewriter, loc, res, {0});
      Value vdst1 = LLVM::ExtractValueOp::create(rewriter, loc, res, {1});

      Value isEqual = LLVM::ICmpOp::create(rewriter, loc,
                                           LLVM::ICmpPredicate::eq, vdst0, v);

      // Per `permlane(16|32)` semantics: if the first extracted element equals
      // 'v', the result is the second element; otherwise it is the first.
      Value vdstNew =
          LLVM::SelectOp::create(rewriter, loc, isEqual, vdst1, vdst0);
      permuted.emplace_back(vdstNew);
    }

    Value result = LLVM::composeValue(rewriter, loc, permuted, src.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

static Value setValueAtOffset(ConversionPatternRewriter &rewriter, Location loc,
                              Value accumulator, Value value, int64_t shift) {
  shift = shift % 32;
  Value shiftAmount;
  if (shift != 0) {
    shiftAmount = createI32Constant(rewriter, loc, shift % 32);
    value = LLVM::ShlOp::create(rewriter, loc, value, shiftAmount);
  }

  if (matchPattern(accumulator, mlir::m_Zero()))
    return value;

  constexpr bool isDisjoint = true;
  return LLVM::OrOp::create(rewriter, loc, accumulator, value, isDisjoint);
}

template <typename BaseOp>
struct AMDGPUMakeDmaBaseLowering : public ConvertOpToLLVMPattern<BaseOp> {
  using ConvertOpToLLVMPattern<BaseOp>::ConvertOpToLLVMPattern;
  using Adaptor = typename ConvertOpToLLVMPattern<BaseOp>::OpAdaptor;

  AMDGPUMakeDmaBaseLowering(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<BaseOp>(converter), chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(BaseOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (chipset < kGfx1250)
      return op->emitOpError("make_dma_base is only supported on gfx1250");

    Location loc = op.getLoc();

    constexpr int32_t constlen = 4;
    Value consts[constlen];
    for (int64_t i = 0; i < constlen; ++i)
      consts[i] = createI32Constant(rewriter, loc, i);

    constexpr int32_t sgprslen = constlen;
    Value sgprs[sgprslen];
    for (int64_t i = 0; i < sgprslen; ++i) {
      sgprs[i] = consts[0];
    }

    sgprs[0] = consts[1];

    if constexpr (BaseOp::isGather()) {
      sgprs[0] = setValueAtOffset(rewriter, loc, sgprs[0], consts[1], 30);

      auto type = cast<TDMGatherBaseType>(op.getResult().getType());
      Type indexType = type.getIndexType();
      unsigned indexSize = indexType.getIntOrFloatBitWidth();
      assert(llvm::is_contained({16u, 32u}, indexSize) &&
             "expected index_size to be 16 or 32");
      unsigned idx = (indexSize / 16) - 1;

      if (idx)
        sgprs[0] = setValueAtOffset(rewriter, loc, sgprs[0], consts[1], 31);
    }

    ValueRange ldsIndices = adaptor.getLdsIndices();
    Value lds = adaptor.getLds();
    auto ldsMemRefType = cast<MemRefType>(op.getLds().getType());

    Value ldsPtr = ConvertToLLVMPattern::getStridedElementPtr(
        rewriter, loc, ldsMemRefType, lds, ldsIndices);

    ValueRange globalIndices = adaptor.getGlobalIndices();
    Value global = adaptor.getGlobal();
    auto globalMemRefType = cast<MemRefType>(op.getGlobal().getType());

    Value globalPtr = ConvertToLLVMPattern::getStridedElementPtr(
        rewriter, loc, globalMemRefType, global, globalIndices);

    Type i32 = rewriter.getI32Type();
    Type i64 = rewriter.getI64Type();

    sgprs[1] = LLVM::PtrToIntOp::create(rewriter, loc, i32, ldsPtr);
    Value castForGlobalAddr =
        LLVM::PtrToIntOp::create(rewriter, loc, i64, globalPtr);

    sgprs[2] = LLVM::TruncOp::create(rewriter, loc, i32, castForGlobalAddr);

    Value shift = LLVM::LShrOp::create(rewriter, loc, castForGlobalAddr,
                                       createI64Constant(rewriter, loc, 32));

    Value highHalf = LLVM::TruncOp::create(rewriter, loc, i32, shift);

    Value mask = createI32Constant(rewriter, loc, (1ull << 25) - 1);
    highHalf = LLVM::AndOp::create(rewriter, loc, highHalf, mask);

    sgprs[3] = setValueAtOffset(rewriter, loc, highHalf, consts[2], 30);

    Type v4i32 = this->typeConverter->convertType(VectorType::get(4, i32));
    assert(v4i32 && "expected type conversion to succeed");
    Value result = LLVM::PoisonOp::create(rewriter, loc, v4i32);

    for (auto [sgpr, constant] : llvm::zip_equal(sgprs, consts))
      result =
          LLVM::InsertElementOp::create(rewriter, loc, result, sgpr, constant);

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename DescriptorOp>
struct AMDGPULowerDescriptor : public ConvertOpToLLVMPattern<DescriptorOp> {
  using ConvertOpToLLVMPattern<DescriptorOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename ConvertOpToLLVMPattern<DescriptorOp>::OpAdaptor;

  AMDGPULowerDescriptor(const LLVMTypeConverter &converter, Chipset chipset)
      : ConvertOpToLLVMPattern<DescriptorOp>(converter), chipset(chipset) {}
  Chipset chipset;

  Value getDGroup0(OpAdaptor adaptor) const { return adaptor.getBase(); }

  Value setWorkgroupMask(DescriptorOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Value sgpr0) const {
    Value mask = op.getWorkgroupMask();
    if (!mask)
      return sgpr0;

    Type i16 = rewriter.getI16Type();
    mask = LLVM::BitcastOp::create(rewriter, loc, i16, mask);
    Type i32 = rewriter.getI32Type();
    Value extendedMask = LLVM::ZExtOp::create(rewriter, loc, i32, mask);
    return setValueAtOffset(rewriter, loc, sgpr0, extendedMask, 0);
  }

  Value setDataSize(DescriptorOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, Location loc,
                    Value sgpr0, ArrayRef<Value> consts) const {
    unsigned elementTypeWidthInBits = op.getElementTypeWidth();
    assert(llvm::is_contained({8u, 16u, 32u, 64u}, elementTypeWidthInBits) &&
           "expected type width to be 8, 16, 32, or 64.");
    int64_t idx = llvm::Log2_32(elementTypeWidthInBits / 8);
    Value size = consts[idx];
    return setValueAtOffset(rewriter, loc, sgpr0, size, 16);
  }

  Value setAtomicBarrier(DescriptorOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Value sgpr0, ArrayRef<Value> consts) const {
    if (!adaptor.getAtomicBarrierAddress())
      return sgpr0;

    return setValueAtOffset(rewriter, loc, sgpr0, consts[1], 18);
  }

  Value setIterateEnable(DescriptorOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Value sgpr0, ArrayRef<Value> consts) const {
    if (!adaptor.getGlobalIncrement())
      return sgpr0;

    // Value is ignored when in gather mode.
    // TODO: emit error earlier?
    return setValueAtOffset(rewriter, loc, sgpr0, consts[1], 19);
  }

  Value setPadEnable(DescriptorOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Location loc,
                     Value sgpr0, ArrayRef<Value> consts) const {
    if (!op.getPadAmount())
      return sgpr0;

    return setValueAtOffset(rewriter, loc, sgpr0, consts[1], 20);
  }

  Value setEarlyTimeout(DescriptorOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter, Location loc,
                        Value sgpr0, ArrayRef<Value> consts) const {
    if (!op.getWorkgroupMask())
      return sgpr0;

    return setValueAtOffset(rewriter, loc, sgpr0, consts[1], 21);
  }

  Value setPadInterval(DescriptorOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter, Location loc,
                       Value sgpr0, ArrayRef<Value> consts) const {
    if (!op.getPadAmount())
      return sgpr0;

    // pre-condition: padInterval can be a power of two between 2 and 256.
    // TODO: Validation if the value breaks the pre-condition.
    // If the pre-condition fails, there is a possibility of
    // affecting the higher bits. In a following PR implement
    // RuntimeVerifiableOpInterface that instruments conditions that need to be
    // checked at runtime.
    IntegerType i32 = rewriter.getI32Type();
    Value padInterval = adaptor.getPadInterval();
    padInterval = LLVM::CountTrailingZerosOp::create(rewriter, loc, i32,
                                                     padInterval, false);
    padInterval = LLVM::SubOp::create(rewriter, loc, padInterval, consts[1]);
    // post-condition: padInterval can be a value between 0 and 7.
    return setValueAtOffset(rewriter, loc, sgpr0, padInterval, 22);
  }

  Value setPadAmount(DescriptorOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Location loc,
                     Value sgpr0, ArrayRef<Value> consts) const {
    if (!op.getPadAmount())
      return sgpr0;

    // pre-condition: padAmount is a value between 1-128.
    // TODO: Validation if the value breaks the pre-condition.
    // If the pre-condition fails, there is a possibility of
    // affecting the higher bits. In a following PR implement
    // RuntimeVerifiableOpInterface that instruments conditions that need to be
    // checked at runtime.
    Value padAmount = adaptor.getPadAmount();
    padAmount = LLVM::SubOp::create(rewriter, loc, padAmount, consts[1]);
    // post-condition: padAmount is a value between 0-127.
    return setValueAtOffset(rewriter, loc, sgpr0, padAmount, 25);
  }

  Value setAtomicBarrierAddress(DescriptorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter,
                                Location loc, Value sgpr1,
                                ArrayRef<Value> consts) const {
    if (!adaptor.getAtomicBarrierAddress())
      return sgpr1;

    Value atomicBarrierAddress = adaptor.getAtomicBarrierAddress();
    auto barrierAddressTy =
        cast<MemRefType>(op.getAtomicBarrierAddress().getType());
    ValueRange atomicBarrierIndices = adaptor.getAtomicBarrierIndices();
    atomicBarrierAddress = ConvertToLLVMPattern::getStridedElementPtr(
        rewriter, loc, barrierAddressTy, atomicBarrierAddress,
        atomicBarrierIndices);
    IntegerType i32 = rewriter.getI32Type();
    // pre-condition: atomicBarrierAddress is aligned to 8 bytes which implies
    // that the 3 LSBs are zero.
    // TODO: Validation if the value breaks the pre-condition.
    // In a following PR implement RuntimeVerifiableOpInterface
    // that instruments conditions that need to be checked at runtime.
    atomicBarrierAddress =
        LLVM::PtrToIntOp::create(rewriter, loc, i32, atomicBarrierAddress);
    atomicBarrierAddress =
        LLVM::LShrOp::create(rewriter, loc, atomicBarrierAddress, consts[3]);
    Value mask = createI32Constant(rewriter, loc, 0xFFFF);
    atomicBarrierAddress =
        LLVM::AndOp::create(rewriter, loc, atomicBarrierAddress, mask);
    return setValueAtOffset(rewriter, loc, sgpr1, atomicBarrierAddress, 32);
  }

  std::pair<Value, Value> setTensorDimX(DescriptorOp op, OpAdaptor adaptor,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc, Value sgpr1, Value sgpr2,
                                        ArrayRef<Value> consts, uint64_t dimX,
                                        uint32_t offset) const {
    ArrayRef<int64_t> globalStaticSizes = adaptor.getGlobalStaticSizes();
    ValueRange globalDynamicSizes = adaptor.getGlobalDynamicSizes();
    SmallVector<OpFoldResult> mixedGlobalSizes =
        getMixedValues(globalStaticSizes, globalDynamicSizes, rewriter);
    if (mixedGlobalSizes.size() <= dimX)
      return {sgpr1, sgpr2};

    OpFoldResult tensorDimXOpFoldResult = *(mixedGlobalSizes.rbegin() + dimX);
    // pre-condition: tensorDimX is less than 2^32-1
    // TODO: Validation if the value breaks the pre-condition.
    // In a following PR implement RuntimeVerifiableOpInterface that instruments
    // conditions that need to be checked at runtime. This could also be fixed
    // by saying that mixedGlobalSizes is a DynamicI32List.
    Value tensorDimX;
    if (auto attr = dyn_cast<Attribute>(tensorDimXOpFoldResult)) {
      tensorDimX =
          createI32Constant(rewriter, loc, cast<IntegerAttr>(attr).getInt());
    } else {
      IntegerType i32 = rewriter.getI32Type();
      tensorDimX = cast<Value>(tensorDimXOpFoldResult);
      tensorDimX = LLVM::TruncOp::create(rewriter, loc, i32, tensorDimX);
    }

    sgpr1 = setValueAtOffset(rewriter, loc, sgpr1, tensorDimX, offset);

    Value c16 = createI32Constant(rewriter, loc, 16);
    Value tensorDimXHigh = LLVM::LShrOp::create(rewriter, loc, tensorDimX, c16);
    sgpr2 = setValueAtOffset(rewriter, loc, sgpr2, tensorDimXHigh, offset + 16);
    return {sgpr1, sgpr2};
  }

  std::pair<Value, Value> setTensorDim0(DescriptorOp op, OpAdaptor adaptor,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc, Value sgpr1, Value sgpr2,
                                        ArrayRef<Value> consts) const {
    return setTensorDimX(op, adaptor, rewriter, loc, sgpr1, sgpr2, consts, 0,
                         48);
  }

  std::pair<Value, Value> setTensorDim1(DescriptorOp op, OpAdaptor adaptor,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc, Value sgpr2, Value sgpr3,
                                        ArrayRef<Value> consts) const {
    return setTensorDimX(op, adaptor, rewriter, loc, sgpr2, sgpr3, consts, 1,
                         80);
  }

  Value setTileDimX(DescriptorOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, Location loc,
                    Value sgpr, ArrayRef<Value> consts, size_t dimX,
                    int64_t offset) const {
    ArrayRef<int64_t> sharedStaticSizes = adaptor.getSharedStaticSizes();
    ValueRange sharedDynamicSizes = adaptor.getSharedDynamicSizes();
    SmallVector<OpFoldResult> mixedSharedSizes =
        getMixedValues(sharedStaticSizes, sharedDynamicSizes, rewriter);
    if (mixedSharedSizes.size() <= dimX)
      return sgpr;

    OpFoldResult tileDimXOpFoldResult = *(mixedSharedSizes.rbegin() + dimX);
    // pre-condition: tileDimX is less than 2^16-1
    // TODO: Validation if the value breaks the pre-condition.
    // If the pre-condition fails, there is a possibility of
    // affecting the higher bits. In a following PR implement
    // RuntimeVerifiableOpInterface that instruments conditions that need to be
    // checked at runtime. This could also be fixed by saying that
    // mixedSharedSizes is a DynamicI16List.
    Value tileDimX;
    if (auto attr = dyn_cast<Attribute>(tileDimXOpFoldResult)) {
      tileDimX =
          createI32Constant(rewriter, loc, cast<IntegerAttr>(attr).getInt());
    } else {
      IntegerType i32 = rewriter.getI32Type();
      tileDimX = cast<Value>(tileDimXOpFoldResult);
      tileDimX = LLVM::TruncOp::create(rewriter, loc, i32, tileDimX);
    }

    return setValueAtOffset(rewriter, loc, sgpr, tileDimX, offset);
  }

  Value setTileDim0(DescriptorOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, Location loc,
                    Value sgpr3, ArrayRef<Value> consts) const {
    return setTileDimX(op, adaptor, rewriter, loc, sgpr3, consts, 0, 112);
  }

  Value setTileDim1(DescriptorOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, Location loc,
                    Value sgpr4, ArrayRef<Value> consts) const {
    return setTileDimX(op, adaptor, rewriter, loc, sgpr4, consts, 1, 128);
  }

  Value setValidIndices(DescriptorOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter, Location loc,
                        Value sgpr4, ArrayRef<Value> consts) const {
    auto type = cast<VectorType>(op.getIndices().getType());
    ArrayRef<int64_t> shape = type.getShape();
    assert(shape.size() == 1 && "expected shape to be of rank 1.");
    unsigned length = shape.back();
    assert(0 < length && length <= 16 && "expected length to be at most 16.");
    Value value = createI32Constant(rewriter, loc, length);
    return setValueAtOffset(rewriter, loc, sgpr4, value, 128);
  }

  Value setTileDim1OrValidIndices(DescriptorOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc, Value sgpr4,
                                  ArrayRef<Value> consts) const {
    if constexpr (DescriptorOp::isGather())
      return setValidIndices(op, adaptor, rewriter, loc, sgpr4, consts);
    return setTileDim1(op, adaptor, rewriter, loc, sgpr4, consts);
  }

  Value setTileDim2(DescriptorOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, Location loc,
                    Value sgpr4, ArrayRef<Value> consts) const {
    // Value is ignored when in gather mode.
    if constexpr (DescriptorOp::isGather())
      return sgpr4;
    return setTileDimX(op, adaptor, rewriter, loc, sgpr4, consts, 2, 144);
  }

  std::pair<Value, Value>
  setTensorDimXStride(DescriptorOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Location loc,
                      Value sgprY, Value sgprZ, ArrayRef<Value> consts,
                      size_t dimX, int64_t offset) const {
    ArrayRef<int64_t> globalStaticStrides = adaptor.getGlobalStaticStrides();
    ValueRange globalDynamicStrides = adaptor.getGlobalDynamicStrides();
    SmallVector<OpFoldResult> mixedGlobalStrides =
        getMixedValues(globalStaticStrides, globalDynamicStrides, rewriter);

    if (mixedGlobalStrides.size() <= (dimX + 1))
      return {sgprY, sgprZ};

    OpFoldResult tensorDimXStrideOpFoldResult =
        *(mixedGlobalStrides.rbegin() + dimX + 1);
    // pre-condition: tensorDimXStride is less than 2^48-1
    // TODO: Validation if the value breaks the pre-condition.
    // In a following PR implement RuntimeVerifiableOpInterface that instruments
    // conditions that need to be checked at runtime.
    Value tensorDimXStride;
    if (auto attr = dyn_cast<Attribute>(tensorDimXStrideOpFoldResult))
      tensorDimXStride =
          createI64Constant(rewriter, loc, cast<IntegerAttr>(attr).getInt());
    else
      tensorDimXStride = cast<Value>(tensorDimXStrideOpFoldResult);

    constexpr int64_t first48bits = (1ll << 48) - 1;
    Value mask = createI64Constant(rewriter, loc, first48bits);
    tensorDimXStride =
        LLVM::AndOp::create(rewriter, loc, mask, tensorDimXStride);
    IntegerType i32 = rewriter.getI32Type();
    Value tensorDimXStrideLow =
        LLVM::TruncOp::create(rewriter, loc, i32, tensorDimXStride);
    sgprY = setValueAtOffset(rewriter, loc, sgprY, tensorDimXStrideLow, offset);

    int64_t shift = (offset % 32) == 0 ? 32 : offset % 32;
    Value shiftVal = createI64Constant(rewriter, loc, shift);
    Value tensorDimXStrideHigh =
        LLVM::LShrOp::create(rewriter, loc, tensorDimXStride, shiftVal);
    tensorDimXStrideHigh =
        LLVM::TruncOp::create(rewriter, loc, i32, tensorDimXStrideHigh);
    sgprZ = setValueAtOffset(rewriter, loc, sgprZ, tensorDimXStrideHigh,
                             offset + shift);
    return {sgprY, sgprZ};
  }

  std::pair<Value, Value>
  setTensorDim0Stride(DescriptorOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Location loc,
                      Value sgpr5, Value sgpr6, ArrayRef<Value> consts) const {
    return setTensorDimXStride(op, adaptor, rewriter, loc, sgpr5, sgpr6, consts,
                               0, 160);
  }

  std::pair<Value, Value>
  setTensorDim1Stride(DescriptorOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Location loc,
                      Value sgpr5, Value sgpr6, ArrayRef<Value> consts) const {
    // Value is ignored when in gather mode.
    if constexpr (DescriptorOp::isGather())
      return {sgpr5, sgpr6};
    return setTensorDimXStride(op, adaptor, rewriter, loc, sgpr5, sgpr6, consts,
                               1, 208);
  }

  Value getDGroup1(DescriptorOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter, Location loc,
                   ArrayRef<Value> consts) const {
    Value sgprs[8];
    for (int64_t i = 0; i < 8; ++i) {
      sgprs[i] = consts[0];
    }

    sgprs[0] = setWorkgroupMask(op, adaptor, rewriter, loc, sgprs[0]);
    sgprs[0] = setDataSize(op, adaptor, rewriter, loc, sgprs[0], consts);
    sgprs[0] = setAtomicBarrier(op, adaptor, rewriter, loc, sgprs[0], consts);
    sgprs[0] = setIterateEnable(op, adaptor, rewriter, loc, sgprs[0], consts);
    sgprs[0] = setPadEnable(op, adaptor, rewriter, loc, sgprs[0], consts);
    sgprs[0] = setEarlyTimeout(op, adaptor, rewriter, loc, sgprs[0], consts);
    sgprs[0] = setPadInterval(op, adaptor, rewriter, loc, sgprs[0], consts);
    sgprs[0] = setPadAmount(op, adaptor, rewriter, loc, sgprs[0], consts);

    sgprs[1] =
        setAtomicBarrierAddress(op, adaptor, rewriter, loc, sgprs[1], consts);
    std::tie(sgprs[1], sgprs[2]) =
        setTensorDim0(op, adaptor, rewriter, loc, sgprs[1], sgprs[2], consts);
    std::tie(sgprs[2], sgprs[3]) =
        setTensorDim1(op, adaptor, rewriter, loc, sgprs[2], sgprs[3], consts);

    sgprs[3] = setTileDim0(op, adaptor, rewriter, loc, sgprs[3], consts);
    sgprs[4] =
        setTileDim1OrValidIndices(op, adaptor, rewriter, loc, sgprs[4], consts);
    sgprs[4] = setTileDim2(op, adaptor, rewriter, loc, sgprs[4], consts);
    std::tie(sgprs[5], sgprs[6]) = setTensorDim0Stride(
        op, adaptor, rewriter, loc, sgprs[5], sgprs[6], consts);
    std::tie(sgprs[6], sgprs[7]) = setTensorDim1Stride(
        op, adaptor, rewriter, loc, sgprs[6], sgprs[7], consts);

    IntegerType i32 = rewriter.getI32Type();
    Type v8i32 = this->typeConverter->convertType(VectorType::get(8, i32));
    assert(v8i32 && "expected type conversion to succeed");
    Value dgroup1 = LLVM::PoisonOp::create(rewriter, loc, v8i32);

    for (auto [sgpr, constant] : llvm::zip_equal(sgprs, consts)) {
      dgroup1 =
          LLVM::InsertElementOp::create(rewriter, loc, dgroup1, sgpr, constant);
    }

    return dgroup1;
  }

  Value setTensorDimX(DescriptorOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Location loc,
                      Value sgpr0, ArrayRef<Value> consts, int64_t dimX,
                      int64_t offset) const {
    ArrayRef<int64_t> globalStaticSizes = adaptor.getGlobalStaticSizes();
    ValueRange globalDynamicSizes = adaptor.getGlobalDynamicSizes();
    SmallVector<OpFoldResult> mixedGlobalSizes =
        getMixedValues(globalStaticSizes, globalDynamicSizes, rewriter);
    if (mixedGlobalSizes.size() <= static_cast<unsigned long>(dimX))
      return sgpr0;

    OpFoldResult tensorDimXOpFoldResult = *(mixedGlobalSizes.rbegin() + dimX);
    Value tensorDimX;
    if (auto attr = dyn_cast<Attribute>(tensorDimXOpFoldResult)) {
      tensorDimX =
          createI32Constant(rewriter, loc, cast<IntegerAttr>(attr).getInt());
    } else {
      IntegerType i32 = rewriter.getI32Type();
      tensorDimX = cast<Value>(tensorDimXOpFoldResult);
      tensorDimX = LLVM::TruncOp::create(rewriter, loc, i32, tensorDimX);
    }

    return setValueAtOffset(rewriter, loc, sgpr0, tensorDimX, offset);
  }

  Value setTensorDim2(DescriptorOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Location loc,
                      Value sgpr0, ArrayRef<Value> consts) const {
    return setTensorDimX(op, adaptor, rewriter, loc, sgpr0, consts, 2, 0);
  }

  Value truncateAndSetValueAtOffset(ConversionPatternRewriter &rewriter,
                                    Location loc, Value accumulator,
                                    Value value, int64_t shift) const {

    IntegerType i32 = rewriter.getI32Type();
    value = LLVM::TruncOp::create(rewriter, loc, i32, value);
    return setValueAtOffset(rewriter, loc, accumulator, value, shift);
  }

  Value setLDSAddrIncrement(DescriptorOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter, Location loc,
                            Value sgpr1, ArrayRef<Value> consts,
                            int64_t offset) const {
    Value ldsAddrIncrement = adaptor.getLdsIncrement();
    return setValueAtOffset(rewriter, loc, sgpr1, ldsAddrIncrement, offset);
  }

  std::pair<Value, Value>
  setGlobalAddrIncrement(DescriptorOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Value sgpr2, Value sgpr3, ArrayRef<Value> consts,
                         int64_t offset) const {
    Value globalAddrIncrement = adaptor.getGlobalIncrement();
    sgpr2 = truncateAndSetValueAtOffset(rewriter, loc, sgpr2,
                                        globalAddrIncrement, offset);
    Value shift = createI64Constant(rewriter, loc, 32);
    globalAddrIncrement =
        LLVM::LShrOp::create(rewriter, loc, globalAddrIncrement, shift);
    constexpr int64_t first16BitsHigh = (1ll << 16) - 1;
    sgpr3 = truncateAndSetValueAtOffset(rewriter, loc, sgpr3,
                                        globalAddrIncrement, offset + 32);
    Value mask = createI32Constant(rewriter, loc, first16BitsHigh);
    sgpr3 = LLVM::AndOp::create(rewriter, loc, sgpr3, mask);
    return {sgpr2, sgpr3};
  }

  Value setTensorDim3OrLDSAddrIncrement(DescriptorOp op, OpAdaptor adaptor,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc, Value sgpr1,
                                        ArrayRef<Value> consts) const {
    Value ldsIncrement = op.getLdsIncrement();
    constexpr int64_t dim = 3;
    constexpr int64_t offset = 32;
    if (!ldsIncrement)
      return setTensorDimX(op, adaptor, rewriter, loc, sgpr1, consts, dim,
                           offset);
    return setLDSAddrIncrement(op, adaptor, rewriter, loc, sgpr1, consts,
                               offset);
  }

  std::pair<Value, Value> setTensorDim2StrideOrGlobalAddrIncrement(
      DescriptorOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
      Location loc, Value sgpr2, Value sgpr3, ArrayRef<Value> consts) const {
    Value globalIncrement = op.getGlobalIncrement();
    constexpr int32_t dim = 2;
    constexpr int32_t offset = 64;
    if (!globalIncrement)
      return setTensorDimXStride(op, adaptor, rewriter, loc, sgpr2, sgpr3,
                                 consts, dim, offset);
    return setGlobalAddrIncrement(op, adaptor, rewriter, loc, sgpr2, sgpr3,
                                  consts, offset);
  }

  Value setIterateCount(DescriptorOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter, Location loc,
                        Value sgpr3, ArrayRef<Value> consts,
                        int32_t offset) const {
    Value iterationCount = adaptor.getIterationCount();
    IntegerType i32 = rewriter.getI32Type();
    // pre-condition: iterationCount is in the inclusive interval [1, 256].
    // TODO: validation if the value breaks the pre-condition.
    // If the pre-condition fails, there is a possibility of
    // affecting the higher bits. In a following PR implement
    // RuntimeVerifiableOpInterface that instruments conditions that need to be
    // checked at runtime.
    iterationCount = LLVM::TruncOp::create(rewriter, loc, i32, iterationCount);
    iterationCount =
        LLVM::SubOp::create(rewriter, loc, iterationCount, consts[1]);
    return setValueAtOffset(rewriter, loc, sgpr3, iterationCount, offset);
  }

  Value setTileDim3OrIterateCount(DescriptorOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc, Value sgpr3,
                                  ArrayRef<Value> consts) const {
    Value iterateCount = op.getIterationCount();
    constexpr int32_t dim = 2;
    constexpr int32_t offset = 112;
    if (!iterateCount)
      return setTileDimX(op, adaptor, rewriter, loc, sgpr3, consts, dim,
                         offset);

    return setIterateCount(op, adaptor, rewriter, loc, sgpr3, consts, offset);
  }

  Value getDGroup2(DescriptorOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter, Location loc,
                   ArrayRef<Value> consts) const {
    if constexpr (DescriptorOp::isGather())
      return getDGroup2Gather(op, adaptor, rewriter, loc, consts);
    return getDGroup2NonGather(op, adaptor, rewriter, loc, consts);
  }

  Value getDGroup2NonGather(DescriptorOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter, Location loc,
                            ArrayRef<Value> consts) const {
    IntegerType i32 = rewriter.getI32Type();
    Type v4i32 = this->typeConverter->convertType(VectorType::get(4, i32));
    assert(v4i32 && "expected type conversion to succeed.");

    bool onlyNeedsTwoDescriptors = !op.getLdsIncrement() && op.getRank() <= 2;
    if (onlyNeedsTwoDescriptors)
      return LLVM::ZeroOp::create(rewriter, loc, v4i32);

    constexpr int64_t sgprlen = 4;
    Value sgprs[sgprlen];
    for (int i = 0; i < sgprlen; ++i)
      sgprs[i] = consts[0];

    sgprs[0] = setTensorDim2(op, adaptor, rewriter, loc, sgprs[0], consts);
    sgprs[1] = setTensorDim3OrLDSAddrIncrement(op, adaptor, rewriter, loc,
                                               sgprs[1], consts);
    std::tie(sgprs[2], sgprs[3]) = setTensorDim2StrideOrGlobalAddrIncrement(
        op, adaptor, rewriter, loc, sgprs[2], sgprs[3], consts);
    sgprs[3] =
        setTileDim3OrIterateCount(op, adaptor, rewriter, loc, sgprs[3], consts);

    Value dgroup2 = LLVM::PoisonOp::create(rewriter, loc, v4i32);
    for (auto [sgpr, constant] : llvm::zip(sgprs, consts))
      dgroup2 =
          LLVM::InsertElementOp::create(rewriter, loc, dgroup2, sgpr, constant);

    return dgroup2;
  }

  Value getGatherIndices(DescriptorOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter, Location loc,
                         ArrayRef<Value> consts, bool firstHalf) const {
    IntegerType i32 = rewriter.getI32Type();
    Type v4i32 = this->typeConverter->convertType(VectorType::get(4, i32));
    assert(v4i32 && "expected type conversion to succeed.");

    Value indices = adaptor.getIndices();
    auto vectorType = cast<VectorType>(indices.getType());
    unsigned length = vectorType.getShape().back();
    Type elementType = vectorType.getElementType();
    unsigned maxLength = elementType == i32 ? 4 : 8;
    int32_t offset = firstHalf ? 0 : maxLength;
    unsigned discountedLength =
        std::max(static_cast<int32_t>(length - offset), 0);

    unsigned targetSize = std::min(maxLength, discountedLength);

    SmallVector<Value> indicesVector;
    for (unsigned i = offset; i < targetSize + offset; ++i) {
      Value idx;
      if (i < consts.size())
        idx = consts[i];
      else
        idx = createI32Constant(rewriter, loc, i);
      Value elem = LLVM::ExtractElementOp::create(rewriter, loc, indices, idx);
      indicesVector.push_back(elem);
    }

    SmallVector<Value> indicesI32Vector;
    if (elementType == i32) {
      indicesI32Vector = indicesVector;
    } else {
      for (unsigned i = 0; i < targetSize; ++i) {
        Value index = indicesVector[i];
        indicesI32Vector.push_back(
            LLVM::ZExtOp::create(rewriter, loc, i32, index));
      }
      if ((targetSize % 2) != 0)
        // Add padding when not divisible by two.
        indicesI32Vector.push_back(consts[0]);
    }

    SmallVector<Value> indicesToInsert;
    if (elementType == i32) {
      indicesToInsert = indicesI32Vector;
    } else {
      unsigned size = indicesI32Vector.size() / 2;
      for (unsigned i = 0; i < size; ++i) {
        Value first = indicesI32Vector[2 * i];
        Value second = indicesI32Vector[2 * i + 1];
        Value joined = setValueAtOffset(rewriter, loc, first, second, 16);
        indicesToInsert.push_back(joined);
      }
    }

    Value dgroup = LLVM::PoisonOp::create(rewriter, loc, v4i32);
    for (auto [sgpr, constant] : llvm::zip_first(indicesToInsert, consts))
      dgroup =
          LLVM::InsertElementOp::create(rewriter, loc, dgroup, sgpr, constant);

    return dgroup;
  }

  Value getDGroup2Gather(DescriptorOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter, Location loc,
                         ArrayRef<Value> consts) const {
    return getGatherIndices(op, adaptor, rewriter, loc, consts, true);
  }

  std::pair<Value, Value>
  setTensorDim3Stride(DescriptorOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Location loc,
                      Value sgpr0, Value sgpr1, ArrayRef<Value> consts) const {
    constexpr int32_t dim = 3;
    constexpr int32_t offset = 0;
    return setTensorDimXStride(op, adaptor, rewriter, loc, sgpr0, sgpr1, consts,
                               dim, offset);
  }

  std::pair<Value, Value> setTensorDim4(DescriptorOp op, OpAdaptor adaptor,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc, Value sgpr1, Value sgpr2,
                                        ArrayRef<Value> consts) const {
    constexpr int32_t dim = 4;
    constexpr int32_t offset = 48;
    return setTensorDimX(op, adaptor, rewriter, loc, sgpr1, sgpr2, consts, dim,
                         offset);
  }

  Value setTileDim4(DescriptorOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter, Location loc,
                    Value sgpr2, ArrayRef<Value> consts) const {
    constexpr int32_t dim = 4;
    constexpr int32_t offset = 80;
    return setTileDimX(op, adaptor, rewriter, loc, sgpr2, consts, dim, offset);
  }

  Value getDGroup3(DescriptorOp op, OpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter, Location loc,
                   ArrayRef<Value> consts) const {
    if constexpr (DescriptorOp::isGather())
      return getDGroup3Gather(op, adaptor, rewriter, loc, consts);
    return getDGroup3NonGather(op, adaptor, rewriter, loc, consts);
  }

  Value getDGroup3NonGather(DescriptorOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter, Location loc,
                            ArrayRef<Value> consts) const {
    IntegerType i32 = rewriter.getI32Type();
    Type v4i32 = this->typeConverter->convertType(VectorType::get(4, i32));
    assert(v4i32 && "expected type conversion to succeed.");
    bool onlyNeedsTwoDescriptors = !op.getLdsIncrement() && op.getRank() <= 2;
    if (onlyNeedsTwoDescriptors)
      return LLVM::ZeroOp::create(rewriter, loc, v4i32);

    constexpr int32_t sgprlen = 4;
    Value sgprs[sgprlen];
    for (int i = 0; i < sgprlen; ++i)
      sgprs[i] = consts[0];

    std::tie(sgprs[0], sgprs[1]) = setTensorDim3Stride(
        op, adaptor, rewriter, loc, sgprs[0], sgprs[1], consts);
    std::tie(sgprs[1], sgprs[2]) =
        setTensorDim4(op, adaptor, rewriter, loc, sgprs[1], sgprs[2], consts);
    sgprs[2] = setTileDim4(op, adaptor, rewriter, loc, sgprs[2], consts);

    Value dgroup3 = LLVM::PoisonOp::create(rewriter, loc, v4i32);
    for (auto [sgpr, constant] : llvm::zip(sgprs, consts))
      dgroup3 =
          LLVM::InsertElementOp::create(rewriter, loc, dgroup3, sgpr, constant);

    return dgroup3;
  }

  Value getDGroup3Gather(DescriptorOp op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter, Location loc,
                         ArrayRef<Value> consts) const {
    return getGatherIndices(op, adaptor, rewriter, loc, consts, false);
  }

  LogicalResult
  matchAndRewrite(DescriptorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (chipset < kGfx1250)
      return op->emitOpError(
          "make_dma_descriptor is only supported on gfx1250");

    Location loc = op.getLoc();

    SmallVector<Value> consts;
    for (int64_t i = 0; i < 8; ++i)
      consts.push_back(createI32Constant(rewriter, loc, i));

    Value dgroup0 = this->getDGroup0(adaptor);
    Value dgroup1 = this->getDGroup1(op, adaptor, rewriter, loc, consts);
    Value dgroup2 = this->getDGroup2(op, adaptor, rewriter, loc, consts);
    Value dgroup3 = this->getDGroup3(op, adaptor, rewriter, loc, consts);
    SmallVector<Value> results = {dgroup0, dgroup1, dgroup2, dgroup3};
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct AMDGPUTensorLoadStoreOpLowering
    : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using Adaptor = typename ConvertOpToLLVMPattern<SourceOp>::OneToNOpAdaptor;
  AMDGPUTensorLoadStoreOpLowering(const LLVMTypeConverter &converter,
                                  Chipset chipset)
      : ConvertOpToLLVMPattern<SourceOp>(converter), chipset(chipset) {}
  Chipset chipset;

  LogicalResult
  matchAndRewrite(SourceOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (chipset < kGfx1250)
      return op->emitOpError("is only supported on gfx1250");

    ValueRange desc = adaptor.getDesc();
    rewriter.replaceOpWithNewOp<TargetOp>(op, desc[0], desc[1], desc[2],
                                          desc[3], /*cachePolicy=*/0,
                                          /*alias_scopes=*/nullptr,
                                          /*noalias_scopes=*/nullptr,
                                          /*tbaa=*/nullptr);
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
    amdgpu::populateCommonGPUTypeAndAttributeConversions(converter);
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

void mlir::amdgpu::populateCommonGPUTypeAndAttributeConversions(
    TypeConverter &typeConverter) {
  populateGpuMemorySpaceAttributeConversions(
      typeConverter, [](gpu::AddressSpace space) {
        switch (space) {
        case gpu::AddressSpace::Global:
          return ROCDL::ROCDLDialect::kGlobalMemoryAddressSpace;
        case gpu::AddressSpace::Workgroup:
          return ROCDL::ROCDLDialect::kSharedMemoryAddressSpace;
        case gpu::AddressSpace::Private:
          return ROCDL::ROCDLDialect::kPrivateMemoryAddressSpace;
        }
        llvm_unreachable("unknown address space enum value");
      });
}

void mlir::populateAMDGPUTypeAndAttributeConversions(
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
  typeConverter.addConversion([&](TDMBaseType type) -> Type {
    Type i32 = IntegerType::get(type.getContext(), 32);
    return typeConverter.convertType(VectorType::get(4, i32));
  });
  typeConverter.addConversion([&](TDMGatherBaseType type) -> Type {
    Type i32 = IntegerType::get(type.getContext(), 32);
    return typeConverter.convertType(VectorType::get(4, i32));
  });
  typeConverter.addConversion(
      [&](TDMDescriptorType type,
          SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        Type i32 = IntegerType::get(type.getContext(), 32);
        Type v4i32 = typeConverter.convertType(VectorType::get(4, i32));
        Type v8i32 = typeConverter.convertType(VectorType::get(8, i32));
        llvm::append_values(result, v4i32, v8i32, v4i32, v4i32);
        return success();
      });

  auto addUnrealizedCast = [](OpBuilder &builder, TypeRange types,
                              ValueRange inputs,
                              Location loc) -> SmallVector<Value> {
    // Only create unrealized_conversion_cast for TDMDescriptorType.
    // All other types which are not expected, should be
    // materialized by other target materialization functions.
    if (inputs.size() != 1)
      return {};

    if (!isa<TDMDescriptorType>(inputs[0].getType()))
      return {};

    auto cast = UnrealizedConversionCastOp::create(builder, loc, types, inputs);
    return cast.getResults();
  };

  typeConverter.addTargetMaterialization(addUnrealizedCast);
}

void mlir::populateAMDGPUToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   Chipset chipset) {
  populateAMDGPUTypeAndAttributeConversions(converter);
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
           AMDGPUDPPLowering, MemoryCounterWaitOpLowering, LDSBarrierOpLowering,
           SchedBarrierOpLowering, MFMAOpLowering, ScaledMFMAOpLowering,
           SparseMFMAOpLowering, WMMAOpLowering, ScaledWMMAOpLowering,
           ExtPackedFp8OpLowering, ScaledExtPackedMatrixOpLowering,
           ScaledExtPackedOpLowering, PackedScaledTruncOpLowering,
           PackedTrunc2xFp8OpLowering, PackedStochRoundFp8OpLowering,
           GatherToLDSOpLowering, TransposeLoadOpLowering,
           AMDGPUPermlaneLowering, AMDGPUMakeDmaBaseLowering<MakeDmaBaseOp>,
           AMDGPUMakeDmaBaseLowering<MakeGatherDmaBaseOp>,
           AMDGPULowerDescriptor<MakeDmaDescriptorOp>,
           AMDGPULowerDescriptor<MakeGatherDmaDescriptorOp>,
           AMDGPUTensorLoadStoreOpLowering<TensorLoadToLDSOp,
                                           ROCDL::TensorLoadToLDSOp>,
           AMDGPUTensorLoadStoreOpLowering<TensorStoreFromLDSOp,
                                           ROCDL::TensorStoreFromLDSOp>>(
          converter, chipset);
  patterns.add<AMDGPUSwizzleBitModeLowering>(converter);
}
