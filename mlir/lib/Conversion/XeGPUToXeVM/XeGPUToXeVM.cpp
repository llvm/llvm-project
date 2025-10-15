//===-- XeGPUToXeVM.cpp - XeGPU to XeVM dialect conversion ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/XeGPUToXeVM/XeGPUToXeVM.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

namespace mlir {
#define GEN_PASS_DEF_CONVERTXEGPUTOXEVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// TODO: Below are uArch dependent values, should move away from hardcoding
static constexpr int32_t systolicDepth{8};
static constexpr int32_t executionSize{16};

// Offsets to individual fields of the 8xi32 layout nd tensor descriptor.
enum class NdTdescOffset : uint32_t {
  BasePtr = 0,       // Base pointer (i64)
  BaseShapeW = 2,    // Base shape width (i32)
  BaseShapeH = 3,    // Base shape height (i32)
  TensorOffsetW = 4, // Tensor offset W (i32)
  TensorOffsetH = 5  // Tensor offset H (i32)
};

static int32_t getNumericXeVMAddrSpace(xegpu::MemorySpace xeGpuMemspace) {
  switch (xeGpuMemspace) {
  case xegpu::MemorySpace::Global:
    return static_cast<int>(xevm::AddrSpace::GLOBAL);
  case xegpu::MemorySpace::SLM:
    return static_cast<int>(xevm::AddrSpace::SHARED);
  }
  llvm_unreachable("Unknown XeGPU memory space");
}

// Get same bitwidth flat vector type of new element type.
static VectorType encodeVectorTypeTo(VectorType currentVecType,
                                     Type toElemType) {
  auto elemType = currentVecType.getElementType();
  auto currentBitWidth = elemType.getIntOrFloatBitWidth();
  auto newBitWidth = toElemType.getIntOrFloatBitWidth();
  const int size =
      currentVecType.getNumElements() * currentBitWidth / newBitWidth;
  return VectorType::get(size, toElemType);
}

static xevm::LoadCacheControl
translateLoadXeGPUCacheHint(std::optional<xegpu::CachePolicy> L1hint,
                            std::optional<xegpu::CachePolicy> L3hint) {
  auto L1hintVal = L1hint.value_or(xegpu::CachePolicy::UNCACHED);
  auto L3hintVal = L3hint.value_or(xegpu::CachePolicy::UNCACHED);
  switch (L1hintVal) {
  case xegpu::CachePolicy::CACHED:
    if (L3hintVal == xegpu::CachePolicy::CACHED)
      return xevm::LoadCacheControl::L1C_L2UC_L3C;
    else if (L3hintVal == xegpu::CachePolicy::UNCACHED)
      return xevm::LoadCacheControl::L1C_L2UC_L3UC;
    else
      llvm_unreachable("Unsupported cache control.");
  case xegpu::CachePolicy::UNCACHED:
    if (L3hintVal == xegpu::CachePolicy::CACHED)
      return xevm::LoadCacheControl::L1UC_L2UC_L3C;
    else if (L3hintVal == xegpu::CachePolicy::UNCACHED)
      return xevm::LoadCacheControl::L1UC_L2UC_L3UC;
    else
      llvm_unreachable("Unsupported cache control.");
  case xegpu::CachePolicy::STREAMING:
    if (L3hintVal == xegpu::CachePolicy::CACHED)
      return xevm::LoadCacheControl::L1S_L2UC_L3C;
    else if (L3hintVal == xegpu::CachePolicy::UNCACHED)
      return xevm::LoadCacheControl::L1S_L2UC_L3UC;
    else
      llvm_unreachable("Unsupported cache control.");
  case xegpu::CachePolicy::READ_INVALIDATE:
    return xevm::LoadCacheControl::INVALIDATE_READ;
  default:
    llvm_unreachable("Unsupported cache control.");
  }
}

static xevm::StoreCacheControl
translateStoreXeGPUCacheHint(std::optional<xegpu::CachePolicy> L1hint,
                             std::optional<xegpu::CachePolicy> L3hint) {
  auto L1hintVal = L1hint.value_or(xegpu::CachePolicy::UNCACHED);
  auto L3hintVal = L3hint.value_or(xegpu::CachePolicy::UNCACHED);
  switch (L1hintVal) {
  case xegpu::CachePolicy::UNCACHED:
    if (L3hintVal == xegpu::CachePolicy::UNCACHED)
      return xevm::StoreCacheControl::L1UC_L2UC_L3UC;
    else if (L3hintVal == xegpu::CachePolicy::WRITE_BACK)
      return xevm::StoreCacheControl::L1UC_L2UC_L3WB;
    else
      llvm_unreachable("Unsupported cache control.");
  case xegpu::CachePolicy::STREAMING:
    if (L3hintVal == xegpu::CachePolicy::UNCACHED)
      return xevm::StoreCacheControl::L1S_L2UC_L3UC;
    else if (L3hintVal == xegpu::CachePolicy::WRITE_BACK)
      return xevm::StoreCacheControl::L1S_L2UC_L3WB;
    else
      llvm_unreachable("Unsupported cache control.");
  case xegpu::CachePolicy::WRITE_BACK:
    if (L3hintVal == xegpu::CachePolicy::UNCACHED)
      return xevm::StoreCacheControl::L1WB_L2UC_L3UC;
    else if (L3hintVal == xegpu::CachePolicy::WRITE_BACK)
      return xevm::StoreCacheControl::L1WB_L2UC_L3WB;
    else
      llvm_unreachable("Unsupported cache control.");
  case xegpu::CachePolicy::WRITE_THROUGH:
    if (L3hintVal == xegpu::CachePolicy::UNCACHED)
      return xevm::StoreCacheControl::L1WT_L2UC_L3UC;
    else if (L3hintVal == xegpu::CachePolicy::WRITE_BACK)
      return xevm::StoreCacheControl::L1WT_L2UC_L3WB;
    else
      llvm_unreachable("Unsupported cache control.");
  default:
    llvm_unreachable("Unsupported cache control.");
  }
}

class CreateNdDescToXeVMPattern
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op,
                  xegpu::CreateNdDescOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
    if (mixedOffsets.size() != 0)
      return rewriter.notifyMatchFailure(op, "Offsets not supported.");
    auto loc = op.getLoc();
    auto source = op.getSource();
    // Op is lowered to a code sequence that populates payload.
    // Payload is a 8xi32 vector. Offset to individual fields are defined in
    // NdTdescOffset enum.
    Type payloadElemTy = rewriter.getI32Type();
    VectorType payloadTy = VectorType::get(8, payloadElemTy);
    Type i64Ty = rewriter.getI64Type();
    // 4xi64 view is used for inserting the base pointer.
    VectorType payloadI64Ty = VectorType::get(4, i64Ty);
    // Initialize payload to zero.
    Value payload = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(payloadTy, IntegerAttr::get(payloadElemTy, 0)));

    Value baseAddr;
    Value baseShapeW;
    Value baseShapeH;
    Value offsetW;
    Value offsetH;

    // Source can be a memref or a pointer (ui64, ui32, i64 or i32).
    SmallVector<OpFoldResult> mixedSizes = op.getMixedSizes();
    // Descriptor shape is expected to be 2D.
    int64_t rank = mixedSizes.size();
    if (rank != 2)
      return rewriter.notifyMatchFailure(op, "Expected 2D shape.");

    auto sourceTy = source.getType();
    auto sourceMemrefTy = dyn_cast<MemRefType>(sourceTy);
    // If source is a memref, we need to extract the aligned pointer as index.
    // Pointer type is passed as i32 or i64 by type converter.
    if (sourceMemrefTy) {
      if (!sourceMemrefTy.hasStaticShape()) {
        return rewriter.notifyMatchFailure(op, "Expected static memref shape.");
      }
      baseAddr =
          memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, source);
    } else {
      baseAddr = adaptor.getSource();
    }
    // Utility for creating offset values from op fold result.
    auto createOffset = [&](SmallVector<OpFoldResult> &ofrVec,
                            unsigned idx) -> Value {
      Value val = getValueOrCreateConstantIntOp(rewriter, loc, ofrVec[idx]);
      val = getValueOrCreateCastToIndexLike(rewriter, loc, payloadElemTy, val);
      return val;
    };
    // Offsets are not supported (0 is used).
    offsetW = arith::ConstantIntOp::create(rewriter, loc, payloadElemTy, 0);
    offsetH = arith::ConstantIntOp::create(rewriter, loc, payloadElemTy, 0);
    // Get shape values from op fold results.
    baseShapeW = createOffset(mixedSizes, 1);
    baseShapeH = createOffset(mixedSizes, 0);
    if (sourceMemrefTy) {
      // Cast index to i64.
      baseAddr = arith::IndexCastUIOp::create(rewriter, loc, i64Ty, baseAddr);
    } else if (baseAddr.getType() != i64Ty) {
      // Pointer type may be i32. Cast to i64 if needed.
      baseAddr = arith::ExtUIOp::create(rewriter, loc, i64Ty, baseAddr);
    }
    // Populate payload.
    Value payLoadAsI64 =
        vector::BitCastOp::create(rewriter, loc, payloadI64Ty, payload);
    payLoadAsI64 =
        vector::InsertOp::create(rewriter, loc, baseAddr, payLoadAsI64,
                                 static_cast<int>(NdTdescOffset::BasePtr));
    payload = vector::BitCastOp::create(rewriter, loc, payloadTy, payLoadAsI64);
    payload =
        vector::InsertOp::create(rewriter, loc, baseShapeW, payload,
                                 static_cast<int>(NdTdescOffset::BaseShapeW));
    payload =
        vector::InsertOp::create(rewriter, loc, baseShapeH, payload,
                                 static_cast<int>(NdTdescOffset::BaseShapeH));
    payload = vector::InsertOp::create(
        rewriter, loc, offsetW, payload,
        static_cast<int>(NdTdescOffset::TensorOffsetW));
    payload = vector::InsertOp::create(
        rewriter, loc, offsetH, payload,
        static_cast<int>(NdTdescOffset::TensorOffsetH));
    rewriter.replaceOp(op, payload);
    return success();
  }
};

template <
    typename OpType,
    typename = std::enable_if_t<llvm::is_one_of<
        OpType, xegpu::LoadNdOp, xegpu::StoreNdOp, xegpu::PrefetchNdOp>::value>>
class LoadStorePrefetchNdToXeVMPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mixedOffsets = op.getMixedOffsets();
    int64_t opOffsetsSize = mixedOffsets.size();
    if (opOffsetsSize != 2)
      return rewriter.notifyMatchFailure(op, "Expected 2D offsets.");
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();

    auto tdesc = adaptor.getTensorDesc();
    auto tdescTy = op.getTensorDescType();
    if (tdescTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "Expected 2D tensor descriptor.");
    auto elemType = tdescTy.getElementType();
    auto elemBitSize = elemType.getIntOrFloatBitWidth();
    if (elemBitSize % 8 != 0)
      return rewriter.notifyMatchFailure(
          op, "Expected element type bit width to be multiple of 8.");

    VectorType payloadI64Ty = VectorType::get(4, rewriter.getI64Type());
    Value payLoadAsI64 =
        vector::BitCastOp::create(rewriter, loc, payloadI64Ty, tdesc);
    Value basePtr = vector::ExtractOp::create(
        rewriter, loc, payLoadAsI64, static_cast<int>(NdTdescOffset::BasePtr));
    Value baseShapeW = vector::ExtractOp::create(
        rewriter, loc, tdesc, static_cast<int>(NdTdescOffset::BaseShapeW));
    Value baseShapeH = vector::ExtractOp::create(
        rewriter, loc, tdesc, static_cast<int>(NdTdescOffset::BaseShapeH));
    // Offsets are provided by the op.
    // convert them to i32.
    Value offsetW =
        getValueOrCreateConstantIntOp(rewriter, loc, mixedOffsets[1]);
    offsetW = getValueOrCreateCastToIndexLike(rewriter, loc,
                                              rewriter.getI32Type(), offsetW);
    Value offsetH =
        getValueOrCreateConstantIntOp(rewriter, loc, mixedOffsets[0]);
    offsetH = getValueOrCreateCastToIndexLike(rewriter, loc,
                                              rewriter.getI32Type(), offsetH);
    // Get address space from tensor descriptor memory space.
    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
    // Convert base pointer (i64) to LLVM pointer type.
    Value basePtrLLVM =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTypeLLVM, basePtr);
    // Compute element byte size and surface width in bytes.
    Value elemByteSize = arith::ConstantIntOp::create(
        rewriter, loc, rewriter.getI32Type(), elemBitSize / 8);
    Value surfaceW =
        arith::MulIOp::create(rewriter, loc, baseShapeW, elemByteSize);

    // Get tile sizes and vblocks from the tensor descriptor type.
    auto tileW = tdescTy.getDimSize(1);
    auto tileH = tdescTy.getDimSize(0);
    int32_t vblocks = tdescTy.getArrayLength();
    if constexpr (std::is_same_v<OpType, xegpu::StoreNdOp>) {
      Value src = adaptor.getValue();
      // If store value is a scalar, get value from op instead of adaptor.
      // Adaptor might have optimized away single element vector
      if (src.getType().isIntOrFloat()) {
        src = op.getValue();
      }
      VectorType srcVecTy = dyn_cast<VectorType>(src.getType());
      if (!srcVecTy)
        return rewriter.notifyMatchFailure(
            op, "Expected store value to be a vector type.");
      // Get flat vector type of integer type with matching element bit size.
      VectorType newSrcVecTy =
          encodeVectorTypeTo(srcVecTy, rewriter.getIntegerType(elemBitSize));
      if (srcVecTy != newSrcVecTy)
        src = vector::BitCastOp::create(rewriter, loc, newSrcVecTy, src);
      auto storeCacheControl =
          translateStoreXeGPUCacheHint(op.getL1Hint(), op.getL3Hint());
      xevm::BlockStore2dOp::create(
          rewriter, loc, basePtrLLVM, surfaceW, baseShapeH, surfaceW, offsetW,
          offsetH, elemBitSize, tileW, tileH, src,
          xevm::StoreCacheControlAttr::get(ctxt, storeCacheControl));
      rewriter.eraseOp(op);
    } else {
      auto loadCacheControl =
          translateLoadXeGPUCacheHint(op.getL1Hint(), op.getL3Hint());
      if constexpr (std::is_same_v<OpType, xegpu::PrefetchNdOp>) {
        xevm::BlockPrefetch2dOp::create(
            rewriter, loc, basePtrLLVM, surfaceW, baseShapeH, surfaceW, offsetW,
            offsetH, elemBitSize, tileW, tileH, vblocks,
            xevm::LoadCacheControlAttr::get(ctxt, loadCacheControl));
        rewriter.eraseOp(op);
      } else {
        VectorType dstVecTy = cast<VectorType>(op.getValue().getType());
        const bool vnni = op.getPacked().value_or(false);
        auto transposeValue = op.getTranspose();
        bool transpose =
            transposeValue.has_value() && transposeValue.value()[0] == 1;
        VectorType loadedTy = encodeVectorTypeTo(
            dstVecTy, vnni ? rewriter.getI32Type()
                           : rewriter.getIntegerType(elemBitSize));

        Value resultFlatVec = xevm::BlockLoad2dOp::create(
            rewriter, loc, loadedTy, basePtrLLVM, surfaceW, baseShapeH,
            surfaceW, offsetW, offsetH, elemBitSize, tileW, tileH, vblocks,
            transpose, vnni,
            xevm::LoadCacheControlAttr::get(ctxt, loadCacheControl));
        resultFlatVec = vector::BitCastOp::create(
            rewriter, loc,
            encodeVectorTypeTo(loadedTy, dstVecTy.getElementType()),
            resultFlatVec);
        rewriter.replaceOp(op, resultFlatVec);
      }
    }
    return success();
  }
};

// Add a builder that creates
// offset * elemByteSize + baseAddr
static Value addOffsetToBaseAddr(ConversionPatternRewriter &rewriter,
                                 Location loc, Value baseAddr, Value offset,
                                 int64_t elemByteSize) {
  Value byteSize = arith::ConstantIntOp::create(
      rewriter, loc, baseAddr.getType(), elemByteSize);
  Value byteOffset = arith::MulIOp::create(rewriter, loc, offset, byteSize);
  Value newAddr = arith::AddIOp::create(rewriter, loc, baseAddr, byteOffset);
  return newAddr;
}

template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<
              OpType, xegpu::LoadGatherOp, xegpu::StoreScatterOp>::value>>
class LoadStoreToXeVMPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value offset = adaptor.getOffsets();
    if (!offset)
      return rewriter.notifyMatchFailure(op, "Expected offset to be provided.");
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto tdescTy = op.getTensorDescType();
    Value basePtrI64;
    // Load result or Store valye Type can be vector or scalar.
    Type valOrResTy;
    if constexpr (std::is_same_v<OpType, xegpu::LoadGatherOp>)
      valOrResTy =
          this->getTypeConverter()->convertType(op.getResult().getType());
    else
      valOrResTy = adaptor.getValue().getType();
    VectorType valOrResVecTy = dyn_cast<VectorType>(valOrResTy);
    bool hasScalarVal = !valOrResVecTy;
    int64_t elemBitWidth =
        hasScalarVal ? valOrResTy.getIntOrFloatBitWidth()
                     : valOrResVecTy.getElementType().getIntOrFloatBitWidth();
    // Element type must be multiple of 8 bits.
    if (elemBitWidth % 8 != 0)
      return rewriter.notifyMatchFailure(
          op, "Expected element type bit width to be multiple of 8.");
    int64_t elemByteSize = elemBitWidth / 8;
    // Default memory space is global.
    LLVM::LLVMPointerType ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(xegpu::MemorySpace::Global));
    // If tensor descriptor is available, we use its memory space.
    if (tdescTy)
      ptrTypeLLVM = LLVM::LLVMPointerType::get(
          ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
    // Base pointer can come from source (load) or dest (store).
    // If they are memrefs, we use their memory space.
    if constexpr (std::is_same_v<OpType, xegpu::LoadGatherOp>) {
      basePtrI64 = adaptor.getSource();
      if (auto memRefTy = dyn_cast<MemRefType>(op.getSource().getType())) {
        auto addrSpace = memRefTy.getMemorySpaceAsInt();
        if (addrSpace != 0)
          ptrTypeLLVM = LLVM::LLVMPointerType::get(ctxt, addrSpace);
      }
    } else {
      basePtrI64 = adaptor.getDest();
      if (auto memRefTy = dyn_cast<MemRefType>(op.getDest().getType())) {
        auto addrSpace = memRefTy.getMemorySpaceAsInt();
        if (addrSpace != 0)
          ptrTypeLLVM = LLVM::LLVMPointerType::get(ctxt, addrSpace);
      }
    }
    // Base pointer is passed as i32 or i64 by adaptor, cast to i64 if needed.
    if (basePtrI64.getType() != rewriter.getI64Type()) {
      basePtrI64 = arith::ExtUIOp::create(rewriter, loc, rewriter.getI64Type(),
                                          basePtrI64);
    }
    Value mask = adaptor.getMask();
    if (dyn_cast<VectorType>(offset.getType())) {
      // Offset needs be scalar. Single element vector is converted to scalar
      // by type converter.
      return rewriter.notifyMatchFailure(op, "Expected offset to be a scalar.");
    } else {
      // If offset is provided, we add them to the base pointer.
      // Offset is in number of elements, we need to multiply by
      // element byte size.
      basePtrI64 =
          addOffsetToBaseAddr(rewriter, loc, basePtrI64, offset, elemByteSize);
    }
    // Convert base pointer (i64) to LLVM pointer type.
    Value basePtrLLVM =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTypeLLVM, basePtrI64);

    Value maskForLane;
    VectorType maskVecTy = dyn_cast<VectorType>(mask.getType());
    if (maskVecTy) {
      // Mask needs be scalar. Single element vector is converted to scalar by
      // type converter.
      return rewriter.notifyMatchFailure(op, "Expected mask to be a scalar.");
    } else
      maskForLane = mask;
    if constexpr (std::is_same_v<OpType, xegpu::LoadGatherOp>) {
      scf::IfOp ifOp = scf::IfOp::create(rewriter, loc, {valOrResTy},
                                         maskForLane, true, true);
      // If mask is true,- then clause - load from memory and yield.
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      if (!hasScalarVal)
        valOrResTy = VectorType::get({valOrResVecTy.getNumElements()},
                                     valOrResVecTy.getElementType());
      Value loaded =
          LLVM::LoadOp::create(rewriter, loc, valOrResTy, basePtrLLVM);
      // Set cache control attribute on the load operation.
      loaded.getDefiningOp()->setAttr(
          "cache_control", xevm::LoadCacheControlAttr::get(
                               ctxt, translateLoadXeGPUCacheHint(
                                         op.getL1Hint(), op.getL3Hint())));
      scf::YieldOp::create(rewriter, loc, ValueRange{loaded});
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      // If mask is false - else clause -yield a vector of zeros.
      auto eTy = hasScalarVal ? valOrResTy : valOrResVecTy.getElementType();
      TypedAttr eVal;
      if (eTy.isFloat())
        eVal = FloatAttr::get(eTy, 0.0);
      else
        eVal = IntegerAttr::get(eTy, 0);
      if (hasScalarVal)
        loaded = arith::ConstantOp::create(rewriter, loc, eVal);
      else
        loaded = arith::ConstantOp::create(
            rewriter, loc, DenseElementsAttr::get(valOrResVecTy, eVal));
      scf::YieldOp::create(rewriter, loc, ValueRange{loaded});
      rewriter.replaceOp(op, ifOp.getResult(0));
    } else {
      // If mask is true, perform the store.
      scf::IfOp ifOp = scf::IfOp::create(rewriter, loc, maskForLane, false);
      auto body = ifOp.getBody();
      rewriter.setInsertionPointToStart(body);
      auto storeOp =
          LLVM::StoreOp::create(rewriter, loc, adaptor.getValue(), basePtrLLVM);
      // Set cache control attribute on the store operation.
      storeOp.getOperation()->setAttr(
          "cache_control", xevm::StoreCacheControlAttr::get(
                               ctxt, translateStoreXeGPUCacheHint(
                                         op.getL1Hint(), op.getL3Hint())));
      rewriter.eraseOp(op);
    }
    return success();
  }
};

// Lower xegpu::CreateMemDescOp to memref::ViewOp. Since SLM access instructions
// on Xe2 and Xe3 operate on 32-bit or 64-bit units, all data types smaller than
// 32 bits will be converted to 32 bits.
class CreateMemDescOpPattern final
    : public OpConversionPattern<xegpu::CreateMemDescOp> {
public:
  using OpConversionPattern<xegpu::CreateMemDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateMemDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resTy = op.getMemDesc();

    // Create the result MemRefType with the same shape, element type, and
    // memory space
    auto newResTy = getTypeConverter()->convertType<MemRefType>(resTy);

    Value zero = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto viewOp = memref::ViewOp::create(rewriter, op.getLoc(), newResTy,
                                         op.getSource(), zero, ValueRange());
    rewriter.replaceOp(op, viewOp);
    return success();
  }
};

template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<
              OpType, xegpu::LoadMatrixOp, xegpu::StoreMatrixOp>::value>>
class LoadStoreMatrixToXeVMPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
    if (offsets.empty())
      return rewriter.notifyMatchFailure(op, "Expected offset to be provided.");

    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    Value basePtrStruct = adaptor.getMemDesc();
    Value mdescVal = op.getMemDesc();
    // Load result or Store value Type can be vector or scalar.
    Value data;
    if constexpr (std::is_same_v<OpType, xegpu::LoadMatrixOp>)
      data = op.getResult();
    else
      data = adaptor.getData();
    VectorType valOrResVecTy = dyn_cast<VectorType>(data.getType());
    if (!valOrResVecTy)
      valOrResVecTy = VectorType::get(1, data.getType());

    int64_t elemBitWidth =
        valOrResVecTy.getElementType().getIntOrFloatBitWidth();
    // Element type must be multiple of 8 bits.
    if (elemBitWidth % 8 != 0)
      return rewriter.notifyMatchFailure(
          op, "Expected element type bit width to be multiple of 8.");
    int64_t elemByteSize = elemBitWidth / 8;

    // Default memory space is SLM.
    LLVM::LLVMPointerType ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(xegpu::MemorySpace::SLM));

    auto mdescTy = cast<xegpu::MemDescType>(mdescVal.getType());

    Value basePtrLLVM = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, basePtrStruct);

    // Convert base pointer (ptr) to i32
    Value basePtrI32 = arith::IndexCastUIOp::create(
        rewriter, loc, rewriter.getI32Type(), basePtrLLVM);

    Value linearOffset = mdescTy.getLinearOffsets(rewriter, loc, offsets);
    linearOffset = arith::IndexCastUIOp::create(
        rewriter, loc, rewriter.getI32Type(), linearOffset);
    basePtrI32 = addOffsetToBaseAddr(rewriter, loc, basePtrI32, linearOffset,
                                     elemByteSize);

    // convert base pointer (i32) to LLVM pointer type
    basePtrLLVM =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTypeLLVM, basePtrI32);

    if (op.getSubgroupBlockIoAttr()) {
      // if the attribute 'subgroup_block_io' is set to true, it lowers to
      // xevm.blockload

      Type intElemTy = rewriter.getIntegerType(elemBitWidth);
      VectorType intVecTy =
          VectorType::get(valOrResVecTy.getShape(), intElemTy);

      if constexpr (std::is_same_v<OpType, xegpu::LoadMatrixOp>) {
        Value loadOp =
            xevm::BlockLoadOp::create(rewriter, loc, intVecTy, basePtrLLVM);
        if (intVecTy != valOrResVecTy) {
          loadOp =
              vector::BitCastOp::create(rewriter, loc, valOrResVecTy, loadOp);
        }
        rewriter.replaceOp(op, loadOp);
      } else {
        Value dataToStore = adaptor.getData();
        if (valOrResVecTy != intVecTy) {
          dataToStore =
              vector::BitCastOp::create(rewriter, loc, intVecTy, dataToStore);
        }
        xevm::BlockStoreOp::create(rewriter, loc, basePtrLLVM, dataToStore,
                                   nullptr);
        rewriter.eraseOp(op);
      }
      return success();
    }

    if (valOrResVecTy.getNumElements() >= 1) {
      auto chipOpt = xegpu::getChipStr(op);
      if (!chipOpt || (*chipOpt != "pvc" && *chipOpt != "bmg")) {
        // the lowering for chunk load only works for pvc and bmg
        return rewriter.notifyMatchFailure(
            op, "The lowering is specific to pvc or bmg.");
      }
    }

    if constexpr (std::is_same_v<OpType, xegpu::LoadMatrixOp>) {
      // if the size of valOrResVecTy is 1, it lowers to a scalar load/store
      // operation. LLVM load/store does not support vector of size 1, so we
      // need to handle this case separately.
      auto scalarTy = valOrResVecTy.getElementType();
      LLVM::LoadOp loadOp;
      if (valOrResVecTy.getNumElements() == 1)
        loadOp = LLVM::LoadOp::create(rewriter, loc, scalarTy, basePtrLLVM);
      else
        loadOp =
            LLVM::LoadOp::create(rewriter, loc, valOrResVecTy, basePtrLLVM);
      rewriter.replaceOp(op, loadOp);
    } else {
      LLVM::StoreOp::create(rewriter, loc, adaptor.getData(), basePtrLLVM);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class PrefetchToXeVMPattern : public OpConversionPattern<xegpu::PrefetchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::PrefetchOp op, xegpu::PrefetchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto tdescTy = op.getTensorDescType();
    Value basePtrI64 = adaptor.getSource();
    // Base pointer is passed as i32 or i64 by adaptor, cast to i64 if needed.
    if (basePtrI64.getType() != rewriter.getI64Type())
      basePtrI64 = arith::ExtUIOp::create(rewriter, loc, rewriter.getI64Type(),
                                          basePtrI64);
    Value offsets = adaptor.getOffsets();
    if (offsets) {
      VectorType offsetsVecTy = dyn_cast<VectorType>(offsets.getType());
      if (offsetsVecTy) {
        // Offset needs be scalar.
        return rewriter.notifyMatchFailure(op,
                                           "Expected offsets to be a scalar.");
      } else {
        int64_t elemBitWidth{0};
        int64_t elemByteSize;
        // Element byte size can come from three sources:
        if (tdescTy) {
          // If tensor descriptor is available, we use its element type to
          // determine element byte size.
          elemBitWidth = tdescTy.getElementType().getIntOrFloatBitWidth();
        } else if (auto memRefTy = dyn_cast<MemRefType>(op.getSourceType())) {
          // If memref is available, we use its element type to
          // determine element byte size.
          elemBitWidth = memRefTy.getElementType().getIntOrFloatBitWidth();
        } else {
          // Otherwise, we use the provided offset byte alignment.
          elemByteSize = *op.getOffsetAlignByte();
        }
        if (elemBitWidth != 0) {
          if (elemBitWidth % 8 != 0)
            return rewriter.notifyMatchFailure(
                op, "Expected element type bit width to be multiple of 8.");
          elemByteSize = elemBitWidth / 8;
        }
        basePtrI64 = addOffsetToBaseAddr(rewriter, loc, basePtrI64, offsets,
                                         elemByteSize);
      }
    }
    // Default memory space is global.
    LLVM::LLVMPointerType ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(xegpu::MemorySpace::Global));
    // If tensor descriptor is available, we use its memory space.
    if (tdescTy)
      ptrTypeLLVM = LLVM::LLVMPointerType::get(
          ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
    // If source is a memref, we use its memory space.
    if (auto memRefTy = dyn_cast<MemRefType>(op.getSource().getType())) {
      auto addrSpace = memRefTy.getMemorySpaceAsInt();
      if (addrSpace != 0)
        ptrTypeLLVM = LLVM::LLVMPointerType::get(ctxt, addrSpace);
    }
    // Convert base pointer (i64) to LLVM pointer type.
    Value ptrLLVM =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTypeLLVM, basePtrI64);
    // Create the prefetch op with cache control attribute.
    xevm::PrefetchOp::create(
        rewriter, loc, ptrLLVM,
        xevm::LoadCacheControlAttr::get(
            ctxt, translateLoadXeGPUCacheHint(op.getL1Hint(), op.getL3Hint())));
    rewriter.eraseOp(op);
    return success();
  }
};

class FenceToXeVMPattern : public OpConversionPattern<xegpu::FenceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::FenceOp op, xegpu::FenceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    xevm::MemScope memScope{xevm::MemScope::WORKGROUP};
    switch (op.getFenceScope()) {
    case xegpu::FenceScope::Workgroup:
      memScope = xevm::MemScope::WORKGROUP;
      break;
    case xegpu::FenceScope::GPU:
      memScope = xevm::MemScope::DEVICE;
      break;
    }
    xevm::AddrSpace addrSpace{xevm::AddrSpace::GLOBAL};
    switch (op.getMemoryKind()) {
    case xegpu::MemorySpace::Global:
      addrSpace = xevm::AddrSpace::GLOBAL;
      break;
    case xegpu::MemorySpace::SLM:
      addrSpace = xevm::AddrSpace::SHARED;
      break;
    }
    xevm::MemfenceOp::create(rewriter, loc, memScope, addrSpace);
    rewriter.eraseOp(op);
    return success();
  }
};

class DpasToXeVMPattern : public OpConversionPattern<xegpu::DpasOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::DpasOp op, xegpu::DpasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto aTy = cast<VectorType>(op.getLhs().getType());
    auto bTy = cast<VectorType>(op.getRhs().getType());
    auto resultType = cast<VectorType>(op.getResultType());

    auto encodePrecision = [&](Type type) -> xevm::ElemType {
      if (type == rewriter.getBF16Type())
        return xevm::ElemType::BF16;
      else if (type == rewriter.getF16Type())
        return xevm::ElemType::F16;
      else if (type == rewriter.getTF32Type())
        return xevm::ElemType::TF32;
      else if (type.isInteger(8)) {
        if (type.isUnsignedInteger())
          return xevm::ElemType::U8;
        return xevm::ElemType::S8;
      } else if (type == rewriter.getF32Type())
        return xevm::ElemType::F32;
      else if (type.isInteger(32))
        return xevm::ElemType::S32;
      llvm_unreachable("add more support for ElemType");
    };
    xevm::ElemType precATy = encodePrecision(aTy.getElementType());
    xevm::ElemType precBTy = encodePrecision(bTy.getElementType());
    Value c = op.getAcc();
    if (!c) {
      auto elementTy = resultType.getElementType();
      Attribute initValueAttr;
      if (isa<FloatType>(elementTy))
        initValueAttr = FloatAttr::get(elementTy, 0.0);
      else
        initValueAttr = IntegerAttr::get(elementTy, 0);
      c = arith::ConstantOp::create(
          rewriter, loc, DenseElementsAttr::get(resultType, initValueAttr));
    }

    Value aVec = op.getLhs();
    Value bVec = op.getRhs();
    auto cvecty = cast<VectorType>(c.getType());
    xevm::ElemType precCTy = encodePrecision(cvecty.getElementType());
    xevm::ElemType precDTy = encodePrecision(resultType.getElementType());
    VectorType cNty =
        VectorType::get(cvecty.getNumElements(), cvecty.getElementType());
    if (cvecty != cNty)
      c = vector::ShapeCastOp::create(rewriter, loc, cNty, c);
    Value dpasRes = xevm::MMAOp::create(
        rewriter, loc, cNty, aVec, bVec, c,
        xevm::MMAShapeAttr::get(ctxt, cvecty.getNumElements(), executionSize,
                                systolicDepth *
                                    getNumOperandsPerDword(precATy)),
        xevm::MMATypesAttr::get(ctxt, precDTy, precATy, precBTy, precCTy));
    if (cvecty != cNty)
      dpasRes = vector::ShapeCastOp::create(rewriter, loc, resultType, dpasRes);
    rewriter.replaceOp(op, dpasRes);
    return success();
  }

private:
  static unsigned getNumOperandsPerDword(xevm::ElemType pTy) {
    switch (pTy) {
    case xevm::ElemType::TF32:
      return 1;
    case xevm::ElemType::BF16:
    case xevm::ElemType::F16:
      return 2;
    case xevm::ElemType::U8:
    case xevm::ElemType::S8:
      return 4;
    default:
      llvm_unreachable("unsupported xevm::ElemType");
    }
  }
};

static std::optional<LLVM::AtomicBinOp>
matchSimpleAtomicOp(arith::AtomicRMWKind arithKind) {
  switch (arithKind) {
  case arith::AtomicRMWKind::addf:
    return LLVM::AtomicBinOp::fadd;
  case arith::AtomicRMWKind::addi:
    return LLVM::AtomicBinOp::add;
  case arith::AtomicRMWKind::assign:
    return LLVM::AtomicBinOp::xchg;
  case arith::AtomicRMWKind::maximumf:
    return LLVM::AtomicBinOp::fmax;
  case arith::AtomicRMWKind::maxs:
    return LLVM::AtomicBinOp::max;
  case arith::AtomicRMWKind::maxu:
    return LLVM::AtomicBinOp::umax;
  case arith::AtomicRMWKind::minimumf:
    return LLVM::AtomicBinOp::fmin;
  case arith::AtomicRMWKind::mins:
    return LLVM::AtomicBinOp::min;
  case arith::AtomicRMWKind::minu:
    return LLVM::AtomicBinOp::umin;
  case arith::AtomicRMWKind::ori:
    return LLVM::AtomicBinOp::_or;
  case arith::AtomicRMWKind::andi:
    return LLVM::AtomicBinOp::_and;
  default:
    return std::nullopt;
  }
}

class AtomicRMWToXeVMPattern : public OpConversionPattern<xegpu::AtomicRMWOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::AtomicRMWOp op, xegpu::AtomicRMWOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto tdesc = op.getTensorDesc().getType();
    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(tdesc.getMemorySpace()));
    Value basePtrI64 = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI64Type(), adaptor.getTensorDesc());
    Value basePtrLLVM =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTypeLLVM, basePtrI64);
    VectorType srcOrDstVecTy = cast<VectorType>(op.getValue().getType());
    VectorType srcOrDstFlatVecTy = VectorType::get(
        srcOrDstVecTy.getNumElements(), srcOrDstVecTy.getElementType());
    Value srcFlatVec = vector::ShapeCastOp::create(
        rewriter, loc, srcOrDstFlatVecTy, op.getValue());
    auto atomicKind = matchSimpleAtomicOp(op.getKind());
    assert(atomicKind.has_value());
    Value resVec = srcFlatVec;
    for (int i = 0; i < srcOrDstVecTy.getNumElements(); i++) {
      auto val = vector::ExtractOp::create(rewriter, loc, resVec, i);
      Value idx = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                           rewriter.getIndexAttr(i));
      Value currPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrTypeLLVM,
                              srcOrDstVecTy.getElementType(), basePtrLLVM, idx);
      Value newVal =
          LLVM::AtomicRMWOp::create(rewriter, loc, atomicKind.value(), currPtr,
                                    val, LLVM::AtomicOrdering::seq_cst);
      resVec = vector::InsertOp::create(rewriter, loc, newVal, resVec, i);
    }
    rewriter.replaceOp(op, resVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertXeGPUToXeVMPass
    : public impl::ConvertXeGPUToXeVMPassBase<ConvertXeGPUToXeVMPass> {
  using Base::Base;

  void runOnOperation() override {
    LLVMTypeConverter typeConverter(&getContext());
    typeConverter.addConversion([&](VectorType type) -> Type {
      unsigned rank = type.getRank();
      auto elemType = type.getElementType();
      // If the element type is index, convert it to i64.
      if (llvm::isa<IndexType>(elemType))
        elemType = IntegerType::get(&getContext(), 64);
      // If the vector is a scalar or has a single element, return the element
      if (rank < 1 || type.getNumElements() == 1)
        return elemType;
      // Otherwise, convert the vector to a flat vector type.
      int64_t sum = llvm::product_of(type.getShape());
      return VectorType::get(sum, elemType);
    });
    typeConverter.addConversion([&](xegpu::TensorDescType type) -> Type {
      if (type.isScattered())
        return IntegerType::get(&getContext(), 64);
      auto i32Type = IntegerType::get(&getContext(), 32);
      return VectorType::get(8, i32Type);
    });
    // Convert MemDescType into flattened MemRefType for SLM
    typeConverter.addConversion([&](xegpu::MemDescType type) -> Type {
      Type elemTy = type.getElementType();
      int numElems = type.getNumElements();
      return MemRefType::get(numElems, elemTy, AffineMap(), 3);
    });

    typeConverter.addConversion([&](MemRefType type) -> Type {
      // Convert MemRefType to i64 type.
      return IntegerType::get(&getContext(), 64);
    });

    // LLVM type converter puts unrealized casts for the following cases:
    // add materialization casts to handle them.

    // Materialization to convert memref to i64
    auto memrefMaterializationCast = [](OpBuilder &builder, Type type,
                                        ValueRange inputs,
                                        Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      auto input = inputs.front();
      if (auto memrefTy = dyn_cast<MemRefType>(input.getType())) {

        Value addr =
            memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, input);
        return arith::IndexCastUIOp::create(builder, loc, type, addr)
            .getResult();
      }
      return {};
    };

    // Materialization to convert ui64 to i64
    auto ui64MaterializationCast = [](OpBuilder &builder, Type type,
                                      ValueRange inputs,
                                      Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      auto input = inputs.front();
      if (input.getType() == builder.getIntegerType(64, false)) {
        Value cast =
            index::CastUOp::create(builder, loc, builder.getIndexType(), input)
                .getResult();
        return arith::IndexCastUIOp::create(builder, loc, type, cast)
            .getResult();
      }
      return {};
    };

    // Materialization to convert ui32 to i32
    auto ui32MaterializationCast = [](OpBuilder &builder, Type type,
                                      ValueRange inputs,
                                      Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      auto input = inputs.front();
      if (input.getType() == builder.getIntegerType(32, false)) {
        Value cast =
            index::CastUOp::create(builder, loc, builder.getIndexType(), input)
                .getResult();
        return arith::IndexCastUIOp::create(builder, loc, type, cast)
            .getResult();
      }
      return {};
    };

    // Materialization to convert
    //   - single element 1D vector to scalar
    //   - bitcast vector of same rank
    //   - shape vector of different rank but same element type
    auto vectorMaterializationCast = [](OpBuilder &builder, Type type,
                                        ValueRange inputs,
                                        Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      auto input = inputs.front();
      if (auto vecTy = dyn_cast<VectorType>(input.getType())) {
        if (vecTy.getNumElements() == 1) {
          // If the vector has a single element, return the element type.
          Value cast =
              vector::ExtractOp::create(builder, loc, input, 0).getResult();
          if (vecTy.getElementType() == builder.getIndexType())
            cast = arith::IndexCastUIOp::create(builder, loc, type, cast)
                       .getResult();
          return cast;
        } else if (auto targetVecTy = dyn_cast<VectorType>(type)) {
          // If the target type is a vector of same rank,
          //   bitcast to the target type.
          if (targetVecTy.getRank() == vecTy.getRank())
            return vector::BitCastOp::create(builder, loc, targetVecTy, input)
                .getResult();
          else if (targetVecTy.getElementType() == vecTy.getElementType()) {
            // If the target type is a vector of different rank but same element
            // type, reshape to the target type.
            return vector::ShapeCastOp::create(builder, loc, targetVecTy, input)
                .getResult();
          }
        }
      }
      return {};
    };

    // If result type of original op is single element vector and lowered type
    // is scalar. This materialization cast creates a single element vector by
    // broadcasting the scalar value.
    auto singleElementVectorMaterializationCast =
        [](OpBuilder &builder, Type type, ValueRange inputs,
           Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      auto input = inputs.front();
      if (input.getType().isIntOrIndexOrFloat()) {
        // If the input is a scalar, and the target type is a vector of single
        // element, create a single element vector by broadcasting.
        if (auto vecTy = dyn_cast<VectorType>(type)) {
          if (vecTy.getNumElements() == 1) {
            return vector::BroadcastOp::create(builder, loc, vecTy, input)
                .getResult();
          }
        }
      }
      return {};
    };
    typeConverter.addSourceMaterialization(
        singleElementVectorMaterializationCast);
    typeConverter.addTargetMaterialization(memrefMaterializationCast);
    typeConverter.addTargetMaterialization(ui32MaterializationCast);
    typeConverter.addTargetMaterialization(ui64MaterializationCast);
    typeConverter.addTargetMaterialization(vectorMaterializationCast);
    ConversionTarget target(getContext());
    target.addLegalDialect<xevm::XeVMDialect, LLVM::LLVMDialect,
                           vector::VectorDialect, arith::ArithDialect,
                           memref::MemRefDialect, gpu::GPUDialect,
                           index::IndexDialect>();
    target.addIllegalDialect<xegpu::XeGPUDialect>();

    RewritePatternSet patterns(&getContext());
    populateXeGPUToXeVMConversionPatterns(typeConverter, patterns);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//
void mlir::populateXeGPUToXeVMConversionPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<CreateNdDescToXeVMPattern,
               LoadStorePrefetchNdToXeVMPattern<xegpu::LoadNdOp>,
               LoadStorePrefetchNdToXeVMPattern<xegpu::StoreNdOp>,
               LoadStorePrefetchNdToXeVMPattern<xegpu::PrefetchNdOp>>(
      typeConverter, patterns.getContext());
  patterns.add<AtomicRMWToXeVMPattern, PrefetchToXeVMPattern,
               LoadStoreToXeVMPattern<xegpu::LoadGatherOp>,
               LoadStoreToXeVMPattern<xegpu::StoreScatterOp>>(
      typeConverter, patterns.getContext());
  patterns.add<LoadStoreMatrixToXeVMPattern<xegpu::LoadMatrixOp>,
               LoadStoreMatrixToXeVMPattern<xegpu::StoreMatrixOp>,
               CreateMemDescOpPattern>(typeConverter, patterns.getContext());
  patterns.add<FenceToXeVMPattern, DpasToXeVMPattern>(typeConverter,
                                                      patterns.getContext());
}
