//===-- XeVMToLLVM.cpp - XeVM to LLVM dialect conversion --------*- C++ -*-===//
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
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTXEGPUTOXEVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

enum class NdDescI32Layout : uint32_t {
  BasePtr = 0,
  BaseShapeW = 2,
  BaseShapeH = 3,
  TensorOffsetW = 4,
  TensorOffsetH = 5
};

static int32_t getNumericXeVMAddrSpace(xegpu::MemorySpace xeGpuMemspace) {
  switch (xeGpuMemspace) {
  case xegpu::MemorySpace::Global:
    return static_cast<int>(xevm::AddrSpace::GLOBAL);
  case xegpu::MemorySpace::SLM:
    return static_cast<int>(xevm::AddrSpace::SHARED);
  }
  llvm_unreachable("Unknown XeGPU memory space.");
}

VectorType encodeVectorTypeTo(VectorType currentVecType, Type toElemType) {
  auto elemType = currentVecType.getElementType();
  auto currentBitWidth = elemType.getIntOrFloatBitWidth();
  auto newBitWidth = toElemType.getIntOrFloatBitWidth();
  const int size =
      currentVecType.getNumElements() * currentBitWidth / newBitWidth;
  return VectorType::get(size, toElemType);
}

xevm::LoadCacheControl
translateLoadXeGPUCacheHint(std::optional<xegpu::CachePolicy> L1hint,
                            std::optional<xegpu::CachePolicy> L3hint) {
  auto L1hintVal =
      L1hint.has_value() ? L1hint.value() : xegpu::CachePolicy::UNCACHED;
  auto L3hintVal =
      L3hint.has_value() ? L3hint.value() : xegpu::CachePolicy::UNCACHED;
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

xevm::StoreCacheControl
translateStoreXeGPUCacheHint(std::optional<xegpu::CachePolicy> L1hint,
                             std::optional<xegpu::CachePolicy> L3hint) {
  auto L1hintVal =
      L1hint.has_value() ? L1hint.value() : xegpu::CachePolicy::UNCACHED;
  auto L3hintVal =
      L3hint.has_value() ? L3hint.value() : xegpu::CachePolicy::UNCACHED;
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
    auto loc = op.getLoc();
    auto source = op.getSource();
    Type payloadElemTy = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    VectorType payloadTy = VectorType::get(8, payloadElemTy);
    VectorType payloadI64Ty = VectorType::get(4, i64Ty);
    Value payload = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(payloadTy, IntegerAttr::get(payloadElemTy, 0)));

    Value baseAddr;
    Value baseShapeW;
    Value baseShapeH;
    Value offsetW;
    Value offsetH;

    bool sourceIsMemref = false;
    auto sourceTy = source.getType();
    int64_t rank;
    if (isa<MemRefType>(sourceTy)) {
      sourceIsMemref = true;
      baseAddr =
          memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, source);
      auto sourceMemrefTy = cast<MemRefType>(sourceTy);
      if (!sourceMemrefTy.hasStaticShape()) {
        op.emitError() << "Expected static memref shape.";
        return failure();
      }
      rank = sourceMemrefTy.getRank();
      if (rank != 2) {
        op.emitError() << "Expected a 2D memref.";
        return failure();
      }
    } else if (sourceTy == rewriter.getIntegerType(64, false)) {
      rank = op.getMixedSizes().size();
    } else {
      op.emitError() << "Expected source to be a 2D memref or ui64.";
      return failure();
    }
    auto createOffset = [&](unsigned idx) -> Value {
      Value val;
      OpFoldResult ofr = op.getMixedOffsets()[idx];
      if (auto v = llvm::dyn_cast_if_present<Value>(ofr)) {
        val = arith::IndexCastOp::create(rewriter, loc, i64Ty, v);
        val = arith::TruncIOp::create(rewriter, loc, payloadElemTy, val);
      } else {
        int32_t off = llvm::cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();
        val = arith::ConstantIntOp::create(rewriter, loc, payloadElemTy, off);
      }
      return val;
    };
    auto offsets = op.getMixedOffsets();
    if (offsets.size() == 2) {
      offsetW = createOffset(rank - 1);
      offsetH = createOffset(rank - 2);
    } else {
      offsetW = arith::ConstantIntOp::create(rewriter, loc, payloadElemTy, 0);
      offsetH = arith::ConstantIntOp::create(rewriter, loc, payloadElemTy, 0);
    }
    auto createShape = [&](unsigned idx) -> Value {
      Value val;
      OpFoldResult ofr = op.getMixedSizes()[idx];
      if (auto v = llvm::dyn_cast_if_present<Value>(ofr)) {
        val = arith::IndexCastOp::create(rewriter, loc, i64Ty, v);
        val = arith::TruncIOp::create(rewriter, loc, payloadElemTy, val);
      } else {
        int32_t off = llvm::cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();
        val = arith::ConstantIntOp::create(rewriter, loc, payloadElemTy, off);
      }
      return val;
    };
    if (sourceIsMemref) {
      auto sourceMemrefTy = cast<MemRefType>(sourceTy);
      baseShapeW = arith::ConstantIntOp::create(
          rewriter, loc, payloadElemTy, sourceMemrefTy.getDimSize(rank - 1));
      baseShapeH = arith::ConstantIntOp::create(
          rewriter, loc, payloadElemTy, sourceMemrefTy.getDimSize(rank - 2));
      baseAddr = arith::IndexCastUIOp::create(rewriter, loc, i64Ty, baseAddr);
    } else {
      baseShapeW = createShape(rank - 1);
      baseShapeH = createShape(rank - 2);
      baseAddr = adaptor.getSource();
    }
    Value payLoadAsI64 =
        vector::BitCastOp::create(rewriter, loc, payloadI64Ty, payload);
    payLoadAsI64 =
        vector::InsertOp::create(rewriter, loc, baseAddr, payLoadAsI64,
                                 static_cast<int>(NdDescI32Layout::BasePtr));
    payload = vector::BitCastOp::create(rewriter, loc, payloadTy, payLoadAsI64);
    payload =
        vector::InsertOp::create(rewriter, loc, baseShapeW, payload,
                                 static_cast<int>(NdDescI32Layout::BaseShapeW));
    payload =
        vector::InsertOp::create(rewriter, loc, baseShapeH, payload,
                                 static_cast<int>(NdDescI32Layout::BaseShapeH));
    payload = vector::InsertOp::create(
        rewriter, loc, offsetW, payload,
        static_cast<int>(NdDescI32Layout::TensorOffsetW));
    payload = vector::InsertOp::create(
        rewriter, loc, offsetH, payload,
        static_cast<int>(NdDescI32Layout::TensorOffsetH));
    rewriter.replaceOp(op, payload);
    return success();
  }
};

class UpdateNdOffsetToXeVMPattern
    : public OpConversionPattern<xegpu::UpdateNdOffsetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::UpdateNdOffsetOp op,
                  xegpu::UpdateNdOffsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto offsets = op.getOffsets();
    auto tdesc = adaptor.getTensorDesc();
    for (size_t offsetDim = 0; offsetDim < offsets.size(); offsetDim++) {
      auto offset = offsets[offsetDim];
      if (auto cst =
              dyn_cast_if_present<arith::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast_if_present<IntegerAttr>(cst.getValue());
            attr && !attr.getInt())
          continue;
      const int offsetPos =
          static_cast<int>(offsetDim ? NdDescI32Layout::TensorOffsetW
                                     : NdDescI32Layout::TensorOffsetH);
      auto oldOffset =
          vector::ExtractOp::create(rewriter, loc, tdesc, offsetPos);
      offset = arith::IndexCastUIOp::create(rewriter, loc,
                                            rewriter.getI32Type(), offset);
      auto newOffset = arith::AddIOp::create(rewriter, loc, oldOffset, offset);
      tdesc =
          vector::InsertOp::create(rewriter, loc, newOffset, tdesc, offsetPos);
    }
    rewriter.replaceOp(op, tdesc);
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
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();

    auto tdesc = adaptor.getTensorDesc();
    auto tdescTy = op.getTensorDescType();
    if (tdescTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "Expected 2D tensor descriptor.");
    }

    VectorType payloadI64Ty = VectorType::get(4, rewriter.getI64Type());
    Value payLoadAsI64 =
        vector::BitCastOp::create(rewriter, loc, payloadI64Ty, tdesc);
    Value basePtr =
        vector::ExtractOp::create(rewriter, loc, payLoadAsI64,
                                  static_cast<int>(NdDescI32Layout::BasePtr));
    Value baseShapeW = vector::ExtractOp::create(
        rewriter, loc, tdesc, static_cast<int>(NdDescI32Layout::BaseShapeW));
    Value baseShapeH = vector::ExtractOp::create(
        rewriter, loc, tdesc, static_cast<int>(NdDescI32Layout::BaseShapeH));
    // Offsets can come from three sources:
    // 1. Constant offsets, which are provided by the op.
    // 2. Offsets as operands, which are provided by the op.
    // 3. Offsets extracted from the tensor descriptor.
    Value offsetW;
    Value offsetH;
    auto cOffsets = op.getConstOffsets();
    auto offsets = op.getOffsets();
    if (cOffsets) {
      offsetW = arith::ConstantIntOp::create(
          rewriter, loc, rewriter.getI32Type(), (*cOffsets)[0]);
      offsetH = arith::ConstantIntOp::create(
          rewriter, loc, rewriter.getI32Type(), (*cOffsets)[1]);
    } else if (offsets.size() != 0) {
      // offsets are provided as operands
      if (offsets[0].getType() != rewriter.getI32Type()) {
        if (offsets[0].getType() != rewriter.getIndexType()) {
          return rewriter.notifyMatchFailure(
              op, "Expected offsets to be of type i32 or index.");
        }
        offsetW = arith::IndexCastUIOp::create(
            rewriter, loc, rewriter.getI32Type(), offsets[0]);
      } else {
        offsetW = offsets[0];
      }
      if (offsets[1].getType() != rewriter.getI32Type()) {
        if (offsets[1].getType() != rewriter.getIndexType()) {
          return rewriter.notifyMatchFailure(
              op, "Expected offsets to be of type i32 or index.");
        }
        offsetH = arith::IndexCastUIOp::create(
            rewriter, loc, rewriter.getI32Type(), offsets[1]);
      } else {
        offsetH = offsets[1];
      }
    } else {
      // If offsets are not available, we need to extract them from the tensor
      // descriptor.
      offsetW = vector::ExtractOp::create(
          rewriter, loc, tdesc,
          static_cast<int>(NdDescI32Layout::TensorOffsetW));
      offsetH = vector::ExtractOp::create(
          rewriter, loc, tdesc,
          static_cast<int>(NdDescI32Layout::TensorOffsetH));
    }
    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
    Value basePtrLLVM =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTypeLLVM, basePtr);
    auto elemType = tdescTy.getElementType();
    auto elemBitSize = elemType.getIntOrFloatBitWidth();
    Value elemByteSize = arith::ConstantIntOp::create(
        rewriter, loc, rewriter.getI32Type(), elemBitSize / 8);
    Value surfaceW =
        arith::MulIOp::create(rewriter, loc, baseShapeW, elemByteSize);

    auto tileW = tdescTy.getDimSize(1);
    auto tileH = tdescTy.getDimSize(0);
    int32_t vblocks = tdescTy.getArrayLength();
    if constexpr (std::is_same_v<OpType, xegpu::StoreNdOp>) {
      VectorType srcVecTy = cast<VectorType>(op.getValue().getType());
      auto storeCacheControl =
          translateStoreXeGPUCacheHint(op.getL1Hint(), op.getL3Hint());
      VectorType srcFlatVecTy =
          VectorType::get(srcVecTy.getNumElements(), srcVecTy.getElementType());
      Value srcFlatVec = op.getValue();
      srcFlatVecTy = encodeVectorTypeTo(srcFlatVecTy,
                                        rewriter.getIntegerType(elemBitSize));
      srcFlatVec =
          vector::BitCastOp::create(rewriter, loc, srcFlatVecTy, srcFlatVec);
      xevm::BlockStore2dOp::create(
          rewriter, loc, basePtrLLVM, surfaceW, baseShapeH, surfaceW, offsetW,
          offsetH, elemBitSize, tileW, tileH, srcFlatVec,
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
auto addOffset = [](ConversionPatternRewriter &rewriter, Location loc,
                    Value baseAddr, Value offset,
                    int64_t elemByteSize) -> Value {
  Value byteSize = arith::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64Type(), elemByteSize);
  Value byteOffset = arith::MulIOp::create(rewriter, loc, offset, byteSize);
  Value newAddr = arith::AddIOp::create(rewriter, loc, baseAddr, byteOffset);
  return newAddr;
};

class CreateDescToXeVMPattern
    : public OpConversionPattern<xegpu::CreateDescOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateDescOp op, xegpu::CreateDescOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto eTy = op.getTensorDescType().getElementType();
    auto eBw = eTy.getIntOrFloatBitWidth();
    if (eBw % 8 != 0) {
      return rewriter.notifyMatchFailure(
          op, "Expected element type bit width to be multiple of 8.");
    }
    auto loc = op.getLoc();
    // offsets are provided as scalar i64 by type converter.
    auto offsets = adaptor.getOffsets();
    // Source type can be a 1D memref or pointer type (ui64, ui32, i64 or i32).
    // But type converter will convert them to integer types.
    Value addr = adaptor.getSource();
    // ui32 or i32 are passed as i32 so they need to be casted to i64.
    if (addr.getType() != rewriter.getI64Type())
      addr = arith::ExtUIOp::create(rewriter, loc, rewriter.getI64Type(), addr);
    auto laneAddr = addOffset(rewriter, loc, addr, offsets, eBw / 8);
    rewriter.replaceOp(op, laneAddr);
    return success();
  }
};

class UpdateOffsetToXeVMPattern
    : public OpConversionPattern<xegpu::UpdateOffsetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::UpdateOffsetOp op,
                  xegpu::UpdateOffsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto eTy = op.getTensorDescType().getElementType();
    auto eBw = eTy.getIntOrFloatBitWidth();
    if (eBw % 8 != 0) {
      return rewriter.notifyMatchFailure(
          op, "Expected element type bit width to be multiple of 8.");
    }
    auto loc = op.getLoc();
    // scatter descriptor is provided as scalar i64 by type converter.
    // offsets are provided as scalar i64 by type converter.
    Value newOffset = addOffset(rewriter, loc, adaptor.getTensorDesc(),
                                adaptor.getOffsets(), eBw / 8);
    rewriter.replaceOp(op, newOffset);
    return success();
  }
};

template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<
              OpType, xegpu::LoadGatherOp, xegpu::StoreScatterOp>::value>>
class LoadStoreToXeVMPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto tdescTy = op.getTensorDescType();
    Value basePtrI64;
    // Load result or Store valye Type can be vector or scalar.
    Type valOrResTy;
    if constexpr (std::is_same_v<OpType, xegpu::LoadGatherOp>) {
      valOrResTy = op.getResult().getType();
    } else {
      valOrResTy = adaptor.getValue().getType();
    }
    VectorType valOrResVecTy = dyn_cast<VectorType>(valOrResTy);
    bool hasScalarVal = !valOrResVecTy;
    int64_t elemBitWidth =
        hasScalarVal ? valOrResTy.getIntOrFloatBitWidth()
                     : valOrResVecTy.getElementType().getIntOrFloatBitWidth();
    // Element type must be multiple of 8 bits.
    if (elemBitWidth % 8 != 0) {
      return rewriter.notifyMatchFailure(
          op, "Expected element type bit width to be multiple of 8.");
    }
    int64_t elemByteSize = elemBitWidth / 8;
    // Default memory space is global.
    LLVM::LLVMPointerType ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(xegpu::MemorySpace::Global));
    // If tensor descriptor is available, we use its memory space.
    if (tdescTy) {
      ptrTypeLLVM = LLVM::LLVMPointerType::get(
          ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
    }
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
    Value offsets = adaptor.getOffsets();
    Value mask = adaptor.getMask();
    if (offsets) {
      if (dyn_cast<VectorType>(offsets.getType())) {
        // Offset needs be scalar. Single element vector is converted to scalar
        // by type converter.
        return rewriter.notifyMatchFailure(op,
                                           "Expected offsets to be a scalar.");
      } else {
        // If offsets are provided, we add them to the base pointer.
        // Offsets are in number of elements, we need to multiply by
        // element byte size.
        basePtrI64 =
            addOffset(rewriter, loc, basePtrI64, offsets, elemByteSize);
      }
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
    } else {
      maskForLane = mask;
    }
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
      // if mask is true, perform the store.
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
    if (basePtrI64.getType() != rewriter.getI64Type()) {
      basePtrI64 = arith::ExtUIOp::create(rewriter, loc, rewriter.getI64Type(),
                                          basePtrI64);
    }
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
          if (elemBitWidth % 8 != 0) {
            return rewriter.notifyMatchFailure(
                op, "Expected element type bit width to be multiple of 8.");
          }
          elemByteSize = elemBitWidth / 8;
        }
        basePtrI64 =
            addOffset(rewriter, loc, basePtrI64, offsets, elemByteSize);
      }
    }
    // Default memory space is global.
    LLVM::LLVMPointerType ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(xegpu::MemorySpace::Global));
    // If tensor descriptor is available, we use its memory space.
    if (tdescTy) {
      ptrTypeLLVM = LLVM::LLVMPointerType::get(
          ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
    }
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
      llvm_unreachable("Unknown XeGPU fence scope.");
    }
    xevm::AddrSpace addrSpace{xevm::AddrSpace::GLOBAL};
    switch (op.getMemoryKind()) {
    case xegpu::MemorySpace::Global:
      addrSpace = xevm::AddrSpace::GLOBAL;
      break;
    case xegpu::MemorySpace::SLM:
      addrSpace = xevm::AddrSpace::SHARED;
      break;
      llvm_unreachable("Unknown XeGPU fence scope.");
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
    // below are uArch dependent values, should move away from hardcoding
    constexpr int32_t systolicDepth{8};
    constexpr int32_t executionSize{16};
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
  llvm_unreachable("Invalid AtomicRMWKind");
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
      unsigned sum = 1;
      for (unsigned i = 0; i < rank; i++) {
        sum *= type.getShape()[i];
      }
      return VectorType::get(sum, elemType);
    });
    typeConverter.addConversion([&](xegpu::TensorDescType type) -> Type {
      if (type.isScattered()) {
        return IntegerType::get(&getContext(), 64);
      }
      auto i32Type = IntegerType::get(&getContext(), 32);
      return VectorType::get(8, i32Type);
    });
    typeConverter.addConversion([&](MemRefType type) -> Type {
      // Convert MemRefType to i64 type.
      return IntegerType::get(&getContext(), 64);
    });

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

    auto vector1DMaterializationCast = [](OpBuilder &builder, Type type,
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
        }
      }
      return {};
    };
    typeConverter.addSourceMaterialization(memrefMaterializationCast);
    typeConverter.addSourceMaterialization(ui64MaterializationCast);
    typeConverter.addSourceMaterialization(ui32MaterializationCast);
    typeConverter.addSourceMaterialization(vector1DMaterializationCast);
    typeConverter.addTargetMaterialization(memrefMaterializationCast);
    typeConverter.addTargetMaterialization(ui32MaterializationCast);
    typeConverter.addTargetMaterialization(ui64MaterializationCast);
    typeConverter.addTargetMaterialization(vector1DMaterializationCast);
    ConversionTarget target(getContext());
    target.addLegalDialect<xevm::XeVMDialect, LLVM::LLVMDialect,
                           vector::VectorDialect, arith::ArithDialect,
                           memref::MemRefDialect, gpu::GPUDialect,
                           index::IndexDialect>();
    target.addIllegalDialect<xegpu::XeGPUDialect>();

    RewritePatternSet patterns(&getContext());
    populateXeGPUToXeVMConversionPatterns(patterns, typeConverter);
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
    RewritePatternSet &patterns, LLVMTypeConverter &typeConverter) {
  patterns.add<CreateNdDescToXeVMPattern, UpdateNdOffsetToXeVMPattern,
               LoadStorePrefetchNdToXeVMPattern<xegpu::LoadNdOp>,
               LoadStorePrefetchNdToXeVMPattern<xegpu::StoreNdOp>,
               LoadStorePrefetchNdToXeVMPattern<xegpu::PrefetchNdOp>>(
      typeConverter, patterns.getContext());
  patterns.add<CreateDescToXeVMPattern, UpdateOffsetToXeVMPattern,
               AtomicRMWToXeVMPattern, PrefetchToXeVMPattern,
               LoadStoreToXeVMPattern<xegpu::LoadGatherOp>,
               LoadStoreToXeVMPattern<xegpu::StoreScatterOp>>(
      typeConverter, patterns.getContext());
  patterns.add<FenceToXeVMPattern, DpasToXeVMPattern>(typeConverter,
                                                      patterns.getContext());
}
