//===- LegalizeForLLVMExport.cpp - Prepare AMX for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/Transforms.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::amx;

namespace {

/// Maps the 2-dim vector shape to the two 16-bit tile sizes. The first
/// dimension directly translates into the number of rows of the tiles.
/// The second dimensions needs to be scaled by the number of bytes.
std::pair<Value, Value> getTileSizes(ConversionPatternRewriter &rewriter,
                                     const LLVMTypeConverter &typeConverter,
                                     VectorType vType, Location loc) {
  Type llvmInt16Type = IntegerType::get(&typeConverter.getContext(), 16);
  unsigned width = vType.getElementType().getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_64(width) && width >= 8);
  unsigned bytes = width >> 3;
  auto mattr = rewriter.getI16IntegerAttr(vType.getDimSize(0));
  auto nattr = rewriter.getI16IntegerAttr(vType.getDimSize(1) * bytes);
  return std::make_pair(
      rewriter.create<LLVM::ConstantOp>(loc, llvmInt16Type, mattr),
      rewriter.create<LLVM::ConstantOp>(loc, llvmInt16Type, nattr));
}

/// Verifies if the stride matches proper tile access.
LogicalResult verifyStride(MemRefType mType) {
  if (mType.getRank() < 2)
    return failure();
  int64_t last = mType.getRank() - 1;
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(mType, strides, offset)) || strides[last] != 1)
    return failure();
  return success();
}

/// Maps the 2-dim memref shape to the 64-bit stride. Note that the buffer
/// shape may "envelop" the actual tile shape, and may be dynamically sized.
Value getStride(ConversionPatternRewriter &rewriter,
                const LLVMTypeConverter &typeConverter, MemRefType mType,
                Value base, Location loc) {
  assert(mType.getRank() >= 2);
  int64_t last = mType.getRank() - 1;
  Type llvmInt64Type = IntegerType::get(&typeConverter.getContext(), 64);
  unsigned width = mType.getElementType().getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_64(width) && width >= 8);
  unsigned bytes = width >> 3;
  if (mType.isDynamicDim(last)) {
    // Dynamic size needs code to compute the stride at runtime.
    MemRefDescriptor memrefDescriptor(base);
    auto attr = rewriter.getI64IntegerAttr(bytes);
    Value scale = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, attr);
    return rewriter.create<LLVM::MulOp>(
        loc, llvmInt64Type, scale, memrefDescriptor.size(rewriter, loc, last));
  }
  // Use direct constant for static size.
  auto attr = rewriter.getI64IntegerAttr(mType.getDimSize(last) * bytes);
  return rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, attr);
}

struct TileZeroConversion : public ConvertOpToLLVMPattern<TileZeroOp> {
  using ConvertOpToLLVMPattern<TileZeroOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TileZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType vType = op.getVectorType();
    // Determine m x n tile sizes.
    std::pair<Value, Value> tsz =
        getTileSizes(rewriter, *getTypeConverter(), vType, op.getLoc());
    // Replace operation with intrinsic.
    Type resType = typeConverter->convertType(vType);
    rewriter.replaceOpWithNewOp<amx::x86_amx_tilezero>(op, resType, tsz.first,
                                                       tsz.second);
    return success();
  }
};

struct TileLoadConversion : public ConvertOpToLLVMPattern<TileLoadOp> {
  using ConvertOpToLLVMPattern<TileLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TileLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType mType = op.getMemRefType();
    VectorType vType = op.getVectorType();
    // Determine m x n tile sizes.
    std::pair<Value, Value> tsz =
        getTileSizes(rewriter, *getTypeConverter(), vType, op.getLoc());
    // Determine stride.
    if (failed(verifyStride(mType)))
      return failure();
    Value stride = getStride(rewriter, *getTypeConverter(), mType,
                             adaptor.getBase(), op.getLoc());
    // Replace operation with intrinsic.
    Value ptr = getStridedElementPtr(op.getLoc(), mType, adaptor.getBase(),
                                     adaptor.getIndices(), rewriter);
    Type resType = typeConverter->convertType(vType);
    rewriter.replaceOpWithNewOp<amx::x86_amx_tileloadd64>(
        op, resType, tsz.first, tsz.second, ptr, stride);
    return success();
  }
};

struct TileStoreConversion : public ConvertOpToLLVMPattern<TileStoreOp> {
  using ConvertOpToLLVMPattern<TileStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TileStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType mType = op.getMemRefType();
    VectorType vType = op.getVectorType();
    // Determine m x n tile sizes.
    std::pair<Value, Value> tsz =
        getTileSizes(rewriter, *getTypeConverter(), vType, op.getLoc());
    // Determine stride.
    if (failed(verifyStride(mType)))
      return failure();
    Value stride = getStride(rewriter, *getTypeConverter(), mType,
                             adaptor.getBase(), op.getLoc());
    // Replace operation with intrinsic.
    Value ptr = getStridedElementPtr(op.getLoc(), mType, adaptor.getBase(),
                                     adaptor.getIndices(), rewriter);
    rewriter.replaceOpWithNewOp<amx::x86_amx_tilestored64>(
        op, tsz.first, tsz.second, ptr, stride, adaptor.getVal());
    return success();
  }
};

struct TileMulFConversion : public ConvertOpToLLVMPattern<TileMulFOp> {
  using ConvertOpToLLVMPattern<TileMulFOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TileMulFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType aType = op.getLhsVectorType();
    VectorType bType = op.getRhsVectorType();
    VectorType cType = op.getVectorType();
    // Determine m x n x k tile sizes.
    std::pair<Value, Value> tsza =
        getTileSizes(rewriter, *getTypeConverter(), aType, op.getLoc());
    std::pair<Value, Value> tszb =
        getTileSizes(rewriter, *getTypeConverter(), bType, op.getLoc());
    // Replace operation with intrinsic.
    Type resType = typeConverter->convertType(cType);
    rewriter.replaceOpWithNewOp<amx::x86_amx_tdpbf16ps>(
        op, resType, tsza.first, tszb.second, tsza.second, adaptor.getAcc(),
        adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct TileMulIConversion : public ConvertOpToLLVMPattern<TileMulIOp> {
  using ConvertOpToLLVMPattern<TileMulIOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TileMulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType aType = op.getLhsVectorType();
    VectorType bType = op.getRhsVectorType();
    VectorType cType = op.getVectorType();
    // Determine m x n x k tile sizes.
    std::pair<Value, Value> tsza =
        getTileSizes(rewriter, *getTypeConverter(), aType, op.getLoc());
    std::pair<Value, Value> tszb =
        getTileSizes(rewriter, *getTypeConverter(), bType, op.getLoc());
    // Replace operation with intrinsic.
    Type resType = typeConverter->convertType(cType);
    bool zexta = op.getIsZextLhs();
    bool zextb = op.getIsZextRhs();
    if (zexta && zextb)
      rewriter.replaceOpWithNewOp<amx::x86_amx_tdpbuud>(
          op, resType, tsza.first, tszb.second, tsza.second, adaptor.getAcc(),
          adaptor.getLhs(), adaptor.getRhs());
    else if (zexta && !zextb)
      rewriter.replaceOpWithNewOp<amx::x86_amx_tdpbusd>(
          op, resType, tsza.first, tszb.second, tsza.second, adaptor.getAcc(),
          adaptor.getLhs(), adaptor.getRhs());
    else if (!zexta && zextb)
      rewriter.replaceOpWithNewOp<amx::x86_amx_tdpbsud>(
          op, resType, tsza.first, tszb.second, tsza.second, adaptor.getAcc(),
          adaptor.getLhs(), adaptor.getRhs());
    else
      rewriter.replaceOpWithNewOp<amx::x86_amx_tdpbssd>(
          op, resType, tsza.first, tszb.second, tsza.second, adaptor.getAcc(),
          adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

} // namespace

void mlir::populateAMXLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<TileZeroConversion, TileLoadConversion, TileStoreConversion,
               TileMulFConversion, TileMulIConversion>(converter);
}

void mlir::configureAMXLegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalOp<x86_amx_tilezero, x86_amx_tileloadd64, x86_amx_tilestored64,
                    x86_amx_tdpbf16ps, x86_amx_tdpbssd, x86_amx_tdpbsud,
                    x86_amx_tdpbusd, x86_amx_tdpbuud>();
  target.addIllegalOp<TileZeroOp, TileLoadOp, TileStoreOp, TileMulIOp,
                      TileMulFOp>();
}
