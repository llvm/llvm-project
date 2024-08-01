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
private:
  const std::optional<std::reference_wrapper<TileScopeAnalysis>>
      &enablingAnalysis;

public:
  using ConvertOpToLLVMPattern<TileZeroOp>::ConvertOpToLLVMPattern;
  TileZeroConversion(
      const LLVMTypeConverter &typeConverter,
      const std::optional<std::reference_wrapper<TileScopeAnalysis>> &analysis)
      : ConvertOpToLLVMPattern<TileZeroOp>(typeConverter),
        enablingAnalysis(analysis) {}

  LogicalResult
  matchAndRewrite(TileZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (enablingAnalysis && enablingAnalysis->get().isValid()) {
      rewriter.setInsertionPoint(op);
      // Routine for lowering tile Ops with binding info.
      auto dstRegIndex = op.getDstRegIndex();
      assert(dstRegIndex && "Incomplete operation attribute for tile binding");
      rewriter.create<amx::x86_amx_tilezero_plain>(op.getLoc(), *dstRegIndex);
      rewriter.eraseOp(op);
      return success();
    }

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
private:
  const std::optional<std::reference_wrapper<TileScopeAnalysis>>
      &enablingAnalysis;

public:
  using ConvertOpToLLVMPattern<TileLoadOp>::ConvertOpToLLVMPattern;
  TileLoadConversion(
      const LLVMTypeConverter &typeConverter,
      const std::optional<std::reference_wrapper<TileScopeAnalysis>> &analysis)
      : ConvertOpToLLVMPattern<TileLoadOp>(typeConverter),
        enablingAnalysis(analysis) {}

  LogicalResult
  matchAndRewrite(TileLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType mType = op.getMemRefType();
    // Determine stride.
    if (failed(verifyStride(mType)))
      return failure();
    Value stride = getStride(rewriter, *getTypeConverter(), mType,
                             adaptor.getBase(), op.getLoc());
    Value ptr = getStridedElementPtr(op.getLoc(), mType, adaptor.getBase(),
                                     adaptor.getIndices(), rewriter);

    if (enablingAnalysis && enablingAnalysis->get().isValid()) {
      rewriter.setInsertionPoint(op);
      // Routine for lowering tile Ops with binding info.
      auto dstRegIndex = op.getDstRegIndex();
      assert(dstRegIndex && "Incomplete operation attribute for tile binding");
      rewriter.create<amx::x86_amx_tileloadd64_plain>(op.getLoc(), *dstRegIndex,
                                                      ptr, stride);
      rewriter.eraseOp(op);
      return success();
    }

    VectorType vType = op.getVectorType();
    // Determine m x n tile sizes.
    std::pair<Value, Value> tsz =
        getTileSizes(rewriter, *getTypeConverter(), vType, op.getLoc());
    // Replace operation with intrinsic.
    Type resType = typeConverter->convertType(vType);
    rewriter.replaceOpWithNewOp<amx::x86_amx_tileloadd64>(
        op, resType, tsz.first, tsz.second, ptr, stride);
    return success();
  }
};

struct TileStoreConversion : public ConvertOpToLLVMPattern<TileStoreOp> {
private:
  const std::optional<std::reference_wrapper<TileScopeAnalysis>>
      &enablingAnalysis;

public:
  using ConvertOpToLLVMPattern<TileStoreOp>::ConvertOpToLLVMPattern;
  TileStoreConversion(
      const LLVMTypeConverter &typeConverter,
      const std::optional<std::reference_wrapper<TileScopeAnalysis>> &analysis)
      : ConvertOpToLLVMPattern<TileStoreOp>(typeConverter),
        enablingAnalysis(analysis) {}

  LogicalResult
  matchAndRewrite(TileStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType mType = op.getMemRefType();
    // Determine stride.
    if (failed(verifyStride(mType)))
      return failure();
    Value stride = getStride(rewriter, *getTypeConverter(), mType,
                             adaptor.getBase(), op.getLoc());
    Value ptr = getStridedElementPtr(op.getLoc(), mType, adaptor.getBase(),
                                     adaptor.getIndices(), rewriter);

    if (enablingAnalysis && enablingAnalysis->get().isValid()) {
      rewriter.setInsertionPoint(op);
      // Routine for lowering tile Ops with binding info.
      auto srcRegIndex = op.getSrcRegIndex();
      assert(srcRegIndex && "Incomplete operation attribute for tile binding");
      rewriter.create<amx::x86_amx_tilestored64_plain>(
          op.getLoc(), *srcRegIndex, ptr, stride);
      rewriter.eraseOp(op);
      return success();
    }

    VectorType vType = op.getVectorType();
    // Determine m x n tile sizes.
    std::pair<Value, Value> tsz =
        getTileSizes(rewriter, *getTypeConverter(), vType, op.getLoc());
    // Replace operation with intrinsic.
    rewriter.replaceOpWithNewOp<amx::x86_amx_tilestored64>(
        op, tsz.first, tsz.second, ptr, stride, adaptor.getVal());
    return success();
  }
};

struct TileMulFConversion : public ConvertOpToLLVMPattern<TileMulFOp> {
private:
  const std::optional<std::reference_wrapper<TileScopeAnalysis>>
      &enablingAnalysis;

public:
  using ConvertOpToLLVMPattern<TileMulFOp>::ConvertOpToLLVMPattern;
  TileMulFConversion(
      const LLVMTypeConverter &typeConverter,
      const std::optional<std::reference_wrapper<TileScopeAnalysis>> &analysis)
      : ConvertOpToLLVMPattern<TileMulFOp>(typeConverter),
        enablingAnalysis(analysis) {}

  LogicalResult
  matchAndRewrite(TileMulFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (enablingAnalysis && enablingAnalysis->get().isValid()) {
      rewriter.setInsertionPoint(op);
      // Routine for lowering tile Ops with binding info.
      auto lhsRegIndex = op.getLhsRegIndex();
      auto rhsRegIndex = op.getRhsRegIndex();
      auto accRegIndex = op.getAccRegIndex();

      assert(lhsRegIndex && rhsRegIndex && accRegIndex &&
             "Incomplete operation attribute for tile binding");
      rewriter.create<amx::x86_amx_tdpbf16ps_plain>(op.getLoc(), *accRegIndex,
                                                    *lhsRegIndex, *rhsRegIndex);
      rewriter.eraseOp(op);
      return success();
    }

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
private:
  const std::optional<std::reference_wrapper<TileScopeAnalysis>>
      &enablingAnalysis;

public:
  using ConvertOpToLLVMPattern<TileMulIOp>::ConvertOpToLLVMPattern;
  TileMulIConversion(
      const LLVMTypeConverter &typeConverter,
      const std::optional<std::reference_wrapper<TileScopeAnalysis>> &analysis)
      : ConvertOpToLLVMPattern<TileMulIOp>(typeConverter),
        enablingAnalysis(analysis) {}

  LogicalResult
  matchAndRewrite(TileMulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool zexta = op.getIsZextLhs();
    bool zextb = op.getIsZextRhs();

    if (enablingAnalysis && enablingAnalysis->get().isValid()) {
      rewriter.setInsertionPoint(op);
      // Routine for lowering tile Ops with binding info.
      auto lhsRegIndex = op.getLhsRegIndex();
      auto rhsRegIndex = op.getRhsRegIndex();
      auto accRegIndex = op.getAccRegIndex();

      assert(lhsRegIndex && rhsRegIndex && accRegIndex &&
             "Incomplete operation attribute for tile binding");
      if (zexta && zextb)
        rewriter.create<amx::x86_amx_tdpbuud_plain>(op.getLoc(), *accRegIndex,
                                                    *lhsRegIndex, *rhsRegIndex);
      else if (zexta && !zextb)
        rewriter.create<amx::x86_amx_tdpbusd_plain>(op.getLoc(), *accRegIndex,
                                                    *lhsRegIndex, *rhsRegIndex);
      else if (!zexta && zextb)
        rewriter.create<amx::x86_amx_tdpbsud_plain>(op.getLoc(), *accRegIndex,
                                                    *lhsRegIndex, *rhsRegIndex);
      else
        rewriter.create<amx::x86_amx_tdpbssd_plain>(op.getLoc(), *accRegIndex,
                                                    *lhsRegIndex, *rhsRegIndex);
      rewriter.eraseOp(op);
      return success();
    }

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
    LLVMTypeConverter &converter,
    std::optional<std::reference_wrapper<TileScopeAnalysis>> &analysis,
    RewritePatternSet &patterns) {
  patterns.add<TileZeroConversion, TileLoadConversion, TileStoreConversion,
               TileMulFConversion, TileMulIConversion>(converter, analysis);
}

void mlir::configureAMXLegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalOp<x86_amx_tilezero, x86_amx_tileloadd64, x86_amx_tilestored64,
                    x86_amx_tdpbf16ps, x86_amx_tdpbssd, x86_amx_tdpbsud,
                    x86_amx_tdpbusd, x86_amx_tdpbuud, x86_amx_ldtilecfg_plain,
                    x86_amx_tilerelease_plain, x86_amx_tilezero_plain,
                    x86_amx_tileloadd64_plain, x86_amx_tileloaddt164_plain,
                    x86_amx_tilestored64_plain, x86_amx_tdpbf16ps_plain,
                    x86_amx_tdpbssd_plain, x86_amx_tdpbsud_plain,
                    x86_amx_tdpbusd_plain, x86_amx_tdpbuud_plain>();
  target.addIllegalOp<TileZeroOp, TileLoadOp, TileStoreOp, TileMulIOp,
                      TileMulFOp>();
}
