//===- LegalizeForLLVMExport.cpp - Prepare AMX for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/Transforms.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
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
                                     amx::TileType tType, Location loc) {
  Type llvmInt16Type = IntegerType::get(&typeConverter.getContext(), 16);
  unsigned width = tType.getElementType().getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_64(width) && width >= 8);
  unsigned bytes = width >> 3;
  auto mattr = rewriter.getI16IntegerAttr(tType.getDimSize(0));
  auto nattr = rewriter.getI16IntegerAttr(tType.getDimSize(1) * bytes);
  return std::make_pair(
      rewriter.create<LLVM::ConstantOp>(loc, llvmInt16Type, mattr),
      rewriter.create<LLVM::ConstantOp>(loc, llvmInt16Type, nattr));
}

/// Maps the 2-dim memref shape to the 64-bit stride. Note that the buffer
/// shape may "envelop" the actual tile shape, and may be dynamically sized.
/// Returns failure if proper stride couldn't be found.
FailureOr<Value> getStride(ConversionPatternRewriter &rewriter,
                           const LLVMTypeConverter &typeConverter,
                           MemRefType mType, Value base, Location loc) {
  if (mType.getRank() < 2)
    return failure();
  int64_t preLast = mType.getRank() - 2;
  Type llvmInt64Type = IntegerType::get(&typeConverter.getContext(), 64);
  unsigned width = mType.getElementType().getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_64(width) && width >= 8);
  unsigned bytes = width >> 3;
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(mType, strides, offset)) ||
      strides.back() != 1)
    return failure();
  if (strides[preLast] == ShapedType::kDynamic) {
    // Dynamic stride needs code to compute the stride at runtime.
    MemRefDescriptor memrefDescriptor(base);
    auto attr = rewriter.getI64IntegerAttr(bytes);
    Value scale = rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, attr);
    return rewriter
        .create<LLVM::MulOp>(loc, llvmInt64Type, scale,
                             memrefDescriptor.stride(rewriter, loc, preLast))
        .getResult();
  }
  // Use direct constant for static stride.
  auto attr = rewriter.getI64IntegerAttr(strides[preLast] * bytes);
  return rewriter.create<LLVM::ConstantOp>(loc, llvmInt64Type, attr)
      .getResult();
}

struct TileZeroConversion : public ConvertOpToLLVMPattern<TileZeroOp> {
  using ConvertOpToLLVMPattern<TileZeroOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TileZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    amx::TileType tType = op.getTileType();
    // Determine m x n tile sizes.
    std::pair<Value, Value> tsz =
        getTileSizes(rewriter, *getTypeConverter(), tType, op.getLoc());
    // Replace operation with intrinsic.
    Type resType = typeConverter->convertType(tType);
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
    amx::TileType tType = op.getTileType();
    // Determine m x n tile sizes.
    std::pair<Value, Value> tsz =
        getTileSizes(rewriter, *getTypeConverter(), tType, op.getLoc());
    // Determine stride.
    auto stride = getStride(rewriter, *getTypeConverter(), mType,
                            adaptor.getBase(), op.getLoc());
    if (failed(stride))
      return failure();
    // Replace operation with intrinsic.
    Value ptr = getStridedElementPtr(op.getLoc(), mType, adaptor.getBase(),
                                     adaptor.getIndices(), rewriter);
    Type resType = typeConverter->convertType(tType);
    rewriter.replaceOpWithNewOp<amx::x86_amx_tileloadd64>(
        op, resType, tsz.first, tsz.second, ptr, stride.value());
    return success();
  }
};

struct TileStoreConversion : public ConvertOpToLLVMPattern<TileStoreOp> {
  using ConvertOpToLLVMPattern<TileStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TileStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType mType = op.getMemRefType();
    amx::TileType tType = op.getTileType();
    // Determine m x n tile sizes.
    std::pair<Value, Value> tsz =
        getTileSizes(rewriter, *getTypeConverter(), tType, op.getLoc());
    // Determine stride.
    auto stride = getStride(rewriter, *getTypeConverter(), mType,
                            adaptor.getBase(), op.getLoc());
    if (failed(stride))
      return failure();
    // Replace operation with intrinsic.
    Value ptr = getStridedElementPtr(op.getLoc(), mType, adaptor.getBase(),
                                     adaptor.getIndices(), rewriter);
    rewriter.replaceOpWithNewOp<amx::x86_amx_tilestored64>(
        op, tsz.first, tsz.second, ptr, stride.value(), adaptor.getVal());
    return success();
  }
};

struct TileMulFConversion : public ConvertOpToLLVMPattern<TileMulFOp> {
  using ConvertOpToLLVMPattern<TileMulFOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TileMulFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    amx::TileType aType = op.getLhsTileType();
    amx::TileType bType = op.getRhsTileType();
    amx::TileType cType = op.getTileType();
    // Determine m x n x k tile sizes.
    std::pair<Value, Value> tsza =
        getTileSizes(rewriter, *getTypeConverter(), aType, op.getLoc());
    std::pair<Value, Value> tszb =
        getTileSizes(rewriter, *getTypeConverter(), bType, op.getLoc());
    // Replace operation with intrinsic.
    Type resType = typeConverter->convertType(cType);
    if (aType.getElementType().isBF16())
      rewriter.replaceOpWithNewOp<amx::x86_amx_tdpbf16ps>(
          op, resType, tsza.first, tszb.second, tsza.second, adaptor.getAcc(),
          adaptor.getLhs(), adaptor.getRhs());
    else if (aType.getElementType().isF16())
      rewriter.replaceOpWithNewOp<amx::x86_amx_tdpfp16ps>(
          op, resType, tsza.first, tszb.second, tsza.second, adaptor.getAcc(),
          adaptor.getLhs(), adaptor.getRhs());
    else
      llvm_unreachable("Unexpected element type for amx.mulf");
    return success();
  }
};

struct TileMulIConversion : public ConvertOpToLLVMPattern<TileMulIOp> {
  using ConvertOpToLLVMPattern<TileMulIOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TileMulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    amx::TileType aType = op.getLhsTileType();
    amx::TileType bType = op.getRhsTileType();
    amx::TileType cType = op.getTileType();
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
  converter.addConversion([&](amx::TileType type) {
    return LLVM::LLVMX86AMXType::get(&converter.getContext());
  });
}

void mlir::configureAMXLegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalOp<x86_amx_tilezero, x86_amx_tileloadd64, x86_amx_tilestored64,
                    x86_amx_tdpbf16ps, x86_amx_tdpfp16ps, x86_amx_tdpbssd,
                    x86_amx_tdpbsud, x86_amx_tdpbusd, x86_amx_tdpbuud>();
  target.addIllegalOp<TileZeroOp, TileLoadOp, TileStoreOp, TileMulIOp,
                      TileMulFOp>();
}

namespace {
/// Implement the interface to convert AMX to LLVM.
struct AMXToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateAMXLegalizeForLLVMExportPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::registerConvertAMXToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, amx::AMXDialect *dialect) {
    dialect->addInterfaces<AMXToLLVMDialectInterface>();
  });
}
