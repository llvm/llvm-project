//===- LegalizeForLLVMExport.cpp - Prepare X86Vector for LLVM translation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/X86Vector/Transforms.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::x86vector;

/// Extracts the "main" vector element type from the given X86Vector operation.
template <typename OpTy>
static Type getSrcVectorElementType(OpTy op) {
  return op.src().getType().template cast<VectorType>().getElementType();
}
template <>
Type getSrcVectorElementType(Vp2IntersectOp op) {
  return op.a().getType().template cast<VectorType>().getElementType();
}

namespace {

/// Base conversion for AVX512 ops that can be lowered to one of the two
/// intrinsics based on the bitwidth of their "main" vector element type. This
/// relies on the to-LLVM-dialect conversion helpers to correctly pack the
/// results of multi-result intrinsic ops.
template <typename OpTy, typename Intr32OpTy, typename Intr64OpTy>
struct LowerToIntrinsic : public OpConversionPattern<OpTy> {
  explicit LowerToIntrinsic(LLVMTypeConverter &converter)
      : OpConversionPattern<OpTy>(converter, &converter.getContext()) {}

  LLVMTypeConverter &getTypeConverter() const {
    return *static_cast<LLVMTypeConverter *>(
        OpConversionPattern<OpTy>::getTypeConverter());
  }

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getSrcVectorElementType<OpTy>(op);
    unsigned bitwidth = elementType.getIntOrFloatBitWidth();
    if (bitwidth == 32)
      return LLVM::detail::oneToOneRewrite(op, Intr32OpTy::getOperationName(),
                                           adaptor.getOperands(),
                                           getTypeConverter(), rewriter);
    if (bitwidth == 64)
      return LLVM::detail::oneToOneRewrite(op, Intr64OpTy::getOperationName(),
                                           adaptor.getOperands(),
                                           getTypeConverter(), rewriter);
    return rewriter.notifyMatchFailure(
        op, "expected 'src' to be either f32 or f64");
  }
};

struct MaskCompressOpConversion
    : public ConvertOpToLLVMPattern<MaskCompressOp> {
  using ConvertOpToLLVMPattern<MaskCompressOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MaskCompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = adaptor.a().getType();

    Value src;
    if (op.src()) {
      src = adaptor.src();
    } else if (op.constant_src()) {
      src = rewriter.create<arith::ConstantOp>(op.getLoc(), opType,
                                               op.constant_srcAttr());
    } else {
      Attribute zeroAttr = rewriter.getZeroAttr(opType);
      src = rewriter.create<arith::ConstantOp>(op->getLoc(), opType, zeroAttr);
    }

    rewriter.replaceOpWithNewOp<MaskCompressIntrOp>(op, opType, adaptor.a(),
                                                    src, adaptor.k());

    return success();
  }
};

struct RsqrtOpConversion : public ConvertOpToLLVMPattern<RsqrtOp> {
  using ConvertOpToLLVMPattern<RsqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = adaptor.a().getType();
    rewriter.replaceOpWithNewOp<RsqrtIntrOp>(op, opType, adaptor.a());
    return success();
  }
};

struct DotOpConversion : public ConvertOpToLLVMPattern<DotOp> {
  using ConvertOpToLLVMPattern<DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = adaptor.a().getType();
    Type llvmIntType = IntegerType::get(&getTypeConverter()->getContext(), 8);
    // Dot product of all elements, broadcasted to all elements.
    auto attr = rewriter.getI8IntegerAttr(static_cast<int8_t>(0xff));
    Value scale =
        rewriter.create<LLVM::ConstantOp>(op.getLoc(), llvmIntType, attr);
    rewriter.replaceOpWithNewOp<DotIntrOp>(op, opType, adaptor.a(), adaptor.b(),
                                           scale);
    return success();
  }
};

/// An entry associating the "main" AVX512 op with its instantiations for
/// vectors of 32-bit and 64-bit elements.
template <typename OpTy, typename Intr32OpTy, typename Intr64OpTy>
struct RegEntry {
  using MainOp = OpTy;
  using Intr32Op = Intr32OpTy;
  using Intr64Op = Intr64OpTy;
};

/// A container for op association entries facilitating the configuration of
/// dialect conversion.
template <typename... Args>
struct RegistryImpl {
  /// Registers the patterns specializing the "main" op to one of the
  /// "intrinsic" ops depending on elemental type.
  static void registerPatterns(LLVMTypeConverter &converter,
                               RewritePatternSet &patterns) {
    patterns
        .add<LowerToIntrinsic<typename Args::MainOp, typename Args::Intr32Op,
                              typename Args::Intr64Op>...>(converter);
  }

  /// Configures the conversion target to lower out "main" ops.
  static void configureTarget(LLVMConversionTarget &target) {
    target.addIllegalOp<typename Args::MainOp...>();
    target.addLegalOp<typename Args::Intr32Op...>();
    target.addLegalOp<typename Args::Intr64Op...>();
  }
};

using Registry = RegistryImpl<
    RegEntry<MaskRndScaleOp, MaskRndScalePSIntrOp, MaskRndScalePDIntrOp>,
    RegEntry<MaskScaleFOp, MaskScaleFPSIntrOp, MaskScaleFPDIntrOp>,
    RegEntry<Vp2IntersectOp, Vp2IntersectDIntrOp, Vp2IntersectQIntrOp>>;

} // namespace

/// Populate the given list with patterns that convert from X86Vector to LLVM.
void mlir::populateX86VectorLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  Registry::registerPatterns(converter, patterns);
  patterns.add<MaskCompressOpConversion, RsqrtOpConversion, DotOpConversion>(
      converter);
}

void mlir::configureX86VectorLegalizeForExportTarget(
    LLVMConversionTarget &target) {
  Registry::configureTarget(target);
  target.addLegalOp<MaskCompressIntrOp>();
  target.addIllegalOp<MaskCompressOp>();
  target.addLegalOp<RsqrtIntrOp>();
  target.addIllegalOp<RsqrtOp>();
  target.addLegalOp<DotIntrOp>();
  target.addIllegalOp<DotOp>();
}
