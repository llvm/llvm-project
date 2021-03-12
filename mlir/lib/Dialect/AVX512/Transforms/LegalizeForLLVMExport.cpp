//===- LegalizeForLLVMExport.cpp - Prepare AVX512 for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AVX512/Transforms.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/AVX512/AVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::avx512;

/// Extracts the "main" vector element type from the given AVX512 operation.
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
  matchAndRewrite(OpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType = getSrcVectorElementType<OpTy>(op);
    unsigned bitwidth = elementType.getIntOrFloatBitWidth();
    if (bitwidth == 32)
      return LLVM::detail::oneToOneRewrite(op, Intr32OpTy::getOperationName(),
                                           operands, getTypeConverter(),
                                           rewriter);
    if (bitwidth == 64)
      return LLVM::detail::oneToOneRewrite(op, Intr64OpTy::getOperationName(),
                                           operands, getTypeConverter(),
                                           rewriter);
    return rewriter.notifyMatchFailure(
        op, "expected 'src' to be either f32 or f64");
  }
};

struct MaskCompressOpConversion
    : public ConvertOpToLLVMPattern<MaskCompressOp> {
  using ConvertOpToLLVMPattern<MaskCompressOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MaskCompressOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    MaskCompressOp::Adaptor adaptor(operands);
    auto opType = adaptor.a().getType();

    Value src;
    if (op.src()) {
      src = adaptor.src();
    } else if (op.constant_src()) {
      src = rewriter.create<ConstantOp>(op.getLoc(), opType,
                                        op.constant_srcAttr());
    } else {
      Attribute zeroAttr = rewriter.getZeroAttr(opType);
      src = rewriter.create<ConstantOp>(op->getLoc(), opType, zeroAttr);
    }

    rewriter.replaceOpWithNewOp<MaskCompressIntrOp>(op, opType, adaptor.a(),
                                                    src, adaptor.k());

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
                               OwningRewritePatternList &patterns) {
    patterns
        .insert<LowerToIntrinsic<typename Args::MainOp, typename Args::Intr32Op,
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

/// Populate the given list with patterns that convert from AVX512 to LLVM.
void mlir::populateAVX512LegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  Registry::registerPatterns(converter, patterns);
  patterns.insert<MaskCompressOpConversion>(converter);
}

void mlir::configureAVX512LegalizeForExportTarget(
    LLVMConversionTarget &target) {
  Registry::configureTarget(target);
  target.addLegalOp<MaskCompressIntrOp>();
  target.addIllegalOp<MaskCompressOp>();
}
