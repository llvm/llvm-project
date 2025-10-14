//===-- MathToROCDL.cpp - conversion from Math to rocdl calls -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToROCDL/MathToROCDL.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOROCDL
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "math-to-rocdl"

template <typename OpTy>
static void populateOpPatterns(const LLVMTypeConverter &converter,
                               RewritePatternSet &patterns, StringRef f32Func,
                               StringRef f64Func, StringRef f16Func,
                               StringRef f32ApproxFunc = "") {
  patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter);
  patterns.add<OpToFuncCallLowering<OpTy>>(converter, f32Func, f64Func,
                                           f32ApproxFunc, f16Func);
}

struct ClampFOpConversion final
    : public ConvertOpToLLVMPattern<math::ClampFOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  ClampFOpConversion(const LLVMTypeConverter &converter,
                     amdgpu::Chipset chipset)
      : ConvertOpToLLVMPattern<math::ClampFOp>(converter), chipset(chipset) {}

  LogicalResult
  matchAndRewrite(math::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only f16 and f32 types are supported by fmed3
    Type opTy = op.getType();
    auto resultType = getTypeConverter()->convertType(opTy);

    if (auto vectorType = dyn_cast<VectorType>(opTy)) {
      opTy = vectorType.getElementType();
    }

    if (!isa<Float16Type, Float32Type>(opTy)) {
      return rewriter.notifyMatchFailure(
          op, "fmed3 only supports f16 and f32 types");
    }

    // Handle multi-dimensional vectors (converted to LLVM arrays)
    if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(resultType)) {
      // Handle multi-dimensional vectors (converted to LLVM arrays)
      return LLVM::detail::handleMultidimensionalVectors(
          op.getOperation(), adaptor.getOperands(), *getTypeConverter(),
          [&](Type llvm1DVectorTy, ValueRange operands) -> Value {
            typename math::ClampFOp::Adaptor adaptor(operands);
            return ROCDL::FMed3Op::create(rewriter, op.getLoc(), llvm1DVectorTy,
                                          adaptor.getValue(), adaptor.getMin(),
                                          adaptor.getMax());
          },
          rewriter);
    }

    // Handle 1D vectors and scalars directly
    rewriter.replaceOpWithNewOp<ROCDL::FMed3Op>(op, op.getType(), op.getValue(),
                                                op.getMin(), op.getMax());
    return success();
  }

  amdgpu::Chipset chipset;
};

static void addChipsetDependentPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns,
                                        amdgpu::Chipset chipset) {

  // V_MED3_F16/F32 only exists in gfx9+ architectures
  if (chipset.majorVersion >= 9) {
    patterns.add<ClampFOpConversion>(converter, chipset);
  }
}

void mlir::populateMathToROCDLConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    amdgpu::Chipset chipset) {
  // Handled by mathToLLVM: math::AbsIOp
  // Handled by mathToLLVM: math::AbsFOp
  // Handled by mathToLLVM: math::CopySignOp
  // Handled by mathToLLVM: math::CountLeadingZerosOp
  // Handled by mathToLLVM: math::CountTrailingZerosOp
  // Handled by mathToLLVM: math::CgPopOp
  // Handled by mathToLLVM: math::ExpOp (32-bit only)
  // Handled by mathToLLVM: math::FmaOp
  // Handled by mathToLLVM: math::LogOp (32-bit only)
  // FIXME: math::IPowIOp
  // Handled by mathToLLVM: math::RoundEvenOp
  // Handled by mathToLLVM: math::RoundOp
  // Handled by mathToLLVM: math::SqrtOp
  // Handled by mathToLLVM: math::TruncOp
  populateOpPatterns<math::AcosOp>(converter, patterns, "__ocml_acos_f32",
                                   "__ocml_acos_f64", "__ocml_acos_f16");
  populateOpPatterns<math::AcoshOp>(converter, patterns, "__ocml_acosh_f32",
                                    "__ocml_acosh_f64", "__ocml_acosh_f16");
  populateOpPatterns<math::AsinOp>(converter, patterns, "__ocml_asin_f32",
                                   "__ocml_asin_f64", "__ocml_asin_f16");
  populateOpPatterns<math::AsinhOp>(converter, patterns, "__ocml_asinh_f32",
                                    "__ocml_asinh_f64", "__ocml_asinh_f16");
  populateOpPatterns<math::AtanOp>(converter, patterns, "__ocml_atan_f32",
                                   "__ocml_atan_f64", "__ocml_atan_f16");
  populateOpPatterns<math::AtanhOp>(converter, patterns, "__ocml_atanh_f32",
                                    "__ocml_atanh_f64", "__ocml_atanh_f16");
  populateOpPatterns<math::Atan2Op>(converter, patterns, "__ocml_atan2_f32",
                                    "__ocml_atan2_f64", "__ocml_atan2_f16");
  populateOpPatterns<math::CbrtOp>(converter, patterns, "__ocml_cbrt_f32",
                                   "__ocml_cbrt_f64", "__ocml_cbrt_f16");
  populateOpPatterns<math::CeilOp>(converter, patterns, "__ocml_ceil_f32",
                                   "__ocml_ceil_f64", "__ocml_ceil_f16");
  populateOpPatterns<math::CosOp>(converter, patterns, "__ocml_cos_f32",
                                  "__ocml_cos_f64", "__ocml_cos_f16");
  populateOpPatterns<math::CoshOp>(converter, patterns, "__ocml_cosh_f32",
                                   "__ocml_cosh_f64", "__ocml_cosh_f16");
  populateOpPatterns<math::SinhOp>(converter, patterns, "__ocml_sinh_f32",
                                   "__ocml_sinh_f64", "__ocml_sinh_f16");
  populateOpPatterns<math::ExpOp>(converter, patterns, "", "__ocml_exp_f64",
                                  "__ocml_exp_f16");
  populateOpPatterns<math::Exp2Op>(converter, patterns, "__ocml_exp2_f32",
                                   "__ocml_exp2_f64", "__ocml_exp2_f16");
  populateOpPatterns<math::ExpM1Op>(converter, patterns, "__ocml_expm1_f32",
                                    "__ocml_expm1_f64", "__ocml_expm1_f16");
  populateOpPatterns<math::FloorOp>(converter, patterns, "__ocml_floor_f32",
                                    "__ocml_floor_f64", "__ocml_floor_f16");
  populateOpPatterns<math::LogOp>(converter, patterns, "", "__ocml_log_f64",
                                  "__ocml_log_f16");
  populateOpPatterns<math::Log10Op>(converter, patterns, "__ocml_log10_f32",
                                    "__ocml_log10_f64", "__ocml_log10_f16");
  populateOpPatterns<math::Log1pOp>(converter, patterns, "__ocml_log1p_f32",
                                    "__ocml_log1p_f64", "__ocml_log1p_f16");
  populateOpPatterns<math::Log2Op>(converter, patterns, "__ocml_log2_f32",
                                   "__ocml_log2_f64", "__ocml_log2_f16");
  populateOpPatterns<math::PowFOp>(converter, patterns, "__ocml_pow_f32",
                                   "__ocml_pow_f64", "__ocml_pow_f16");
  populateOpPatterns<math::RsqrtOp>(converter, patterns, "__ocml_rsqrt_f32",
                                    "__ocml_rsqrt_f64", "__ocml_rsqrt_f16");
  populateOpPatterns<math::SinOp>(converter, patterns, "__ocml_sin_f32",
                                  "__ocml_sin_f64", "__ocml_sin_f16");
  populateOpPatterns<math::TanhOp>(converter, patterns, "__ocml_tanh_f32",
                                   "__ocml_tanh_f64", "__ocml_tanh_f16");
  populateOpPatterns<math::TanOp>(converter, patterns, "__ocml_tan_f32",
                                  "__ocml_tan_f64", "__ocml_tan_f16");
  populateOpPatterns<math::ErfOp>(converter, patterns, "__ocml_erf_f32",
                                  "__ocml_erf_f64", "__ocml_erf_f16");
  populateOpPatterns<math::ErfcOp>(converter, patterns, "__ocml_erfc_f32",
                                   "__ocml_erfc_f64", "__ocml_erfc_f16");
  populateOpPatterns<math::FPowIOp>(converter, patterns, "__ocml_pown_f32",
                                    "__ocml_pown_f64", "__ocml_pown_f16");
  // Single arith pattern that needs a ROCDL call, probably not
  // worth creating a separate pass for it.
  populateOpPatterns<arith::RemFOp>(converter, patterns, "__ocml_fmod_f32",
                                    "__ocml_fmod_f64", "__ocml_fmod_f16");

  addChipsetDependentPatterns(converter, patterns, chipset);
}

struct ConvertMathToROCDLPass final
    : impl::ConvertMathToROCDLBase<ConvertMathToROCDLPass> {
  using impl::ConvertMathToROCDLBase<
      ConvertMathToROCDLPass>::ConvertMathToROCDLBase;

  void runOnOperation() override;
};

void ConvertMathToROCDLPass::runOnOperation() {
  auto m = getOperation();
  MLIRContext *ctx = m.getContext();

  RewritePatternSet patterns(&getContext());
  LowerToLLVMOptions options(ctx, DataLayout(m));
  LLVMTypeConverter converter(ctx, options);

  // Only populate chipset-dependent patterns if chipset is specified
  if (!chipset.empty()) {
    FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      return signalPassFailure();
    }
    populateMathToROCDLConversionPatterns(converter, patterns, *maybeChipset);
  }

  ConversionTarget target(getContext());
  target
      .addLegalDialect<BuiltinDialect, func::FuncDialect, vector::VectorDialect,
                       LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::FRemOp, LLVM::LogOp,
                      LLVM::Log10Op, LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp,
                      LLVM::SqrtOp>();
  if (failed(applyPartialConversion(m, target, std::move(patterns))))
    signalPassFailure();
}
