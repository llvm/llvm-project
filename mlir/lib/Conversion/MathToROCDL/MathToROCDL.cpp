//===-- MathToROCDL.cpp - conversion from Math to rocdl calls -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToROCDL/MathToROCDL.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOROCDL
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "math-to-rocdl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

template <typename OpTy>
static void populateOpPatterns(LLVMTypeConverter &converter,
                               RewritePatternSet &patterns, StringRef f32Func,
                               StringRef f64Func) {
  patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter);
  patterns.add<OpToFuncCallLowering<OpTy>>(converter, f32Func, f64Func);
}

void mlir::populateMathToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns) {
  // Handled by mathToLLVM: math::AbsIOp
  // Handled by mathToLLVM: math::CopySignOp
  // Handled by mathToLLVM: math::CountLeadingZerosOp
  // Handled by mathToLLVM: math::CountTrailingZerosOp
  // Handled by mathToLLVM: math::CgPopOp
  // Handled by mathToLLVM: math::FmaOp
  // FIXME: math::IPowIOp
  // FIXME: math::FPowIOp
  // Handled by mathToLLVM: math::RoundEvenOp
  // Handled by mathToLLVM: math::RoundOp
  // Handled by mathToLLVM: math::TruncOp
  populateOpPatterns<math::AbsFOp>(converter, patterns, "__ocml_fabs_f32",
                                   "__ocml_fabs_f64");
  populateOpPatterns<math::AcosOp>(converter, patterns, "__ocml_acos_f32",
                                   "__ocml_acos_f64");
  populateOpPatterns<math::AcoshOp>(converter, patterns, "__ocml_acosh_f32",
                                    "__ocml_acosh_f64");
  populateOpPatterns<math::AsinOp>(converter, patterns, "__ocml_asin_f32",
                                   "__ocml_asin_f64");
  populateOpPatterns<math::AsinhOp>(converter, patterns, "__ocml_asinh_f32",
                                    "__ocml_asinh_f64");
  populateOpPatterns<math::AtanOp>(converter, patterns, "__ocml_atan_f32",
                                   "__ocml_atan_f64");
  populateOpPatterns<math::AtanhOp>(converter, patterns, "__ocml_atanh_f32",
                                    "__ocml_atanh_f64");
  populateOpPatterns<math::Atan2Op>(converter, patterns, "__ocml_atan2_f32",
                                    "__ocml_atan2_f64");
  populateOpPatterns<math::CbrtOp>(converter, patterns, "__ocml_cbrt_f32",
                                   "__ocml_cbrt_f64");
  populateOpPatterns<math::CeilOp>(converter, patterns, "__ocml_ceil_f32",
                                   "__ocml_ceil_f64");
  populateOpPatterns<math::CosOp>(converter, patterns, "__ocml_cos_f32",
                                  "__ocml_cos_f64");
  populateOpPatterns<math::CoshOp>(converter, patterns, "__ocml_cosh_f32",
                                   "__ocml_cosh_f64");
  populateOpPatterns<math::SinhOp>(converter, patterns, "__ocml_sinh_f32",
                                   "__ocml_sinh_f64");
  populateOpPatterns<math::ExpOp>(converter, patterns, "__ocml_exp_f32",
                                  "__ocml_exp_f64");
  populateOpPatterns<math::Exp2Op>(converter, patterns, "__ocml_exp2_f32",
                                   "__ocml_exp2_f64");
  populateOpPatterns<math::ExpM1Op>(converter, patterns, "__ocml_expm1_f32",
                                    "__ocml_expm1_f64");
  populateOpPatterns<math::FloorOp>(converter, patterns, "__ocml_floor_f32",
                                    "__ocml_floor_f64");
  populateOpPatterns<math::LogOp>(converter, patterns, "__ocml_log_f32",
                                  "__ocml_log_f64");
  populateOpPatterns<math::Log10Op>(converter, patterns, "__ocml_log10_f32",
                                    "__ocml_log10_f64");
  populateOpPatterns<math::Log1pOp>(converter, patterns, "__ocml_log1p_f32",
                                    "__ocml_log1p_f64");
  populateOpPatterns<math::Log2Op>(converter, patterns, "__ocml_log2_f32",
                                   "__ocml_log2_f64");
  populateOpPatterns<math::PowFOp>(converter, patterns, "__ocml_pow_f32",
                                   "__ocml_pow_f64");
  populateOpPatterns<math::RsqrtOp>(converter, patterns, "__ocml_rsqrt_f32",
                                    "__ocml_rsqrt_f64");
  populateOpPatterns<math::SinOp>(converter, patterns, "__ocml_sin_f32",
                                  "__ocml_sin_f64");
  populateOpPatterns<math::SqrtOp>(converter, patterns, "__ocml_sqrt_f32",
                                   "__ocml_sqrt_f64");
  populateOpPatterns<math::TanhOp>(converter, patterns, "__ocml_tanh_f32",
                                   "__ocml_tanh_f64");
  populateOpPatterns<math::TanOp>(converter, patterns, "__ocml_tan_f32",
                                  "__ocml_tan_f64");
  populateOpPatterns<math::ErfOp>(converter, patterns, "__ocml_erf_f32",
                                  "__ocml_erf_f64");
  // Single arith pattern that needs a ROCDL call, probably not
  // worth creating a separate pass for it.
  populateOpPatterns<arith::RemFOp>(converter, patterns, "__ocml_fmod_f32",
                                    "__ocml_fmod_f64");
}

namespace {
struct ConvertMathToROCDLPass
    : public impl::ConvertMathToROCDLBase<ConvertMathToROCDLPass> {
  ConvertMathToROCDLPass() = default;
  void runOnOperation() override;
};
} // namespace

void ConvertMathToROCDLPass::runOnOperation() {
  auto m = getOperation();
  MLIRContext *ctx = m.getContext();

  RewritePatternSet patterns(&getContext());
  LowerToLLVMOptions options(ctx, DataLayout(m));
  LLVMTypeConverter converter(ctx, options);
  populateMathToROCDLConversionPatterns(converter, patterns);
  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, func::FuncDialect,
                         vector::VectorDialect, LLVM::LLVMDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::FRemOp, LLVM::LogOp,
                      LLVM::Log10Op, LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp,
                      LLVM::SqrtOp>();
  if (failed(applyPartialConversion(m, target, std::move(patterns))))
    signalPassFailure();
}
