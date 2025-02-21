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
static void populateOpPatterns(const LLVMTypeConverter &converter,
                               RewritePatternSet &patterns,
                               MathToROCDLConversionPatternKind patternKind,
                               StringRef f32Func, StringRef f64Func,
                               StringRef f16Func,
                               StringRef f32ApproxFunc = "") {
  if (patternKind == MathToROCDLConversionPatternKind::All ||
      patternKind == MathToROCDLConversionPatternKind::Scalarizations) {
    patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter);
  }
  if (patternKind == MathToROCDLConversionPatternKind::All ||
      patternKind == MathToROCDLConversionPatternKind::Lowerings) {
    patterns.add<OpToFuncCallLowering<OpTy>>(converter, f32Func, f64Func,
                                             f32ApproxFunc, f16Func);
  }
}

void mlir::populateMathToROCDLConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    MathToROCDLConversionPatternKind patternKind) {
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
  populateOpPatterns<math::AcosOp>(converter, patterns, patternKind,
                                   "__ocml_acos_f32", "__ocml_acos_f64",
                                   "__ocml_acos_f16");
  populateOpPatterns<math::AcoshOp>(converter, patterns, patternKind,
                                    "__ocml_acosh_f32", "__ocml_acosh_f64",
                                    "__ocml_acosh_f16");
  populateOpPatterns<math::AsinOp>(converter, patterns, patternKind,
                                   "__ocml_asin_f32", "__ocml_asin_f64",
                                   "__ocml_asin_f16");
  populateOpPatterns<math::AsinhOp>(converter, patterns, patternKind,
                                    "__ocml_asinh_f32", "__ocml_asinh_f64",
                                    "__ocml_asinh_f16");
  populateOpPatterns<math::AtanOp>(converter, patterns, patternKind,
                                   "__ocml_atan_f32", "__ocml_atan_f64",
                                   "__ocml_atan_f16");
  populateOpPatterns<math::AtanhOp>(converter, patterns, patternKind,
                                    "__ocml_atanh_f32", "__ocml_atanh_f64",
                                    "__ocml_atanh_f16");
  populateOpPatterns<math::Atan2Op>(converter, patterns, patternKind,
                                    "__ocml_atan2_f32", "__ocml_atan2_f64",
                                    "__ocml_atan2_f16");
  populateOpPatterns<math::CbrtOp>(converter, patterns, patternKind,
                                   "__ocml_cbrt_f32", "__ocml_cbrt_f64",
                                   "__ocml_cbrt_f16");
  populateOpPatterns<math::CeilOp>(converter, patterns, patternKind,
                                   "__ocml_ceil_f32", "__ocml_ceil_f64",
                                   "__ocml_ceil_f16");
  populateOpPatterns<math::CosOp>(converter, patterns, patternKind,
                                  "__ocml_cos_f32", "__ocml_cos_f64",
                                  "__ocml_cos_f16");
  populateOpPatterns<math::CoshOp>(converter, patterns, patternKind,
                                   "__ocml_cosh_f32", "__ocml_cosh_f64",
                                   "__ocml_cosh_f16");
  populateOpPatterns<math::SinhOp>(converter, patterns, patternKind,
                                   "__ocml_sinh_f32", "__ocml_sinh_f64",
                                   "__ocml_sinh_f16");
  populateOpPatterns<math::ExpOp>(converter, patterns, patternKind, "",
                                  "__ocml_exp_f64", "__ocml_exp_f16");
  populateOpPatterns<math::Exp2Op>(converter, patterns, patternKind,
                                   "__ocml_exp2_f32", "__ocml_exp2_f64",
                                   "__ocml_exp2_f16");
  populateOpPatterns<math::ExpM1Op>(converter, patterns, patternKind,
                                    "__ocml_expm1_f32", "__ocml_expm1_f64",
                                    "__ocml_expm1_f16");
  populateOpPatterns<math::FloorOp>(converter, patterns, patternKind,
                                    "__ocml_floor_f32", "__ocml_floor_f64",
                                    "__ocml_floor_f16");
  populateOpPatterns<math::LogOp>(converter, patterns, patternKind, "",
                                  "__ocml_log_f64", "__ocml_log_f16");
  populateOpPatterns<math::Log10Op>(converter, patterns, patternKind,
                                    "__ocml_log10_f32", "__ocml_log10_f64",
                                    "__ocml_log10_f16");
  populateOpPatterns<math::Log1pOp>(converter, patterns, patternKind,
                                    "__ocml_log1p_f32", "__ocml_log1p_f64",
                                    "__ocml_log1p_f16");
  populateOpPatterns<math::Log2Op>(converter, patterns, patternKind,
                                   "__ocml_log2_f32", "__ocml_log2_f64",
                                   "__ocml_log2_f16");
  populateOpPatterns<math::PowFOp>(converter, patterns, patternKind,
                                   "__ocml_pow_f32", "__ocml_pow_f64",
                                   "__ocml_pow_f16");
  populateOpPatterns<math::RsqrtOp>(converter, patterns, patternKind,
                                    "__ocml_rsqrt_f32", "__ocml_rsqrt_f64",
                                    "__ocml_rsqrt_f16");
  populateOpPatterns<math::SinOp>(converter, patterns, patternKind,
                                  "__ocml_sin_f32", "__ocml_sin_f64",
                                  "__ocml_sin_f16");
  populateOpPatterns<math::TanhOp>(converter, patterns, patternKind,
                                   "__ocml_tanh_f32", "__ocml_tanh_f64",
                                   "__ocml_tanh_f16");
  populateOpPatterns<math::TanOp>(converter, patterns, patternKind,
                                  "__ocml_tan_f32", "__ocml_tan_f64",
                                  "__ocml_tan_f16");
  populateOpPatterns<math::ErfOp>(converter, patterns, patternKind,
                                  "__ocml_erf_f32", "__ocml_erf_f64",
                                  "__ocml_erf_f16");
  populateOpPatterns<math::FPowIOp>(converter, patterns, patternKind,
                                    "__ocml_pown_f32", "__ocml_pown_f64",
                                    "__ocml_pown_f16");
  // Single arith pattern that needs a ROCDL call, probably not
  // worth creating a separate pass for it.
  populateOpPatterns<arith::RemFOp>(converter, patterns, patternKind,
                                    "__ocml_fmod_f32", "__ocml_fmod_f64",
                                    "__ocml_fmod_f16");
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

  LowerToLLVMOptions options(ctx, DataLayout(m));
  LLVMTypeConverter converter(ctx, options);

  // The two pattern applications below will use distinct ConversionTarget's,
  // but this is the common denominator.
  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, func::FuncDialect,
                         vector::VectorDialect, LLVM::LLVMDialect>();

  // Perform the scalarizations. This is done in a separate pattern application
  // to ensure that scalarizations are done regardless of lowerings. It is
  // normal for some lowerings may fail to apply, when we purposely do not lower
  // a math op to a function call.
  RewritePatternSet scalarizationPatterns(&getContext());
  ConversionTarget scalarizationTarget(target);
  // Math ops are legal if their operands are not vectors.
  scalarizationTarget.addDynamicallyLegalDialect<math::MathDialect>(
      [&](Operation *op) {
        return llvm::none_of(op->getOperandTypes(), llvm::IsaPred<VectorType>);
      });
  populateMathToROCDLConversionPatterns(
      converter, scalarizationPatterns,
      MathToROCDLConversionPatternKind::Scalarizations);
  if (failed(applyPartialConversion(m, scalarizationTarget,
                                    std::move(scalarizationPatterns))))
    signalPassFailure();

  // Perform the lowerings. The ops that must lower to function calls become
  // illegal.
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::FRemOp, LLVM::LogOp,
                      LLVM::Log10Op, LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp,
                      LLVM::SqrtOp>();
  RewritePatternSet loweringPatterns(&getContext());
  populateMathToROCDLConversionPatterns(
      converter, loweringPatterns, MathToROCDLConversionPatternKind::Lowerings);
  if (failed(applyPartialConversion(m, target, std::move(loweringPatterns))))
    signalPassFailure();
}
