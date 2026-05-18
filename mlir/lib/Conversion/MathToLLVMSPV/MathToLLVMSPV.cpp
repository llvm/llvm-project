//===-- MathToLLVMSPV.cpp - conversion from Math to SPIR-V builtin calls --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToLLVMSPV/MathToLLVMSPV.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOLLVMSPV
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "math-to-llvm-spv"

static bool isExtensionSetSupported(StringRef name) {
  return name == "OpenCL.std";
}

template <typename OpTy>
static void populateOpPatterns(const LLVMTypeConverter &converter,
                               RewritePatternSet &patterns,
                               PatternBenefit benefit, StringRef f32Func,
                               StringRef f64Func) {
  patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter, benefit);
  patterns.add<OpToFuncCallLowering<OpTy>>(converter, f32Func, f64Func,
                                           /*f32ApproxFunc=*/"", /*f16Func=*/"",
                                           /*i32Func=*/"", benefit,
                                           LLVM::cconv::CConv::SPIR_FUNC);
}

template <typename OpTy>
static void populateOCLExtSetOpPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit,
                                        StringRef opName) {
  std::string mangledName =
      "_Z" + std::to_string(12 + opName.size()) + "__spirv_ocl_" + opName.str();
  populateOpPatterns<OpTy>(converter, patterns, benefit, mangledName + "f",
                           mangledName + "d");
}

void mlir::populateMathToOCLExtSetLLVMSPVConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  populateOCLExtSetOpPatterns<math::AcosOp>(converter, patterns, benefit,
                                            "acos");
  populateOCLExtSetOpPatterns<math::AcoshOp>(converter, patterns, benefit,
                                             "acosh");
  populateOCLExtSetOpPatterns<math::AsinOp>(converter, patterns, benefit,
                                            "asin");
  populateOCLExtSetOpPatterns<math::AsinhOp>(converter, patterns, benefit,
                                             "asinh");
  populateOCLExtSetOpPatterns<math::AtanOp>(converter, patterns, benefit,
                                            "atan");
  populateOCLExtSetOpPatterns<math::Atan2Op>(converter, patterns, benefit,
                                             "atan2");
  populateOCLExtSetOpPatterns<math::AtanhOp>(converter, patterns, benefit,
                                             "atanh");
  populateOCLExtSetOpPatterns<math::CbrtOp>(converter, patterns, benefit,
                                            "cbrt");
  populateOCLExtSetOpPatterns<math::CopySignOp>(converter, patterns, benefit,
                                                "copysign");
  populateOCLExtSetOpPatterns<math::CosOp>(converter, patterns, benefit, "cos");
  populateOCLExtSetOpPatterns<math::CoshOp>(converter, patterns, benefit,
                                            "cosh");
  populateOCLExtSetOpPatterns<math::ErfOp>(converter, patterns, benefit, "erf");
  populateOCLExtSetOpPatterns<math::ErfcOp>(converter, patterns, benefit,
                                            "erfc");
  populateOCLExtSetOpPatterns<math::ExpOp>(converter, patterns, benefit, "exp");
  populateOCLExtSetOpPatterns<math::Exp2Op>(converter, patterns, benefit,
                                            "exp2");
  populateOCLExtSetOpPatterns<math::ExpM1Op>(converter, patterns, benefit,
                                             "expm1");
  populateOCLExtSetOpPatterns<math::LogOp>(converter, patterns, benefit, "log");
  populateOCLExtSetOpPatterns<math::Log10Op>(converter, patterns, benefit,
                                             "log10");
  populateOCLExtSetOpPatterns<math::Log1pOp>(converter, patterns, benefit,
                                             "log1p");
  populateOCLExtSetOpPatterns<math::Log2Op>(converter, patterns, benefit,
                                            "log2");
  populateOCLExtSetOpPatterns<math::PowFOp>(converter, patterns, benefit,
                                            "pow");
  populateOCLExtSetOpPatterns<math::RsqrtOp>(converter, patterns, benefit,
                                             "rsqrt");
  populateOCLExtSetOpPatterns<math::SinOp>(converter, patterns, benefit, "sin");
  populateOCLExtSetOpPatterns<math::SinhOp>(converter, patterns, benefit,
                                            "sinh");
  populateOCLExtSetOpPatterns<math::SqrtOp>(converter, patterns, benefit,
                                            "sqrt");
  populateOCLExtSetOpPatterns<math::TanOp>(converter, patterns, benefit, "tan");
  populateOCLExtSetOpPatterns<math::TanhOp>(converter, patterns, benefit,
                                            "tanh");
}

namespace {
struct ConvertMathToLLVMSPVPass final
    : impl::ConvertMathToLLVMSPVBase<ConvertMathToLLVMSPVPass> {
  using impl::ConvertMathToLLVMSPVBase<
      ConvertMathToLLVMSPVPass>::ConvertMathToLLVMSPVBase;

  void runOnOperation() override;
};
} // namespace

void ConvertMathToLLVMSPVPass::runOnOperation() {
  auto m = getOperation();
  MLIRContext *ctx = m.getContext();

  if (!isExtensionSetSupported(extensionSetName)) {
    m.emitError() << "Unsupported extension set '" << extensionSetName << "'!";
    return signalPassFailure();
  }

  RewritePatternSet patterns(&getContext());
  LowerToLLVMOptions options(ctx, DataLayout(m));
  LLVMTypeConverter converter(ctx, options);
  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, func::FuncDialect,
                         vector::VectorDialect, LLVM::LLVMDialect>();
  if (extensionSetName == "OpenCL.std") {
    populateMathToOCLExtSetLLVMSPVConversionPatterns(converter, patterns,
                                                     /*benefit=*/1);
    target
        .addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::LogOp,
                      LLVM::Log10Op, LLVM::Log2Op, LLVM::SinOp, LLVM::SqrtOp>();
  }
  if (failed(applyPartialConversion(m, target, std::move(patterns))))
    signalPassFailure();
}
