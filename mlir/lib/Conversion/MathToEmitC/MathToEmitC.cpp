//===- MathToEmitC.cpp - Math to EmitC Patterns -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToEmitC/MathToEmitC.h"

#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Dialect/Arith/IR/ArithOpInterfaces.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
/// Implement the interface to convert Math to EmitC.
struct MathToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  MathToEmitCDialectInterface(Dialect *dialect)
      : ConvertToEmitCPatternInterface(dialect) {}

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToEmitCConversionPatterns(
      ConversionTarget &target, TypeConverter &typeConverter,
      RewritePatternSet &patterns, std::optional<bool> lowerToCpp) const final {
    populateConvertMathToEmitCPatterns(
        patterns, lowerToCpp.value_or(true) ? emitc::LanguageTarget::cpp11
                                            : emitc::LanguageTarget::c99);
  }
};

template <typename OpType>
class LowerToEmitCCallOpaque : public OpRewritePattern<OpType> {
  std::string calleeStr;
  emitc::LanguageTarget languageTarget;
  bool typeSuffix;

public:
  LowerToEmitCCallOpaque(MLIRContext *context, std::string calleeStr,
                         emitc::LanguageTarget languageTarget,
                         bool typeSuffix = true)
      : OpRewritePattern<OpType>(context), calleeStr(std::move(calleeStr)),
        languageTarget(languageTarget), typeSuffix(typeSuffix) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override;
};

template <typename OpType>
LogicalResult LowerToEmitCCallOpaque<OpType>::matchAndRewrite(
    OpType op, PatternRewriter &rewriter) const {
  if (!llvm::all_of(op->getOperandTypes(),
                    llvm::IsaPred<Float32Type, Float64Type>) ||
      !llvm::all_of(op->getResultTypes(), [](Type type) {
        return isa<Float32Type, Float64Type>(type) || type.isInteger(1);
      }))
    return rewriter.notifyMatchFailure(
        op.getLoc(), "expected all operands to be of type f32 or f64, and "
                     "all results to be of type f32, f64 or i1");
  if (auto roundingModeOp =
          dyn_cast<arith::ArithRoundingModeInterface>(op.getOperation());
      roundingModeOp && roundingModeOp.getRoundingModeAttr())
    return rewriter.notifyMatchFailure(op.getLoc(),
                                       "static rounding mode not supported");
  std::string modifiedCalleeStr = calleeStr;
  if (languageTarget == emitc::LanguageTarget::cpp11) {
    modifiedCalleeStr = "std::" + calleeStr;
  } else if (languageTarget == emitc::LanguageTarget::c99 && typeSuffix) {
    auto operandType = op->getOperandTypes()[0];
    if (operandType.isF32())
      modifiedCalleeStr = calleeStr + "f";
  }
  rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
      op, op.getType(), modifiedCalleeStr, op->getOperands());
  return success();
}

} // namespace

void mlir::registerConvertMathToEmitCInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, math::MathDialect *dialect) {
    dialect->addInterfaces<MathToEmitCDialectInterface>();
  });
}

// Populates patterns to replace `math` operations with `emitc.call_opaque`,
// using function names consistent with those in <math.h> or a builtin.
void mlir::populateConvertMathToEmitCPatterns(
    RewritePatternSet &patterns, emitc::LanguageTarget languageTarget) {
  auto *context = patterns.getContext();
  patterns.insert<LowerToEmitCCallOpaque<math::FloorOp>>(context, "floor",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::RoundOp>>(context, "round",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::RoundEvenOp>>(
      context, "__builtin_roundeven", emitc::LanguageTarget::c99);
  patterns.insert<LowerToEmitCCallOpaque<math::ExpOp>>(context, "exp",
                                                       languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::CosOp>>(context, "cos",
                                                       languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::SinOp>>(context, "sin",
                                                       languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::AcosOp>>(context, "acos",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::AsinOp>>(context, "asin",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::Atan2Op>>(context, "atan2",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::CeilOp>>(context, "ceil",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::AbsFOp>>(context, "fabs",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::PowFOp>>(context, "pow",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::SqrtOp>>(context, "sqrt",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::TanOp>>(context, "tan",
                                                       languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::AtanOp>>(context, "atan",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::SinhOp>>(context, "sinh",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::CoshOp>>(context, "cosh",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::TanhOp>>(context, "tanh",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::AsinhOp>>(context, "asinh",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::AcoshOp>>(context, "acosh",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::AtanhOp>>(context, "atanh",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::Exp2Op>>(context, "exp2",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::ExpM1Op>>(context, "expm1",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::LogOp>>(context, "log",
                                                       languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::Log2Op>>(context, "log2",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::Log10Op>>(context, "log10",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::Log1pOp>>(context, "log1p",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::CbrtOp>>(context, "cbrt",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::ErfOp>>(context, "erf",
                                                       languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::ErfcOp>>(context, "erfc",
                                                        languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::TruncOp>>(context, "trunc",
                                                         languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::CopySignOp>>(context, "copysign",
                                                            languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::FmaOp>>(context, "fma",
                                                       languageTarget);
  patterns.insert<LowerToEmitCCallOpaque<math::IsNaNOp>>(
      context, "isnan", languageTarget, /*typeSuffix=*/false);
  patterns.insert<LowerToEmitCCallOpaque<math::IsInfOp>>(
      context, "isinf", languageTarget, /*typeSuffix=*/false);
  patterns.insert<LowerToEmitCCallOpaque<math::IsFiniteOp>>(
      context, "isfinite", languageTarget, /*typeSuffix=*/false);
  patterns.insert<LowerToEmitCCallOpaque<math::IsNormalOp>>(
      context, "isnormal", languageTarget, /*typeSuffix=*/false);
}
