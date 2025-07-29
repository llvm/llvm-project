//=== ComplexToROCDLLibraryCalls.cpp - convert from Complex to ROCDL calls ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToROCDLLibraryCalls/ComplexToROCDLLibraryCalls.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTCOMPLEXTOROCDLLIBRARYCALLS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

template <typename Op, typename FloatTy>
// Pattern to convert Complex ops to ROCDL function calls.
struct ComplexOpToROCDLLibraryCalls : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  ComplexOpToROCDLLibraryCalls(MLIRContext *context, StringRef funcName,
                               PatternBenefit benefit = 1)
      : OpRewritePattern<Op>(context, benefit), funcName(funcName) {}

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
    Operation *symTable = SymbolTable::getNearestSymbolTable(op);
    Type resType = op.getType();
    if (auto complexType = dyn_cast<ComplexType>(resType))
      resType = complexType.getElementType();
    if (!isa<FloatTy>(resType))
      return failure();

    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(symTable, funcName));
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&symTable->getRegion(0).front());
      auto funcTy = FunctionType::get(
          rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
      opFunc = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(),
                                    funcName, funcTy);
      opFunc.setPrivate();
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(op, funcName, op.getType(),
                                              op->getOperands());
    return success();
  }

private:
  std::string funcName;
};
} // namespace

void mlir::populateComplexToROCDLLibraryCallsConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::AbsOp, Float32Type>>(
      patterns.getContext(), "__ocml_cabs_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::AbsOp, Float64Type>>(
      patterns.getContext(), "__ocml_cabs_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::AngleOp, Float32Type>>(
      patterns.getContext(), "__ocml_carg_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::AngleOp, Float64Type>>(
      patterns.getContext(), "__ocml_carg_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::ConjOp, Float32Type>>(
      patterns.getContext(), "__ocml_conj_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::ConjOp, Float64Type>>(
      patterns.getContext(), "__ocml_conj_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::CosOp, Float32Type>>(
      patterns.getContext(), "__ocml_ccos_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::CosOp, Float64Type>>(
      patterns.getContext(), "__ocml_ccos_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::ExpOp, Float32Type>>(
      patterns.getContext(), "__ocml_cexp_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::ExpOp, Float64Type>>(
      patterns.getContext(), "__ocml_cexp_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::LogOp, Float32Type>>(
      patterns.getContext(), "__ocml_clog_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::LogOp, Float64Type>>(
      patterns.getContext(), "__ocml_clog_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::PowOp, Float32Type>>(
      patterns.getContext(), "__ocml_cpow_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::PowOp, Float64Type>>(
      patterns.getContext(), "__ocml_cpow_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::SinOp, Float32Type>>(
      patterns.getContext(), "__ocml_csin_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::SinOp, Float64Type>>(
      patterns.getContext(), "__ocml_csin_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::SqrtOp, Float32Type>>(
      patterns.getContext(), "__ocml_csqrt_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::SqrtOp, Float64Type>>(
      patterns.getContext(), "__ocml_csqrt_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::TanOp, Float32Type>>(
      patterns.getContext(), "__ocml_ctan_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::TanOp, Float64Type>>(
      patterns.getContext(), "__ocml_ctan_f64");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::TanhOp, Float32Type>>(
      patterns.getContext(), "__ocml_ctanh_f32");
  patterns.add<ComplexOpToROCDLLibraryCalls<complex::TanhOp, Float64Type>>(
      patterns.getContext(), "__ocml_ctanh_f64");
}

namespace {
struct ConvertComplexToROCDLLibraryCallsPass
    : public impl::ConvertComplexToROCDLLibraryCallsBase<
          ConvertComplexToROCDLLibraryCallsPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertComplexToROCDLLibraryCallsPass::runOnOperation() {
  Operation *op = getOperation();

  RewritePatternSet patterns(&getContext());
  populateComplexToROCDLLibraryCallsConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addIllegalOp<complex::AbsOp, complex::AngleOp, complex::ConjOp,
                      complex::CosOp, complex::ExpOp, complex::LogOp,
                      complex::PowOp, complex::SinOp, complex::SqrtOp,
                      complex::TanOp, complex::TanhOp>();
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}
