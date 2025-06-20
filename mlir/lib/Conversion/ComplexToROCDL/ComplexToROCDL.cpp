//===-- ComplexToROCDL.cpp - conversion from Complex to ROCDL calls -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToROCDL/ComplexToROCDL.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTCOMPLEXTOROCDL
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

template <typename Op>
// Pattern to convert Complex ops to ROCDL function calls.
struct ComplexOpToROCDLCall : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  ComplexOpToROCDLCall(MLIRContext *context, StringRef floatFunc,
                       StringRef doubleFunc, PatternBenefit benefit = 1)
      : OpRewritePattern<Op>(context, benefit), floatFunc(floatFunc),
        doubleFunc(doubleFunc) {}

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
    Operation *symTable = SymbolTable::getNearestSymbolTable(op);
    Type resType = op.getType();
    if (auto complexType = dyn_cast<ComplexType>(resType))
      resType = complexType.getElementType();
    FloatType floatTy = dyn_cast<FloatType>(resType);
    if (!floatTy)
      return failure();

    StringRef name;
    if (floatTy.isF64())
      name = doubleFunc;
    else if (floatTy.isF32())
      name = floatFunc;
    else
      return failure();

    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(symTable, name));
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&symTable->getRegion(0).front());
      auto funcTy = FunctionType::get(
          rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
      opFunc =
          rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), name, funcTy);
      opFunc.setPrivate();
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(op, name, op.getType(),
                                              op->getOperands());
    return success();
  }

private:
  std::string floatFunc, doubleFunc;
};
} // namespace

void mlir::populateComplexToROCDLConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ComplexOpToROCDLCall<complex::AbsOp>>(
      patterns.getContext(), "__ocml_cabs_f32", "__ocml_cabs_f64");
  patterns.add<ComplexOpToROCDLCall<complex::ExpOp>>(
      patterns.getContext(), "__ocml_cexp_f32", "__ocml_cexp_f64");
}

namespace {
struct ConvertComplexToROCDLPass
    : public impl::ConvertComplexToROCDLBase<ConvertComplexToROCDLPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertComplexToROCDLPass::runOnOperation() {
  Operation *op = getOperation();

  RewritePatternSet patterns(&getContext());
  populateComplexToROCDLConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addIllegalOp<complex::AbsOp, complex::ExpOp>();
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}
