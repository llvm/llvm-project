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
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTCOMPLEXTOROCDL
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct FloatTypeResolver {
  std::optional<bool> operator()(Type type) const {
    auto elementType = cast<FloatType>(type);
    if (!isa<Float32Type, Float64Type>(elementType))
      return {};
    return elementType.getIntOrFloatBitWidth() == 64;
  }
};

template <typename Op, typename TypeResolver = FloatTypeResolver>
struct ScalarOpToROCDLCall : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  ScalarOpToROCDLCall(MLIRContext *context, StringRef floatFunc,
                      StringRef doubleFunc, PatternBenefit benefit)
      : OpRewritePattern<Op>(context, benefit), floatFunc(floatFunc),
        doubleFunc(doubleFunc) {}

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
    auto module = SymbolTable::getNearestSymbolTable(op);
    auto isDouble = TypeResolver()(op.getType());
    if (!isDouble.has_value())
      return failure();

    auto name = *isDouble ? doubleFunc : floatFunc;

    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, name));
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());
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

void mlir::populateComplexToROCDLConversionPatterns(RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  patterns.add<ScalarOpToROCDLCall<complex::AbsOp>>(
      patterns.getContext(), "__ocml_cabs_f32", "__ocml_cabs_f64", benefit);
}

namespace {
struct ConvertComplexToROCDLPass
    : public impl::ConvertComplexToROCDLBase<ConvertComplexToROCDLPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertComplexToROCDLPass::runOnOperation() {
  auto module = getOperation();

  RewritePatternSet patterns(&getContext());
  populateComplexToROCDLConversionPatterns(patterns, /*benefit=*/1);

  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addIllegalOp<complex::AbsOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
