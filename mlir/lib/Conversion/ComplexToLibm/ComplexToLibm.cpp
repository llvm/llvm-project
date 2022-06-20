//===-- ComplexToLibm.cpp - conversion from Complex to libm calls ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToLibm/ComplexToLibm.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {
// Pattern to convert scalar complex operations to calls to libm functions.
// Additionally the libm function signatures are declared.
template <typename Op>
struct ScalarOpToLibmCall : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  ScalarOpToLibmCall<Op>(MLIRContext *context, StringRef floatFunc,
                         StringRef doubleFunc, PatternBenefit benefit)
      : OpRewritePattern<Op>(context, benefit), floatFunc(floatFunc),
        doubleFunc(doubleFunc){};

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;

private:
  std::string floatFunc, doubleFunc;
};
} // namespace

template <typename Op>
LogicalResult
ScalarOpToLibmCall<Op>::matchAndRewrite(Op op,
                                        PatternRewriter &rewriter) const {
  auto module = SymbolTable::getNearestSymbolTable(op);
  auto type = op.getType().template cast<ComplexType>();
  Type elementType = type.getElementType();
  if (!elementType.isa<Float32Type, Float64Type>())
    return failure();

  auto name =
      elementType.getIntOrFloatBitWidth() == 64 ? doubleFunc : floatFunc;
  auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
      SymbolTable::lookupSymbolIn(module, name));
  // Forward declare function if it hasn't already been
  if (!opFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&module->getRegion(0).front());
    auto opFunctionTy = FunctionType::get(
        rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
    opFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), name,
                                           opFunctionTy);
    opFunc.setPrivate();
  }
  assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

  rewriter.replaceOpWithNewOp<func::CallOp>(op, name, type, op->getOperands());

  return success();
}

void mlir::populateComplexToLibmConversionPatterns(RewritePatternSet &patterns,
                                                   PatternBenefit benefit) {
  patterns.add<ScalarOpToLibmCall<complex::PowOp>>(patterns.getContext(),
                                                   "cpowf", "cpow", benefit);
  patterns.add<ScalarOpToLibmCall<complex::SqrtOp>>(patterns.getContext(),
                                                    "csqrtf", "csqrt", benefit);
  patterns.add<ScalarOpToLibmCall<complex::TanhOp>>(patterns.getContext(),
                                                    "ctanhf", "ctanh", benefit);
  patterns.add<ScalarOpToLibmCall<complex::CosOp>>(patterns.getContext(),
                                                   "ccosf", "ccos", benefit);
  patterns.add<ScalarOpToLibmCall<complex::SinOp>>(patterns.getContext(),
                                                   "csinf", "csin", benefit);
  patterns.add<ScalarOpToLibmCall<complex::ConjOp>>(patterns.getContext(),
                                                    "conjf", "conj", benefit);
}

namespace {
struct ConvertComplexToLibmPass
    : public ConvertComplexToLibmBase<ConvertComplexToLibmPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertComplexToLibmPass::runOnOperation() {
  auto module = getOperation();

  RewritePatternSet patterns(&getContext());
  populateComplexToLibmConversionPatterns(patterns, /*benefit=*/1);

  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addIllegalOp<complex::PowOp, complex::SqrtOp, complex::TanhOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertComplexToLibmPass() {
  return std::make_unique<ConvertComplexToLibmPass>();
}
