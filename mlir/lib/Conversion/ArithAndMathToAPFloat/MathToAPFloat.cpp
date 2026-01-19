//===- MathToAPFloat.cpp - Mathmetic to APFloat Conversion ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "mlir/Conversion/ArithAndMathToAPFloat/MathToAPFloat.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MATHTOAPFLOATCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::func;

struct AbsFOpToAPFloatConversion final : OpRewritePattern<math::AbsFOp> {
  AbsFOpToAPFloatConversion(MLIRContext *context, SymbolOpInterface symTable,
                            PatternBenefit benefit = 1)
      : OpRewritePattern<math::AbsFOp>(context, benefit), symTable(symTable) {}

  LogicalResult matchAndRewrite(math::AbsFOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(rewriter, op)))
      return failure();
    // Get APFloat function from runtime library.
    auto i32Type = IntegerType::get(symTable->getContext(), 32);
    auto i64Type = IntegerType::get(symTable->getContext(), 64);
    FailureOr<FuncOp> fn = lookupOrCreateFnDecl(
        rewriter, symTable, "_mlir_apfloat_abs", {i32Type, i64Type});
    if (failed(fn))
      return fn;
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    // Scalarize and convert to APFloat runtime calls.
    Value repl = forEachScalarValue(
        rewriter, loc, op.getOperand(), /*operand2=*/Value(), op.getType(),
        [&](Value operand, Value, Type resultType) {
          auto floatTy = cast<FloatType>(operand.getType());
          auto intWType = rewriter.getIntegerType(floatTy.getWidth());
          Value operandBits = arith::ExtUIOp::create(
              rewriter, loc, i64Type,
              arith::BitcastOp::create(rewriter, loc, intWType, operand));
          // Call APFloat function.
          Value semValue = getAPFloatSemanticsValue(rewriter, loc, floatTy);
          SmallVector<Value> params = {semValue, operandBits};
          Value negatedBits =
              func::CallOp::create(rewriter, loc, TypeRange(i64Type),
                                   SymbolRefAttr::get(*fn), params)
                  ->getResult(0);
          // Truncate result to the original width.
          auto truncatedBits =
              arith::TruncIOp::create(rewriter, loc, intWType, negatedBits);
          return arith::BitcastOp::create(rewriter, loc, floatTy,
                                          truncatedBits);
        });

    rewriter.replaceOp(op, repl);
    return success();
  }

  SymbolOpInterface symTable;
};

template <typename OpTy>
struct IsOpToAPFloatConversion final : OpRewritePattern<OpTy> {
  IsOpToAPFloatConversion(MLIRContext *context, const char *APFloatName,
                          SymbolOpInterface symTable,
                          PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit), symTable(symTable),
        APFloatName(APFloatName) {};

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(rewriter, op)))
      return failure();
    // Get APFloat function from runtime library.
    auto i1 = IntegerType::get(symTable->getContext(), 1);
    auto i32Type = IntegerType::get(symTable->getContext(), 32);
    auto i64Type = IntegerType::get(symTable->getContext(), 64);
    std::string funcName =
        (llvm::Twine("_mlir_apfloat_is") + APFloatName).str();
    FailureOr<FuncOp> fn = lookupOrCreateFnDecl(
        rewriter, symTable, funcName, {i32Type, i64Type}, nullptr, i1);
    if (failed(fn))
      return fn;
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    // Scalarize and convert to APFloat runtime calls.
    Value repl = forEachScalarValue(
        rewriter, loc, op.getOperand(), /*operand2=*/Value(), op.getType(),
        [&](Value operand, Value, Type resultType) {
          auto floatTy = cast<FloatType>(operand.getType());
          auto intWType = rewriter.getIntegerType(floatTy.getWidth());
          Value operandBits = arith::ExtUIOp::create(
              rewriter, loc, i64Type,
              arith::BitcastOp::create(rewriter, loc, intWType, operand));

          // Call APFloat function.
          Value semValue = getAPFloatSemanticsValue(rewriter, loc, floatTy);
          Value params[] = {semValue, operandBits};
          return func::CallOp::create(rewriter, loc, TypeRange(i1),
                                      SymbolRefAttr::get(*fn), params)
              .getResult(0);
        });
    rewriter.replaceOp(op, repl);
    return success();
  }

  SymbolOpInterface symTable;
  const char *APFloatName;
};

struct FmaOpToAPFloatConversion final : OpRewritePattern<math::FmaOp> {
  FmaOpToAPFloatConversion(MLIRContext *context, SymbolOpInterface symTable,
                           PatternBenefit benefit = 1)
      : OpRewritePattern<math::FmaOp>(context, benefit), symTable(symTable) {};

  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(rewriter, op)))
      return failure();
    // Cast operands to 64-bit integers.
    mlir::Type resType = op.getResult().getType();
    auto floatTy = dyn_cast<FloatType>(resType);
    if (!floatTy) {
      auto vecTy1 = cast<VectorType>(resType);
      floatTy = llvm::cast<FloatType>(vecTy1.getElementType());
    }
    auto i32Type = IntegerType::get(symTable->getContext(), 32);
    auto i64Type = IntegerType::get(symTable->getContext(), 64);
    FailureOr<FuncOp> fn = lookupOrCreateFnDecl(
        rewriter, symTable, "_mlir_apfloat_fused_multiply_add",
        {i32Type, i64Type, i64Type, i64Type});
    if (failed(fn))
      return fn;
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);

    IntegerType intWType = rewriter.getIntegerType(floatTy.getWidth());
    IntegerType int64Type = rewriter.getI64Type();

    auto scalarFMA = [&rewriter, &loc, &floatTy, &fn, &intWType,
                      &int64Type](Value a, Value b, Value c) {
      Value operand = arith::ExtUIOp::create(
          rewriter, loc, int64Type,
          arith::BitcastOp::create(rewriter, loc, intWType, a));
      Value multiplicand = arith::ExtUIOp::create(
          rewriter, loc, int64Type,
          arith::BitcastOp::create(rewriter, loc, intWType, b));
      Value addend = arith::ExtUIOp::create(
          rewriter, loc, int64Type,
          arith::BitcastOp::create(rewriter, loc, intWType, c));
      // Call APFloat function.
      Value semValue = getAPFloatSemanticsValue(rewriter, loc, floatTy);
      SmallVector<Value> params = {semValue, operand, multiplicand, addend};
      auto resultOp =
          func::CallOp::create(rewriter, loc, TypeRange(rewriter.getI64Type()),
                               SymbolRefAttr::get(*fn), params);

      // Truncate result to the original width.
      auto trunc = arith::TruncIOp::create(rewriter, loc, intWType,
                                           resultOp->getResult(0));
      return arith::BitcastOp::create(rewriter, loc, floatTy, trunc);
    };

    if (auto vecTy1 = dyn_cast<VectorType>(op.getA().getType())) {
      // Sanity check: Operand types must match.
      assert(vecTy1 == dyn_cast<VectorType>(op.getB().getType()) &&
             "expected same vector types");
      assert(vecTy1 == dyn_cast<VectorType>(op.getC().getType()) &&
             "expected same vector types");
      // Prepare scalar operands.
      ResultRange scalarOperands =
          vector::ToElementsOp::create(rewriter, loc, op.getA())->getResults();
      ResultRange scalarMultiplicands =
          vector::ToElementsOp::create(rewriter, loc, op.getB())->getResults();
      ResultRange scalarAddends =
          vector::ToElementsOp::create(rewriter, loc, op.getC())->getResults();
      // Call the function for each pair of scalar operands.
      SmallVector<Value> results;
      for (auto [operand, multiplicand, addend] : llvm::zip_equal(
               scalarOperands, scalarMultiplicands, scalarAddends)) {
        results.push_back(scalarFMA(operand, multiplicand, addend));
      }
      // Package the results into a vector.
      auto fromElements = vector::FromElementsOp::create(
          rewriter, loc,
          vecTy1.cloneWith(/*shape=*/std::nullopt, results.front().getType()),
          results);
      rewriter.replaceOp(op, fromElements);
      return success();
    }

    Value repl = scalarFMA(op.getA(), op.getB(), op.getC());
    rewriter.replaceOp(op, repl);
    return success();
  }

  SymbolOpInterface symTable;
};

namespace {
struct MathToAPFloatConversionPass final
    : impl::MathToAPFloatConversionPassBase<MathToAPFloatConversionPass> {
  using Base::Base;

  void runOnOperation() override;
};

void MathToAPFloatConversionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);

  patterns.add<AbsFOpToAPFloatConversion>(context, getOperation());
  patterns.add<IsOpToAPFloatConversion<math::IsFiniteOp>>(context, "finite",
                                                          getOperation());
  patterns.add<IsOpToAPFloatConversion<math::IsInfOp>>(context, "infinite",
                                                       getOperation());
  patterns.add<IsOpToAPFloatConversion<math::IsNaNOp>>(context, "nan",
                                                       getOperation());
  patterns.add<IsOpToAPFloatConversion<math::IsNormalOp>>(context, "normal",
                                                          getOperation());
  patterns.add<FmaOpToAPFloatConversion>(context, getOperation());

  LogicalResult result = success();
  ScopedDiagnosticHandler scopedHandler(context, [&result](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) {
      result = failure();
    }
    // NB: if you don't return failure, no other diag handlers will fire (see
    // mlir/lib/IR/Diagnostics.cpp:DiagnosticEngineImpl::emit).
    return failure();
  });
  walkAndApplyPatterns(getOperation(), std::move(patterns));
  if (failed(result))
    return signalPassFailure();
}
} // namespace
