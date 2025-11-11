//===- ArithToAPFloat.cpp - Arithmetic to APFloat Conversion --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToAPFloat/ArithToAPFloat.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arith-to-apfloat"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOAPFLOATCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::func;

static FuncOp createFnDecl(OpBuilder &b, SymbolOpInterface symTable,
                           StringRef name, FunctionType funcT, bool setPrivate,
                           SymbolTableCollection *symbolTables = nullptr) {
  OpBuilder::InsertionGuard g(b);
  assert(!symTable->getRegion(0).empty() && "expected non-empty region");
  b.setInsertionPointToStart(&symTable->getRegion(0).front());
  FuncOp funcOp = FuncOp::create(b, symTable->getLoc(), name, funcT);
  if (setPrivate)
    funcOp.setPrivate();
  if (symbolTables) {
    SymbolTable &symbolTable = symbolTables->getSymbolTable(symTable);
    symbolTable.insert(funcOp, symTable->getRegion(0).front().begin());
  }
  return funcOp;
}

/// Helper function to look up or create the symbol for a runtime library
/// function for a binary arithmetic operation.
///
/// Parameter 1: APFloat semantics
/// Parameter 2: Left-hand side operand
/// Parameter 3: Right-hand side operand
///
/// This function will return a failure if the function is found but has an
/// unexpected signature.
///
static FailureOr<FuncOp>
lookupOrCreateBinaryFn(OpBuilder &b, SymbolOpInterface symTable, StringRef name,
                       SymbolTableCollection *symbolTables = nullptr) {
  auto i32Type = IntegerType::get(symTable->getContext(), 32);
  auto i64Type = IntegerType::get(symTable->getContext(), 64);

  std::string funcName = (llvm::Twine("__mlir_apfloat_") + name).str();
  FunctionType funcT =
      FunctionType::get(b.getContext(), {i32Type, i64Type, i64Type}, {i64Type});
  FailureOr<FuncOp> func =
      lookupFnDecl(symTable, funcName, funcT, symbolTables);
  // Failed due to type mismatch.
  if (failed(func))
    return func;
  // Successfully matched existing decl.
  if (*func)
    return *func;

  return createFnDecl(b, symTable, funcName, funcT,
                      /*setPrivate=*/true, symbolTables);
}

/// Rewrite a binary arithmetic operation to an APFloat function call.
template <typename OpTy, const char *APFloatName>
struct BinaryArithOpToAPFloatConversion final : OpRewritePattern<OpTy> {
  BinaryArithOpToAPFloatConversion(MLIRContext *context, PatternBenefit benefit,
                                   SymbolOpInterface symTable)
      : OpRewritePattern<OpTy>(context, benefit), symTable(symTable) {};

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Get APFloat function from runtime library.
    FailureOr<FuncOp> fn =
        lookupOrCreateBinaryFn(rewriter, symTable, APFloatName);
    if (failed(fn))
      return fn;

    rewriter.setInsertionPoint(op);
    // Cast operands to 64-bit integers.
    Location loc = op.getLoc();
    auto floatTy = cast<FloatType>(op.getType());
    auto intWType = rewriter.getIntegerType(floatTy.getWidth());
    auto int64Type = rewriter.getI64Type();
    Value lhsBits = arith::ExtUIOp::create(
        rewriter, loc, int64Type,
        arith::BitcastOp::create(rewriter, loc, intWType, op.getLhs()));
    Value rhsBits = arith::ExtUIOp::create(
        rewriter, loc, int64Type,
        arith::BitcastOp::create(rewriter, loc, intWType, op.getRhs()));

    // Call APFloat function.
    int32_t sem =
        llvm::APFloatBase::SemanticsToEnum(floatTy.getFloatSemantics());
    Value semValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), sem));
    SmallVector<Value> params = {semValue, lhsBits, rhsBits};
    auto resultOp =
        func::CallOp::create(rewriter, loc, TypeRange(rewriter.getI64Type()),
                             SymbolRefAttr::get(*fn), params);

    // Truncate result to the original width.
    Value truncatedBits = arith::TruncIOp::create(rewriter, loc, intWType,
                                                  resultOp->getResult(0));
    rewriter.replaceOp(
        op, arith::BitcastOp::create(rewriter, loc, floatTy, truncatedBits));
    return success();
  }

  SymbolOpInterface symTable;
};

namespace {
struct ArithToAPFloatConversionPass final
    : impl::ArithToAPFloatConversionPassBase<ArithToAPFloatConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    static const char add[] = "add";
    static const char subtract[] = "subtract";
    static const char multiply[] = "multiply";
    static const char divide[] = "divide";
    static const char remainder[] = "remainder";
    patterns.add<BinaryArithOpToAPFloatConversion<arith::AddFOp, add>,
                 BinaryArithOpToAPFloatConversion<arith::SubFOp, subtract>,
                 BinaryArithOpToAPFloatConversion<arith::MulFOp, multiply>,
                 BinaryArithOpToAPFloatConversion<arith::DivFOp, divide>,
                 BinaryArithOpToAPFloatConversion<arith::RemFOp, remainder>>(
        context, 1, getOperation());
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
};
} // namespace
