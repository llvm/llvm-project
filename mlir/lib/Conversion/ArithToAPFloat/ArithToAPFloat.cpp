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

namespace mlir {
#define GEN_PASS_DEF_ARITHTOAPFLOATCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::func;

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
static FailureOr<Operation *>
lookupOrCreateBinaryFn(OpBuilder &b, Operation *moduleOp, StringRef name,
                       SymbolTableCollection *symbolTables = nullptr) {
  auto i32Type = IntegerType::get(moduleOp->getContext(), 32);
  auto i64Type = IntegerType::get(moduleOp->getContext(), 64);
  return lookupOrCreateFnDecl(b, moduleOp,
                              (llvm::Twine("_mlir_apfloat_") + name).str(),
                              {i32Type, i64Type, i64Type}, {i64Type},
                              /*setPrivate=*/true, symbolTables);
}

/// Rewrite a binary arithmetic operation to an APFloat function call.
template <typename OpTy, const char *APFloatName>
struct ArithOpToAPFloatConversion final : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto moduleOp = op->template getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return rewriter.notifyMatchFailure(
          op, "arith op must be contained within a builtin.module");
    }
    // Get APFloat function from runtime library.
    FailureOr<Operation *> fn =
        lookupOrCreateBinaryFn(rewriter, moduleOp, APFloatName);
    if (failed(fn))
      return op->emitError("failed to lookup or create APFloat function");

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
    patterns.add<ArithOpToAPFloatConversion<arith::AddFOp, add>,
                 ArithOpToAPFloatConversion<arith::SubFOp, subtract>,
                 ArithOpToAPFloatConversion<arith::MulFOp, multiply>,
                 ArithOpToAPFloatConversion<arith::DivFOp, divide>,
                 ArithOpToAPFloatConversion<arith::RemFOp, remainder>>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace
