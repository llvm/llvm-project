//===- ArithToAPFloat.cpp - Arithmetic to APFloat impl conversion ---------===//
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
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOAPFLOATCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::func;

#define APFLOAT_BIN_OPS(X)                                                     \
  X(add)                                                                       \
  X(subtract)                                                                  \
  X(multiply)                                                                  \
  X(divide)                                                                    \
  X(remainder)                                                                 \
  X(mod)

#define APFLOAT_EXTERN_K(OP) kApFloat_##OP

#define APFLOAT_EXTERN_NAME(OP)                                                \
  static constexpr llvm::StringRef APFLOAT_EXTERN_K(OP) = "_mlir_"             \
                                                          "apfloat_" #OP;

namespace mlir::func {
#define LOOKUP_OR_CREATE_APFLOAT_FN_DECL(OP)                                   \
  FailureOr<FuncOp> lookupOrCreateApFloat##OP##Fn(                             \
      OpBuilder &b, Operation *moduleOp,                                       \
      SymbolTableCollection *symbolTables = nullptr);

APFLOAT_BIN_OPS(LOOKUP_OR_CREATE_APFLOAT_FN_DECL)

#undef LOOKUP_OR_CREATE_APFLOAT_FN_DECL

APFLOAT_BIN_OPS(APFLOAT_EXTERN_NAME)

#define LOOKUP_OR_CREATE_APFLOAT_FN_DEFN(OP)                                   \
  FailureOr<FuncOp> lookupOrCreateApFloat##OP##Fn(                             \
      OpBuilder &b, Operation *moduleOp,                                       \
      SymbolTableCollection *symbolTables) {                                   \
    return lookupOrCreateFn(b, moduleOp, APFLOAT_EXTERN_K(OP),                 \
                            {IntegerType::get(moduleOp->getContext(), 32),     \
                             IntegerType::get(moduleOp->getContext(), 64),     \
                             IntegerType::get(moduleOp->getContext(), 64)},    \
                            {IntegerType::get(moduleOp->getContext(), 64)},    \
                            /*setPrivate*/ true, symbolTables);                \
  }

APFLOAT_BIN_OPS(LOOKUP_OR_CREATE_APFLOAT_FN_DEFN)
#undef LOOKUP_OR_CREATE_APFLOAT_FN_DEFN
} // namespace mlir::func

struct FancyAddFLowering : OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {
    // Get APFloat adder function from runtime library.
    auto parent = op->getParentOfType<ModuleOp>();
    if (!parent)
      return failure();
    if (!llvm::isa<Float8E5M2Type, Float8E4M3Type, Float8E4M3FNType,
                   Float8E5M2FNUZType, Float8E4M3FNUZType,
                   Float8E4M3B11FNUZType, Float8E3M4Type, Float4E2M1FNType,
                   Float6E2M3FNType, Float6E3M2FNType, Float8E8M0FNUType>(
            op.getType()))
      return failure();
    FailureOr<Operation *> adder = lookupOrCreateApFloataddFn(rewriter, parent);

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

    // Call software implementation of floating point addition.
    int32_t sem =
        llvm::APFloatBase::SemanticsToEnum(floatTy.getFloatSemantics());
    Value semValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), sem));
    SmallVector<Value> params = {semValue, lhsBits, rhsBits};
    auto resultOp =
        func::CallOp::create(rewriter, loc, TypeRange(rewriter.getI64Type()),
                             SymbolRefAttr::get(*adder), params);

    // Truncate result to the original width.
    Value truncatedBits = arith::TruncIOp::create(rewriter, loc, intWType,
                                                  resultOp->getResult(0));
    rewriter.replaceAllUsesWith(
        op, arith::BitcastOp::create(rewriter, loc, floatTy, truncatedBits));
    return success();
  }
};

void arith::populateArithToAPFloatConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FancyAddFLowering>(patterns.getContext());
}

namespace {
struct ArithToAPFloatConversionPass final
    : impl::ArithToAPFloatConversionPassBase<ArithToAPFloatConversionPass> {
  using impl::ArithToAPFloatConversionPassBase<
      ArithToAPFloatConversionPass>::ArithToAPFloatConversionPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    arith::populateArithToAPFloatConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
