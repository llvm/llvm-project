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
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOAPFLOATCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::func;

/// Helper function to lookup or create the symbol for a runtime library
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
  return lookupOrCreateFn(b, moduleOp,
                          (llvm::Twine("_mlir_apfloat_") + name).str(),
                          {IntegerType::get(moduleOp->getContext(), 32),
                           IntegerType::get(moduleOp->getContext(), 64),
                           IntegerType::get(moduleOp->getContext(), 64)},
                          {IntegerType::get(moduleOp->getContext(), 64)},
                          /*setPrivate=*/true, symbolTables);
}

/// Rewrite a binary arithmetic operation to an APFloat function call.
template <typename OpTy>
static LogicalResult rewriteBinaryOp(RewriterBase &rewriter, ModuleOp module,
                                     OpTy op, StringRef apfloatName) {
  // Get APFloat function from runtime library.
  FailureOr<Operation *> fn =
      lookupOrCreateBinaryFn(rewriter, module, apfloatName);
  if (failed(fn))
    return op->emitError("failed to lookup or create APFloat function");

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
  int32_t sem = llvm::APFloatBase::SemanticsToEnum(floatTy.getFloatSemantics());
  Value semValue = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32Type(),
      rewriter.getIntegerAttr(rewriter.getI32Type(), sem));
  SmallVector<Value> params = {semValue, lhsBits, rhsBits};
  auto resultOp =
      func::CallOp::create(rewriter, loc, TypeRange(rewriter.getI64Type()),
                           SymbolRefAttr::get(*fn), params);

  // Truncate result to the original width.
  Value truncatedBits =
      arith::TruncIOp::create(rewriter, loc, intWType, resultOp->getResult(0));
  rewriter.replaceOp(
      op, arith::BitcastOp::create(rewriter, loc, floatTy, truncatedBits));
  return success();
}

namespace {
struct ArithToAPFloatConversionPass final
    : impl::ArithToAPFloatConversionPassBase<ArithToAPFloatConversionPass> {
  using impl::ArithToAPFloatConversionPassBase<
      ArithToAPFloatConversionPass>::ArithToAPFloatConversionPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(getOperation()->getContext());
    SmallVector<arith::AddFOp> addOps;
    WalkResult status = module->walk([&](Operation *op) {
      rewriter.setInsertionPoint(op);
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<arith::AddFOp>([&](arith::AddFOp op) {
                return rewriteBinaryOp(rewriter, module, op, "add");
              })
              .Case<arith::SubFOp>([&](arith::SubFOp op) {
                return rewriteBinaryOp(rewriter, module, op, "subtract");
              })
              .Case<arith::MulFOp>([&](arith::MulFOp op) {
                return rewriteBinaryOp(rewriter, module, op, "multiply");
              })
              .Case<arith::DivFOp>([&](arith::DivFOp op) {
                return rewriteBinaryOp(rewriter, module, op, "divide");
              })
              .Case<arith::RemFOp>([&](arith::RemFOp op) {
                return rewriteBinaryOp(rewriter, module, op, "remainder");
              })
              .Default([](Operation *op) { return success(); });
      if (failed(result))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (status.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
