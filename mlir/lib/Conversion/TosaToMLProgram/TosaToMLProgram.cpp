//===- TosaToMLProgram.cpp - Lowering Tosa to MLProgram Dialect------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the TOSA dialect to the MLProgram dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToMLProgram/TosaToMLProgram.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace tosa;
namespace {

class VariableOpConverter : public OpRewritePattern<tosa::VariableOp> {
public:
  using OpRewritePattern<tosa::VariableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::VariableOp op,
                                PatternRewriter &rewriter) const final {
    auto newVariable = rewriter.create<mlir::ml_program::GlobalOp>(
        op.getLoc(), op.getName(), op.getType(), /*is_mutable=*/true,
        op.getInitialValueAttr(), /*sym_visibility=*/nullptr);
    newVariable.setPrivate();
    rewriter.replaceOp(op, newVariable);
    return success();
  }
};

class VariableWriteOpConverter
    : public OpRewritePattern<tosa::VariableWriteOp> {
public:
  using OpRewritePattern<tosa::VariableWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::VariableWriteOp op,
                                PatternRewriter &rewriter) const final {
    auto globalSymbolRef =
        SymbolRefAttr::get(rewriter.getContext(), op.getName());
    auto newVariableWrite = rewriter.create<ml_program::GlobalStoreOp>(
        op.getLoc(), globalSymbolRef, op.getValue());
    rewriter.replaceOp(op, newVariableWrite);
    return success();
  }
};

class VariableReadOpConverter : public OpRewritePattern<tosa::VariableReadOp> {
public:
  using OpRewritePattern<tosa::VariableReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::VariableReadOp op,
                                PatternRewriter &rewriter) const final {
    auto globalSymbolRef =
        SymbolRefAttr::get(rewriter.getContext(), op.getName());
    auto newVariableRead = rewriter.create<ml_program::GlobalLoadOp>(
        op.getLoc(), op.getType(), globalSymbolRef);
    rewriter.replaceOp(op, newVariableRead);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToMLProgramConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<VariableOpConverter, VariableWriteOpConverter,
                VariableReadOpConverter>(patterns->getContext());
}
