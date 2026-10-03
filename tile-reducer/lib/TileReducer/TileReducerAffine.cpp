//===- TileReducerAffine.cpp - Milestones 14-15 -----------------*- C++ -*-===//
//
// Raise genuinely affine index arithmetic to affine.apply / affine.for.
// Lower those ops back to scf.for + arith with upstream patterns.
//
//===----------------------------------------------------------------------===//

#include "TileReducer/TileReducerPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tr {
#define GEN_PASS_DEF_CONVERTTRINDEXTOAFFINE
#define GEN_PASS_DEF_LOWERTRAFFINE
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

struct MulIToAffineApply : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isIndex())
      return failure();
    std::optional<int64_t> lhsC = getConstantIntValue(op.getLhs());
    std::optional<int64_t> rhsC = getConstantIntValue(op.getRhs());
    if (lhsC && rhsC)
      return failure();
    if (!lhsC && !rhsC)
      return rewriter.notifyMatchFailure(
          op, "product of two SSA values is not affine");
    Value var = lhsC ? op.getRhs() : op.getLhs();
    int64_t cst = lhsC ? *lhsC : *rhsC;
    AffineExpr expr =
        getAffineDimExpr(0, rewriter.getContext()) * cst;
    auto apply = affine::makeComposedAffineApply(rewriter, op.getLoc(), expr,
                                                 ArrayRef<OpFoldResult>{var});
    rewriter.replaceOp(op, apply.getResult());
    return success();
  }
};

struct AddIToAffineApply : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isIndex())
      return failure();
    if (getConstantIntValue(op.getLhs()) && getConstantIntValue(op.getRhs()))
      return failure();
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr expr = getAffineDimExpr(0, ctx) + getAffineDimExpr(1, ctx);
    auto apply = affine::makeComposedAffineApply(
        rewriter, op.getLoc(), expr,
        ArrayRef<OpFoldResult>{op.getLhs(), op.getRhs()});
    rewriter.replaceOp(op, apply.getResult());
    return success();
  }
};

struct ConvertTRIndexToAffine
    : impl::ConvertTRIndexToAffineBase<ConvertTRIndexToAffine> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MulIToAffineApply, AddIToAffineApply>(&getContext());
    affine::AffineApplyOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct LowerTRAffine : impl::LowerTRAffineBase<LowerTRAffine> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateAffineToStdConversionPatterns(patterns);
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           scf::SCFDialect, func::FuncDialect>();
    target.addIllegalOp<affine::AffineApplyOp, affine::AffineForOp,
                        affine::AffineIfOp, affine::AffineLoadOp,
                        affine::AffineStoreOp, affine::AffineYieldOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tr
