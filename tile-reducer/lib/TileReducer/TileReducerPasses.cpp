//===- TileReducerPasses.cpp - TileReducer rewrite passes -------*- C++ -*-===//

#include "TileReducer/TileReducerPasses.h"

#include "TileReducer/TileReducerAnalyses.h"
#include "TileReducer/TileReducerDialect.h"
#include "TileReducer/TileReducerOps.h"
#include "TileReducer/TileReducerTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tr {
#define GEN_PASS_DEF_FOLDTRADDZERO
#define GEN_PASS_DEF_RECOGNIZELOADREDUCE
#define GEN_PASS_DEF_ANNOTATEREDUCTIONPLAN
#define GEN_PASS_DEF_CONVERTTRFORBOUNDSTOARITH
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

bool isZeroSplat(Value v) {
  auto cst = v.getDefiningOp<ConstantOp>();
  if (!cst)
    return false;
  Attribute attr = cst.getValue();
  if (auto f = dyn_cast<FloatAttr>(attr))
    return f.getValue().isZero();
  if (auto i = dyn_cast<IntegerAttr>(attr))
    return i.getValue().isZero();
  return false;
}

struct FoldAddZeroPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (isZeroSplat(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if (isZeroSplat(op.getLhs())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

struct RecognizeLoadReducePattern : public OpRewritePattern<ReduceSumOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceSumOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("tr.load_reduce"))
      return failure();
    auto load = op.getInput().getDefiningOp<LoadOp>();
    if (!load)
      return rewriter.notifyMatchFailure(op, "input is not a tr.load");
    if (!load->hasOneUse())
      return rewriter.notifyMatchFailure(op, "load has other users");
    auto inTy = dyn_cast<TileType>(load.getType());
    auto outTy = dyn_cast<TileType>(op.getType());
    if (!inTy || !outTy || inTy.getElementType() != outTy.getElementType())
      return rewriter.notifyMatchFailure(op, "element type mismatch");
    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr("tr.load_reduce", rewriter.getUnitAttr());
    });
    return success();
  }
};

struct FoldTRAddZero : impl::FoldTRAddZeroBase<FoldTRAddZero> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldAddZeroPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct RecognizeLoadReduce : impl::RecognizeLoadReduceBase<RecognizeLoadReduce> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RecognizeLoadReducePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct AnnotateReductionPlan
    : impl::AnnotateReductionPlanBase<AnnotateReductionPlan> {
  void runOnOperation() override {
    auto &reds = getAnalysis<ReductionAnalysis>();
    (void)getAnalysis<BoundsAnalysis>();
    (void)getAnalysis<LayoutAnalysis>();
    getOperation().walk([&](ReduceSumOp op) {
      const ReductionInfo *info = reds.get(op);
      if (!info)
        return;
      StringRef plan = "unknown";
      switch (info->kind) {
      case ReductionKind::Row:
        plan = "row";
        break;
      case ReductionKind::Column:
        plan = "column";
        break;
      case ReductionKind::Full:
        plan = "full";
        break;
      case ReductionKind::Unknown:
        break;
      }
      op->setAttr("tr.plan", StringAttr::get(op.getContext(), plan));
    });
  }
};

struct ConvertTRForBoundsToArith
    : impl::ConvertTRForBoundsToArithBase<ConvertTRForBoundsToArith> {
  void runOnOperation() override {
    getOperation().walk([&](ForOp op) {
      OpBuilder b(op);
      Location loc = op.getLoc();
      if (!op.getLowerBound()) {
        if (auto lb = op.getConstantLowerBound()) {
          Value v = arith::ConstantIndexOp::create(b, loc, *lb);
          op.getLowerBoundMutable().assign(v);
          op.removeStaticLowerBoundAttr();
        }
      }
      if (!op.getStep()) {
        if (auto st = op.getConstantStep()) {
          Value v = arith::ConstantIndexOp::create(b, loc, *st);
          op.getStepMutable().assign(v);
          op.removeStaticStepAttr();
        }
      }
    });
  }
};

} // namespace

void populateFoldAddZeroPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldAddZeroPattern>(patterns.getContext());
}
void populateRecognizeLoadReducePatterns(RewritePatternSet &patterns) {
  patterns.add<RecognizeLoadReducePattern>(patterns.getContext());
}

} // namespace mlir::tr
