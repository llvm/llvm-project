//===- TestTensorTransforms.cpp - Test Tensor transformation patterns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Tensor transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/TransformUtils.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestTensorTransforms
    : public PassWrapper<TestTensorTransforms, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTensorTransforms)

  TestTensorTransforms() = default;
  TestTensorTransforms(const TestTensorTransforms &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, linalg::LinalgDialect,
                    transform::TransformDialect>();
  }

  StringRef getArgument() const final {
    return "test-tensor-transform-patterns";
  }
  StringRef getDescription() const final {
    return "Test Tensor transformation patterns by applying them greedily.";
  }

  void runOnOperation() override;

  Option<bool> testFoldConstantExtractSlice{
      *this, "test-fold-constant-extract-slice",
      llvm::cl::desc("Test folding arith.constant and tensor.extract_slice"),
      llvm::cl::init(false)};

  Option<bool> testFoldConsecutiveInsertExtractSlice{
      *this, "test-fold-consecutive-insert-extract-slice",
      llvm::cl::desc(
          "Test folding consecutive tensor.insert_slice/tensor.extract_slice"),
      llvm::cl::init(false)};

  Option<bool> testRewriteExtractSliceWithTiledCollapseShape{
      *this, "test-rewrite-extract-slice-from-collapse-shape",
      llvm::cl::desc("Test swapping tensor.extract_slice of a collapse_shape "
                     "with loop nest"),
      llvm::cl::init(false)};

  Option<bool> testDropRedundantInsertSliceRankExpansion{
      *this, "test-drop-redundant-insert-slice-rank-expansion",
      llvm::cl::desc("Test dropping redundant insert_slice rank expansions"),
      llvm::cl::init(false)};

  Option<bool> testReassociativeReshapeFolding{
      *this, "test-reassociative-reshape-folding",
      llvm::cl::desc("Test folding of expand_shape/collapse_shape"),
      llvm::cl::init(false)};

  Option<bool> testFoldIntoPackAndUnpack{
      *this, "test-fold-into-pack-and-unpack",
      llvm::cl::desc("Test folding ops into tensor.pack and tensor.unpack"),
      llvm::cl::init(false)};

  Option<bool> useForeach{
      *this, "use-foreach",
      llvm::cl::desc(
          "Use the scf.forall operation when generating loop nests for "
          "the extract_slice of collapse_shape pattern"),
      llvm::cl::init(false)};

  Option<bool> testSimplifyPackUnpackPatterns{
      *this, "test-simplify-pack-unpack-patterns",
      llvm::cl::desc("Test patterns to simplify tensor.pack and tensor.unpack"),
      llvm::cl::init(false)};

  Option<bool> testTrackingListener{
      *this, "test-tracking-listener",
      llvm::cl::desc("Test tensor TrackingListener for the transform dialect"),
      llvm::cl::init(false)};
};
} // namespace

static void applyReassociativeReshapeFoldingPatterns(Operation *rootOp) {
  RewritePatternSet patterns(rootOp->getContext());
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

static void applyFoldIntoPackAndUnpackPatterns(Operation *rootOp) {
  RewritePatternSet patterns(rootOp->getContext());
  tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

static void applyFoldConstantExtractSlicePatterns(Operation *rootOp) {
  RewritePatternSet patterns(rootOp->getContext());
  tensor::ControlConstantExtractSliceFusionFn controlFn =
      [](tensor::ExtractSliceOp op) {
        if (!op.getSource().hasOneUse())
          return false;

        auto resultType = cast<ShapedType>(op.getResult().getType());
        constexpr int64_t kConstantFoldingMaxNumElements = 1024;
        return resultType.getNumElements() <= kConstantFoldingMaxNumElements;
      };

  tensor::populateFoldConstantExtractSlicePatterns(patterns, controlFn);
  (void)applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

static void applyFoldConsecutiveInsertExtractSlicePatterns(Operation *rootOp) {
  RewritePatternSet patterns(rootOp->getContext());
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  (void)applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

static void
applyDropRedundantInsertSliceRankExpansionPatterns(Operation *rootOp) {
  RewritePatternSet patterns(rootOp->getContext());
  tensor::populateDropRedundantInsertSliceRankExpansionPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

static void applySimplifyPackUnpackPatterns(Operation *rootOp) {
  RewritePatternSet patterns(rootOp->getContext());
  tensor::populateSimplifyPackAndUnpackPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

namespace {
/// Base pattern to rewrite  a `tensor.collapse_shape -> tensor.extract_slice`.
/// The `tensor.extract_slice` is replaced by a loop or gather operation that
/// stitches together the desired tile from slices of the source of the collapse
/// shape op.
struct RewriteExtractSliceFromCollapseShapeBase
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  RewriteExtractSliceFromCollapseShapeBase(MLIRContext *context)
      : mlir::OpRewritePattern<tensor::ExtractSliceOp>(context) {}

  /// Emit a loop or gather operation that uses `helper` to take each point in
  /// the parallel iteration space bounds, extract a slice from the source
  /// tensor and insert it into `dest`. For examples, see below for `scf.for`
  /// and `scf.foreach`.
  virtual LogicalResult
  emitReplacement(tensor::ExtractSliceOp op, Value dest,
                  tensor::ExtractSliceFromCollapseHelper &helper,
                  PatternRewriter &rewriter) const = 0;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto collapseOp = op.getSource().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseOp)
      return rewriter.notifyMatchFailure(
          op, "producer is not a tensor.collapse_shape op");

    // Try to simplify the collapse shape using a rank-reducing slice, if
    // possible.
    FailureOr<Operation *> simplifiedCollapseShapeResult =
        tensor::simplifyCollapseShapeWithRankReducingExtractSlice(collapseOp,
                                                                  rewriter);
    if (succeeded(simplifiedCollapseShapeResult)) {
      auto newCollapseOp =
          dyn_cast<tensor::CollapseShapeOp>(*simplifiedCollapseShapeResult);
      // The collapse shape op might have been simplified away, so we can just
      // return.
      if (!newCollapseOp)
        return success();
      collapseOp = newCollapseOp;
    }

    // Materialize the output shape values of the slice operation.
    ReifiedRankedShapedTypeDims reifiedShapes;
    if (failed(reifyResultShapes(rewriter, op, reifiedShapes)))
      return rewriter.notifyMatchFailure(op, "failed to reify result shapes");

    // Create the destination tensor using the above values.
    Type elementType = op.getSourceType().getElementType();
    SmallVector<OpFoldResult> outputShape = reifiedShapes[0];
    Value dest = rewriter.create<tensor::EmptyOp>(op->getLoc(), outputShape,
                                                  elementType);

    // Calculate the parameters for the tile loop nest.
    FailureOr<tensor::ExtractSliceFromCollapseHelper> params =
        tensor::ExtractSliceFromCollapseHelper::create(rewriter, collapseOp,
                                                       op);
    if (failed(params))
      return rewriter.notifyMatchFailure(
          op, "could not calculate tiling parameters");
    return emitReplacement(op, dest, *params, rewriter);
  }
};

struct RewriteExtractSliceFromCollapseShapeUsingScfFor
    : public RewriteExtractSliceFromCollapseShapeBase {
  RewriteExtractSliceFromCollapseShapeUsingScfFor(MLIRContext *context)
      : RewriteExtractSliceFromCollapseShapeBase(context) {}
  LogicalResult emitReplacement(tensor::ExtractSliceOp op, Value dest,
                                tensor::ExtractSliceFromCollapseHelper &helper,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    const unsigned numTiledDims = helper.getIterationSpaceSizes().size();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> lbs(numTiledDims, zero);
    SmallVector<Value> steps(numTiledDims, one);

    scf::LoopNest nest = scf::buildLoopNest(
        rewriter, loc, lbs, helper.getIterationSpaceSizes(), steps, dest,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
            ValueRange iterArgs) -> scf::ValueVector {
          auto [tile, insertParams] =
              helper.emitLoopNestBody(nestedBuilder, loc, outputIvs);

          // Insert the slice into the destination.
          return {nestedBuilder.create<tensor::InsertSliceOp>(
              loc, tile, iterArgs[0], insertParams)};
        });
    rewriter.replaceOp(op, nest.results);

    return success();
  }
};

struct RewriteExtractSliceFromCollapseShapeUsingScfForeach
    : public RewriteExtractSliceFromCollapseShapeBase {
  RewriteExtractSliceFromCollapseShapeUsingScfForeach(MLIRContext *context)
      : RewriteExtractSliceFromCollapseShapeBase(context) {}
  LogicalResult emitReplacement(tensor::ExtractSliceOp op, Value dest,
                                tensor::ExtractSliceFromCollapseHelper &helper,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto forallOp = rewriter.create<scf::ForallOp>(
        loc, /*numThreads=*/getAsOpFoldResult(helper.getIterationSpaceSizes()),
        /*outputs=*/dest,
        /*mapping=*/std::nullopt,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange regionArgs) {
          unsigned numThreadIdRegionArgs =
              helper.getIterationSpaceSizes().size();
          unsigned numOutputRegionArgs =
              regionArgs.size() - numThreadIdRegionArgs;
          ValueRange outputIvs = regionArgs.take_front(numThreadIdRegionArgs);
          ValueRange outputArgs = regionArgs.take_back(numOutputRegionArgs);
          assert(outputArgs.size() == 1 &&
                 "there should only be one output region argument");
          auto [tile, insertParams] =
              helper.emitLoopNestBody(nestedBuilder, loc, outputIvs);
          // Insert the slice into the destination.
          auto term = nestedBuilder.create<scf::InParallelOp>(loc);
          nestedBuilder.setInsertionPointToStart(term.getBody());
          nestedBuilder.create<tensor::ParallelInsertSliceOp>(
              loc, tile, outputArgs[0], insertParams);
        });
    rewriter.replaceOp(op, forallOp->getResult(0));
    return success();
  }
};
} // namespace

static LogicalResult
applyRewriteExtractFromCollapseShapePatterns(Operation *rootOp,
                                             bool useForeach) {
  RewritePatternSet patterns(rootOp->getContext());
  if (useForeach)
    patterns.add<RewriteExtractSliceFromCollapseShapeUsingScfForeach>(
        rootOp->getContext());
  else
    patterns.add<RewriteExtractSliceFromCollapseShapeUsingScfFor>(
        rootOp->getContext());
  return applyPatternsAndFoldGreedily(rootOp, std::move(patterns));
}

namespace {
class DummyTrackingListener : public transform::TrackingListener {
public:
  using transform::TrackingListener::TrackingListener;

  // Expose `findReplacementOp` as a public function, so that it can be tested.
  Operation *getReplacementOp(Operation *op, ValueRange newValues) const {
    Operation *replacementOp;
    if (!findReplacementOp(replacementOp, op, newValues).succeeded())
      return nullptr;
    return replacementOp;
  }
};
} // namespace

static LogicalResult testTrackingListenerReplacements(Operation *rootOp) {
  // Find replaced op.
  Operation *replaced = nullptr;
  WalkResult status = rootOp->walk([&](Operation *op) {
    if (op->hasAttr("replaced")) {
      if (replaced) {
        op->emitError("only one 'replaced' op is allowed per test case");
        replaced->emitRemark("other 'replaced' op");
        return WalkResult::interrupt();
      }
      replaced = op;
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted())
    return failure();
  if (!replaced) {
    rootOp->emitError("could not find 'replaced' op");
    return failure();
  }

  // Find replacements.
  SmallVector<Value> replacements(replaced->getNumResults(), Value());
  status = rootOp->walk([&](Operation *op) {
    for (int64_t i = 0; i < replaced->getNumResults(); ++i) {
      if (auto attr = op->getAttrOfType<IntegerAttr>("replacement_" +
                                                     std::to_string(i))) {
        if (replacements[i]) {
          op->emitError("only one 'replacement_" + std::to_string(i) +
                        "' is allowed per test case");
          replacements[i].getDefiningOp()->emitRemark("other 'replacement_" +
                                                      std::to_string(i) + "'");
          return WalkResult::interrupt();
        }
        replacements[i] = op->getResult(attr.getInt());
      }
    }
    return WalkResult::advance();
  });
  if (status.wasInterrupted())
    return failure();

  if (!llvm::all_of(replacements,
                    [](Value v) { return static_cast<bool>(v); })) {
    replaced->emitError("insufficient replacement values");
    return failure();
  }

  // Find the replacement op (if any) and emit a remark/error.
  transform::TransformState transformState =
      transform::detail::makeTransformStateForTesting(/*region=*/nullptr,
                                                      /*payloadRoot=*/nullptr);
  MLIRContext *context = rootOp->getContext();
  OpBuilder builder(context);
  OwningOpRef<transform::NamedSequenceOp> transformOp =
      builder.create<transform::NamedSequenceOp>(
          rootOp->getLoc(),
          /*sym_name=*/"test_sequence",
          /*function_type=*/
          TypeAttr::get(FunctionType::get(context, TypeRange{}, TypeRange{})),
          /*sym_visibility*/ StringAttr::get(context, "public"),
          /*arg_attrs=*/ArrayAttr::get(context, ArrayRef<Attribute>()),
          /*res_attrs=*/ArrayAttr::get(context, ArrayRef<Attribute>()));
  DummyTrackingListener listener(transformState, transformOp.get());
  Operation *replacement = listener.getReplacementOp(replaced, replacements);
  if (!replacement) {
    replaced->emitError("listener could not find replacement op");
    return failure();
  }

  replacement->emitRemark("replacement found");
  return success();
}

void TestTensorTransforms::runOnOperation() {
  Operation *rootOp = getOperation();
  if (testSimplifyPackUnpackPatterns)
    applySimplifyPackUnpackPatterns(rootOp);
  if (testFoldConstantExtractSlice)
    applyFoldConstantExtractSlicePatterns(rootOp);
  if (testFoldConsecutiveInsertExtractSlice)
    applyFoldConsecutiveInsertExtractSlicePatterns(rootOp);
  if (testDropRedundantInsertSliceRankExpansion)
    applyDropRedundantInsertSliceRankExpansionPatterns(rootOp);
  if (testReassociativeReshapeFolding)
    applyReassociativeReshapeFoldingPatterns(rootOp);
  if (testFoldIntoPackAndUnpack)
    applyFoldIntoPackAndUnpackPatterns(rootOp);
  if (testRewriteExtractSliceWithTiledCollapseShape) {
    if (failed(
            applyRewriteExtractFromCollapseShapePatterns(rootOp, useForeach)))
      return signalPassFailure();
  }
  if (testTrackingListener)
    if (failed(testTrackingListenerReplacements(rootOp)))
      return signalPassFailure();
}

namespace mlir {
namespace test {
void registerTestTensorTransforms() {
  PassRegistration<TestTensorTransforms>();
}
} // namespace test
} // namespace mlir
