//===- Loops.cpp - conversion from Linalg named and generic ops to loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linalg;

static SmallVector<Value> makeCanonicalAffineApplies(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> vals) {
  if (map.isEmpty())
    return {};

  assert(map.getNumInputs() == vals.size());
  SmallVector<Value> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
    SmallVector<Value> operands(vals.begin(), vals.end());
    canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(b.create<AffineApplyOp>(loc, exprMap, operands));
  }
  return res;
}

template <typename LoadOpTy, typename StoreOpTy, typename OpType>
static void inlineRegionAndEmitStore(OpBuilder &b, Location loc, OpType op,
                                     ArrayRef<Value> indexedValues,
                                     ArrayRef<SmallVector<Value>> indexing,
                                     ArrayRef<Value> outputBuffers) {
  auto &block = op->getRegion(0).front();
  BlockAndValueMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    auto *newOp = b.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }

  Operation *terminator = block.getTerminator();
  for (OpOperand &operand : terminator->getOpOperands()) {
    Value toStore = map.lookupOrDefault(operand.get());
    b.create<StoreOpTy>(loc, toStore, outputBuffers[operand.getOperandNumber()],
                        indexing[operand.getOperandNumber()]);
  }
}

// Returns a pair that contains input indices and output indices of a
// SingleInputPoolingOp `op`.
struct InputAndOutputIndices {
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
};
template <typename SingleInputPoolingOp>
static InputAndOutputIndices
getInputAndOutputIndices(OpBuilder &b, Location loc, ArrayRef<Value> allIvs,
                         SingleInputPoolingOp op) {
  auto mapsRange = op.getIndexingMapsArray();
  auto maps = llvm::to_vector<8>(
      llvm::map_range(mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
  return InputAndOutputIndices{
      makeCanonicalAffineApplies(b, loc, maps[0], allIvs),
      makeCanonicalAffineApplies(b, loc, maps[2], allIvs)};
}

/// Emits the MLIR for the scalar part of the generic op by:
///   1. Emitting load ops for each input and output view in order. This is
///      achieved by applying the appropriate input or output map to the
///      enclosing induction variables.
///   2. Emitting a call to `op.fun()` that takes as arguments the scalars
///      from point 1. above.
///   3. Emitting store ops to store the results of 2. to the output
///      views.
///
/// An example output may resemble:
///
/// ```
///    scf.for %i = %c0 to %0 step %c1 {
///      scf.for %j = %c0 to %1 step %c1 {
///        scf.for %k = %c0 to %4 step %c1 {
///          %11 = load %arg0[%i, %j] :
///            memref<?x?xf32, stride_specification>
///          %12 = load %arg1[%i, %j, %k] :
///            memref<?x?x?xf32, stride_specification>
///          %13 = load %arg2[%i, %k, %j] :
///            memref<?x?x?xf32, stride_specification>
///          %14:2 = call @foo(%11, %12, %13) : (f32, f32, f32) -> (f32, f32)
///          store %14#0, %arg1[%i, %j, %k] :
///            memref<?x?x?Xf32, stride_specification>
///          store %14#1, %arg2[%i, %k, %j] :
///            memref<?x?x?Xf32, stride_specification>
///       }
///      }
///    }
/// ```
template <typename LoadOpTy, typename StoreOpTy>
static void emitScalarImplementation(OpBuilder &b, Location loc,
                                     ArrayRef<Value> allIvs,
                                     LinalgOp linalgOp) {
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp.getNumInputsAndOutputs());

  auto allIvsPlusDims = SmallVector<Value>(allIvs.begin(), allIvs.end());

  // TODO: Avoid the loads if the corresponding argument of the
  // region has no uses.
  // 1.a. Emit load from input operand or for scalars access the operand itself.
  for (OpOperand *inputOperand : linalgOp.getInputOperands()) {
    if (linalgOp.isScalar(inputOperand)) {
      indexedValues.push_back(inputOperand->get());
      continue;
    }
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getTiedIndexingMap(inputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<LoadOpTy>(loc, inputOperand->get(), indexing));
  }
  // 1.b. Emit load from output views.
  for (OpOperand *outputOperand : linalgOp.getOutputOperands()) {
    SmallVector<Value> indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getTiedIndexingMap(outputOperand), allIvsPlusDims);
    indexedValues.push_back(
        b.create<LoadOpTy>(loc, outputOperand->get(), indexing));
  }

  // TODO: When a region inliner exists, use it.
  // 2. Inline region, currently only works for a single basic block.
  // 3. Emit store.
  SmallVector<SmallVector<Value>, 8> indexing;
  SmallVector<Value> outputBuffers;
  for (OpOperand *outputOperand : linalgOp.getOutputBufferOperands()) {
    indexing.push_back(makeCanonicalAffineApplies(
        b, loc, linalgOp.getTiedIndexingMap(outputOperand), allIvsPlusDims));
    outputBuffers.push_back(outputOperand->get());
  }
  inlineRegionAndEmitStore<LoadOpTy, StoreOpTy>(b, loc, linalgOp, indexedValues,
                                                indexing, outputBuffers);
}

/// Replace the index operations in the body of the loop nest by the matching
/// induction variables.
static void replaceIndexOpsByInductionVariables(LinalgOp linalgOp,
                                                PatternRewriter &rewriter,
                                                ArrayRef<Operation *> loopOps) {
  // Extract the induction variables of the loop nest from outer to inner.
  SmallVector<Value> allIvs;
  for (Operation *loopOp : loopOps) {
    llvm::TypeSwitch<Operation *>(loopOp)
        .Case([&](scf::ParallelOp parallelOp) {
          allIvs.append(parallelOp.getInductionVars().begin(),
                        parallelOp.getInductionVars().end());
        })
        .Case([&](scf::ForOp forOp) {
          allIvs.push_back(forOp.getInductionVar());
        })
        .Case([&](AffineForOp affineForOp) {
          allIvs.push_back(affineForOp.getInductionVar());
        })
        .Default([&](Operation *op) { assert(false && "unexpected op"); });
  }
  assert(linalgOp.getNumLoops() == allIvs.size() &&
         "expected the number of loops and induction variables to match");
  // Replace the index operations in the body of the innermost loop op.
  if (!loopOps.empty()) {
    LoopLikeOpInterface loopOp = loopOps.back();
    for (IndexOp indexOp :
         llvm::make_early_inc_range(loopOp.getLoopBody().getOps<IndexOp>()))
      rewriter.replaceOp(indexOp, allIvs[indexOp.dim()]);
  }
}

template <typename LoopTy>
static FailureOr<LinalgLoops> linalgOpToLoopsImpl(PatternRewriter &rewriter,
                                                  LinalgOp linalgOp) {
  using LoadOpTy =
      typename std::conditional<std::is_same<LoopTy, AffineForOp>::value,
                                AffineLoadOp, memref::LoadOp>::type;
  using StoreOpTy =
      typename std::conditional<std::is_same<LoopTy, AffineForOp>::value,
                                AffineStoreOp, memref::StoreOp>::type;

  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");

  auto loopRanges = linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
  auto iteratorTypes = llvm::to_vector<4>(linalgOp.iterator_types().getValue());

  SmallVector<Value> allIvs;
  GenerateLoopNest<LoopTy>::doit(
      rewriter, linalgOp.getLoc(), loopRanges, linalgOp, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange ivs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
        assert(operandValuesToUse == linalgOp->getOperands() &&
               "expect operands are captured and not passed by loop argument");
        allIvs.append(ivs.begin(), ivs.end());
        emitScalarImplementation<LoadOpTy, StoreOpTy>(b, loc, allIvs, linalgOp);
        return scf::ValueVector{};
      });
  // Number of loop ops might be different from the number of ivs since some
  // loops like affine.parallel and scf.parallel have multiple ivs.
  SetVector<Operation *> loopSet;
  for (Value iv : allIvs) {
    if (!iv)
      return failure();
    // The induction variable is a block argument of the entry block of the
    // loop operation.
    BlockArgument ivVal = iv.dyn_cast<BlockArgument>();
    if (!ivVal)
      return failure();
    loopSet.insert(ivVal.getOwner()->getParentOp());
  }
  LinalgLoops loops(loopSet.begin(), loopSet.end());
  // Replace all index operations in the loop body.
  replaceIndexOpsByInductionVariables(linalgOp, rewriter, loops);
  return loops;
}

namespace {
template <typename LoopType>
class LinalgRewritePattern : public RewritePattern {
public:
  LinalgRewritePattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<LinalgOp>(op);
    if (!isa<LinalgOp>(op))
      return failure();
    if (failed(linalgOpToLoopsImpl<LoopType>(rewriter, linalgOp)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

/// Local folding pattern for AffineApplyOp that we can apply greedily.
/// This replaces AffineApplyOp by the proper value in cases where the
/// associated map is trivial.
/// A trivial map here is defined as a map with a single result and either:
///   1. Zero operand + returns a single AffineConstantExpr
///   2. One operand + returns a single AffineDimExpr
///   3. One operand + returns a single AffineSymbolExpr
//
/// In the first case, the AffineApplyOp is replaced by a new constant. In the
/// other cases, it is replaced by its unique operand.
struct FoldAffineOp : public RewritePattern {
  FoldAffineOp(MLIRContext *context)
      : RewritePattern(AffineApplyOp::getOperationName(), 0, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    AffineApplyOp affineApplyOp = cast<AffineApplyOp>(op);
    auto map = affineApplyOp.getAffineMap();
    if (map.getNumResults() != 1 || map.getNumInputs() > 1)
      return failure();

    AffineExpr expr = map.getResult(0);
    if (map.getNumInputs() == 0) {
      if (auto val = expr.dyn_cast<AffineConstantExpr>()) {
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, val.getValue());
        return success();
      }
      return failure();
    }
    if (expr.dyn_cast<AffineDimExpr>() || expr.dyn_cast<AffineSymbolExpr>()) {
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }
    return failure();
  }
};

template <typename LoopType>
static void lowerLinalgToLoopsImpl(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  patterns.add<LinalgRewritePattern<LoopType>>(context);
  memref::DimOp::getCanonicalizationPatterns(patterns, context);
  tensor::DimOp::getCanonicalizationPatterns(patterns, context);
  AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  patterns.add<FoldAffineOp>(context);
  // Just apply the patterns greedily.
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

struct LowerToAffineLoops
    : public LinalgLowerToAffineLoopsBase<LowerToAffineLoops> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
  void runOnOperation() override {
    lowerLinalgToLoopsImpl<AffineForOp>(getOperation());
  }
};

struct LowerToLoops : public LinalgLowerToLoopsBase<LowerToLoops> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    lowerLinalgToLoopsImpl<scf::ForOp>(getOperation());
  }
};

struct LowerToParallelLoops
    : public LinalgLowerToParallelLoopsBase<LowerToParallelLoops> {
  void runOnOperation() override {
    lowerLinalgToLoopsImpl<scf::ParallelOp>(getOperation());
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertLinalgToLoopsPass() {
  return std::make_unique<LowerToLoops>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertLinalgToParallelLoopsPass() {
  return std::make_unique<LowerToParallelLoops>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertLinalgToAffineLoopsPass() {
  return std::make_unique<LowerToAffineLoops>();
}

/// Emits a loop nest of `affine.for` with the proper body for `linalgOp`.
FailureOr<LinalgLoops>
mlir::linalg::linalgOpToAffineLoops(PatternRewriter &rewriter,
                                    LinalgOp linalgOp) {
  return linalgOpToLoopsImpl<AffineForOp>(rewriter, linalgOp);
}

/// Emits a loop nest of `scf.for` with the proper body for `linalgOp`.
FailureOr<LinalgLoops> mlir::linalg::linalgOpToLoops(PatternRewriter &rewriter,
                                                     LinalgOp linalgOp) {
  return linalgOpToLoopsImpl<scf::ForOp>(rewriter, linalgOp);
}

/// Emits a loop nest of `scf.parallel` with the proper body for `linalgOp`.
FailureOr<LinalgLoops>
mlir::linalg::linalgOpToParallelLoops(PatternRewriter &rewriter,
                                      LinalgOp linalgOp) {
  return linalgOpToLoopsImpl<scf::ParallelOp>(rewriter, linalgOp);
}
