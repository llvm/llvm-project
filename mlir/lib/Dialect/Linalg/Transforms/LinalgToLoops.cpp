//===- LinalgToLoops.cpp - conversion from Linalg library ops to loops-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AffineOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/LoopOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using edsc::op::operator+;
using edsc::op::operator==;

static SmallVector<ValueHandle, 8>
makeCanonicalAffineApplies(OpBuilder &b, Location loc, AffineMap map,
                           ArrayRef<Value> vals) {
  assert(map.getNumSymbols() == 0);
  assert(map.getNumInputs() == vals.size());
  SmallVector<ValueHandle, 8> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, 0, e);
    SmallVector<Value, 4> operands(vals.begin(), vals.end());
    canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(affine_apply(exprMap, operands));
  }
  return res;
}

static SmallVector<Value, 4> permuteIvs(ArrayRef<Value> ivs,
                                        Optional<AffineMap> permutation) {
  return permutation ? applyMapToValues(ScopedContext::getBuilder(),
                                        ScopedContext::getLocation(),
                                        permutation.getValue(), ivs)
                     : SmallVector<Value, 4>(ivs.begin(), ivs.end());
}

// Creates a number of ranges equal to the number of results in `map`.
// The returned ranges correspond to the loop ranges, in the proper order, for
// which new loops will be created.
static SmallVector<Value, 4> emitLoopRanges(OpBuilder &b, Location loc,
                                            AffineMap map,
                                            ArrayRef<Value> allViewSizes);
SmallVector<Value, 4> emitLoopRanges(OpBuilder &b, Location loc, AffineMap map,
                                     ArrayRef<Value> allViewSizes) {
  // Apply `map` to get view sizes in loop order.
  auto sizes = applyMapToValues(b, loc, map, allViewSizes);
  // Create a new range with the applied tile sizes.
  ScopedContext scope(b, loc);
  SmallVector<Value, 4> res;
  for (unsigned idx = 0, e = map.getNumResults(); idx < e; ++idx) {
    res.push_back(
        linalg_range(std_constant_index(0), sizes[idx], std_constant_index(1)));
  }
  return res;
}

namespace {
template <typename IndexedValueType, typename LinalgOpType>
class LinalgScopedEmitter {};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, CopyOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs, CopyOp copyOp) {
    assert(copyOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");
    auto nPar = copyOp.getNumParallelLoops();
    assert(nPar == allIvs.size());
    auto inputIvs =
        permuteIvs(allIvs.take_front(nPar), copyOp.inputPermutation());
    auto outputIvs =
        permuteIvs(allIvs.take_front(nPar), copyOp.outputPermutation());
    SmallVector<ValueHandle, 8> iivs(inputIvs.begin(), inputIvs.end());
    SmallVector<ValueHandle, 8> oivs(outputIvs.begin(), outputIvs.end());
    IndexedValueType O(copyOp.getOutputBuffer(0)), I(copyOp.getInput(0));
    // Emit the proper scalar assignment, whether we are dealing with a 0-D or
    // an n-D loop nest; with or without permutations.
    // clang-format off
    nPar > 0 ? O(oivs) = I(iivs) :
               O() = I();
    // clang-format on
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, FillOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs, FillOp fillOp) {
    assert(fillOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");
    auto nPar = fillOp.getNumParallelLoops();
    assert(nPar == allIvs.size());
    auto ivs =
        SmallVector<ValueHandle, 4>(allIvs.begin(), allIvs.begin() + nPar);
    IndexedValueType O(fillOp.getOutputBuffer(0));
    // Emit the proper scalar assignment, whether we are dealing with a 0-D or
    // an n-D loop nest; with or without permutations.
    nPar > 0 ? O(ivs) = ValueHandle(fillOp.value())
             : O() = ValueHandle(fillOp.value());
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, DotOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs, DotOp dotOp) {
    assert(dotOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");
    assert(allIvs.size() == 1);
    ValueHandle r_i(allIvs[0]);
    IndexedValueType A(dotOp.getInput(0)), B(dotOp.getInput(1)),
        C(dotOp.getOutputBuffer(0));
    // Emit scalar form.
    C() = C() + A(r_i) * B(r_i);
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, MatvecOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs,
                                       MatvecOp matvecOp) {
    assert(matvecOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");
    assert(allIvs.size() == 2);
    ValueHandle i(allIvs[0]), r_j(allIvs[1]);
    IndexedValueType A(matvecOp.getInput(0)), B(matvecOp.getInput(1)),
        C(matvecOp.getOutputBuffer(0));
    // Emit scalar form.
    C(i) = C(i) + A(i, r_j) * B(r_j);
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, MatmulOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs,
                                       MatmulOp matmulOp) {
    assert(matmulOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");
    assert(allIvs.size() == 3);
    ValueHandle i(allIvs[0]), j(allIvs[1]), r_k(allIvs[2]);
    IndexedValueType A(matmulOp.getInput(0)), B(matmulOp.getInput(1)),
        C(matmulOp.getOutputBuffer(0));
    // Emit scalar form.
    C(i, j) = C(i, j) + A(i, r_k) * B(r_k, j);
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, ConvOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs, ConvOp convOp) {
    assert(convOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    auto maps = loopToOperandRangesMaps(convOp);
    SmallVector<ValueHandle, 8> fIdx(
        makeCanonicalAffineApplies(b, loc, maps[0], allIvs));
    SmallVector<ValueHandle, 8> imIdx(
        makeCanonicalAffineApplies(b, loc, maps[1], allIvs));
    SmallVector<ValueHandle, 8> oIdx(
        makeCanonicalAffineApplies(b, loc, maps[2], allIvs));
    IndexedValueType F(convOp.filter()), I(convOp.input()), O(convOp.output());
    // Emit scalar form.
    O(oIdx) += F(fIdx) * I(imIdx);
  }
};

// Emits the MLIR for the scalar part of the generic op by:
//   1. Emitting std_load and std_store ops for each input and output
//      view in order. This is achieved by applying the appropriate input or
//      output map to the enclosing induction variables.
//   2. Emitting a call to `op.fun()` that takes as arguments the scalars
//      from point 1. above.
//   3. Emitting std_store to store the results of 2. to the output
//      views.
//
// An example output may resemble:
//
// ```
//    loop.for %i = %c0 to %0 step %c1 {
//      loop.for %j = %c0 to %1 step %c1 {
//        loop.for %k = %c0 to %4 step %c1 {
//          %11 = load %arg0[%i, %j] :
//            memref<?x?xf32, stride_specification>
//          %12 = load %arg1[%i, %j, %k] :
//            memref<?x?x?xf32, stride_specification>
//          %13 = load %arg2[%i, %k, %j] :
//            memref<?x?x?xf32, stride_specification>
//          %14:2 = call @foo(%11, %12, %13) : (f32, f32, f32) -> (f32, f32)
//          store %14#0, %arg1[%i, %j, %k] :
//            memref<?x?x?Xf32, stride_specification>
//          store %14#1, %arg2[%i, %k, %j] :
//            memref<?x?x?Xf32, stride_specification>
//       }
//      }
//    }
// ```
template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, GenericOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs,
                                       GenericOp genericOp) {
    assert(genericOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    using edsc::intrinsics::detail::ValueHandleArray;
    unsigned nInputs = genericOp.getNumInputs();
    unsigned nOutputs = genericOp.getNumOutputs();
    SmallVector<Value, 4> indexedValues(nInputs + nOutputs);

    // 1.a. Emit std_load from input views.
    for (unsigned i = 0; i < nInputs; ++i) {
      Value input = genericOp.getInput(i);
      if (!input.getType().cast<ShapedType>().getRank()) {
        indexedValues[i] = std_load(input);
      } else {
        ValueHandleArray indexing(makeCanonicalAffineApplies(
            b, loc, genericOp.getInputIndexingMap(i), allIvs));
        indexedValues[i] = std_load(input, indexing);
      }
    }

    // 1.b. Emit std_load from output views.
    for (unsigned i = 0; i < nOutputs; ++i) {
      ValueHandleArray indexing(makeCanonicalAffineApplies(
          b, loc, genericOp.getOutputIndexingMap(i), allIvs));
      indexedValues[nInputs + i] =
          std_load(genericOp.getOutputBuffer(i), indexing);
    }

    auto funcOp = genericOp.getFunction();
    if (funcOp) {
      // 2. Emit call.
      Operation *callOp = std_call(funcOp, indexedValues);
      assert(callOp->getNumResults() == genericOp.getNumOutputs());

      // 3. Emit std_store.
      for (unsigned i = 0; i < nOutputs; ++i) {
        ValueHandleArray indexing(makeCanonicalAffineApplies(
            b, loc, genericOp.getOutputIndexingMap(i), allIvs));
        std_store(callOp->getResult(i), genericOp.getOutputBuffer(i), indexing);
      }
      return;
    }
    // TODO(ntv): When a region inliner exists, use it.
    // 2. Inline region, currently only works for a single basic block.
    BlockAndValueMapping map;
    auto &block = genericOp.region().front();
    map.map(block.getArguments(), indexedValues);
    for (auto &op : block.without_terminator()) {
      assert(op.getNumRegions() == 0);
      auto *newOp = b.clone(op, map);
      map.map(op.getResults(), newOp->getResults());
    }

    // 3. Emit std_store.
    auto *yieldOp = cast<YieldOp>(block.back()).getOperation();
    assert(yieldOp->getNumOperands() == nOutputs);
    for (unsigned i = 0; i < nOutputs; ++i) {
      ValueHandleArray indexing(makeCanonicalAffineApplies(
          b, loc, genericOp.getOutputIndexingMap(i), allIvs));
      std_store(map.lookup(yieldOp->getOperand(i)),
                genericOp.getOutputBuffer(i), indexing);
    }
  }
};

// Emits the MLIR for the scalar part of the indexed generic op by:
//   1. Emitting std_load and std_store ops for each input and output view in
//      order. This is achieved by applying the appropriate input or output map
//      to the enclosing induction variables.
//   2. Emitting a call to `op.fun()` that takes as arguments the induction
//      variables and the scalars from point 1. above.
//   3. Emitting std_store to store the results of 2. to the output views.
//
// An example output may resemble:
//
// ```
//    loop.for %i = %c0 to %0 step %c1 {
//      loop.for %j = %c0 to %1 step %c1 {
//        loop.for %k = %c0 to %4 step %c1 {
//          %11 = load %arg0[%i, %j] :
//            memref<?x?xf32, stride_specification>
//          %12 = load %arg1[%i, %j, %k] :
//            memref<?x?x?xf32, stride_specification>
//          %13 = load %arg2[%i, %k, %j] :
//            memref<?x?x?xf32, stride_specification>
//          %14:2 = call @foo(%i, %j, %k, %11, %12, %13) :
//            (index, index, index, f32, f32, f32) -> (f32, f32)
//          store %14#0, %arg1[%i, %j, %k] :
//            memref<?x?x?Xf32, stride_specification>
//          store %14#1, %arg2[%i, %k, %j] :
//            memref<?x?x?Xf32, stride_specification>
//       }
//      }
//    }
// ```
template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, IndexedGenericOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs,
                                       IndexedGenericOp indexedGenericOp) {
    assert(indexedGenericOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    using edsc::intrinsics::detail::ValueHandleArray;
    unsigned nInputs = indexedGenericOp.getNumInputs();
    unsigned nOutputs = indexedGenericOp.getNumOutputs();
    unsigned nLoops = allIvs.size();
    SmallVector<Value, 4> indexedValues(nLoops + nInputs + nOutputs);

    for (unsigned i = 0; i < nLoops; ++i) {
      indexedValues[i] = allIvs[i];
    }

    // 1.a. Emit std_load from input views.
    for (unsigned i = 0; i < nInputs; ++i) {
      Value input = indexedGenericOp.getInput(i);
      if (!input.getType().cast<ShapedType>().getRank()) {
        indexedValues[nLoops + i] = std_load(input);
      } else {
        ValueHandleArray indexing(makeCanonicalAffineApplies(
            b, loc, indexedGenericOp.getInputIndexingMap(i), allIvs));
        indexedValues[nLoops + i] = std_load(input, indexing);
      }
    }

    // 1.b. Emit std_load from output views.
    for (unsigned i = 0; i < nOutputs; ++i) {
      ValueHandleArray indexing(makeCanonicalAffineApplies(
          b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs));
      indexedValues[nLoops + nInputs + i] =
          std_load(indexedGenericOp.getOutputBuffer(i), indexing);
    }

    if (auto funcOp = indexedGenericOp.getFunction()) {
      // 2. Emit call.
      Operation *callOp = std_call(funcOp, indexedValues);
      assert(callOp->getNumResults() == indexedGenericOp.getNumOutputs());

      // 3. Emit std_store.
      for (unsigned i = 0; i < nOutputs; ++i) {
        ValueHandleArray indexing(makeCanonicalAffineApplies(
            b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs));
        std_store(callOp->getResult(i), indexedGenericOp.getOutputBuffer(i),
                  indexing);
      }
      return;
    }
    // TODO(ntv): When a region inliner exists, use it.
    // 2. Inline region, currently only works for a single basic block.
    BlockAndValueMapping map;
    auto &block = indexedGenericOp.region().front();
    map.map(block.getArguments(), indexedValues);
    for (auto &op : block.without_terminator()) {
      assert(op.getNumRegions() == 0);
      auto *newOp = b.clone(op, map);
      map.map(op.getResults(), newOp->getResults());
    }

    // 3. Emit std_store.
    auto *yieldOp = cast<YieldOp>(block.back()).getOperation();
    assert(yieldOp->getNumOperands() == nOutputs);
    for (unsigned i = 0; i < nOutputs; ++i) {
      ValueHandleArray indexing(makeCanonicalAffineApplies(
          b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs));
      std_store(map.lookup(yieldOp->getOperand(i)),
                indexedGenericOp.getOutputBuffer(i), indexing);
    }
  }
};

// This struct is for factoring out the implementation and support template
// instantiations in the following 2 cases:
//   1. Appending to a list of patterns via RewritePatternList.
//   2. Direct invocation via `linalgOpToLoops` and `linalgOpToAffineLoops`.
// The implementation must work both in DRR and inside a RewritePattern. As a
// consequence, (1) it is only allowed to emit new ops if the match is
// guaranteed to be a success, (2) it is not allowed erase/replace, and (3) an
// encompassing pattern must take care of the erasure logic.
template <typename LoopTy, typename IndexedValueTy, typename ConcreteOpTy>
class LinalgOpToLoopsImpl {
public:
  static LogicalResult doit(Operation *op, PatternRewriter &rewriter);
};

template <typename LoopTy>
bool loweringIsAllowed(int numParallelLoops, int numLoops) {
  return true;
}
template <>
bool loweringIsAllowed<loop::ParallelOp>(int numParallelLoops, int numLoops) {
  return numParallelLoops == numLoops;
}

template <typename LoopTy, typename IndexedValueTy, typename ConcreteOpTy>
LogicalResult LinalgOpToLoopsImpl<LoopTy, IndexedValueTy, ConcreteOpTy>::doit(
    Operation *op, PatternRewriter &rewriter) {
  OpBuilder b(op);
  ScopedContext scope(b, op->getLoc());

  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  auto linalgOp = cast<ConcreteOpTy>(op);
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto nPar = linalgOp.getNumParallelLoops();
  auto nRed = linalgOp.getNumReductionLoops();
  auto nWin = linalgOp.getNumWindowLoops();
  auto nLoops = nPar + nRed + nWin;
  if (!loweringIsAllowed<LoopTy>(nPar, nLoops))
    return failure();
  auto invertedMap =
      inversePermutation(concatAffineMaps(loopToOperandRangesMaps(linalgOp)));
  if (!invertedMap) {
    LinalgScopedEmitter<IndexedValueTy, ConcreteOpTy>::emitScalarImplementation(
        {}, linalgOp);
    return success();
  }

  SmallVector<ValueHandle, 4> allIvs(nLoops, ValueHandle(b.getIndexType()));
  SmallVector<ValueHandle *, 4> allPIvs =
      makeHandlePointers(MutableArrayRef<ValueHandle>(allIvs));
  auto loopRanges = emitLoopRanges(scope.getBuilder(), scope.getLocation(),
                                   invertedMap, getViewSizes(b, linalgOp));
  assert(loopRanges.size() == allIvs.size());

  GenericLoopNestRangeBuilder<LoopTy>(allPIvs, loopRanges)([&] {
    SmallVector<Value, 4> allIvValues(allIvs.begin(), allIvs.end());
    LinalgScopedEmitter<IndexedValueTy, ConcreteOpTy>::emitScalarImplementation(
        allIvValues, linalgOp);
  });
  return success();
}

template <typename LoopType, typename IndexedValueType, typename ConcreteOp>
class LinalgRewritePattern : public RewritePattern {
public:
  explicit LinalgRewritePattern(MLIRContext *context)
      : RewritePattern(ConcreteOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    using Impl = LinalgOpToLoopsImpl<LoopType, IndexedValueType, ConcreteOp>;
    if (failed(Impl::doit(op, rewriter)))
      return matchFailure();
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

// Helper classes for type list expansion.
template <typename LoopType, typename IndexedValueType, typename... LinalgOps>
class RewritePatternList;

template <typename LoopType, typename IndexedValueType>
class RewritePatternList<LoopType, IndexedValueType> {
public:
  static void build(OwningRewritePatternList &patterns, MLIRContext *ctx) {}
};

template <typename LoopType, typename IndexedValueType, typename ConcreteOp,
          typename... LinalgOps>
class RewritePatternList<LoopType, IndexedValueType, ConcreteOp, LinalgOps...> {
public:
  static void build(OwningRewritePatternList &patterns, MLIRContext *ctx) {
    patterns
        .insert<LinalgRewritePattern<LoopType, IndexedValueType, ConcreteOp>>(
            ctx);
    RewritePatternList<LoopType, IndexedValueType, LinalgOps...>::build(
        patterns, ctx);
  }
};

/// Populate the given list with patterns that convert from Linalg to LLVM.
template <typename LoopType, typename IndexedValueType>
void FillRewritePatterns(OwningRewritePatternList &patterns, MLIRContext *ctx) {
  RewritePatternList<LoopType, IndexedValueType,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                     >::build(patterns, ctx);
}

namespace {
template <typename LoopType, typename IndexedValueType>
struct LowerLinalgToLoopsPass
    : public FunctionPass<LowerLinalgToLoopsPass<LoopType, IndexedValueType>> {
  void runOnFunction() override;
};
} // namespace

// Local folding pattern for AffineApplyOp that we can apply greedily.
// This replaces AffineApplyOp by the proper value in cases where the associated
// map is trivial. A trivial map here is defined as a map with a single result
// and either:
//   1. Zero operand + returns a single AffineConstantExpr
//   2. One operand + returns a single AffineDimExpr
//   3. One operands + returns a single AffineSymbolExpr
//
// In the first case, the AffineApplyOp is replaced by a new constant. In the
// other cases, it is replaced by its unique operand.
struct FoldAffineOp : public RewritePattern {
  FoldAffineOp(MLIRContext *context)
      : RewritePattern(AffineApplyOp::getOperationName(), 0, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    AffineApplyOp affineApplyOp = cast<AffineApplyOp>(op);
    auto map = affineApplyOp.getAffineMap();
    if (map.getNumResults() != 1 || map.getNumInputs() > 1)
      return matchFailure();

    AffineExpr expr = map.getResult(0);
    if (map.getNumInputs() == 0) {
      if (auto val = expr.dyn_cast<AffineConstantExpr>()) {
        rewriter.replaceOpWithNewOp<ConstantIndexOp>(op, val.getValue());
        return matchSuccess();
      }
      return matchFailure();
    }
    if (expr.dyn_cast<AffineDimExpr>() || expr.dyn_cast<AffineSymbolExpr>()) {
      rewriter.replaceOp(op, op->getOperand(0));
      return matchSuccess();
    }
    return matchFailure();
  }
};
} // namespace

template <typename LoopType, typename IndexedValueType>
void LowerLinalgToLoopsPass<LoopType, IndexedValueType>::runOnFunction() {
  auto *context = &this->getContext();
  OwningRewritePatternList patterns;
  // Canonicalization and folding patterns applied greedily allow cleaning up
  // the emitted IR on the fly.
  // TODO(ntv) fold view and subview ops?
  FillRewritePatterns<LoopType, IndexedValueType>(patterns, context);
  DimOp::getCanonicalizationPatterns(patterns, context);
  AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  patterns.insert<FoldAffineOp>(context);
  // Just apply the patterns greedily.
  applyPatternsGreedily(this->getFunction(), patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertLinalgToLoopsPass() {
  return std::make_unique<
      LowerLinalgToLoopsPass<loop::ForOp, StdIndexedValue>>();
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createConvertLinalgToParallelLoopsPass() {
  return std::make_unique<
      LowerLinalgToLoopsPass<loop::ParallelOp, StdIndexedValue>>();
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createConvertLinalgToAffineLoopsPass() {
  return std::make_unique<
      LowerLinalgToLoopsPass<AffineForOp, AffineIndexedValue>>();
}

/// Emits a loop nest of `loop.for` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult mlir::linalg::linalgOpToLoops(PatternRewriter &rewriter,
                                            Operation *op) {
  return LinalgOpToLoopsImpl<loop::ForOp, StdIndexedValue, ConcreteOp>::doit(
      op, rewriter);
}

/// Emits a loop nest of `affine.for` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult mlir::linalg::linalgOpToAffineLoops(PatternRewriter &rewriter,
                                                  Operation *op) {
  return LinalgOpToLoopsImpl<AffineForOp, AffineIndexedValue, ConcreteOp>::doit(
      op, rewriter);
}

/// Emits a loop nest of `loop.parallel` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult mlir::linalg::linalgOpToParallelLoops(PatternRewriter &rewriter,
                                                    Operation *op) {
  return LinalgOpToLoopsImpl<loop::ParallelOp, StdIndexedValue,
                             ConcreteOp>::doit(op, rewriter);
}

// TODO(ntv) Need to make these instantiations more future-proof to avoid the
// need to update as soon as we add new ops.
#define INSTANTIATE_LINALG_OP_TO_LOOPS(OP_TYPE)                                \
  template LogicalResult mlir::linalg::linalgOpToLoops<OP_TYPE>(               \
      PatternRewriter & rewriter, Operation * op);                             \
  template LogicalResult mlir::linalg::linalgOpToAffineLoops<OP_TYPE>(         \
      PatternRewriter & rewriter, Operation * op);

INSTANTIATE_LINALG_OP_TO_LOOPS(CopyOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(FillOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(DotOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(MatvecOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(MatmulOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(ConvOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(GenericOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(IndexedGenericOp)

// TODO(pifon): Enable lowering to parallel loops for ops other than
// linalg.generic for now to be on the safe side.
template LogicalResult
mlir::linalg::linalgOpToParallelLoops<GenericOp>(PatternRewriter &rewriter,
                                                 Operation *op);

static PassRegistration<LowerLinalgToLoopsPass<loop::ForOp, StdIndexedValue>>
    structuredLoopsPass(
        "convert-linalg-to-loops",
        "Lower the operations from the linalg dialect into loops");

static PassRegistration<
    LowerLinalgToLoopsPass<loop::ParallelOp, StdIndexedValue>>
    parallelLoopsPass(
        "convert-linalg-to-parallel-loops",
        "Lower the operations from the linalg dialect into parallel loops");

static PassRegistration<LowerLinalgToLoopsPass<AffineForOp, AffineIndexedValue>>
    affineLoopsPass(
        "convert-linalg-to-affine-loops",
        "Lower the operations from the linalg dialect into affine loops");
