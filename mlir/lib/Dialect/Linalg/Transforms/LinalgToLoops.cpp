//===- LinalgToLoops.cpp - conversion from Linalg library ops to loops-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/FoldedIntrinsics.h"
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
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using edsc::op::operator+;
using edsc::op::operator==;
using mlir::edsc::intrinsics::detail::ValueHandleArray;

static SmallVector<ValueHandle, 8>
makeCanonicalAffineApplies(OpBuilder &b, Location loc, AffineMap map,
                           ArrayRef<Value> vals) {
  if (map.isEmpty())
    return {};
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

template <typename OpType>
static void inlineRegionAndEmitStdStore(OpType op,
                                        ArrayRef<Value> indexedValues,
                                        ArrayRef<ValueHandleArray> indexing,
                                        ArrayRef<Value> outputBuffers) {
  auto &b = ScopedContext::getBuilder();
  auto &block = op.region().front();
  BlockAndValueMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    assert(op.getNumRegions() == 0 && "expected a non-nested region");
    auto *newOp = b.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }

  Operation &terminator = block.back();
  assert(isa<YieldOp>(terminator) &&
         "expected an yield op in the end of the region");
  for (unsigned i = 0, e = terminator.getNumOperands(); i < e; ++i) {
    std_store(map.lookupOrDefault(terminator.getOperand(i)), outputBuffers[i],
              indexing[i]);
  }
}

// Returns a pair that contains input indices and output indices of a
// SingleInputPoolingOp `op`.
template <typename SingleInputPoolingOp>
static std::pair<SmallVector<ValueHandle, 8>, SmallVector<ValueHandle, 8>>
getInputAndOutputIndices(ArrayRef<Value> allIvs, SingleInputPoolingOp op) {
  auto &b = ScopedContext::getBuilder();
  auto loc = ScopedContext::getLocation();
  auto mapsRange = op.indexing_maps().template getAsRange<AffineMapAttr>();
  auto maps = llvm::to_vector<8>(
      llvm::map_range(mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
  SmallVector<ValueHandle, 8> iIdx(
      makeCanonicalAffineApplies(b, loc, maps[0], allIvs));
  SmallVector<ValueHandle, 8> oIdx(
      makeCanonicalAffineApplies(b, loc, maps[2], allIvs));
  return {iIdx, oIdx};
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
  /// Returns the input value of convOp. If the indices in `imIdx` is out of
  /// boundary, returns 0 instead.
  static ValueHandle getConvOpInput(ConvOp convOp, IndexedValueType im,
                                    ArrayRef<ValueHandle> imIdx) {
    // TODO(ntv): add a level of indirection to linalg.generic.
    if (!convOp.padding())
      return im(imIdx);

    auto *context = ScopedContext::getContext();
    ValueHandle zeroIndex = std_constant_index(0);
    SmallVector<ValueHandle, 8> conds;
    SmallVector<ValueHandle, 8> clampedImIdx;
    for (auto iter : llvm::enumerate(imIdx)) {
      int idx = iter.index();
      auto dim = iter.value();
      // Only need to iterate over the window dimensions.
      if (idx == 0 || idx == static_cast<int>(imIdx.size()) - 1) {
        clampedImIdx.push_back(dim);
        continue;
      }

      using edsc::op::operator<;
      using edsc::op::operator>=;
      using edsc::op::operator||;
      ValueHandle leftOutOfBound = dim < zeroIndex;
      if (conds.empty())
        conds.push_back(leftOutOfBound);
      else
        conds.push_back(conds.back() || leftOutOfBound);
      ValueHandle rightBound = std_dim(convOp.input(), idx);
      conds.push_back(conds.back() || (dim >= rightBound));

      // When padding is involved, the indices will only be shifted to negative,
      // so having a max op is enough.
      auto maxMap = AffineMap::get(/*dimCount=*/1, 0,
                                   {getAffineDimExpr(/*position=*/0, context),
                                    getAffineConstantExpr(0, context)},
                                   context);
      clampedImIdx.push_back(
          affine_max(dim.getType(), maxMap, ValueRange{dim}));
    }

    auto b = ScopedContext::getBuilder();
    Type type = convOp.input().getType().cast<MemRefType>().getElementType();
    ValueHandle zero = std_constant(type, b.getZeroAttr(type));
    ValueHandle readInput = im(clampedImIdx);
    return conds.empty() ? readInput
                         : std_select(conds.back(), zero, readInput);
  }

  static void emitScalarImplementation(ArrayRef<Value> allIvs, ConvOp convOp) {
    assert(convOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    auto mapsRange = convOp.indexing_maps().getAsRange<AffineMapAttr>();
    auto maps = llvm::to_vector<8>(llvm::map_range(
        mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
    SmallVector<ValueHandle, 8> fIdx(
        makeCanonicalAffineApplies(b, loc, maps[0], allIvs));
    SmallVector<ValueHandle, 8> imIdx(
        makeCanonicalAffineApplies(b, loc, maps[1], allIvs));
    SmallVector<ValueHandle, 8> oIdx(
        makeCanonicalAffineApplies(b, loc, maps[2], allIvs));
    IndexedValueType F(convOp.filter()), I(convOp.input()), O(convOp.output());

    // Emit scalar form.
    ValueHandle paddedInput = getConvOpInput(convOp, I, imIdx);
    O(oIdx) += F(fIdx) * paddedInput;
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, PoolingMaxOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs,
                                       PoolingMaxOp op) {
    auto indices = getInputAndOutputIndices(allIvs, op);
    ValueHandleArray iIdx(indices.first);
    ValueHandleArray oIdx(indices.second);

    // Emit scalar form.
    ValueHandle lhs = std_load(op.output(), oIdx);
    ValueHandle rhs = std_load(op.input(), iIdx);
    using edsc::op::operator>;
    ValueHandle maxValue = std_select(lhs > rhs, lhs, rhs);
    std_store(maxValue, op.output(), oIdx);
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, PoolingMinOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs,
                                       PoolingMinOp op) {
    auto indices = getInputAndOutputIndices(allIvs, op);
    ValueHandleArray iIdx(indices.first);
    ValueHandleArray oIdx(indices.second);

    // Emit scalar form.
    ValueHandle lhs = std_load(op.output(), oIdx);
    ValueHandle rhs = std_load(op.input(), iIdx);
    using edsc::op::operator<;
    ValueHandle minValue = std_select(lhs < rhs, lhs, rhs);
    std_store(minValue, op.output(), oIdx);
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, PoolingSumOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value> allIvs,
                                       PoolingSumOp op) {
    auto indices = getInputAndOutputIndices(allIvs, op);
    SmallVector<ValueHandle, 8> iIdx = indices.first;
    SmallVector<ValueHandle, 8> oIdx = indices.second;
    IndexedValueType input(op.input()), output(op.output());

    // Emit scalar form.
    output(oIdx) += input(iIdx);
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
      ValueHandleArray indexing(makeCanonicalAffineApplies(
          b, loc, genericOp.getInputIndexingMap(i), allIvs));
      indexedValues[i] = std_load(genericOp.getInput(i), indexing);
    }

    // 1.b. Emit std_load from output views.
    // TODO(mravishankar): Avoid the loads if the corresponding argument of the
    // region has no uses.
    for (unsigned i = 0; i < nOutputs; ++i) {
      Value output = genericOp.getOutputBuffer(i);
      ValueHandleArray indexing(makeCanonicalAffineApplies(
          b, loc, genericOp.getOutputIndexingMap(i), allIvs));
      indexedValues[nInputs + i] = std_load(output, indexing);
    }

    // TODO(ntv): When a region inliner exists, use it.
    // 2. Inline region, currently only works for a single basic block.
    // 3. Emit std_store.
    SmallVector<ValueHandleArray, 8> indexing;
    SmallVector<Value, 8> outputBuffers;
    for (unsigned i = 0; i < nOutputs; ++i) {
      indexing.emplace_back(makeCanonicalAffineApplies(
          b, loc, genericOp.getOutputIndexingMap(i), allIvs));
      outputBuffers.push_back(genericOp.getOutputBuffer(i));
    }
    inlineRegionAndEmitStdStore(genericOp, indexedValues, indexing,
                                outputBuffers);
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
      ValueHandleArray indexing(makeCanonicalAffineApplies(
          b, loc, indexedGenericOp.getInputIndexingMap(i), allIvs));
      indexedValues[nLoops + i] = std_load(input, indexing);
    }

    // 1.b. Emit std_load from output views.
    for (unsigned i = 0; i < nOutputs; ++i) {
      Value output = indexedGenericOp.getOutputBuffer(i);
      ValueHandleArray indexing(makeCanonicalAffineApplies(
          b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs));
      indexedValues[nLoops + nInputs + i] = std_load(output, indexing);
    }

    // TODO(ntv): When a region inliner exists, use it.
    // 2. Inline region, currently only works for a single basic block.
    // 3. Emit std_store.
    SmallVector<ValueHandleArray, 8> indexing;
    SmallVector<Value, 8> outputBuffers;
    for (unsigned i = 0; i < nOutputs; ++i) {
      indexing.emplace_back(makeCanonicalAffineApplies(
          b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs));
      outputBuffers.push_back(indexedGenericOp.getOutputBuffer(i));
    }
    inlineRegionAndEmitStdStore(indexedGenericOp, indexedValues, indexing,
                                outputBuffers);
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
template <typename LoopTy, typename ConcreteOpTy>
class LinalgOpToLoopsImpl {
public:
  static Optional<LinalgLoops> doit(Operation *op, PatternRewriter &rewriter);
};

namespace {
/// Helper struct to generate the loop nest for the op. This factored out here
/// to be able to partially specialize this for different LoopTy.
template <typename LoopTy, typename ConcreteOpTy>
class GenerateLoopNest {
public:
  using IndexedValueTy =
      typename std::conditional<std::is_same<LoopTy, AffineForOp>::value,
                                AffineIndexedValue, StdIndexedValue>::type;
  static void doit(ConcreteOpTy linalgOp, ArrayRef<Value> loopRanges,
                   MutableArrayRef<ValueHandle> allIvs) {
    SmallVector<ValueHandle *, 4> allPIvs =
        makeHandlePointers(MutableArrayRef<ValueHandle>(allIvs));

    GenericLoopNestRangeBuilder<LoopTy>(allPIvs, loopRanges)([&] {
      SmallVector<Value, 4> allIvValues(allIvs.begin(), allIvs.end());
      LinalgScopedEmitter<IndexedValueTy,
                          ConcreteOpTy>::emitScalarImplementation(allIvValues,
                                                                  linalgOp);
    });
  }
};

/// Generates loops nest using loop.parallel. loop.parallel is only used for the
/// outer parallel loops. All other loops are generated using loop.for
/// operation.
template <typename ConcreteOpTy>
class GenerateLoopNest<loop::ParallelOp, ConcreteOpTy> {
public:
  using IndexedValueTy = StdIndexedValue;

  static void doit(ConcreteOpTy linalgOp, ArrayRef<Value> loopRanges,
                   MutableArrayRef<ValueHandle> allIvs) {
    // Only generate loop.parallel for outer consecutive "parallel"
    // iterator_types.
    // TODO(ravishankarm): Generate loop.parallel for all "parallel" iterator
    // types, not just the outer most ones. Also handle "reduction" iterator
    // types.
    auto nPar = linalgOp.getNumParallelLoops();
    auto nRed = linalgOp.getNumReductionLoops();
    auto nWin = linalgOp.getNumWindowLoops();
    auto nLoops = nPar + nRed + nWin;
    auto nOuterPar = linalgOp.iterator_types()
                         .getValue()
                         .take_while([](Attribute attr) {
                           return attr.cast<StringAttr>().getValue() ==
                                  getParallelIteratorTypeName();
                         })
                         .size();
    // If there are no outer parallel loops, then number of loop ops is same as
    // the number of loops, and they are all loop.for ops.
    auto nLoopOps = (nOuterPar ? nLoops - nOuterPar + 1 : nLoops);
    SmallVector<ValueHandle *, 4> allPIvs =
        makeHandlePointers(MutableArrayRef<ValueHandle>(allIvs));

    SmallVector<OperationHandle, 4> allLoops(nLoopOps, OperationHandle());
    SmallVector<OperationHandle *, 4> allPLoops;
    allPLoops.reserve(allLoops.size());
    for (OperationHandle &loop : allLoops)
      allPLoops.push_back(&loop);

    ArrayRef<ValueHandle *> allPIvsRef(allPIvs);
    ArrayRef<OperationHandle *> allPLoopsRef(allPLoops);

    if (nOuterPar) {
      GenericLoopNestRangeBuilder<loop::ParallelOp>(
          allPIvsRef.take_front(nOuterPar),
          loopRanges.take_front(nOuterPar))([&] {
        GenericLoopNestRangeBuilder<loop::ForOp>(
            allPIvsRef.drop_front(nOuterPar),
            loopRanges.drop_front(nOuterPar))([&] {
          SmallVector<Value, 4> allIvValues(allIvs.begin(), allIvs.end());
          LinalgScopedEmitter<StdIndexedValue, ConcreteOpTy>::
              emitScalarImplementation(allIvValues, linalgOp);
        });
      });
    } else {
      // If there are no parallel loops then fallback to generating all loop.for
      // operations.
      GenericLoopNestRangeBuilder<loop::ForOp>(allPIvsRef, loopRanges)([&] {
        SmallVector<Value, 4> allIvValues(allIvs.begin(), allIvs.end());
        LinalgScopedEmitter<StdIndexedValue,
                            ConcreteOpTy>::emitScalarImplementation(allIvValues,
                                                                    linalgOp);
      });
    }
  }
};
} // namespace

template <typename LoopTy, typename ConcreteOpTy>
Optional<LinalgLoops>
LinalgOpToLoopsImpl<LoopTy, ConcreteOpTy>::doit(Operation *op,
                                                PatternRewriter &rewriter) {
  using Impl = GenerateLoopNest<LoopTy, ConcreteOpTy>;
  using IndexedValueTy =
      typename GenerateLoopNest<LoopTy, ConcreteOpTy>::IndexedValueTy;

  ScopedContext scope(rewriter, op->getLoc());

  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  auto linalgOp = cast<ConcreteOpTy>(op);
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto nPar = linalgOp.getNumParallelLoops();
  auto nRed = linalgOp.getNumReductionLoops();
  auto nWin = linalgOp.getNumWindowLoops();
  auto nLoops = nPar + nRed + nWin;
  auto mapsRange =
      linalgOp.indexing_maps().template getAsRange<AffineMapAttr>();
  auto maps = llvm::to_vector<8>(
      llvm::map_range(mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
  AffineMap invertedMap = inversePermutation(concatAffineMaps(maps));
  if (!invertedMap)
    return {};
  if (invertedMap.isEmpty()) {
    LinalgScopedEmitter<IndexedValueTy, ConcreteOpTy>::emitScalarImplementation(
        {}, linalgOp);
    return LinalgLoops();
  }

  SmallVector<ValueHandle, 4> allIvs(nLoops,
                                     ValueHandle(rewriter.getIndexType()));
  auto loopRanges =
      emitLoopRanges(scope.getBuilder(), scope.getLocation(), invertedMap,
                     getViewSizes(rewriter, linalgOp));
  assert(loopRanges.size() == allIvs.size());
  Impl::doit(linalgOp, loopRanges, allIvs);
  // Number of loop ops might be different from the number of ivs since some
  // loops like affine.parallel and loop.parallel have multiple ivs.
  llvm::SetVector<Operation *> loopSet;
  for (ValueHandle &iv : allIvs) {
    if (!iv.hasValue())
      return {};
    // The induction variable is a block argument of the entry block of the
    // loop operation.
    BlockArgument ivVal = iv.getValue().dyn_cast<BlockArgument>();
    if (!ivVal)
      return {};
    loopSet.insert(ivVal.getOwner()->getParentOp());
  }
  LinalgLoops loops(loopSet.begin(), loopSet.end());
  return loops;
}

template <typename LoopType, typename ConcreteOp>
class LinalgRewritePattern : public RewritePattern {
public:
  explicit LinalgRewritePattern(MLIRContext *context)
      : RewritePattern(ConcreteOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    using Impl = LinalgOpToLoopsImpl<LoopType, ConcreteOp>;
    if (!Impl::doit(op, rewriter))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

// Helper classes for type list expansion.
template <typename LoopType, typename... LinalgOps>
class RewritePatternList;

template <typename LoopType>
class RewritePatternList<LoopType> {
public:
  static void build(OwningRewritePatternList &patterns, MLIRContext *ctx) {}
};

template <typename LoopType, typename ConcreteOp, typename... LinalgOps>
class RewritePatternList<LoopType, ConcreteOp, LinalgOps...> {
public:
  static void build(OwningRewritePatternList &patterns, MLIRContext *ctx) {
    patterns.insert<LinalgRewritePattern<LoopType, ConcreteOp>>(ctx);
    RewritePatternList<LoopType, LinalgOps...>::build(patterns, ctx);
  }
};

/// Populate the given list with patterns that convert from Linalg to LLVM.
template <typename LoopType>
void FillRewritePatterns(OwningRewritePatternList &patterns, MLIRContext *ctx) {
  RewritePatternList<LoopType,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                     >::build(patterns, ctx);
}

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

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    AffineApplyOp affineApplyOp = cast<AffineApplyOp>(op);
    auto map = affineApplyOp.getAffineMap();
    if (map.getNumResults() != 1 || map.getNumInputs() > 1)
      return failure();

    AffineExpr expr = map.getResult(0);
    if (map.getNumInputs() == 0) {
      if (auto val = expr.dyn_cast<AffineConstantExpr>()) {
        rewriter.replaceOpWithNewOp<ConstantIndexOp>(op, val.getValue());
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
} // namespace

template <typename LoopType>
static void lowerLinalgToLoopsImpl(Operation *op, MLIRContext *context) {
  OwningRewritePatternList patterns;
  // Canonicalization and folding patterns applied greedily allow cleaning up
  // the emitted IR on the fly.
  // TODO(ntv) fold view and subview ops?
  FillRewritePatterns<LoopType>(patterns, context);
  DimOp::getCanonicalizationPatterns(patterns, context);
  AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  patterns.insert<FoldAffineOp>(context);
  // Just apply the patterns greedily.
  applyPatternsAndFoldGreedily(op, patterns);
}

namespace {
struct LowerToAffineLoops
    : public LinalgLowerToAffineLoopsBase<LowerToAffineLoops> {
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<AffineForOp>(getFunction(), &getContext());
  }
};
struct LowerToLoops : public LinalgLowerToLoopsBase<LowerToLoops> {
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<loop::ForOp>(getFunction(), &getContext());
  }
};
struct LowerToParallelLoops
    : public LinalgLowerToParallelLoopsBase<LowerToParallelLoops> {
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<loop::ParallelOp>(getFunction(), &getContext());
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createConvertLinalgToLoopsPass() {
  return std::make_unique<LowerToLoops>();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertLinalgToParallelLoopsPass() {
  return std::make_unique<LowerToParallelLoops>();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertLinalgToAffineLoopsPass() {
  return std::make_unique<LowerToAffineLoops>();
}

/// Emits a loop nest with the proper body for `op`.
template <typename LoopTy, typename ConcreteOp>
Optional<LinalgLoops>
mlir::linalg::linalgLowerOpToLoops(PatternRewriter &rewriter, Operation *op) {
  return LinalgOpToLoopsImpl<LoopTy, ConcreteOp>::doit(op, rewriter);
}

/// Emits a loop nest of `loop.for` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult mlir::linalg::linalgOpToLoops(PatternRewriter &rewriter,
                                            Operation *op) {
  Optional<LinalgLoops> loops =
      linalgLowerOpToLoops<loop::ForOp, ConcreteOp>(rewriter, op);
  return loops ? success() : failure();
}

/// Emits a loop nest of `affine.for` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult mlir::linalg::linalgOpToAffineLoops(PatternRewriter &rewriter,
                                                  Operation *op) {
  Optional<LinalgLoops> loops =
      linalgLowerOpToLoops<AffineForOp, ConcreteOp>(rewriter, op);
  return loops ? success() : failure();
}

/// Emits a loop nest of `loop.parallel` with the proper body for `op`.
template <typename ConcreteOp>
LogicalResult mlir::linalg::linalgOpToParallelLoops(PatternRewriter &rewriter,
                                                    Operation *op) {
  Optional<LinalgLoops> loops =
      linalgLowerOpToLoops<loop::ParallelOp, ConcreteOp>(rewriter, op);
  return loops ? success() : failure();
}

// TODO(ntv) Need to make these instantiations more future-proof to avoid the
// need to update as soon as we add new ops.
#define INSTANTIATE_LINALG_OP_TO_LOOPS(OP_TYPE)                                \
  template LogicalResult mlir::linalg::linalgOpToLoops<OP_TYPE>(               \
      PatternRewriter & rewriter, Operation * op);                             \
  template LogicalResult mlir::linalg::linalgOpToAffineLoops<OP_TYPE>(         \
      PatternRewriter & rewriter, Operation * op);                             \
  template LogicalResult mlir::linalg::linalgOpToParallelLoops<OP_TYPE>(       \
      PatternRewriter & rewriter, Operation * op);                             \
  template Optional<LinalgLoops>                                               \
      mlir::linalg::linalgLowerOpToLoops<loop::ParallelOp, OP_TYPE>(           \
          PatternRewriter & rewriter, Operation * op);

INSTANTIATE_LINALG_OP_TO_LOOPS(CopyOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(FillOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(DotOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(MatvecOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(MatmulOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(ConvOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(PoolingMaxOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(PoolingMinOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(PoolingSumOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(GenericOp)
INSTANTIATE_LINALG_OP_TO_LOOPS(IndexedGenericOp)
