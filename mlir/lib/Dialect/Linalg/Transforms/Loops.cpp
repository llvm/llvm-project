//===- Loops.cpp - conversion from Linalg named and generic ops to loops --===//
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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
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

static SmallVector<Value, 8> makeCanonicalAffineApplies(OpBuilder &b,
                                                        Location loc,
                                                        AffineMap map,
                                                        ArrayRef<Value> vals) {
  if (map.isEmpty())
    return {};

  assert(map.getNumInputs() == vals.size());
  SmallVector<Value, 8> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, map.getNumSymbols(), e);
    SmallVector<Value, 4> operands(vals.begin(), vals.end());
    canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(affine_apply(exprMap, operands));
  }
  return res;
}

static SmallVector<Value, 4> permuteIvs(ArrayRef<Value> ivs,
                                        Optional<AffineMap> permutation) {
  return permutation ? applyMapToValues(ScopedContext::getBuilderRef(),
                                        ScopedContext::getLocation(),
                                        permutation.getValue(), ivs)
                     : SmallVector<Value, 4>(ivs.begin(), ivs.end());
}

/// Creates a number of ranges equal to the number of dimensions in the `map`.
/// The returned ranges correspond to the loop ranges, in the proper order, for
/// which new loops will be created.
/// The function supports only maps that are invertible and have results of type
/// DimExpr or (DimExpr + DimExpr - SymbolExpr floordiv ConstExpr).
/// It expects a non-inverted, concatenated map and last values in
/// allViewSizes will be applied to the symbols in the map if it contains any.
static SmallVector<SubViewOp::Range, 4> emitLoopRanges(OpBuilder &b,
                                                       Location loc,
                                                       AffineMap map,
                                                       ValueRange viewSizes) {
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  unsigned numSym = map.getNumSymbols();
  assert(viewSizes.size() == numRes + numSym &&
         "viewSizes must contain sizes of all views and values for symbols");
  SmallVector<SubViewOp::Range, 4> res(numDims);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = result.dyn_cast<AffineDimExpr>()) {
      if (res[d.getPosition()].offset)
        continue;
      res[d.getPosition()] = SubViewOp::Range{
          std_constant_index(0), viewSizes[idx], std_constant_index(1)};
    }

    // If the access pattern is of form (m, n)[s] -> (m + n - s floordiv 2),
    // then the bounds are:
    //   (s floordiv 2) <= m <= (size(m) + s floordiv 2 - s + 1).
    // where size(n) is applied to the symbol s.
    // This is done statically now.
    if (auto binOp = result.dyn_cast<AffineBinaryOpExpr>()) {
      auto lhs = binOp.getLHS().dyn_cast<AffineBinaryOpExpr>();
      auto rhs = binOp.getRHS().dyn_cast<AffineBinaryOpExpr>();
      if (!lhs || !rhs || binOp.getKind() != AffineExprKind::Add ||
          lhs.getKind() != AffineExprKind::Add ||
          rhs.getKind() != mlir::AffineExprKind::Mul)
        continue;

      auto m = lhs.getLHS().dyn_cast<AffineDimExpr>();
      auto n = lhs.getRHS().dyn_cast<AffineDimExpr>();
      auto fDiv = rhs.getLHS().dyn_cast<AffineBinaryOpExpr>();
      auto minusOne = rhs.getRHS().dyn_cast<AffineConstantExpr>();
      if (!m || !n || !fDiv || !minusOne ||
          fDiv.getKind() != AffineExprKind::FloorDiv ||
          fDiv.getLHS().getKind() != AffineExprKind::SymbolId ||
          fDiv.getRHS().getKind() != AffineExprKind::Constant)
        continue;

      auto s = fDiv.getLHS().dyn_cast<AffineSymbolExpr>();
      if (minusOne.getValue() != -1)
        continue;

      int mPos = m.getPosition();
      AffineExpr one = getAffineConstantExpr(1, s.getContext());
      AffineExpr sizeOfM = getAffineSymbolExpr(numSym, s.getContext());
      // Construction of upper bound (size(m) + s floordiv 2 - s + 1).
      AffineExpr upperOffsetExpr = sizeOfM + fDiv + one - s;
      AffineMap fromMap = AffineMap::get(numDims, numSym + 1, fDiv);
      AffineMap toMap = AffineMap::get(numDims, numSym + 1, upperOffsetExpr);
      SmallVector<Value, 8> values(viewSizes.begin(),
                                   viewSizes.begin() + numDims);
      values.insert(values.end(), viewSizes.begin() + numRes, viewSizes.end());
      values.push_back(viewSizes[mPos]);
      // Construction of the lower bound (s floordiv 2).
      Value from = applyMapToValues(b, loc, fromMap, values).front();
      Value to = applyMapToValues(b, loc, toMap, values).front();
      res[mPos] = SubViewOp::Range{from, to, std_constant_index(1)};
    }
  }
  return res;
}

template <typename IndexedValueType, typename OpType>
static void inlineRegionAndEmitStore(OpType op, ArrayRef<Value> indexedValues,
                                     ArrayRef<SmallVector<Value, 8>> indexing,
                                     ArrayRef<Value> outputBuffers) {
  assert(op.getOperation()->getNumRegions() == 1 &&
         "Expected single region op");
  auto &b = ScopedContext::getBuilderRef();
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
         "expected a yield op in the end of the region");
  for (unsigned i = 0, e = terminator.getNumOperands(); i < e; ++i) {
    IndexedValueType O(outputBuffers[i]);
    O(indexing[i]) = map.lookupOrDefault(terminator.getOperand(i));
  }
}

// Returns a pair that contains input indices and output indices of a
// SingleInputPoolingOp `op`.
struct InputAndOutputIndices {
  SmallVector<Value, 8> inputs;
  SmallVector<Value, 8> outputs;
};
template <typename SingleInputPoolingOp>
static InputAndOutputIndices getInputAndOutputIndices(ArrayRef<Value> allIvs,
                                                      SingleInputPoolingOp op) {
  auto &b = ScopedContext::getBuilderRef();
  auto loc = ScopedContext::getLocation();
  auto mapsRange = op.indexing_maps().template getAsRange<AffineMapAttr>();
  auto maps = llvm::to_vector<8>(
      llvm::map_range(mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
  return InputAndOutputIndices{
      makeCanonicalAffineApplies(b, loc, maps[0], allIvs),
      makeCanonicalAffineApplies(b, loc, maps[2], allIvs)};
}

namespace {

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
// TODO: need a LinalgStructuredOpInterface.
template <typename IndexedValueType, typename LinalgStructuredOpType>
void emitScalarImplementation(ArrayRef<Value> allIvs,
                              LinalgStructuredOpType linalgOp) {
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto &b = ScopedContext::getBuilderRef();
  auto loc = ScopedContext::getLocation();
  unsigned nInputs = linalgOp.getNumInputs();
  unsigned nOutputs = linalgOp.getNumOutputs();
  SmallVector<Value, 4> indexedValues;
  indexedValues.reserve(nInputs + nOutputs);

  auto attr = linalgOp.template getAttrOfType<IntegerAttr>("symbol_source");
  auto allIvsPlusDims = SmallVector<Value, 4>(allIvs.begin(), allIvs.end());
  if (attr) {
    auto operand = linalgOp.getOperand(attr.getInt());
    auto shapedType = operand.getType().template cast<ShapedType>();
    allIvsPlusDims.reserve(allIvs.size() + shapedType.getRank());
    for (unsigned idx = 0, e = shapedType.getRank(); idx < e; ++idx)
      allIvsPlusDims.push_back(b.create<DimOp>(loc, operand, idx));
  }

  // TODO: Avoid the loads if the corresponding argument of the
  // region has no uses.
  // 1.a. Emit load from input views.
  for (unsigned i = 0; i < nInputs; ++i) {
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getInputIndexingMap(i), allIvsPlusDims);
    // Passing through IndexedValueType emits the proper load operation.
    indexedValues.push_back(IndexedValueType(linalgOp.getInput(i))(indexing));
  }
  // 1.b. Emit load from output views.
  for (unsigned i = 0; i < nOutputs; ++i) {
    auto indexing = makeCanonicalAffineApplies(
        b, loc, linalgOp.getOutputIndexingMap(i), allIvsPlusDims);
    // Passing through IndexedValueType emits the proper load operation.
    indexedValues.push_back(
        IndexedValueType(linalgOp.getOutputBuffer(i))(indexing));
  }

  // TODO: When a region inliner exists, use it.
  // 2. Inline region, currently only works for a single basic block.
  // 3. Emit store.
  SmallVector<SmallVector<Value, 8>, 8> indexing;
  SmallVector<Value, 8> outputBuffers;
  for (unsigned i = 0; i < nOutputs; ++i) {
    indexing.push_back(makeCanonicalAffineApplies(
        b, loc, linalgOp.getOutputIndexingMap(i), allIvsPlusDims));
    outputBuffers.push_back(linalgOp.getOutputBuffer(i));
  }
  inlineRegionAndEmitStore<IndexedValueType>(linalgOp, indexedValues, indexing,
                                             outputBuffers);
}

template <typename IndexedValueType>
void emitScalarImplementation(ArrayRef<Value> allIvs, CopyOp copyOp) {
  assert(copyOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto nPar = copyOp.getNumParallelLoops();
  assert(nPar == allIvs.size());
  auto inputIvs =
      permuteIvs(allIvs.take_front(nPar), copyOp.inputPermutation());
  auto outputIvs =
      permuteIvs(allIvs.take_front(nPar), copyOp.outputPermutation());
  SmallVector<Value, 8> iivs(inputIvs.begin(), inputIvs.end());
  SmallVector<Value, 8> oivs(outputIvs.begin(), outputIvs.end());
  IndexedValueType O(copyOp.getOutputBuffer(0)), I(copyOp.getInput(0));
  // Emit the proper scalar assignment, whether we are dealing with a 0-D or
  // an n-D loop nest; with or without permutations.
  // clang-format off
    nPar > 0 ? O(oivs) = I(iivs) :
               O() = I();
  // clang-format on
}

template <typename IndexedValueType>
void emitScalarImplementation(ArrayRef<Value> allIvs, FillOp fillOp) {
  assert(fillOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto nPar = fillOp.getNumParallelLoops();
  assert(nPar == allIvs.size());
  auto ivs = SmallVector<Value, 4>(allIvs.begin(), allIvs.begin() + nPar);
  IndexedValueType O(fillOp.getOutputBuffer(0));
  // Emit the proper scalar assignment, whether we are dealing with a 0-D or
  // an n-D loop nest; with or without permutations.
  nPar > 0 ? O(ivs) = fillOp.value() : O() = fillOp.value();
}

template <typename IndexedValueType>
Value getConvOpInput(ConvOp convOp, StdIndexedValue im,
                     MutableArrayRef<Value> imIdx) {
  // TODO: add a level of indirection to linalg.generic.
  if (!convOp.padding())
    return im(imIdx);

  auto *context = ScopedContext::getContext();
  Value zeroIndex = std_constant_index(0);
  SmallVector<Value, 8> conds;
  SmallVector<Value, 8> clampedImIdx;
  for (auto iter : llvm::enumerate(imIdx)) {
    int idx = iter.index();
    auto dim = iter.value();
    // Only need to iterate over the window dimensions.
    if (idx == 0 || idx == static_cast<int>(imIdx.size()) - 1) {
      clampedImIdx.push_back(dim);
      continue;
    }

    using edsc::op::sge;
    using edsc::op::slt;
    using edsc::op::operator||;
    Value leftOutOfBound = slt(dim, zeroIndex);
    if (conds.empty())
      conds.push_back(leftOutOfBound);
    else
      conds.push_back(conds.back() || leftOutOfBound);
    Value rightBound = std_dim(convOp.input(), idx);
    conds.push_back(conds.back() || (sge(dim, rightBound)));

    // When padding is involved, the indices will only be shifted to negative,
    // so having a max op is enough.
    auto maxMap = AffineMap::get(/*dimCount=*/1, 0,
                                 {getAffineDimExpr(/*position=*/0, context),
                                  getAffineConstantExpr(0, context)},
                                 context);
    clampedImIdx.push_back(affine_max(dim.getType(), maxMap, ValueRange{dim}));
  }

  auto &b = ScopedContext::getBuilderRef();
  Type type = convOp.input().getType().cast<MemRefType>().getElementType();
  Value zero = std_constant(type, b.getZeroAttr(type));
  Value readInput = im(clampedImIdx);
  return conds.empty() ? readInput
                       : (Value)std_select(conds.back(), zero, readInput);
}

/// Returns true is `convOp` has a non-zero padding.
static bool hasPadding(ConvOp convOp) {
  for (unsigned i = 0, e = convOp.getNumSpatialDimensions(); i < e; ++i) {
    if (convOp.getLowPad(i) > 0 || convOp.getHighPad(i) > 0)
      return true;
  }
  return false;
}

template <typename IndexedValueType>
static void emitScalarImplementation(ArrayRef<Value> allIvs, ConvOp convOp) {
  assert(convOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto &b = ScopedContext::getBuilderRef();
  auto loc = ScopedContext::getLocation();
  auto mapsRange = convOp.indexing_maps().getAsRange<AffineMapAttr>();
  auto maps = llvm::to_vector<8>(
      llvm::map_range(mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
  SmallVector<Value, 8> fIdx(
      makeCanonicalAffineApplies(b, loc, maps[0], allIvs));
  SmallVector<Value, 8> imIdx(
      makeCanonicalAffineApplies(b, loc, maps[1], allIvs));
  SmallVector<Value, 8> oIdx(
      makeCanonicalAffineApplies(b, loc, maps[2], allIvs));

  IndexedValueType F(convOp.filter()), O(convOp.output());

  // Emit scalar form. Padded conv involves an affine.max in the memory access
  // which is not allowed by affine.load. Override to use an StdIndexedValue
  // when there is non-zero padding.
  if (hasPadding(convOp)) {
    StdIndexedValue I(convOp.input());
    Value paddedInput = getConvOpInput<IndexedValueType>(convOp, I, imIdx);
    O(oIdx) += F(fIdx) * paddedInput;
  } else {
    IndexedValueType I(convOp.input());
    O(oIdx) += F(fIdx) * I(imIdx);
  }
}

template <typename IndexedValueType>
void emitScalarImplementation(ArrayRef<Value> allIvs, PoolingMaxOp op) {
  InputAndOutputIndices indices = getInputAndOutputIndices(allIvs, op);
  // Emit scalar form.
  IndexedValueType output(op.output());
  IndexedValueType input(op.input());
  Value lhs = output(indices.outputs);
  Value rhs = input(indices.inputs);
  using edsc::op::sgt;
  Value maxValue = std_select(sgt(lhs, rhs), lhs, rhs);
  output(indices.outputs) = maxValue;
}

template <typename IndexedValueType>
void emitScalarImplementation(ArrayRef<Value> allIvs, PoolingMinOp op) {
  InputAndOutputIndices indices = getInputAndOutputIndices(allIvs, op);
  // Emit scalar form.
  IndexedValueType output(op.output());
  IndexedValueType input(op.input());
  Value lhs = output(indices.outputs);
  Value rhs = input(indices.inputs);
  using edsc::op::slt;
  Value minValue = std_select(slt(lhs, rhs), lhs, rhs);
  output(indices.outputs) = minValue;
}
template <typename IndexedValueType>
void emitScalarImplementation(ArrayRef<Value> allIvs, PoolingSumOp op) {
  auto indices = getInputAndOutputIndices(allIvs, op);
  IndexedValueType input(op.input()), output(op.output());

  // Emit scalar form.
  output(indices.outputs) += input(indices.inputs);
}
/// Emits the MLIR for the scalar part of the indexed generic op by:
///   1. Emitting load ops for each input and output view in order. This is
///      achieved by applying the appropriate input or output map to the
///      enclosing induction variables.
///   2. Emitting a call to `op.fun()` that takes as arguments the induction
///      variables and the scalars from point 1. above.
///   3. Emitting store ops to store the results of 2. to the output views.
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
///          %14:2 = call @foo(%i, %j, %k, %11, %12, %13) :
///            (index, index, index, f32, f32, f32) -> (f32, f32)
///          store %14#0, %arg1[%i, %j, %k] :
///            memref<?x?x?Xf32, stride_specification>
///          store %14#1, %arg2[%i, %k, %j] :
///            memref<?x?x?Xf32, stride_specification>
///       }
///      }
///    }
/// ```
template <typename IndexedValueType>
static void emitScalarImplementation(ArrayRef<Value> allIvs,
                                     IndexedGenericOp indexedGenericOp) {
  assert(indexedGenericOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto &b = ScopedContext::getBuilderRef();
  auto loc = ScopedContext::getLocation();
  unsigned nInputs = indexedGenericOp.getNumInputs();
  unsigned nOutputs = indexedGenericOp.getNumOutputs();
  unsigned nLoops = allIvs.size();
  SmallVector<Value, 4> indexedValues;
  indexedValues.reserve(nLoops + nInputs + nOutputs);
  for (unsigned i = 0; i < nLoops; ++i)
    indexedValues.push_back(allIvs[i]);

  // TODO: Avoid the loads if the corresponding argument of the
  // region has no uses.
  // 1.a. Emit load from input views.
  for (unsigned i = 0; i < nInputs; ++i) {
    auto indexing = makeCanonicalAffineApplies(
        b, loc, indexedGenericOp.getInputIndexingMap(i), allIvs);
    // Pass input i through IndexedValueType emits the proper load operation.
    indexedValues.push_back(
        IndexedValueType(indexedGenericOp.getInput(i))(indexing));
  }
  // 1.b. Emit load from output views.
  for (unsigned i = 0; i < nOutputs; ++i) {
    auto indexing = makeCanonicalAffineApplies(
        b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs);
    // Pass output i through IndexedValueType emits the proper load operation.
    indexedValues.push_back(
        IndexedValueType(indexedGenericOp.getOutputBuffer(i))(indexing));
  }

  // TODO: When a region inliner exists, use it.
  // 2. Inline region, currently only works for a single basic block.
  // 3. Emit store.
  SmallVector<SmallVector<Value, 8>, 8> indexing;
  SmallVector<Value, 8> outputBuffers;
  for (unsigned i = 0; i < nOutputs; ++i) {
    indexing.push_back(makeCanonicalAffineApplies(
        b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs));
    outputBuffers.push_back(indexedGenericOp.getOutputBuffer(i));
  }
  inlineRegionAndEmitStore<IndexedValueType>(indexedGenericOp, indexedValues,
                                             indexing, outputBuffers);
}

template <typename LoopTy, typename ConcreteOpTy>
Optional<LinalgLoops> linalgOpToLoopsImpl(Operation *op, OpBuilder &builder) {
  using IndexedValueTy = typename GenerateLoopNest<LoopTy>::IndexedValueTy;

  ScopedContext scope(builder, op->getLoc());

  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  auto linalgOp = cast<ConcreteOpTy>(op);
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto mapsRange =
      linalgOp.indexing_maps().template getAsRange<AffineMapAttr>();
  auto maps = llvm::to_vector<8>(
      llvm::map_range(mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
  SmallVector<Value, 8> sizes = getViewSizes(builder, linalgOp);
  AffineMap map = concatAffineMaps(maps);
  auto loopRanges = emitLoopRanges(scope.getBuilderRef(), scope.getLocation(),
                                   map, getViewSizes(builder, linalgOp));
  SmallVector<Value, 4> allIvs;
  GenerateLoopNest<LoopTy>::doit(
      loopRanges, linalgOp.iterator_types().getValue(), [&](ValueRange ivs) {
        allIvs.append(ivs.begin(), ivs.end());
        emitScalarImplementation<IndexedValueTy>(allIvs, linalgOp);
      });
  // Number of loop ops might be different from the number of ivs since some
  // loops like affine.parallel and scf.parallel have multiple ivs.
  llvm::SetVector<Operation *> loopSet;
  for (Value iv : allIvs) {
    if (!iv)
      return {};
    // The induction variable is a block argument of the entry block of the
    // loop operation.
    BlockArgument ivVal = iv.dyn_cast<BlockArgument>();
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
    if (!linalgOpToLoopsImpl<LoopType, ConcreteOp>(op, rewriter))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename LoopType, typename ConcreteOp>
void insertOnePattern(OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<LinalgRewritePattern<LoopType, ConcreteOp>>(ctx);
}

template <typename LoopType, typename... Args>
void insertPatterns(OwningRewritePatternList &patterns, MLIRContext *ctx) {
  (void)std::initializer_list<int>{
      0, (insertOnePattern<LoopType, Args>(patterns, ctx), 0)...};
}

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
static void lowerLinalgToLoopsImpl(FuncOp funcOp, MLIRContext *context) {
  OwningRewritePatternList patterns;
  // Canonicalization and folding patterns applied greedily allow cleaning up
  // the emitted IR on the fly.
  // TODO: fold view and subview ops?
  insertPatterns<LoopType,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                 >(patterns, context);

  DimOp::getCanonicalizationPatterns(patterns, context);
  AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  patterns.insert<FoldAffineOp>(context);
  // Just apply the patterns greedily.
  applyPatternsAndFoldGreedily(funcOp, patterns);
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
    lowerLinalgToLoopsImpl<scf::ForOp>(getFunction(), &getContext());
  }
};
struct LowerToParallelLoops
    : public LinalgLowerToParallelLoopsBase<LowerToParallelLoops> {
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<scf::ParallelOp>(getFunction(), &getContext());
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

// TODO: gradually remove this layer as more ops become "named".
template <typename LoopTy>
static Optional<LinalgLoops> linalgOpToLoopsImplSwitch(Operation *op,
                                                       OpBuilder &builder) {
  assert(isa<LinalgOp>(op) && "LinalgOp expected");
  if (isa<CopyOp>(op))
    return linalgOpToLoopsImpl<LoopTy, CopyOp>(op, builder);
  if (isa<FillOp>(op))
    return linalgOpToLoopsImpl<LoopTy, FillOp>(op, builder);
  if (isa<ConvOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvOp>(op, builder);
  if (isa<PoolingMaxOp>(op))
    return linalgOpToLoopsImpl<LoopTy, PoolingMaxOp>(op, builder);
  if (isa<PoolingMinOp>(op))
    return linalgOpToLoopsImpl<LoopTy, PoolingMinOp>(op, builder);
  if (isa<PoolingSumOp>(op))
    return linalgOpToLoopsImpl<LoopTy, PoolingSumOp>(op, builder);
  if (isa<IndexedGenericOp>(op))
    return linalgOpToLoopsImpl<LoopTy, IndexedGenericOp>(op, builder);

  // TODO: Cases below are generic and need a LinalgStructuredOpInterface.
  if (isa<GenericOp>(op))
    return linalgOpToLoopsImpl<LoopTy, GenericOp>(op, builder);
  if (isa<MatmulOp>(op))
    return linalgOpToLoopsImpl<LoopTy, MatmulOp>(op, builder);
  if (isa<MatvecOp>(op))
    return linalgOpToLoopsImpl<LoopTy, MatvecOp>(op, builder);
  if (isa<DotOp>(op))
    return linalgOpToLoopsImpl<LoopTy, DotOp>(op, builder);
  if (isa<BatchMatmulOp>(op))
    return linalgOpToLoopsImpl<LoopTy, BatchMatmulOp>(op, builder);
  if (isa<ConvWOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvWOp>(op, builder);
  if (isa<ConvNWCOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvNWCOp>(op, builder);
  if (isa<ConvNCWOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvNCWOp>(op, builder);
  if (isa<ConvHWOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvHWOp>(op, builder);
  if (isa<ConvNHWCOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvNHWCOp>(op, builder);
  if (isa<ConvNCHWOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvNCHWOp>(op, builder);
  if (isa<ConvDHWOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvDHWOp>(op, builder);
  if (isa<ConvNDHWCOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvNDHWCOp>(op, builder);
  if (isa<ConvNCDHWOp>(op))
    return linalgOpToLoopsImpl<LoopTy, ConvNCDHWOp>(op, builder);
  llvm_unreachable("Unexpected op in linalgOpToLoopsImpl");
}

/// Emits a loop nest with the proper body for `op`.
template <typename LoopTy>
Optional<LinalgLoops> mlir::linalg::linalgLowerOpToLoops(OpBuilder &builder,
                                                         Operation *op) {
  return linalgOpToLoopsImplSwitch<LoopTy>(op, builder);
}

template Optional<LinalgLoops>
mlir::linalg::linalgLowerOpToLoops<AffineForOp>(OpBuilder &builder,
                                                Operation *op);
template Optional<LinalgLoops>
mlir::linalg::linalgLowerOpToLoops<scf::ForOp>(OpBuilder &builder,
                                               Operation *op);
template Optional<LinalgLoops>
mlir::linalg::linalgLowerOpToLoops<scf::ParallelOp>(OpBuilder &builder,
                                                    Operation *op);

/// Emits a loop nest of `affine.for` with the proper body for `op`.
LogicalResult mlir::linalg::linalgOpToAffineLoops(OpBuilder &builder,
                                                  Operation *op) {
  Optional<LinalgLoops> loops = linalgLowerOpToLoops<AffineForOp>(builder, op);
  return loops ? success() : failure();
}

/// Emits a loop nest of `scf.for` with the proper body for `op`.
LogicalResult mlir::linalg::linalgOpToLoops(OpBuilder &builder, Operation *op) {
  Optional<LinalgLoops> loops = linalgLowerOpToLoops<scf::ForOp>(builder, op);
  return loops ? success() : failure();
}

/// Emits a loop nest of `scf.parallel` with the proper body for `op`.
LogicalResult mlir::linalg::linalgOpToParallelLoops(OpBuilder &builder,
                                                    Operation *op) {
  Optional<LinalgLoops> loops =
      linalgLowerOpToLoops<scf::ParallelOp>(builder, op);
  return loops ? success() : failure();
}
