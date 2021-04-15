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
#include "mlir/Dialect/MemRef/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

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

template <typename IndexedValueType, typename OpType>
static void inlineRegionAndEmitStore(OpType op, ArrayRef<Value> indexedValues,
                                     ArrayRef<SmallVector<Value, 8>> indexing,
                                     ArrayRef<Value> outputBuffers) {
  assert(op->getNumRegions() == 1 && "Expected single region op");
  auto &b = ScopedContext::getBuilderRef();
  auto &block = op->getRegion(0).front();
  BlockAndValueMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    auto *newOp = b.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }

  Operation &terminator = block.back();
  assert(isa<linalg::YieldOp>(terminator) &&
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
template <typename IndexedValueType>
static void emitScalarImplementation(ArrayRef<Value> allIvs,
                                     LinalgOp linalgOp) {
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto &b = ScopedContext::getBuilderRef();
  auto loc = ScopedContext::getLocation();
  unsigned nInputs = linalgOp.getNumInputs();
  unsigned nOutputs = linalgOp.getNumOutputs();
  SmallVector<Value, 4> indexedValues;
  indexedValues.reserve(nInputs + nOutputs);

  auto allIvsPlusDims = SmallVector<Value, 4>(allIvs.begin(), allIvs.end());

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

// Create a padded view into the given `input` tensor using the 'indices'
// to access the tensor. `skipPadding` lists the dimensions for which no padding
// is needed e.g. the non-spatial dimensions for convolutions.
template <typename IndexedValueType>
Value getPaddedInput(Value input, ArrayRef<Value> indices,
                     ArrayRef<int> skipPadding, Value padValue) {
  // TODO: add a level of indirection to linalg.generic.

  IndexedValueType indexedInput(input);

  auto *context = ScopedContext::getContext();
  Value zeroIndex = std_constant_index(0);
  SmallVector<Value, 8> conds;
  SmallVector<Value, 8> clampedImIdx;
  for (auto iter : llvm::enumerate(indices)) {
    int idx = iter.index();
    auto dim = iter.value();
    if (is_contained(skipPadding, idx)) {
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
    Value rightBound = memref_dim(input, idx);
    conds.push_back(conds.back() || (sge(dim, rightBound)));

    // When padding is involved, the indices will only be shifted to negative,
    // so having a max op is enough.
    auto maxMap = AffineMap::get(/*dimCount=*/1, 0,
                                 {getAffineDimExpr(/*position=*/0, context),
                                  getAffineConstantExpr(0, context)},
                                 context);
    clampedImIdx.push_back(affine_max(dim.getType(), maxMap, ValueRange{dim}));
  }

  Value readInput = indexedInput(clampedImIdx);
  return conds.empty() ? readInput
                       : (Value)std_select(conds.back(), padValue, readInput);
}

namespace {

/// The padding value for a given Op depends on the semantics of the Op.
/// The identity value for ConvOp and PoolingSumOp is 0, for PoolingMaxOp is
/// -inf or minInt and for PoolingMinOp is inf or maxInt.
template <typename OpType> Attribute getPadValueAttr(Type type) {
  llvm_unreachable("Unexpected op type for getPadValueAttr");
  return {};
}

template <> Attribute getPadValueAttr<PoolingMaxOp>(Type type) {
  auto &b = ScopedContext::getBuilderRef();
  if (auto floatType = type.dyn_cast<FloatType>()) {
    return b.getFloatAttr(
        floatType,
        APFloat::getInf(floatType.getFloatSemantics(), /*Negative*/ true));
  }
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    // The select instruction used to lower the PoolingMin uses a signed
    // comparison, use a signed constant irrespective of the signedness of the
    // integer type.
    return b.getIntegerAttr(intType, APInt::getSignedMinValue(width));
  }
  llvm_unreachable("Unsupported data type for PoolingMaxOp");
  return {};
}

template <> Attribute getPadValueAttr<PoolingMinOp>(Type type) {
  auto &b = ScopedContext::getBuilderRef();
  if (auto floatType = type.dyn_cast<FloatType>()) {
    return b.getFloatAttr(floatType,
                          APFloat::getInf(floatType.getFloatSemantics()));
  }
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    // The select instruction used to lower the PoolingMin uses a signed
    // comparison, use a signed constant irrespective of the signedness of the
    // integer type.
    return b.getIntegerAttr(intType, APInt::getSignedMaxValue(width));
  }
  llvm_unreachable("Unsupported data type for PoolingMinOp");
  return {};
}

template <> Attribute getPadValueAttr<PoolingSumOp>(Type type) {
  auto &b = ScopedContext::getBuilderRef();
  return b.getZeroAttr(type);
}

template <> Attribute getPadValueAttr<ConvOp>(Type type) {
  auto &b = ScopedContext::getBuilderRef();
  return b.getZeroAttr(type);
}

} // namespace

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
  // which is not allowed by affine.load. Override to use an MemRefIndexedValue
  // when there is non-zero padding.
  if (hasPadding(convOp)) {
    Type type = convOp.input().getType().cast<MemRefType>().getElementType();
    Value padValue = std_constant(type, getPadValueAttr<ConvOp>(type));
    Value paddedInput = getPaddedInput<MemRefIndexedValue>(
        convOp.input(), imIdx,
        /* Only need to pad the window dimensions */
        {0, static_cast<int>(imIdx.size()) - 1}, padValue);
    O(oIdx) += F(fIdx) * paddedInput;
  } else {
    IndexedValueType I(convOp.input());
    O(oIdx) += F(fIdx) * I(imIdx);
  }
}

template <typename PoolingOp> static bool hasPadding(PoolingOp poolingOp) {
  for (unsigned i = 0, e = poolingOp.getNumWindowLoops(); i < e; ++i) {
    if (poolingOp.getLowPad(i) > 0 || poolingOp.getHighPad(i) > 0)
      return true;
  }
  return false;
}

template <typename IndexedValueType, typename PoolingOp>
static Value getPoolingInput(PoolingOp op, ArrayRef<Value> inputIndices) {
  if (hasPadding(op)) {
    Type type =
        op.input().getType().template cast<MemRefType>().getElementType();
    Value padValue = std_constant(type, getPadValueAttr<PoolingOp>(type));
    return getPaddedInput<MemRefIndexedValue>(op.input(), inputIndices,
                                              /*Pad every dimension*/ {},
                                              padValue);
  }
  IndexedValueType input(op.input());
  return input(inputIndices);
}

template <typename IndexedValueType, typename OpType>
void emitPoolingMinMaxScalarImplementation(ArrayRef<Value> allIvs, OpType op) {
  InputAndOutputIndices indices = getInputAndOutputIndices(allIvs, op);
  // Emit scalar form.
  IndexedValueType output(op.output());
  Value lhs = output(indices.outputs);
  Value rhs = getPoolingInput<IndexedValueType>(op, indices.inputs);
  using edsc::op::sgt;
  using edsc::op::slt;
  Value value = std::is_same<OpType, PoolingMinOp>()
                    ? std_select(slt(lhs, rhs), lhs, rhs)
                    : std_select(sgt(lhs, rhs), lhs, rhs);
  output(indices.outputs) = value;
}

template <typename IndexedValueType>
static void emitScalarImplementation(ArrayRef<Value> allIvs, PoolingMaxOp op) {
  emitPoolingMinMaxScalarImplementation<IndexedValueType, PoolingMaxOp>(allIvs,
                                                                        op);
}

template <typename IndexedValueType>
static void emitScalarImplementation(ArrayRef<Value> allIvs, PoolingMinOp op) {
  emitPoolingMinMaxScalarImplementation<IndexedValueType, PoolingMinOp>(allIvs,
                                                                        op);
}

template <typename IndexedValueType>
static void emitScalarImplementation(ArrayRef<Value> allIvs, PoolingSumOp op) {
  auto indices = getInputAndOutputIndices(allIvs, op);
  IndexedValueType output(op.output());

  // Emit scalar form.
  output(indices.outputs) +=
      getPoolingInput<IndexedValueType>(op, indices.inputs);
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

template <typename LoopTy>
static Optional<LinalgLoops>
linalgOpToLoopsImpl(Operation *op, OpBuilder &builder,
                    ArrayRef<unsigned> interchangeVector) {
  using IndexedValueTy = typename GenerateLoopNest<LoopTy>::IndexedValueTy;
  ScopedContext scope(builder, op->getLoc());

  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  auto linalgOp = cast<LinalgOp>(op);
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");

  auto loopRanges = linalgOp.createLoopRanges(builder, op->getLoc());
  auto iteratorTypes = llvm::to_vector<4>(linalgOp.iterator_types().getValue());

  if (!interchangeVector.empty()) {
    assert(interchangeVector.size() == loopRanges.size());
    assert(interchangeVector.size() == iteratorTypes.size());
    applyPermutationToVector(loopRanges, interchangeVector);
    applyPermutationToVector(iteratorTypes, interchangeVector);
  }

  SmallVector<Value, 4> allIvs;
  GenerateLoopNest<LoopTy>::doit(
      loopRanges, /*iterInitArgs=*/{}, iteratorTypes,
      [&](ValueRange ivs, ValueRange iterArgs) -> scf::ValueVector {
        assert(iterArgs.empty() && "unexpected iterArgs");
        allIvs.append(ivs.begin(), ivs.end());
        llvm::TypeSwitch<Operation *>(op)
            .Case<ConvOp, PoolingMaxOp, PoolingMinOp, PoolingSumOp,
                  IndexedGenericOp, LinalgOp>([&](auto op) {
              emitScalarImplementation<IndexedValueTy>(allIvs, op);
            })
            .Default([&](Operation *op) { assert(false && "unexpected op"); });
        return scf::ValueVector{};
      });
  // Number of loop ops might be different from the number of ivs since some
  // loops like affine.parallel and scf.parallel have multiple ivs.
  SetVector<Operation *> loopSet;
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

/// Replace the index operations in the body of the loop nest by the matching
/// induction variables. If available use the interchange vector to map the
/// interchanged induction variables to the dimension of the index operation.
static void replaceIndexOpsByInductionVariables(
    LinalgOp linalgOp, PatternRewriter &rewriter, ArrayRef<Operation *> loopOps,
    ArrayRef<unsigned> interchangeVector) {
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
         llvm::make_early_inc_range(loopOp.getLoopBody().getOps<IndexOp>())) {
      // Search the indexing dimension in the interchange vector if available.
      assert(interchangeVector.empty() ||
             interchangeVector.size() == linalgOp.getNumLoops());
      const auto *it = llvm::find(interchangeVector, indexOp.dim());
      uint64_t dim = it != interchangeVector.end()
                         ? std::distance(interchangeVector.begin(), it)
                         : indexOp.dim();
      rewriter.replaceOp(indexOp, allIvs[dim]);
    }
  }
}

namespace {
template <typename LoopType>
class LinalgRewritePattern : public RewritePattern {
public:
  LinalgRewritePattern(MLIRContext *context,
                       ArrayRef<unsigned> interchangeVector)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        interchangeVector(interchangeVector.begin(), interchangeVector.end()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<LinalgOp>(op);
    if (!isa<LinalgOp>(op))
      return failure();
    Optional<LinalgLoops> loopOps =
        linalgOpToLoopsImpl<LoopType>(op, rewriter, interchangeVector);
    if (!loopOps.hasValue())
      return failure();
    replaceIndexOpsByInductionVariables(linalgOp, rewriter, loopOps.getValue(),
                                        interchangeVector);
    rewriter.eraseOp(op);
    return success();
  }

private:
  SmallVector<unsigned, 4> interchangeVector;
};

struct FoldAffineOp;
} // namespace

template <typename LoopType>
static void lowerLinalgToLoopsImpl(FuncOp funcOp,
                                   ArrayRef<unsigned> interchangeVector) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  patterns.add<LinalgRewritePattern<LoopType>>(context, interchangeVector);
  memref::DimOp::getCanonicalizationPatterns(patterns, context);
  AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  patterns.add<FoldAffineOp>(context);
  // Just apply the patterns greedily.
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

namespace {
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

struct LowerToAffineLoops
    : public LinalgLowerToAffineLoopsBase<LowerToAffineLoops> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<AffineForOp>(getFunction(), interchangeVector);
  }
};

struct LowerToLoops : public LinalgLowerToLoopsBase<LowerToLoops> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<scf::ForOp>(getFunction(), interchangeVector);
  }
};

struct LowerToParallelLoops
    : public LinalgLowerToParallelLoopsBase<LowerToParallelLoops> {
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<scf::ParallelOp>(getFunction(), interchangeVector);
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
template <typename LoopTy>
Optional<LinalgLoops>
mlir::linalg::linalgLowerOpToLoops(OpBuilder &builder, Operation *op,
                                   ArrayRef<unsigned> interchangeVector) {
  return linalgOpToLoopsImpl<LoopTy>(op, builder, interchangeVector);
}

template Optional<LinalgLoops> mlir::linalg::linalgLowerOpToLoops<AffineForOp>(
    OpBuilder &builder, Operation *op, ArrayRef<unsigned> interchangeVector);
template Optional<LinalgLoops> mlir::linalg::linalgLowerOpToLoops<scf::ForOp>(
    OpBuilder &builder, Operation *op, ArrayRef<unsigned> interchangeVector);
template Optional<LinalgLoops>
mlir::linalg::linalgLowerOpToLoops<scf::ParallelOp>(
    OpBuilder &builder, Operation *op, ArrayRef<unsigned> interchangeVector);

/// Emits a loop nest of `affine.for` with the proper body for `op`.
LogicalResult
mlir::linalg::linalgOpToAffineLoops(OpBuilder &builder, Operation *op,
                                    ArrayRef<unsigned> interchangeVector) {
  Optional<LinalgLoops> loops =
      linalgLowerOpToLoops<AffineForOp>(builder, op, interchangeVector);
  return loops ? success() : failure();
}

/// Emits a loop nest of `scf.for` with the proper body for `op`.
LogicalResult
mlir::linalg::linalgOpToLoops(OpBuilder &builder, Operation *op,
                              ArrayRef<unsigned> interchangeVector) {
  Optional<LinalgLoops> loops =
      linalgLowerOpToLoops<scf::ForOp>(builder, op, interchangeVector);
  return loops ? success() : failure();
}

/// Emits a loop nest of `scf.parallel` with the proper body for `op`.
LogicalResult
mlir::linalg::linalgOpToParallelLoops(OpBuilder &builder, Operation *op,
                                      ArrayRef<unsigned> interchangeVector) {
  Optional<LinalgLoops> loops =
      linalgLowerOpToLoops<scf::ParallelOp>(builder, op, interchangeVector);
  return loops ? success() : failure();
}
