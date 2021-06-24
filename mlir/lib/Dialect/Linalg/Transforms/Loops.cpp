//===- Loops.cpp - conversion from Linalg named and generic ops to loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
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

// Create a padded view into the given `input` tensor using the 'indices'
// to access the tensor. `skipPadding` lists the dimensions for which no padding
// is needed e.g. the non-spatial dimensions for convolutions.
Value getPaddedInput(OpBuilder &b, Location loc, Value input,
                     ArrayRef<Value> indices, ArrayRef<int> skipPadding,
                     Value padValue) {
  Value zeroIndex = b.create<ConstantIndexOp>(loc, 0);
  SmallVector<Value> conds;
  SmallVector<Value> clampedImIdx;
  for (auto iter : llvm::enumerate(indices)) {
    int idx = iter.index();
    auto dim = iter.value();
    if (is_contained(skipPadding, idx)) {
      clampedImIdx.push_back(dim);
      continue;
    }

    Value leftOutOfBound =
        b.create<CmpIOp>(loc, CmpIPredicate::slt, dim, zeroIndex);
    if (conds.empty())
      conds.push_back(leftOutOfBound);
    else
      conds.push_back(b.create<OrOp>(loc, conds.back(), leftOutOfBound));
    Value rightBound = b.create<memref::DimOp>(loc, input, idx);
    Value rightOutOfBound =
        b.create<CmpIOp>(loc, CmpIPredicate::sge, dim, rightBound);
    conds.push_back(b.create<OrOp>(loc, conds.back(), rightOutOfBound));

    // When padding is involved, the indices will only be shifted to negative,
    // so having a max op is enough.
    MLIRContext *ctx = input.getContext();
    AffineExpr m = getAffineDimExpr(/*position=*/0, ctx),
               zero = getAffineConstantExpr(0, ctx);
    AffineMap maxMap =
        AffineMap::inferFromExprList(ArrayRef<ArrayRef<AffineExpr>>{{m, zero}})
            .front();
    clampedImIdx.push_back(b.create<AffineMaxOp>(loc, maxMap, ValueRange{dim}));
  }

  Value readInput = b.create<memref::LoadOp>(loc, input, clampedImIdx);
  if (conds.empty())
    return readInput;

  return b.create<SelectOp>(loc, conds.back(), padValue, readInput);
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
  if (auto floatType = type.dyn_cast<FloatType>()) {
    return OpBuilder(type.getContext())
        .getFloatAttr(floatType, APFloat::getInf(floatType.getFloatSemantics(),
                                                 /*Negative*/ true));
  }
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    // The select instruction used to lower the PoolingMin uses a signed
    // comparison, use a signed constant irrespective of the signedness of the
    // integer type.
    return OpBuilder(type.getContext())
        .getIntegerAttr(intType, APInt::getSignedMinValue(width));
  }
  llvm_unreachable("Unsupported data type for PoolingMaxOp");
  return {};
}

template <> Attribute getPadValueAttr<PoolingMinOp>(Type type) {
  if (auto floatType = type.dyn_cast<FloatType>()) {
    return OpBuilder(type.getContext())
        .getFloatAttr(floatType,
                      APFloat::getInf(floatType.getFloatSemantics()));
  }
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    // The select instruction used to lower the PoolingMin uses a signed
    // comparison, use a signed constant irrespective of the signedness of the
    // integer type.
    return OpBuilder(type.getContext())
        .getIntegerAttr(intType, APInt::getSignedMaxValue(width));
  }
  llvm_unreachable("Unsupported data type for PoolingMinOp");
  return {};
}

template <> Attribute getPadValueAttr<PoolingSumOp>(Type type) {
  return OpBuilder(type.getContext()).getZeroAttr(type);
}

template <> Attribute getPadValueAttr<ConvOp>(Type type) {
  return OpBuilder(type.getContext()).getZeroAttr(type);
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

template <typename LoadOpTy, typename StoreOpTy>
static void emitScalarImplementation(OpBuilder &b, Location loc,
                                     ArrayRef<Value> allIvs, ConvOp convOp) {
  assert(convOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto mapsRange = convOp.indexing_maps().getAsRange<AffineMapAttr>();
  auto maps = llvm::to_vector<8>(
      llvm::map_range(mapsRange, [](AffineMapAttr a) { return a.getValue(); }));
  SmallVector<Value> fIdx(makeCanonicalAffineApplies(b, loc, maps[0], allIvs));
  SmallVector<Value> imIdx(makeCanonicalAffineApplies(b, loc, maps[1], allIvs));
  SmallVector<Value> oIdx(makeCanonicalAffineApplies(b, loc, maps[2], allIvs));

  Value filter = convOp.filter(), output = convOp.output();

  // Emit scalar form. Padded conv involves an affine.max in the memory access
  // which is not allowed by affine.load. Override to use an MemRefIndexedValue
  // when there is non-zero padding.
  if (hasPadding(convOp)) {
    Type type = convOp.input().getType().cast<MemRefType>().getElementType();
    Value padValue =
        b.create<ConstantOp>(loc, type, getPadValueAttr<ConvOp>(type));
    Value paddedInput =
        getPaddedInput(b, loc, convOp.input(), imIdx,
                       /* Only need to pad the window dimensions */
                       {0, static_cast<int>(imIdx.size()) - 1}, padValue);
    Value filterVal = b.create<LoadOpTy>(loc, filter, fIdx);
    Value mulVal = ArithBuilder(b, loc).mul(filterVal, paddedInput);
    Value outputVal = b.create<LoadOpTy>(loc, output, oIdx);
    Value addVal = ArithBuilder(b, loc).add(mulVal, outputVal);
    b.create<StoreOpTy>(loc, addVal, output, oIdx);
  } else {
    Value inputVal = b.create<LoadOpTy>(loc, convOp.input(), imIdx);
    Value filterVal = b.create<LoadOpTy>(loc, filter, fIdx);
    Value mulVal = ArithBuilder(b, loc).mul(filterVal, inputVal);
    Value outputVal = b.create<LoadOpTy>(loc, output, oIdx);
    Value addVal = ArithBuilder(b, loc).add(mulVal, outputVal);
    b.create<StoreOpTy>(loc, addVal, output, oIdx);
  }
}

template <typename PoolingOp> static bool hasPadding(PoolingOp poolingOp) {
  for (unsigned i = 0, e = poolingOp.getNumWindowLoops(); i < e; ++i) {
    if (poolingOp.getLowPad(i) > 0 || poolingOp.getHighPad(i) > 0)
      return true;
  }
  return false;
}

template <typename LoadOpTy, typename StoreOpTy, typename PoolingOp>
static Value getPoolingInput(OpBuilder &b, Location loc, PoolingOp op,
                             ArrayRef<Value> inputIndices) {
  if (hasPadding(op)) {
    Type type =
        op.input().getType().template cast<MemRefType>().getElementType();
    Value padValue =
        b.create<ConstantOp>(loc, type, getPadValueAttr<PoolingOp>(type));
    return getPaddedInput(b, loc, op.input(), inputIndices,
                          /*Pad every dimension*/ {}, padValue);
  }
  return b.create<LoadOpTy>(loc, op.input(), inputIndices);
}

template <typename LoadOpTy, typename StoreOpTy, typename OpType>
void emitPoolingMinMaxScalarImplementation(OpBuilder &b, Location loc,
                                           ArrayRef<Value> allIvs, OpType op) {
  InputAndOutputIndices indices = getInputAndOutputIndices(b, loc, allIvs, op);
  Value lhs = b.create<LoadOpTy>(loc, op.output(), indices.outputs);
  Value rhs = getPoolingInput<LoadOpTy, StoreOpTy>(b, loc, op, indices.inputs);
  Value value = llvm::TypeSwitch<Operation *, Value>(op)
                    .Case([&](PoolingMinOp poolingOp) {
                      return ArithBuilder(b, loc).select(
                          ArithBuilder(b, loc).slt(lhs, rhs), lhs, rhs);
                    })
                    .Case([&](PoolingMaxOp poolingOp) {
                      return ArithBuilder(b, loc).select(
                          ArithBuilder(b, loc).sgt(lhs, rhs), lhs, rhs);
                    })
                    .Default([&](auto) { return Value(); });
  b.create<StoreOpTy>(loc, value, op.output(), indices.outputs);
}

template <typename LoadOpTy, typename StoreOpTy>
static void emitScalarImplementation(OpBuilder &b, Location loc,
                                     ArrayRef<Value> allIvs, PoolingMaxOp op) {
  emitPoolingMinMaxScalarImplementation<LoadOpTy, StoreOpTy, PoolingMaxOp>(
      b, loc, allIvs, op);
}

template <typename LoadOpTy, typename StoreOpTy>
static void emitScalarImplementation(OpBuilder &b, Location loc,
                                     ArrayRef<Value> allIvs, PoolingMinOp op) {
  emitPoolingMinMaxScalarImplementation<LoadOpTy, StoreOpTy, PoolingMinOp>(
      b, loc, allIvs, op);
}

template <typename LoadOpTy, typename StoreOpTy>
static void emitScalarImplementation(OpBuilder &b, Location loc,
                                     ArrayRef<Value> allIvs, PoolingSumOp op) {
  auto indices = getInputAndOutputIndices(b, loc, allIvs, op);
  Value inputVal =
      getPoolingInput<LoadOpTy, StoreOpTy>(b, loc, op, indices.inputs);
  Value outputVal = b.create<LoadOpTy>(loc, op.output(), indices.outputs);
  Value added = ArithBuilder(b, loc).add(outputVal, inputVal);
  b.create<StoreOpTy>(loc, added, op.output(), indices.outputs);
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
static Optional<LinalgLoops> linalgOpToLoopsImpl(PatternRewriter &rewriter,
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
          ValueRange iterArgs) -> scf::ValueVector {
        assert(iterArgs.empty() && "unexpected iterArgs");
        allIvs.append(ivs.begin(), ivs.end());
        llvm::TypeSwitch<Operation *>(linalgOp)
            .Case<ConvOp, PoolingMaxOp, PoolingMinOp, PoolingSumOp, LinalgOp>(
                [&](auto op) {
                  emitScalarImplementation<LoadOpTy, StoreOpTy>(b, loc, allIvs,
                                                                op);
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
    if (!linalgOpToLoopsImpl<LoopType>(rewriter, linalgOp))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct TiledLoopToSCFPattern : public OpRewritePattern<TiledLoopOp> {
  using OpRewritePattern<TiledLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledLoopOp tiledLoop,
                                PatternRewriter &rewriter) const override {
    Location loc = tiledLoop.getLoc();

    // Fail conversion if the `tiled_loop` has not been bufferized.
    if (!llvm::all_of(tiledLoop.outputs(), [&](Value arg) {
          return arg.getType().isa<MemRefType>();
        }))
      return failure();

    // TODO: Build loop nest with `scf.for` and `scf.parallel` depending on the
    // iterator type.
    scf::buildLoopNest(rewriter, loc, tiledLoop.lowerBound(),
                       tiledLoop.upperBound(), tiledLoop.step(),
                       [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                         // Move body without its terminator.
                         SmallVector<Value> newBlockArgs;
                         newBlockArgs.append(ivs.begin(), ivs.end());
                         newBlockArgs.append(tiledLoop.inputs().begin(),
                                             tiledLoop.inputs().end());
                         newBlockArgs.append(tiledLoop.outputs().begin(),
                                             tiledLoop.outputs().end());
                         Block *newBody = rewriter.getInsertionBlock();
                         rewriter.mergeBlocks(tiledLoop.getBody(), newBody,
                                              newBlockArgs);
                         rewriter.eraseOp(newBody->getTerminator());
                       });
    rewriter.eraseOp(tiledLoop);
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

template <typename LoopType>
static void lowerLinalgToLoopsImpl(FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  patterns.add<LinalgRewritePattern<LoopType>>(context);
  memref::DimOp::getCanonicalizationPatterns(patterns, context);
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
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<AffineForOp>(getFunction());
  }
};

struct LowerToLoops : public LinalgLowerToLoopsBase<LowerToLoops> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<scf::ForOp>(getFunction());
  }
};

struct LowerToParallelLoops
    : public LinalgLowerToParallelLoopsBase<LowerToParallelLoops> {
  void runOnFunction() override {
    lowerLinalgToLoopsImpl<scf::ParallelOp>(getFunction());
  }
};

struct LowerTiledLoopsToSCF
    : public LinalgLowerTiledLoopsToSCFBase<LowerTiledLoopsToSCF> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateTiledLoopToSCFPattern(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};
} // namespace

void mlir::linalg::populateTiledLoopToSCFPattern(RewritePatternSet &patterns) {
  patterns.add<TiledLoopToSCFPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createConvertLinalgTiledLoopsToSCFPass() {
  return std::make_unique<LowerTiledLoopsToSCF>();
}

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

/// Emits a loop nest of `affine.for` with the proper body for `linalgOp`.
Optional<LinalgLoops>
mlir::linalg::linalgOpToAffineLoops(PatternRewriter &rewriter,
                                    LinalgOp linalgOp) {
  return linalgOpToLoopsImpl<AffineForOp>(rewriter, linalgOp);
}

/// Emits a loop nest of `scf.for` with the proper body for `linalgOp`.
Optional<LinalgLoops> mlir::linalg::linalgOpToLoops(PatternRewriter &rewriter,
                                                    LinalgOp linalgOp) {
  return linalgOpToLoopsImpl<scf::ForOp>(rewriter, linalgOp);
}

/// Emits a loop nest of `scf.parallel` with the proper body for `linalgOp`.
Optional<LinalgLoops>
mlir::linalg::linalgOpToParallelLoops(PatternRewriter &rewriter,
                                      LinalgOp linalgOp) {
  return linalgOpToLoopsImpl<scf::ParallelOp>(rewriter, linalgOp);
}
