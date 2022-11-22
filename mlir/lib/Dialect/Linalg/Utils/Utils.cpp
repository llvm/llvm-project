//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-utils"

using namespace mlir;
using namespace presburger;
using namespace mlir::linalg;
using namespace mlir::scf;

static bool isZero(OpFoldResult v) {
  if (!v)
    return false;
  if (auto attr = v.dyn_cast<Attribute>()) {
    IntegerAttr intAttr = attr.dyn_cast<IntegerAttr>();
    return intAttr && intAttr.getValue().isZero();
  }
  if (auto cst = v.get<Value>().getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == 0;
  return false;
}

namespace {

// Helper visitor to determine whether an AffineExpr is tiled.
// This is achieved by traversing every AffineDimExpr with position `pos` and
// checking whether the corresponding `tileSizes[pos]` is non-zero.
// This also enforces only positive coefficients occur in multiplications.
//
// Example:
//   `d0 + 2 * d1 + d3` is tiled by [0, 0, 0, 2] but not by [0, 0, 2, 0]
//
struct TileCheck : public AffineExprVisitor<TileCheck> {
  TileCheck(ArrayRef<OpFoldResult> tileSizes) : tileSizes(tileSizes) {}

  void visitDimExpr(AffineDimExpr expr) {
    isTiled |= !isZero(tileSizes[expr.getPosition()]);
  }
  void visitAffineBinaryOpExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
    if (expr.getKind() == mlir::AffineExprKind::Mul)
      assert(expr.getRHS().cast<AffineConstantExpr>().getValue() > 0 &&
             "nonpositive multiplying coefficient");
  }
  bool isTiled = false;
  ArrayRef<OpFoldResult> tileSizes;
};

} // namespace

static bool isTiled(AffineExpr expr, ArrayRef<OpFoldResult> tileSizes) {
  if (!expr)
    return false;
  TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the `map  varies with respect to a non-zero `tileSize`.
static bool isTiled(AffineMap map, ArrayRef<OpFoldResult> tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

Optional<RegionMatcher::BinaryOpKind>
RegionMatcher::matchAsScalarBinaryOp(GenericOp op) {
  auto &region = op.getRegion();
  if (!llvm::hasSingleElement(region))
    return llvm::None;

  Block &block = region.front();
  if (block.getNumArguments() != 2 ||
      !block.getArgument(0).getType().isSignlessIntOrFloat() ||
      !block.getArgument(1).getType().isSignlessIntOrFloat())
    return llvm::None;

  auto &ops = block.getOperations();
  if (!llvm::hasSingleElement(block.without_terminator()))
    return llvm::None;

  using mlir::matchers::m_Val;
  auto a = m_Val(block.getArgument(0));
  auto b = m_Val(block.getArgument(1));

  auto addPattern = m_Op<linalg::YieldOp>(m_Op<arith::AddIOp>(a, b));
  if (addPattern.match(&ops.back()))
    return BinaryOpKind::IAdd;

  return llvm::None;
}

/// Explicit instantiation of loop nest generator for different loop types.
template struct mlir::linalg::GenerateLoopNest<scf::ForOp>;
template struct mlir::linalg::GenerateLoopNest<scf::ParallelOp>;
template struct mlir::linalg::GenerateLoopNest<AffineForOp>;

/// Given a list of subview ranges, extract individual values for lower, upper
/// bounds and steps and put them into the corresponding vectors.
static void unpackRanges(OpBuilder &builder, Location loc,
                         ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                         SmallVectorImpl<Value> &ubs,
                         SmallVectorImpl<Value> &steps) {
  for (Range range : ranges) {
    lbs.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.offset));
    ubs.emplace_back(getValueOrCreateConstantIndexOp(builder, loc, range.size));
    steps.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.stride));
  }
}

namespace mlir {
namespace linalg {

bool allIndexingsAreProjectedPermutation(LinalgOp op) {
  return llvm::all_of(op.getIndexingMapsArray(), [](AffineMap m) {
    return m.isProjectedPermutation(/*allowZeroInResults=*/true);
  });
}

bool hasOnlyScalarElementwiseOp(Region &r) {
  if (!llvm::hasSingleElement(r))
    return false;
  for (Operation &op : r.front()) {
    if (!(isa<arith::ConstantOp, func::ConstantOp, tensor::ExtractOp,
              linalg::YieldOp, linalg::IndexOp>(op) ||
          OpTrait::hasElementwiseMappableTraits(&op)) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type type) { return !type.isIntOrIndexOrFloat(); }))
      return false;
  }
  return true;
}

bool isElementwise(LinalgOp op) {
  if (op.getNumLoops() != op.getNumParallelLoops())
    return false;

  if (!allIndexingsAreProjectedPermutation(op))
    return false;

  // TODO: relax the restrictions on indexing map.
  for (OpOperand *opOperand : op.getDpsInitOperands()) {
    if (!op.getMatchingIndexingMap(opOperand).isPermutation())
      return false;
  }
  return hasOnlyScalarElementwiseOp(op->getRegion(0));
}

bool isParallelIterator(utils::IteratorType iteratorType) {
  return iteratorType == utils::IteratorType::parallel;
}

bool isReductionIterator(utils::IteratorType iteratorType) {
  return iteratorType == utils::IteratorType::reduction;
}

/// Helper function that creates a memref::DimOp or tensor::DimOp depending on
/// the type of `source`.
Value createOrFoldDimOp(OpBuilder &b, Location loc, Value source, int64_t dim) {
  if (source.getType().isa<UnrankedMemRefType, MemRefType>())
    return b.createOrFold<memref::DimOp>(loc, source, dim);
  if (source.getType().isa<UnrankedTensorType, RankedTensorType>())
    return b.createOrFold<tensor::DimOp>(loc, source, dim);
  llvm_unreachable("Expected MemRefType or TensorType");
}

OpFoldResult createFoldedDimOp(OpBuilder &b, Location loc, Value source,
                               int64_t dim) {
  auto shapedType = source.getType().cast<ShapedType>();
  if (!shapedType.hasRank() || shapedType.isDynamicDim(dim))
    return createOrFoldDimOp(b, loc, source, dim);
  return b.getIndexAttr(shapedType.getDimSize(dim));
}

/// Given an operation, retrieves the value of each dynamic dimension through
/// constructing the necessary DimOp operators.
SmallVector<Value, 4> getDynOperands(Location loc, Value val, OpBuilder &b) {
  SmallVector<Value, 4> dynOperands;
  auto shapedType = val.getType().cast<ShapedType>();
  for (const auto &dim : llvm::enumerate(shapedType.getShape())) {
    if (dim.value() == ShapedType::kDynamicSize)
      dynOperands.push_back(createOrFoldDimOp(b, loc, val, dim.index()));
  }
  return dynOperands;
}

void getUpperBoundForIndex(Value value, AffineMap &boundMap,
                           SmallVectorImpl<Value> &boundOperands,
                           bool constantRequired) {
  // Initialize `boundMap` and `boundOperands` to the identity returning
  // `value`. This combination is the default result of the method if no
  // simplification is possible.
  assert(value.getType().isIndex() && "expect value to have index type");
  boundMap = AffineMap::getMultiDimIdentityMap(1, value.getContext());
  boundOperands.assign({value});
  canonicalizeMapAndOperands(&boundMap, &boundOperands);

  // Continue only if there is an affine index computation to simplify.
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp || !isa<AffineApplyOp, AffineMinOp>(definingOp))
    return;

  // Get the backward slice containing the affine index computation.
  SetVector<Operation *> backwardSlice;
  getBackwardSlice(definingOp, &backwardSlice, [](Operation *op) {
    return isa<AffineApplyOp, AffineMinOp>(op);
  });
  backwardSlice.insert(definingOp);

  // Setup a system of affine constraints that describe the index computation.
  FlatAffineValueConstraints constraints;

  // Helper to find or create an identifier for the given value.
  auto findOrCreateId = [&](Value value) {
    if (!constraints.containsVar(value)) {
      constraints.appendDimVar(value);
      return true;
    }
    unsigned pos;
    constraints.findVar(value, &pos);
    return pos < constraints.getNumDimVars();
  };
  // Helper to get the position for the given value.
  auto getPosition = [&](Value value) {
    unsigned pos;
    bool exists = constraints.findVar(value, &pos);
    (void)exists;
    assert(exists && "expect to find the identifier");
    return pos;
  };

  // Add the affine operations in `backwardSlice` to the constraints.
  for (Operation *op : llvm::reverse(backwardSlice)) {
    // Add an identifier for all op results and operands.
    if (!(llvm::all_of(op->getResults(), findOrCreateId) &&
          llvm::all_of(op->getOperands(), findOrCreateId)))
      return;

    // Add AffineApplyOps to the constraints.
    if (auto applyOp = dyn_cast<AffineApplyOp>(op)) {
      AffineMap map = constraints.computeAlignedMap(applyOp.getAffineMap(),
                                                    applyOp.getOperands());
      if (failed(constraints.addBound(IntegerPolyhedron::EQ,
                                      getPosition(applyOp.getResult()), map)))
        return;
      continue;
    }
    // Add AffineMinOps to the constraints.
    auto minOp = cast<AffineMinOp>(op);
    AffineMap map = constraints.computeAlignedMap(minOp.getAffineMap(),
                                                  minOp.getOperands());
    if (failed(constraints.addBound(IntegerPolyhedron::UB,
                                    getPosition(minOp.getResult()), map,
                                    /*isClosedBound=*/true)))
      return;
  }

  // Obtain an upper bound for the affine index computation by projecting out
  // all temporary results and expressing the upper bound for `value` in terms
  // of the terminals of the index computation.
  unsigned pos = getPosition(value);
  if (constantRequired) {
    auto ubConst = constraints.getConstantBound64(
        FlatAffineValueConstraints::BoundType::UB, pos);
    if (!ubConst)
      return;

    boundMap = AffineMap::getConstantMap(*ubConst, value.getContext());
    return;
  }

  SmallVector<AffineMap> lowerBounds(1), upperBounds(1);
  constraints.getSliceBounds(pos, 1, value.getContext(), &lowerBounds,
                             &upperBounds,
                             /*getClosedUB=*/true);
  // Verify `upperBounds[0]` is valid and has at least one result.
  if (!upperBounds[0] || upperBounds[0].getNumResults() == 0)
    return;

  // Set `boundMap` and `boundOperands` to the computed upper bound.
  boundMap = upperBounds[0];
  constraints.getAllValues(&boundOperands);
  erase_value(boundOperands, value);
  canonicalizeMapAndOperands(&boundMap, &boundOperands);
}

FailureOr<int64_t> getConstantUpperBoundForIndex(Value value) {
  // Compute an upper bound for `value`.
  AffineMap boundMap;
  SmallVector<Value> boundOperands;
  getUpperBoundForIndex(value, boundMap, boundOperands,
                        /*constantRequired=*/true);

  // Search the results of `boundMap` for constant upper bounds.
  SmallVector<int64_t> constantBounds;
  for (AffineExpr result : boundMap.getResults())
    if (auto constExpr = result.dyn_cast<AffineConstantExpr>())
      constantBounds.push_back(constExpr.getValue());

  // Return the minimal upper bound or failure if none is found.
  if (constantBounds.empty())
    return failure();
  return *std::min_element(constantBounds.begin(), constantBounds.end());
}

Value makeComposedPadHighOp(OpBuilder &b, Location loc, RankedTensorType type,
                            Value source, Value pad, bool nofold) {
  // Exit if `source` is not defined by an ExtractSliceOp.
  auto sliceOp = source.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b);

  // Search the `source` use-def chain for padded LinalgOps.
  Value current = sliceOp.getSource();
  while (current) {
    auto linalgOp = current.getDefiningOp<LinalgOp>();
    if (!linalgOp)
      break;
    OpResult opResult = current.cast<OpResult>();
    current = linalgOp.getDpsInitOperand(opResult.getResultNumber())->get();
  }
  auto padOp = current ? current.getDefiningOp<tensor::PadOp>() : nullptr;

  // Exit if the search fails to match a tensor::PadOp at the end of the matched
  // LinalgOp sequence.
  if (!padOp)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b);

  // Exit if the padded result type does not match.
  if (sliceOp.getSource().getType() != type)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b);

  // Exit if the LinalgOps are not high padded.
  if (llvm::any_of(padOp.getMixedLowPad(), [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) != static_cast<int64_t>(0);
      }))
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b);

  // Exit if `padOpSliceOp`, which defines the slice used by
  // `padOp`, is rank-reducing.
  auto padOpSliceOp = padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!padOpSliceOp ||
      sliceOp.getMixedSizes().size() != padOpSliceOp.getMixedSizes().size())
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b);

  // Exit if the sizes of the dynamic sizes of `sliceOp` do not match the size
  // of the slice padded by `padOp`.
  if (llvm::any_of(
          llvm::zip(sliceOp.getMixedSizes(), padOpSliceOp.getMixedSizes()),
          [](std::tuple<OpFoldResult, OpFoldResult> it) {
            return !isEqualConstantIntOrValue(std::get<0>(it), std::get<1>(it));
          }))
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b);

  // Exit if the padding values do not match.
  Attribute padOpPadAttr, padAttr;
  Value padOpPad = padOp.getConstantPaddingValue();
  if (!padOpPad || !matchPattern(padOpPad, m_Constant(&padOpPadAttr)) ||
      !matchPattern(pad, m_Constant(&padAttr)) || padOpPadAttr != padAttr)
    return tensor::createPadHighOp(type, source, pad, nofold, loc, b);

  // Return the padded result if the padding values and sizes match.
  return sliceOp.getSource();
}

GenericOp makeTransposeOp(OpBuilder &b, Location loc, Value inputTensor,
                          Value outputTensor,
                          ArrayRef<int64_t> transposeVector) {
  auto resultTensorType = outputTensor.getType().cast<RankedTensorType>();
  Type elementType = resultTensorType.getElementType();

  assert(isPermutation(transposeVector) &&
         "expect transpose vector to be a permutation");
  assert(transposeVector.size() ==
             static_cast<size_t>(resultTensorType.getRank()) &&
         "expect transpose vector size to match result tensor rank");

  // Compute the transpose and the indentity indexing maps.
  SmallVector<AffineMap> indexingMaps = {
      inversePermutation(AffineMap::getPermutationMap(
          SmallVector<unsigned>(transposeVector.begin(), transposeVector.end()),
          b.getContext())),
      AffineMap::getMultiDimIdentityMap(transposeVector.size(),
                                        b.getContext())};
  SmallVector<utils::IteratorType> iteratorTypes(transposeVector.size(),
                                                 utils::IteratorType::parallel);

  // Create a GenericOp to transpose `inputTensor` into `outputTensor`.
  auto transposeOp =
      b.create<GenericOp>(loc, resultTensorType, inputTensor, outputTensor,
                          indexingMaps, iteratorTypes);
  Region &body = transposeOp.getRegion();
  body.push_back(new Block());
  body.front().addArguments({elementType, elementType}, {loc, loc});

  // Create the body of the transpose operation.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(&body.front());
  b.create<YieldOp>(loc, transposeOp.getRegion().front().getArgument(0));
  return transposeOp;
}

GenericOp makeMemRefCopyOp(OpBuilder &b, Location loc, Value from, Value to) {
  auto memrefTypeTo = to.getType().cast<MemRefType>();
#ifndef NDEBUG
  auto memrefTypeFrom = from.getType().cast<MemRefType>();
  assert(memrefTypeFrom.getRank() == memrefTypeTo.getRank() &&
         "`from` and `to` memref must have the same rank");
#endif // NDEBUG

  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::makeArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args.front());
      });
}

/// Specialization to build an scf "for" nest.
template <>
void GenerateLoopNest<scf::ForOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    ArrayRef<linalg::ProcInfo> procInfo) {
  assert((procInfo.empty() || (procInfo.size() == loopRanges.size())) &&
         "expected as many entries for proc info as number of loops, even if "
         "they are null entries");
  SmallVector<Value> iterArgInitValues = linalgOp.hasBufferSemantics()
                                             ? SmallVector<Value>{}
                                             : linalgOp.getDpsInitOperands();

  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(b, loc, loopRanges, lbs, ubs, steps);
  LoopNest loopNest = mlir::scf::buildLoopNest(
      b, loc, lbs, ubs, steps, iterArgInitValues,
      [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange iterArgs) {
        assert(iterArgs.size() == iterArgInitValues.size() &&
               "expect the number of output tensors and iter args to match");
        SmallVector<Value> operandValuesToUse = linalgOp->getOperands();
        if (!iterArgs.empty()) {
          operandValuesToUse = linalgOp.getDpsInputOperands();
          operandValuesToUse.append(iterArgs.begin(), iterArgs.end());
        }
        return bodyBuilderFn(b, loc, ivs, operandValuesToUse);
      });

  if (loopNest.loops.empty() || procInfo.empty())
    return;

  // Filter out scf.for loops that were created out of parallel dimensions.
  for (const auto &loop : llvm::enumerate(loopNest.loops)) {
    if (procInfo[loop.index()].distributionMethod ==
        DistributionMethod::Cyclic) {
      mapLoopToProcessorIds(loop.value(), procInfo[loop.index()].procId,
                            procInfo[loop.index()].nprocs);
    }
  }
}

/// Specialization to build affine "for" nest.
template <>
void GenerateLoopNest<AffineForOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    ArrayRef<linalg::ProcInfo> /*procInfo*/) {
  SmallVector<Value> iterArgInitValues = linalgOp.hasBufferSemantics()
                                             ? SmallVector<Value>{}
                                             : linalgOp.getDpsInitOperands();
  assert(iterArgInitValues.empty() && "unexpected AffineForOp init values");
  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(b, loc, loopRanges, lbs, ubs, steps);

  // Affine loops require constant steps.
  SmallVector<int64_t, 4> constantSteps;
  constantSteps.reserve(steps.size());
  for (Value v : steps) {
    auto op = v.getDefiningOp<arith::ConstantIndexOp>();
    assert(op && "Affine loops require constant steps");
    constantSteps.push_back(op.value());
  }

  mlir::buildAffineLoopNest(b, loc, lbs, ubs, constantSteps,
                            [&](OpBuilder &b, Location loc, ValueRange ivs) {
                              bodyBuilderFn(b, loc, ivs,
                                            linalgOp->getOperands());
                            });
}

/// Update the `lb`, `ub` and `step` to get per processor `lb`, `ub` and `step`.
void updateBoundsForCyclicDistribution(OpBuilder &b, Location loc, Value procId,
                                       Value nprocs, Value &lb, Value &ub,
                                       Value &step) {
  AffineExpr d0, d1;
  bindDims(b.getContext(), d0, d1);
  AffineExpr s0 = getAffineSymbolExpr(0, b.getContext());
  lb = makeComposedAffineApply(b, loc, d0 + d1 * s0, {lb, procId, step});
  step = makeComposedAffineApply(b, loc, d0 * s0, {nprocs, step});
}

/// Generates a loop nest consisting of scf.parallel and scf.for, depending
/// on the `iteratorTypes.` Consecutive parallel loops create a single
/// scf.parallel operation; each sequential loop creates a new scf.for
/// operation. The body of the innermost loop is populated by
/// `bodyBuilderFn` that accepts a range of induction variables for all
/// loops. `ivStorage` is used to store the partial list of induction
/// variables.
// TODO: this function can be made iterative instead. However, it
// will have at most as many recursive calls as nested loops, which rarely
// exceeds 10.
static void generateParallelLoopNest(
    OpBuilder &b, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps, ArrayRef<utils::IteratorType> iteratorTypes,
    ArrayRef<linalg::ProcInfo> procInfo,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn,
    SmallVectorImpl<Value> &ivStorage) {
  assert(lbs.size() == ubs.size());
  assert(lbs.size() == steps.size());
  assert(lbs.size() == iteratorTypes.size());
  assert(procInfo.empty() || (lbs.size() == procInfo.size()));

  // If there are no (more) loops to be generated, generate the body and be
  // done with it.
  if (iteratorTypes.empty()) {
    bodyBuilderFn(b, loc, ivStorage);
    return;
  }

  // If there are no outer parallel loops, generate one sequential loop and
  // recurse.
  if (!isParallelIterator(iteratorTypes.front())) {
    LoopNest singleLoop = buildLoopNest(
        b, loc, lbs.take_front(), ubs.take_front(), steps.take_front(),
        [&](OpBuilder &b, Location loc, ValueRange ivs) {
          ivStorage.append(ivs.begin(), ivs.end());
          generateParallelLoopNest(
              b, loc, lbs.drop_front(), ubs.drop_front(), steps.drop_front(),
              iteratorTypes.drop_front(),
              procInfo.empty() ? procInfo : procInfo.drop_front(),
              bodyBuilderFn, ivStorage);
        });
    return;
  }

  unsigned nLoops = iteratorTypes.size();
  unsigned numProcessed = 0;
  DistributionMethod distributionMethod = DistributionMethod::None;
  if (procInfo.empty()) {
    numProcessed = nLoops - iteratorTypes.drop_while(isParallelIterator).size();
  } else {
    distributionMethod = procInfo.front().distributionMethod;
    numProcessed =
        nLoops - procInfo
                     .drop_while([&](linalg::ProcInfo p) {
                       return p.distributionMethod == distributionMethod;
                     })
                     .size();
  }

  auto remainderProcInfo =
      procInfo.empty() ? procInfo : procInfo.drop_front(numProcessed);
  switch (distributionMethod) {
  case DistributionMethod::None: {
    // Generate a single parallel loop-nest operation for all outermost
    // parallel loops and recurse.
    b.create<scf::ParallelOp>(
        loc, lbs.take_front(numProcessed), ubs.take_front(numProcessed),
        steps.take_front(numProcessed),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
          ivStorage.append(localIvs.begin(), localIvs.end());
          generateParallelLoopNest(
              nestedBuilder, nestedLoc, lbs.drop_front(numProcessed),
              ubs.drop_front(numProcessed), steps.drop_front(numProcessed),
              iteratorTypes.drop_front(numProcessed), remainderProcInfo,
              bodyBuilderFn, ivStorage);
        });
    return;
  }
  case DistributionMethod::Cyclic: {
    // Generate a single parallel loop-nest operation for all outermost
    // parallel loops and recurse.
    b.create<scf::ParallelOp>(
        loc, lbs.take_front(numProcessed), ubs.take_front(numProcessed),
        steps.take_front(numProcessed),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
          ivStorage.append(localIvs.begin(), localIvs.end());
          generateParallelLoopNest(
              nestedBuilder, nestedLoc, lbs.drop_front(numProcessed),
              ubs.drop_front(numProcessed), steps.drop_front(numProcessed),
              iteratorTypes.drop_front(numProcessed), remainderProcInfo,
              bodyBuilderFn, ivStorage);
        });
    return;
  }
  case DistributionMethod::CyclicNumProcsGeNumIters: {
    // Check (for the processed loops) that the iteration is in-bounds.
    ArithBuilder ab(b, loc);
    Value cond = ab.slt(lbs[0], ubs[0]);
    for (unsigned i = 1; i < numProcessed; ++i)
      cond = ab._and(cond, ab.slt(lbs[i], ubs[i]));
    ivStorage.append(lbs.begin(), std::next(lbs.begin(), numProcessed));
    b.create<scf::IfOp>(loc, cond, [&](OpBuilder &b, Location loc) {
      generateParallelLoopNest(b, loc, lbs.drop_front(numProcessed),
                               ubs.drop_front(numProcessed),
                               steps.drop_front(numProcessed),
                               iteratorTypes.drop_front(numProcessed),
                               remainderProcInfo, bodyBuilderFn, ivStorage);
      b.create<scf::YieldOp>(loc, ValueRange{});
    });
    return;
  }
  case DistributionMethod::CyclicNumProcsEqNumIters:
    // No check/loops needed here. Set the `%iv` to be the `%lb` and proceed
    // with inner loop generation.
    ivStorage.append(lbs.begin(), std::next(lbs.begin(), numProcessed));
    generateParallelLoopNest(
        b, loc, lbs.drop_front(numProcessed), ubs.drop_front(numProcessed),
        steps.drop_front(numProcessed), iteratorTypes.drop_front(numProcessed),
        remainderProcInfo, bodyBuilderFn, ivStorage);
    return;
  }
}

/// Specialization for generating a mix of parallel and sequential scf loops.
template <>
void GenerateLoopNest<scf::ParallelOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    ArrayRef<linalg::ProcInfo> procInfo) {
  SmallVector<Value> iterArgInitValues = linalgOp.hasBufferSemantics()
                                             ? SmallVector<Value>{}
                                             : linalgOp.getDpsInitOperands();
  assert(iterArgInitValues.empty() && "unexpected ParallelOp init values");
  // This function may be passed more iterator types than ranges.
  assert(iteratorTypes.size() >= loopRanges.size() &&
         "expected iterator type for all ranges");
  assert((procInfo.empty() || (procInfo.size() == loopRanges.size())) &&
         "expected proc information for all loops when present");
  iteratorTypes = iteratorTypes.take_front(loopRanges.size());
  SmallVector<Value, 8> lbsStorage, ubsStorage, stepsStorage, ivs;
  unsigned numLoops = iteratorTypes.size();
  ivs.reserve(numLoops);
  lbsStorage.reserve(numLoops);
  ubsStorage.reserve(numLoops);
  stepsStorage.reserve(numLoops);

  // Get the loop lb, ub, and step.
  unpackRanges(b, loc, loopRanges, lbsStorage, ubsStorage, stepsStorage);

  // Modify the lb, ub, and step based on the distribution options.
  for (const auto &it : llvm::enumerate(procInfo)) {
    if (it.value().distributionMethod != linalg::DistributionMethod::None) {
      updateBoundsForCyclicDistribution(
          b, loc, it.value().procId, it.value().nprocs, lbsStorage[it.index()],
          ubsStorage[it.index()], stepsStorage[it.index()]);
    }
  }
  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);
  generateParallelLoopNest(
      b, loc, lbs, ubs, steps, iteratorTypes, procInfo,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        bodyBuilderFn(b, loc, ivs, linalgOp->getOperands());
      },
      ivs);

  assert(ivs.size() == iteratorTypes.size() && "did not generate enough loops");
}

static Value materializeTiledShape(OpBuilder &builder, Location loc,
                                   Value valueToTile,
                                   const SliceParameters &sliceParams) {
  auto shapedType = valueToTile.getType().dyn_cast<ShapedType>();
  auto *sliceOp = TypeSwitch<ShapedType, Operation *>(shapedType)
                      .Case([&](MemRefType) {
                        return builder.create<memref::SubViewOp>(
                            loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .Case([&](RankedTensorType) {
                        return builder.create<tensor::ExtractSliceOp>(
                            loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .Default([](ShapedType) -> Operation * {
                        llvm_unreachable("Unexpected shaped type");
                      });
  return sliceOp->getResult(0);
}

Value makeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
                     ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                     ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs,
                     ArrayRef<OpFoldResult> subShapeSizes,
                     bool omitPartialTileCheck) {
  SliceParameters sliceParams =
      computeSliceParameters(builder, loc, valueToTile, tileSizes, map, lbs,
                             ubs, subShapeSizes, omitPartialTileCheck);
  return materializeTiledShape(builder, loc, valueToTile, sliceParams);
}

SliceParameters
computeSliceParameters(OpBuilder &builder, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> tileSizes, AffineMap map,
                       ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs,
                       ArrayRef<OpFoldResult> subShapeSizes,
                       bool omitPartialTileCheck) {
  auto shapedType = valueToTile.getType().dyn_cast<ShapedType>();
  assert(shapedType && "only shaped types can be tiled");
  ArrayRef<int64_t> shape = shapedType.getShape();
  int64_t rank = shapedType.getRank();

  // Compute offsets/sizes/strides for the tile.
  SliceParameters sliceParams;
  sliceParams.offsets.reserve(rank);
  sliceParams.sizes.reserve(rank);
  sliceParams.strides.reserve(rank);
  for (unsigned r = 0; r < rank; ++r) {
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: for dim#" << r);
    if (!isTiled(map.getSubMap({r}), tileSizes)) {
      sliceParams.offsets.push_back(builder.getIndexAttr(0));
      OpFoldResult dim = createFoldedDimOp(builder, loc, valueToTile, r);
      sliceParams.sizes.push_back(dim);
      sliceParams.strides.push_back(builder.getIndexAttr(1));
      LLVM_DEBUG(llvm::dbgs() << ": not tiled: use size: " << dim << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subsize...\n");

    // Tiling creates a new slice at the proper index, the slice step is 1
    // (i.e. the op does not subsample, stepping occurs in the loop).
    auto m = map.getSubMap({r});
    LLVM_DEBUG(llvm::dbgs() << "computeSliceParameters: submap: " << m << "\n");
    IRRewriter rewriter(builder);
    OpFoldResult offset = makeComposedFoldedAffineApply(rewriter, loc, m, lbs);
    sliceParams.offsets.push_back(offset);
    OpFoldResult closedIntSize =
        makeComposedFoldedAffineApply(rewriter, loc, m, subShapeSizes);
    // Resulting size needs to be made half open interval again.
    AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
    OpFoldResult size =
        makeComposedFoldedAffineApply(rewriter, loc, s0 + 1, closedIntSize);
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: raw size: " << size << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "computeSliceParameters: new offset: " << offset << "\n");
    sliceParams.strides.push_back(builder.getIndexAttr(1));

    if (omitPartialTileCheck) {
      // We statically know that the partial/boundary tile condition is
      // unnecessary.
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
      sliceParams.sizes.push_back(size);
      continue;
    }

    // The size of the subview / extract_slice should be trimmed to avoid
    // out-of-bounds accesses, unless:
    // a. We statically know the subshape size divides the shape size evenly.
    // b. The subshape size is 1. According to the way the loops are set up,
    //    tensors with "0" dimensions would never be constructed.
    int64_t shapeSize = shape[r];
    Optional<int64_t> sizeCst = getConstantIntValue(size);
    auto hasTileSizeOne = sizeCst && *sizeCst == 1;
    auto dividesEvenly = sizeCst && !ShapedType::isDynamic(shapeSize) &&
                         ((shapeSize % *sizeCst) == 0);
    if (!hasTileSizeOne && !dividesEvenly) {
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: shapeSize=" << shapeSize
                              << ", size: " << size
                              << ": make sure in bound with affine.min\n");

      AffineExpr dim0, dim1, dim2;
      bindDims(builder.getContext(), dim0, dim1, dim2);

      // Get the dimension size for this dimension. We need to first calculate
      // the max index and then plus one. This is important because for
      // convolution ops, we have its input window dimension's affine map of the
      // form `(d0 * s0 + d1)`, where `d0`/`d1 is an output/filter window
      // dimension and `s0` is stride. Directly use the dimension size of
      // output/filer window dimensions will cause incorrect calculation.
      AffineMap minusOneMap =
          AffineMap::inferFromExprList({ArrayRef<AffineExpr>{dim0 - 1}})
              .front();
      AffineMap plusOneMap =
          AffineMap::inferFromExprList({ArrayRef<AffineExpr>{dim0 + 1}})
              .front();
      SmallVector<OpFoldResult> maxIndices =
          llvm::to_vector(llvm::map_range(ubs, [&](OpFoldResult ub) {
            return makeComposedFoldedAffineApply(rewriter, loc, minusOneMap,
                                                 {ub});
          }));
      OpFoldResult maxIndex =
          makeComposedFoldedAffineApply(rewriter, loc, m, maxIndices);
      OpFoldResult d =
          makeComposedFoldedAffineApply(rewriter, loc, plusOneMap, {maxIndex});

      // Compute min(dim - offset, size) to avoid out-of-bounds accesses.
      AffineMap minMap = AffineMap::inferFromExprList(
                             {ArrayRef<AffineExpr>{dim1 - dim2, dim0}})
                             .front();
      size =
          makeComposedFoldedAffineMin(rewriter, loc, minMap, {size, d, offset});
    }
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
    sliceParams.sizes.push_back(size);
  }
  return sliceParams;
}

SmallVector<OpFoldResult> computeTileOffsets(OpBuilder &b, Location loc,
                                             ArrayRef<OpFoldResult> ivs,
                                             ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<OpFoldResult> offsets;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for loop#" << idx << "\n");
    bool isTiled = !isZero(tileSizes[idx]);
    offsets.push_back(isTiled ? ivs[idxIvs++] : b.getIndexAttr(0));
    LLVM_DEBUG(llvm::dbgs()
               << "computeTileOffsets: " << offsets.back() << "\n");
  }
  return offsets;
}

SmallVector<OpFoldResult> computeTileSizes(OpBuilder &b, Location loc,
                                           ArrayRef<OpFoldResult> tileSizes,
                                           ArrayRef<OpFoldResult> sizeBounds) {
  SmallVector<OpFoldResult> sizes;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx) {
    bool isTiled = !isZero(tileSizes[idx]);
    // Before composing, we need to make range a closed interval.
    OpFoldResult size = isTiled ? tileSizes[idx] : sizeBounds[idx];
    AffineExpr d0 = getAffineDimExpr(0, b.getContext());
    IRRewriter rewriter(b);
    sizes.push_back(makeComposedFoldedAffineApply(rewriter, loc, d0 - 1, size));
    LLVM_DEBUG(llvm::dbgs() << "computeTileSizes: " << sizes.back() << "\n");
  }
  return sizes;
}

SmallVector<Type> getTensorOutputTypes(LinalgOp op, ValueRange operands) {
  if (op.hasBufferSemantics())
    return {};
  return llvm::to_vector(
      llvm::map_range(op.getDpsInitOperands(), [&](OpOperand *opOperand) {
        return operands[opOperand->getOperandNumber()].getType();
      }));
}

SmallVector<Value> insertSlicesBack(OpBuilder &builder, Location loc,
                                    LinalgOp op, ValueRange operands,
                                    ValueRange results) {
  if (op.hasBufferSemantics())
    return {};
  SmallVector<Value> tensorResults;
  tensorResults.reserve(results.size());
  // Insert a insert_slice for each output tensor.
  unsigned resultIdx = 0;
  for (OpOperand *opOperand : op.getDpsInitOperands()) {
    // TODO: use an interface/adaptor to avoid leaking position in
    // `tiledOperands`.
    Value outputTensor = operands[opOperand->getOperandNumber()];
    if (auto sliceOp = outputTensor.getDefiningOp<tensor::ExtractSliceOp>()) {
      Value inserted = builder.create<tensor::InsertSliceOp>(
          loc, sliceOp.getSource().getType(), results[resultIdx],
          sliceOp.getSource(), sliceOp.getOffsets(), sliceOp.getSizes(),
          sliceOp.getStrides(), sliceOp.getStaticOffsets(),
          sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
      tensorResults.push_back(inserted);
    } else {
      tensorResults.push_back(results[resultIdx]);
    }
    ++resultIdx;
  }
  return tensorResults;
}

SmallVector<Optional<SliceParameters>>
computeAllSliceParameters(OpBuilder &builder, Location loc, LinalgOp linalgOp,
                          ValueRange valuesToTile, ArrayRef<OpFoldResult> ivs,
                          ArrayRef<OpFoldResult> tileSizes,
                          ArrayRef<OpFoldResult> sizeBounds,
                          bool omitPartialTileCheck) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](OpFoldResult v) { return !isZero(v); })) &&
         "expected as many ivs as non-zero sizes");

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<OpFoldResult> lbs =
      computeTileOffsets(builder, loc, ivs, tileSizes);
  SmallVector<OpFoldResult> subShapeSizes =
      computeTileSizes(builder, loc, tileSizes, sizeBounds);

  assert(static_cast<int64_t>(valuesToTile.size()) <=
             linalgOp->getNumOperands() &&
         "more value to tile than operands.");
  SmallVector<Optional<SliceParameters>> allSliceParams;
  allSliceParams.reserve(valuesToTile.size());
  for (auto [opOperand, val] :
       llvm::zip(linalgOp->getOpOperands(), valuesToTile)) {
    Value shapedOp = val;
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for operand " << shapedOp);
    AffineMap map = linalgOp.getMatchingIndexingMap(&opOperand);
    // Use `opOperand` as is if it is not tiled and not an output tensor. Having
    // an extract/insert slice pair for all output tensors simplifies follow up
    // transformations such as padding and bufferization since the
    // extract/insert slice pairs make the accessed iteration argument
    // subdomains explicit.

    Type operandType = opOperand.get().getType();
    if (!isTiled(map, tileSizes) && !(operandType.isa<RankedTensorType>() &&
                                      linalgOp.isDpsInit(&opOperand))) {
      allSliceParams.push_back(llvm::None);
      LLVM_DEBUG(llvm::dbgs()
                 << ": not tiled: use shape: " << operandType << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subshape...\n");

    allSliceParams.push_back(computeSliceParameters(
        builder, loc, shapedOp, tileSizes, map, lbs, sizeBounds, subShapeSizes,
        omitPartialTileCheck));
  }

  return allSliceParams;
}

SmallVector<Value> makeTiledShapes(OpBuilder &builder, Location loc,
                                   LinalgOp linalgOp, ValueRange valuesToTile,
                                   ArrayRef<OpFoldResult> ivs,
                                   ArrayRef<OpFoldResult> tileSizes,
                                   ArrayRef<OpFoldResult> sizeBounds,
                                   bool omitPartialTileCheck) {
  SmallVector<Optional<SliceParameters>> allSliceParameter =
      computeAllSliceParameters(builder, loc, linalgOp, valuesToTile, ivs,
                                tileSizes, sizeBounds, omitPartialTileCheck);
  SmallVector<Value> tiledShapes;
  for (auto item : llvm::zip(valuesToTile, allSliceParameter)) {
    Value valueToTile = std::get<0>(item);
    Optional<SliceParameters> sliceParams = std::get<1>(item);
    tiledShapes.push_back(
        sliceParams.has_value()
            ? materializeTiledShape(builder, loc, valueToTile, *sliceParams)
            : valueToTile);
  }
  return tiledShapes;
}

void offsetIndices(OpBuilder &b, LinalgOp linalgOp,
                   ArrayRef<OpFoldResult> offsets) {
  IRRewriter rewriter(b);
  offsetIndices(rewriter, linalgOp, offsets);
}

void offsetIndices(RewriterBase &b, LinalgOp linalgOp,
                   ArrayRef<OpFoldResult> offsets) {
  if (!linalgOp.hasIndexSemantics())
    return;

  for (IndexOp indexOp : linalgOp.getBlock()->getOps<IndexOp>()) {
    if (indexOp.getDim() >= offsets.size() || !offsets[indexOp.getDim()])
      continue;
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(indexOp);
    AffineExpr index, offset;
    bindDims(b.getContext(), index, offset);
    OpFoldResult applied = makeComposedFoldedAffineApply(
        b, indexOp.getLoc(), index + offset,
        {getAsOpFoldResult(indexOp.getResult()), offsets[indexOp.getDim()]});
    Value materialized =
        getValueOrCreateConstantIndexOp(b, indexOp.getLoc(), applied);
    b.replaceOpWithIf(indexOp, materialized, [&](OpOperand &use) {
      return use.getOwner() != materialized.getDefiningOp();
    });
  }
}

/// Get the reassociation maps to fold the result of a extract_slice (or source
/// of a insert_slice) operation with given offsets, and sizes to its
/// rank-reduced version. This is only done for the cases where the size is 1
/// and offset is 0. Strictly speaking the offset 0 is not required in general,
/// but non-zero offsets are not handled by SPIR-V backend at this point (and
/// potentially cannot be handled).
Optional<SmallVector<ReassociationIndices>>
getReassociationMapForFoldingUnitDims(ArrayRef<OpFoldResult> mixedSizes) {
  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices curr;
  for (const auto &it : llvm::enumerate(mixedSizes)) {
    auto dim = it.index();
    auto size = it.value();
    curr.push_back(dim);
    auto attr = size.dyn_cast<Attribute>();
    if (attr && attr.cast<IntegerAttr>().getInt() == 1)
      continue;
    reassociation.emplace_back(ReassociationIndices{});
    std::swap(reassociation.back(), curr);
  }
  // When the reassociations are not empty, then fold the remaining
  // unit-dimensions into the last dimension.  If the reassociations so far is
  // empty, then leave it emtpy. This will fold everything to a rank-0 tensor.
  if (!curr.empty() && !reassociation.empty())
    reassociation.back().append(curr.begin(), curr.end());
  return reassociation;
}

/// Return the identity numeric value associated to the give op.
Optional<Attribute> getNeutralElement(Operation *op) {
  // Builder only used as helper for attribute creation.
  OpBuilder b(op->getContext());
  Type resultType = op->getResult(0).getType();
  if (auto floatType = resultType.dyn_cast<FloatType>()) {
    const llvm::fltSemantics &semantic = floatType.getFloatSemantics();
    if (isa<arith::AddFOp>(op))
      return b.getFloatAttr(resultType, llvm::APFloat::getZero(semantic));
    if (isa<arith::MulFOp>(op))
      return b.getFloatAttr(resultType, llvm::APFloat(semantic, 1));
    if (isa<arith::MaxFOp>(op))
      return b.getFloatAttr(resultType,
                            llvm::APFloat::getInf(semantic, /*Negative=*/true));
    if (isa<arith::MinFOp>(op))
      return b.getFloatAttr(
          resultType, llvm::APFloat::getInf(semantic, /*Negative=*/false));
    return Attribute();
  }
  if (isa<arith::AddIOp, arith::OrIOp, arith::XOrIOp>(op))
    return b.getIntegerAttr(resultType, 0);
  if (isa<arith::AndIOp>(op))
    return b.getIntegerAttr(resultType, -1);
  if (isa<arith::MaxSIOp>(op))
    return b.getIntegerAttr(resultType, std::numeric_limits<int64_t>::min());
  if (isa<arith::MinSIOp>(op))
    return b.getIntegerAttr(resultType, std::numeric_limits<int64_t>::max());
  if (isa<arith::MulIOp>(op))
    return b.getIntegerAttr(resultType, 1);
  return llvm::None;
}

} // namespace linalg
} // namespace mlir
