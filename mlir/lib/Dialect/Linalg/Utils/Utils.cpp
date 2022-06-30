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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
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

static bool isZero(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
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
  TileCheck(ValueRange tileSizes) : tileSizes(tileSizes) {}

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
  ValueRange tileSizes;
};

} // namespace

static bool isTiled(AffineExpr expr, ValueRange tileSizes) {
  if (!expr)
    return false;
  TileCheck t(tileSizes);
  t.visit(expr);
  return t.isTiled;
}

// Checks whether the `map  varies with respect to a non-zero `tileSize`.
static bool isTiled(AffineMap map, ValueRange tileSizes) {
  if (!map)
    return false;
  for (unsigned r = 0; r < map.getNumResults(); ++r)
    if (isTiled(map.getResult(r), tileSizes))
      return true;
  return false;
}

Optional<RegionMatcher::BinaryOpKind>
RegionMatcher::matchAsScalarBinaryOp(GenericOp op) {
  auto &region = op.region();
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
static void unpackRanges(ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                         SmallVectorImpl<Value> &ubs,
                         SmallVectorImpl<Value> &steps) {
  for (Range range : ranges) {
    lbs.emplace_back(range.offset);
    ubs.emplace_back(range.size);
    steps.emplace_back(range.stride);
  }
}

namespace mlir {
namespace linalg {

bool allIndexingsAreProjectedPermutation(LinalgOp op) {
  return llvm::all_of(op.getIndexingMaps(), [](AffineMap m) {
    return m.isProjectedPermutation(/*allowZeroInResults=*/true);
  });
}

bool hasOnlyScalarElementwiseOp(Region &r) {
  if (!llvm::hasSingleElement(r))
    return false;
  for (Operation &op : r.front()) {
    if (!(isa<arith::ConstantOp, func::ConstantOp, linalg::YieldOp,
              linalg::IndexOp>(op) ||
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
  for (OpOperand *opOperand : op.getOutputOperands()) {
    if (!op.getTiedIndexingMap(opOperand).isPermutation())
      return false;
  }
  return hasOnlyScalarElementwiseOp(op->getRegion(0));
}

bool isPermutation(ArrayRef<int64_t> permutation) {
  // Count the number of appearances for all indices.
  SmallVector<int64_t> indexCounts(permutation.size(), 0);
  for (auto index : permutation) {
    // Exit if the index is out-of-range.
    if (index < 0 || index >= static_cast<int64_t>(permutation.size()))
      return false;
    indexCounts[index]++;
  }
  // Return true if all indices appear once.
  return count(indexCounts, 1) == static_cast<int64_t>(permutation.size());
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
    auto ubConst = constraints.getConstantBound(
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

tensor::ExtractSliceOp makeComposedExtractSliceOp(
    OpBuilder &b, Location loc, Value source, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides) {
  assert(source && "expect source to be nonzero");

  // Do not fold if the producer is not an ExtractSliceOp.
  auto producerOp = source.getDefiningOp<tensor::ExtractSliceOp>();
  if (!producerOp)
    return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                            strides);

  // Do not fold if the producer is rank reducing or if there are any non-unit
  // strides. Supporting non-unit strides complicates the offset computation
  // since the consumer offsets need to be multiplied by the producer strides.
  // TODO: support non-unit strides once there are use cases.
  SmallVector<OpFoldResult> allStrides = producerOp.getMixedStrides();
  allStrides.append(strides.begin(), strides.end());
  bool hasNonUnitStride = any_of(allStrides, [](OpFoldResult ofr) {
    return getConstantIntValue(ofr) != static_cast<int64_t>(1);
  });
  if (hasNonUnitStride ||
      producerOp.getSourceType().getRank() !=
          producerOp.getResult().getType().cast<ShapedType>().getRank())
    return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                            strides);

  // Fold the producer by adding the offests and extracting the slice directly
  // from the producer source tensor.
  SmallVector<OpFoldResult> foldedOffsets(offsets.begin(), offsets.end());
  AffineExpr dim1, dim2;
  bindDims(b.getContext(), dim1, dim2);
  for (const auto &en : enumerate(producerOp.getMixedOffsets())) {
    SmallVector<Value> offsetValues = {
        getValueOrCreateConstantIndexOp(b, loc, foldedOffsets[en.index()]),
        getValueOrCreateConstantIndexOp(b, loc, en.value())};
    foldedOffsets[en.index()] =
        makeComposedAffineApply(b, loc, dim1 + dim2, offsetValues).getResult();
  }
  return b.create<tensor::ExtractSliceOp>(loc, producerOp.getSource(),
                                          foldedOffsets, sizes, strides);
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
    current = linalgOp.getOutputOperand(opResult.getResultNumber())->get();
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
  SmallVector<llvm::StringRef> iteratorTypes(transposeVector.size(),
                                             getParallelIteratorTypeName());

  // Create a GenericOp to transpose `inputTensor` into `outputTensor`.
  auto transposeOp = b.create<GenericOp>(
      loc, resultTensorType, inputTensor, outputTensor,
      b.getAffineMapArrayAttr(indexingMaps), b.getStrArrayAttr(iteratorTypes),
      /*doc=*/nullptr,
      /*library_call=*/nullptr);
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
  SmallVector<StringRef> iteratorTypes(memrefTypeTo.getRank(),
                                       getParallelIteratorTypeName());
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
    ArrayRef<Attribute> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    Optional<LinalgLoopDistributionOptions> distributionOptions,
    ArrayRef<StringRef> distributionTypes) {
  SmallVector<Value> iterArgInitValues = linalgOp.getOutputTensorOperands();
  // Create procInfo so it dominates loops, if appropriate.
  SmallVector<ProcInfo, 4> procInfo;
  SmallVector<DistributionMethod, 0> distributionMethod;
  if (distributionOptions) {
    // Collect loop ranges for parallel dimensions.
    SmallVector<Range, 2> parallelLoopRanges;
    for (const auto &iteratorType : enumerate(iteratorTypes))
      if (isParallelIterator(iteratorType.value()))
        parallelLoopRanges.push_back(loopRanges[iteratorType.index()]);

    // Get their distribution schemes.
    distributionMethod = distributionOptions->distributionMethod;
    if (distributionMethod.size() < parallelLoopRanges.size())
      parallelLoopRanges.resize(distributionMethod.size());
    procInfo = distributionOptions->procInfo(b, loc, parallelLoopRanges);
  }

  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(loopRanges, lbs, ubs, steps);
  LoopNest loopNest = mlir::scf::buildLoopNest(
      b, loc, lbs, ubs, steps, iterArgInitValues,
      [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange iterArgs) {
        assert(iterArgs.size() == linalgOp.getOutputTensorOperands().size() &&
               "expect the number of output tensors and iter args to match");
        SmallVector<Value> operandValuesToUse =
            linalgOp.getInputAndOutputOperands();
        if (!iterArgs.empty()) {
          operandValuesToUse = linalgOp.getInputOperands();
          operandValuesToUse.append(iterArgs.begin(), iterArgs.end());
        }
        return bodyBuilderFn(b, loc, ivs, operandValuesToUse);
      });

  if (!distributionOptions || loopNest.loops.empty())
    return;

  // Filter out scf.for loops that were created out of parallel dimensions.
  SmallVector<scf::ForOp, 4> loops;
  for (const auto &iteratorType : enumerate(iteratorTypes))
    if (isParallelIterator(iteratorType.value()))
      loops.push_back(loopNest.loops[iteratorType.index()]);

  // Distribute - only supports cyclic distribution for now.
  for (auto it : llvm::zip(loops, procInfo, distributionMethod))
    if (std::get<2>(it) == DistributionMethod::Cyclic)
      mapLoopToProcessorIds(std::get<0>(it), std::get<1>(it).procId,
                            std::get<1>(it).nprocs);
}

/// Specialization to build affine "for" nest.
template <>
void GenerateLoopNest<AffineForOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<Attribute> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    Optional<LinalgLoopDistributionOptions>, ArrayRef<StringRef>) {
  SmallVector<Value> iterArgInitValues = linalgOp.getOutputTensorOperands();
  assert(iterArgInitValues.empty() && "unexpected AffineForOp init values");
  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(loopRanges, lbs, ubs, steps);

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
                              SmallVector<Value> operandValuesToUse =
                                  linalgOp.getInputAndOutputOperands();
                              bodyBuilderFn(b, loc, ivs, operandValuesToUse);
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
    ValueRange steps, ArrayRef<Attribute> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn,
    SmallVectorImpl<Value> &ivStorage,
    ArrayRef<DistributionMethod> distributionMethod = {}) {
  assert(lbs.size() == ubs.size());
  assert(lbs.size() == steps.size());
  assert(lbs.size() == iteratorTypes.size());

  // If there are no (more) loops to be generated, generate the body and be
  // done with it.
  if (iteratorTypes.empty()) {
    bodyBuilderFn(b, loc, ivStorage);
    return;
  }

  // Find the outermost parallel loops and drop their types from the list.
  unsigned nLoops = iteratorTypes.size();
  unsigned nOuterPar =
      nLoops - iteratorTypes.drop_while(isParallelIterator).size();

  // If there are no outer parallel loops, generate one sequential loop and
  // recurse. Note that we wouldn't have dropped anything from `iteratorTypes`
  // in this case.
  if (nOuterPar == 0) {
    LoopNest singleLoop = buildLoopNest(
        b, loc, lbs.take_front(), ubs.take_front(), steps.take_front(),
        [&](OpBuilder &b, Location loc, ValueRange ivs) {
          ivStorage.append(ivs.begin(), ivs.end());
          generateParallelLoopNest(b, loc, lbs.drop_front(), ubs.drop_front(),
                                   steps.drop_front(),
                                   iteratorTypes.drop_front(), bodyBuilderFn,
                                   ivStorage, distributionMethod);
        });
    return;
  }
  if (distributionMethod.empty()) {
    // Generate a single parallel loop-nest operation for all outermost
    // parallel loops and recurse.
    b.create<scf::ParallelOp>(
        loc, lbs.take_front(nOuterPar), ubs.take_front(nOuterPar),
        steps.take_front(nOuterPar),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
          ivStorage.append(localIvs.begin(), localIvs.end());
          generateParallelLoopNest(
              nestedBuilder, nestedLoc, lbs.drop_front(nOuterPar),
              ubs.drop_front(nOuterPar), steps.drop_front(nOuterPar),
              iteratorTypes.drop_front(nOuterPar), bodyBuilderFn, ivStorage,
              (distributionMethod.size() < nOuterPar)
                  ? ArrayRef<DistributionMethod>()
                  : distributionMethod.drop_front(nOuterPar));
        });
    return;
  }

  // Process all consecutive similarly distributed loops simultaneously.
  DistributionMethod methodToUse = distributionMethod[0];
  unsigned numProcessed = 1;
  for (unsigned i = 1; i < nOuterPar && i < distributionMethod.size(); ++i) {
    if (distributionMethod[i] != methodToUse)
      break;
    numProcessed++;
  }

  switch (methodToUse) {
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
              iteratorTypes.drop_front(numProcessed), bodyBuilderFn, ivStorage,
              (distributionMethod.size() < numProcessed)
                  ? ArrayRef<DistributionMethod>()
                  : distributionMethod.drop_front(numProcessed));
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
      generateParallelLoopNest(
          b, loc, lbs.drop_front(numProcessed), ubs.drop_front(numProcessed),
          steps.drop_front(numProcessed),
          iteratorTypes.drop_front(numProcessed), bodyBuilderFn, ivStorage,
          distributionMethod.drop_front(numProcessed));
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
        bodyBuilderFn, ivStorage, distributionMethod.drop_front(numProcessed));
    return;
  }
}

/// Specialization for generating a mix of parallel and sequential scf loops.
template <>
void GenerateLoopNest<scf::ParallelOp>::doit(
    OpBuilder &b, Location loc, ArrayRef<Range> loopRanges, LinalgOp linalgOp,
    ArrayRef<Attribute> iteratorTypes,
    function_ref<scf::ValueVector(OpBuilder &, Location, ValueRange,
                                  ValueRange)>
        bodyBuilderFn,
    Optional<LinalgLoopDistributionOptions> distributionOptions,
    ArrayRef<StringRef> distributionTypes) {
  SmallVector<Value> iterArgInitValues = linalgOp.getOutputTensorOperands();
  assert(iterArgInitValues.empty() && "unexpected ParallelOp init values");
  // This function may be passed more iterator types than ranges.
  assert(iteratorTypes.size() >= loopRanges.size() &&
         "expected iterator type for all ranges");
  iteratorTypes = iteratorTypes.take_front(loopRanges.size());
  SmallVector<Value, 8> lbsStorage, ubsStorage, stepsStorage, ivs;
  unsigned numLoops = iteratorTypes.size();
  ivs.reserve(numLoops);
  lbsStorage.reserve(numLoops);
  ubsStorage.reserve(numLoops);
  stepsStorage.reserve(numLoops);

  // Get the loop lb, ub, and step.
  unpackRanges(loopRanges, lbsStorage, ubsStorage, stepsStorage);

  // Modify the lb, ub, and step based on the distribution options.
  SmallVector<DistributionMethod, 0> distributionMethod;
  if (distributionOptions) {
    auto &options = *distributionOptions;
    distributionMethod.assign(distributionOptions->distributionMethod.begin(),
                              distributionOptions->distributionMethod.end());
    SmallVector<Range, 2> parallelLoopRanges;
    for (const auto &iteratorType : enumerate(iteratorTypes)) {
      if (isParallelIterator(iteratorType.value()))
        parallelLoopRanges.push_back(loopRanges[iteratorType.index()]);
    }
    if (distributionMethod.size() < parallelLoopRanges.size())
      parallelLoopRanges.resize(distributionMethod.size());
    SmallVector<ProcInfo, 2> procInfo =
        options.procInfo(b, loc, parallelLoopRanges);
    unsigned index = 0;
    for (const auto &iteratorType : enumerate(iteratorTypes)) {
      if (index >= procInfo.size())
        break;
      if (isParallelIterator(iteratorType.value())) {
        unsigned i = iteratorType.index();
        updateBoundsForCyclicDistribution(b, loc, procInfo[index].procId,
                                          procInfo[index].nprocs, lbsStorage[i],
                                          ubsStorage[i], stepsStorage[i]);
        index++;
      }
    }
  }
  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);
  generateParallelLoopNest(
      b, loc, lbs, ubs, steps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        SmallVector<Value> operandValuesToUse =
            linalgOp.getInputAndOutputOperands();
        bodyBuilderFn(b, loc, ivs, operandValuesToUse);
      },
      ivs, distributionMethod);

  assert(ivs.size() == iteratorTypes.size() && "did not generate enough loops");
}

static Value fullyComposeAndAffineApply(OpBuilder &b, Location loc,
                                        AffineExpr expr, ValueRange operands) {
  AffineMap map = AffineMap::inferFromExprList({expr}).front();
  SmallVector<Value> normalizedOperands(operands.begin(), operands.end());
  mlir::fullyComposeAffineMapAndOperands(&map, &normalizedOperands);
  canonicalizeMapAndOperands(&map, &normalizedOperands);
  return b.createOrFold<AffineApplyOp>(loc, map, normalizedOperands);
}

Value makeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
                     ValueRange tileSizes, AffineMap map, ValueRange lbs,
                     ValueRange ubs, ValueRange subShapeSizes,
                     bool omitPartialTileCheck) {
  auto shapedType = valueToTile.getType().dyn_cast<ShapedType>();
  assert(shapedType && "only shaped types can be tiled");
  ArrayRef<int64_t> shape = shapedType.getShape();
  int64_t rank = shapedType.getRank();

  // Construct a new subview / extract_slice for the tile.
  SmallVector<OpFoldResult, 4> offsets, sizes, strides;
  offsets.reserve(rank);
  sizes.reserve(rank);
  strides.reserve(rank);
  for (unsigned r = 0; r < rank; ++r) {
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: for dim#" << r);
    if (!isTiled(map.getSubMap({r}), tileSizes)) {
      offsets.push_back(builder.getIndexAttr(0));
      Value dim = createOrFoldDimOp(builder, loc, valueToTile, r);
      sizes.push_back(getAsOpFoldResult(dim));
      strides.push_back(builder.getIndexAttr(1));
      LLVM_DEBUG(llvm::dbgs() << ": not tiled: use size: " << dim << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subsize...\n");

    // Tiling creates a new slice at the proper index, the slice step is 1
    // (i.e. the op does not subsample, stepping occurs in the loop).
    auto m = map.getSubMap({r});
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: submap: " << m << "\n");
    auto offset = applyMapToValues(builder, loc, m, lbs).front();
    offsets.push_back(getAsOpFoldResult(offset));
    auto closedIntSize =
        applyMapToValues(builder, loc, m, subShapeSizes).front();
    // Resulting size needs to be made half open interval again.
    AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
    Value size =
        fullyComposeAndAffineApply(builder, loc, s0 + 1, closedIntSize);
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: raw size: " << size << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "makeTiledShape: new offset: " << offset << "\n");
    strides.push_back(builder.getIndexAttr(1));

    if (omitPartialTileCheck) {
      // We statically know that the partial/boundary tile condition is
      // unnecessary.
      LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
      sizes.push_back(getAsOpFoldResult(size));
      continue;
    }

    // The size of the subview / extract_slice should be trimmed to avoid
    // out-of-bounds accesses, unless:
    // a. We statically know the subshape size divides the shape size evenly.
    // b. The subshape size is 1. According to the way the loops are set up,
    //    tensors with "0" dimensions would never be constructed.
    int64_t shapeSize = shape[r];
    auto sizeCst = size.getDefiningOp<arith::ConstantIndexOp>();
    auto hasTileSizeOne = sizeCst && sizeCst.value() == 1;
    auto dividesEvenly = sizeCst && !ShapedType::isDynamic(shapeSize) &&
                         ((shapeSize % sizeCst.value()) == 0);
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
      auto maxIndices = llvm::to_vector<8>(llvm::map_range(ubs, [&](Value ub) {
        return makeComposedAffineApply(builder, loc, minusOneMap, {ub})
            .getResult();
      }));
      Value maxIndex = applyMapToValues(builder, loc, m, maxIndices).front();
      Value d = makeComposedAffineApply(builder, loc, plusOneMap, {maxIndex});

      // Compute min(dim - offset, size) to avoid out-of-bounds accesses.
      AffineMap minMap = AffineMap::inferFromExprList(
                             {ArrayRef<AffineExpr>{dim1 - dim2, dim0}})
                             .front();
      SmallVector<Value, 4> operands{size, d, offset};
      fullyComposeAffineMapAndOperands(&minMap, &operands);
      canonicalizeMapAndOperands(&minMap, &operands);
      size = builder.create<AffineMinOp>(loc, builder.getIndexType(), minMap,
                                         operands);
    }
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShape: new size: " << size << "\n");
    sizes.push_back(getAsOpFoldResult(size));
  }

  auto *sliceOp = TypeSwitch<ShapedType, Operation *>(shapedType)
                      .Case([&](MemRefType) {
                        return builder.create<memref::SubViewOp>(
                            loc, valueToTile, offsets, sizes, strides);
                      })
                      .Case([&](RankedTensorType) {
                        return makeComposedExtractSliceOp(
                            builder, loc, valueToTile, offsets, sizes, strides);
                      })
                      .Default([](ShapedType) -> Operation * {
                        llvm_unreachable("Unexpected shaped type");
                      });
  return sliceOp->getResult(0);
}

SmallVector<Value> computeTileOffsets(OpBuilder &b, Location loc,
                                      ValueRange ivs, ValueRange tileSizes) {
  SmallVector<Value> offsets;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for loop#" << idx << "\n");
    bool isTiled = !isZero(tileSizes[idx]);
    offsets.push_back(
        isTiled ? ivs[idxIvs++]
                : b.create<arith::ConstantIndexOp>(loc, 0).getResult());
    LLVM_DEBUG(llvm::dbgs()
               << "computeTileOffsets: " << offsets.back() << "\n");
  }
  return offsets;
}

SmallVector<Value> computeTileSizes(OpBuilder &b, Location loc,
                                    ValueRange tileSizes,
                                    ArrayRef<Value> sizeBounds) {
  SmallVector<Value> sizes;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx) {
    bool isTiled = !isZero(tileSizes[idx]);
    // Before composing, we need to make range a closed interval.
    Value size = isTiled ? tileSizes[idx] : sizeBounds[idx];
    AffineExpr d0 = getAffineDimExpr(0, b.getContext());
    sizes.push_back(fullyComposeAndAffineApply(b, loc, d0 - 1, size));
    LLVM_DEBUG(llvm::dbgs() << "computeTileSizes: " << sizes.back() << "\n");
  }
  return sizes;
}

SmallVector<Value, 4> makeTiledShapes(OpBuilder &b, Location loc,
                                      LinalgOp linalgOp,
                                      ArrayRef<Value> valuesToTile,
                                      ValueRange ivs, ValueRange tileSizes,
                                      ArrayRef<Value> sizeBounds,
                                      bool omitPartialTileCheck) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](Value v) { return !isZero(v); })) &&
         "expected as many ivs as non-zero sizes");

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<Value> lbs = computeTileOffsets(b, loc, ivs, tileSizes);
  SmallVector<Value> subShapeSizes =
      computeTileSizes(b, loc, tileSizes, sizeBounds);

  assert(static_cast<int64_t>(valuesToTile.size()) ==
             linalgOp.getNumInputsAndOutputs() &&
         "expected one value to tile for every operand");
  SmallVector<Value, 4> tiledShapes;
  tiledShapes.reserve(valuesToTile.size());
  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    Value shapedOp = valuesToTile[opOperand->getOperandNumber()];
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for operand " << shapedOp);
    AffineMap map = linalgOp.getTiedIndexingMap(opOperand);
    // Use `opOperand` as is if it is not tiled and not an output tensor. Having
    // an extract/insert slice pair for all output tensors simplifies follow up
    // transformations such as padding and bufferization since the
    // extract/insert slice pairs make the accessed iteration argument
    // subdomains explicit.
    if (!isTiled(map, tileSizes) && !linalgOp.isOutputTensor(opOperand)) {
      tiledShapes.push_back(shapedOp);
      LLVM_DEBUG(llvm::dbgs() << ": not tiled: use shape: "
                              << opOperand->get().getType() << "\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subshape...\n");

    tiledShapes.push_back(makeTiledShape(b, loc, shapedOp, tileSizes, map, lbs,
                                         sizeBounds, subShapeSizes,
                                         omitPartialTileCheck));
  }

  return tiledShapes;
}

void addTileLoopIvsToIndexOpResults(OpBuilder &b, LinalgOp tiledOp,
                                    ArrayRef<Value> ivs) {
  if (tiledOp.hasIndexSemantics()) {
    for (IndexOp indexOp : tiledOp.getBlock()->getOps<IndexOp>()) {
      if (ivs[indexOp.dim()] == nullptr)
        continue;
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointAfter(indexOp);
      AffineExpr index, offset;
      bindDims(b.getContext(), index, offset);
      AffineApplyOp applyOp = makeComposedAffineApply(
          b, indexOp.getLoc(), index + offset,
          ValueRange{indexOp.getResult(), ivs[indexOp.dim()]});
      indexOp.getResult().replaceAllUsesExcept(applyOp, applyOp);
    }
  }
}

} // namespace linalg
} // namespace mlir
