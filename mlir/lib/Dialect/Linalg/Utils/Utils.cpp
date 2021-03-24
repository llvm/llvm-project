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

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::scf;

static bool isZero(Value v) {
  if (auto cst = v.getDefiningOp<ConstantIndexOp>())
    return cst.getValue() == 0;
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
  TileCheck(ValueRange tileSizes) : isTiled(false), tileSizes(tileSizes) {}

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
  bool isTiled;
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

  auto addPattern = m_Op<linalg::YieldOp>(m_Op<AddIOp>(a, b));
  if (addPattern.match(&ops.back()))
    return BinaryOpKind::IAdd;

  return llvm::None;
}

bool mlir::linalg::isParallelIteratorType(Attribute attr) {
  if (auto strAttr = attr.dyn_cast<StringAttr>()) {
    return strAttr.getValue() == getParallelIteratorTypeName();
  }
  return false;
}

bool mlir::linalg::isReductionIteratorType(Attribute attr) {
  if (auto strAttr = attr.dyn_cast<StringAttr>()) {
    return strAttr.getValue() == getReductionIteratorTypeName();
  }
  return false;
}

bool mlir::linalg::isWindowIteratorType(Attribute attr) {
  if (auto strAttr = attr.dyn_cast<StringAttr>()) {
    return strAttr.getValue() == getWindowIteratorTypeName();
  }
  return false;
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

/// If `size` comes from an AffineMinOp and one of the values of AffineMinOp
/// is a constant then return a new value set to the smallest such constant.
/// Otherwise returngetSmallestBoundingIndex nullptr.
IntegerAttr getSmallestBoundingIndex(Value size) {
  Optional<int64_t> boundingConst = {};
  if (auto affineMinOp = size.getDefiningOp<AffineMinOp>()) {
    for (auto e : affineMinOp.getAffineMap().getResults())
      if (auto cst = e.dyn_cast<AffineConstantExpr>())
        boundingConst = boundingConst
                            ? std::min(boundingConst.getValue(), cst.getValue())
                            : cst.getValue();
  } else if (auto constIndexOp = size.getDefiningOp<ConstantOp>()) {
    if (constIndexOp.getType().isa<IndexType>())
      boundingConst = constIndexOp.value().cast<IntegerAttr>().getInt();
  } else if (auto affineApplyOp = size.getDefiningOp<AffineApplyOp>()) {
    if (auto cExpr = affineApplyOp.getAffineMap()
                         .getResult(0)
                         .dyn_cast<AffineConstantExpr>())
      boundingConst = cExpr.getValue();
  }
  if (boundingConst && *boundingConst >= 0)
    return Builder(size.getContext()).getIndexAttr(*boundingConst);
  return nullptr;
}

/// Specialization to build an scf "for" nest.
template <>
void GenerateLoopNest<scf::ForOp>::doit(
    ArrayRef<Range> loopRanges, ValueRange iterArgInitValues,
    ArrayRef<Attribute> iteratorTypes,
    function_ref<scf::ValueVector(ValueRange, ValueRange)> bodyBuilderFn,
    Optional<LinalgLoopDistributionOptions> distributionOptions) {
  // Create procInfo so it dominates loops, if appropriate.
  OpBuilder &builder = edsc::ScopedContext::getBuilderRef();
  Location loc = edsc::ScopedContext::getLocation();
  SmallVector<ProcInfo, 2> procInfo;
  if (distributionOptions.hasValue())
    procInfo = distributionOptions->procInfo(builder, loc, loopRanges);

  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(loopRanges, lbs, ubs, steps);
  LoopNest loopNest =
      edsc::loopNestBuilder(lbs, ubs, steps, iterArgInitValues, bodyBuilderFn);

  if (!distributionOptions.hasValue() || loopNest.loops.empty())
    return;

  // Only supports cyclic distribution for now.
  for (auto it : llvm::zip(loopNest.loops, procInfo,
                           distributionOptions->distributionMethod))
    if (std::get<2>(it) == DistributionMethod::Cyclic)
      mapLoopToProcessorIds(std::get<0>(it), std::get<1>(it).procId,
                            std::get<1>(it).nprocs);
}

/// Specialization to build affine "for" nest.
template <>
void GenerateLoopNest<AffineForOp>::doit(
    ArrayRef<Range> loopRanges, ValueRange iterArgInitValues,
    ArrayRef<Attribute> iteratorTypes,
    function_ref<scf::ValueVector(ValueRange, ValueRange)> bodyBuilderFn,
    Optional<LinalgLoopDistributionOptions>) {
  assert(iterArgInitValues.empty() && "unexpected AffineForOp init values");
  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(loopRanges, lbs, ubs, steps);

  // Affine loops require constant steps.
  SmallVector<int64_t, 4> constantSteps;
  constantSteps.reserve(steps.size());
  for (Value v : steps) {
    auto op = v.getDefiningOp<ConstantIndexOp>();
    assert(op && "Affine loops require constant steps");
    constantSteps.push_back(op.getValue());
  }

  auto bodyBuilderWithoutIterArgsFn = [&](ValueRange ivs) {
    bodyBuilderFn(ivs, {});
  };
  edsc::affineLoopNestBuilder(lbs, ubs, constantSteps,
                              bodyBuilderWithoutIterArgsFn);
}

/// Update the `lb`, `ub` and `step` to get per processor `lb`, `ub` and `step`.
static void updateBoundsForCyclicDistribution(OpBuilder &builder, Location loc,
                                              Value procId, Value nprocs,
                                              Value &lb, Value &ub,
                                              Value &step) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  lb = lb + (procId * step);
  step = nprocs * step;
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
static void
generateParallelLoopNest(ValueRange lbs, ValueRange ubs, ValueRange steps,
                         ArrayRef<Attribute> iteratorTypes,
                         function_ref<void(ValueRange)> bodyBuilderFn,
                         SmallVectorImpl<Value> &ivStorage,
                         ArrayRef<DistributionMethod> distributionMethod = {}) {
  assert(lbs.size() == ubs.size());
  assert(lbs.size() == steps.size());
  assert(lbs.size() == iteratorTypes.size());

  // If there are no (more) loops to be generated, generate the body and be
  // done with it.
  if (iteratorTypes.empty())
    return bodyBuilderFn(ivStorage);

  // Find the outermost parallel loops and drop their types from the list.
  unsigned nLoops = iteratorTypes.size();
  unsigned nOuterPar =
      nLoops - iteratorTypes.drop_while(isParallelIteratorType).size();

  // If there are no outer parallel loops, generate one sequential loop and
  // recurse. Note that we wouldn't have dropped anything from `iteratorTypes`
  // in this case.
  if (nOuterPar == 0) {
    edsc::loopNestBuilder(lbs[0], ubs[0], steps[0], [&](Value iv) {
      ivStorage.push_back(iv);
      generateParallelLoopNest(lbs.drop_front(), ubs.drop_front(),
                               steps.drop_front(), iteratorTypes.drop_front(),
                               bodyBuilderFn, ivStorage, distributionMethod);
    });
    return;
  }
  if (distributionMethod.empty()) {
    // Generate a single parallel loop-nest operation for all outermost
    // parallel loops and recurse.
    edsc::OperationBuilder<scf::ParallelOp>(
        lbs.take_front(nOuterPar), ubs.take_front(nOuterPar),
        steps.take_front(nOuterPar),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
          edsc::ScopedContext context(nestedBuilder, nestedLoc);
          ivStorage.append(localIvs.begin(), localIvs.end());
          generateParallelLoopNest(
              lbs.drop_front(nOuterPar), ubs.drop_front(nOuterPar),
              steps.drop_front(nOuterPar), iteratorTypes.drop_front(nOuterPar),
              bodyBuilderFn, ivStorage,
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
    edsc::OperationBuilder<scf::ParallelOp>(
        lbs.take_front(numProcessed), ubs.take_front(numProcessed),
        steps.take_front(numProcessed),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
          edsc::ScopedContext context(nestedBuilder, nestedLoc);
          ivStorage.append(localIvs.begin(), localIvs.end());
          generateParallelLoopNest(
              lbs.drop_front(numProcessed), ubs.drop_front(numProcessed),
              steps.drop_front(numProcessed),
              iteratorTypes.drop_front(numProcessed), bodyBuilderFn, ivStorage,
              (distributionMethod.size() < numProcessed)
                  ? ArrayRef<DistributionMethod>()
                  : distributionMethod.drop_front(numProcessed));
        });
    return;
  }
  case DistributionMethod::CyclicNumProcsGeNumIters: {
    // Check (for the processed loops) that the iteration is in-bounds.
    using edsc::op::slt;
    using edsc::op::operator&&;
    Value cond = slt(lbs[0], ubs[0]);
    for (unsigned i = 1; i < numProcessed; ++i)
      cond = cond && slt(lbs[i], ubs[i]);
    ivStorage.append(lbs.begin(), std::next(lbs.begin(), numProcessed));
    edsc::conditionBuilder(cond, [&]() {
      generateParallelLoopNest(
          lbs.drop_front(numProcessed), ubs.drop_front(numProcessed),
          steps.drop_front(numProcessed),
          iteratorTypes.drop_front(numProcessed), bodyBuilderFn, ivStorage,
          distributionMethod.drop_front(numProcessed));
    });
    return;
  }
  case DistributionMethod::CyclicNumProcsEqNumIters:
    // No check/loops needed here. Set the `%iv` to be the `%lb` and proceed
    // with inner loop generation.
    ivStorage.append(lbs.begin(), std::next(lbs.begin(), numProcessed));
    generateParallelLoopNest(
        lbs.drop_front(numProcessed), ubs.drop_front(numProcessed),
        steps.drop_front(numProcessed), iteratorTypes.drop_front(numProcessed),
        bodyBuilderFn, ivStorage, distributionMethod.drop_front(numProcessed));
    return;
  }
}

/// Specialization for generating a mix of parallel and sequential scf loops.
template <>
void GenerateLoopNest<scf::ParallelOp>::doit(
    ArrayRef<Range> loopRanges, ValueRange iterArgInitValues,
    ArrayRef<Attribute> iteratorTypes,
    function_ref<scf::ValueVector(ValueRange, ValueRange)> bodyBuilderFn,
    Optional<LinalgLoopDistributionOptions> distributionOptions) {
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
    auto &options = distributionOptions.getValue();
    OpBuilder &builder = edsc::ScopedContext::getBuilderRef();
    Location loc = edsc::ScopedContext::getLocation();
    distributionMethod.assign(distributionOptions->distributionMethod.begin(),
                              distributionOptions->distributionMethod.end());
    SmallVector<Range, 2> parallelLoopRanges;
    for (auto iteratorType : enumerate(iteratorTypes)) {
      if (isParallelIteratorType(iteratorType.value()))
        parallelLoopRanges.push_back(loopRanges[iteratorType.index()]);
    }
    if (distributionMethod.size() < parallelLoopRanges.size())
      parallelLoopRanges.resize(distributionMethod.size());
    SmallVector<ProcInfo, 2> procInfo =
        options.procInfo(builder, loc, parallelLoopRanges);
    unsigned index = 0;
    for (auto iteratorType : enumerate(iteratorTypes)) {
      if (index >= procInfo.size())
        break;
      if (isParallelIteratorType(iteratorType.value())) {
        unsigned i = iteratorType.index();
        updateBoundsForCyclicDistribution(builder, loc, procInfo[index].procId,
                                          procInfo[index].nprocs, lbsStorage[i],
                                          ubsStorage[i], stepsStorage[i]);
        index++;
      }
    }
  }
  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);
  auto bodyBuilderWithoutIterArgsFn = [&](ValueRange ivs) {
    bodyBuilderFn(ivs, {});
  };
  generateParallelLoopNest(lbs, ubs, steps, iteratorTypes,
                           bodyBuilderWithoutIterArgsFn, ivs,
                           distributionMethod);

  assert(ivs.size() == iteratorTypes.size() && "did not generate enough loops");
}

SmallVector<Value, 4> makeTiledShapes(OpBuilder &builder, Location loc,
                                      LinalgOp linalgOp,
                                      ArrayRef<Value> tiledOperands,
                                      ValueRange ivs, ValueRange tileSizes,
                                      ArrayRef<Value> sizeBounds) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](Value v) { return !isZero(v); })) &&
         "expected as many ivs as non-zero sizes");

  using namespace edsc::op;

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<Value, 8> lbs, subShapeSizes;
  for (unsigned idx = 0, idxIvs = 0, e = tileSizes.size(); idx < e; ++idx) {
    bool isTiled = !isZero(tileSizes[idx]);
    lbs.push_back(isTiled ? ivs[idxIvs++] : (Value)std_constant_index(0));
    // Before composing, we need to make range a closed interval.
    Value size = isTiled ? tileSizes[idx] : sizeBounds[idx];
    subShapeSizes.push_back(size - std_constant_index(1));
  }

  MLIRContext *context = builder.getContext();
  SmallVector<Value, 4> tiledShapes;
  tiledShapes.reserve(tiledOperands.size());
  for (auto en : llvm::enumerate(tiledOperands)) {
    Value shapedOp = en.value();
    ShapedType shapedType = shapedOp.getType().cast<ShapedType>();
    unsigned rank = shapedType.getRank();
    AffineMap map = linalgOp.getIndexingMap(en.index());
    // If the shape is not tiled, we can use it as is.
    if (!isTiled(map, tileSizes)) {
      tiledShapes.push_back(shapedOp);
      continue;
    }

    // Construct a new subview / subtensor for the tile.
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (unsigned r = 0; r < rank; ++r) {
      if (!isTiled(map.getSubMap({r}), tileSizes)) {
        offsets.push_back(builder.getIndexAttr(0));
        sizes.push_back(memref_dim(shapedOp, r).value);
        strides.push_back(builder.getIndexAttr(1));
        continue;
      }

      // Tiling creates a new slice at the proper index, the slice step is 1
      // (i.e. the op does not subsample, stepping occurs in the loop).
      auto m = map.getSubMap({r});
      auto offset = applyMapToValues(builder, loc, m, lbs).front();
      offsets.push_back(offset);
      auto closedIntSize =
          applyMapToValues(builder, loc, m, subShapeSizes).front();
      // Resulting size needs to be made half open interval again.
      auto size = closedIntSize + std_constant_index(1);

      // The size of the subview / subtensor should be trimmed to avoid
      // out-of-bounds accesses, unless we statically know the subshape size
      // divides the shape size evenly.
      int64_t shapeSize = shapedType.getDimSize(r);
      auto sizeCst = size.getDefiningOp<ConstantIndexOp>();
      if (ShapedType::isDynamic(shapeSize) || !sizeCst ||
          (shapeSize % sizeCst.getValue()) != 0) {
        AffineExpr dim0, dim1, dim2;
        bindDims(context, dim0, dim1, dim2);
        // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
        auto minMap = AffineMap::get(
            /*dimCount=*/3, /*symbolCount=*/0, {dim0, dim1 - dim2}, context);
        Value d = memref_dim(shapedOp, r);
        SmallVector<Value, 4> operands{size, d, offset};
        fullyComposeAffineMapAndOperands(&minMap, &operands);
        size = affine_min(builder.getIndexType(), minMap, operands);
      }

      sizes.push_back(size);
      strides.push_back(builder.getIndexAttr(1));
    }

    if (shapedType.isa<MemRefType>())
      tiledShapes.push_back(builder.create<memref::SubViewOp>(
          loc, shapedOp, offsets, sizes, strides));
    else
      tiledShapes.push_back(
          builder.create<SubTensorOp>(loc, shapedOp, offsets, sizes, strides));
  }

  return tiledShapes;
}

} // namespace linalg
} // namespace mlir
