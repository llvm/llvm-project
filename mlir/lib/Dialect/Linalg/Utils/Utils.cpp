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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::scf;

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

static Value createFoldedComposedAffineApply(OpBuilder &b, Location loc,
                                             AffineMap map,
                                             ValueRange operandsRef) {
  SmallVector<Value, 4> operands(operandsRef.begin(), operandsRef.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  return b.createOrFold<AffineApplyOp>(loc, map, operands);
}

SmallVector<Value, 4> mlir::linalg::applyMapToValues(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ValueRange values) {
  SmallVector<Value, 4> res;
  res.reserve(map.getNumResults());
  unsigned numDims = map.getNumDims(), numSym = map.getNumSymbols();
  // For each `expr` in `map`, applies the `expr` to the values extracted from
  // ranges. If the resulting application can be folded into a Value, the
  // folding occurs eagerly.
  for (auto expr : map.getResults()) {
    AffineMap map = AffineMap::get(numDims, numSym, expr);
    res.push_back(createFoldedComposedAffineApply(b, loc, map, values));
  }
  return res;
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

/// Return the linearized list of all view dimensions in a linalgOp.
SmallVector<Value, 8> getShape(OpBuilder &builder, LinalgOp linalgOp) {
  auto loc = linalgOp.getLoc();
  SmallVector<Value, 8> res;
  SmallVector<unsigned, 4> ranks;
  for (Value v : linalgOp.getShapedOperands()) {
    ShapedType t = v.getType().template cast<ShapedType>();
    ranks.push_back(t.getRank());
    for (unsigned i = 0; i < t.getRank(); ++i)
      res.push_back(builder.create<DimOp>(loc, v, i));
  }

  auto attr = linalgOp.template getAttrOfType<IntegerAttr>("symbol_source");
  if (attr) {
    // Find the correct position for inserting values for symbols.
    unsigned numSymb = ranks[attr.getInt()], symbolsPos = 0;
    for (unsigned idx = 0; idx < attr.getInt(); idx++)
      symbolsPos += ranks[idx];

    // Append the end of the value list that corresponds to the
    // values mapping to symbols. Since inside concatinated map symbols are
    // repeated we have to repeat the sizes as well.

    // Reserve is mandatory to avoid a potential undefined behavior with
    // pushing back to smallvector from itself.
    res.reserve(res.size() + ranks.size() * numSymb);
    for (unsigned idx = 0, s = ranks.size(); idx < s; ++idx)
      for (unsigned idx2 = 0; idx2 < numSymb; ++idx2)
        res.push_back(res[symbolsPos + idx2]);
  }
  return res;
}

Optional<SmallVector<Value, 4>> getLoopRanges(OpBuilder &builder,
                                              LinalgOp linalgOp) {
  SmallVector<Value, 8> viewSizes = getShape(builder, linalgOp);
  AffineMap invertedMap =
      inversePermutation(concatAffineMaps(linalgOp.getIndexingMaps()));
  if (!invertedMap)
    return {};
  return applyMapToValues(builder, linalgOp.getLoc(), invertedMap, viewSizes);
}

/// Specialization to build an scf "for" nest.
template <>
void GenerateLoopNest<scf::ForOp>::doit(
    ArrayRef<Range> loopRanges, ValueRange iterArgInitValues,
    ArrayRef<Attribute> iteratorTypes,
    function_ref<scf::ValueVector(ValueRange, ValueRange)> bodyBuilderFn,
    Optional<LinalgLoopDistributionOptions>) {
  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(loopRanges, lbs, ubs, steps);
  edsc::loopNestBuilder(lbs, ubs, steps, iterArgInitValues, bodyBuilderFn);
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

} // namespace linalg
} // namespace mlir
