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
#include "mlir/Transforms/FoldUtils.h"

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

static Value emitOrFoldComposedAffineApply(OpBuilder &b, Location loc,
                                           AffineMap map,
                                           ValueRange operandsRef,
                                           OperationFolder *folder) {
  SmallVector<Value, 4> operands(operandsRef.begin(), operandsRef.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  return folder ? folder->create<AffineApplyOp>(b, loc, map, operands)
                : b.create<AffineApplyOp>(loc, map, operands);
}

SmallVector<Value, 4> mlir::linalg::applyMapToValues(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ValueRange values,
                                                     OperationFolder *folder) {
  SmallVector<Value, 4> res;
  res.reserve(map.getNumResults());
  unsigned numDims = map.getNumDims(), numSym = map.getNumSymbols();
  // For each `expr` in `map`, applies the `expr` to the values extracted from
  // ranges. If the resulting application can be folded into a Value, the
  // folding occurs eagerly. Otherwise, an affine.apply operation is emitted.
  for (auto expr : map.getResults()) {
    AffineMap map = AffineMap::get(numDims, numSym, expr);
    res.push_back(emitOrFoldComposedAffineApply(b, loc, map, values, folder));
  }
  return res;
}

/// Returns all the operands of `linalgOp` that are not views.
/// Asserts that these operands are value types to allow transformations like
/// tiling to just use the values when cloning `linalgOp`.
SmallVector<Value, 4>
mlir::linalg::getAssumedNonViewOperands(LinalgOp linalgOp) {
  auto *op = linalgOp.getOperation();
  unsigned numViews = linalgOp.getNumInputsAndOutputs();
  unsigned nOperands = op->getNumOperands() - numViews;
  SmallVector<Value, 4> res;
  res.reserve(nOperands);
  for (unsigned i = 0; i < nOperands; ++i) {
    res.push_back(op->getOperand(numViews + i));
    auto t = res.back().getType();
    (void)t;
    assert((t.isSignlessIntOrIndexOrFloat() || t.isa<VectorType>()) &&
           "expected scalar or vector type");
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
static void unpackRanges(ArrayRef<SubViewOp::Range> ranges,
                         SmallVectorImpl<Value> &lbs,
                         SmallVectorImpl<Value> &ubs,
                         SmallVectorImpl<Value> &steps) {
  for (SubViewOp::Range range : ranges) {
    lbs.emplace_back(range.offset);
    ubs.emplace_back(range.size);
    steps.emplace_back(range.stride);
  }
}

namespace mlir {
namespace linalg {

/// Specialization to build an scf "for" nest.
template <>
void GenerateLoopNest<scf::ForOp>::doit(
    ArrayRef<SubViewOp::Range> loopRanges, ArrayRef<Attribute> iteratorTypes,
    function_ref<void(ValueRange)> bodyBuilderFn) {
  SmallVector<Value, 4> lbs, ubs, steps;
  unpackRanges(loopRanges, lbs, ubs, steps);
  edsc::loopNestBuilder(lbs, ubs, steps, bodyBuilderFn);
}

/// Specialization to build affine "for" nest.
template <>
void GenerateLoopNest<AffineForOp>::doit(
    ArrayRef<SubViewOp::Range> loopRanges, ArrayRef<Attribute> iteratorTypes,
    function_ref<void(ValueRange)> bodyBuilderFn) {
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

  edsc::affineLoopNestBuilder(lbs, ubs, constantSteps, bodyBuilderFn);
}

/// Generates a loop nest consisting of scf.parallel and scf.for, depending on
/// the `iteratorTypes.` Consecutive parallel loops create a single scf.parallel
/// operation; each sequential loop creates a new scf.for operation. The body
/// of the innermost loop is populated by `bodyBuilderFn` that accepts a range
/// of induction variables for all loops. `ivStorage` is used to store the
/// partial list of induction variables.
// TODO: this function can be made iterative instead. However, it
// will have at most as many recursive calls as nested loops, which rarely
// exceeds 10.
static void
generateParallelLoopNest(ValueRange lbs, ValueRange ubs, ValueRange steps,
                         ArrayRef<Attribute> iteratorTypes,
                         function_ref<void(ValueRange)> bodyBuilderFn,
                         SmallVectorImpl<Value> &ivStorage) {
  assert(lbs.size() == ubs.size());
  assert(lbs.size() == steps.size());
  assert(lbs.size() == iteratorTypes.size());

  // If there are no (more) loops to be generated, generate the body and be
  // done with it.
  if (iteratorTypes.empty())
    return bodyBuilderFn(ivStorage);

  // Find the outermost parallel loops and drop their types from the list.
  unsigned nLoops = iteratorTypes.size();
  iteratorTypes = iteratorTypes.drop_while(isParallelIteratorType);
  unsigned nOuterPar = nLoops - iteratorTypes.size();

  // If there are no outer parallel loops, generate one sequential loop and
  // recurse. Note that we wouldn't have dropped anything from `iteratorTypes`
  // in this case.
  if (nOuterPar == 0) {
    edsc::loopNestBuilder(lbs[0], ubs[0], steps[0], [&](Value iv) {
      ivStorage.push_back(iv);
      generateParallelLoopNest(lbs.drop_front(), ubs.drop_front(),
                               steps.drop_front(), iteratorTypes.drop_front(),
                               bodyBuilderFn, ivStorage);
    });
    return;
  }

  // Generate a single parallel loop-nest operation for all outermost parallel
  // loops and recurse.
  edsc::OperationBuilder<scf::ParallelOp>(
      lbs.take_front(nOuterPar), ubs.take_front(nOuterPar),
      steps.take_front(nOuterPar),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange localIvs) {
        edsc::ScopedContext context(nestedBuilder, nestedLoc);
        ivStorage.append(localIvs.begin(), localIvs.end());
        generateParallelLoopNest(lbs.drop_front(nOuterPar),
                                 ubs.drop_front(nOuterPar),
                                 steps.drop_front(nOuterPar), iteratorTypes,
                                 bodyBuilderFn, ivStorage);
      });
}

/// Specialization for generating a mix of parallel and sequential scf loops.
template <>
void GenerateLoopNest<scf::ParallelOp>::doit(
    ArrayRef<SubViewOp::Range> loopRanges, ArrayRef<Attribute> iteratorTypes,
    function_ref<void(ValueRange)> bodyBuilderFn) {
  SmallVector<Value, 8> lbsStorage, ubsStorage, stepsStorage, ivs;
  unpackRanges(loopRanges, lbsStorage, ubsStorage, stepsStorage);
  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);

  // This function may be passed more iterator types than ranges.
  assert(iteratorTypes.size() >= loopRanges.size() &&
         "expected iterator type for all ranges");
  iteratorTypes = iteratorTypes.take_front(loopRanges.size());
  ivs.reserve(iteratorTypes.size());
  generateParallelLoopNest(lbs, ubs, steps, iteratorTypes, bodyBuilderFn, ivs);
  assert(ivs.size() == iteratorTypes.size() && "did not generate enough loops");
}

} // namespace linalg
} // namespace mlir
