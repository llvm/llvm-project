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
                                           ArrayRef<Value> operandsRef,
                                           OperationFolder *folder) {
  SmallVector<Value, 4> operands(operandsRef.begin(), operandsRef.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  return folder ? folder->create<AffineApplyOp>(b, loc, map, operands)
                : b.create<AffineApplyOp>(loc, map, operands);
}

SmallVector<Value, 4> mlir::linalg::applyMapToValues(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> values,
                                                     OperationFolder *folder) {
  SmallVector<Value, 4> res;
  res.reserve(map.getNumResults());
  unsigned numDims = map.getNumDims();
  // For each `expr` in `map`, applies the `expr` to the values extracted from
  // ranges. If the resulting application can be folded into a Value, the
  // folding occurs eagerly. Otherwise, an affine.apply operation is emitted.
  for (auto expr : map.getResults()) {
    AffineMap map = AffineMap::get(numDims, 0, expr);
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

/// Specialization of loop nest generator for scf.parallel loops to handle
/// iterator types that are not parallel. These are generated as sequential
/// loops.
template <>
void mlir::linalg::GenerateLoopNest<scf::ForOp>::doit(
    MutableArrayRef<Value> allIvs, ArrayRef<SubViewOp::Range> loopRanges,
    ArrayRef<Attribute> iteratorTypes, std::function<void(void)> fun) {
  edsc::GenericLoopNestRangeBuilder<scf::ForOp>(allIvs, loopRanges)(fun);
}

template <>
void mlir::linalg::GenerateLoopNest<AffineForOp>::doit(
    MutableArrayRef<Value> allIvs, ArrayRef<SubViewOp::Range> loopRanges,
    ArrayRef<Attribute> iteratorTypes, std::function<void(void)> fun) {
  edsc::GenericLoopNestRangeBuilder<AffineForOp>(allIvs, loopRanges)(fun);
}

template <>
void mlir::linalg::GenerateLoopNest<scf::ParallelOp>::doit(
    MutableArrayRef<Value> allIvs, ArrayRef<SubViewOp::Range> loopRanges,
    ArrayRef<Attribute> iteratorTypes, std::function<void(void)> fun) {
  // Check if there is nothing to do here. This is also the recursion
  // termination.
  if (loopRanges.empty())
    return;
  size_t nOuterPar = iteratorTypes.take_front(loopRanges.size())
                         .take_while(isParallelIteratorType)
                         .size();
  if (nOuterPar == 0 && loopRanges.size() == 1)
    // Generate the sequential for loop for the remaining non-parallel loop.
    return GenerateLoopNest<scf::ForOp>::doit(allIvs, loopRanges, iteratorTypes,
                                              fun);
  if (nOuterPar == 0) {
    // The immediate outer loop is not parallel. Generate a scf.for op for this
    // loop, but there might be subsequent loops that are parallel. Use
    // recursion to find those.
    auto nestedFn = [&]() {
      GenerateLoopNest<scf::ParallelOp>::doit(allIvs.drop_front(),
                                              loopRanges.drop_front(),
                                              iteratorTypes.drop_front(), fun);
    };
    return GenerateLoopNest<scf::ForOp>::doit(allIvs[0], loopRanges[0],
                                              iteratorTypes[0], nestedFn);
  }
  if (nOuterPar == loopRanges.size()) {
    // All loops are parallel, so generate the scf.parallel op.
    return edsc::GenericLoopNestRangeBuilder<scf::ParallelOp>(allIvs,
                                                              loopRanges)(fun);
  }
  // Generate scf.parallel for the outer parallel loops. The next inner loop is
  // sequential, but there might be more parallel loops after that. So recurse
  // into the same method.
  auto nestedFn = [&]() {
    GenerateLoopNest<scf::ParallelOp>::doit(
        allIvs.drop_front(nOuterPar), loopRanges.drop_front(nOuterPar),
        iteratorTypes.drop_front(nOuterPar), fun);
  };
  return GenerateLoopNest<scf::ParallelOp>::doit(
      allIvs.take_front(nOuterPar), loopRanges.take_front(nOuterPar),
      iteratorTypes.take_front(nOuterPar), nestedFn);
}
