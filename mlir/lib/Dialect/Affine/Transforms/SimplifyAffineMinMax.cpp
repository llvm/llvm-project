//===- SimplifyAffineMinMax.cpp - Simplify affine min/max ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a transform to simplify mix/max affine operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "affine-min-max"

using namespace mlir;
using namespace mlir::affine;

/// Simplifies an affine min/max operation by proving there's a lower or upper
/// bound.
template <typename AffineOp>
static bool simplifyAffineMinMaxOp(RewriterBase &rewriter, AffineOp affineOp) {
  using Variable = ValueBoundsConstraintSet::Variable;
  using ComparisonOperator = ValueBoundsConstraintSet::ComparisonOperator;

  AffineMap affineMap = affineOp.getMap();
  ValueRange operands = affineOp.getOperands();
  static constexpr bool isMin = std::is_same_v<AffineOp, AffineMinOp>;

  LDBG() << "analyzing value: `" << affineOp;

  // Create a `Variable` list with values corresponding to each of the results
  // in the affine affineMap.
  SmallVector<Variable> variables = llvm::map_to_vector(
      llvm::iota_range<unsigned>(0u, affineMap.getNumResults(), false),
      [&](unsigned i) {
        return Variable(affineMap.getSliceMap(i, 1), operands);
      });
  LDBG() << "- constructed variables are: "
         << llvm::interleaved_array(llvm::map_range(
                variables, [](const Variable &v) { return v.getMap(); }));

  // Get the comparison operation.
  ComparisonOperator cmpOp =
      isMin ? ComparisonOperator::LT : ComparisonOperator::GT;

  // Find disjoint sets bounded by a common value.
  llvm::IntEqClasses boundedClasses(variables.size());
  DenseMap<unsigned, Variable *> bounds;
  for (auto &&[i, v] : llvm::enumerate(variables)) {
    unsigned eqClass = boundedClasses.findLeader(i);

    // If the class already has a bound continue.
    if (bounds.contains(eqClass))
      continue;

    // Initialize the bound.
    Variable *bound = &v;

    LDBG() << "- inspecting variable: #" << i << ", with map: `" << v.getMap()
           << "`\n";

    // Check against the other variables.
    for (size_t j = i + 1; j < variables.size(); ++j) {
      unsigned jEqClass = boundedClasses.findLeader(j);
      // Skip if the class is the same.
      if (jEqClass == eqClass)
        continue;

      // Get the bound of the equivalence class or itself.
      Variable *nv = bounds.lookup_or(jEqClass, &variables[j]);

      LDBG() << "- comparing with variable: #" << jEqClass
             << ", with map: " << nv->getMap();

      // Compare the variables.
      FailureOr<bool> cmpResult =
          ValueBoundsConstraintSet::strongCompare(*bound, cmpOp, *nv);

      // The variables cannot be compared.
      if (failed(cmpResult)) {
        LDBG() << "-- classes: #" << i << ", #" << jEqClass
               << " cannot be merged";
        continue;
      }

      // Join the equivalent classes and update the bound if necessary.
      LDBG() << "-- merging classes: #" << i << ", #" << jEqClass
             << ", is cmp(lhs, rhs): " << *cmpResult << "`";
      if (*cmpResult) {
        boundedClasses.join(eqClass, jEqClass);
      } else {
        // In this case we have lhs > rhs if isMin == true, or lhs < rhs if
        // isMin == false.
        bound = nv;
        boundedClasses.join(eqClass, jEqClass);
      }
    }
    bounds[boundedClasses.findLeader(i)] = bound;
  }

  // Return if there's no simplification.
  if (bounds.size() >= affineMap.getNumResults()) {
    LDBG() << "- the affine operation couldn't get simplified";
    return false;
  }

  // Construct the new affine affineMap.
  SmallVector<AffineExpr> results;
  results.reserve(bounds.size());
  for (auto [k, bound] : bounds)
    results.push_back(bound->getMap().getResult(0));

  LDBG() << "- starting from map: " << affineMap;
  LDBG() << "- creating new map with:";
  LDBG() << "--- dims: " << affineMap.getNumDims();
  LDBG() << "--- syms: " << affineMap.getNumSymbols();
  LDBG() << "--- res: " << llvm::interleaved_array(results);

  affineMap =
      AffineMap::get(0, affineMap.getNumSymbols() + affineMap.getNumDims(),
                     results, rewriter.getContext());

  // Update the affine op.
  rewriter.modifyOpInPlace(affineOp, [&]() { affineOp.setMap(affineMap); });
  LDBG() << "- simplified affine op: `" << affineOp << "`";
  return true;
}

bool mlir::affine::simplifyAffineMinOp(RewriterBase &rewriter, AffineMinOp op) {
  return simplifyAffineMinMaxOp(rewriter, op);
}

bool mlir::affine::simplifyAffineMaxOp(RewriterBase &rewriter, AffineMaxOp op) {
  return simplifyAffineMinMaxOp(rewriter, op);
}

LogicalResult mlir::affine::simplifyAffineMinMaxOps(RewriterBase &rewriter,
                                                    ArrayRef<Operation *> ops,
                                                    bool *modified) {
  bool changed = false;
  for (Operation *op : ops) {
    if (auto minOp = dyn_cast<AffineMinOp>(op)) {
      changed = simplifyAffineMinOp(rewriter, minOp) || changed;
      continue;
    }
    auto maxOp = cast<AffineMaxOp>(op);
    changed = simplifyAffineMaxOp(rewriter, maxOp) || changed;
  }
  RewritePatternSet patterns(rewriter.getContext());
  AffineMaxOp::getCanonicalizationPatterns(patterns, rewriter.getContext());
  AffineMinOp::getCanonicalizationPatterns(patterns, rewriter.getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (modified)
    *modified = changed;
  // Canonicalize to a fixpoint.
  if (failed(applyOpPatternsGreedily(
          ops, frozenPatterns,
          GreedyRewriteConfig()
              .setListener(
                  static_cast<RewriterBase::Listener *>(rewriter.getListener()))
              .setStrictness(GreedyRewriteStrictness::ExistingAndNewOps),
          &changed))) {
    return failure();
  }
  if (modified)
    *modified = changed;
  return success();
}

namespace {

struct SimplifyAffineMaxOp : public OpRewritePattern<AffineMaxOp> {
  using OpRewritePattern<AffineMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMaxOp affineOp,
                                PatternRewriter &rewriter) const override {
    return success(simplifyAffineMaxOp(rewriter, affineOp));
  }
};

struct SimplifyAffineMinOp : public OpRewritePattern<AffineMinOp> {
  using OpRewritePattern<AffineMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp affineOp,
                                PatternRewriter &rewriter) const override {
    return success(simplifyAffineMinOp(rewriter, affineOp));
  }
};

struct SimplifyAffineApplyOp : public OpRewritePattern<AffineApplyOp> {
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp affineOp,
                                PatternRewriter &rewriter) const override {
    AffineMap map = affineOp.getAffineMap();
    SmallVector<Value> operands{affineOp->getOperands().begin(),
                                affineOp->getOperands().end()};
    fullyComposeAffineMapAndOperands(&map, &operands,
                                     /*composeAffineMin=*/true);

    // No change => failure to apply.
    if (map == affineOp.getAffineMap())
      return failure();

    rewriter.modifyOpInPlace(affineOp, [&]() {
      affineOp.setMap(map);
      affineOp->setOperands(operands);
    });
    return success();
  }
};

} // namespace

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_SIMPLIFYAFFINEMINMAXPASS
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

/// Creates a simplification pass for affine min/max/apply.
struct SimplifyAffineMinMaxPass
    : public affine::impl::SimplifyAffineMinMaxPassBase<
          SimplifyAffineMinMaxPass> {
  void runOnOperation() override;
};

void SimplifyAffineMinMaxPass::runOnOperation() {
  FunctionOpInterface func = getOperation();
  RewritePatternSet patterns(func.getContext());
  AffineMaxOp::getCanonicalizationPatterns(patterns, func.getContext());
  AffineMinOp::getCanonicalizationPatterns(patterns, func.getContext());
  patterns.add<SimplifyAffineMaxOp, SimplifyAffineMinOp, SimplifyAffineApplyOp>(
      func.getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPatternsGreedily(func, std::move(frozenPatterns))))
    return signalPassFailure();
}
