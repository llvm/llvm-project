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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-min-max"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

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

  LLVM_DEBUG({ DBGS() << "analyzing value: `" << affineOp << "`\n"; });

  // Create a `Variable` list with values corresponding to each of the results
  // in the affine affineMap.
  SmallVector<Variable> variables = llvm::map_to_vector(
      llvm::iota_range<unsigned>(0u, affineMap.getNumResults(), false),
      [&](unsigned i) {
        return Variable(affineMap.getSliceMap(i, 1), operands);
      });

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

    LLVM_DEBUG({
      DBGS() << "- inspecting variable: #" << i << ", with map: `" << v.getMap()
             << "`\n";
    });

    // Check against the other variables.
    for (size_t j = i + 1; j < variables.size(); ++j) {
      unsigned jEqClass = boundedClasses.findLeader(j);
      // Skip if the class is the same.
      if (jEqClass == eqClass)
        continue;

      // Get the bound of the equivalence class or itself.
      Variable *nv = bounds.lookup_or(jEqClass, &variables[j]);

      LLVM_DEBUG({
        DBGS() << "- comparing with variable: #" << jEqClass
               << ", with map: " << nv->getMap() << "\n";
      });

      // Compare the variables.
      FailureOr<bool> cmpResult =
          ValueBoundsConstraintSet::strongCompare(*bound, cmpOp, *nv);

      // The variables cannot be compared.
      if (failed(cmpResult)) {
        LLVM_DEBUG({
          DBGS() << "-- classes: #" << i << ", #" << jEqClass
                 << " cannot be merged\n";
        });
        continue;
      }

      // Join the equivalent classes and update the bound if necessary.
      LLVM_DEBUG({
        DBGS() << "-- merging classes: #" << i << ", #" << jEqClass
               << ", is cmp(lhs, rhs): " << *cmpResult << "`\n";
      });
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
    LLVM_DEBUG(
        { DBGS() << "- the affine operation couldn't get simplified\n"; });
    return false;
  }

  // Construct the new affine affineMap.
  SmallVector<AffineExpr> results;
  results.reserve(bounds.size());
  for (auto [k, bound] : bounds)
    results.push_back(bound->getMap().getResult(0));

  affineMap = AffineMap::get(affineMap.getNumDims(), affineMap.getNumSymbols(),
                             results, rewriter.getContext());

  // Update the affine op.
  rewriter.modifyOpInPlace(affineOp, [&]() { affineOp.setMap(affineMap); });
  LLVM_DEBUG({ DBGS() << "- simplified affine op: `" << affineOp << "`\n"; });
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
    if (auto minOp = dyn_cast<AffineMinOp>(op))
      changed = simplifyAffineMinOp(rewriter, minOp) || changed;
    else if (auto maxOp = cast<AffineMaxOp>(op))
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
