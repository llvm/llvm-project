//===- LoopUnrollAndJam.cpp - Code to perform loop unroll and jam ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop unroll and jam. Unroll and jam is a transformation
// that improves locality, in particular, register reuse, while also improving
// operation level parallelism. The example below shows what it does in nearly
// the general case. Loop unroll and jam currently works if the bounds of the
// loops inner to the loop being unroll-jammed do not depend on the latter.
//
// Before      After unroll and jam of i by factor 2:
//
//             for i, step = 2
// for i         S1(i);
//   S1;         S2(i);
//   S2;         S1(i+1);
//   for j       S2(i+1);
//     S3;       for j
//     S4;         S3(i, j);
//   S5;           S4(i, j);
//   S6;           S3(i+1, j)
//                 S4(i+1, j)
//               S5(i);
//               S6(i);
//               S5(i+1);
//               S6(i+1);
//
// Note: 'if/else' blocks are not jammed. So, if there are loops inside if
// op's, bodies of those loops will not be jammed.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPUNROLLANDJAM
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-unroll-jam"

using namespace mlir;
using namespace mlir::affine;

namespace {
struct MultiplyAddReduction {
  AffineStoreOp store;
  AffineLoadOp accLoad;
  AffineLoadOp input0;
  AffineLoadOp input1;
};

/// Match a body containing only three loads, a multiply, an add and a store.
static std::optional<MultiplyAddReduction>
matchMultiplyAddReduction(AffineForOp inner) {
  Block *body = inner.getBody();
  auto stores = body->getOps<AffineStoreOp>();
  if (!llvm::hasNItems(body->without_terminator(), 6) ||
      !llvm::hasSingleElement(stores))
    return std::nullopt;

  AffineStoreOp store = *stores.begin();
  auto add = store.getValue().getDefiningOp<arith::AddFOp>();
  if (!add)
    return std::nullopt;

  auto accLoad = add.getLhs().getDefiningOp<AffineLoadOp>();
  auto mul = add.getRhs().getDefiningOp<arith::MulFOp>();
  if (!accLoad || !mul) {
    accLoad = add.getRhs().getDefiningOp<AffineLoadOp>();
    mul = add.getLhs().getDefiningOp<arith::MulFOp>();
  }
  if (!accLoad || !mul)
    return std::nullopt;

  auto input0 = mul.getLhs().getDefiningOp<AffineLoadOp>();
  auto input1 = mul.getRhs().getDefiningOp<AffineLoadOp>();
  if (!input0 || !input1 || input0 == input1 || input0 == accLoad ||
      input1 == accLoad)
    return std::nullopt;

  // These six distinct operations must all belong to this body, leaving no
  // room for additional operations, effects or subregions.
  Operation *matchedOps[] = {store, add, mul, accLoad, input0, input1};
  if (llvm::any_of(matchedOps,
                   [&](Operation *op) { return op->getBlock() != body; }))
    return std::nullopt;

  return MultiplyAddReduction{store, accLoad, input0, input1};
}

static bool isSharedInputReduction(AffineForOp outer, unsigned factor,
                                   AliasAnalysis &aliasAnalysis) {
  // Full output grouping only: no cleanup/remainder or dynamic bounds.
  if (factor < 2 || !outer.hasConstantBounds() || outer.getStepAsInt() != 1 ||
      outer.getConstantLowerBound() != 0 ||
      outer.getConstantUpperBound() != factor || outer.getNumResults())
    return false;

  SmallVector<AffineForOp> loops;
  getPerfectlyNestedLoops(loops, outer);
  if (loops.size() < 2)
    return false;

  // The outer loop was checked above.
  for (AffineForOp loop : llvm::drop_begin(loops)) {
    if (!loop.hasConstantBounds() || loop.getNumResults())
      return false;
  }

  auto reduction = matchMultiplyAddReduction(loops.back());
  if (!reduction)
    return false;
  auto [store, accLoad, input0, input1] = *reduction;
  if (!(MemRefAccess(accLoad) == MemRefAccess(store)))
    return false;

  // An entire output coordinate is exactly the outer IV, so different outer
  // iterations access distinct index tuples. Valid memref layouts do not alias
  // distinct in-bounds tuples, regardless of allocation site, shape or strides.
  Value outerIV = outer.getInductionVar();
  auto operands = store.getMapOperands();
  bool hasOuterIVCoordinate =
      llvm::any_of(store.getAffineMap().getResults(), [&](AffineExpr expr) {
        auto dim = dyn_cast<AffineDimExpr>(expr);
        return dim && operands[dim.getPosition()] == outerIV;
      });
  if (!hasOuterIVCoordinate)
    return false;

  // Select reductions with exactly one input access invariant in the outer
  // loop.
  bool input0Invariant = isInvariantAccess(input0, outer);
  bool input1Invariant = isInvariantAccess(input1, outer);
  if (input0Invariant == input1Invariant)
    return false;

  // Require NoAlias between the accumulator and each input. The read-only
  // inputs may alias each other. Query the accessed memrefs to retain facts
  // from memref.distinct_objects.
  Value accumulator = store.getMemRef();
  return aliasAnalysis.alias(accumulator, input0.getMemRef()).isNo() &&
         aliasAnalysis.alias(accumulator, input1.getMemRef()).isNo();
}

struct LoopUnrollAndJam
    : public affine::impl::AffineLoopUnrollAndJamBase<LoopUnrollAndJam> {
  explicit LoopUnrollAndJam(
      std::optional<unsigned> unrollJamFactor = std::nullopt) {
    if (unrollJamFactor)
      this->unrollJamFactor = *unrollJamFactor;
  }

  void runOnOperation() override;
};
} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::affine::createLoopUnrollAndJamPass(int unrollJamFactor) {
  return std::make_unique<LoopUnrollAndJam>(
      unrollJamFactor == -1 ? std::nullopt
                            : std::optional<unsigned>(unrollJamFactor));
}

void LoopUnrollAndJam::runOnOperation() {
  if (getOperation().isExternal())
    return;

  if (selectSharedInputReductions) {
    // This is a structural selection policy, not a profitability model. Keep
    // the legacy path below unchanged when the pass-local option is disabled.
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    // Finish all alias queries before any candidate is rewritten.
    SmallVector<AffineForOp> candidates;
    getOperation()->walk<WalkOrder::PreOrder>([&](AffineForOp loop) {
      if (isSharedInputReduction(loop, unrollJamFactor, aliasAnalysis)) {
        candidates.push_back(loop);
        return WalkResult::skip(); // Never retain overlapping loop handles.
      }
      return WalkResult::advance();
    });
    for (AffineForOp loop : candidates)
      (void)loopUnrollJamByFactor(loop, unrollJamFactor);
    // No fallback to the legacy path if no candidate was selected.
    return;
  }

  // Currently, just the outermost loop from the first loop nest is
  // unroll-and-jammed by this pass. However, runOnAffineForOp can be called on
  // any for operation.
  auto &entryBlock = getOperation().front();
  if (auto forOp = dyn_cast<AffineForOp>(entryBlock.front()))
    (void)loopUnrollJamByFactor(forOp, unrollJamFactor);
}
