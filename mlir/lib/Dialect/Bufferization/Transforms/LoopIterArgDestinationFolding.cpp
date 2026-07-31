//===- LoopIterArgDestinationFolding.cpp - Reuse iter_arg buffers ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pre-bufferization rewrite that turns a loop-carried
// value into an in-place update by folding the destination of its yielded write
// onto the loop's iter_arg.
//
// After vectorization, a tiled reduction typically threads an accumulator as a
// read-only `scf.for` iter_arg and writes the updated value into a *fresh*
// `tensor.empty` that is then yielded:
//
//   %r = scf.for ... iter_args(%acc = %init) -> (tensor<...>) {
//     %v = vector.transfer_read %acc[...]        // read the incoming value
//     ... compute %new ...
//     %e = tensor.empty()
//     %w = vector.transfer_write %new, %e[...]   // write into a fresh tensor
//     scf.yield %w                                // yield != iter_arg
//   }
//
// Because the yielded tensor is not the iter_arg, one-shot bufferization must
// allocate a fresh buffer and copy into it every iteration (its result may not
// alias a buffer defined outside the loop other than its own init operand).
// When the iter_arg is read-then-fully-overwritten, the same buffer can serve
// both roles, so redirecting the write destination to the iter_arg exposes the
// reuse:
//
//     %w = vector.transfer_write %new, %acc[...]  // destination = iter_arg
//     scf.yield %w                                 // yield == iter_arg (in place)
//
// This folds the iter_arg to loop-invariant reuse; bufferization then keeps it
// in place with no per-iteration copy.
//
// Correctness: one-shot bufferization's in-place analysis is the final arbiter.
// If the reuse would be unsound (e.g. the iter_arg is read again *after* the
// write), the analysis declines the in-place update and reinserts the copy, so
// this rewrite is a hint that never changes program semantics. The legality
// check below is nonetheless conservative so the pass only fires where the
// reuse is expected to be honored.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_LOOPITERARGDESTINATIONFOLDINGPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::scf;

/// Returns the whole-tensor `vector.transfer_write` that produces `value` and
/// writes into a loop-invariant `tensor.empty`, or nullptr if `value` is not
/// such a write. A whole-tensor write has all-zero constant indices, an
/// identity permutation, and all dims in-bounds, so it fully defines the tensor
/// and its destination's prior contents are dead.
static vector::TransferWriteOp
getFoldableYieldWrite(Value value, ForOp loop) {
  auto write = value.getDefiningOp<vector::TransferWriteOp>();
  if (!write)
    return nullptr;
  // The write must be inside the loop body (it defines the yielded value).
  if (!loop->isProperAncestor(write))
    return nullptr;
  // Destination must be a tensor.empty defined outside the loop: a pure scratch
  // whose contents are undefined, so this write fully defines the result and
  // redirecting only this write's destination operand is safe regardless of the
  // empty's other uses.
  auto empty = write.getBase().getDefiningOp<tensor::EmptyOp>();
  if (!empty || loop->isProperAncestor(empty))
    return nullptr;
  // Whole-tensor write: no mask, in-bounds, identity permutation, zero indices.
  if (write.getMask())
    return nullptr;
  if (!write.getPermutationMap().isIdentity())
    return nullptr;
  if (llvm::any_of(write.getInBoundsValues(), [](bool b) { return !b; }))
    return nullptr;
  if (!llvm::all_of(write.getIndices(), [](Value idx) {
        return matchPattern(idx, m_Zero());
      }))
    return nullptr;
  return write;
}

/// Checks that folding the yielded write for iter_arg index `idx` onto the
/// iter_arg is legal: the iter_arg's every in-loop read must not observe the
/// write, i.e. all reads must properly precede the write in the (single-block)
/// loop body. A read after the write would, once the buffer is reused, observe
/// this iteration's own store instead of the incoming value.
static bool readsPrecedeWrite(ForOp loop, unsigned idx,
                              vector::TransferWriteOp write,
                              DominanceInfo &dominance) {
  BlockArgument iterArg = loop.getRegionIterArgs()[idx];
  for (OpOperand &use : iterArg.getUses()) {
    Operation *user = use.getOwner();
    // The yield use is the loop-carry itself; ignore it.
    if (isa<scf::YieldOp>(user) && user->getParentOp() == loop)
      continue;
    // Any read must strictly dominate the write within the body.
    if (!dominance.properlyDominates(user, write.getOperation()))
      return false;
  }
  return true;
}

/// Attempts to fold the yielded write of iter_arg `idx` onto the iter_arg.
/// Returns true if the IR was modified.
static bool tryFoldIterArg(ForOp loop, unsigned idx, DominanceInfo &dominance) {
  // Only shaped (tensor) iter_args participate.
  BlockArgument iterArg = loop.getRegionIterArgs()[idx];
  if (!isa<TensorType>(iterArg.getType()))
    return false;

  auto yieldOp = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  Value yielded = yieldOp.getOperand(idx);

  // Already in place.
  if (yielded == iterArg)
    return false;

  vector::TransferWriteOp write = getFoldableYieldWrite(yielded, loop);
  if (!write)
    return false;

  // The write must produce exactly the yielded value (single use into yield).
  if (!write.getResult().hasOneUse())
    return false;

  // Types must match so the destination swap is a pure rewrite.
  if (write.getBase().getType() != iterArg.getType())
    return false;

  if (!readsPrecedeWrite(loop, idx, write, dominance))
    return false;

  // Redirect the write's destination from the fresh empty to the iter_arg.
  // The now-dead empty is left for later DCE/canonicalization.
  write.getBaseMutable().assign(iterArg);
  return true;
}

namespace {
struct LoopIterArgDestinationFoldingPass
    : public bufferization::impl::LoopIterArgDestinationFoldingPassBase<
          LoopIterArgDestinationFoldingPass> {
  void runOnOperation() override {
    DominanceInfo dominance(getOperation());
    getOperation()->walk([&](ForOp loop) {
      for (unsigned i = 0, e = loop.getInitArgs().size(); i < e; ++i)
        (void)tryFoldIterArg(loop, i, dominance);
    });
  }
};
} // namespace
