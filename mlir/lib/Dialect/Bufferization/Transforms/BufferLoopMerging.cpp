//===- BufferLoopMerging.cpp - Merge converging scf.for buffers -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements merging of `scf.for` memref iter_args that converge onto
// a single buffer after the first iteration.
//
// Code that stages a value through scratch memory across loop iterations often
// threads one buffer in as the initial value and yields a *different* buffer
// from the body:
//
//   %yield = memref.alloca()
//   %init  = memref.alloca()
//   store %v, %init[]
//   %r = scf.for ... iter_args(%it = %init) -> (memref<f32>) {
//     %x = load %it[]           // %init on iteration 0, %yield afterwards
//     store f(%x), %yield[]
//     scf.yield %yield
//   }
//
// The iter_arg is not loop-invariant, so the `scf.for` remains a blocking use of
// both allocations and Mem2Reg abandons them (`scf.for` implements
// `PromotableRegionOpInterface`, but is not itself a promotable or aliasing op).
// The buffers are nonetheless interchangeable: `%it` denotes `%init` only on the
// first iteration and `%yield` on every later one, and no iteration reads a
// buffer it has not first written except through `%it`. Rewriting all uses of
// `%yield` to `%init` makes the iter_arg loop-invariant, after which existing
// canonicalization drops it and Mem2Reg promotes the slot.
//
// Merging redirects the body's stores from `%yield` into `%init`, so it is only
// sound when the buffers cannot be distinguished outside the loop after that
// redirection: every non-threading use of `%init` must strictly precede the loop
// (its contents change from the loop onward), and `%yield` must never be written
// outside the loop (such a write would be lost or reordered).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_BUFFERLOOPMERGINGPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::scf;

namespace {

/// How an operation uses a buffer value.
enum class BufferUseKind {
  /// The use is the `scf.yield` of the loop being considered, or an init operand
  /// of the loop itself. Both are the threading of the buffer through the
  /// iter_arg, which this transform is rewriting.
  Yield,
  /// The operation only reads the buffer contents.
  Read,
  /// The operation only writes the buffer contents.
  Write,
  /// The operation both reads and writes the buffer contents.
  ReadWrite,
  /// The operation may expose the buffer's identity (view, cast, call, ...) or
  /// its effects on the buffer cannot be determined.
  Opaque
};

} // namespace

/// Classifies how `use` accesses the buffer it refers to. Only operations whose
/// memory effects are fully known and limited to reads and writes of the operand
/// are safe to redirect; anything else may observe the buffer's address rather
/// than just its contents, which merging does not preserve.
static BufferUseKind classifyBufferUse(OpOperand &use, ForOp loop) {
  Operation *user = use.getOwner();
  if (isa<scf::YieldOp>(user) && user->getParentOp() == loop)
    return BufferUseKind::Yield;
  // The loop's own init operand: the caller has already verified that this
  // threading is the converging pattern being merged.
  if (user == loop.getOperation())
    return BufferUseKind::Yield;

  auto effectOp = dyn_cast<MemoryEffectOpInterface>(user);
  if (!effectOp)
    return BufferUseKind::Opaque;

  // Operations with regions could hide accesses that the effect list does not
  // attribute to this operand.
  if (user->getNumRegions() != 0)
    return BufferUseKind::Opaque;

  SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffects(effects);

  bool reads = false;
  bool writes = false;
  for (const MemoryEffects::EffectInstance &effect : effects) {
    // An effect that is not pinned to a specific value may apply to this
    // buffer; conservatively treat the operation as opaque.
    Value effectValue = effect.getValue();
    if (!effectValue)
      return BufferUseKind::Opaque;
    if (effectValue != use.get())
      continue;
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      reads = true;
      continue;
    }
    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      writes = true;
      continue;
    }
    // Allocate/Free on the buffer means the operation controls its lifetime.
    return BufferUseKind::Opaque;
  }

  // A use that carries no effect on the buffer is a plain capture of the
  // pointer, e.g. a view or cast that forwards it.
  if (!reads && !writes)
    return BufferUseKind::Opaque;
  if (reads && writes)
    return BufferUseKind::ReadWrite;
  return reads ? BufferUseKind::Read : BufferUseKind::Write;
}

/// Returns true if `value` is produced by an allocation whose lifetime is the
/// enclosing scope and which cannot be freed explicitly. Only `memref.alloca`
/// qualifies, so redirecting its uses is not observable outside the scope.
static bool isMergeableAlloc(Value value) {
  return value.getDefiningOp<memref::AllocaOp>() != nullptr;
}

/// Attempts to merge a converging memref iter_arg of `loop` at `argIdx`.
/// Returns true if the IR was modified.
static bool tryMergeIterArg(ForOp loop, unsigned argIdx,
                            DominanceInfo &dominance) {
  Value init = loop.getInitArgs()[argIdx];
  auto yieldOp = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  Value yielded = yieldOp.getOperands()[argIdx];
  Value iterArg = loop.getRegionIterArgs()[argIdx];

  // Only memref-typed slots participate; the merge is about buffer identity.
  if (!isa<MemRefType>(init.getType()))
    return false;

  // Already loop-invariant: canonicalization handles this case on its own.
  if (yielded == init)
    return false;

  // A yielded value derived from the block argument means the rotation is
  // genuinely dynamic (a ping-pong swap), which this transform cannot merge.
  if (yielded == iterArg)
    return false;

  if (!isMergeableAlloc(init) || !isMergeableAlloc(yielded))
    return false;

  // Identical types keep the rewrite a pure use replacement.
  if (init.getType() != yielded.getType())
    return false;

  // Both allocations must dominate the loop. One defined in the body would be
  // fresh per iteration and could not be unified with the initial buffer.
  if (!loop.isDefinedOutsideOfLoop(init) ||
      !loop.isDefinedOutsideOfLoop(yielded))
    return false;

  // The two buffers must not be threaded through any other iter_arg of this
  // loop, where they could be rotated under a different schedule.
  for (unsigned i = 0, e = loop.getInitArgs().size(); i < e; ++i) {
    if (i == argIdx)
      continue;
    if (loop.getInitArgs()[i] == init || loop.getInitArgs()[i] == yielded ||
        yieldOp.getOperands()[i] == init || yieldOp.getOperands()[i] == yielded)
      return false;
  }

  // Merging collapses two buffers into one, so the two buffers must remain
  // indistinguishable once the body's stores are redirected from `yielded` into
  // `init`. The two allocations play asymmetric roles:
  //
  //   * `init` is read inside the body only through the iter_arg. Any direct
  //     in-loop use would, after merging, observe this iteration's own store.
  //     Outside the loop, redirecting the body's stores into `init` changes its
  //     contents from the loop onward, so every non-threading use must strictly
  //     precede the loop.
  //   * `yielded` is written inside the body to define what the next iteration
  //     reads through the iter_arg; only pure writes are allowed there. Outside
  //     the loop it may be read (a read observes the same bytes as `init` after
  //     merging) but never written -- an outside write would be lost.
  auto usesAreMergeable = [&](Value buffer, bool isInit) {
    for (OpOperand &use : buffer.getUses()) {
      BufferUseKind kind = classifyBufferUse(use, loop);
      if (kind == BufferUseKind::Opaque)
        return false;
      // The threading through the iter_arg is what this transform rewrites.
      if (kind == BufferUseKind::Yield)
        continue;

      if (loop->isProperAncestor(use.getOwner())) {
        // Inside the loop body.
        if (isInit)
          return false;
        if (kind != BufferUseKind::Write)
          return false;
        continue;
      }

      // Outside the loop.
      if (isInit) {
        if (!dominance.properlyDominates(use.getOwner(), loop))
          return false;
      } else if (kind != BufferUseKind::Read) {
        return false;
      }
    }
    return true;
  };

  if (!usesAreMergeable(init, /*isInit=*/true) ||
      !usesAreMergeable(yielded, /*isInit=*/false))
    return false;

  // All preconditions hold: fold `yielded` into `init`. This makes the iter_arg
  // loop-invariant and leaves `yielded` dead.
  yielded.replaceAllUsesWith(init);
  return true;
}

namespace {
struct BufferLoopMergingPass
    : public bufferization::impl::BufferLoopMergingPassBase<
          BufferLoopMergingPass> {
  void runOnOperation() override {
    DominanceInfo dominance(getOperation());
    getOperation()->walk([&](ForOp loop) {
      for (unsigned i = 0, e = loop.getInitArgs().size(); i < e; ++i)
        (void)tryMergeIterArg(loop, i, dominance);
    });
  }
};
} // namespace
