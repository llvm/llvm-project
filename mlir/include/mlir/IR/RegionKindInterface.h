//===- RegionKindInterface.h - Region Kind Interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the infer op interfaces defined in
// `RegionKindInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGIONKINDINTERFACE_H_
#define MLIR_IR_REGIONKINDINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// The kinds of regions contained in an operation. SSACFG regions
/// require the SSA-Dominance property to hold. Graph regions do not
/// require SSA-Dominance. If a registered operation does not implement
/// RegionKindInterface, then any regions it contains are assumed to be
/// SSACFG regions.
enum class RegionKind {
  SSACFG,
  Graph,
};

namespace OpTrait {
/// A trait that specifies that an operation only defines graph regions.
template <typename ConcreteType>
class HasOnlyGraphRegion : public TraitBase<ConcreteType, HasOnlyGraphRegion> {
public:
  static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
  static bool hasSSADominance(unsigned index) { return false; }
};

/// Indicates that this operation is transparent to breaking control flow:
/// a RegionTerminator (e.g. scf.break / scf.continue) with
/// num-breaking-regions > 1 can propagate through this op on its way to the
/// enclosing HasBreakingControlFlowOpInterface ancestor. The op does NOT
/// consume the break; it simply passes it upward. All ops that are "skipped
/// over" by a multi-level region terminator must carry this trait.
template <typename ConcreteType>
class PropagateControlFlowBreak
    : public TraitBase<ConcreteType, PropagateControlFlowBreak> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // Verify the operation has regions and can handle breaking control flow
    if (op->getNumRegions() == 0)
      return op->emitOpError(
          "operation with PropagateControlFlowBreak trait must have regions");
    return success();
  }
};

/// Indicates that this operation is a block terminator that can exit multiple
/// nested region levels in one step. The operation must carry a
/// num-breaking-regions value N > 0:
///   N = 1  — exits its own immediately enclosing region (normal yield).
///   N = K  — exits K region levels; the K-1 intermediate parent ops must
///            each carry PropagateControlFlowBreak; the op at the K-th level
///            must implement HasBreakingControlFlowOpInterface.
/// This trait also requires IsTerminator (enforced by verifyTrait).
template <typename ConcreteType>
class RegionTerminator : public TraitBase<ConcreteType, RegionTerminator> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    if (op->getNumBreakingControlRegions() == 0)
      return op->emitOpError("operation with region terminator trait must have "
                             "breaking control regions > 0");
    if (!op->hasTrait<OpTrait::IsTerminator>())
      return op->emitOpError(
          "operation with region terminator trait must be a terminator");
    return success();
  }
};

} // namespace OpTrait

/// Return "true" if the given region may have SSA dominance. This function also
/// returns "true" in case the owner op is an unregistered op or an op that does
/// not implement the RegionKindInterface.
bool mayHaveSSADominance(Region &region);

/// Return "true" if the given region may be a graph region without SSA
/// dominance. This function returns "true" in case the owner op is an
/// unregistered op. It returns "false" if it is a registered op that does not
/// implement the RegionKindInterface.
bool mayBeGraphRegion(Region &region);

/// Return true if `op` (which implements HasBreakingControlFlowOpInterface)
/// contains at least one RegionTerminator that directly targets it — i.e. a
/// terminator whose num-breaking-regions equals the region-nesting depth from
/// `op`'s body to that terminator. Such a terminator is a "nested predecessor"
/// of `op` because control flow may re-enter or exit `op` from a deeply nested
/// site rather than only through the immediately enclosing terminator.
bool hasNestedPredecessors(Operation *op);

/// Return true if `op` contains any RegionTerminator that would "break
/// through" `op` towards an outer HasBreakingControlFlowOpInterface ancestor,
/// i.e. a terminator whose num-breaking-regions exceeds the nesting depth at
/// which it appears inside `op`. This is used to detect whether an op's
/// post-dominance is disrupted by an early-exit path that bypasses it.
bool hasBreakingControlFlowOps(Operation *op);

/// Collect all RegionTerminator operations nested inside `op` that directly
/// target `op` (num-breaking-regions == their region-nesting depth from `op`).
/// These are the ops that will transfer control flow to `op` on an early exit.
void collectAllNestedPredecessors(Operation *op,
                                  SmallVector<Operation *> &predecessors);

namespace detail {
/// Implementation helper for visitNestedBreakingControlFlowOps. Walks the
/// regions of `op` and invokes `callback` for every RegionTerminator whose
/// num-breaking-regions is >= its current nesting depth (i.e. the terminator
/// either targets `op` or propagates further upward through `op`).
/// The `nestedLevel` argument passed to the callback is the 1-based depth of
/// the terminator relative to `op`'s outermost region.
void visitNestedBreakingControlFlowOpsImpl(
    Operation *op,
    function_ref<WalkResult(Operation *, int nestedLevel)> callback);
} // namespace detail

/// Walk all RegionTerminator operations that are relevant to breaking control
/// flow inside `op` (see visitNestedBreakingControlFlowOpsImpl). The callback
/// receives the terminator op and its 1-based nesting level. The WalkResult-
/// returning overload supports early termination via WalkResult::interrupt().
template <typename CallbackT>
std::enable_if_t<
    std::is_same_v<decltype(std::declval<CallbackT>()(
                       std::declval<Operation *>(), std::declval<int>())),
                   WalkResult>>
visitNestedBreakingControlFlowOps(Operation *op, CallbackT &&callback) {
  detail::visitNestedBreakingControlFlowOpsImpl(op, callback);
}

/// Walk all RegionTerminator operations relevant to breaking control flow
/// inside `op`. Void-returning callback overload (no early termination).
template <typename CallbackT>
std::enable_if_t<
    std::is_same_v<decltype(std::declval<CallbackT>()(
                       std::declval<Operation *>(), std::declval<int>())),
                   void>>
visitNestedBreakingControlFlowOps(Operation *op, CallbackT &&callback) {
  detail::visitNestedBreakingControlFlowOpsImpl(
      op, [&](Operation *visitedOp, int nestedLevel) {
        callback(visitedOp, nestedLevel);
        return WalkResult::advance();
      });
}

/// Walk all RegionTerminator operations relevant to breaking control flow
/// across all top-level ops in `region`. WalkResult-returning overload.
template <typename CallbackT>
std::enable_if_t<
    std::is_same_v<decltype(std::declval<CallbackT>()(
                       std::declval<Operation *>(), std::declval<int>())),
                   WalkResult>>
visitNestedBreakingControlFlowOps(Region &region, CallbackT &&callback) {
  for (Operation &op : region.getOps())
    detail::visitNestedBreakingControlFlowOpsImpl(&op, callback);
}

/// Walk all RegionTerminator operations relevant to breaking control flow
/// across all top-level ops in `region`. Void-returning overload.
template <typename CallbackT>
std::enable_if_t<
    std::is_same_v<decltype(std::declval<CallbackT>()(
                       std::declval<Operation *>(), std::declval<int>())),
                   void>>
visitNestedBreakingControlFlowOps(Region &region, CallbackT &&callback) {
  for (Operation &op : region.getOps())
    detail::visitNestedBreakingControlFlowOpsImpl(
        &op, [&](Operation *visitedOp, int nestedLevel) {
          callback(visitedOp, nestedLevel);
          return WalkResult::advance();
        });
}

} // namespace mlir

#include "mlir/IR/RegionKindInterface.h.inc"

namespace mlir {

/// Return true if the given region may contain breaking control flow — either
/// because its parent op propagates breaks (PropagateControlFlowBreak) or
/// because it is the body of a HasBreakingControlFlowOpInterface op. Used to
/// decide whether post-dominance analysis must account for early-exit paths.
inline bool mightHaveBreakingControlFlow(Region *region) {
  Operation *parentOp = region->getParentOp();
  return (!parentOp->isRegistered() ||
          parentOp->hasTrait<OpTrait::PropagateControlFlowBreak>() ||
          isa<HasBreakingControlFlowOpInterface>(parentOp));
}

} // namespace mlir

#endif // MLIR_IR_REGIONKINDINTERFACE_H_
