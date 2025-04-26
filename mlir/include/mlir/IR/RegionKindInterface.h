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
/// a terminator (e.g. scf.break / scf.continue) can propagate through
/// this op on its way to the addressed HasBreakingControlFlowOpInterface
/// ancestor. The op does NOT consume the break; it simply passes it upward.
/// All intermediate ops that receive and forward the break request must carry
/// this trait.
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

/// Summary of breaking terminator operations nested under an op.
struct NestedBreakingControlFlowInfo {
  /// Breaking terminators that may target the queried op directly. These
  /// transfer control to the queried op rather than only to an intermediate
  /// PropagateControlFlowBreak op.
  SmallVector<Operation *> predecessors;

  /// True if any direct predecessor targets the queried op from below the
  /// queried op's immediate region.
  bool hasNestedPredecessors = false;

  /// True if any breaking terminator under the queried op targets one of the
  /// queried op's ancestors, escaping through the queried op.
  bool hasBreakingControlFlowOps = false;
};

/// Collect nested breaking-control-flow information for `op` with a single walk
/// of its nested region tree.
NestedBreakingControlFlowInfo getNestedBreakingControlFlowInfo(Operation *op);

/// Return true if `op` (which implements HasBreakingControlFlowOpInterface)
/// contains at least one breaking terminator that directly targets it from a
/// nested region. Such a terminator is a "nested predecessor" of `op` because
/// control flow may re-enter or exit `op` through a request propagated from a
/// deeply nested site rather than only through an immediately enclosing
/// terminator.
bool hasNestedPredecessors(Operation *op);

/// Return true if `op` contains any breaking terminator that would "break
/// through" `op` towards an outer HasBreakingControlFlowOpInterface ancestor.
/// This is used to detect whether an op's post-dominance is disrupted by an
/// early-exit request that propagates through it.
bool hasBreakingControlFlowOps(Operation *op);

/// Collect all breaking terminators nested inside `op` that potentially
/// directly target `op`. These are the ops that may transfer control flow to
/// `op` on an early exit.
void collectAllNestedPredecessors(Operation *op,
                                  SmallVector<Operation *> &predecessors);
} // namespace mlir

#include "mlir/IR/RegionKindInterface.h.inc"

namespace mlir {
namespace detail {
/// Implementation helper for visitNestedBreakingControlFlowOps. Walks the
/// regions of `op` and invokes `callback` for every breaking terminator that
/// either targets `op` or propagates further upward through `op`.
/// The `nestedLevel` argument passed to the callback is the 1-based depth of
/// the terminator relative to `op`'s outermost region.
void visitNestedBreakingControlFlowOpsImpl(
    Operation *op,
    function_ref<WalkResult(RegionExitTerminatorOpInterface, int nestedLevel)>
        callback);
} // namespace detail

/// Walk all breaking terminators that are relevant to breaking control
/// flow inside `op` (see visitNestedBreakingControlFlowOpsImpl). The callback
/// receives the terminator op and its 1-based nesting level. Callbacks
/// returning WalkResult support early termination via WalkResult::interrupt();
/// void-returning callbacks always continue.
template <typename CallbackT>
void visitNestedBreakingControlFlowOps(Operation *op, CallbackT &&callback) {
  using RetT =
      decltype(callback(std::declval<Operation *>(), std::declval<int>()));
  if constexpr (std::is_same_v<RetT, WalkResult>) {
    detail::visitNestedBreakingControlFlowOpsImpl(op, callback);
  } else {
    detail::visitNestedBreakingControlFlowOpsImpl(
        op, [&](RegionExitTerminatorOpInterface visitedOp, int nestedLevel) {
          callback(visitedOp, nestedLevel);
          return WalkResult::advance();
        });
  }
}

/// Walk all breaking terminators relevant to breaking control flow
/// across all top-level ops in `region`.
template <typename CallbackT>
void visitNestedBreakingControlFlowOps(Region &region, CallbackT &&callback) {
  // Pass `callback` as an lvalue: it is reused across iterations, so it must
  // not be forwarded (which could move from it on the first iteration).
  for (Operation &op : region.getOps())
    visitNestedBreakingControlFlowOps(&op, callback);
}

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

/// Return the HasBreakingControlFlowOpInterface operations potentially
/// addressed by the given terminator. Returns an empty vector for non-breaking
/// terminators or malformed target designators.
SmallVector<HasBreakingControlFlowOpInterface>
findPotentialBreakTargets(Operation *terminator);

} // namespace mlir

#endif // MLIR_IR_REGIONKINDINTERFACE_H_
