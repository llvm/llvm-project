//===- CheckUses.cpp - Expensive transform value validity checks ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass that performs expensive opt-in checks for Transform
// dialect values being potentially used after they have been consumed.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Transforms/Passes.h"

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetOperations.h"

namespace mlir {
namespace transform {
#define GEN_PASS_DEF_CHECKUSESPASS
#include "mlir/Dialect/Transform/Transforms/Passes.h.inc"
} // namespace transform
} // namespace mlir

using namespace mlir;

namespace {

/// Returns a reference to a cached set of blocks that are reachable from the
/// given block via edges computed by the `getNextNodes` function. For example,
/// if `getNextNodes` returns successors of a block, this will return the set of
/// reachable blocks; if it returns predecessors of a block, this will return
/// the set of blocks from which the given block can be reached. The block is
/// considered reachable form itself only if there is a cycle.
template <typename FnTy>
const llvm::SmallPtrSet<Block *, 4> &
getReachableImpl(Block *block, FnTy getNextNodes,
                 DenseMap<Block *, llvm::SmallPtrSet<Block *, 4>> &cache) {
  auto it = cache.find(block);
  if (it != cache.end())
    return it->getSecond();

  llvm::SmallPtrSet<Block *, 4> &reachable = cache[block];
  SmallVector<Block *> worklist;
  worklist.push_back(block);
  while (!worklist.empty()) {
    Block *current = worklist.pop_back_val();
    for (Block *predecessor : getNextNodes(current)) {
      // The block is reachable from its transitive predecessors. Only add
      // them to the worklist if they weren't already visited.
      if (reachable.insert(predecessor).second)
        worklist.push_back(predecessor);
    }
  }
  return reachable;
}

/// An analysis that identifies whether a value allocated by a Transform op may
/// be used by another such op after it may have been freed by a third op on
/// some control flow path. This is conceptually similar to a data flow
/// analysis, but relies on side effects related to particular values that
/// currently cannot be modeled by the MLIR data flow analysis framework (also,
/// the lattice element would be rather expensive as it would need to include
/// live and/or freed values for each operation).
///
/// This analysis is conservatively pessimisic: it will consider that a value
/// may be freed if it is freed on any possible control flow path between its
/// allocation and a relevant use, even if the control never actually flows
/// through the operation that frees the value. It also does not differentiate
/// between may- (freed on at least one control flow path) and must-free (freed
/// on all possible control flow paths) because it would require expensive graph
/// algorithms.
///
/// It is intended as an additional non-blocking verification or debugging aid
/// for ops in the Transform dialect. It leverages the requirement for Transform
/// dialect ops to implement the MemoryEffectsOpInterface, and expects the
/// values in the Transform IR to have an allocation effect on the
/// TransformMappingResource when defined.
class TransformOpMemFreeAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformOpMemFreeAnalysis)

  /// Computes the analysis for Transform ops nested in the given operation.
  explicit TransformOpMemFreeAnalysis(Operation *root) {
    root->walk([&](Operation *op) {
      if (isa<transform::TransformOpInterface>(op)) {
        collectFreedValues(op);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }

  /// A list of operations that may be deleting a value. Non-empty list
  /// contextually converts to boolean "true" value.
  class PotentialDeleters {
  public:
    /// Creates an empty list that corresponds to the value being live.
    static PotentialDeleters live() { return PotentialDeleters({}); }

    /// Creates a list from the operations that may be deleting the value.
    static PotentialDeleters maybeFreed(ArrayRef<Operation *> deleters) {
      return PotentialDeleters(deleters);
    }

    /// Converts to "true" if there are operations that may be deleting the
    /// value.
    explicit operator bool() const { return !deleters.empty(); }

    /// Concatenates the lists of operations that may be deleting the value. The
    /// value is known to be live if the reuslting list is still empty.
    PotentialDeleters &operator|=(const PotentialDeleters &other) {
      llvm::append_range(deleters, other.deleters);
      return *this;
    }

    /// Returns the list of ops that may be deleting the value.
    ArrayRef<Operation *> getOps() const { return deleters; }

  private:
    /// Constructs the list from the given operations.
    explicit PotentialDeleters(ArrayRef<Operation *> ops) {
      llvm::append_range(deleters, ops);
    }

    /// The list of operations that may be deleting the value.
    SmallVector<Operation *> deleters;
  };

  /// Returns the list of operations that may be deleting the operand value on
  /// any control flow path between the definition of the value and its use as
  /// the given operand. For the purposes of this analysis, the value is
  /// considered to be allocated at its definition point and never re-allocated.
  PotentialDeleters isUseLive(OpOperand &operand) {
    const llvm::SmallPtrSet<Operation *, 2> &deleters = freedBy[operand.get()];
    if (deleters.empty())
      return live();

#ifndef NDEBUG
    // Check that the definition point actually allocates the value. If the
    // definition is a block argument, it may be just forwarding the operand of
    // the parent op without doing a new allocation, allow that. We currently
    // don't have the capability to analyze region-based control flow here.
    //
    // TODO: when this ported to the dataflow analysis infra, we should have
    // proper support for region-based control flow.
    Operation *valueSource =
        isa<OpResult>(operand.get())
            ? operand.get().getDefiningOp()
            : operand.get().getParentBlock()->getParentOp();
    auto iface = cast<MemoryEffectOpInterface>(valueSource);
    SmallVector<MemoryEffects::EffectInstance> instances;
    iface.getEffectsOnResource(transform::TransformMappingResource::get(),
                               instances);
    assert((isa<BlockArgument>(operand.get()) ||
            hasEffect<MemoryEffects::Allocate>(instances, operand.get())) &&
           "expected the op defining the value to have an allocation effect "
           "on it");
#endif

    // Collect ancestors of the use operation.
    Block *defBlock = operand.get().getParentBlock();
    SmallVector<Operation *> ancestors;
    Operation *ancestor = operand.getOwner();
    do {
      ancestors.push_back(ancestor);
      if (ancestor->getParentRegion() == defBlock->getParent())
        break;
      ancestor = ancestor->getParentOp();
    } while (true);
    std::reverse(ancestors.begin(), ancestors.end());

    // Consider the control flow from the definition point of the value to its
    // use point. If the use is located in some nested region, consider the path
    // from the entry block of the region to the use.
    for (Operation *ancestor : ancestors) {
      // The block should be considered partially if it is the block that
      // contains the definition (allocation) of the value being used, and the
      // value is defined in the middle of the block, i.e., is not a block
      // argument.
      bool isOutermost = ancestor == ancestors.front();
      bool isFromBlockPartial = isOutermost && isa<OpResult>(operand.get());

      // Check if the value may be freed by operations between its definition
      // (allocation) point in its block and the terminator of the block or the
      // ancestor of the use if it is located in the same block. This is only
      // done for partial blocks here, full blocks will be considered below
      // similarly to other blocks.
      if (isFromBlockPartial) {
        bool defUseSameBlock = ancestor->getBlock() == defBlock;
        // Consider all ops from the def to its block terminator, except the
        // when the use is in the same block, in which case only consider the
        // ops until the user.
        if (PotentialDeleters potentialDeleters = isFreedInBlockAfter(
                operand.get().getDefiningOp(), operand.get(),
                defUseSameBlock ? ancestor : nullptr))
          return potentialDeleters;
      }

      // Check if the value may be freed by opeations preceding the ancestor in
      // its block. Skip the check for partial blocks that contain both the
      // definition and the use point, as this has been already checked above.
      if (!isFromBlockPartial || ancestor->getBlock() != defBlock) {
        if (PotentialDeleters potentialDeleters =
                isFreedInBlockBefore(ancestor, operand.get()))
          return potentialDeleters;
      }

      // Check if the value may be freed by operations in any of the blocks
      // between the definition point (in the outermost region) or the entry
      // block of the region (in other regions) and the operand or its ancestor
      // in the region. This includes the entire "form" block if (1) the block
      // has not been considered as partial above and (2) the block can be
      // reached again through some control-flow loop. This includes the entire
      // "to" block if it can be reached form itself through some control-flow
      // cycle, regardless of whether it has been visited before.
      Block *ancestorBlock = ancestor->getBlock();
      Block *from =
          isOutermost ? defBlock : &ancestorBlock->getParent()->front();
      if (PotentialDeleters potentialDeleters =
              isMaybeFreedOnPaths(from, ancestorBlock, operand.get(),
                                  /*alwaysIncludeFrom=*/!isFromBlockPartial))
        return potentialDeleters;
    }
    return live();
  }

private:
  /// Make PotentialDeleters constructors available with shorter names.
  static PotentialDeleters maybeFreed(ArrayRef<Operation *> deleters) {
    return PotentialDeleters::maybeFreed(deleters);
  }
  static PotentialDeleters live() { return PotentialDeleters::live(); }

  /// Returns the list of operations that may be deleting the given value betwen
  /// the first and last operations, non-inclusive. `getNext` indicates the
  /// direction of the traversal.
  PotentialDeleters
  isFreedBetween(Value value, Operation *first, Operation *last,
                 llvm::function_ref<Operation *(Operation *)> getNext) const {
    auto it = freedBy.find(value);
    if (it == freedBy.end())
      return live();
    const llvm::SmallPtrSet<Operation *, 2> &deleters = it->getSecond();
    for (Operation *op = getNext(first); op != last; op = getNext(op)) {
      if (deleters.contains(op))
        return maybeFreed(op);
    }
    return live();
  }

  /// Returns the list of operations that may be deleting the given value
  /// between `root` and `before` values. `root` is expected to be in the same
  /// block as `before` and precede it. If `before` is null, consider all
  /// operations until the end of the block including the terminator.
  PotentialDeleters isFreedInBlockAfter(Operation *root, Value value,
                                        Operation *before = nullptr) const {
    return isFreedBetween(value, root, before,
                          [](Operation *op) { return op->getNextNode(); });
  }

  /// Returns the list of operations that may be deleting the given value
  /// between the entry of the block and the `root` operation.
  PotentialDeleters isFreedInBlockBefore(Operation *root, Value value) const {
    return isFreedBetween(value, root, nullptr,
                          [](Operation *op) { return op->getPrevNode(); });
  }

  /// Returns the list of operations that may be deleting the given value on
  /// any of the control flow paths between the "form" and the "to" block. The
  /// operations from any block visited on any control flow path are
  /// consdiered. The "from" block is considered if there is a control flow
  /// cycle going through it, i.e., if there is a possibility that all
  /// operations in this block are visited or if the `alwaysIncludeFrom` flag is
  /// set. The "to" block is considered only if there is a control flow cycle
  /// going through it.
  PotentialDeleters isMaybeFreedOnPaths(Block *from, Block *to, Value value,
                                        bool alwaysIncludeFrom) {
    // Find all blocks that lie on any path between "from" and "to", i.e., the
    // intersection of blocks reachable from "from" and blocks from which "to"
    // is rechable.
    const llvm::SmallPtrSet<Block *, 4> &sources = getReachableFrom(to);
    if (!sources.contains(from))
      return live();

    llvm::SmallPtrSet<Block *, 4> reachable(getReachable(from));
    llvm::set_intersect(reachable, sources);

    // If requested, include the "from" block that may not be present in the set
    // of visited blocks when there is no cycle going through it.
    if (alwaysIncludeFrom)
      reachable.insert(from);

    // Join potential deleters from all blocks as we don't know here which of
    // the paths through the control flow is taken.
    PotentialDeleters potentialDeleters = live();
    for (Block *block : reachable) {
      for (Operation &op : *block) {
        if (freedBy[value].count(&op))
          potentialDeleters |= maybeFreed(&op);
      }
    }
    return potentialDeleters;
  }

  /// Popualtes `reachable` with the set of blocks that are rechable from the
  /// given block. A block is considered reachable from itself if there is a
  /// cycle in the control-flow graph that invovles the block.
  const llvm::SmallPtrSet<Block *, 4> &getReachable(Block *block) {
    return getReachableImpl(
        block, [](Block *b) { return b->getSuccessors(); }, reachableCache);
  }

  /// Populates `sources` with the set of blocks from which the given block is
  /// reachable.
  const llvm::SmallPtrSet<Block *, 4> &getReachableFrom(Block *block) {
    return getReachableImpl(
        block, [](Block *b) { return b->getPredecessors(); },
        reachableFromCache);
  }

  /// Returns true of `instances` contains an effect of `EffectTy` on `value`.
  template <typename EffectTy>
  static bool hasEffect(ArrayRef<MemoryEffects::EffectInstance> instances,
                        Value value) {
    return llvm::any_of(instances,
                        [&](const MemoryEffects::EffectInstance &instance) {
                          return instance.getValue() == value &&
                                 isa<EffectTy>(instance.getEffect());
                        });
  }

  /// Records the values that are being freed by an operation or any of its
  /// children in `freedBy`.
  void collectFreedValues(Operation *root) {
    SmallVector<MemoryEffects::EffectInstance> instances;
    root->walk([&](Operation *child) {
      // TODO: extend this to conservatively handle operations with undeclared
      // side effects as maybe freeing the operands.
      auto iface = cast<MemoryEffectOpInterface>(child);
      instances.clear();
      iface.getEffectsOnResource(transform::TransformMappingResource::get(),
                                 instances);
      for (Value operand : child->getOperands()) {
        if (hasEffect<MemoryEffects::Free>(instances, operand)) {
          // All parents of the operation that frees a value should be
          // considered as potentially freeing the value as well.
          //
          // TODO: differentiate between must-free/may-free as well as between
          // this op having the effect and children having the effect. This may
          // require some analysis of all control flow paths through the nested
          // regions as well as a mechanism to separate proper side effects from
          // those obtained by nesting.
          Operation *parent = child;
          do {
            freedBy[operand].insert(parent);
            if (parent == root)
              break;
            parent = parent->getParentOp();
          } while (true);
        }
      }
    });
  }

  /// The mapping from a value to operations that have a Free memory effect on
  /// the TransformMappingResource and associated with this value, or to
  /// Transform operations transitively containing such operations.
  DenseMap<Value, llvm::SmallPtrSet<Operation *, 2>> freedBy;

  /// Caches for sets of reachable blocks.
  DenseMap<Block *, llvm::SmallPtrSet<Block *, 4>> reachableCache;
  DenseMap<Block *, llvm::SmallPtrSet<Block *, 4>> reachableFromCache;
};

//// A simple pass that warns about any use of a value by a transform operation
// that may be using the value after it has been freed.
class CheckUsesPass : public transform::impl::CheckUsesPassBase<CheckUsesPass> {
public:
  void runOnOperation() override {
    auto &analysis = getAnalysis<TransformOpMemFreeAnalysis>();

    getOperation()->walk([&](Operation *child) {
      for (OpOperand &operand : child->getOpOperands()) {
        TransformOpMemFreeAnalysis::PotentialDeleters deleters =
            analysis.isUseLive(operand);
        if (!deleters)
          continue;

        InFlightDiagnostic diag = child->emitWarning()
                                  << "operand #" << operand.getOperandNumber()
                                  << " may be used after free";
        diag.attachNote(operand.get().getLoc()) << "allocated here";
        for (Operation *d : deleters.getOps()) {
          diag.attachNote(d->getLoc()) << "freed here";
        }
      }
    });
  }
};

} // namespace

