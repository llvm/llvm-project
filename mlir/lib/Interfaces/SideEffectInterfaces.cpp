//===- SideEffectInterfaces.cpp - SideEffects in MLIR ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/SymbolTable.h"
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// SideEffect Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the side effect interfaces.
#include "mlir/Interfaces/SideEffectInterfaces.cpp.inc"

template MemoryEffects::Allocate *
SideEffects::Effect::get<MemoryEffects::Allocate>();
template MemoryEffects::Free *SideEffects::Effect::get<MemoryEffects::Free>();
template MemoryEffects::Read *SideEffects::Effect::get<MemoryEffects::Read>();
template MemoryEffects::Write *SideEffects::Effect::get<MemoryEffects::Write>();
template SideEffects::DefaultResource *
SideEffects::Resource::Base<SideEffects::DefaultResource>::get();
template SideEffects::AutomaticAllocationScopeResource *
SideEffects::Resource::Base<SideEffects::AutomaticAllocationScopeResource,
                            SideEffects::DefaultResource>::get();

//===----------------------------------------------------------------------===//
// MemoryEffects
//===----------------------------------------------------------------------===//

bool MemoryEffects::Effect::classof(const SideEffects::Effect *effect) {
  return isa<Allocate, Free, Read, Write>(effect);
}

//===----------------------------------------------------------------------===//
// SideEffect Utilities
//===----------------------------------------------------------------------===//

bool mlir::isOpTriviallyDead(Operation *op) {
  return op->use_empty() && wouldOpBeTriviallyDead(op);
}

/// Return true if the op's own memory effects do not prevent erasure. Allowed
/// effects are reads and effects on results allocated by the op itself.
///
/// If the op only has recursive effects, treat the op itself as having no
/// effects. Without a memory effect interface or recursive effects, return
/// false to conservatively assume live effects.
///
/// The caller must also check nested effects when `hasRecursiveEffects` is
/// true. This function does not check uses or restrictions on erasing
/// terminators and symbols.
static bool hasOnlyRemovableEffects(Operation *op, bool hasRecursiveEffects) {
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);

    // Gather all results of this op that are allocated.
    SmallPtrSet<Value, 4> allocResults;
    for (const MemoryEffects::EffectInstance &effect : effects)
      if (isa<MemoryEffects::Allocate>(effect.getEffect()) &&
          effect.getValue() && effect.getValue().getDefiningOp() == op)
        allocResults.insert(effect.getValue());
    return llvm::all_of(effects,
                        [&](const MemoryEffects::EffectInstance &effect) {
                          // We can drop effects on values allocated by this
                          // operation and returned as results. Otherwise, the
                          // effect must be a read.
                          return allocResults.contains(effect.getValue()) ||
                                 isa<MemoryEffects::Read>(effect.getEffect());
                        });
  }
  return hasRecursiveEffects;
}

/// Internal implementation of `mlir::wouldOpBeTriviallyDead` that also
/// considers terminator operations as dead if they have no side effects. This
/// allows for marking region operations as trivially dead without always being
/// conservative of terminators.
static bool wouldOpBeTriviallyDeadImpl(
    Operation *root,
    DenseMap<Operation *, DeadnessCache::State> *cache = nullptr) {
  using State = DeadnessCache::State;
  // Repeated queries on nested operations would otherwise scan the same
  // subtrees. Cache each subtree after checking its effects, and skip cached
  // subtrees on later queries. An explicit worklist also bounds C++ stack use
  // for deeply nested IR. Uncached queries do not look up or record summaries.
  struct WorkItem {
    // Keep the unvisited range so a block can resume after a nested operation
    // without adding every sibling to the worklist.
    Block::iterator current;
    Block::iterator end;
    Operation *complete = nullptr;
  };
  SmallVector<WorkItem, 1> worklist = {
      {Block::iterator(root), ++Block::iterator(root)}};
  while (!worklist.empty()) {
    WorkItem &item = worklist.back();
    if (item.current == item.end) {
      if (cache && item.complete)
        (*cache)[item.complete] = State::Dead;
      worklist.pop_back();
      continue;
    }
    Operation *op = &*item.current++;
    bool cachedAlive = false;
    if (cache) {
      auto cached = cache->find(op);
      if (cached != cache->end()) {
        if (cached->second == State::Dead)
          continue;
        cachedAlive = true;
      }
    }
    bool recursive = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (cachedAlive || !hasOnlyRemovableEffects(op, recursive)) {
      // One live effect keeps every ancestor up to the root live. The
      // remaining subtrees cannot change the answer.
      if (cache) {
        for (Operation *ancestor = op;; ancestor = ancestor->getParentOp()) {
          (*cache)[ancestor] = State::Alive;
          if (ancestor == root)
            break;
        }
      }
      return false;
    }
    if (!recursive) {
      if (cache)
        (*cache)[op] = State::Dead;
      continue;
    }
    // The first pushed block is processed last. Record the parent's summary
    // only when that block is complete and all other blocks have been checked.
    Operation *complete = cache ? op : nullptr;
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        worklist.push_back({block.begin(), block.end(), complete});
        complete = nullptr;
      }
    }
    if (complete)
      (*cache)[complete] = State::Dead;
  }
  return true;
}

bool DeadnessCache::isOpTriviallyDead(Operation *root) {
  // Cache only effects. Uses can change without a change to the defining op.
  // Nested operations may have internal uses or be terminators even when the
  // whole root can be erased, so these checks apply only to the root.
  if (!root->use_empty() || root->mightHaveTrait<OpTrait::IsTerminator>() ||
      isa<SymbolOpInterface>(root))
    return false;

  auto cachedRoot = cache.find(root);
  if (cachedRoot != cache.end())
    return cachedRoot->second == State::Dead;

  return wouldOpBeTriviallyDeadImpl(root, &cache);
}

void DeadnessCache::invalidate(Operation *op) {
  op->walk([&](Operation *nested) { cache.erase(nested); });
  invalidateAncestors(op);
}

void DeadnessCache::invalidateAncestors(Operation *op) {
  // An ancestor may report nested effects through its own effect interface,
  // without querying the intermediate cache entries. A missing entry cannot
  // stop invalidation of its ancestors.
  for (Operation *parent = op; parent; parent = parent->getParentOp())
    cache.erase(parent);
}

template <typename EffectTy>
bool mlir::hasSingleEffect(Operation *op) {
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memOp.getEffects(effects);
  bool hasSingleEffectOnVal = false;
  // Iterate through `effects` and check if an effect of type `EffectTy` and
  // only of that type is present.
  for (auto &effect : effects) {
    hasSingleEffectOnVal = isa<EffectTy>(effect.getEffect());
    if (!hasSingleEffectOnVal)
      return false;
  }
  return hasSingleEffectOnVal;
}
template bool mlir::hasSingleEffect<MemoryEffects::Allocate>(Operation *);
template bool mlir::hasSingleEffect<MemoryEffects::Free>(Operation *);
template bool mlir::hasSingleEffect<MemoryEffects::Read>(Operation *);
template bool mlir::hasSingleEffect<MemoryEffects::Write>(Operation *);

template <typename EffectTy>
bool mlir::hasSingleEffect(Operation *op, Value value) {
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memOp.getEffects(effects);
  bool hasSingleEffectOnVal = false;
  // Iterate through `effects` and check if an effect of type `EffectTy` and
  // only of that type is present.
  for (auto &effect : effects) {
    if (effect.getValue() != value)
      continue;
    hasSingleEffectOnVal = isa<EffectTy>(effect.getEffect());
    if (!hasSingleEffectOnVal)
      return false;
  }
  return hasSingleEffectOnVal;
}

template bool mlir::hasSingleEffect<MemoryEffects::Allocate>(Operation *,
                                                             Value value);
template bool mlir::hasSingleEffect<MemoryEffects::Free>(Operation *,
                                                         Value value);
template bool mlir::hasSingleEffect<MemoryEffects::Read>(Operation *,
                                                         Value value);
template bool mlir::hasSingleEffect<MemoryEffects::Write>(Operation *,
                                                          Value value);

template <typename ValueTy, typename EffectTy>
bool mlir::hasSingleEffect(Operation *op, ValueTy value) {
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memOp.getEffects(effects);
  bool hasSingleEffectOnVal = false;
  // Iterate through `effects` and check if an effect of type `EffectTy` and
  // only of that type is present on value.
  for (auto &effect : effects) {
    if (effect.getEffectValue<ValueTy>() != value)
      continue;
    hasSingleEffectOnVal = isa<EffectTy>(effect.getEffect());
    if (!hasSingleEffectOnVal)
      return false;
  }
  return hasSingleEffectOnVal;
}

template bool
mlir::hasSingleEffect<OpOperand *, MemoryEffects::Allocate>(Operation *,
                                                            OpOperand *);
template bool
mlir::hasSingleEffect<OpOperand *, MemoryEffects::Free>(Operation *,
                                                        OpOperand *);
template bool
mlir::hasSingleEffect<OpOperand *, MemoryEffects::Read>(Operation *,
                                                        OpOperand *);
template bool
mlir::hasSingleEffect<OpOperand *, MemoryEffects::Write>(Operation *,
                                                         OpOperand *);
template bool
mlir::hasSingleEffect<OpResult, MemoryEffects::Allocate>(Operation *, OpResult);
template bool mlir::hasSingleEffect<OpResult, MemoryEffects::Free>(Operation *,
                                                                   OpResult);
template bool mlir::hasSingleEffect<OpResult, MemoryEffects::Read>(Operation *,
                                                                   OpResult);
template bool mlir::hasSingleEffect<OpResult, MemoryEffects::Write>(Operation *,
                                                                    OpResult);
template bool
mlir::hasSingleEffect<BlockArgument, MemoryEffects::Allocate>(Operation *,
                                                              BlockArgument);
template bool
mlir::hasSingleEffect<BlockArgument, MemoryEffects::Free>(Operation *,
                                                          BlockArgument);
template bool
mlir::hasSingleEffect<BlockArgument, MemoryEffects::Read>(Operation *,
                                                          BlockArgument);
template bool
mlir::hasSingleEffect<BlockArgument, MemoryEffects::Write>(Operation *,
                                                           BlockArgument);

template <typename... EffectTys>
bool mlir::hasEffect(Operation *op) {
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memOp.getEffects(effects);
  return llvm::any_of(effects, [&](MemoryEffects::EffectInstance &effect) {
    return isa<EffectTys...>(effect.getEffect());
  });
}
template bool mlir::hasEffect<MemoryEffects::Allocate>(Operation *);
template bool mlir::hasEffect<MemoryEffects::Free>(Operation *);
template bool mlir::hasEffect<MemoryEffects::Read>(Operation *);
template bool mlir::hasEffect<MemoryEffects::Write>(Operation *);
template bool
mlir::hasEffect<MemoryEffects::Write, MemoryEffects::Free>(Operation *);

template <typename... EffectTys>
bool mlir::hasEffect(Operation *op, Value value) {
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memOp.getEffects(effects);
  return llvm::any_of(effects, [&](MemoryEffects::EffectInstance &effect) {
    if (effect.getValue() != value)
      return false;
    return isa<EffectTys...>(effect.getEffect());
  });
}
template bool mlir::hasEffect<MemoryEffects::Allocate>(Operation *,
                                                       Value value);
template bool mlir::hasEffect<MemoryEffects::Free>(Operation *, Value value);
template bool mlir::hasEffect<MemoryEffects::Read>(Operation *, Value value);
template bool mlir::hasEffect<MemoryEffects::Write>(Operation *, Value value);
template bool
mlir::hasEffect<MemoryEffects::Write, MemoryEffects::Free>(Operation *,
                                                           Value value);

template <typename ValueTy, typename... EffectTys>
bool mlir::hasEffect(Operation *op, ValueTy value) {
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memOp.getEffects(effects);
  return llvm::any_of(effects, [&](MemoryEffects::EffectInstance &effect) {
    if (effect.getEffectValue<ValueTy>() != value)
      return false;
    return isa<EffectTys...>(effect.getEffect());
  });
}
template bool
mlir::hasEffect<OpOperand *, MemoryEffects::Allocate>(Operation *, OpOperand *);
template bool mlir::hasEffect<OpOperand *, MemoryEffects::Free>(Operation *,
                                                                OpOperand *);
template bool mlir::hasEffect<OpOperand *, MemoryEffects::Read>(Operation *,
                                                                OpOperand *);
template bool mlir::hasEffect<OpOperand *, MemoryEffects::Write>(Operation *,
                                                                 OpOperand *);
template bool
mlir::hasEffect<OpOperand *, MemoryEffects::Write, MemoryEffects::Free>(
    Operation *, OpOperand *);

template bool mlir::hasEffect<OpResult, MemoryEffects::Allocate>(Operation *,
                                                                 OpResult);
template bool mlir::hasEffect<OpResult, MemoryEffects::Free>(Operation *,
                                                             OpResult);
template bool mlir::hasEffect<OpResult, MemoryEffects::Read>(Operation *,
                                                             OpResult);
template bool mlir::hasEffect<OpResult, MemoryEffects::Write>(Operation *,
                                                              OpResult);
template bool
mlir::hasEffect<OpResult, MemoryEffects::Write, MemoryEffects::Free>(
    Operation *, OpResult);

template bool
mlir::hasEffect<BlockArgument, MemoryEffects::Allocate>(Operation *,
                                                        BlockArgument);
template bool
mlir::hasEffect<BlockArgument, MemoryEffects::Free>(Operation *, BlockArgument);
template bool
mlir::hasEffect<BlockArgument, MemoryEffects::Read>(Operation *, BlockArgument);
template bool
mlir::hasEffect<BlockArgument, MemoryEffects::Write>(Operation *,
                                                     BlockArgument);
template bool
mlir::hasEffect<BlockArgument, MemoryEffects::Write, MemoryEffects::Free>(
    Operation *, BlockArgument);

bool mlir::hasUnknownEffects(Operation *op) {
  return !isa<MemoryEffectOpInterface>(op) &&
         !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
}

bool mlir::wouldOpBeTriviallyDead(Operation *op) {
  if (op->mightHaveTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<SymbolOpInterface>(op))
    return false;
  return wouldOpBeTriviallyDeadImpl(op);
}

bool mlir::isMemoryEffectFree(Operation *op) {
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (!memInterface.hasNoEffect())
      return false;
    // If the op does not have recursive side effects, then it is memory effect
    // free.
    if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return true;
  } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    // Otherwise, if the op does not implement the memory effect interface and
    // it does not have recursive side effects, then it cannot be known that the
    // op is moveable.
    return false;
  }

  // Recurse into the regions and ensure that all nested ops are memory effect
  // free.
  for (Region &region : op->getRegions())
    for (Operation &op : region.getOps())
      if (!isMemoryEffectFree(&op))
        return false;
  return true;
}

// the returned vector may contain duplicate effects
std::optional<llvm::SmallVector<MemoryEffects::EffectInstance>>
mlir::getEffectsRecursively(Operation *rootOp) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  SmallVector<Operation *> effectingOps(1, rootOp);
  while (!effectingOps.empty()) {
    Operation *op = effectingOps.pop_back_val();

    // If the operation has recursive effects, push all of the nested
    // operations on to the stack to consider.
    bool hasRecursiveEffects =
        op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (hasRecursiveEffects) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (Operation &nestedOp : block) {
            effectingOps.push_back(&nestedOp);
          }
        }
      }
    }

    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      effectInterface.getEffects(effects);
    } else if (!hasRecursiveEffects) {
      // the operation does not have recursive memory effects or implement
      // the memory effect op interface. Its effects are unknown.
      return std::nullopt;
    }
  }
  return effects;
}

bool mlir::isSpeculatable(Operation *op) {
  auto conditionallySpeculatable = dyn_cast<ConditionallySpeculatable>(op);
  if (!conditionallySpeculatable)
    return false;

  switch (conditionallySpeculatable.getSpeculatability()) {
  case Speculation::RecursivelySpeculatable:
    for (Region &region : op->getRegions()) {
      for (Operation &op : region.getOps())
        if (!isSpeculatable(&op))
          return false;
    }
    return true;

  case Speculation::Speculatable:
    return true;

  case Speculation::NotSpeculatable:
    return false;
  }

  llvm_unreachable("Unhandled enum in mlir::isSpeculatable!");
}

/// The implementation of this function replicates the `def Pure : TraitList`
/// in `SideEffectInterfaces.td` and has to be kept in sync manually.
bool mlir::isPure(Operation *op) {
  return isSpeculatable(op) && isMemoryEffectFree(op);
}
