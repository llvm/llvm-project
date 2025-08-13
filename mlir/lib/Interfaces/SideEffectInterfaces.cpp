//===- SideEffectInterfaces.cpp - SideEffects in MLIR ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <unordered_set>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// SideEffect Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the side effect interfaces.
#include "mlir/Interfaces/SideEffectInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// MemoryEffects
//===----------------------------------------------------------------------===//

bool MemoryEffects::Effect::classof(const SideEffects::Effect *effect) {
  return isa<Allocate, Free, Read, Write, Init>(effect);
}

//===----------------------------------------------------------------------===//
// SideEffect Utilities
//===----------------------------------------------------------------------===//

bool mlir::isOpTriviallyDead(Operation *op) {
  return op->use_empty() && wouldOpBeTriviallyDead(op);
}

/// Internal implementation of `mlir::wouldOpBeTriviallyDead` that also
/// considers terminator operations as dead if they have no side effects. This
/// allows for marking region operations as trivially dead without always being
/// conservative of terminators.
static bool wouldOpBeTriviallyDeadImpl(Operation *rootOp) {
  // The set of operation intervals (end-exclusive) to consider when checking
  // for side effects.
  SmallVector<std::pair<Block::iterator, Block::iterator>, 1> effectingOps = {
      std::make_pair(Block::iterator(rootOp), ++Block::iterator(rootOp))};
  while (!effectingOps.empty()) {
    Block::iterator &it = effectingOps.back().first;
    Block::iterator end = effectingOps.back().second;
    if (it == end) {
      effectingOps.pop_back();
      continue;
    }
    mlir::Operation *op = &*(it++);

    // If the operation has recursive effects, push all of the nested operations
    // on to the stack to consider.
    bool hasRecursiveEffects =
        op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (hasRecursiveEffects) {
      for (Region &region : op->getRegions()) {
        for (auto &block : region) {
          effectingOps.push_back(std::make_pair(block.begin(), block.end()));
        }
      }
    }

    // If the op has memory effects, try to characterize them to see if the op
    // is trivially dead here.
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Check to see if this op either has no effects, or only allocates/reads
      // memory.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      effectInterface.getEffects(effects);

      // Gather all results of this op that are allocated.
      SmallPtrSet<Value, 4> allocResults;
      for (const MemoryEffects::EffectInstance &it : effects)
        if (isa<MemoryEffects::Allocate>(it.getEffect()) && it.getValue() &&
            it.getValue().getDefiningOp() == op)
          allocResults.insert(it.getValue());

      if (!llvm::all_of(effects, [&allocResults](
                                     const MemoryEffects::EffectInstance &it) {
            // We can drop effects if the value is an allocation and is a result
            // of the operation.
            if (allocResults.contains(it.getValue()))
              return true;
            // Otherwise, the effect must be a read.
            return isa<MemoryEffects::Read>(it.getEffect());
          })) {
        return false;
      }
      continue;
    }
    // Otherwise, if the op only has recursive side effects we can treat the
    // operation itself as having no effects. We will visit its children next.
    if (hasRecursiveEffects)
      continue;

    // If there were no effect interfaces, we treat this op as conservatively
    // having effects.
    return false;
  }

  // If we get here, none of the operations had effects that prevented marking
  // 'op' as dead.
  return true;
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
template bool mlir::hasSingleEffect<MemoryEffects::Init>(Operation *);

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
template bool mlir::hasSingleEffect<MemoryEffects::Init>(Operation *,
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
mlir::hasSingleEffect<OpOperand *, MemoryEffects::Init>(Operation *,
                                                        OpOperand *);
template bool
mlir::hasSingleEffect<OpResult, MemoryEffects::Allocate>(Operation *, OpResult);
template bool mlir::hasSingleEffect<OpResult, MemoryEffects::Free>(Operation *,
                                                                   OpResult);
template bool mlir::hasSingleEffect<OpResult, MemoryEffects::Read>(Operation *,
                                                                   OpResult);
template bool mlir::hasSingleEffect<OpResult, MemoryEffects::Write>(Operation *,
                                                                    OpResult);
template bool mlir::hasSingleEffect<OpResult, MemoryEffects::Init>(Operation *,
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
template bool
mlir::hasSingleEffect<BlockArgument, MemoryEffects::Init>(Operation *,
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
template bool mlir::hasEffect<MemoryEffects::Init>(Operation *);
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
template bool mlir::hasEffect<MemoryEffects::Init>(Operation *, Value value);
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
template bool mlir::hasEffect<OpOperand *, MemoryEffects::Init>(Operation *,
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
template bool mlir::hasEffect<OpResult, MemoryEffects::Init>(Operation *,
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
mlir::hasEffect<BlockArgument, MemoryEffects::Init>(Operation *, BlockArgument);
template bool
mlir::hasEffect<BlockArgument, MemoryEffects::Write, MemoryEffects::Free>(
    Operation *, BlockArgument);

bool mlir::wouldOpBeTriviallyDead(Operation *op) {
  if (op->mightHaveTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<SymbolOpInterface>(op))
    return false;
  return wouldOpBeTriviallyDeadImpl(op);
}

bool mlir::isMemoryEffectMovable(Operation *op) {
  return (isMemoryEffectFree(op) || isMemoryInitMovable(op));
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

bool mlir::isMemoryInitMovable(Operation *op) {
  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
  // op does not implement the memory effect op interface
  // meaning it doesn't have any memory init effects and
  // shouldn't be flagged as movable to be conservative
  if (!memInterface) return false;

  // gather all effects on op
  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  memInterface.getEffects(effects);

  // op has interface but no effects, be conservative
  if (effects.empty()) return false;


  std::unordered_map<std::string, int> resources;

  // ensure op only has Init effects and gather unique
  // resource names
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (!isa<MemoryEffects::Init>(effect.getEffect()))
      return false;

    std::string name = effect.getResource()->getName().str();
    resources.try_emplace(name, 0);
  }

  // op itself is good, need to check rest of its parent region
  Operation *parent = op->getParentOp();

  for (Region &region : parent->getRegions())
    for (Operation &op_i : region.getOps())
      if (hasMemoryEffectInitConflict(&op_i, resources))
        return false;

  return true;
}

bool mlir::hasMemoryEffectInitConflict(
    Operation *op, std::unordered_map<std::string, int> &resources) {

  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (!memInterface.hasNoEffect()) {
      llvm::SmallVector<MemoryEffects::EffectInstance> effects;
      memInterface.getEffects(effects);

      // ensure op only has Init effects and gather unique
      // resource names
      for (const MemoryEffects::EffectInstance &effect : effects) {
        if (!isa<MemoryEffects::Init>(effect.getEffect()))
          return true;

        // only care about resources of the op that called
        // this recursive function for the first time
        std::string name = effect.getResource()->getName().str();

        if (resources.find(name) != resources.end())
          if (++resources[name] > 1)
            return true;
      }
      return false;
    }
  }

  // Recurse into the regions and ensure that nested ops don't
  // conflict with each others MemInits
  for (Region &region : op->getRegions())
    for (Operation &op : region.getOps()) 
      if (hasMemoryEffectInitConflict(&op, resources))
        return true;
      
  return false;
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
