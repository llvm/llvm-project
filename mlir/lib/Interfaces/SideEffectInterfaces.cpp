//===- SideEffectInterfaces.cpp - SideEffects in MLIR ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallPtrSet.h"

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
  return isa<Allocate, Free, Read, Write>(effect);
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
  // The set of operations to consider when checking for side effects.
  SmallVector<Operation *, 1> effectingOps(1, rootOp);
  while (!effectingOps.empty()) {
    Operation *op = effectingOps.pop_back_val();

    // If the operation has recursive effects, push all of the nested operations
    // on to the stack to consider.
    bool hasRecursiveEffects =
        op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (hasRecursiveEffects) {
      for (Region &region : op->getRegions()) {
        for (auto &block : region) {
          for (auto &nestedOp : block)
            effectingOps.push_back(&nestedOp);
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

      // Otherwise, if the op has recursive side effects we can treat the
      // operation itself as having no effects.
    }
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
bool mlir::hasSingleEffect(Operation *op, Value value) {
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memOp.getEffects(effects);
  bool hasSingleEffectOnVal = false;
  // Iterate through `effects` and check if an effect of type `EffectTy` and
  // only of that type is present. A `value` to check the effect on may or may
  // not have been provided.
  for (auto &effect : effects) {
    if (value && effect.getValue() != value)
      continue;
    hasSingleEffectOnVal = isa<EffectTy>(effect.getEffect());
    if (!hasSingleEffectOnVal)
      return false;
  }
  return hasSingleEffectOnVal;
}

template bool mlir::hasSingleEffect<MemoryEffects::Allocate>(Operation *,
                                                             Value);
template bool mlir::hasSingleEffect<MemoryEffects::Free>(Operation *, Value);
template bool mlir::hasSingleEffect<MemoryEffects::Read>(Operation *, Value);
template bool mlir::hasSingleEffect<MemoryEffects::Write>(Operation *, Value);

template <typename... EffectTys>
bool mlir::hasEffect(Operation *op, Value value) {
  auto memOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memOp)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memOp.getEffects(effects);
  return llvm::any_of(effects, [&](MemoryEffects::EffectInstance &effect) {
    if (value && effect.getValue() != value)
      return false;
    return isa<EffectTys...>(effect.getEffect());
  });
}
template bool mlir::hasEffect<MemoryEffects::Allocate>(Operation *, Value);
template bool mlir::hasEffect<MemoryEffects::Free>(Operation *, Value);
template bool mlir::hasEffect<MemoryEffects::Read>(Operation *, Value);
template bool mlir::hasEffect<MemoryEffects::Write>(Operation *, Value);
template bool
mlir::hasEffect<MemoryEffects::Write, MemoryEffects::Free>(Operation *, Value);

bool mlir::wouldOpBeTriviallyDead(Operation *op) {
  if (op->mightHaveTrait<OpTrait::IsTerminator>())
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
