//===- SideEffectInterfaces.cpp - SideEffects in MLIR ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SideEffectInterfaces.h"

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
  return isa<Allocate>(effect) || isa<Free>(effect) || isa<Read>(effect) ||
         isa<Write>(effect);
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
    bool hasRecursiveEffects = op->hasTrait<OpTrait::HasRecursiveSideEffects>();
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
      if (!llvm::all_of(effects, [op](const MemoryEffects::EffectInstance &it) {
            // We can drop allocations if the value is a result of the
            // operation.
            if (isa<MemoryEffects::Allocate>(it.getEffect()))
              return it.getValue() && it.getValue().getDefiningOp() == op;
            // Otherwise, the effect must be a read.
            return isa<MemoryEffects::Read>(it.getEffect());
          })) {
        return false;
      }
      continue;

      // Otherwise, if the op has recursive side effects we can treat the
      // operation itself as having no effects.
    } else if (hasRecursiveEffects) {
      continue;
    }

    // If there were no effect interfaces, we treat this op as conservatively
    // having effects.
    return false;
  }

  // If we get here, none of the operations had effects that prevented marking
  // 'op' as dead.
  return true;
}

bool mlir::wouldOpBeTriviallyDead(Operation *op) {
  if (!op->isKnownNonTerminator())
    return false;
  return wouldOpBeTriviallyDeadImpl(op);
}
