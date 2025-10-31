//===- OptimizeAllocationLiveness.cpp - impl. optimize allocation liveness pass
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for optimizing allocation liveness.
// The pass moves the deallocation operation after the last user of the
// allocated buffer.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "optimize-allocation-liveness"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_OPTIMIZEALLOCATIONLIVENESSPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Return true if `a` happens before `b`, i.e., `a` or one of its ancestors
/// properly dominates `b` and `b` is not inside `a`.
static bool happensBefore(Operation *a, Operation *b) {
  do {
    if (a->isProperAncestor(b))
      return false;
    if (Operation *bAncestor = a->getBlock()->findAncestorOpInBlock(*b)) {
      return a->isBeforeInBlock(bAncestor);
    }
  } while ((a = a->getParentOp()));
  return false;
}

/// This method searches for a user of value that is a dealloc operation.
/// If multiple users with free effect are found, return nullptr.
Operation *findUserWithFreeSideEffect(Value value) {
  Operation *freeOpUser = nullptr;
  for (Operation *user : value.getUsers()) {
    if (MemoryEffectOpInterface memEffectOp =
            dyn_cast<MemoryEffectOpInterface>(user)) {
      SmallVector<MemoryEffects::EffectInstance, 2> effects;
      memEffectOp.getEffects(effects);

      for (const auto &effect : effects) {
        if (isa<MemoryEffects::Free>(effect.getEffect())) {
          if (freeOpUser) {
            LDBG() << "Multiple users with free effect found: " << *freeOpUser
                   << " and " << *user;
            return nullptr;
          }
          freeOpUser = user;
        }
      }
    }
  }
  return freeOpUser;
}

/// Checks if the given op allocates memory.
static bool hasMemoryAllocEffect(MemoryEffectOpInterface memEffectOp) {
  SmallVector<MemoryEffects::EffectInstance, 2> effects;
  memEffectOp.getEffects(effects);
  for (const auto &effect : effects) {
    if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
      return true;
    }
  }
  return false;
}

/// Extracts OpResult's with Allocate effects from given op
static SmallVector<OpResult>
collectAllocations(MemoryEffectOpInterface allocOp) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  allocOp.getEffects(effects);
  SmallVector<OpResult> allocResults;
  for (const MemoryEffects::EffectInstance &it : effects)
    if (isa<MemoryEffects::Allocate>(it.getEffect()))
      if (auto val = it.getValue(); val && val.getDefiningOp() == allocOp)
        allocResults.push_back(cast<OpResult>(val));
  return allocResults;
}

struct OptimizeAllocationLiveness
    : public bufferization::impl::OptimizeAllocationLivenessPassBase<
          OptimizeAllocationLiveness> {
public:
  OptimizeAllocationLiveness() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.isExternal())
      return;

    BufferViewFlowAnalysis analysis = BufferViewFlowAnalysis(func);

    func.walk([&](MemoryEffectOpInterface memEffectOp) -> WalkResult {
      if (!hasMemoryAllocEffect(memEffectOp))
        return WalkResult::advance();

      auto allocOp = memEffectOp;
      LDBG() << "Checking alloc op: " << allocOp;

      SmallVector<OpResult> allocationResults = collectAllocations(allocOp);
      // Multiple allocations from a single op are not considered here yet.
      if (allocationResults.size() != 1)
        return WalkResult::advance();

      OpResult allocResult = allocationResults[0];
      LDBG() << "On allocation result: " << allocResult;

      auto *deallocOp = findUserWithFreeSideEffect(allocResult);
      if (!deallocOp || (deallocOp->getBlock() != allocOp->getBlock())) {
        // The pass handles allocations that have a single dealloc op in the
        // same block. We also should not hoist the dealloc op out of
        // conditionals.
        return WalkResult::advance();
      }

      Operation *lastUser = nullptr;
      const BufferViewFlowAnalysis::ValueSetT &deps =
          analysis.resolve(allocResult);
      for (auto dep : llvm::make_early_inc_range(deps)) {
        for (auto *user : dep.getUsers()) {
          // We are looking for a non dealloc op user.
          // check if user is the dealloc op itself.
          if (user == deallocOp)
            continue;

          // find the ancestor of user that is in the same block as the allocOp.
          auto *topUser = allocOp->getBlock()->findAncestorOpInBlock(*user);
          if (!lastUser || happensBefore(lastUser, topUser)) {
            lastUser = topUser;
          }
        }
      }
      if (lastUser == nullptr) {
        return WalkResult::advance();
      }
      LDBG() << "Last user found: " << *lastUser;
      assert(lastUser->getBlock() == allocOp->getBlock());
      assert(lastUser->getBlock() == deallocOp->getBlock());
      // Move the dealloc op after the last user.
      deallocOp->moveAfter(lastUser);
      LDBG() << "Moved dealloc op after: " << *lastUser;

      return WalkResult::advance();
    });
  }
};

} // end anonymous namespace
