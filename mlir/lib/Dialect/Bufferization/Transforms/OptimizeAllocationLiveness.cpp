//===- OptimizeAllocationliveness.cpp - impl. for buffer dealloc. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an algorithem for optimization of allocation liveness,
// The algorithm moves the dealloc op to right after the last user of the
// allocation and on the same block as the allocation.
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <utility>

#define DEBUG_TYPE "optimize-allocation-liveness"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_OPTIMIZEALLOCATIONLIVENESS
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
/// TODO find proper location for this function, since its copied from the llvm project.
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

/// This method will find all the users of an op according to given templete
/// user type.
/// TODO find proper location for this helper function.
template <typename T> FailureOr<T> getUserOfType(Value val) {
  auto isTOp = [](Operation *op) { return isa<T>(op); };
  auto userItr = llvm::find_if(val.getUsers(), isTOp);
  if (userItr == val.getUsers().end())
    return failure();
  assert(llvm::count_if(val.getUsers(), isTOp) == 1 &&
         "expecting one user of type T");
  return cast<T>(*userItr);
}

struct OptimizeAllocationliveness
    : public bufferization::impl::OptimizeAllocationlivenessBase<
          OptimizeAllocationliveness> {
public:
  OptimizeAllocationliveness() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.isExternal())
      return;
    if (func.empty() || func.getOps<memref::DeallocOp>().empty())
      return;
    
    BufferViewFlowAnalysis analysis = BufferViewFlowAnalysis(func);
    func.walk([&](memref::AllocOp allocOp) {
      LDBG("Checking alloc op: " << allocOp);

      auto deallocOp = getUserOfType<memref::DeallocOp>(allocOp);
      if (failed(deallocOp)) {
        return WalkResult::advance();
      }

      // Find the last user of the alloc op and its aliases.
      Operation *lastUser = nullptr;
      const BufferViewFlowAnalysis::ValueSetT& deps = analysis.resolve(allocOp.getMemref());
      for (auto dep : llvm::make_early_inc_range(deps)) {
        for (auto user : dep.getUsers()) {
          // We are looking for a non dealloc op user.
          if (isa<memref::DeallocOp>(user))
            continue;
          // Not expecting a return op to be a user of the alloc op.
          if (isa<func::ReturnOp>(user))
            continue;

          // find the ancestor of user that is in the same block as the allocOp.
          auto topUser = allocOp->getBlock()->findAncestorOpInBlock(*user);
          if (!lastUser || happensBefore(lastUser, topUser)) {
            lastUser = topUser;
          }
        }
      }
      if (lastUser == nullptr) {
        return WalkResult::advance();
      }
      LDBG("Last user found: " << *lastUser);
      assert(lastUser->getBlock() == allocOp->getBlock());
      assert(lastUser->getBlock() == (*deallocOp)->getBlock());
      // Move the dealloc op after the last user.
      (*deallocOp)->moveAfter(lastUser);
      LDBG("Moved dealloc op after: " << *lastUser);

      return WalkResult::advance();
    });
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// OptimizeAllocatinliveness construction
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
mlir::bufferization::createOptimizeAllocationlivenessPass() {
  return std::make_unique<OptimizeAllocationliveness>();
}