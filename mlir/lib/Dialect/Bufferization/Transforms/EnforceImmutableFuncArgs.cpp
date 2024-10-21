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

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "enforce-immutable-func-args"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_ENFORCEIMMUTABLEFUNCARGS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

// Checks if there is any operation which tries to write
// into `buffer`.
// This method assumes buffer has `MemRefType`.
static bool isWrittenTo(Value buffer);

namespace {
/// This pass allocates a new a buffer for each input argument of the function
/// which is being written to, also copying it into the allocated buffer.
/// This will avoid in place memory updates for the kernel's arguments and
/// make them immutable/read-only buffers.
struct EnforceImmutableFuncArgsPass
    : public bufferization::impl::EnforceImmutableFuncArgsBase<
          EnforceImmutableFuncArgsPass> {
  void runOnOperation() final;
};
} // end anonymous namespace.

static bool isWrittenTo(Value buffer) {
  assert(isa<MemRefType>(buffer.getType()));

  for (auto user : buffer.getUsers()) {
    if (hasEffect<MemoryEffects::Write>(user, buffer))
      return true;
    if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(user)) {
      assert(viewLikeOp->getNumResults() == 1);
      if (isWrittenTo(viewLikeOp->getResult(0)))
        return true;
    }
  }
  return false;
}

void EnforceImmutableFuncArgsPass::runOnOperation() {

  func::FuncOp funcOp = getOperation();

  LDBG("enforcing immutable function arguments in func " << funcOp.getName());

  IRRewriter rewriter(funcOp->getContext());
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  for (auto argument : funcOp.getArguments()) {

    auto argType = dyn_cast<MemRefType>(argument.getType());
    if (!argType) {
      emitError(argument.getLoc(),
                "function has argument with non memref type");
      return signalPassFailure();
    }

    if (!isWrittenTo(argument))
      continue;

    LDBG("Found a function argument is being written to " << argument);
    Value allocatedMemref =
        rewriter.create<memref::AllocOp>(funcOp.getLoc(), argType);
    rewriter.replaceAllUsesWith(argument, allocatedMemref);
    rewriter.create<memref::CopyOp>(funcOp.getLoc(), argument, allocatedMemref);
  }
}

//===----------------------------------------------------------------------===//
// EnforceImmutableFuncArgs construction
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
mlir::bufferization::createEnforceImmutableFuncArgsPass() {
  return std::make_unique<EnforceImmutableFuncArgsPass>();
}
