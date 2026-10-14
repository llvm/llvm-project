//===- DropEquivalentBufferResults.cpp - Calling convention conversion ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass drops return values from functions if they are equivalent to one of
// their arguments. E.g.:
//
// ```
// func.func @foo(%m : memref<?xf32>) -> (memref<?xf32>) {
//   return %m : memref<?xf32>
// }
// ```
//
// This functions is rewritten to:
//
// ```
// func.func @foo(%m : memref<?xf32>) {
//   return
// }
// ```
//
// All call sites are updated accordingly. If a function returns a cast of a
// function argument, it is also considered equivalent. A cast is inserted at
// the call site in that case.

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_DROPEQUIVALENTBUFFERRESULTSPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;

/// Get all the ReturnOp in the funcOp.
static SmallVector<func::ReturnOp> getReturnOps(func::FuncOp funcOp) {
  SmallVector<func::ReturnOp> returnOps;
  for (Block &b : funcOp.getBody()) {
    if (auto candidateOp = dyn_cast<func::ReturnOp>(b.getTerminator())) {
      returnOps.push_back(candidateOp);
    }
  }
  return returnOps;
}

/// Get the operands at the specified position for all returnOps.
static SmallVector<Value>
getReturnOpsOperandInPos(ArrayRef<func::ReturnOp> returnOps, size_t pos) {
  return llvm::map_to_vector(returnOps, [&](func::ReturnOp returnOp) {
    return returnOp.getOperand(pos);
  });
}

/// Check if all given values are the same buffer as the block argument (modulo
/// cast ops).
static bool operandsEqualFuncArgument(ArrayRef<Value> operands,
                                      BlockArgument argument) {
  for (Value val : operands) {
    while (auto castOp = val.getDefiningOp<memref::CastOp>())
      val = castOp.getSource();

    if (val != argument)
      return false;
  }
  return true;
}

LogicalResult mlir::bufferization::dropEquivalentBufferResults(
    ModuleOp module, DropBufferResultsOpts options) {
  IRRewriter rewriter(module.getContext());

  auto canModify = [&](func::FuncOp funcOp) {
    return funcOp->getParentOp() == module.getOperation() &&
           !funcOp.isExternal() &&
           (!funcOp.isPublic() || options.modifyPublicFunctions);
  };

  DenseMap<func::FuncOp, SmallVector<func::CallOp>> callerMap;
  // Collect the mapping of functions to their call sites.
  module.walk([&](func::CallOp callOp) {
    if (func::FuncOp calledFunc =
            dyn_cast_or_null<func::FuncOp>(callOp.resolveCallable())) {
      if (canModify(calledFunc))
        callerMap[calledFunc].push_back(callOp);
    }
    return WalkResult::advance();
  });

  // Dropping a callee result can make a caller result equivalent to one of the
  // caller's arguments. Revisit such callers until no more results can be
  // dropped. A SetVector avoids adding the same function to the worklist more
  // than once. Every revisit is triggered by deleting a result, so the
  // algorithm also terminates for recursive call graphs.
  llvm::SetVector<func::FuncOp> worklist;
  for (auto funcOp : module.getOps<func::FuncOp>())
    if (canModify(funcOp))
      worklist.insert(funcOp);

  while (!worklist.empty()) {
    func::FuncOp funcOp = worklist.pop_back_val();
    SmallVector<func::ReturnOp> returnOps = getReturnOps(funcOp);
    if (returnOps.empty())
      continue;

    // Compute erased results.
    size_t numReturnOps = returnOps.size();
    size_t numReturnValues = funcOp.getFunctionType().getNumResults();
    SmallVector<SmallVector<Value>> newReturnValues(numReturnOps);
    BitVector erasedResultIndices(numReturnValues);
    DenseMap<int64_t, int64_t> resultToArgs;
    for (size_t i = 0; i < numReturnValues; ++i) {
      bool erased = false;
      SmallVector<Value> returnOperands =
          getReturnOpsOperandInPos(returnOps, i);
      for (BlockArgument bbArg : funcOp.getArguments()) {
        if (operandsEqualFuncArgument(returnOperands, bbArg)) {
          resultToArgs[i] = bbArg.getArgNumber();
          erased = true;
          break;
        }
      }

      if (erased) {
        erasedResultIndices.set(i);
      } else {
        for (auto [newReturnValue, operand] :
             llvm::zip(newReturnValues, returnOperands)) {
          newReturnValue.push_back(operand);
        }
      }
    }

    if (erasedResultIndices.none())
      continue;

    // Update function.
    if (failed(funcOp.eraseResults(erasedResultIndices)))
      return failure();

    for (auto [returnOp, newReturnValue] :
         llvm::zip(returnOps, newReturnValues))
      returnOp.getOperandsMutable().assign(newReturnValue);

    // Update function calls.
    SmallVector<func::CallOp> callOps;
    callOps.swap(callerMap[funcOp]);
    for (func::CallOp callOp : callOps) {
      func::FuncOp caller = callOp->getParentOfType<func::FuncOp>();
      rewriter.setInsertionPoint(callOp);
      auto newCallOp = func::CallOp::create(rewriter, callOp.getLoc(), funcOp,
                                            callOp.getOperands());
      SmallVector<Value> newResults;
      int64_t nextResult = 0;
      for (int64_t i = 0; i < callOp.getNumResults(); ++i) {
        if (!resultToArgs.count(i)) {
          // This result was not erased.
          newResults.push_back(newCallOp.getResult(nextResult++));
          continue;
        }

        // This result was erased.
        Value replacement = callOp.getOperand(resultToArgs[i]);
        Type expectedType = callOp.getResult(i).getType();
        if (replacement.getType() != expectedType) {
          // A cast must be inserted at the call site.
          replacement = memref::CastOp::create(rewriter, callOp.getLoc(),
                                               expectedType, replacement);
        }
        newResults.push_back(replacement);
      }
      rewriter.replaceOp(callOp, newResults);
      callerMap[funcOp].push_back(newCallOp);
      if (caller && canModify(caller))
        worklist.insert(caller);
    }
  }

  return success();
}

namespace {
struct DropEquivalentBufferResultsPass
    : bufferization::impl::DropEquivalentBufferResultsPassBase<
          DropEquivalentBufferResultsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Convert pass options.
    options.modifyPublicFunctions = modifyPublicFunctions;

    if (failed(bufferization::dropEquivalentBufferResults(getOperation(),
                                                          options)))
      return signalPassFailure();
  }

private:
  bufferization::DropBufferResultsOpts options;
};
} // namespace
