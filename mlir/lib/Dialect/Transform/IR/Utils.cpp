//===- Utils.cpp - Utils related to the transform dialect -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/Utils.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

using namespace mlir;

#define DEBUG_TYPE "transform-dialect-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

/// Return whether `func1` can be merged into `func2`. For that to work
/// `func1` has to be a declaration (aka has to be external) and `func2`
/// either has to be a declaration as well, or it has to be public (otherwise,
/// it wouldn't be visible by `func1`).
static bool canMergeInto(FunctionOpInterface func1, FunctionOpInterface func2) {
  return func1.isExternal() && (func2.isPublic() || func2.isExternal());
}

/// Merge `func1` into `func2`. The two ops must be inside the same parent op
/// and mergable according to `canMergeInto`. The function erases `func1` such
/// that only `func2` exists when the function returns.
static LogicalResult mergeInto(FunctionOpInterface func1,
                               FunctionOpInterface func2) {
  assert(canMergeInto(func1, func2));
  assert(func1->getParentOp() == func2->getParentOp() &&
         "expected func1 and func2 to be in the same parent op");

  // Check that function signatures match.
  if (func1.getFunctionType() != func2.getFunctionType()) {
    return func1.emitError()
           << "external definition has a mismatching signature ("
           << func2.getFunctionType() << ")";
  }

  // Check and merge argument attributes.
  MLIRContext *context = func1->getContext();
  auto *td = context->getLoadedDialect<transform::TransformDialect>();
  StringAttr consumedName = td->getConsumedAttrName();
  StringAttr readOnlyName = td->getReadOnlyAttrName();
  for (unsigned i = 0, e = func1.getNumArguments(); i < e; ++i) {
    bool isExternalConsumed = func2.getArgAttr(i, consumedName) != nullptr;
    bool isExternalReadonly = func2.getArgAttr(i, readOnlyName) != nullptr;
    bool isConsumed = func1.getArgAttr(i, consumedName) != nullptr;
    bool isReadonly = func1.getArgAttr(i, readOnlyName) != nullptr;
    if (!isExternalConsumed && !isExternalReadonly) {
      if (isConsumed)
        func2.setArgAttr(i, consumedName, UnitAttr::get(context));
      else if (isReadonly)
        func2.setArgAttr(i, readOnlyName, UnitAttr::get(context));
      continue;
    }

    if ((isExternalConsumed && !isConsumed) ||
        (isExternalReadonly && !isReadonly)) {
      return func1.emitError()
             << "external definition has mismatching consumption "
                "annotations for argument #"
             << i;
    }
  }

  // `func1` is the external one, so we can remove it.
  assert(func1.isExternal());
  func1->erase();

  return success();
}

LogicalResult transform::detail::verifyNoRecursionInCallGraph(Operation *root) {
  const mlir::CallGraph callgraph(root);
  for (auto scc = llvm::scc_begin(&callgraph); !scc.isAtEnd(); ++scc) {
    if (!scc.hasCycle())
      continue;

    // Need to check this here additionally because this verification may run
    // before we check the nested operations.
    if ((*scc->begin())->isExternal())
      return root->emitOpError() << "contains a call to an external "
                                    "operation, which is not allowed";

    Operation *first = (*scc->begin())->getCallableRegion()->getParentOp();
    InFlightDiagnostic diag = emitError(first->getLoc())
                              << "recursion not allowed in named sequences";
    for (auto it = std::next(scc->begin()); it != scc->end(); ++it) {
      // Need to check this here additionally because this verification may
      // run before we check the nested operations.
      if ((*it)->isExternal()) {
        return root->emitOpError() << "contains a call to an external "
                                      "operation, which is not allowed";
      }

      Operation *current = (*it)->getCallableRegion()->getParentOp();
      diag.attachNote(current->getLoc()) << "operation on recursion stack";
    }
    return diag;
  }
  return success();
}

LogicalResult
transform::detail::mergeSymbolsInto(Operation *target,
                                    OwningOpRef<Operation *> other) {
  assert(target->hasTrait<OpTrait::SymbolTable>() &&
         "requires target to implement the 'SymbolTable' trait");
  assert(other->hasTrait<OpTrait::SymbolTable>() &&
         "requires target to implement the 'SymbolTable' trait");

  SymbolTable targetSymbolTable(target);
  InlinerInterface inliner(target->getContext());

  // Collect all the functions that are called in `target` that cannot be
  // inlined into `target`.
  SmallPtrSet<Operation *, 1> noInlineCalls;
  target->walk([&](CallOpInterface call) {
    Operation *callable = nullptr;
    CallInterfaceCallable callee = call.getCallableForCallee();
    if (auto symRef = dyn_cast<SymbolRefAttr>(callee)) {
      // Fall back to full resolution for nested symbols, the table is
      // one-level only.
      if (isa<FlatSymbolRefAttr>(symRef))
        callable = targetSymbolTable.lookup(symRef.getLeafReference());
      else
        callable = SymbolTable::lookupNearestSymbolFrom(call, symRef);
    } else if (auto value = dyn_cast<Value>(callee)) {
      callable = value.getDefiningOp();
    }

    if (!callable)
      return;

    if (!inliner.isLegalToInline(call, callable, /*wouldBeCloned=*/false)) {
      noInlineCalls.insert(call.getOperation());
    }
    return;
  });

  SymbolTable otherSymbolTable(*other);

  // Step 1:
  //
  // Rename private symbols in both ops in order to resolve conflicts that can
  // be resolved that way.
  LDBG() << "renaming private symbols to resolve conflicts:";
  // TODO: Do we *actually* need to test in both directions?
  for (auto &&[symbolTable, otherSymbolTable] : llvm::zip(
           SmallVector<SymbolTable *, 2>{&targetSymbolTable, &otherSymbolTable},
           SmallVector<SymbolTable *, 2>{&otherSymbolTable,
                                         &targetSymbolTable})) {
    Operation *symbolTableOp = symbolTable->getOp();
    for (Operation &op : symbolTableOp->getRegion(0).front()) {
      auto symbolOp = dyn_cast<SymbolOpInterface>(op);
      if (!symbolOp)
        continue;
      StringAttr name = symbolOp.getNameAttr();
      LDBG() << "  found @" << name.getValue();

      // Check if there is a colliding op in the other module.
      auto collidingOp =
          cast_or_null<SymbolOpInterface>(otherSymbolTable->lookup(name));
      if (!collidingOp)
        continue;

      LDBG() << "    collision found for @" << name.getValue();

      // Collisions are fine if both opt are functions and can be merged.
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op),
          collidingFuncOp =
              dyn_cast<FunctionOpInterface>(collidingOp.getOperation());
          funcOp && collidingFuncOp) {
        if (canMergeInto(funcOp, collidingFuncOp) ||
            canMergeInto(collidingFuncOp, funcOp)) {
          LDBG() << " but both ops are functions and will be merged";
          continue;
        }

        // If they can't be merged, proceed like any other collision.
        LDBG() << " and both ops are function definitions";
      }

      // Collision can be resolved by renaming if one of the ops is private.
      auto renameToUnique =
          [&](SymbolOpInterface op, SymbolOpInterface otherOp,
              SymbolTable &symbolTable,
              SymbolTable &otherSymbolTable) -> LogicalResult {
        LDBG() << ", renaming";
        FailureOr<StringAttr> maybeNewName =
            symbolTable.renameToUnique(op, {&otherSymbolTable});
        if (failed(maybeNewName)) {
          InFlightDiagnostic diag = op->emitError("failed to rename symbol");
          diag.attachNote(otherOp->getLoc())
              << "attempted renaming due to collision with this op";
          return diag;
        }
        LDBG() << "      renamed to @" << maybeNewName->getValue();
        return success();
      };

      if (symbolOp.isPrivate()) {
        if (failed(renameToUnique(symbolOp, collidingOp, *symbolTable,
                                  *otherSymbolTable)))
          return failure();
        continue;
      }
      if (collidingOp.isPrivate()) {
        if (failed(renameToUnique(collidingOp, symbolOp, *otherSymbolTable,
                                  *symbolTable)))
          return failure();
        continue;
      }
      LDBG() << ", emitting error";
      InFlightDiagnostic diag = symbolOp.emitError()
                                << "doubly defined symbol @" << name.getValue();
      diag.attachNote(collidingOp->getLoc()) << "previously defined here";
      return diag;
    }
  }

  // We only modified symbols above, so there is no need to verify everything
  // again, just the symbol table.
  for (auto *op : SmallVector<Operation *>{target, *other}) {
    if (failed(mlir::detail::verifySymbolTable(op)))
      return op->emitError()
             << "failed to verify symbol table after symbol renaming";
  }

  // Step 2:
  //
  // Move all ops from `other` into target and merge public symbols.
  LDBG() << "moving all symbols into target";
  {
    SmallVector<SymbolOpInterface> opsToMove;
    for (Operation &op : other->getRegion(0).front()) {
      if (auto symbol = dyn_cast<SymbolOpInterface>(op))
        opsToMove.push_back(symbol);
    }

    for (SymbolOpInterface op : opsToMove) {
      // Remember potentially colliding op in the target module.
      auto collidingOp = cast_or_null<SymbolOpInterface>(
          targetSymbolTable.lookup(op.getNameAttr()));

      // Move op even if we get a collision.
      LDBG() << "  moving @" << op.getName();
      op->moveBefore(&target->getRegion(0).front(),
                     target->getRegion(0).front().end());

      // If there is no collision, we are done -- keep the target symbol
      // table in sync with the moved op so that subsequent lookups (and the
      // post-merge validation below) remain efficient.
      if (!collidingOp) {
        LDBG() << " without collision";
        targetSymbolTable.insert(op);
        continue;
      }

      // The two colliding ops must both be functions because we have already
      // emitted errors otherwise earlier.
      auto funcOp = cast<FunctionOpInterface>(op.getOperation());
      auto collidingFuncOp =
          cast<FunctionOpInterface>(collidingOp.getOperation());

      // Both ops are in the target module now and can be treated
      // symmetrically, so w.l.o.g. we can reduce to merging `funcOp` into
      // `collidingFuncOp`.
      if (!canMergeInto(funcOp, collidingFuncOp)) {
        std::swap(funcOp, collidingFuncOp);
      }
      assert(canMergeInto(funcOp, collidingFuncOp));

      LDBG() << " with collision, trying to keep op at "
             << collidingFuncOp.getLoc() << ":\n"
             << collidingFuncOp;

      // Update symbol table. This works with or without the previous `swap`.
      targetSymbolTable.remove(funcOp);
      targetSymbolTable.insert(collidingFuncOp);
      assert(targetSymbolTable.lookup(funcOp.getName()) == collidingFuncOp);

      // Do the actual merging.
      if (failed(mergeInto(funcOp, collidingFuncOp)))
        return failure();
    }
  }

  // Symbol merging only moves callable ops between symbol tables; it does not
  // alter the bodies that were already valid in the source modules. The only
  // invariants that may newly be violated after merging are:
  //   1. a call now refers to a callee whose body is structurally not legal to
  //      inline at the call site (caught by the transform dialect's
  //      `DialectInlinerInterface` implementation), or
  //   2. the merged call graph contains a recursive cycle, which is forbidden
  //      for `transform.named_sequence` callables (caught by the shared
  //      `verifyNoRecursionInCallGraph` helper).
  // Use the inliner interface methods directly (without running the inlining
  // pass) to validate (1), and reuse the dialect's call-graph verifier for
  // (2). The call graph builder requires call/callable ops to be well-formed,
  // so pre-verify them here without recursing into their bodies.
  WalkResult preVerify = target->walk([](Operation *nested) {
    if (!isa<CallableOpInterface, CallOpInterface>(nested))
      return WalkResult::advance();
    if (failed(mlir::verify(nested, /*verifyRecursively=*/false)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (preVerify.wasInterrupted())
    return failure();

  WalkResult inlineCheck = target->walk([&](CallOpInterface call) {
    Operation *callable = nullptr;
    CallInterfaceCallable callee = call.getCallableForCallee();
    if (auto symRef = dyn_cast<SymbolRefAttr>(callee)) {
      // Fall back to full resolution for nested symbols, the table is
      // one-level only.
      if (isa<FlatSymbolRefAttr>(symRef))
        callable = targetSymbolTable.lookup(symRef.getLeafReference());
      else
        callable = SymbolTable::lookupNearestSymbolFrom(call, symRef);
    } else if (auto value = dyn_cast<Value>(callee)) {
      callable = value.getDefiningOp();
    }

    if (!callable)
      return WalkResult::advance();
    if (!noInlineCalls.contains(call.getOperation()) &&
        !inliner.isLegalToInline(call, callable, /*wouldBeCloned=*/false)) {
      InFlightDiagnostic diag =
          call->emitError()
          << "merged call is not legal to inline into its caller";
      diag.attachNote(callable->getLoc()) << "callee defined here";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (inlineCheck.wasInterrupted())
    return failure();

  LDBG() << "done merging ops";
  return verifyNoRecursionInCallGraph(target);
}
