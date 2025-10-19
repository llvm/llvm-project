//===- Utils.cpp - Utils related to the transform dialect -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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
static InFlightDiagnostic mergeInto(FunctionOpInterface func1,
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

  return InFlightDiagnostic();
}

InFlightDiagnostic
transform::detail::mergeSymbolsInto(Operation *target,
                                    OwningOpRef<Operation *> other) {
  assert(target->hasTrait<OpTrait::SymbolTable>() &&
         "requires target to implement the 'SymbolTable' trait");
  assert(other->hasTrait<OpTrait::SymbolTable>() &&
         "requires target to implement the 'SymbolTable' trait");

  SymbolTable targetSymbolTable(target);
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
              SymbolTable &otherSymbolTable) -> InFlightDiagnostic {
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
        return InFlightDiagnostic();
      };

      if (symbolOp.isPrivate()) {
        InFlightDiagnostic diag = renameToUnique(
            symbolOp, collidingOp, *symbolTable, *otherSymbolTable);
        if (failed(diag))
          return diag;
        continue;
      }
      if (collidingOp.isPrivate()) {
        InFlightDiagnostic diag = renameToUnique(
            collidingOp, symbolOp, *otherSymbolTable, *symbolTable);
        if (failed(diag))
          return diag;
        continue;
      }
      LDBG() << ", emitting error";
      InFlightDiagnostic diag = symbolOp.emitError()
                                << "doubly defined symbol @" << name.getValue();
      diag.attachNote(collidingOp->getLoc()) << "previously defined here";
      return diag;
    }
  }

  // TODO: This duplicates pass infrastructure. We should split this pass into
  //       several and let the pass infrastructure do the verification.
  for (auto *op : SmallVector<Operation *>{target, *other}) {
    if (failed(mlir::verify(op)))
      return op->emitError() << "failed to verify input op after renaming";
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

      // If there is no collision, we are done.
      if (!collidingOp) {
        LDBG() << " without collision";
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
      {
        InFlightDiagnostic diag = mergeInto(funcOp, collidingFuncOp);
        if (failed(diag))
          return diag;
      }
    }
  }

  if (failed(mlir::verify(target)))
    return target->emitError()
           << "failed to verify target op after merging symbols";

  LDBG() << "done merging ops";
  return InFlightDiagnostic();
}
