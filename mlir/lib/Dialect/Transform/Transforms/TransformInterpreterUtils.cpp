//===- TransformInterpreterUtils.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lightweight transform dialect interpreter utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "transform-dialect-interpreter-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

LogicalResult transform::detail::parseTransformModuleFromFile(
    MLIRContext *context, llvm::StringRef transformFileName,
    OwningOpRef<ModuleOp> &transformModule) {
  if (transformFileName.empty()) {
    LLVM_DEBUG(
        DBGS() << "no transform file name specified, assuming the transform "
                  "module is embedded in the IR next to the top-level\n");
    return success();
  }
  // Parse transformFileName content into a ModuleOp.
  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(transformFileName, &errorMessage);
  if (!memoryBuffer) {
    return emitError(FileLineColLoc::get(
               StringAttr::get(context, transformFileName), 0, 0))
           << "failed to open transform file: " << errorMessage;
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  transformModule =
      OwningOpRef<ModuleOp>(parseSourceFile<ModuleOp>(sourceMgr, context));
  return mlir::verify(*transformModule);
}

ModuleOp transform::detail::getPreloadedTransformModule(MLIRContext *context) {
  auto preloadedLibraryRange =
      context->getOrLoadDialect<transform::TransformDialect>()
          ->getLibraryModules();
  if (!preloadedLibraryRange.empty())
    return *preloadedLibraryRange.begin();
  return ModuleOp();
}

transform::TransformOpInterface
transform::detail::findTransformEntryPoint(Operation *root, ModuleOp module,
                                           StringRef entryPoint) {
  SmallVector<Operation *, 2> l{root};
  if (module)
    l.push_back(module);
  for (Operation *op : l) {
    transform::TransformOpInterface transform = nullptr;
    op->walk<WalkOrder::PreOrder>(
        [&](transform::NamedSequenceOp namedSequenceOp) {
          if (namedSequenceOp.getSymName() == entryPoint) {
            transform = cast<transform::TransformOpInterface>(
                namedSequenceOp.getOperation());
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (transform)
      return transform;
  }
  auto diag = root->emitError()
              << "could not find a nested named sequence with name: "
              << entryPoint;
  return nullptr;
}

/// Return whether `func1` can be merged into `func2`. For that to work `func1`
/// has to be a declaration (aka has to be external) and `func2` either has to
/// be a declaration as well, or it has to be public (otherwise, it wouldn't
/// be visible by `func1`).
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

LogicalResult
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
  LLVM_DEBUG(DBGS() << "renaming private symbols to resolve conflicts:\n");
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
      LLVM_DEBUG(DBGS() << "  found @" << name.getValue() << "\n");

      // Check if there is a colliding op in the other module.
      auto collidingOp =
          cast_or_null<SymbolOpInterface>(otherSymbolTable->lookup(name));
      if (!collidingOp)
        continue;

      LLVM_DEBUG(DBGS() << "    collision found for @" << name.getValue());

      // Collisions are fine if both opt are functions and can be merged.
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op),
          collidingFuncOp =
              dyn_cast<FunctionOpInterface>(collidingOp.getOperation());
          funcOp && collidingFuncOp) {
        if (canMergeInto(funcOp, collidingFuncOp) ||
            canMergeInto(collidingFuncOp, funcOp)) {
          LLVM_DEBUG(llvm::dbgs() << " but both ops are functions and "
                                     "will be merged\n");
          continue;
        }

        // If they can't be merged, proceed like any other collision.
        LLVM_DEBUG(llvm::dbgs() << " and both ops are function definitions");
      }

      // Collision can be resolved by renaming if one of the ops is private.
      auto renameToUnique =
          [&](SymbolOpInterface op, SymbolOpInterface otherOp,
              SymbolTable &symbolTable,
              SymbolTable &otherSymbolTable) -> LogicalResult {
        LLVM_DEBUG(llvm::dbgs() << ", renaming\n");
        FailureOr<StringAttr> maybeNewName =
            symbolTable.renameToUnique(op, {&otherSymbolTable});
        if (failed(maybeNewName)) {
          InFlightDiagnostic diag = op->emitError("failed to rename symbol");
          diag.attachNote(otherOp->getLoc())
              << "attempted renaming due to collision with this op";
          return diag;
        }
        LLVM_DEBUG(DBGS() << "      renamed to @" << maybeNewName->getValue()
                          << "\n");
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
      LLVM_DEBUG(llvm::dbgs() << ", emitting error\n");
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
  LLVM_DEBUG(DBGS() << "moving all symbols into target\n");
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
      LLVM_DEBUG(DBGS() << "  moving @" << op.getName());
      op->moveBefore(&target->getRegion(0).front(),
                     target->getRegion(0).front().end());

      // If there is no collision, we are done.
      if (!collidingOp) {
        LLVM_DEBUG(llvm::dbgs() << " without collision\n");
        continue;
      }

      // The two colliding ops must both be functions because we have already
      // emitted errors otherwise earlier.
      auto funcOp = cast<FunctionOpInterface>(op.getOperation());
      auto collidingFuncOp =
          cast<FunctionOpInterface>(collidingOp.getOperation());

      // Both ops are in the target module now and can be treated symmetrically,
      // so w.l.o.g. we can reduce to merging `funcOp` into `collidingFuncOp`.
      if (!canMergeInto(funcOp, collidingFuncOp)) {
        std::swap(funcOp, collidingFuncOp);
      }
      assert(canMergeInto(funcOp, collidingFuncOp));

      LLVM_DEBUG(llvm::dbgs() << " with collision, trying to keep op at "
                              << collidingFuncOp.getLoc() << ":\n"
                              << collidingFuncOp << "\n");

      // Update symbol table. This works with or without the previous `swap`.
      targetSymbolTable.remove(funcOp);
      targetSymbolTable.insert(collidingFuncOp);
      assert(targetSymbolTable.lookup(funcOp.getName()) == collidingFuncOp);

      // Do the actual merging.
      if (failed(mergeInto(funcOp, collidingFuncOp))) {
        return failure();
      }
    }
  }

  if (failed(mlir::verify(target)))
    return target->emitError()
           << "failed to verify target op after merging symbols";

  LLVM_DEBUG(DBGS() << "done merging ops\n");
  return success();
}

LogicalResult transform::applyTransformNamedSequence(
    Operation *payload, ModuleOp transformModule,
    const TransformOptions &options, StringRef entryPoint) {
  Operation *transformRoot =
      detail::findTransformEntryPoint(payload, transformModule, entryPoint);
  if (!transformRoot)
    return failure();

  // `transformModule` may not be modified.
  OwningOpRef<Operation *> clonedTransformModule(transformModule->clone());
  if (transformModule && !transformModule->isAncestor(transformRoot)) {
    if (failed(detail::mergeSymbolsInto(
            SymbolTable::getNearestSymbolTable(transformRoot),
            std::move(clonedTransformModule))))
      return failure();
  }

  // Apply the transform to the IR, do not enforce top-level constraints.
  RaggedArray<MappedValue> noExtraMappings;
  return applyTransforms(payload, cast<TransformOpInterface>(transformRoot),
                         noExtraMappings, options,
                         /*enforceToplevelTransformOp=*/false);
}
