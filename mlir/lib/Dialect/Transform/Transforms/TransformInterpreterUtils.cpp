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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "transform-dialect-interpreter-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

/// Expands the given list of `paths` to a list of `.mlir` files.
///
/// Each entry in `paths` may either be a regular file, in which case it ends up
/// in the result list, or a directory, in which case all (regular) `.mlir`
/// files in that directory are added. Any other file types lead to a failure.
LogicalResult transform::detail::expandPathsToMLIRFiles(
    ArrayRef<std::string> &paths, MLIRContext *context,
    SmallVectorImpl<std::string> &fileNames) {
  for (const std::string &path : paths) {
    auto loc = FileLineColLoc::get(context, path, 0, 0);

    if (llvm::sys::fs::is_regular_file(path)) {
      LLVM_DEBUG(DBGS() << "Adding '" << path << "' to list of files\n");
      fileNames.push_back(path);
      continue;
    }

    if (!llvm::sys::fs::is_directory(path)) {
      return emitError(loc)
             << "'" << path << "' is neither a file nor a directory";
    }

    LLVM_DEBUG(DBGS() << "Looking for files in '" << path << "':\n");

    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(path, ec), itEnd;
         it != itEnd && !ec; it.increment(ec)) {
      const std::string &fileName = it->path();

      if (it->type() != llvm::sys::fs::file_type::regular_file) {
        LLVM_DEBUG(DBGS() << "  Skipping non-regular file '" << fileName
                          << "'\n");
        continue;
      }

      if (!StringRef(fileName).endswith(".mlir")) {
        LLVM_DEBUG(DBGS() << "  Skipping '" << fileName
                          << "' because it does not end with '.mlir'\n");
        continue;
      }

      LLVM_DEBUG(DBGS() << "  Adding '" << fileName << "' to list of files\n");
      fileNames.push_back(fileName);
    }

    if (ec)
      return emitError(loc) << "error while opening files in '" << path
                            << "': " << ec.message();
  }

  return success();
}

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

LogicalResult transform::detail::assembleTransformLibraryFromPaths(
    MLIRContext *context, ArrayRef<std::string> transformLibraryPaths,
    OwningOpRef<ModuleOp> &transformModule) {
  // Assemble list of library files.
  SmallVector<std::string> libraryFileNames;
  if (failed(detail::expandPathsToMLIRFiles(transformLibraryPaths, context,
                                            libraryFileNames)))
    return failure();

  // Parse modules from library files.
  SmallVector<OwningOpRef<ModuleOp>> parsedLibraries;
  for (const std::string &libraryFileName : libraryFileNames) {
    OwningOpRef<ModuleOp> parsedLibrary;
    if (failed(transform::detail::parseTransformModuleFromFile(
            context, libraryFileName, parsedLibrary)))
      return failure();
    parsedLibraries.push_back(std::move(parsedLibrary));
  }

  // Merge parsed libraries into one module.
  auto loc = FileLineColLoc::get(context, "<shared-library-module>", 0, 0);
  OwningOpRef<ModuleOp> mergedParsedLibraries =
      ModuleOp::create(loc, "__transform");
  {
    mergedParsedLibraries.get()->setAttr("transform.with_named_sequence",
                                         UnitAttr::get(context));
    IRRewriter rewriter(context);
    // TODO: extend `mergeSymbolsInto` to support multiple `other` modules.
    for (OwningOpRef<ModuleOp> &parsedLibrary : parsedLibraries) {
      if (failed(transform::detail::mergeSymbolsInto(
              mergedParsedLibraries.get(), std::move(parsedLibrary))))
        return parsedLibrary->emitError()
               << "failed to merge symbols into shared library module";
    }
  }

  transformModule = std::move(mergedParsedLibraries);
  return success();
}

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
              SymbolTable &otherSymbolTable) -> InFlightDiagnostic {
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

      // Both ops are in the target module now and can be treated
      // symmetrically, so w.l.o.g. we can reduce to merging `funcOp` into
      // `collidingFuncOp`.
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

  LLVM_DEBUG(DBGS() << "done merging ops\n");
  return InFlightDiagnostic();
}

LogicalResult transform::applyTransformNamedSequence(
    Operation *payload, ModuleOp transformModule,
    const TransformOptions &options, StringRef entryPoint) {
  Operation *transformRoot =
      detail::findTransformEntryPoint(payload, transformModule, entryPoint);
  if (!transformRoot) {
    return payload->emitError()
           << "could not find transform entry point: " << entryPoint
           << " in either payload or transform module";
  }

  // `transformModule` may not be modified.
  if (transformModule && !transformModule->isAncestor(transformRoot)) {
    OwningOpRef<Operation *> clonedTransformModule(transformModule->clone());
    if (failed(detail::mergeSymbolsInto(
            SymbolTable::getNearestSymbolTable(transformRoot),
            std::move(clonedTransformModule)))) {
      return payload->emitError() << "failed to merge symbols";
    }
  }

  LLVM_DEBUG(DBGS() << "Apply\n" << *transformRoot << "\n");
  LLVM_DEBUG(DBGS() << "To\n" << *payload << "\n");

  // Apply the transform to the IR, do not enforce top-level constraints.
  RaggedArray<MappedValue> noExtraMappings;
  return applyTransforms(payload, cast<TransformOpInterface>(transformRoot),
                         noExtraMappings, options,
                         /*enforceToplevelTransformOp=*/false);
}
