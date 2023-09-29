//===- TransformInterpreterPassBase.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class with shared implementation for transform dialect interpreter
// passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "transform-dialect-interpreter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DEBUG_TYPE_DUMP_STDERR "transform-dialect-dump-repro"
#define DEBUG_TYPE_DUMP_FILE "transform-dialect-save-repro"

/// Name of the attribute used for targeting the transform dialect interpreter
/// at specific operations.
constexpr static llvm::StringLiteral kTransformDialectTagAttrName =
    "transform.target_tag";
/// Value of the attribute indicating the root payload operation.
constexpr static llvm::StringLiteral kTransformDialectTagPayloadRootValue =
    "payload_root";
/// Value of the attribute indicating the container of transform operations
/// (containing the top-level transform operation).
constexpr static llvm::StringLiteral
    kTransformDialectTagTransformContainerValue = "transform_container";

namespace {

/// Utility to parse the content of a `transformFileName` MLIR file containing
/// a transform dialect specification.
static LogicalResult
parseTransformModuleFromFile(MLIRContext *context,
                             llvm::StringRef transformFileName,
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
  return success();
}

/// Finds the single top-level transform operation with `root` as ancestor.
/// Reports an error if there is more than one such operation and returns the
/// first one found. Reports an error returns nullptr if no such operation
/// found.
static Operation *findTopLevelTransform(Operation *root,
                                        StringRef filenameOption) {
  ::mlir::transform::TransformOpInterface topLevelTransform = nullptr;
  WalkResult walkResult = root->walk<WalkOrder::PreOrder>(
      [&](::mlir::transform::TransformOpInterface transformOp) {
        if (!transformOp
                 ->hasTrait<transform::PossibleTopLevelTransformOpTrait>())
          return WalkResult::skip();
        if (!topLevelTransform) {
          topLevelTransform = transformOp;
          return WalkResult::skip();
        }
        auto diag = transformOp.emitError()
                    << "more than one top-level transform op";
        diag.attachNote(topLevelTransform.getLoc())
            << "previous top-level transform op";
        return WalkResult::interrupt();
      });
  if (walkResult.wasInterrupted())
    return nullptr;
  if (!topLevelTransform) {
    auto diag = root->emitError()
                << "could not find a nested top-level transform op";
    diag.attachNote() << "use the '" << filenameOption
                      << "' option to provide transform as external file";
    return nullptr;
  }
  return topLevelTransform;
}

/// Finds an operation nested in `root` that has the transform dialect tag
/// attribute with the value specified as `tag`. Assumes only one operation
/// may have the tag. Returns nullptr if there is no such operation.
static Operation *findOpWithTag(Operation *root, StringRef tagKey,
                                StringRef tagValue) {
  Operation *found = nullptr;
  WalkResult walkResult = root->walk<WalkOrder::PreOrder>(
      [tagKey, tagValue, &found, root](Operation *op) {
        auto attr = op->getAttrOfType<StringAttr>(tagKey);
        if (!attr || attr.getValue() != tagValue)
          return WalkResult::advance();

        if (found) {
          InFlightDiagnostic diag = root->emitError()
                                    << "more than one operation with " << tagKey
                                    << "=\"" << tagValue << "\" attribute";
          diag.attachNote(found->getLoc()) << "first operation";
          diag.attachNote(op->getLoc()) << "other operation";
          return WalkResult::interrupt();
        }

        found = op;
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted())
    return nullptr;

  if (!found) {
    root->emitError() << "could not find the operation with " << tagKey << "=\""
                      << tagValue << "\" attribute";
  }
  return found;
}

/// Returns the ancestor of `target` that doesn't have a parent.
static Operation *getRootOperation(Operation *target) {
  Operation *root = target;
  while (root->getParentOp())
    root = root->getParentOp();
  return root;
}

/// Prints the CLI command running the repro with the current path.
// TODO: make binary name optional by querying LLVM command line API for the
// name of the current binary.
static llvm::raw_ostream &
printReproCall(llvm::raw_ostream &os, StringRef rootOpName, StringRef passName,
               const Pass::Option<std::string> &debugPayloadRootTag,
               const Pass::Option<std::string> &debugTransformRootTag,
               const Pass::Option<std::string> &transformLibraryFileName,
               StringRef binaryName) {
  std::string transformLibraryOption = "";
  if (!transformLibraryFileName.empty()) {
    transformLibraryOption =
        llvm::formatv(" {0}={1}", transformLibraryFileName.getArgStr(),
                      transformLibraryFileName.getValue())
            .str();
  }
  os << llvm::formatv(
      "{7} --pass-pipeline=\"{0}({1}{{{2}={3} {4}={5}{6}})\"", rootOpName,
      passName, debugPayloadRootTag.getArgStr(),
      debugPayloadRootTag.empty()
          ? StringRef(kTransformDialectTagPayloadRootValue)
          : debugPayloadRootTag,
      debugTransformRootTag.getArgStr(),
      debugTransformRootTag.empty()
          ? StringRef(kTransformDialectTagTransformContainerValue)
          : debugTransformRootTag,
      transformLibraryOption, binaryName);
  return os;
}

/// Prints the module rooted at `root` to `os` and appends
/// `transformContainer` if it is not nested in `root`.
llvm::raw_ostream &printModuleForRepro(llvm::raw_ostream &os, Operation *root,
                                       Operation *transform) {
  root->print(os);
  if (!root->isAncestor(transform))
    transform->print(os);
  return os;
}

/// Saves the payload and the transform IR into a temporary file and reports
/// the file name to `os`.
void saveReproToTempFile(
    llvm::raw_ostream &os, Operation *target, Operation *transform,
    StringRef passName, const Pass::Option<std::string> &debugPayloadRootTag,
    const Pass::Option<std::string> &debugTransformRootTag,
    const Pass::Option<std::string> &transformLibraryFileName,
    StringRef binaryName) {
  using llvm::sys::fs::TempFile;
  Operation *root = getRootOperation(target);

  SmallVector<char, 128> tmpPath;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, tmpPath);
  llvm::sys::path::append(tmpPath, "transform_dialect_%%%%%%.mlir");
  llvm::Expected<TempFile> tempFile = TempFile::create(tmpPath);
  if (!tempFile) {
    os << "could not open temporary file to save the repro\n";
    return;
  }

  llvm::raw_fd_ostream fout(tempFile->FD, /*shouldClose=*/false);
  printModuleForRepro(fout, root, transform);
  fout.flush();
  std::string filename = tempFile->TmpName;

  if (tempFile->keep()) {
    os << "could not preserve the temporary file with the repro\n";
    return;
  }

  os << "=== Transform Interpreter Repro ===\n";
  printReproCall(os, root->getName().getStringRef(), passName,
                 debugPayloadRootTag, debugTransformRootTag,
                 transformLibraryFileName, binaryName)
      << " " << filename << "\n";
  os << "===================================\n";
}

// Optionally perform debug actions requested by the user to dump IR and a
// repro to stderr and/or a file.
static void performOptionalDebugActions(
    Operation *target, Operation *transform, StringRef passName,
    const Pass::Option<std::string> &debugPayloadRootTag,
    const Pass::Option<std::string> &debugTransformRootTag,
    const Pass::Option<std::string> &transformLibraryFileName,
    StringRef binaryName) {
  MLIRContext *context = target->getContext();

  // If we are not planning to print, bail early.
  bool hasDebugFlags = false;
  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_STDERR, { hasDebugFlags = true; });
  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_FILE, { hasDebugFlags = true; });
  if (!hasDebugFlags)
    return;

  // We will be mutating the IR to set attributes. If this is running
  // concurrently on several parts of a container or using a shared transform
  // script, this would create a race. Bail in multithreaded mode and require
  // the user to disable threading to dump repros.
  static llvm::sys::SmartMutex<true> dbgStreamMutex;
  if (target->getContext()->isMultithreadingEnabled()) {
    llvm::sys::SmartScopedLock<true> lock(dbgStreamMutex);
    llvm::dbgs() << "=======================================================\n";
    llvm::dbgs() << "|      Transform reproducers cannot be produced       |\n";
    llvm::dbgs() << "|              in multi-threaded mode!                |\n";
    llvm::dbgs() << "=======================================================\n";
    return;
  }

  Operation *root = getRootOperation(target);

  // Add temporary debug / repro attributes, these must never leak out.
  if (debugPayloadRootTag.empty()) {
    target->setAttr(
        kTransformDialectTagAttrName,
        StringAttr::get(context, kTransformDialectTagPayloadRootValue));
  }
  if (debugTransformRootTag.empty()) {
    transform->setAttr(
        kTransformDialectTagAttrName,
        StringAttr::get(context, kTransformDialectTagTransformContainerValue));
  }

  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_STDERR, {
    llvm::dbgs() << "=== Transform Interpreter Repro ===\n";
    printReproCall(llvm::dbgs() << "cat <<EOF | ",
                   root->getName().getStringRef(), passName,
                   debugPayloadRootTag, debugTransformRootTag,
                   transformLibraryFileName, binaryName)
        << "\n";
    printModuleForRepro(llvm::dbgs(), root, transform);
    llvm::dbgs() << "\nEOF\n";
    llvm::dbgs() << "===================================\n";
  });
  (void)root;
  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_FILE, {
    saveReproToTempFile(llvm::dbgs(), target, transform, passName,
                        debugPayloadRootTag, debugTransformRootTag,
                        transformLibraryFileName, binaryName);
  });

  // Remove temporary attributes if they were set.
  if (debugPayloadRootTag.empty())
    target->removeAttr(kTransformDialectTagAttrName);
  if (debugTransformRootTag.empty())
    transform->removeAttr(kTransformDialectTagAttrName);
}

/// Rename `op` to avoid a collision with `otherOp`. `symbolTable` and
/// `otherSymbolTable` are the symbol tables of the two ops, respectively.
/// `uniqueId` is used to generate a unique name in the context of the caller.
LogicalResult renameToUnique(SymbolOpInterface op, SymbolOpInterface otherOp,
                             SymbolTable &symbolTable,
                             SymbolTable &otherSymbolTable, int &uniqueId) {
  assert(symbolTable.lookup(op.getNameAttr()) == op &&
         "symbol table does not contain op");
  assert(otherSymbolTable.lookup(otherOp.getNameAttr()) == otherOp &&
         "other symbol table does not contain other op");

  // Determine new name that is unique in both symbol tables.
  StringAttr oldName = op.getNameAttr();
  StringAttr newName;
  {
    MLIRContext *context = op->getContext();
    SmallString<64> prefix = oldName.getValue();
    prefix.push_back('_');
    while (true) {
      newName = StringAttr::get(context, prefix + Twine(uniqueId++));
      if (!symbolTable.lookup(newName) && !otherSymbolTable.lookup(newName)) {
        break;
      }
    }
  }

  // Apply renaming.
  LLVM_DEBUG(llvm::dbgs() << ", renaming to @" << newName.getValue() << "\n");
  Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(op);
  if (failed(SymbolTable::replaceAllSymbolUses(op, newName, symbolTableOp))) {
    InFlightDiagnostic diag =
        emitError(op->getLoc(),
                  Twine("failed to rename symbol to @") + newName.getValue());
    diag.attachNote(otherOp->getLoc())
        << "attempted renaming due to collision with this op";
    return diag;
  }

  // Change the symbol in the op itself and update the symbol table.
  symbolTable.remove(op);
  SymbolTable::setSymbolName(op, newName);
  symbolTable.insert(op);

  assert(symbolTable.lookup(newName) == op &&
         "symbol table does not resolve to renamed op");
  assert(symbolTable.lookup(oldName) == nullptr &&
         "symbol table still resolves old name");

  return success();
}

/// Return whether `func1` can be merged into `func2`.
bool canMergeInto(FunctionOpInterface func1, FunctionOpInterface func2) {
  return func1.isExternal() && (func2.isPublic() || func2.isExternal());
}

/// Merge `func1` into `func2`. The two ops must be inside the same parent op
/// and mergable according to `canMergeInto`. The function erases `func1` such
/// that only `func2` exists when the function returns.
LogicalResult mergeInto(FunctionOpInterface func1, FunctionOpInterface func2) {
  assert(canMergeInto(func1, func2));
  assert(func1->getParentOp() == func2->getParentOp() &&
         "expected func1 and func2 to be in the same parent op");

  MLIRContext *context = func1->getContext();
  auto consumedName = StringAttr::get(
      context, transform::TransformDialect::kArgConsumedAttrName);
  auto readOnlyName = StringAttr::get(
      context, transform::TransformDialect::kArgReadOnlyAttrName);

  // Check that function signatures match.
  if (func1.getFunctionType() != func2.getFunctionType()) {
    return func1.emitError()
           << "external definition has a mismatching signature ("
           << func2.getFunctionType() << ")";
  }

  // Check and merge argument attributes.
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

/// Merge all symbols from `other` into `target`. Both ops need to implement the
/// `SymbolTable` trait. Operations are moved from `other`, i.e., `other` may be
/// modified by this function and might not verify after the function returns.
/// Upon merging, private symbols may be renamed in order to avoid collisions in
/// the result. Public symbols may not collide, with the exception of
/// instances of `SymbolOpInterface`, where collisions are allowed if at least
/// one of the two is external, in which case the other op preserved (or any one
/// of the two if both are external).
static LogicalResult mergeSymbolsInto(Operation *target,
                                      OwningOpRef<Operation *> other) {
  assert(target->hasTrait<OpTrait::SymbolTable>() &&
         "requires target to implement the 'SymbolTable' trait");
  assert(other->hasTrait<OpTrait::SymbolTable>() &&
         "requires target to implement the 'SymbolTable' trait");

  SymbolTable targetSymbolTable(target);
  SymbolTable otherSymbolTable(*other);

  int uniqueId = 0;

  // Step 1:
  //
  // Rename private symbols in both ops in order to resolve conflicts that can
  // be resolved that way.
  LLVM_DEBUG(DBGS() << "renaming private symbols to resolve conflicts:\n");
  for (auto [symbolTable, otherSymbolTable] : llvm::zip(
           SmallVector<SymbolTable *>{&targetSymbolTable, &otherSymbolTable},
           SmallVector<SymbolTable *>{&otherSymbolTable, &targetSymbolTable})) {
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

      // Collision can be resolved if one of the ops is private.
      if (symbolOp.isPrivate()) {
        if (failed(renameToUnique(symbolOp, collidingOp, *symbolTable,
                                  *otherSymbolTable, uniqueId)))
          return failure();
        continue;
      }
      if (collidingOp.isPrivate()) {
        if (failed(renameToUnique(collidingOp, symbolOp, *otherSymbolTable,
                                  *symbolTable, uniqueId)))
          return failure();
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << ", emitting error\n");
      InFlightDiagnostic diag =
          emitError(symbolOp->getLoc(),
                    Twine("doubly defined symbol @") + name.getValue());
      diag.attachNote(collidingOp->getLoc()) << "previously defined here";
      return diag;
    }
  }

  for (auto *op : SmallVector<Operation *>{target, *other}) {
    if (failed(mlir::verify(op)))
      return emitError(op->getLoc(),
                       "failed to verify input op after renaming");
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
    return emitError(target->getLoc(),
                     "failed to verify target op after merging symbols");

  LLVM_DEBUG(DBGS() << "done merging ops\n");
  return success();
}

} // namespace

LogicalResult transform::detail::interpreterBaseRunOnOperationImpl(
    Operation *target, StringRef passName,
    const std::shared_ptr<OwningOpRef<ModuleOp>> &sharedTransformModule,
    const std::shared_ptr<OwningOpRef<ModuleOp>> &transformLibraryModule,
    const RaggedArray<MappedValue> &extraMappings,
    const TransformOptions &options,
    const Pass::Option<std::string> &transformFileName,
    const Pass::Option<std::string> &transformLibraryFileName,
    const Pass::Option<std::string> &debugPayloadRootTag,
    const Pass::Option<std::string> &debugTransformRootTag,
    StringRef binaryName) {
  bool hasSharedTransformModule =
      sharedTransformModule && *sharedTransformModule;
  bool hasTransformLibraryModule =
      transformLibraryModule && *transformLibraryModule;
  assert((!hasSharedTransformModule || !hasTransformLibraryModule) &&
         "at most one of shared or library transform module can be set");

  // Step 1
  // ------
  // If debugPayloadRootTag was passed, then we are in user-specified selection
  // of the transformed IR. This corresponds to REPL debug mode. Otherwise, just
  // apply to `target`.
  Operation *payloadRoot = target;
  if (!debugPayloadRootTag.empty()) {
    payloadRoot = findOpWithTag(target, kTransformDialectTagAttrName,
                                debugPayloadRootTag);
    if (!payloadRoot)
      return failure();
  }

  // Step 2
  // ------
  // If a shared transform was specified separately, use it. Otherwise, the
  // transform is embedded in the payload IR. If debugTransformRootTag was
  // passed, then we are in user-specified selection of the transforming IR.
  // This corresponds to REPL debug mode.
  Operation *transformContainer =
      hasSharedTransformModule ? sharedTransformModule->get() : target;
  Operation *transformRoot =
      debugTransformRootTag.empty()
          ? findTopLevelTransform(transformContainer,
                                  transformFileName.getArgStr())
          : findOpWithTag(transformContainer, kTransformDialectTagAttrName,
                          debugTransformRootTag);
  if (!transformRoot)
    return failure();

  if (!transformRoot->hasTrait<PossibleTopLevelTransformOpTrait>()) {
    return emitError(transformRoot->getLoc())
           << "expected the transform entry point to be a top-level transform "
              "op";
  }

  // Step 3
  // ------
  // Copy external defintions for symbols if provided. Be aware of potential
  // concurrent execution (normally, the error shouldn't be triggered unless the
  // transform IR modifies itself in a pass, which is also forbidden elsewhere).
  if (hasTransformLibraryModule) {
    if (!target->isProperAncestor(transformRoot)) {
      InFlightDiagnostic diag =
          transformRoot->emitError()
          << "cannot inject transform definitions next to pass anchor op";
      diag.attachNote(target->getLoc()) << "pass anchor op";
      return diag;
    }
    if (failed(
            mergeSymbolsInto(SymbolTable::getNearestSymbolTable(transformRoot),
                             transformLibraryModule->get()->clone())))
      return failure();
  }

  // Step 4
  // ------
  // Optionally perform debug actions requested by the user to dump IR and a
  // repro to stderr and/or a file.
  performOptionalDebugActions(target, transformRoot, passName,
                              debugPayloadRootTag, debugTransformRootTag,
                              transformLibraryFileName, binaryName);

  // Step 5
  // ------
  // Apply the transform to the IR
  return applyTransforms(payloadRoot, cast<TransformOpInterface>(transformRoot),
                         extraMappings, options);
}

LogicalResult transform::detail::interpreterBaseInitializeImpl(
    MLIRContext *context, StringRef transformFileName,
    StringRef transformLibraryFileName,
    std::shared_ptr<OwningOpRef<ModuleOp>> &sharedTransformModule,
    std::shared_ptr<OwningOpRef<ModuleOp>> &transformLibraryModule,
    function_ref<std::optional<LogicalResult>(OpBuilder &, Location)>
        moduleBuilder) {
  OwningOpRef<ModuleOp> parsedTransformModule;
  if (failed(parseTransformModuleFromFile(context, transformFileName,
                                          parsedTransformModule)))
    return failure();
  if (parsedTransformModule && failed(mlir::verify(*parsedTransformModule)))
    return failure();

  OwningOpRef<ModuleOp> parsedLibraryModule;
  if (failed(parseTransformModuleFromFile(context, transformLibraryFileName,
                                          parsedLibraryModule)))
    return failure();
  if (parsedLibraryModule && failed(mlir::verify(*parsedLibraryModule)))
    return failure();

  if (parsedTransformModule) {
    sharedTransformModule = std::make_shared<OwningOpRef<ModuleOp>>(
        std::move(parsedTransformModule));
  } else if (moduleBuilder) {
    // TODO: better location story.
    auto location = UnknownLoc::get(context);
    auto localModule = std::make_shared<OwningOpRef<ModuleOp>>(
        ModuleOp::create(location, "__transform"));

    OpBuilder b(context);
    b.setInsertionPointToEnd(localModule->get().getBody());
    if (std::optional<LogicalResult> result = moduleBuilder(b, location)) {
      if (failed(*result))
        return failure();
      sharedTransformModule = std::move(localModule);
    }
  }

  if (!parsedLibraryModule || !*parsedLibraryModule)
    return success();

  if (sharedTransformModule && *sharedTransformModule) {
    if (failed(mergeSymbolsInto(sharedTransformModule->get(),
                                std::move(parsedLibraryModule))))
      return failure();
  } else {
    transformLibraryModule =
        std::make_shared<OwningOpRef<ModuleOp>>(std::move(parsedLibraryModule));
  }
  return success();
}
