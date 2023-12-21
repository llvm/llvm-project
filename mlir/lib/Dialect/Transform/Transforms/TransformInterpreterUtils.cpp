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
#include "mlir/Dialect/Transform/IR/Utils.h"
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
    ArrayRef<std::string> paths, MLIRContext *context,
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

      if (it->type() != llvm::sys::fs::file_type::regular_file &&
          it->type() != llvm::sys::fs::file_type::symlink_file) {
        LLVM_DEBUG(DBGS() << "  Skipping non-regular file '" << fileName
                          << "'\n");
        continue;
      }

      if (!StringRef(fileName).ends_with(".mlir")) {
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
  return context->getOrLoadDialect<transform::TransformDialect>()
      ->getLibraryModule();
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

LogicalResult transform::applyTransformNamedSequence(
    Operation *payload, Operation *transformRoot, ModuleOp transformModule,
    const TransformOptions &options) {
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
