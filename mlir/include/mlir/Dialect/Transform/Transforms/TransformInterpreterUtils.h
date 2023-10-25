//===- TransformInterpreterUtils.h - Transform Utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERUTILS_H
#define MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERUTILS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include <memory>

namespace mlir {
struct LogicalResult;
class MLIRContext;
class ModuleOp;
class Operation;
template <typename>
class OwningOpRef;
class Region;

namespace transform {
namespace detail {

/// Expands the given list of `paths` to a list of `.mlir` files.
///
/// Each entry in `paths` may either be a regular file, in which case it ends up
/// in the result list, or a directory, in which case all (regular) `.mlir`
/// files in that directory are added. Any other file types lead to a failure.
LogicalResult expandPathsToMLIRFiles(ArrayRef<std::string> &paths,
                                     MLIRContext *context,
                                     SmallVectorImpl<std::string> &fileNames);

/// Utility to parse and verify the content of a `transformFileName` MLIR file
/// containing a transform dialect specification.
LogicalResult
parseTransformModuleFromFile(MLIRContext *context,
                             llvm::StringRef transformFileName,
                             OwningOpRef<ModuleOp> &transformModule);

/// Utility to parse, verify, aggregate and link the content of all mlir files
/// nested under `transformLibraryPaths` and containing transform dialect
/// specifications.
LogicalResult
assembleTransformLibraryFromPaths(MLIRContext *context,
                                  ArrayRef<std::string> transformLibraryPaths,
                                  OwningOpRef<ModuleOp> &transformModule);

/// Utility to load a transform interpreter `module` from a module that has
/// already been preloaded in the context.
/// This mode is useful in cases where explicit parsing of a transform library
/// from file is expected to be prohibitively expensive.
/// In such cases, the transform module is expected to be found in the preloaded
/// library modules of the transform dialect.
/// Returns null if the module is not found.
ModuleOp getPreloadedTransformModule(MLIRContext *context);

/// Finds the first TransformOpInterface named `kTransformEntryPointSymbolName`
/// that is either:
///   1. nested under `root` (takes precedence).
///   2. nested under `module`, if not found in `root`.
/// Reports errors and returns null if no such operation found.
TransformOpInterface findTransformEntryPoint(
    Operation *root, ModuleOp module,
    StringRef entryPoint = TransformDialect::kTransformEntryPointSymbolName);

/// Merge all symbols from `other` into `target`. Both ops need to implement the
/// `SymbolTable` trait. Operations are moved from `other`, i.e., `other` may be
/// modified by this function and might not verify after the function returns.
/// Upon merging, private symbols may be renamed in order to avoid collisions in
/// the result. Public symbols may not collide, with the exception of
/// instances of `SymbolOpInterface`, where collisions are allowed if at least
/// one of the two is external, in which case the other op preserved (or any one
/// of the two if both are external).
// TODO: Reconsider cloning individual ops rather than forcing users of the
//       function to clone (or move) `other` in order to improve efficiency.
//       This might primarily make sense if we can also prune the symbols that
//       are merged to a subset (such as those that are actually used).
InFlightDiagnostic mergeSymbolsInto(Operation *target,
                                    OwningOpRef<Operation *> other);
} // namespace detail

/// Standalone util to apply the named sequence `transformRoot` to `payload` IR.
/// This is done in 2 steps:
///   1. If `transformModule` is provided and is not nested under
///      `transformRoot`, it will be "linked into" the IR containing
///      `transformRoot` to resolve undefined named sequences.
///   2. The transforms specified in `transformRoot` are applied to `payload`,
///      assuming the named sequence has a single argument handle that will be
///      associated with `payload` on run.
LogicalResult applyTransformNamedSequence(Operation *payload,
                                          Operation *transformRoot,
                                          ModuleOp transformModule,
                                          const TransformOptions &options);

} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERUTILS_H
