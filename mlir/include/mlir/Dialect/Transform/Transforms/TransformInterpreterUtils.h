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
/// Finds the first TransformOpInterface named `kTransformEntryPointSymbolName`
/// that is either:
///   1. nested under `root` (takes precedence).
///   2. nested under `module`, if not found in `root`.
/// Reports errors and returns null if no such operation found.
TransformOpInterface findTransformEntryPoint(
    Operation *root, ModuleOp module,
    StringRef entryPoint = TransformDialect::kTransformEntryPointSymbolName);
} // namespace detail

/// Standalone util to apply the named sequence `entryPoint` to the payload.
/// This is done in 3 steps:
///   1. lookup the `entryPoint` symbol in `{payload, sharedTransformModule}` by
///   calling detail::findTransformEntryPoint.
///   2. if the entry point is found and not nested under
///   `sharedTransformModule`, call `detail::defineDeclaredSymbols` to "link" in
///   the `sharedTransformModule`. Note: this may modify the transform IR
///   embedded with the payload IR.
///   3. apply the transform IR to the payload IR, relaxing the requirement that
///   the transform IR is a top-level transform op. We are applying a named
///   sequence anyway.
LogicalResult applyTransformNamedSequence(
    Operation *payload, ModuleOp transformModule,
    const TransformOptions &options,
    StringRef entryPoint = TransformDialect::kTransformEntryPointSymbolName);

} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERUTILS_H
