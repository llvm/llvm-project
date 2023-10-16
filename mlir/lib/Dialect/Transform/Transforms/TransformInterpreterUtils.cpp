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
