//===- LoopAnnotationImporter.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between LLVMIR loop metadata and the
// corresponding MLIR representation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONIMPORTER_H_
#define MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONIMPORTER_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

namespace mlir {
namespace LLVM {
namespace detail {

/// A helper class that converts a `llvm.loop` metadata node into a
/// corresponding LoopAnnotationAttr.
class LoopAnnotationImporter {
public:
  explicit LoopAnnotationImporter(ModuleImport &moduleImport)
      : moduleImport(moduleImport) {}
  LoopAnnotationAttr translate(const llvm::MDNode *node, Location loc);

private:
  /// Returns the LLVM metadata corresponding to a llvm loop metadata attribute.
  LoopAnnotationAttr lookupLoopMetadata(const llvm::MDNode *node) const {
    return loopMetadataMapping.lookup(node);
  }

  void mapLoopMetadata(const llvm::MDNode *metadata, LoopAnnotationAttr attr) {
    auto result = loopMetadataMapping.try_emplace(metadata, attr);
    (void)result;
    assert(result.second &&
           "attempting to map loop options that was already mapped");
  }

  ModuleImport &moduleImport;
  DenseMap<const llvm::MDNode *, LoopAnnotationAttr> loopMetadataMapping;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONIMPORTER_H_
