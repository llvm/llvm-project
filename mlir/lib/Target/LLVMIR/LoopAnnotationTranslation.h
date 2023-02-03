//===- LoopAnnotationTranslation.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between an MLIR loop annotations and
// the corresponding LLVMIR metadata representation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONTRANSLATION_H_
#define MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONTRANSLATION_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

namespace mlir {
namespace LLVM {
namespace detail {

/// A helper class that converts a LoopAnnotationAttr into a corresponding
/// llvm::MDNode.
class LoopAnnotationTranslation {
public:
  LoopAnnotationTranslation(LLVM::ModuleTranslation &moduleTranslation)
      : moduleTranslation(moduleTranslation) {}

  llvm::MDNode *translate(LoopAnnotationAttr attr, Operation *op);

private:
  /// Returns the LLVM metadata corresponding to a llvm loop metadata attribute.
  llvm::MDNode *lookupLoopMetadata(Attribute options) const {
    return loopMetadataMapping.lookup(options);
  }

  void mapLoopMetadata(Attribute options, llvm::MDNode *metadata) {
    auto result = loopMetadataMapping.try_emplace(options, metadata);
    (void)result;
    assert(result.second &&
           "attempting to map loop options that was already mapped");
  }

  /// Mapping from an attribute describing loop metadata to its LLVM metadata.
  /// The metadata is attached to Latch block branches with this attribute.
  DenseMap<Attribute, llvm::MDNode *> loopMetadataMapping;

  LLVM::ModuleTranslation &moduleTranslation;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONTRANSLATION_H_
