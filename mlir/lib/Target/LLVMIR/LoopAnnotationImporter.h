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

/// A helper class that converts llvm.loop metadata nodes into corresponding
/// LoopAnnotationAttrs and llvm.access.group nodes into
/// AccessGroupMetadataOps.
class LoopAnnotationImporter {
public:
  explicit LoopAnnotationImporter(OpBuilder &builder) : builder(builder) {}
  LoopAnnotationAttr translateLoopAnnotation(const llvm::MDNode *node,
                                             Location loc);

  /// Converts all LLVM access groups starting from node to MLIR access group
  /// operations mested in the region of metadataOp. It stores a mapping from
  /// every nested access group nod to the symbol pointing to the translated
  /// operation. Returns success if all conversions succeed and failure
  /// otherwise.
  LogicalResult translateAccessGroup(const llvm::MDNode *node, Location loc,
                                     MetadataOp metadataOp);

  /// Returns the symbol references pointing to the access group operations that
  /// map to the access group nodes starting from the access group metadata
  /// node. Returns failure, if any of the symbol references cannot be found.
  FailureOr<SmallVector<SymbolRefAttr>>
  lookupAccessGroupAttrs(const llvm::MDNode *node) const;

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

  OpBuilder &builder;
  DenseMap<const llvm::MDNode *, LoopAnnotationAttr> loopMetadataMapping;
  /// Mapping between original LLVM access group metadata nodes and the symbol
  /// references pointing to the imported MLIR access group operations.
  DenseMap<const llvm::MDNode *, SymbolRefAttr> accessGroupMapping;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONIMPORTER_H_
