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
  LoopAnnotationImporter(ModuleImport &moduleImport, OpBuilder &builder)
      : moduleImport(moduleImport), builder(builder) {}
  LoopAnnotationAttr translateLoopAnnotation(const llvm::MDNode *node,
                                             Location loc);

  /// Converts all LLVM access groups starting from node to MLIR access group
  /// attributes. It stores a mapping from every nested access group node to the
  /// translated attribute. Returns success if all conversions succeed and
  /// failure otherwise.
  LogicalResult translateAccessGroup(const llvm::MDNode *node, Location loc);

  /// Returns the access group attribute that map to the access group nodes
  /// starting from the access group metadata node. Returns failure, if any of
  /// the attributes cannot be found.
  FailureOr<SmallVector<AccessGroupAttr>>
  lookupAccessGroupAttrs(const llvm::MDNode *node) const;

  /// The ModuleImport owning this instance.
  ModuleImport &moduleImport;

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
  /// Mapping between original LLVM access group metadata nodes and the imported
  /// MLIR access group attributes.
  DenseMap<const llvm::MDNode *, AccessGroupAttr> accessGroupMapping;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONIMPORTER_H_
