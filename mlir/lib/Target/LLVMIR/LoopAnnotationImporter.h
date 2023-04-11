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

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
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
  explicit LoopAnnotationImporter(MLIRContext *context) : context(context) {}
  LoopAnnotationAttr translateLoopAnnotation(const llvm::MDNode *node,
                                             Location loc);

  /// Converts all LLVM access groups starting from `node` to MLIR access group
  /// attributes. Uses `distinctSequence` to generate the function specific
  /// access group identifiers. It stores a mapping from every nested access
  /// group node to the translated access group attribute. Returns success if
  /// all conversions succeed and failure otherwise.
  LogicalResult translateAccessGroup(const llvm::MDNode *node, Location loc,
                                     DistinctSequenceAttr distinctSequence);

  /// Returns the access group attributes that map to the access group nodes
  /// starting from the access group metadata `node`. Returns failure if the
  /// lookup fails for any of the access groups.
  FailureOr<SmallVector<AccessGroupAttr>>
  lookupAccessGroupsAttr(const llvm::MDNode *node) const;

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

  MLIRContext *context;
  DenseMap<const llvm::MDNode *, LoopAnnotationAttr> loopMetadataMapping;
  /// Mapping between original LLVM access group metadata nodes and the
  /// corresponding MLIR access group attribute.
  DenseMap<const llvm::MDNode *, AccessGroupAttr> accessGroupMapping;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_LOOPANNOTATIONIMPORTER_H_
