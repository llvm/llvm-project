//===- DebugImporter.h - LLVM to MLIR Debug conversion -------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between LLVMIR debug information and
// the corresponding MLIR representation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_LLVMIR_DEBUGIMPORTER_H_
#define MLIR_LIB_TARGET_LLVMIR_DEBUGIMPORTER_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/CyclicReplacerCache.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/IR/DebugInfoMetadata.h"

namespace mlir {
class Operation;

namespace LLVM {
class LLVMFuncOp;

namespace detail {

class DebugImporter {
public:
  DebugImporter(ModuleOp mlirModule, bool dropDICompositeTypeElements);

  /// Translates the given LLVM debug location to an MLIR location.
  Location translateLoc(llvm::DILocation *loc);

  /// Translates the LLVM DWARF expression metadata to MLIR.
  DIExpressionAttr translateExpression(llvm::DIExpression *node);

  /// Translates the LLVM DWARF global variable expression metadata to MLIR.
  DIGlobalVariableExpressionAttr
  translateGlobalVariableExpression(llvm::DIGlobalVariableExpression *node);

  /// Translates the debug information for the given function into a Location.
  /// Returns UnknownLoc if `func` has no debug information attached to it.
  Location translateFuncLocation(llvm::Function *func);

  /// Translates the given LLVM debug metadata to MLIR.
  DINodeAttr translate(llvm::DINode *node);

  /// Infers the metadata type and translates it to MLIR.
  template <typename DINodeT>
  auto translate(DINodeT *node) {
    // Infer the MLIR type from the LLVM metadata type.
    using MLIRTypeT = decltype(translateImpl(node));
    return cast_or_null<MLIRTypeT>(
        translate(static_cast<llvm::DINode *>(node)));
  }

private:
  /// Translates the given LLVM debug metadata to the corresponding attribute.
  DIBasicTypeAttr translateImpl(llvm::DIBasicType *node);
  DICompileUnitAttr translateImpl(llvm::DICompileUnit *node);
  DICompositeTypeAttr translateImpl(llvm::DICompositeType *node);
  DIDerivedTypeAttr translateImpl(llvm::DIDerivedType *node);
  DIStringTypeAttr translateImpl(llvm::DIStringType *node);
  DIFileAttr translateImpl(llvm::DIFile *node);
  DILabelAttr translateImpl(llvm::DILabel *node);
  DILexicalBlockAttr translateImpl(llvm::DILexicalBlock *node);
  DILexicalBlockFileAttr translateImpl(llvm::DILexicalBlockFile *node);
  DIGlobalVariableAttr translateImpl(llvm::DIGlobalVariable *node);
  DILocalVariableAttr translateImpl(llvm::DILocalVariable *node);
  DIVariableAttr translateImpl(llvm::DIVariable *node);
  DIModuleAttr translateImpl(llvm::DIModule *node);
  DINamespaceAttr translateImpl(llvm::DINamespace *node);
  DIImportedEntityAttr translateImpl(llvm::DIImportedEntity *node);
  DIScopeAttr translateImpl(llvm::DIScope *node);
  DISubprogramAttr translateImpl(llvm::DISubprogram *node);
  DISubrangeAttr translateImpl(llvm::DISubrange *node);
  DISubroutineTypeAttr translateImpl(llvm::DISubroutineType *node);
  DITypeAttr translateImpl(llvm::DIType *node);

  /// Constructs a StringAttr from the MDString if it is non-null. Returns a
  /// null attribute otherwise.
  StringAttr getStringAttrOrNull(llvm::MDString *stringNode);

  /// Get the DistinctAttr used to represent `node` if one was already created
  /// for it, or create a new one if not.
  DistinctAttr getOrCreateDistinctID(llvm::DINode *node);

  std::optional<DINodeAttr> createRecSelf(llvm::DINode *node);

  /// A mapping between distinct LLVM debug metadata nodes and the corresponding
  /// distinct id attribute.
  DenseMap<llvm::DINode *, DistinctAttr> nodeToDistinctAttr;

  /// A mapping between DINodes that are recursive, and their assigned recId.
  /// This is kept so that repeated occurrences of the same node can reuse the
  /// same ID and be deduplicated.
  DenseMap<llvm::DINode *, DistinctAttr> nodeToRecId;

  CyclicReplacerCache<llvm::DINode *, DINodeAttr> cache;

  MLIRContext *context;
  ModuleOp mlirModule;

  /// An option to control if DICompositeTypes should always be imported without
  /// converting their elements. If set, the option avoids the recursive
  /// traversal of composite type debug information, which can be expensive for
  /// adversarial inputs.
  bool dropDICompositeTypeElements;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_DEBUGIMPORTER_H_
