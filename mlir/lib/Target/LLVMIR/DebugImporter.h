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
#include "llvm/IR/DebugInfoMetadata.h"

namespace mlir {
class Operation;

namespace LLVM {
class LLVMFuncOp;

namespace detail {

class DebugImporter {
public:
  DebugImporter(ModuleOp mlirModule)
      : context(mlirModule.getContext()), mlirModule(mlirModule) {}

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
    // Infer the result MLIR type from the LLVM metadata type.
    // If the result is a DIType, it can also be wrapped in a recursive type,
    // so the result is wrapped into a DIRecursiveTypeAttrOf.
    // Otherwise, the exact result type is used.
    constexpr bool isDIType = std::is_base_of_v<llvm::DIType, DINodeT>;
    using RawMLIRTypeT = decltype(translateImpl(node));
    using MLIRTypeT =
        std::conditional_t<isDIType, DIRecursiveTypeAttrOf<RawMLIRTypeT>,
                           RawMLIRTypeT>;
    return cast_or_null<MLIRTypeT>(
        translate(static_cast<llvm::DINode *>(node)));
  }

private:
  /// Translates the given LLVM debug metadata to the corresponding attribute.
  DIBasicTypeAttr translateImpl(llvm::DIBasicType *node);
  DICompileUnitAttr translateImpl(llvm::DICompileUnit *node);
  DICompositeTypeAttr translateImpl(llvm::DICompositeType *node);
  DIDerivedTypeAttr translateImpl(llvm::DIDerivedType *node);
  DIFileAttr translateImpl(llvm::DIFile *node);
  DILabelAttr translateImpl(llvm::DILabel *node);
  DILexicalBlockAttr translateImpl(llvm::DILexicalBlock *node);
  DILexicalBlockFileAttr translateImpl(llvm::DILexicalBlockFile *node);
  DIGlobalVariableAttr translateImpl(llvm::DIGlobalVariable *node);
  DILocalVariableAttr translateImpl(llvm::DILocalVariable *node);
  DIModuleAttr translateImpl(llvm::DIModule *node);
  DINamespaceAttr translateImpl(llvm::DINamespace *node);
  DIScopeAttr translateImpl(llvm::DIScope *node);
  DISubprogramAttr translateImpl(llvm::DISubprogram *node);
  DISubrangeAttr translateImpl(llvm::DISubrange *node);
  DISubroutineTypeAttr translateImpl(llvm::DISubroutineType *node);
  DITypeAttr translateImpl(llvm::DIType *node);

  /// Constructs a StringAttr from the MDString if it is non-null. Returns a
  /// null attribute otherwise.
  StringAttr getStringAttrOrNull(llvm::MDString *stringNode);

  DistinctAttr getOrCreateDistinctID(llvm::DINode *node);

  /// A mapping between LLVM debug metadata and the corresponding attribute.
  DenseMap<llvm::DINode *, DINodeAttr> nodeToAttr;
  /// A mapping between LLVM debug metadata and the distinct ID attr for DI
  /// nodes that require distinction.
  DenseMap<llvm::DINode *, DistinctAttr> nodeToDistinctAttr;

  /// A stack that stores the metadata type nodes that are being traversed. The
  /// stack is used to detect cyclic dependencies during the metadata
  /// translation. Nodes are pushed with a null value. If it is ever seen twice,
  /// it is given a DistinctAttr, indicating that it is a recursive node and
  /// should take on that DistinctAttr as ID.
  llvm::MapVector<llvm::DIType *, DistinctAttr> typeTranslationStack;
  /// All the unbound recursive self references in the translation stack.
  SmallVector<DenseSet<DistinctAttr>> unboundRecursiveSelfRefs;
  /// A stack that stores the non-type metadata nodes that are being traversed.
  /// Each node is associated with the size of the `typeTranslationStack` at the
  /// time of push. This is used to identify a recursion purely in the non-type
  /// metadata nodes, which is not supported yet.
  SetVector<std::pair<llvm::DINode *, unsigned>> nonTypeTranslationStack;

  MLIRContext *context;
  ModuleOp mlirModule;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_DEBUGIMPORTER_H_
