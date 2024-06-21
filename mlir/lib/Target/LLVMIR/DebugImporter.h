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

  /// A mapping between LLVM debug metadata and the corresponding attribute.
  DenseMap<llvm::DINode *, DINodeAttr> nodeToAttr;
  /// A mapping between distinct LLVM debug metadata nodes and the corresponding
  /// distinct id attribute.
  DenseMap<llvm::DINode *, DistinctAttr> nodeToDistinctAttr;

  /// Translation helper for recursive DINodes.
  /// Works alongside a stack-based DINode translator (the "main translator")
  /// for gracefully handling DINodes that are recursive.
  ///
  /// Usage:
  /// - Before translating a node, call `pruneOrPushTranslationStack` to see if
  ///   the pruner can preempt this translation. If this is a node that the
  ///   pruner already knows how to handle, it will return the translated
  ///   DINodeAttr.
  /// - After a node is successfully translated by the main translator, call
  ///   `finalizeTranslation` to save the translated result with the pruner, and
  ///   give it a chance to further modify the result.
  /// - Regardless of success or failure by the main translator, always call
  ///   `popTranslationStack` at the end of translating a node. This is
  ///   necessary to keep the internal book-keeping in sync.
  ///
  /// This helper maintains an internal cache so that no recursive type will
  /// be translated more than once by the main translator.
  /// This internal cache is different from the cache maintained by the main
  /// translator because it may store nodes that are not self-contained (i.e.
  /// contain unbounded recursive self-references).
  class RecursionPruner {
  public:
    RecursionPruner(MLIRContext *context) : context(context) {}

    /// If this node is a recursive instance that was previously seen, returns a
    /// self-reference. If this node was previously cached, returns the cached
    /// result. Otherwise, returns null attr, and a translation stack frame is
    /// created for this node. Expects `finalizeTranslation` &
    /// `popTranslationStack` to be called on this node later.
    DINodeAttr pruneOrPushTranslationStack(llvm::DINode *node);

    /// Register the translated result of `node`. Returns the finalized result
    /// (with recId if recursive) and whether the result is self-contained
    /// (i.e. contains no unbound self-refs).
    std::pair<DINodeAttr, bool> finalizeTranslation(llvm::DINode *node,
                                                    DINodeAttr result);

    /// Pop off a frame from the translation stack after a node is done being
    /// translated.
    void popTranslationStack(llvm::DINode *node);

  private:
    /// Returns the cached result (if exists) or null.
    /// The cache entry will be removed if not all of its dependent self-refs
    /// exists.
    DINodeAttr lookup(llvm::DINode *node);

    MLIRContext *context;

    /// A cached translation that contains the translated attribute as well
    /// as any unbound self-references that it depends on.
    struct DependentTranslation {
      /// The translated attr. May contain unbound self-references for other
      /// recursive attrs.
      DINodeAttr attr;
      /// The set of unbound self-refs that this cached entry refers to. All
      /// these self-refs must exist for the cached entry to be valid.
      DenseSet<DIRecursiveTypeAttrInterface> unboundSelfRefs;
    };
    /// A mapping between LLVM debug metadata and the corresponding attribute.
    /// Only contains those with unboundSelfRefs. Fully self-contained attrs
    /// will be cached by the outer main translator.
    DenseMap<llvm::DINode *, DependentTranslation> dependentCache;

    /// Each potentially recursive node will have a TranslationState pushed onto
    /// the `translationStack` to keep track of whether this node is actually
    /// recursive (i.e. has self-references inside), and other book-keeping.
    struct TranslationState {
      /// The rec-self if this node is indeed a recursive node (i.e. another
      /// instance of itself is seen while translating it). Null if this node
      /// has not been seen again deeper in the translation stack.
      DIRecursiveTypeAttrInterface recSelf;
      /// All the unbound recursive self references in this layer of the
      /// translation stack.
      DenseSet<DIRecursiveTypeAttrInterface> unboundSelfRefs;
    };
    /// A stack that stores the metadata nodes that are being traversed. The
    /// stack is used to handle cyclic dependencies during metadata translation.
    /// Each node is pushed with an empty TranslationState. If it is ever seen
    /// later when the stack is deeper, the node is recursive, and its
    /// TranslationState is assigned a recSelf.
    llvm::MapVector<llvm::DINode *, TranslationState> translationStack;

    /// A mapping between DINodes that are recursive, and their assigned recId.
    /// This is kept so that repeated occurrences of the same node can reuse the
    /// same ID and be deduplicated.
    DenseMap<llvm::DINode *, DistinctAttr> nodeToRecId;
  };
  RecursionPruner recursionPruner;

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
