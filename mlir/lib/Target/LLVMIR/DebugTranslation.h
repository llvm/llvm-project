//===- DebugTranslation.h - MLIR to LLVM Debug conversion -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between an MLIR debug information and
// the corresponding LLVMIR representation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_LLVMIR_DEBUGTRANSLATION_H_
#define MLIR_LIB_TARGET_LLVMIR_DEBUGTRANSLATION_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/DIBuilder.h"

namespace mlir {
class Operation;

namespace LLVM {
class LLVMFuncOp;

namespace detail {
class DebugTranslation {
public:
  DebugTranslation(Operation *module, llvm::Module &llvmModule);

  /// Finalize the translation of debug information.
  void finalize();

  /// Translate the given location to an llvm debug location.
  const llvm::DILocation *translateLoc(Location loc, llvm::DILocalScope *scope);

  /// Translate the debug information for the given function.
  void translate(LLVMFuncOp func, llvm::Function &llvmFunc);

  /// Translate the given LLVM debug metadata to LLVM.
  llvm::DINode *translate(DINodeAttr attr);

  /// Translate the given derived LLVM debug metadata to LLVM.
  template <typename DIAttrT>
  auto translate(DIAttrT attr) {
    // Infer the LLVM type from the attribute type.
    using LLVMTypeT = std::remove_pointer_t<decltype(translateImpl(attr))>;
    return cast_or_null<LLVMTypeT>(translate(DINodeAttr(attr)));
  }

private:
  /// Translate the given location to an llvm debug location with the given
  /// scope and inlinedAt parameters.
  const llvm::DILocation *translateLoc(Location loc, llvm::DILocalScope *scope,
                                       const llvm::DILocation *inlinedAt);

  /// Create an llvm debug file for the given file path.
  llvm::DIFile *translateFile(StringRef fileName);

  /// Translate the given attribute to the corresponding llvm debug metadata.
  llvm::DIBasicType *translateImpl(DIBasicTypeAttr attr);
  llvm::DICompileUnit *translateImpl(DICompileUnitAttr attr);
  llvm::DICompositeType *translateImpl(DICompositeTypeAttr attr);
  llvm::DIDerivedType *translateImpl(DIDerivedTypeAttr attr);
  llvm::DIFile *translateImpl(DIFileAttr attr);
  llvm::DILexicalBlock *translateImpl(DILexicalBlockAttr attr);
  llvm::DILexicalBlockFile *translateImpl(DILexicalBlockFileAttr attr);
  llvm::DILocalVariable *translateImpl(DILocalVariableAttr attr);
  llvm::DIScope *translateImpl(DIScopeAttr attr);
  llvm::DISubprogram *translateImpl(DISubprogramAttr attr);
  llvm::DISubrange *translateImpl(DISubrangeAttr attr);
  llvm::DISubroutineType *translateImpl(DISubroutineTypeAttr attr);
  llvm::DIType *translateImpl(DITypeAttr attr);

  /// A mapping between mlir location+scope and the corresponding llvm debug
  /// metadata.
  DenseMap<std::tuple<Location, llvm::DILocalScope *, const llvm::DILocation *>,
           const llvm::DILocation *>
      locationToLoc;

  /// A mapping between debug attribute and the corresponding llvm debug
  /// metadata.
  DenseMap<Attribute, llvm::DINode *> attrToNode;

  /// A mapping between filename and llvm debug file.
  /// TODO: Change this to DenseMap<Identifier, ...> when we can
  /// access the Identifier filename in FileLineColLoc.
  llvm::StringMap<llvm::DIFile *> fileMap;

  /// A string containing the current working directory of the compiler.
  SmallString<256> currentWorkingDir;

  /// Flag indicating if debug information should be emitted.
  bool debugEmissionIsEnabled;

  /// Debug information fields.
  llvm::Module &llvmModule;
  llvm::LLVMContext &llvmCtx;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_DEBUGTRANSLATION_H_
