//===- llvm/Transforms/IPO/FunctionImport.h - ThinLTO importing -*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUNCTIONIMPORT_H
#define LLVM_FUNCTIONIMPORT_H

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
class LLVMContext;
class Module;
class FunctionInfoIndex;

/// The function importer is automatically importing function from other modules
/// based on the provided summary informations.
class FunctionImporter {

  /// Cache of lazily loaded module for import.
  StringMap<std::unique_ptr<Module>> ModuleMap;

  /// The context that will be used for importing.
  LLVMContext &Context;

  /// The summaries index used to trigger importing.
  const FunctionInfoIndex &Index;

  /// Diagnostic will be sent to this handler.
  DiagnosticHandlerFunction DiagnosticHandler;

  /// Retrieve a Module from the cache or lazily load it on demand.
  Module &getOrLoadModule(StringRef FileName);

public:
  /// Create a Function Importer.
  FunctionImporter(LLVMContext &Context, const FunctionInfoIndex &Index,
                   DiagnosticHandlerFunction DiagnosticHandler)
      : Context(Context), Index(Index), DiagnosticHandler(DiagnosticHandler) {}

  /// Import functions in Module \p M based on the summary informations.
  bool importFunctions(Module &M);
};
}

#endif // LLVM_FUNCTIONIMPORT_H
