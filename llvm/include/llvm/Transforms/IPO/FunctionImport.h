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

/// Helper to load on demand a Module from file and cache it for subsequent
/// queries. It can be used with the FunctionImporter.
class ModuleLazyLoaderCache {
  /// The context that will be used for importing.
  LLVMContext &Context;

  /// Cache of lazily loaded module for import.
  StringMap<std::unique_ptr<Module>> ModuleMap;

public:
  /// Create the loader, Module will be initialized in \p Context.
  ModuleLazyLoaderCache(LLVMContext &Context) : Context(Context) {}

  /// Retrieve a Module from the cache or lazily load it on demand.
  Module &operator()(StringRef FileName);
};

/// The function importer is automatically importing function from other modules
/// based on the provided summary informations.
class FunctionImporter {

  /// The summaries index used to trigger importing.
  const FunctionInfoIndex &Index;

  /// Diagnostic will be sent to this handler.
  DiagnosticHandlerFunction DiagnosticHandler;

  /// Retrieve a Module from the cache or lazily load it on demand.
  std::function<Module &(StringRef FileName)> getLazyModule;

public:
  /// Create a Function Importer.
  FunctionImporter(const FunctionInfoIndex &Index,
                   DiagnosticHandlerFunction DiagnosticHandler,
                   std::function<Module &(StringRef FileName)> ModuleLoader)
      : Index(Index), DiagnosticHandler(DiagnosticHandler),
        getLazyModule(ModuleLoader) {}

  /// Import functions in Module \p M based on the summary informations.
  bool importFunctions(Module &M);
};
}

#endif // LLVM_FUNCTIONIMPORT_H
