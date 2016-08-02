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

#include "llvm/ADT/StringMap.h"
#include <functional>

namespace llvm {
class LLVMContext;
class Module;
class FunctionInfoIndex;

/// The function importer is automatically importing function from other modules
/// based on the provided summary informations.
class FunctionImporter {

  /// The summaries index used to trigger importing.
  const FunctionInfoIndex &Index;

  /// Factory function to load a Module for a given identifier
  std::function<std::unique_ptr<Module>(StringRef Identifier)> ModuleLoader;

public:
  /// Create a Function Importer.
  FunctionImporter(
      const FunctionInfoIndex &Index,
      std::function<std::unique_ptr<Module>(StringRef Identifier)> ModuleLoader)
      : Index(Index), ModuleLoader(ModuleLoader) {}

  /// Import functions in Module \p M based on the summary informations.
  bool importFunctions(Module &M);
};
}

#endif // LLVM_FUNCTIONIMPORT_H
