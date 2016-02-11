//===- FunctionImportUtils.h - Importing support utilities -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionImportGlobalProcessing class which is used
// to perform the necessary global value handling for function importing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_FUNCTIONIMPORTUTILS_H
#define LLVM_TRANSFORMS_UTILS_FUNCTIONIMPORTUTILS_H

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/FunctionInfo.h"

namespace llvm {
class Module;

/// Class to handle necessary GlobalValue changes required by ThinLTO
/// function importing, including linkage changes and any necessary renaming.
class FunctionImportGlobalProcessing {
  /// The Module which we are exporting or importing functions from.
  Module &M;

  /// Function index passed in for function importing/exporting handling.
  const FunctionInfoIndex *ImportIndex;

  /// Functions to import from this module, all other functions will be
  /// imported as declarations instead of definitions.
  DenseSet<const GlobalValue *> *FunctionsToImport;

  /// Set to true if the given FunctionInfoIndex contains any functions
  /// from this source module, in which case we must conservatively assume
  /// that any of its functions may be imported into another module
  /// as part of a different backend compilation process.
  bool HasExportedFunctions = false;

  /// Populated during ThinLTO global processing with locals promoted
  /// to global scope in an exporting module, which now need to be linked
  /// in if calling from the ModuleLinker.
  SetVector<GlobalValue *> NewExportedValues;

  /// Check if we should promote the given local value to global scope.
  bool doPromoteLocalToGlobal(const GlobalValue *SGV);

  /// Helper methods to check if we are importing from or potentially
  /// exporting from the current source module.
  bool isPerformingImport() const { return FunctionsToImport != nullptr; }
  bool isModuleExporting() const { return HasExportedFunctions; }

  /// If we are importing from the source module, checks if we should
  /// import SGV as a definition, otherwise import as a declaration.
  bool doImportAsDefinition(const GlobalValue *SGV);

  /// Get the name for SGV that should be used in the linked destination
  /// module. Specifically, this handles the case where we need to rename
  /// a local that is being promoted to global scope.
  std::string getName(const GlobalValue *SGV);

  /// Process globals so that they can be used in ThinLTO. This includes
  /// promoting local variables so that they can be reference externally by
  /// thin lto imported globals and converting strong external globals to
  /// available_externally.
  void processGlobalsForThinLTO();
  void processGlobalForThinLTO(GlobalValue &GV);

  /// Get the new linkage for SGV that should be used in the linked destination
  /// module. Specifically, for ThinLTO importing or exporting it may need
  /// to be adjusted.
  GlobalValue::LinkageTypes getLinkage(const GlobalValue *SGV);

public:
  FunctionImportGlobalProcessing(
      Module &M, const FunctionInfoIndex *Index,
      DenseSet<const GlobalValue *> *FunctionsToImport = nullptr)
      : M(M), ImportIndex(Index), FunctionsToImport(FunctionsToImport) {
    // If we have a FunctionInfoIndex but no function to import,
    // then this is the primary module being compiled in a ThinLTO
    // backend compilation, and we need to see if it has functions that
    // may be exported to another backend compilation.
    if (!FunctionsToImport)
      HasExportedFunctions = ImportIndex->hasExportedFunctions(M);
  }

  bool run();

  static bool
  doImportAsDefinition(const GlobalValue *SGV,
                       DenseSet<const GlobalValue *> *FunctionsToImport);

  /// Access the promoted globals that are now exported and need to be linked.
  SetVector<GlobalValue *> &getNewExportedValues() { return NewExportedValues; }
};

/// Perform in-place global value handling on the given Module for
/// exported local functions renamed and promoted for ThinLTO.
bool renameModuleForThinLTO(Module &M, const FunctionInfoIndex *Index);

} // End llvm namespace

#endif
