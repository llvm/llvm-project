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
#include "llvm/IR/GlobalValue.h"

#include <functional>
#include <map>
#include <unordered_set>

namespace llvm {
class LLVMContext;
class Module;
class ModuleSummaryIndex;

/// The function importer is automatically importing function from other modules
/// based on the provided summary informations.
class FunctionImporter {
public:
  /// Set of functions to import from a source module. Each entry is a map
  /// containing all the functions to import for a source module.
  /// The keys is the GUID identifying a function to import, and the value
  /// is the threshold applied when deciding to import it.
  typedef std::map<GlobalValue::GUID, unsigned> FunctionsToImportTy;

  /// The map contains an entry for every module to import from, the key being
  /// the module identifier to pass to the ModuleLoader. The value is the set of
  /// functions to import.
  typedef StringMap<FunctionsToImportTy> ImportMapTy;

  /// The set contains an entry for every global value the module exports.
  typedef std::unordered_set<GlobalValue::GUID> ExportSetTy;

  /// Create a Function Importer.
  FunctionImporter(
      const ModuleSummaryIndex &Index,
      std::function<std::unique_ptr<Module>(StringRef Identifier)> ModuleLoader)
      : Index(Index), ModuleLoader(ModuleLoader) {}

  /// Import functions in Module \p M based on the supplied import list.
  bool importFunctions(Module &M, const ImportMapTy &ImportList);

private:
  /// The summaries index used to trigger importing.
  const ModuleSummaryIndex &Index;

  /// Factory function to load a Module for a given identifier
  std::function<std::unique_ptr<Module>(StringRef Identifier)> ModuleLoader;
};

/// Compute all the imports and exports for every module in the Index.
///
/// \p ImportLists will be populated with an entry for every Module we are
/// importing into. This entry is itself a map that can be passed to
/// FunctionImporter::importFunctions() above (see description there).
///
/// \p ExportLists contains for each Module the set of globals (GUID) that will
/// be imported by another module, or referenced by such a function. I.e. this
/// is the set of globals that need to be promoted/renamed appropriately.
void ComputeCrossModuleImport(
    const ModuleSummaryIndex &Index,
    StringMap<FunctionImporter::ImportMapTy> &ImportLists,
    StringMap<FunctionImporter::ExportSetTy> &ExportLists);
}

#endif // LLVM_FUNCTIONIMPORT_H
