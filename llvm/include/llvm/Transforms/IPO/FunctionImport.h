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
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <map>
#include <unordered_set>
#include <utility>

namespace llvm {
class LLVMContext;
class GlobalValueSummary;
class Module;

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

  /// A function of this type is used to load modules referenced by the index.
  typedef std::function<Expected<std::unique_ptr<Module>>(StringRef Identifier)>
      ModuleLoaderTy;

  /// Create a Function Importer.
  FunctionImporter(const ModuleSummaryIndex &Index, ModuleLoaderTy ModuleLoader)
      : Index(Index), ModuleLoader(std::move(ModuleLoader)) {}

  /// Import functions in Module \p M based on the supplied import list.
  Expected<bool> importFunctions(
      Module &M, const ImportMapTy &ImportList,
      Optional<std::function<void(Module &)>> PostBitcodeLoading = None);

private:
  /// The summaries index used to trigger importing.
  const ModuleSummaryIndex &Index;

  /// Factory function to load a Module for a given identifier
  ModuleLoaderTy ModuleLoader;
};

/// The function importing pass
class FunctionImportPass : public PassInfoMixin<FunctionImportPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// Compute all the imports and exports for every module in the Index.
///
/// \p ModuleToDefinedGVSummaries contains for each Module a map
/// (GUID -> Summary) for every global defined in the module.
///
/// \p ImportLists will be populated with an entry for every Module we are
/// importing into. This entry is itself a map that can be passed to
/// FunctionImporter::importFunctions() above (see description there).
///
/// \p ExportLists contains for each Module the set of globals (GUID) that will
/// be imported by another module, or referenced by such a function. I.e. this
/// is the set of globals that need to be promoted/renamed appropriately.
///
/// \p DeadSymbols (optional) contains a list of GUID that are deemed "dead" and
/// will be ignored for the purpose of importing.
void ComputeCrossModuleImport(
    const ModuleSummaryIndex &Index,
    const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    StringMap<FunctionImporter::ImportMapTy> &ImportLists,
    StringMap<FunctionImporter::ExportSetTy> &ExportLists,
    const DenseSet<GlobalValue::GUID> *DeadSymbols = nullptr);

/// Compute all the imports for the given module using the Index.
///
/// \p ImportList will be populated with a map that can be passed to
/// FunctionImporter::importFunctions() above (see description there).
void ComputeCrossModuleImportForModule(
    StringRef ModulePath, const ModuleSummaryIndex &Index,
    FunctionImporter::ImportMapTy &ImportList);

/// Compute all the symbols that are "dead": i.e these that can't be reached
/// in the graph from any of the given symbols listed in
/// \p GUIDPreservedSymbols.
DenseSet<GlobalValue::GUID>
computeDeadSymbols(const ModuleSummaryIndex &Index,
                   const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols);

/// Compute the set of summaries needed for a ThinLTO backend compilation of
/// \p ModulePath.
//
/// This includes summaries from that module (in case any global summary based
/// optimizations were recorded) and from any definitions in other modules that
/// should be imported.
//
/// \p ModuleToSummariesForIndex will be populated with the needed summaries
/// from each required module path. Use a std::map instead of StringMap to get
/// stable order for bitcode emission.
void gatherImportedSummariesForModule(
    StringRef ModulePath,
    const StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    const FunctionImporter::ImportMapTy &ImportList,
    std::map<std::string, GVSummaryMapTy> &ModuleToSummariesForIndex);

/// Emit into \p OutputFilename the files module \p ModulePath will import from.
std::error_code
EmitImportsFiles(StringRef ModulePath, StringRef OutputFilename,
                 const FunctionImporter::ImportMapTy &ModuleImports);

/// Resolve WeakForLinker values in \p TheModule based on the information
/// recorded in the summaries during global summary-based analysis.
void thinLTOResolveWeakForLinkerModule(Module &TheModule,
                                       const GVSummaryMapTy &DefinedGlobals);

/// Internalize \p TheModule based on the information recorded in the summaries
/// during global summary-based analysis.
void thinLTOInternalizeModule(Module &TheModule,
                              const GVSummaryMapTy &DefinedGlobals);
}

#endif // LLVM_FUNCTIONIMPORT_H
