//===- llvm/Transforms/IPO/FunctionImport.h - ThinLTO importing -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_FUNCTIONIMPORT_H
#define LLVM_TRANSFORMS_IPO_FUNCTIONIMPORT_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <memory>
#include <system_error>
#include <utility>

namespace llvm {

class Module;

/// The function importer is automatically importing function from other modules
/// based on the provided summary informations.
class FunctionImporter {
public:
  /// The different reasons selectCallee will chose not to import a
  /// candidate.
  enum class ImportFailureReason {
    None,
    // We can encounter a global variable instead of a function in rare
    // situations with SamplePGO. See comments where this failure type is
    // set for more details.
    GlobalVar,
    // Found to be globally dead, so we don't bother importing.
    NotLive,
    // Instruction count over the current threshold.
    TooLarge,
    // Don't import something with interposable linkage as we can't inline it
    // anyway.
    InterposableLinkage,
    // Generally we won't end up failing due to this reason, as we expect
    // to find at least one summary for the GUID that is global or a local
    // in the referenced module for direct calls.
    LocalLinkageNotInModule,
    // This corresponds to the NotEligibleToImport being set on the summary,
    // which can happen in a few different cases (e.g. local that can't be
    // renamed or promoted because it is referenced on a llvm*.used variable).
    NotEligible,
    // This corresponds to NoInline being set on the function summary,
    // which will happen if it is known that the inliner will not be able
    // to inline the function (e.g. it is marked with a NoInline attribute).
    NoInline
  };

  /// Information optionally tracked for candidates the importer decided
  /// not to import. Used for optional stat printing.
  struct ImportFailureInfo {
    // The ValueInfo corresponding to the candidate. We save an index hash
    // table lookup for each GUID by stashing this here.
    ValueInfo VI;
    // The maximum call edge hotness for all failed imports of this candidate.
    CalleeInfo::HotnessType MaxHotness;
    // most recent reason for failing to import (doesn't necessarily correspond
    // to the attempt with the maximum hotness).
    ImportFailureReason Reason;
    // The number of times we tried to import candidate but failed.
    unsigned Attempts;
    ImportFailureInfo(ValueInfo VI, CalleeInfo::HotnessType MaxHotness,
                      ImportFailureReason Reason, unsigned Attempts)
        : VI(VI), MaxHotness(MaxHotness), Reason(Reason), Attempts(Attempts) {}
  };

  /// Map of callee GUID considered for import into a given module to a pair
  /// consisting of the largest threshold applied when deciding whether to
  /// import it and, if we decided to import, a pointer to the summary instance
  /// imported. If we decided not to import, the summary will be nullptr.
  using ImportThresholdsTy =
      DenseMap<GlobalValue::GUID,
               std::tuple<unsigned, const GlobalValueSummary *,
                          std::unique_ptr<ImportFailureInfo>>>;

  // Issues import IDs.  Each ID uniquely corresponds to a tuple of
  // (FromModule, GUID, Definition/Declaration).
  //
  // The import IDs make the import list space efficient by referring to each
  // import with a 32-bit integer ID while maintaining a central table that maps
  // those integer IDs to tuples of (FromModule, GUID, Def/Decl).
  //
  // In one large application, a pair of (FromModule, GUID) is mentioned in
  // import lists more than 50 times on average across all destination modules.
  // Mentioning the 32-byte tuple:
  //
  // std::tuple<StringRef, GlobalValue::GUID, GlobalValueSummary::ImportKind>
  //
  // 50 times by value in various import lists would be costly.  We can reduce
  // the memory footprint of import lists by placing one copy in a central table
  // and referring to it with 32-bit integer IDs.
  //
  // To save space within the central table, we only store pairs of
  // (FromModule, GUID) in the central table.  In the actual 32-bit integer ID,
  // the top 31 bits index into the central table while the bottom 1 bit
  // indicates whether an ID is for GlobalValueSummary::Declaration or
  // GlobalValueSummary::Definition.
  class ImportIDTable {
  public:
    using ImportIDTy = uint32_t;

    ImportIDTable() = default;

    // Something is wrong with the application logic if we need to make a copy
    // of this and potentially make a fork.
    ImportIDTable(const ImportIDTable &) = delete;
    ImportIDTable &operator=(const ImportIDTable &) = delete;

    // Create a pair of import IDs [Def, Decl] for a given pair of FromModule
    // and GUID.
    std::pair<ImportIDTy, ImportIDTy> createImportIDs(StringRef FromModule,
                                                      GlobalValue::GUID GUID) {
      auto Key = std::make_pair(FromModule, GUID);
      auto InsertResult = TheTable.try_emplace(Key, TheTable.size());
      return makeIDPair(InsertResult.first->second);
    }

    // Get a pair of previously created import IDs [Def, Decl] for a given pair
    // of FromModule and GUID.  Returns std::nullopt if not available.
    std::optional<std::pair<ImportIDTy, ImportIDTy>>
    getImportIDs(StringRef FromModule, GlobalValue::GUID GUID) {
      auto Key = std::make_pair(FromModule, GUID);
      auto It = TheTable.find(Key);
      if (It != TheTable.end())
        return makeIDPair(It->second);
      return std::nullopt;
    }

    // Return a tuple of [FromModule, GUID, Def/Decl] that a given ImportID
    // corresponds to.
    std::tuple<StringRef, GlobalValue::GUID, GlobalValueSummary::ImportKind>
    lookup(ImportIDTy ImportID) const {
      GlobalValueSummary::ImportKind Kind =
          (ImportID & 1) ? GlobalValueSummary::Declaration
                         : GlobalValueSummary::Definition;
      auto It = TheTable.begin() + (ImportID >> 1);
      StringRef FromModule = It->first.first;
      GlobalValue::GUID GUID = It->first.second;
      return std::make_tuple(FromModule, GUID, Kind);
    }

    // The same as lookup above.  Useful for map_iterator.
    std::tuple<StringRef, GlobalValue::GUID, GlobalValueSummary::ImportKind>
    operator()(ImportIDTable::ImportIDTy ImportID) const {
      return lookup(ImportID);
    }

  private:
    // Make a pair of import IDs [Def, Decl] from an index into TheTable.
    static std::pair<ImportIDTy, ImportIDTy> makeIDPair(ImportIDTy Index) {
      ImportIDTy Def = Index << 1;
      ImportIDTy Decl = Def | 1;
      return std::make_pair(Def, Decl);
    }

    MapVector<std::pair<StringRef, GlobalValue::GUID>, ImportIDTy> TheTable;
  };

  // Forward-declare SortedImportList for ImportMapTy.
  class SortedImportList;

  /// The map maintains the list of imports.  Conceptually, it is a collection
  /// of tuples of the form:
  ///
  ///   (The name of the source module, GUID, Definition/Declaration)
  ///
  /// The name of the source module is the module identifier to pass to the
  /// ModuleLoader.  The module identifier strings must be owned elsewhere,
  /// typically by the in-memory ModuleSummaryIndex the importing decisions are
  /// made from (the module path for each summary is owned by the index's module
  /// path string table).
  class ImportMapTy {
  public:
    enum class AddDefinitionStatus {
      // No change was made to the list of imports or whether each import should
      // be imported as a declaration or definition.
      NoChange,
      // Successfully added the given GUID to be imported as a definition. There
      // was no existing entry with the same GUID as a declaration.
      Inserted,
      // An existing with the given GUID was changed to a definition.
      ChangedToDefinition,
    };

    ImportMapTy() = delete;
    ImportMapTy(ImportIDTable &IDs) : IDs(IDs) {}

    // Add the given GUID to ImportList as a definition.  If the same GUID has
    // been added as a declaration previously, that entry is overridden.
    AddDefinitionStatus addDefinition(StringRef FromModule,
                                      GlobalValue::GUID GUID);

    // Add the given GUID to ImportList as a declaration.  If the same GUID has
    // been added as a definition previously, that entry takes precedence, and
    // no change is made.
    void maybeAddDeclaration(StringRef FromModule, GlobalValue::GUID GUID);

    void addGUID(StringRef FromModule, GlobalValue::GUID GUID,
                 GlobalValueSummary::ImportKind ImportKind) {
      if (ImportKind == GlobalValueSummary::Definition)
        addDefinition(FromModule, GUID);
      else
        maybeAddDeclaration(FromModule, GUID);
    }

    // Return the list of source modules sorted in the ascending alphabetical
    // order.
    SmallVector<StringRef, 0> getSourceModules() const;

    std::optional<GlobalValueSummary::ImportKind>
    getImportType(StringRef FromModule, GlobalValue::GUID GUID) const;

    // Iterate over the import list.  The caller gets tuples of FromModule,
    // GUID, and ImportKind instead of import IDs.  std::cref below prevents
    // map_iterator from deep-copying IDs.
    auto begin() const { return map_iterator(Imports.begin(), std::cref(IDs)); }
    auto end() const { return map_iterator(Imports.end(), std::cref(IDs)); }

    friend class SortedImportList;

  private:
    ImportIDTable &IDs;
    DenseSet<ImportIDTable::ImportIDTy> Imports;
  };

  // A read-only copy of ImportMapTy with its contents sorted according to the
  // given comparison function.
  class SortedImportList {
  public:
    SortedImportList(const ImportMapTy &ImportMap,
                     llvm::function_ref<
                         bool(const std::pair<StringRef, GlobalValue::GUID> &,
                              const std::pair<StringRef, GlobalValue::GUID> &)>
                         Comp)
        : IDs(ImportMap.IDs), Imports(iterator_range(ImportMap.Imports)) {
      llvm::sort(Imports, [&](ImportIDTable::ImportIDTy L,
                              ImportIDTable::ImportIDTy R) {
        auto Lookup = [&](ImportIDTable::ImportIDTy Id)
            -> std::pair<StringRef, GlobalValue::GUID> {
          auto Tuple = IDs.lookup(Id);
          return std::make_pair(std::get<0>(Tuple), std::get<1>(Tuple));
        };
        return Comp(Lookup(L), Lookup(R));
      });
    }

    // Iterate over the import list.  The caller gets tuples of FromModule,
    // GUID, and ImportKind instead of import IDs.  std::cref below prevents
    // map_iterator from deep-copying IDs.
    auto begin() const { return map_iterator(Imports.begin(), std::cref(IDs)); }
    auto end() const { return map_iterator(Imports.end(), std::cref(IDs)); }

  private:
    const ImportIDTable &IDs;
    SmallVector<ImportIDTable::ImportIDTy, 0> Imports;
  };

  // A map from destination modules to lists of imports.
  class ImportListsTy {
  public:
    ImportListsTy() : EmptyList(ImportIDs) {}
    ImportListsTy(size_t Size) : EmptyList(ImportIDs), ListsImpl(Size) {}

    ImportMapTy &operator[](StringRef DestMod) {
      return ListsImpl.try_emplace(DestMod, ImportIDs).first->second;
    }

    const ImportMapTy &lookup(StringRef DestMod) const {
      auto It = ListsImpl.find(DestMod);
      if (It != ListsImpl.end())
        return It->second;
      return EmptyList;
    }

    size_t size() const { return ListsImpl.size(); }

    using const_iterator = DenseMap<StringRef, ImportMapTy>::const_iterator;
    const_iterator begin() const { return ListsImpl.begin(); }
    const_iterator end() const { return ListsImpl.end(); }

  private:
    ImportMapTy EmptyList;
    DenseMap<StringRef, ImportMapTy> ListsImpl;
    ImportIDTable ImportIDs;
  };

  /// The set contains an entry for every global value that the module exports.
  /// Depending on the user context, this container is allowed to contain
  /// definitions, declarations or a mix of both.
  using ExportSetTy = DenseSet<ValueInfo>;

  /// A function of this type is used to load modules referenced by the index.
  using ModuleLoaderTy =
      std::function<Expected<std::unique_ptr<Module>>(StringRef Identifier)>;

  /// Create a Function Importer.
  FunctionImporter(const ModuleSummaryIndex &Index, ModuleLoaderTy ModuleLoader,
                   bool ClearDSOLocalOnDeclarations)
      : Index(Index), ModuleLoader(std::move(ModuleLoader)),
        ClearDSOLocalOnDeclarations(ClearDSOLocalOnDeclarations) {}

  /// Import functions in Module \p M based on the supplied import list.
  Expected<bool> importFunctions(Module &M, const ImportMapTy &ImportList);

private:
  /// The summaries index used to trigger importing.
  const ModuleSummaryIndex &Index;

  /// Factory function to load a Module for a given identifier
  ModuleLoaderTy ModuleLoader;

  /// See the comment of ClearDSOLocalOnDeclarations in
  /// Utils/FunctionImportUtils.h.
  bool ClearDSOLocalOnDeclarations;
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
/// \p isPrevailing is a callback that will be called with a global value's GUID
/// and summary and should return whether the module corresponding to the
/// summary contains the linker-prevailing copy of that value.
///
/// \p ImportLists will be populated with an entry for every Module we are
/// importing into. This entry is itself a map that can be passed to
/// FunctionImporter::importFunctions() above (see description there).
///
/// \p ExportLists contains for each Module the set of globals (GUID) that will
/// be imported by another module, or referenced by such a function. I.e. this
/// is the set of globals that need to be promoted/renamed appropriately.
///
/// The module identifier strings that are the keys of the above two maps
/// are owned by the in-memory ModuleSummaryIndex the importing decisions
/// are made from (the module path for each summary is owned by the index's
/// module path string table).
void ComputeCrossModuleImport(
    const ModuleSummaryIndex &Index,
    const DenseMap<StringRef, GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        isPrevailing,
    FunctionImporter::ImportListsTy &ImportLists,
    DenseMap<StringRef, FunctionImporter::ExportSetTy> &ExportLists);

/// PrevailingType enum used as a return type of callback passed
/// to computeDeadSymbolsAndUpdateIndirectCalls. Yes and No values used when
/// status explicitly set by symbols resolution, otherwise status is Unknown.
enum class PrevailingType { Yes, No, Unknown };

/// Update call edges for indirect calls to local functions added from
/// SamplePGO when needed. Normally this is done during
/// computeDeadSymbolsAndUpdateIndirectCalls, but can be called standalone
/// when that is not called (e.g. during testing).
void updateIndirectCalls(ModuleSummaryIndex &Index);

/// Compute all the symbols that are "dead": i.e these that can't be reached
/// in the graph from any of the given symbols listed in
/// \p GUIDPreservedSymbols. Non-prevailing symbols are symbols without a
/// prevailing copy anywhere in IR and are normally dead, \p isPrevailing
/// predicate returns status of symbol.
/// Also update call edges for indirect calls to local functions added from
/// SamplePGO when needed.
void computeDeadSymbolsAndUpdateIndirectCalls(
    ModuleSummaryIndex &Index,
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols,
    function_ref<PrevailingType(GlobalValue::GUID)> isPrevailing);

/// Compute dead symbols and run constant propagation in combined index
/// after that.
void computeDeadSymbolsWithConstProp(
    ModuleSummaryIndex &Index,
    const DenseSet<GlobalValue::GUID> &GUIDPreservedSymbols,
    function_ref<PrevailingType(GlobalValue::GUID)> isPrevailing,
    bool ImportEnabled);

/// Converts value \p GV to declaration, or replaces with a declaration if
/// it is an alias. Returns true if converted, false if replaced.
bool convertToDeclaration(GlobalValue &GV);

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
///
/// \p DecSummaries will be popluated with the subset of of summary pointers
/// that have 'declaration' import type among all summaries the module need.
void gatherImportedSummariesForModule(
    StringRef ModulePath,
    const DenseMap<StringRef, GVSummaryMapTy> &ModuleToDefinedGVSummaries,
    const FunctionImporter::ImportMapTy &ImportList,
    ModuleToSummariesForIndexTy &ModuleToSummariesForIndex,
    GVSummaryPtrSet &DecSummaries);

/// Emit into \p OutputFilename the files module \p ModulePath will import from.
Error EmitImportsFiles(
    StringRef ModulePath, StringRef OutputFilename,
    const ModuleToSummariesForIndexTy &ModuleToSummariesForIndex);

/// Call \p F passing each of the files module \p ModulePath will import from.
void processImportsFiles(
    StringRef ModulePath,
    const ModuleToSummariesForIndexTy &ModuleToSummariesForIndex,
    function_ref<void(const std::string &)> F);

/// Based on the information recorded in the summaries during global
/// summary-based analysis:
/// 1. Resolve prevailing symbol linkages and constrain visibility (CanAutoHide
///    and consider visibility from other definitions for ELF) in \p TheModule
/// 2. (optional) Apply propagated function attributes to \p TheModule if
///    PropagateAttrs is true
void thinLTOFinalizeInModule(Module &TheModule,
                             const GVSummaryMapTy &DefinedGlobals,
                             bool PropagateAttrs);

/// Internalize \p TheModule based on the information recorded in the summaries
/// during global summary-based analysis.
void thinLTOInternalizeModule(Module &TheModule,
                              const GVSummaryMapTy &DefinedGlobals);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_FUNCTIONIMPORT_H
