//===-- llvm/ModuleSummaryIndex.h - Module Summary Index --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// ModuleSummaryIndex.h This file contains the declarations the classes that
///  hold the module index and summary for function importing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_MODULESUMMARYINDEX_H
#define LLVM_IR_MODULESUMMARYINDEX_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <array>

namespace llvm {

/// \brief Class to accumulate and hold information about a callee.
struct CalleeInfo {
  /// The static number of callsites calling corresponding function.
  unsigned CallsiteCount;
  /// The cumulative profile count of calls to corresponding function
  /// (if using PGO, otherwise 0).
  uint64_t ProfileCount;
  CalleeInfo() : CallsiteCount(0), ProfileCount(0) {}
  CalleeInfo(unsigned CallsiteCount, uint64_t ProfileCount)
      : CallsiteCount(CallsiteCount), ProfileCount(ProfileCount) {}
  CalleeInfo &operator+=(uint64_t RHSProfileCount) {
    CallsiteCount++;
    ProfileCount += RHSProfileCount;
    return *this;
  }
};

/// \brief Function and variable summary information to aid decisions and
/// implementation of importing.
///
/// This is a separate class from GlobalValueInfo to enable lazy reading of this
/// summary information from the combined index file during imporing.
class GlobalValueSummary {
public:
  /// \brief Sububclass discriminator (for dyn_cast<> et al.)
  enum SummaryKind { FunctionKind, GlobalVarKind };

private:
  /// Kind of summary for use in dyn_cast<> et al.
  SummaryKind Kind;

  /// \brief Path of module IR containing value's definition, used to locate
  /// module during importing.
  ///
  /// This is only used during parsing of the combined index, or when
  /// parsing the per-module index for creation of the combined summary index,
  /// not during writing of the per-module index which doesn't contain a
  /// module path string table.
  StringRef ModulePath;

  /// \brief The linkage type of the associated global value.
  ///
  /// One use is to flag values that have local linkage types and need to
  /// have module identifier appended before placing into the combined
  /// index, to disambiguate from other values with the same name.
  /// In the future this will be used to update and optimize linkage
  /// types based on global summary-based analysis.
  GlobalValue::LinkageTypes Linkage;

  /// List of GUIDs of values referenced by this global value's definition
  /// (either by the initializer of a global variable, or referenced
  /// from within a function). This does not include functions called, which
  /// are listed in the derived FunctionSummary object.
  std::vector<GlobalValue::GUID> RefEdgeList;

protected:
  /// GlobalValueSummary constructor.
  GlobalValueSummary(SummaryKind K, GlobalValue::LinkageTypes Linkage)
      : Kind(K), Linkage(Linkage) {}

public:
  virtual ~GlobalValueSummary() = default;

  /// Which kind of summary subclass this is.
  SummaryKind getSummaryKind() const { return Kind; }

  /// Set the path to the module containing this function, for use in
  /// the combined index.
  void setModulePath(StringRef ModPath) { ModulePath = ModPath; }

  /// Get the path to the module containing this function.
  StringRef modulePath() const { return ModulePath; }

  /// Return linkage type recorded for this global value.
  GlobalValue::LinkageTypes linkage() const { return Linkage; }

  /// Record a reference from this global value to the global value identified
  /// by \p RefGUID.
  void addRefEdge(GlobalValue::GUID RefGUID) { RefEdgeList.push_back(RefGUID); }

  /// Record a reference from this global value to each global value identified
  /// in \p RefEdges.
  void addRefEdges(DenseSet<unsigned> &RefEdges) {
    for (auto &RI : RefEdges)
      addRefEdge(RI);
  }

  /// Return the list of GUIDs referenced by this global value definition.
  std::vector<GlobalValue::GUID> &refs() { return RefEdgeList; }
  const std::vector<GlobalValue::GUID> &refs() const { return RefEdgeList; }
};

/// \brief Function summary information to aid decisions and implementation of
/// importing.
class FunctionSummary : public GlobalValueSummary {
public:
  /// <CalleeGUID, CalleeInfo> call edge pair.
  typedef std::pair<GlobalValue::GUID, CalleeInfo> EdgeTy;

private:
  /// Number of instructions (ignoring debug instructions, e.g.) computed
  /// during the initial compile step when the summary index is first built.
  unsigned InstCount;

  /// List of <CalleeGUID, CalleeInfo> call edge pairs from this function.
  std::vector<EdgeTy> CallGraphEdgeList;

public:
  /// Summary constructors.
  FunctionSummary(GlobalValue::LinkageTypes Linkage, unsigned NumInsts)
      : GlobalValueSummary(FunctionKind, Linkage), InstCount(NumInsts) {}

  /// Check if this is a function summary.
  static bool classof(const GlobalValueSummary *GVS) {
    return GVS->getSummaryKind() == FunctionKind;
  }

  /// Get the instruction count recorded for this function.
  unsigned instCount() const { return InstCount; }

  /// Record a call graph edge from this function to the function identified
  /// by \p CalleeGUID, with \p CalleeInfo including the cumulative profile
  /// count (across all calls from this function) or 0 if no PGO.
  void addCallGraphEdge(GlobalValue::GUID CalleeGUID, CalleeInfo Info) {
    CallGraphEdgeList.push_back(std::make_pair(CalleeGUID, Info));
  }

  /// Record a call graph edge from this function to each function recorded
  /// in \p CallGraphEdges.
  void addCallGraphEdges(DenseMap<unsigned, CalleeInfo> &CallGraphEdges) {
    for (auto &EI : CallGraphEdges)
      addCallGraphEdge(EI.first, EI.second);
  }

  /// Return the list of <CalleeGUID, ProfileCount> pairs.
  std::vector<EdgeTy> &calls() { return CallGraphEdgeList; }
  const std::vector<EdgeTy> &calls() const { return CallGraphEdgeList; }
};

/// \brief Global variable summary information to aid decisions and
/// implementation of importing.
///
/// Currently this doesn't add anything to the base \p GlobalValueSummary,
/// but is a placeholder as additional info may be added to the summary
/// for variables.
class GlobalVarSummary : public GlobalValueSummary {

public:
  /// Summary constructors.
  GlobalVarSummary(GlobalValue::LinkageTypes Linkage)
      : GlobalValueSummary(GlobalVarKind, Linkage) {}

  /// Check if this is a global variable summary.
  static bool classof(const GlobalValueSummary *GVS) {
    return GVS->getSummaryKind() == GlobalVarKind;
  }
};

/// \brief Class to hold pointer to summary object and information required
/// for parsing or writing it.
class GlobalValueInfo {
private:
  /// Summary information used to help make ThinLTO importing decisions.
  std::unique_ptr<GlobalValueSummary> Summary;

  /// \brief The bitcode offset corresponding to either an associated
  /// function's function body record, or to an associated summary record,
  /// depending on whether this is a per-module or combined index.
  ///
  /// This bitcode offset is written to or read from the associated
  /// \a ValueSymbolTable entry for a function.
  /// For the per-module index this holds the bitcode offset of a
  /// function's body record within bitcode module block in its module,
  /// although this field is currently only used when writing the VST
  /// (it is set to 0 and also unused when this is a global variable).
  /// For the combined index this holds the offset of the corresponding
  /// summary record, to enable associating the combined index
  /// VST records with the summary records.
  uint64_t BitcodeIndex;

public:
  GlobalValueInfo(uint64_t Offset = 0,
                  std::unique_ptr<GlobalValueSummary> Summary = nullptr)
      : Summary(std::move(Summary)), BitcodeIndex(Offset) {}

  /// Record the summary information parsed out of the summary block during
  /// parsing or combined index creation.
  void setSummary(std::unique_ptr<GlobalValueSummary> GVSummary) {
    Summary = std::move(GVSummary);
  }

  /// Get the summary recorded for this global value.
  GlobalValueSummary *summary() const { return Summary.get(); }

  /// Get the bitcode index recorded for this value symbol table entry.
  uint64_t bitcodeIndex() const { return BitcodeIndex; }

  /// Set the bitcode index recorded for this value symbol table entry.
  void setBitcodeIndex(uint64_t Offset) { BitcodeIndex = Offset; }
};

/// 160 bits SHA1
typedef std::array<uint32_t, 5> ModuleHash;

/// List of global value info structures for a particular value held
/// in the GlobalValueMap. Requires a vector in the case of multiple
/// COMDAT values of the same name.
typedef std::vector<std::unique_ptr<GlobalValueInfo>> GlobalValueInfoList;

/// Map from global value GUID to corresponding info structures.
/// Use a std::map rather than a DenseMap since it will likely incur
/// less overhead, as the value type is not very small and the size
/// of the map is unknown, resulting in inefficiencies due to repeated
/// insertions and resizing.
typedef std::map<GlobalValue::GUID, GlobalValueInfoList> GlobalValueInfoMapTy;

/// Type used for iterating through the global value info map.
typedef GlobalValueInfoMapTy::const_iterator const_globalvalueinfo_iterator;
typedef GlobalValueInfoMapTy::iterator globalvalueinfo_iterator;

/// String table to hold/own module path strings, which additionally holds the
/// module ID assigned to each module during the plugin step, as well as a hash
/// of the module. The StringMap makes a copy of and owns inserted strings.
typedef StringMap<std::pair<uint64_t, ModuleHash>> ModulePathStringTableTy;

/// Class to hold module path string table and global value map,
/// and encapsulate methods for operating on them.
class ModuleSummaryIndex {
private:
  /// Map from value name to list of information instances for values of that
  /// name (may be duplicates in the COMDAT case, e.g.).
  GlobalValueInfoMapTy GlobalValueMap;

  /// Holds strings for combined index, mapping to the corresponding module ID.
  ModulePathStringTableTy ModulePathStringTable;

public:
  ModuleSummaryIndex() = default;

  // Disable the copy constructor and assignment operators, so
  // no unexpected copying/moving occurs.
  ModuleSummaryIndex(const ModuleSummaryIndex &) = delete;
  void operator=(const ModuleSummaryIndex &) = delete;

  globalvalueinfo_iterator begin() { return GlobalValueMap.begin(); }
  const_globalvalueinfo_iterator begin() const {
    return GlobalValueMap.begin();
  }
  globalvalueinfo_iterator end() { return GlobalValueMap.end(); }
  const_globalvalueinfo_iterator end() const { return GlobalValueMap.end(); }

  /// Get the list of global value info objects for a given value name.
  const GlobalValueInfoList &getGlobalValueInfoList(StringRef ValueName) {
    return GlobalValueMap[GlobalValue::getGUID(ValueName)];
  }

  /// Get the list of global value info objects for a given value name.
  const const_globalvalueinfo_iterator
  findGlobalValueInfoList(StringRef ValueName) const {
    return GlobalValueMap.find(GlobalValue::getGUID(ValueName));
  }

  /// Get the list of global value info objects for a given value GUID.
  const const_globalvalueinfo_iterator
  findGlobalValueInfoList(GlobalValue::GUID ValueGUID) const {
    return GlobalValueMap.find(ValueGUID);
  }

  /// Add a global value info for a value of the given name.
  void addGlobalValueInfo(StringRef ValueName,
                          std::unique_ptr<GlobalValueInfo> Info) {
    GlobalValueMap[GlobalValue::getGUID(ValueName)].push_back(std::move(Info));
  }

  /// Add a global value info for a value of the given GUID.
  void addGlobalValueInfo(GlobalValue::GUID ValueGUID,
                          std::unique_ptr<GlobalValueInfo> Info) {
    GlobalValueMap[ValueGUID].push_back(std::move(Info));
  }

  /// Table of modules, containing module hash and id.
  const StringMap<std::pair<uint64_t, ModuleHash>> &modulePaths() const {
    return ModulePathStringTable;
  }

  /// Table of modules, containing hash and id.
  StringMap<std::pair<uint64_t, ModuleHash>> &modulePaths() {
    return ModulePathStringTable;
  }

  /// Get the module ID recorded for the given module path.
  uint64_t getModuleId(const StringRef ModPath) const {
    return ModulePathStringTable.lookup(ModPath).first;
  }

  /// Get the module SHA1 hash recorded for the given module path.
  const ModuleHash &getModuleHash(const StringRef ModPath) const {
    auto It = ModulePathStringTable.find(ModPath);
    assert(It != ModulePathStringTable.end() && "Module not registered");
    return It->second.second;
  }

  /// Add the given per-module index into this module index/summary,
  /// assigning it the given module ID. Each module merged in should have
  /// a unique ID, necessary for consistent renaming of promoted
  /// static (local) variables.
  void mergeFrom(std::unique_ptr<ModuleSummaryIndex> Other,
                 uint64_t NextModuleId);

  /// Convenience method for creating a promoted global name
  /// for the given value name of a local, and its original module's ID.
  static std::string getGlobalNameForLocal(StringRef Name, uint64_t ModId) {
    SmallString<256> NewName(Name);
    NewName += ".llvm.";
    raw_svector_ostream(NewName) << ModId;
    return NewName.str();
  }

  /// Add a new module path with the given \p Hash, mapped to the given \p
  /// ModID, and return an iterator to the entry in the index.
  ModulePathStringTableTy::iterator
  addModulePath(StringRef ModPath, uint64_t ModId,
                ModuleHash Hash = ModuleHash{{0}}) {
    return ModulePathStringTable.insert(std::make_pair(
                                            ModPath,
                                            std::make_pair(ModId, Hash))).first;
  }

  /// Check if the given Module has any functions available for exporting
  /// in the index. We consider any module present in the ModulePathStringTable
  /// to have exported functions.
  bool hasExportedFunctions(const Module &M) const {
    return ModulePathStringTable.count(M.getModuleIdentifier());
  }

  /// Remove entries in the GlobalValueMap that have empty summaries due to the
  /// eager nature of map entry creation during VST parsing. These would
  /// also be suppressed during combined index generation in mergeFrom(),
  /// but if there was only one module or this was the first module we might
  /// not invoke mergeFrom.
  void removeEmptySummaryEntries();
};

} // End llvm namespace

#endif
