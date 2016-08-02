//===-- llvm/FunctionInfo.h - Function Info Index ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// FunctionInfo.h This file contains the declarations the classes that hold
///  the function info index and summary.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_FUNCTIONINFO_H
#define LLVM_IR_FUNCTIONINFO_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// \brief Function summary information to aid decisions and implementation of
/// importing.
///
/// This is a separate class from FunctionInfo to enable lazy reading of this
/// function summary information from the combined index file during imporing.
class FunctionSummary {
private:
  /// \brief Path of module containing function IR, used to locate module when
  /// importing this function.
  ///
  /// This is only used during parsing of the combined function index, or when
  /// parsing the per-module index for creation of the combined function index,
  /// not during writing of the per-module index which doesn't contain a
  /// module path string table.
  StringRef ModulePath;

  /// \brief Used to flag functions that have local linkage types and need to
  /// have module identifier appended before placing into the combined
  /// index, to disambiguate from other functions with the same name.
  ///
  /// This is only used in the per-module function index, as it is consumed
  /// while creating the combined index.
  bool IsLocalFunction;

  // The rest of the information is used to help decide whether importing
  // is likely to be profitable.
  // Other information will be added as the importing is tuned, such
  // as hotness (when profile available), and other function characteristics.

  /// Number of instructions (ignoring debug instructions, e.g.) computed
  /// during the initial compile step when the function index is first built.
  unsigned InstCount;

public:
  /// Construct a summary object from summary data expected for all
  /// summary records.
  FunctionSummary(unsigned NumInsts) : InstCount(NumInsts) {}

  /// Set the path to the module containing this function, for use in
  /// the combined index.
  void setModulePath(StringRef ModPath) { ModulePath = ModPath; }

  /// Get the path to the module containing this function.
  StringRef modulePath() const { return ModulePath; }

  /// Record whether this is a local function in the per-module index.
  void setLocalFunction(bool IsLocal) { IsLocalFunction = IsLocal; }

  /// Check whether this was a local function, for use in creating
  /// the combined index.
  bool isLocalFunction() const { return IsLocalFunction; }

  /// Get the instruction count recorded for this function.
  unsigned instCount() const { return InstCount; }
};

/// \brief Class to hold pointer to function summary and information required
/// for parsing it.
///
/// For the per-module index, this holds the bitcode offset
/// of the corresponding function block. For the combined index,
/// after parsing of the \a ValueSymbolTable, this initially
/// holds the offset of the corresponding function summary bitcode
/// record. After parsing the associated summary information from the summary
/// block the \a FunctionSummary is populated and stored here.
class FunctionInfo {
private:
  /// Function summary information used to help make ThinLTO importing
  /// decisions.
  std::unique_ptr<FunctionSummary> Summary;

  /// \brief The bitcode offset corresponding to either the associated
  /// function's function body record, or its function summary record,
  /// depending on whether this is a per-module or combined index.
  ///
  /// This bitcode offset is written to or read from the associated
  /// \a ValueSymbolTable entry for the function.
  /// For the per-module index this holds the bitcode offset of the
  /// function's body record  within bitcode module block in its module,
  /// which is used during lazy function parsing or ThinLTO importing.
  /// For the combined index this holds the offset of the corresponding
  /// function summary record, to enable associating the combined index
  /// VST records with the summary records.
  uint64_t BitcodeIndex;

public:
  /// Constructor used during parsing of VST entries.
  FunctionInfo(uint64_t FuncOffset)
      : Summary(nullptr), BitcodeIndex(FuncOffset) {}

  /// Constructor used for per-module index bitcode writing.
  FunctionInfo(uint64_t FuncOffset,
               std::unique_ptr<FunctionSummary> FuncSummary)
      : Summary(std::move(FuncSummary)), BitcodeIndex(FuncOffset) {}

  /// Record the function summary information parsed out of the function
  /// summary block during parsing or combined index creation.
  void setFunctionSummary(std::unique_ptr<FunctionSummary> FuncSummary) {
    Summary = std::move(FuncSummary);
  }

  /// Get the function summary recorded for this function.
  FunctionSummary *functionSummary() const { return Summary.get(); }

  /// Get the bitcode index recorded for this function, depending on
  /// the index type.
  uint64_t bitcodeIndex() const { return BitcodeIndex; }

  /// Record the bitcode index for this function, depending on
  /// the index type.
  void setBitcodeIndex(uint64_t FuncOffset) { BitcodeIndex = FuncOffset; }
};

/// List of function info structures for a particular function name held
/// in the FunctionMap. Requires a vector in the case of multiple
/// COMDAT functions of the same name.
typedef std::vector<std::unique_ptr<FunctionInfo>> FunctionInfoList;

/// Map from function name to corresponding function info structures.
typedef StringMap<FunctionInfoList> FunctionInfoMapTy;

/// Type used for iterating through the function info map.
typedef FunctionInfoMapTy::const_iterator const_funcinfo_iterator;
typedef FunctionInfoMapTy::iterator funcinfo_iterator;

/// String table to hold/own module path strings, which additionally holds the
/// module ID assigned to each module during the plugin step. The StringMap
/// makes a copy of and owns inserted strings.
typedef StringMap<uint64_t> ModulePathStringTableTy;

/// Class to hold module path string table and function map,
/// and encapsulate methods for operating on them.
class FunctionInfoIndex {
private:
  /// Map from function name to list of function information instances
  /// for functions of that name (may be duplicates in the COMDAT case, e.g.).
  FunctionInfoMapTy FunctionMap;

  /// Holds strings for combined index, mapping to the corresponding module ID.
  ModulePathStringTableTy ModulePathStringTable;

public:
  FunctionInfoIndex() = default;

  // Disable the copy constructor and assignment operators, so
  // no unexpected copying/moving occurs.
  FunctionInfoIndex(const FunctionInfoIndex &) = delete;
  void operator=(const FunctionInfoIndex &) = delete;

  funcinfo_iterator begin() { return FunctionMap.begin(); }
  const_funcinfo_iterator begin() const { return FunctionMap.begin(); }
  funcinfo_iterator end() { return FunctionMap.end(); }
  const_funcinfo_iterator end() const { return FunctionMap.end(); }

  /// Get the list of function info objects for a given function.
  const FunctionInfoList &getFunctionInfoList(StringRef FuncName) {
    return FunctionMap[FuncName];
  }

  /// Get the list of function info objects for a given function.
  const const_funcinfo_iterator findFunctionInfoList(StringRef FuncName) const {
    return FunctionMap.find(FuncName);
  }

  /// Add a function info for a function of the given name.
  void addFunctionInfo(StringRef FuncName, std::unique_ptr<FunctionInfo> Info) {
    FunctionMap[FuncName].push_back(std::move(Info));
  }

  /// Iterator to allow writer to walk through table during emission.
  iterator_range<StringMap<uint64_t>::const_iterator>
  modPathStringEntries() const {
    return llvm::make_range(ModulePathStringTable.begin(),
                            ModulePathStringTable.end());
  }

  /// Get the module ID recorded for the given module path.
  uint64_t getModuleId(const StringRef ModPath) const {
    return ModulePathStringTable.lookup(ModPath);
  }

  /// Add the given per-module index into this function index/summary,
  /// assigning it the given module ID. Each module merged in should have
  /// a unique ID, necessary for consistent renaming of promoted
  /// static (local) variables.
  void mergeFrom(std::unique_ptr<FunctionInfoIndex> Other,
                 uint64_t NextModuleId);

  /// Convenience method for creating a promoted global name
  /// for the given value name of a local, and its original module's ID.
  static std::string getGlobalNameForLocal(StringRef Name, uint64_t ModId) {
    SmallString<256> NewName(Name);
    NewName += ".llvm.";
    raw_svector_ostream(NewName) << ModId;
    return NewName.str();
  }

  /// Add a new module path, mapped to the given module Id, and return StringRef
  /// owned by string table map.
  StringRef addModulePath(StringRef ModPath, uint64_t ModId) {
    return ModulePathStringTable.insert(std::make_pair(ModPath, ModId))
        .first->first();
  }

  /// Check if the given Module has any functions available for exporting
  /// in the index. We consider any module present in the ModulePathStringTable
  /// to have exported functions.
  bool hasExportedFunctions(const Module &M) const {
    return ModulePathStringTable.count(M.getModuleIdentifier());
  }
};

} // End llvm namespace

#endif
