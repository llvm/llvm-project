//===- llvm/ModuleSummaryIndex.h - Module Summary Index ---------*- C++ -*-===//
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

namespace yaml {

template <typename T> struct MappingTraits;

} // end namespace yaml

/// \brief Class to accumulate and hold information about a callee.
struct CalleeInfo {
  enum class HotnessType : uint8_t {
    Unknown = 0,
    Cold = 1,
    None = 2,
    Hot = 3,
    Critical = 4
  };
  HotnessType Hotness = HotnessType::Unknown;

  CalleeInfo() = default;
  explicit CalleeInfo(HotnessType Hotness) : Hotness(Hotness) {}

  void updateHotness(const HotnessType OtherHotness) {
    Hotness = std::max(Hotness, OtherHotness);
  }
};

class GlobalValueSummary;

using GlobalValueSummaryList = std::vector<std::unique_ptr<GlobalValueSummary>>;

struct GlobalValueSummaryInfo {
  /// The GlobalValue corresponding to this summary. This is only used in
  /// per-module summaries.
  const GlobalValue *GV = nullptr;

  /// List of global value summary structures for a particular value held
  /// in the GlobalValueMap. Requires a vector in the case of multiple
  /// COMDAT values of the same name.
  GlobalValueSummaryList SummaryList;
};

/// Map from global value GUID to corresponding summary structures. Use a
/// std::map rather than a DenseMap so that pointers to the map's value_type
/// (which are used by ValueInfo) are not invalidated by insertion. Also it will
/// likely incur less overhead, as the value type is not very small and the size
/// of the map is unknown, resulting in inefficiencies due to repeated
/// insertions and resizing.
using GlobalValueSummaryMapTy =
    std::map<GlobalValue::GUID, GlobalValueSummaryInfo>;

/// Struct that holds a reference to a particular GUID in a global value
/// summary.
struct ValueInfo {
  const GlobalValueSummaryMapTy::value_type *Ref = nullptr;

  ValueInfo() = default;
  ValueInfo(const GlobalValueSummaryMapTy::value_type *Ref) : Ref(Ref) {}

  operator bool() const { return Ref; }

  GlobalValue::GUID getGUID() const { return Ref->first; }
  const GlobalValue *getValue() const { return Ref->second.GV; }

  ArrayRef<std::unique_ptr<GlobalValueSummary>> getSummaryList() const {
    return Ref->second.SummaryList;
  }
};

template <> struct DenseMapInfo<ValueInfo> {
  static inline ValueInfo getEmptyKey() {
    return ValueInfo((GlobalValueSummaryMapTy::value_type *)-1);
  }

  static inline ValueInfo getTombstoneKey() {
    return ValueInfo((GlobalValueSummaryMapTy::value_type *)-2);
  }

  static bool isEqual(ValueInfo L, ValueInfo R) { return L.Ref == R.Ref; }
  static unsigned getHashValue(ValueInfo I) { return (uintptr_t)I.Ref; }
};

/// \brief Function and variable summary information to aid decisions and
/// implementation of importing.
class GlobalValueSummary {
public:
  /// \brief Sububclass discriminator (for dyn_cast<> et al.)
  enum SummaryKind : unsigned { AliasKind, FunctionKind, GlobalVarKind };

  /// Group flags (Linkage, NotEligibleToImport, etc.) as a bitfield.
  struct GVFlags {
    /// \brief The linkage type of the associated global value.
    ///
    /// One use is to flag values that have local linkage types and need to
    /// have module identifier appended before placing into the combined
    /// index, to disambiguate from other values with the same name.
    /// In the future this will be used to update and optimize linkage
    /// types based on global summary-based analysis.
    unsigned Linkage : 4;

    /// Indicate if the global value cannot be imported (e.g. it cannot
    /// be renamed or references something that can't be renamed).
    unsigned NotEligibleToImport : 1;

    /// In per-module summary, indicate that the global value must be considered
    /// a live root for index-based liveness analysis. Used for special LLVM
    /// values such as llvm.global_ctors that the linker does not know about.
    ///
    /// In combined summary, indicate that the global value is live.
    unsigned Live : 1;

    /// Convenience Constructors
    explicit GVFlags(GlobalValue::LinkageTypes Linkage,
                     bool NotEligibleToImport, bool Live)
        : Linkage(Linkage), NotEligibleToImport(NotEligibleToImport),
          Live(Live) {}
  };

private:
  /// Kind of summary for use in dyn_cast<> et al.
  SummaryKind Kind;

  GVFlags Flags;

  /// This is the hash of the name of the symbol in the original file. It is
  /// identical to the GUID for global symbols, but differs for local since the
  /// GUID includes the module level id in the hash.
  GlobalValue::GUID OriginalName = 0;

  /// \brief Path of module IR containing value's definition, used to locate
  /// module during importing.
  ///
  /// This is only used during parsing of the combined index, or when
  /// parsing the per-module index for creation of the combined summary index,
  /// not during writing of the per-module index which doesn't contain a
  /// module path string table.
  StringRef ModulePath;

  /// List of values referenced by this global value's definition
  /// (either by the initializer of a global variable, or referenced
  /// from within a function). This does not include functions called, which
  /// are listed in the derived FunctionSummary object.
  std::vector<ValueInfo> RefEdgeList;

  bool isLive() const { return Flags.Live; }

protected:
  GlobalValueSummary(SummaryKind K, GVFlags Flags, std::vector<ValueInfo> Refs)
      : Kind(K), Flags(Flags), RefEdgeList(std::move(Refs)) {
    assert((K != AliasKind || Refs.empty()) &&
           "Expect no references for AliasSummary");
  }

public:
  virtual ~GlobalValueSummary() = default;

  /// Returns the hash of the original name, it is identical to the GUID for
  /// externally visible symbols, but not for local ones.
  GlobalValue::GUID getOriginalName() { return OriginalName; }

  /// Initialize the original name hash in this summary.
  void setOriginalName(GlobalValue::GUID Name) { OriginalName = Name; }

  /// Which kind of summary subclass this is.
  SummaryKind getSummaryKind() const { return Kind; }

  /// Set the path to the module containing this function, for use in
  /// the combined index.
  void setModulePath(StringRef ModPath) { ModulePath = ModPath; }

  /// Get the path to the module containing this function.
  StringRef modulePath() const { return ModulePath; }

  /// Get the flags for this GlobalValue (see \p struct GVFlags).
  GVFlags flags() { return Flags; }

  /// Return linkage type recorded for this global value.
  GlobalValue::LinkageTypes linkage() const {
    return static_cast<GlobalValue::LinkageTypes>(Flags.Linkage);
  }

  /// Sets the linkage to the value determined by global summary-based
  /// optimization. Will be applied in the ThinLTO backends.
  void setLinkage(GlobalValue::LinkageTypes Linkage) {
    Flags.Linkage = Linkage;
  }

  /// Return true if this global value can't be imported.
  bool notEligibleToImport() const { return Flags.NotEligibleToImport; }

  void setLive(bool Live) { Flags.Live = Live; }

  /// Flag that this global value cannot be imported.
  void setNotEligibleToImport() { Flags.NotEligibleToImport = true; }

  /// Return the list of values referenced by this global value definition.
  ArrayRef<ValueInfo> refs() const { return RefEdgeList; }

  /// If this is an alias summary, returns the summary of the aliased object (a
  /// global variable or function), otherwise returns itself.
  GlobalValueSummary *getBaseObject();

  friend class ModuleSummaryIndex;
  friend void computeDeadSymbols(class ModuleSummaryIndex &,
                                 const DenseSet<GlobalValue::GUID> &);
};

/// \brief Alias summary information.
class AliasSummary : public GlobalValueSummary {
  GlobalValueSummary *AliaseeSummary;

public:
  AliasSummary(GVFlags Flags)
      : GlobalValueSummary(AliasKind, Flags, ArrayRef<ValueInfo>{}) {}

  /// Check if this is an alias summary.
  static bool classof(const GlobalValueSummary *GVS) {
    return GVS->getSummaryKind() == AliasKind;
  }

  void setAliasee(GlobalValueSummary *Aliasee) { AliaseeSummary = Aliasee; }

  const GlobalValueSummary &getAliasee() const {
    assert(AliaseeSummary && "Unexpected missing aliasee summary");
    return *AliaseeSummary;
  }

  GlobalValueSummary &getAliasee() {
    return const_cast<GlobalValueSummary &>(
                         static_cast<const AliasSummary *>(this)->getAliasee());
  }
};

inline GlobalValueSummary *GlobalValueSummary::getBaseObject() {
  if (auto *AS = dyn_cast<AliasSummary>(this))
    return &AS->getAliasee();
  return this;
}

/// \brief Function summary information to aid decisions and implementation of
/// importing.
class FunctionSummary : public GlobalValueSummary {
public:
  /// <CalleeValueInfo, CalleeInfo> call edge pair.
  using EdgeTy = std::pair<ValueInfo, CalleeInfo>;

  /// An "identifier" for a virtual function. This contains the type identifier
  /// represented as a GUID and the offset from the address point to the virtual
  /// function pointer, where "address point" is as defined in the Itanium ABI:
  /// https://itanium-cxx-abi.github.io/cxx-abi/abi.html#vtable-general
  struct VFuncId {
    GlobalValue::GUID GUID;
    uint64_t Offset;
  };

  /// A specification for a virtual function call with all constant integer
  /// arguments. This is used to perform virtual constant propagation on the
  /// summary.
  struct ConstVCall {
    VFuncId VFunc;
    std::vector<uint64_t> Args;
  };

  /// Function attribute flags. Used to track if a function accesses memory,
  /// recurses or aliases.
  struct FFlags {
    unsigned ReadNone : 1;
    unsigned ReadOnly : 1;
    unsigned NoRecurse : 1;
    unsigned ReturnDoesNotAlias : 1;
  };

private:
  /// Number of instructions (ignoring debug instructions, e.g.) computed
  /// during the initial compile step when the summary index is first built.
  unsigned InstCount;

  /// Function attribute flags. Used to track if a function accesses memory,
  /// recurses or aliases.
  FFlags FunFlags;

  /// List of <CalleeValueInfo, CalleeInfo> call edge pairs from this function.
  std::vector<EdgeTy> CallGraphEdgeList;

  /// All type identifier related information. Because these fields are
  /// relatively uncommon we only allocate space for them if necessary.
  struct TypeIdInfo {
    /// List of type identifiers used by this function in llvm.type.test
    /// intrinsics other than by an llvm.assume intrinsic, represented as GUIDs.
    std::vector<GlobalValue::GUID> TypeTests;

    /// List of virtual calls made by this function using (respectively)
    /// llvm.assume(llvm.type.test) or llvm.type.checked.load intrinsics that do
    /// not have all constant integer arguments.
    std::vector<VFuncId> TypeTestAssumeVCalls, TypeCheckedLoadVCalls;

    /// List of virtual calls made by this function using (respectively)
    /// llvm.assume(llvm.type.test) or llvm.type.checked.load intrinsics with
    /// all constant integer arguments.
    std::vector<ConstVCall> TypeTestAssumeConstVCalls,
        TypeCheckedLoadConstVCalls;
  };

  std::unique_ptr<TypeIdInfo> TIdInfo;

public:
  FunctionSummary(GVFlags Flags, unsigned NumInsts, FFlags FunFlags,
                  std::vector<ValueInfo> Refs, std::vector<EdgeTy> CGEdges,
                  std::vector<GlobalValue::GUID> TypeTests,
                  std::vector<VFuncId> TypeTestAssumeVCalls,
                  std::vector<VFuncId> TypeCheckedLoadVCalls,
                  std::vector<ConstVCall> TypeTestAssumeConstVCalls,
                  std::vector<ConstVCall> TypeCheckedLoadConstVCalls)
      : GlobalValueSummary(FunctionKind, Flags, std::move(Refs)),
        InstCount(NumInsts), FunFlags(FunFlags),
        CallGraphEdgeList(std::move(CGEdges)) {
    if (!TypeTests.empty() || !TypeTestAssumeVCalls.empty() ||
        !TypeCheckedLoadVCalls.empty() || !TypeTestAssumeConstVCalls.empty() ||
        !TypeCheckedLoadConstVCalls.empty())
      TIdInfo = llvm::make_unique<TypeIdInfo>(TypeIdInfo{
          std::move(TypeTests), std::move(TypeTestAssumeVCalls),
          std::move(TypeCheckedLoadVCalls),
          std::move(TypeTestAssumeConstVCalls),
          std::move(TypeCheckedLoadConstVCalls)});
  }

  /// Check if this is a function summary.
  static bool classof(const GlobalValueSummary *GVS) {
    return GVS->getSummaryKind() == FunctionKind;
  }

  /// Get function attribute flags.
  FFlags &fflags() { return FunFlags; }

  /// Get the instruction count recorded for this function.
  unsigned instCount() const { return InstCount; }

  /// Return the list of <CalleeValueInfo, CalleeInfo> pairs.
  ArrayRef<EdgeTy> calls() const { return CallGraphEdgeList; }

  /// Returns the list of type identifiers used by this function in
  /// llvm.type.test intrinsics other than by an llvm.assume intrinsic,
  /// represented as GUIDs.
  ArrayRef<GlobalValue::GUID> type_tests() const {
    if (TIdInfo)
      return TIdInfo->TypeTests;
    return {};
  }

  /// Returns the list of virtual calls made by this function using
  /// llvm.assume(llvm.type.test) intrinsics that do not have all constant
  /// integer arguments.
  ArrayRef<VFuncId> type_test_assume_vcalls() const {
    if (TIdInfo)
      return TIdInfo->TypeTestAssumeVCalls;
    return {};
  }

  /// Returns the list of virtual calls made by this function using
  /// llvm.type.checked.load intrinsics that do not have all constant integer
  /// arguments.
  ArrayRef<VFuncId> type_checked_load_vcalls() const {
    if (TIdInfo)
      return TIdInfo->TypeCheckedLoadVCalls;
    return {};
  }

  /// Returns the list of virtual calls made by this function using
  /// llvm.assume(llvm.type.test) intrinsics with all constant integer
  /// arguments.
  ArrayRef<ConstVCall> type_test_assume_const_vcalls() const {
    if (TIdInfo)
      return TIdInfo->TypeTestAssumeConstVCalls;
    return {};
  }

  /// Returns the list of virtual calls made by this function using
  /// llvm.type.checked.load intrinsics with all constant integer arguments.
  ArrayRef<ConstVCall> type_checked_load_const_vcalls() const {
    if (TIdInfo)
      return TIdInfo->TypeCheckedLoadConstVCalls;
    return {};
  }

  /// Add a type test to the summary. This is used by WholeProgramDevirt if we
  /// were unable to devirtualize a checked call.
  void addTypeTest(GlobalValue::GUID Guid) {
    if (!TIdInfo)
      TIdInfo = llvm::make_unique<TypeIdInfo>();
    TIdInfo->TypeTests.push_back(Guid);
  }
};

template <> struct DenseMapInfo<FunctionSummary::VFuncId> {
  static FunctionSummary::VFuncId getEmptyKey() { return {0, uint64_t(-1)}; }

  static FunctionSummary::VFuncId getTombstoneKey() {
    return {0, uint64_t(-2)};
  }

  static bool isEqual(FunctionSummary::VFuncId L, FunctionSummary::VFuncId R) {
    return L.GUID == R.GUID && L.Offset == R.Offset;
  }

  static unsigned getHashValue(FunctionSummary::VFuncId I) { return I.GUID; }
};

template <> struct DenseMapInfo<FunctionSummary::ConstVCall> {
  static FunctionSummary::ConstVCall getEmptyKey() {
    return {{0, uint64_t(-1)}, {}};
  }

  static FunctionSummary::ConstVCall getTombstoneKey() {
    return {{0, uint64_t(-2)}, {}};
  }

  static bool isEqual(FunctionSummary::ConstVCall L,
                      FunctionSummary::ConstVCall R) {
    return DenseMapInfo<FunctionSummary::VFuncId>::isEqual(L.VFunc, R.VFunc) &&
           L.Args == R.Args;
  }

  static unsigned getHashValue(FunctionSummary::ConstVCall I) {
    return I.VFunc.GUID;
  }
};

/// \brief Global variable summary information to aid decisions and
/// implementation of importing.
///
/// Currently this doesn't add anything to the base \p GlobalValueSummary,
/// but is a placeholder as additional info may be added to the summary
/// for variables.
class GlobalVarSummary : public GlobalValueSummary {

public:
  GlobalVarSummary(GVFlags Flags, std::vector<ValueInfo> Refs)
      : GlobalValueSummary(GlobalVarKind, Flags, std::move(Refs)) {}

  /// Check if this is a global variable summary.
  static bool classof(const GlobalValueSummary *GVS) {
    return GVS->getSummaryKind() == GlobalVarKind;
  }
};

struct TypeTestResolution {
  /// Specifies which kind of type check we should emit for this byte array.
  /// See http://clang.llvm.org/docs/ControlFlowIntegrityDesign.html for full
  /// details on each kind of check; the enumerators are described with
  /// reference to that document.
  enum Kind {
    Unsat,     ///< Unsatisfiable type (i.e. no global has this type metadata)
    ByteArray, ///< Test a byte array (first example)
    Inline,    ///< Inlined bit vector ("Short Inline Bit Vectors")
    Single,    ///< Single element (last example in "Short Inline Bit Vectors")
    AllOnes,   ///< All-ones bit vector ("Eliminating Bit Vector Checks for
               ///  All-Ones Bit Vectors")
  } TheKind = Unsat;

  /// Range of size-1 expressed as a bit width. For example, if the size is in
  /// range [1,256], this number will be 8. This helps generate the most compact
  /// instruction sequences.
  unsigned SizeM1BitWidth = 0;

  // The following fields are only used if the target does not support the use
  // of absolute symbols to store constants. Their meanings are the same as the
  // corresponding fields in LowerTypeTestsModule::TypeIdLowering in
  // LowerTypeTests.cpp.

  uint64_t AlignLog2 = 0;
  uint64_t SizeM1 = 0;
  uint8_t BitMask = 0;
  uint64_t InlineBits = 0;
};

struct WholeProgramDevirtResolution {
  enum Kind {
    Indir,      ///< Just do a regular virtual call
    SingleImpl, ///< Single implementation devirtualization
  } TheKind = Indir;

  std::string SingleImplName;

  struct ByArg {
    enum Kind {
      Indir,            ///< Just do a regular virtual call
      UniformRetVal,    ///< Uniform return value optimization
      UniqueRetVal,     ///< Unique return value optimization
      VirtualConstProp, ///< Virtual constant propagation
    } TheKind = Indir;

    /// Additional information for the resolution:
    /// - UniformRetVal: the uniform return value.
    /// - UniqueRetVal: the return value associated with the unique vtable (0 or
    ///   1).
    uint64_t Info = 0;

    // The following fields are only used if the target does not support the use
    // of absolute symbols to store constants.

    uint32_t Byte = 0;
    uint32_t Bit = 0;
  };

  /// Resolutions for calls with all constant integer arguments (excluding the
  /// first argument, "this"), where the key is the argument vector.
  std::map<std::vector<uint64_t>, ByArg> ResByArg;
};

struct TypeIdSummary {
  TypeTestResolution TTRes;

  /// Mapping from byte offset to whole-program devirt resolution for that
  /// (typeid, byte offset) pair.
  std::map<uint64_t, WholeProgramDevirtResolution> WPDRes;
};

/// 160 bits SHA1
using ModuleHash = std::array<uint32_t, 5>;

/// Type used for iterating through the global value summary map.
using const_gvsummary_iterator = GlobalValueSummaryMapTy::const_iterator;
using gvsummary_iterator = GlobalValueSummaryMapTy::iterator;

/// String table to hold/own module path strings, which additionally holds the
/// module ID assigned to each module during the plugin step, as well as a hash
/// of the module. The StringMap makes a copy of and owns inserted strings.
using ModulePathStringTableTy = StringMap<std::pair<uint64_t, ModuleHash>>;

/// Map of global value GUID to its summary, used to identify values defined in
/// a particular module, and provide efficient access to their summary.
using GVSummaryMapTy = DenseMap<GlobalValue::GUID, GlobalValueSummary *>;

/// Class to hold module path string table and global value map,
/// and encapsulate methods for operating on them.
class ModuleSummaryIndex {
private:
  /// Map from value name to list of summary instances for values of that
  /// name (may be duplicates in the COMDAT case, e.g.).
  GlobalValueSummaryMapTy GlobalValueMap;

  /// Holds strings for combined index, mapping to the corresponding module ID.
  ModulePathStringTableTy ModulePathStringTable;

  /// Mapping from type identifiers to summary information for that type
  /// identifier.
  // FIXME: Add bitcode read/write support for this field.
  std::map<std::string, TypeIdSummary> TypeIdMap;

  /// Mapping from original ID to GUID. If original ID can map to multiple
  /// GUIDs, it will be mapped to 0.
  std::map<GlobalValue::GUID, GlobalValue::GUID> OidGuidMap;

  /// Indicates that summary-based GlobalValue GC has run, and values with
  /// GVFlags::Live==false are really dead. Otherwise, all values must be
  /// considered live.
  bool WithGlobalValueDeadStripping = false;

  std::set<std::string> CfiFunctionDefs;
  std::set<std::string> CfiFunctionDecls;

  // YAML I/O support.
  friend yaml::MappingTraits<ModuleSummaryIndex>;

  GlobalValueSummaryMapTy::value_type *
  getOrInsertValuePtr(GlobalValue::GUID GUID) {
    return &*GlobalValueMap.emplace(GUID, GlobalValueSummaryInfo{}).first;
  }

public:
  gvsummary_iterator begin() { return GlobalValueMap.begin(); }
  const_gvsummary_iterator begin() const { return GlobalValueMap.begin(); }
  gvsummary_iterator end() { return GlobalValueMap.end(); }
  const_gvsummary_iterator end() const { return GlobalValueMap.end(); }
  size_t size() const { return GlobalValueMap.size(); }

  bool withGlobalValueDeadStripping() const {
    return WithGlobalValueDeadStripping;
  }
  void setWithGlobalValueDeadStripping() {
    WithGlobalValueDeadStripping = true;
  }

  bool isGlobalValueLive(const GlobalValueSummary *GVS) const {
    return !WithGlobalValueDeadStripping || GVS->isLive();
  }
  bool isGUIDLive(GlobalValue::GUID GUID) const;

  /// Return a ValueInfo for GUID if it exists, otherwise return ValueInfo().
  ValueInfo getValueInfo(GlobalValue::GUID GUID) const {
    auto I = GlobalValueMap.find(GUID);
    return ValueInfo(I == GlobalValueMap.end() ? nullptr : &*I);
  }

  /// Return a ValueInfo for \p GUID.
  ValueInfo getOrInsertValueInfo(GlobalValue::GUID GUID) {
    return ValueInfo(getOrInsertValuePtr(GUID));
  }

  /// Return a ValueInfo for \p GV and mark it as belonging to GV.
  ValueInfo getOrInsertValueInfo(const GlobalValue *GV) {
    auto VP = getOrInsertValuePtr(GV->getGUID());
    VP->second.GV = GV;
    return ValueInfo(VP);
  }

  /// Return the GUID for \p OriginalId in the OidGuidMap.
  GlobalValue::GUID getGUIDFromOriginalID(GlobalValue::GUID OriginalID) const {
    const auto I = OidGuidMap.find(OriginalID);
    return I == OidGuidMap.end() ? 0 : I->second;
  }

  std::set<std::string> &cfiFunctionDefs() { return CfiFunctionDefs; }
  const std::set<std::string> &cfiFunctionDefs() const { return CfiFunctionDefs; }

  std::set<std::string> &cfiFunctionDecls() { return CfiFunctionDecls; }
  const std::set<std::string> &cfiFunctionDecls() const { return CfiFunctionDecls; }

  /// Add a global value summary for a value of the given name.
  void addGlobalValueSummary(StringRef ValueName,
                             std::unique_ptr<GlobalValueSummary> Summary) {
    addGlobalValueSummary(getOrInsertValueInfo(GlobalValue::getGUID(ValueName)),
                          std::move(Summary));
  }

  /// Add a global value summary for the given ValueInfo.
  void addGlobalValueSummary(ValueInfo VI,
                             std::unique_ptr<GlobalValueSummary> Summary) {
    addOriginalName(VI.getGUID(), Summary->getOriginalName());
    // Here we have a notionally const VI, but the value it points to is owned
    // by the non-const *this.
    const_cast<GlobalValueSummaryMapTy::value_type *>(VI.Ref)
        ->second.SummaryList.push_back(std::move(Summary));
  }

  /// Add an original name for the value of the given GUID.
  void addOriginalName(GlobalValue::GUID ValueGUID,
                       GlobalValue::GUID OrigGUID) {
    if (OrigGUID == 0 || ValueGUID == OrigGUID)
      return;
    if (OidGuidMap.count(OrigGUID) && OidGuidMap[OrigGUID] != ValueGUID)
      OidGuidMap[OrigGUID] = 0;
    else
      OidGuidMap[OrigGUID] = ValueGUID;
  }

  /// Find the summary for global \p GUID in module \p ModuleId, or nullptr if
  /// not found.
  GlobalValueSummary *findSummaryInModule(GlobalValue::GUID ValueGUID,
                                          StringRef ModuleId) const {
    auto CalleeInfo = getValueInfo(ValueGUID);
    if (!CalleeInfo) {
      return nullptr; // This function does not have a summary
    }
    auto Summary =
        llvm::find_if(CalleeInfo.getSummaryList(),
                      [&](const std::unique_ptr<GlobalValueSummary> &Summary) {
                        return Summary->modulePath() == ModuleId;
                      });
    if (Summary == CalleeInfo.getSummaryList().end())
      return nullptr;
    return Summary->get();
  }

  /// Returns the first GlobalValueSummary for \p GV, asserting that there
  /// is only one if \p PerModuleIndex.
  GlobalValueSummary *getGlobalValueSummary(const GlobalValue &GV,
                                            bool PerModuleIndex = true) const {
    assert(GV.hasName() && "Can't get GlobalValueSummary for GV with no name");
    return getGlobalValueSummary(GlobalValue::getGUID(GV.getName()),
                                 PerModuleIndex);
  }

  /// Returns the first GlobalValueSummary for \p ValueGUID, asserting that
  /// there
  /// is only one if \p PerModuleIndex.
  GlobalValueSummary *getGlobalValueSummary(GlobalValue::GUID ValueGUID,
                                            bool PerModuleIndex = true) const;

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

  /// Convenience method for creating a promoted global name
  /// for the given value name of a local, and its original module's ID.
  static std::string getGlobalNameForLocal(StringRef Name, ModuleHash ModHash) {
    SmallString<256> NewName(Name);
    NewName += ".llvm.";
    NewName += utostr(ModHash[0]); // Take the first 32 bits
    return NewName.str();
  }

  /// Helper to obtain the unpromoted name for a global value (or the original
  /// name if not promoted).
  static StringRef getOriginalNameBeforePromote(StringRef Name) {
    std::pair<StringRef, StringRef> Pair = Name.split(".llvm.");
    return Pair.first;
  }

  typedef ModulePathStringTableTy::value_type ModuleInfo;

  /// Add a new module with the given \p Hash, mapped to the given \p
  /// ModID, and return a reference to the module.
  ModuleInfo *addModule(StringRef ModPath, uint64_t ModId,
                        ModuleHash Hash = ModuleHash{{0}}) {
    return &*ModulePathStringTable.insert({ModPath, {ModId, Hash}}).first;
  }

  /// Check if the given Module has any functions available for exporting
  /// in the index. We consider any module present in the ModulePathStringTable
  /// to have exported functions.
  bool hasExportedFunctions(const Module &M) const {
    return ModulePathStringTable.count(M.getModuleIdentifier());
  }

  const std::map<std::string, TypeIdSummary> &typeIds() const {
    return TypeIdMap;
  }

  /// This accessor should only be used when exporting because it can mutate the
  /// map.
  TypeIdSummary &getOrInsertTypeIdSummary(StringRef TypeId) {
    return TypeIdMap[TypeId];
  }

  /// This returns either a pointer to the type id summary (if present in the
  /// summary map) or null (if not present). This may be used when importing.
  const TypeIdSummary *getTypeIdSummary(StringRef TypeId) const {
    auto I = TypeIdMap.find(TypeId);
    if (I == TypeIdMap.end())
      return nullptr;
    return &I->second;
  }

  /// Collect for the given module the list of function it defines
  /// (GUID -> Summary).
  void collectDefinedFunctionsForModule(StringRef ModulePath,
                                        GVSummaryMapTy &GVSummaryMap) const;

  /// Collect for each module the list of Summaries it defines (GUID ->
  /// Summary).
  void collectDefinedGVSummariesPerModule(
      StringMap<GVSummaryMapTy> &ModuleToDefinedGVSummaries) const;
};

} // end namespace llvm

#endif // LLVM_IR_MODULESUMMARYINDEX_H
