//===- llvm/TextAPI/MachO/IntefaceFile.h - TAPI Interface File --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A generic and abstract interface representation for linkable objects. This
// could be an MachO executable, bundle, dylib, or text-based stub file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_MACHO_INTERFACE_FILE_H
#define LLVM_TEXTAPI_MACHO_INTERFACE_FILE_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/TextAPI/MachO/Architecture.h"
#include "llvm/TextAPI/MachO/ArchitectureSet.h"
#include "llvm/TextAPI/MachO/PackedVersion.h"
#include "llvm/TextAPI/MachO/Platform.h"
#include "llvm/TextAPI/MachO/Symbol.h"
#include "llvm/TextAPI/MachO/Target.h"

namespace llvm {
namespace MachO {

/// Defines a list of Objective-C constraints.
enum class ObjCConstraintType : unsigned {
  /// No constraint.
  None = 0,

  /// Retain/Release.
  Retain_Release = 1,

  /// Retain/Release for Simulator.
  Retain_Release_For_Simulator = 2,

  /// Retain/Release or Garbage Collection.
  Retain_Release_Or_GC = 3,

  /// Garbage Collection.
  GC = 4,
};

// clang-format off

/// Defines the file type this file represents.
enum FileType : unsigned {
  /// Invalid file type.
  Invalid = 0U,

  /// Text-based stub file (.tbd) version 1.0
  TBD_V1  = 1U <<  0,

  /// Text-based stub file (.tbd) version 2.0
  TBD_V2  = 1U <<  1,

  /// Text-based stub file (.tbd) version 3.0
  TBD_V3  = 1U <<  2,

  /// Text-based stub file (.tbd) version 4.0
  TBD_V4  = 1U <<  3,

  All     = ~0U,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/All),
};

// clang-format on

/// Reference to an interface file.
class InterfaceFileRef {
public:
  InterfaceFileRef() = default;

  InterfaceFileRef(StringRef InstallName) : InstallName(InstallName) {}

  InterfaceFileRef(StringRef InstallName, const TargetList Targets)
      : InstallName(InstallName), Targets(std::move(Targets)) {}

  StringRef getInstallName() const { return InstallName; };

  void addTarget(const Target &Target);
  template <typename RangeT> void addTargets(RangeT &&Targets) {
    for (const auto &Target : Targets)
      addTarget(Target(Target));
  }

  using const_target_iterator = TargetList::const_iterator;
  using const_target_range = llvm::iterator_range<const_target_iterator>;
  const_target_range targets() const { return {Targets}; }

  ArchitectureSet getArchitectures() const {
    return mapToArchitectureSet(Targets);
  }

  PlatformSet getPlatforms() const { return mapToPlatformSet(Targets); }

  bool operator==(const InterfaceFileRef &O) const {
    return std::tie(InstallName, Targets) == std::tie(O.InstallName, O.Targets);
  }

  bool operator!=(const InterfaceFileRef &O) const {
    return std::tie(InstallName, Targets) != std::tie(O.InstallName, O.Targets);
  }

  bool operator<(const InterfaceFileRef &O) const {
    return std::tie(InstallName, Targets) < std::tie(O.InstallName, O.Targets);
  }

private:
  std::string InstallName;
  TargetList Targets;
};

} // end namespace MachO.

struct SymbolsMapKey {
  MachO::SymbolKind Kind;
  StringRef Name;

  SymbolsMapKey(MachO::SymbolKind Kind, StringRef Name)
      : Kind(Kind), Name(Name) {}
};
template <> struct DenseMapInfo<SymbolsMapKey> {
  static inline SymbolsMapKey getEmptyKey() {
    return SymbolsMapKey(MachO::SymbolKind::GlobalSymbol, StringRef{});
  }

  static inline SymbolsMapKey getTombstoneKey() {
    return SymbolsMapKey(MachO::SymbolKind::ObjectiveCInstanceVariable,
                         StringRef{});
  }

  static unsigned getHashValue(const SymbolsMapKey &Key) {
    return hash_combine(hash_value(Key.Kind), hash_value(Key.Name));
  }

  static bool isEqual(const SymbolsMapKey &LHS, const SymbolsMapKey &RHS) {
    return std::tie(LHS.Kind, LHS.Name) == std::tie(RHS.Kind, RHS.Name);
  }
};

namespace MachO {

/// Defines the interface file.
class InterfaceFile {
public:
  /// Set the path from which this file was generated (if applicable).
  ///
  /// \param Path_ The path to the source file.
  void setPath(StringRef Path_) { Path = std::string(Path_); }

  /// Get the path from which this file was generated (if applicable).
  ///
  /// \return The path to the source file or empty.
  StringRef getPath() const { return Path; }

  /// Set the file type.
  ///
  /// This is used by the YAML writer to identify the specification it should
  /// use for writing the file.
  ///
  /// \param Kind The file type.
  void setFileType(FileType Kind) { FileKind = Kind; }

  /// Get the file type.
  ///
  /// \return The file type.
  FileType getFileType() const { return FileKind; }

  /// Get the architectures.
  ///
  /// \return The applicable architectures.
  ArchitectureSet getArchitectures() const {
    return mapToArchitectureSet(Targets);
  }

  /// Get the platforms.
  ///
  /// \return The applicable platforms.
  PlatformSet getPlatforms() const { return mapToPlatformSet(Targets); }

  /// Set and add target.
  ///
  /// \param Target the target to add into.
  void addTarget(const Target &Target);

  /// Set and add targets.
  ///
  /// Add the subset of llvm::triples that is supported by Tapi
  ///
  /// \param Targets the collection of targets.
  template <typename RangeT> void addTargets(RangeT &&Targets) {
    for (const auto &Target_ : Targets)
      addTarget(Target(Target_));
  }

  using const_target_iterator = TargetList::const_iterator;
  using const_target_range = llvm::iterator_range<const_target_iterator>;
  const_target_range targets() const { return {Targets}; }

  using const_filtered_target_iterator =
      llvm::filter_iterator<const_target_iterator,
                            std::function<bool(const Target &)>>;
  using const_filtered_target_range =
      llvm::iterator_range<const_filtered_target_iterator>;
  const_filtered_target_range targets(ArchitectureSet Archs) const;

  /// Set the install name of the library.
  void setInstallName(StringRef InstallName_) {
    InstallName = std::string(InstallName_);
  }

  /// Get the install name of the library.
  StringRef getInstallName() const { return InstallName; }

  /// Set the current version of the library.
  void setCurrentVersion(PackedVersion Version) { CurrentVersion = Version; }

  /// Get the current version of the library.
  PackedVersion getCurrentVersion() const { return CurrentVersion; }

  /// Set the compatibility version of the library.
  void setCompatibilityVersion(PackedVersion Version) {
    CompatibilityVersion = Version;
  }

  /// Get the compatibility version of the library.
  PackedVersion getCompatibilityVersion() const { return CompatibilityVersion; }

  /// Set the Swift ABI version of the library.
  void setSwiftABIVersion(uint8_t Version) { SwiftABIVersion = Version; }

  /// Get the Swift ABI version of the library.
  uint8_t getSwiftABIVersion() const { return SwiftABIVersion; }

  /// Specify if the library uses two-level namespace (or flat namespace).
  void setTwoLevelNamespace(bool V = true) { IsTwoLevelNamespace = V; }

  /// Check if the library uses two-level namespace.
  bool isTwoLevelNamespace() const { return IsTwoLevelNamespace; }

  /// Specify if the library is application extension safe (or not).
  void setApplicationExtensionSafe(bool V = true) { IsAppExtensionSafe = V; }

  /// Check if the library is application extension safe.
  bool isApplicationExtensionSafe() const { return IsAppExtensionSafe; }

  /// Set the Objective-C constraint.
  void setObjCConstraint(ObjCConstraintType Constraint) {
    ObjcConstraint = Constraint;
  }

  /// Get the Objective-C constraint.
  ObjCConstraintType getObjCConstraint() const { return ObjcConstraint; }

  /// Specify if this file was generated during InstallAPI (or not).
  void setInstallAPI(bool V = true) { IsInstallAPI = V; }

  /// Check if this file was generated during InstallAPI.
  bool isInstallAPI() const { return IsInstallAPI; }

  /// Set the parent umbrella frameworks.
  /// \param Target_ The target applicable to Parent
  /// \param Parent  The name of Parent
  void addParentUmbrella(const Target &Target_, StringRef Parent);
  const std::vector<std::pair<Target, std::string>> &umbrellas() const {
    return ParentUmbrellas;
  }

  /// Get the parent umbrella framework.
  const std::vector<std::pair<Target, std::string>> getParentUmbrellas() const {
    return ParentUmbrellas;
  }

  /// Add an allowable client.
  ///
  /// Mach-O Dynamic libraries have the concept of allowable clients that are
  /// checked during static link time. The name of the application or library
  /// that is being generated needs to match one of the allowable clients or the
  /// linker refuses to link this library.
  ///
  /// \param InstallName The name of the client that is allowed to link this library.
  /// \param Target The target triple for which this applies.
  void addAllowableClient(StringRef InstallName, const Target &Target);

  /// Get the list of allowable clients.
  ///
  /// \return Returns a list of allowable clients.
  const std::vector<InterfaceFileRef> &allowableClients() const {
    return AllowableClients;
  }

  /// Add a re-exported library.
  ///
  /// \param InstallName The name of the library to re-export.
  /// \param Target The target triple for which this applies.
  void addReexportedLibrary(StringRef InstallName, const Target &Target);

  /// Get the list of re-exported libraries.
  ///
  /// \return Returns a list of re-exported libraries.
  const std::vector<InterfaceFileRef> &reexportedLibraries() const {
    return ReexportedLibraries;
  }

  /// Add an Target/UUID pair.
  ///
  /// \param Target The target triple for which this applies.
  /// \param UUID The UUID of the library for the specified architecture.
  void addUUID(const Target &Target, StringRef UUID);

  /// Add an Target/UUID pair.
  ///
  /// \param Target The target triple for which this applies.
  /// \param UUID The UUID of the library for the specified architecture.
  void addUUID(const Target &Target, uint8_t UUID[16]);

  /// Get the list of Target/UUID pairs.
  ///
  /// \return Returns a list of Target/UUID pairs.
  const std::vector<std::pair<Target, std::string>> &uuids() const {
    return UUIDs;
  }

  /// Add a symbol to the symbols list or extend an existing one.
  void addSymbol(SymbolKind Kind, StringRef Name, const TargetList &Targets,
                 SymbolFlags Flags = SymbolFlags::None);

  using SymbolMapType = DenseMap<SymbolsMapKey, Symbol *>;
  struct const_symbol_iterator
      : public iterator_adaptor_base<
            const_symbol_iterator, SymbolMapType::const_iterator,
            std::forward_iterator_tag, const Symbol *, ptrdiff_t,
            const Symbol *, const Symbol *> {
    const_symbol_iterator() = default;

    template <typename U>
    const_symbol_iterator(U &&u)
        : iterator_adaptor_base(std::forward<U &&>(u)) {}

    reference operator*() const { return I->second; }
    pointer operator->() const { return I->second; }
  };

  using const_symbol_range = iterator_range<const_symbol_iterator>;

  using const_filtered_symbol_iterator =
      filter_iterator<const_symbol_iterator,
                      std::function<bool(const Symbol *)>>;
  using const_filtered_symbol_range =
      iterator_range<const_filtered_symbol_iterator>;

  const_symbol_range symbols() const {
    return {Symbols.begin(), Symbols.end()};
  }

  const_filtered_symbol_range exports() const {
    std::function<bool(const Symbol *)> fn = [](const Symbol *Symbol) {
      return !Symbol->isUndefined();
    };
    return make_filter_range(
        make_range<const_symbol_iterator>({Symbols.begin()}, {Symbols.end()}),
        fn);
  }

  const_filtered_symbol_range undefineds() const {
    std::function<bool(const Symbol *)> fn = [](const Symbol *Symbol) {
      return Symbol->isUndefined();
    };
    return make_filter_range(
        make_range<const_symbol_iterator>({Symbols.begin()}, {Symbols.end()}),
        fn);
  }

private:
  llvm::BumpPtrAllocator Allocator;
  StringRef copyString(StringRef String) {
    if (String.empty())
      return {};

    void *Ptr = Allocator.Allocate(String.size(), 1);
    memcpy(Ptr, String.data(), String.size());
    return StringRef(reinterpret_cast<const char *>(Ptr), String.size());
  }

  TargetList Targets;
  std::string Path;
  FileType FileKind;
  std::string InstallName;
  PackedVersion CurrentVersion;
  PackedVersion CompatibilityVersion;
  uint8_t SwiftABIVersion{0};
  bool IsTwoLevelNamespace{false};
  bool IsAppExtensionSafe{false};
  bool IsInstallAPI{false};
  ObjCConstraintType ObjcConstraint = ObjCConstraintType::None;
  std::vector<std::pair<Target, std::string>> ParentUmbrellas;
  std::vector<InterfaceFileRef> AllowableClients;
  std::vector<InterfaceFileRef> ReexportedLibraries;
  std::vector<std::pair<Target, std::string>> UUIDs;
  SymbolMapType Symbols;
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_MACHO_INTERFACE_FILE_H
