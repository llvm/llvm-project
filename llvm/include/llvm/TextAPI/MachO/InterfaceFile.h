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
#include "llvm/TextAPI/MachO/Symbol.h"

namespace llvm {
namespace MachO {

/// Defines the list of MachO platforms.
enum class PlatformKind : unsigned {
  unknown,
  macOS = MachO::PLATFORM_MACOS,
  iOS = MachO::PLATFORM_IOS,
  tvOS = MachO::PLATFORM_TVOS,
  watchOS = MachO::PLATFORM_WATCHOS,
  bridgeOS = MachO::PLATFORM_BRIDGEOS,
};

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

  All     = ~0U,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/All),
};

// clang-format on

/// Reference to an interface file.
class InterfaceFileRef {
public:
  InterfaceFileRef() = default;

  InterfaceFileRef(StringRef InstallName) : InstallName(InstallName) {}

  InterfaceFileRef(StringRef InstallName, ArchitectureSet Archs)
      : InstallName(InstallName), Architectures(Archs) {}

  StringRef getInstallName() const { return InstallName; };
  void addArchitectures(ArchitectureSet Archs) { Architectures |= Archs; }
  ArchitectureSet getArchitectures() const { return Architectures; }
  bool hasArchitecture(Architecture Arch) const {
    return Architectures.has(Arch);
  }

  bool operator==(const InterfaceFileRef &O) const {
    return std::tie(InstallName, Architectures) ==
           std::tie(O.InstallName, O.Architectures);
  }

  bool operator<(const InterfaceFileRef &O) const {
    return std::tie(InstallName, Architectures) <
           std::tie(O.InstallName, O.Architectures);
  }

private:
  std::string InstallName;
  ArchitectureSet Architectures;
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
  void setPath(StringRef Path_) { Path = Path_; }

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

  /// Set the platform.
  void setPlatform(PlatformKind Platform_) { Platform = Platform_; }

  /// Get the platform.
  PlatformKind getPlatform() const { return Platform; }

  /// Specify the set of supported architectures by this file.
  void setArchitectures(ArchitectureSet Architectures_) {
    Architectures = Architectures_;
  }

  /// Add the set of supported architectures by this file.
  void addArchitectures(ArchitectureSet Architectures_) {
    Architectures |= Architectures_;
  }

  /// Add supported architecture by this file..
  void addArch(Architecture Arch) { Architectures.set(Arch); }

  /// Get the set of supported architectures.
  ArchitectureSet getArchitectures() const { return Architectures; }

  /// Set the install name of the library.
  void setInstallName(StringRef InstallName_) { InstallName = InstallName_; }

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

  /// Set the parent umbrella framework.
  void setParentUmbrella(StringRef Parent) { ParentUmbrella = Parent; }

  /// Get the parent umbrella framework.
  StringRef getParentUmbrella() const { return ParentUmbrella; }

  /// Add an allowable client.
  ///
  /// Mach-O Dynamic libraries have the concept of allowable clients that are
  /// checked during static link time. The name of the application or library
  /// that is being generated needs to match one of the allowable clients or the
  /// linker refuses to link this library.
  ///
  /// \param Name The name of the client that is allowed to link this library.
  /// \param Architectures The set of architecture for which this applies.
  void addAllowableClient(StringRef Name, ArchitectureSet Architectures);

  /// Get the list of allowable clients.
  ///
  /// \return Returns a list of allowable clients.
  const std::vector<InterfaceFileRef> &allowableClients() const {
    return AllowableClients;
  }

  /// Add a re-exported library.
  ///
  /// \param InstallName The name of the library to re-export.
  /// \param Architectures The set of architecture for which this applies.
  void addReexportedLibrary(StringRef InstallName,
                            ArchitectureSet Architectures);

  /// Get the list of re-exported libraries.
  ///
  /// \return Returns a list of re-exported libraries.
  const std::vector<InterfaceFileRef> &reexportedLibraries() const {
    return ReexportedLibraries;
  }

  /// Add an architecture/UUID pair.
  ///
  /// \param Arch The architecture for which this applies.
  /// \param UUID The UUID of the library for the specified architecture.
  void addUUID(Architecture Arch, StringRef UUID);

  /// Add an architecture/UUID pair.
  ///
  /// \param Arch The architecture for which this applies.
  /// \param UUID The UUID of the library for the specified architecture.
  void addUUID(Architecture Arch, uint8_t UUID[16]);

  /// Get the list of architecture/UUID pairs.
  ///
  /// \return Returns a list of architecture/UUID pairs.
  const std::vector<std::pair<Architecture, std::string>> &uuids() const {
    return UUIDs;
  }

  /// Add a symbol to the symbols list or extend an existing one.
  void addSymbol(SymbolKind Kind, StringRef Name, ArchitectureSet Architectures,
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

  // Custom iterator to return only exported symbols.
  struct const_export_iterator
      : public iterator_adaptor_base<
            const_export_iterator, const_symbol_iterator,
            std::forward_iterator_tag, const Symbol *> {
    const_symbol_iterator _end;

    void skipToNextSymbol() {
      while (I != _end && I->isUndefined())
        ++I;
    }

    const_export_iterator() = default;
    template <typename U>
    const_export_iterator(U &&it, U &&end)
        : iterator_adaptor_base(std::forward<U &&>(it)),
          _end(std::forward<U &&>(end)) {
      skipToNextSymbol();
    }

    const_export_iterator &operator++() {
      ++I;
      skipToNextSymbol();
      return *this;
    }

    const_export_iterator operator++(int) {
      const_export_iterator tmp(*this);
      ++(*this);
      return tmp;
    }
  };
  using const_export_range = llvm::iterator_range<const_export_iterator>;

  // Custom iterator to return only undefined symbols.
  struct const_undefined_iterator
      : public iterator_adaptor_base<
            const_undefined_iterator, const_symbol_iterator,
            std::forward_iterator_tag, const Symbol *> {
    const_symbol_iterator _end;

    void skipToNextSymbol() {
      while (I != _end && !I->isUndefined())
        ++I;
    }

    const_undefined_iterator() = default;
    template <typename U>
    const_undefined_iterator(U &&it, U &&end)
        : iterator_adaptor_base(std::forward<U &&>(it)),
          _end(std::forward<U &&>(end)) {
      skipToNextSymbol();
    }

    const_undefined_iterator &operator++() {
      ++I;
      skipToNextSymbol();
      return *this;
    }

    const_undefined_iterator operator++(int) {
      const_undefined_iterator tmp(*this);
      ++(*this);
      return tmp;
    }
  };
  using const_undefined_range = llvm::iterator_range<const_undefined_iterator>;

  const_symbol_range symbols() const {
    return {Symbols.begin(), Symbols.end()};
  }
  const_export_range exports() const {
    return {{Symbols.begin(), Symbols.end()}, {Symbols.end(), Symbols.end()}};
  }
  const_undefined_range undefineds() const {
    return {{Symbols.begin(), Symbols.end()}, {Symbols.end(), Symbols.end()}};
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

  std::string Path;
  FileType FileKind;
  PlatformKind Platform;
  ArchitectureSet Architectures;
  std::string InstallName;
  PackedVersion CurrentVersion;
  PackedVersion CompatibilityVersion;
  uint8_t SwiftABIVersion{0};
  bool IsTwoLevelNamespace{false};
  bool IsAppExtensionSafe{false};
  bool IsInstallAPI{false};
  ObjCConstraintType ObjcConstraint = ObjCConstraintType::None;
  std::string ParentUmbrella;
  std::vector<InterfaceFileRef> AllowableClients;
  std::vector<InterfaceFileRef> ReexportedLibraries;
  std::vector<std::pair<Architecture, std::string>> UUIDs;
  SymbolMapType Symbols;
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_MACHO_INTERFACE_FILE_H
