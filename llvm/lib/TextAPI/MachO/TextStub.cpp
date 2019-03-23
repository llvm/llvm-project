//===- TextStub.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the text stub file reader/writer.
//
//===----------------------------------------------------------------------===//

#include "TextAPIContext.h"
#include "TextStubCommon.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TextAPI/MachO/Architecture.h"
#include "llvm/TextAPI/MachO/ArchitectureSet.h"
#include "llvm/TextAPI/MachO/InterfaceFile.h"
#include "llvm/TextAPI/MachO/PackedVersion.h"
#include "llvm/TextAPI/MachO/TextAPIReader.h"
#include "llvm/TextAPI/MachO/TextAPIWriter.h"
#include <algorithm>
#include <set>

// clang-format off
/*

 YAML Format specification.

 The TBD v1 format only support two level address libraries and is per
 definition application extension safe.

---                              # the tag !tapi-tbd-v1 is optional and
                                 # shouldn't be emitted to support older linker.
archs: [ armv7, armv7s, arm64 ]  # the list of architecture slices that are
                                 # supported by this file.
platform: ios                    # Specifies the platform (macosx, ios, etc)
install-name: /u/l/libfoo.dylib  #
current-version: 1.2.3           # Optional: defaults to 1.0
compatibility-version: 1.0       # Optional: defaults to 1.0
swift-version: 0                 # Optional: defaults to 0
objc-constraint: none            # Optional: defaults to none
exports:                         # List of export sections
...

Each export section is defined as following:

 - archs: [ arm64 ]                   # the list of architecture slices
   allowed-clients: [ client ]        # Optional: List of clients
   re-exports: [ ]                    # Optional: List of re-exports
   symbols: [ _sym ]                  # Optional: List of symbols
   objc-classes: []                   # Optional: List of Objective-C classes
   objc-ivars: []                     # Optional: List of Objective C Instance
                                      #           Variables
   weak-def-symbols: []               # Optional: List of weak defined symbols
   thread-local-symbols: []           # Optional: List of thread local symbols
*/

/*

 YAML Format specification.

--- !tapi-tbd-v2
archs: [ armv7, armv7s, arm64 ]  # the list of architecture slices that are
                                 # supported by this file.
uuids: [ armv7:... ]             # Optional: List of architecture and UUID pairs.
platform: ios                    # Specifies the platform (macosx, ios, etc)
flags: []                        # Optional:
install-name: /u/l/libfoo.dylib  #
current-version: 1.2.3           # Optional: defaults to 1.0
compatibility-version: 1.0       # Optional: defaults to 1.0
swift-version: 0                 # Optional: defaults to 0
objc-constraint: retain_release  # Optional: defaults to retain_release
parent-umbrella:                 # Optional:
exports:                         # List of export sections
...
undefineds:                      # List of undefineds sections
...

Each export section is defined as following:

- archs: [ arm64 ]                   # the list of architecture slices
  allowed-clients: [ client ]        # Optional: List of clients
  re-exports: [ ]                    # Optional: List of re-exports
  symbols: [ _sym ]                  # Optional: List of symbols
  objc-classes: []                   # Optional: List of Objective-C classes
  objc-ivars: []                     # Optional: List of Objective C Instance
                                     #           Variables
  weak-def-symbols: []               # Optional: List of weak defined symbols
  thread-local-symbols: []           # Optional: List of thread local symbols

Each undefineds section is defined as following:
- archs: [ arm64 ]     # the list of architecture slices
  symbols: [ _sym ]    # Optional: List of symbols
  objc-classes: []     # Optional: List of Objective-C classes
  objc-ivars: []       # Optional: List of Objective C Instance Variables
  weak-ref-symbols: [] # Optional: List of weak defined symbols
*/

/*

 YAML Format specification.

--- !tapi-tbd-v3
archs: [ armv7, armv7s, arm64 ]  # the list of architecture slices that are
                                 # supported by this file.
uuids: [ armv7:... ]             # Optional: List of architecture and UUID pairs.
platform: ios                    # Specifies the platform (macosx, ios, etc)
flags: []                        # Optional:
install-name: /u/l/libfoo.dylib  #
current-version: 1.2.3           # Optional: defaults to 1.0
compatibility-version: 1.0       # Optional: defaults to 1.0
swift-abi-version: 0             # Optional: defaults to 0
objc-constraint: retain_release  # Optional: defaults to retain_release
parent-umbrella:                 # Optional:
exports:                         # List of export sections
...
undefineds:                      # List of undefineds sections
...

Each export section is defined as following:

- archs: [ arm64 ]                   # the list of architecture slices
  allowed-clients: [ client ]        # Optional: List of clients
  re-exports: [ ]                    # Optional: List of re-exports
  symbols: [ _sym ]                  # Optional: List of symbols
  objc-classes: []                   # Optional: List of Objective-C classes
  objc-eh-types: []                  # Optional: List of Objective-C classes
                                     #           with EH
  objc-ivars: []                     # Optional: List of Objective C Instance
                                     #           Variables
  weak-def-symbols: []               # Optional: List of weak defined symbols
  thread-local-symbols: []           # Optional: List of thread local symbols

Each undefineds section is defined as following:
- archs: [ arm64 ]     # the list of architecture slices
  symbols: [ _sym ]    # Optional: List of symbols
  objc-classes: []     # Optional: List of Objective-C classes
  objc-eh-types: []                  # Optional: List of Objective-C classes
                                     #           with EH
  objc-ivars: []       # Optional: List of Objective C Instance Variables
  weak-ref-symbols: [] # Optional: List of weak defined symbols
*/
// clang-format on

using namespace llvm;
using namespace llvm::yaml;
using namespace llvm::MachO;

namespace {
struct ExportSection {
  std::vector<Architecture> Architectures;
  std::vector<FlowStringRef> AllowableClients;
  std::vector<FlowStringRef> ReexportedLibraries;
  std::vector<FlowStringRef> Symbols;
  std::vector<FlowStringRef> Classes;
  std::vector<FlowStringRef> ClassEHs;
  std::vector<FlowStringRef> IVars;
  std::vector<FlowStringRef> WeakDefSymbols;
  std::vector<FlowStringRef> TLVSymbols;
};

struct UndefinedSection {
  std::vector<Architecture> Architectures;
  std::vector<FlowStringRef> Symbols;
  std::vector<FlowStringRef> Classes;
  std::vector<FlowStringRef> ClassEHs;
  std::vector<FlowStringRef> IVars;
  std::vector<FlowStringRef> WeakRefSymbols;
};

// clang-format off
enum TBDFlags : unsigned {
  None                         = 0U,
  FlatNamespace                = 1U << 0,
  NotApplicationExtensionSafe  = 1U << 1,
  InstallAPI                   = 1U << 2,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/InstallAPI),
};
// clang-format on
} // end anonymous namespace.

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(Architecture)
LLVM_YAML_IS_SEQUENCE_VECTOR(ExportSection)
LLVM_YAML_IS_SEQUENCE_VECTOR(UndefinedSection)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<ExportSection> {
  static void mapping(IO &IO, ExportSection &Section) {
    const auto *Ctx = reinterpret_cast<TextAPIContext *>(IO.getContext());
    assert((!Ctx || (Ctx && Ctx->FileKind != FileType::Invalid)) &&
           "File type is not set in YAML context");

    IO.mapRequired("archs", Section.Architectures);
    if (Ctx->FileKind == FileType::TBD_V1)
      IO.mapOptional("allowed-clients", Section.AllowableClients);
    else
      IO.mapOptional("allowable-clients", Section.AllowableClients);
    IO.mapOptional("re-exports", Section.ReexportedLibraries);
    IO.mapOptional("symbols", Section.Symbols);
    IO.mapOptional("objc-classes", Section.Classes);
    if (Ctx->FileKind == FileType::TBD_V3)
      IO.mapOptional("objc-eh-types", Section.ClassEHs);
    IO.mapOptional("objc-ivars", Section.IVars);
    IO.mapOptional("weak-def-symbols", Section.WeakDefSymbols);
    IO.mapOptional("thread-local-symbols", Section.TLVSymbols);
  }
};

template <> struct MappingTraits<UndefinedSection> {
  static void mapping(IO &IO, UndefinedSection &Section) {
    const auto *Ctx = reinterpret_cast<TextAPIContext *>(IO.getContext());
    assert((!Ctx || (Ctx && Ctx->FileKind != FileType::Invalid)) &&
           "File type is not set in YAML context");

    IO.mapRequired("archs", Section.Architectures);
    IO.mapOptional("symbols", Section.Symbols);
    IO.mapOptional("objc-classes", Section.Classes);
    if (Ctx->FileKind == FileType::TBD_V3)
      IO.mapOptional("objc-eh-types", Section.ClassEHs);
    IO.mapOptional("objc-ivars", Section.IVars);
    IO.mapOptional("weak-ref-symbols", Section.WeakRefSymbols);
  }
};

template <> struct ScalarBitSetTraits<TBDFlags> {
  static void bitset(IO &IO, TBDFlags &Flags) {
    IO.bitSetCase(Flags, "flat_namespace", TBDFlags::FlatNamespace);
    IO.bitSetCase(Flags, "not_app_extension_safe",
                  TBDFlags::NotApplicationExtensionSafe);
    IO.bitSetCase(Flags, "installapi", TBDFlags::InstallAPI);
  }
};

template <> struct MappingTraits<const InterfaceFile *> {
  struct NormalizedTBD {
    explicit NormalizedTBD(IO &IO) {}
    NormalizedTBD(IO &IO, const InterfaceFile *&File) {
      Architectures = File->getArchitectures();
      UUIDs = File->uuids();
      Platform = File->getPlatform();
      InstallName = File->getInstallName();
      CurrentVersion = PackedVersion(File->getCurrentVersion());
      CompatibilityVersion = PackedVersion(File->getCompatibilityVersion());
      SwiftABIVersion = File->getSwiftABIVersion();
      ObjCConstraint = File->getObjCConstraint();

      Flags = TBDFlags::None;
      if (!File->isApplicationExtensionSafe())
        Flags |= TBDFlags::NotApplicationExtensionSafe;

      if (!File->isTwoLevelNamespace())
        Flags |= TBDFlags::FlatNamespace;

      if (File->isInstallAPI())
        Flags |= TBDFlags::InstallAPI;

      ParentUmbrella = File->getParentUmbrella();

      std::set<ArchitectureSet> ArchSet;
      for (const auto &Library : File->allowableClients())
        ArchSet.insert(Library.getArchitectures());

      for (const auto &Library : File->reexportedLibraries())
        ArchSet.insert(Library.getArchitectures());

      std::map<const Symbol *, ArchitectureSet> SymbolToArchSet;
      for (const auto *Symbol : File->exports()) {
        auto Architectures = Symbol->getArchitectures();
        SymbolToArchSet[Symbol] = Architectures;
        ArchSet.insert(Architectures);
      }

      for (auto Architectures : ArchSet) {
        ExportSection Section;
        Section.Architectures = Architectures;

        for (const auto &Library : File->allowableClients())
          if (Library.getArchitectures() == Architectures)
            Section.AllowableClients.emplace_back(Library.getInstallName());

        for (const auto &Library : File->reexportedLibraries())
          if (Library.getArchitectures() == Architectures)
            Section.ReexportedLibraries.emplace_back(Library.getInstallName());

        for (const auto &SymArch : SymbolToArchSet) {
          if (SymArch.second != Architectures)
            continue;

          const auto *Symbol = SymArch.first;
          switch (Symbol->getKind()) {
          case SymbolKind::GlobalSymbol:
            if (Symbol->isWeakDefined())
              Section.WeakDefSymbols.emplace_back(Symbol->getName());
            else if (Symbol->isThreadLocalValue())
              Section.TLVSymbols.emplace_back(Symbol->getName());
            else
              Section.Symbols.emplace_back(Symbol->getName());
            break;
          case SymbolKind::ObjectiveCClass:
            if (File->getFileType() != FileType::TBD_V3)
              Section.Classes.emplace_back(
                  copyString("_" + Symbol->getName().str()));
            else
              Section.Classes.emplace_back(Symbol->getName());
            break;
          case SymbolKind::ObjectiveCClassEHType:
            if (File->getFileType() != FileType::TBD_V3)
              Section.Symbols.emplace_back(
                  copyString("_OBJC_EHTYPE_$_" + Symbol->getName().str()));
            else
              Section.ClassEHs.emplace_back(Symbol->getName());
            break;
          case SymbolKind::ObjectiveCInstanceVariable:
            if (File->getFileType() != FileType::TBD_V3)
              Section.IVars.emplace_back(
                  copyString("_" + Symbol->getName().str()));
            else
              Section.IVars.emplace_back(Symbol->getName());
            break;
          }
        }
        llvm::sort(Section.Symbols.begin(), Section.Symbols.end());
        llvm::sort(Section.Classes.begin(), Section.Classes.end());
        llvm::sort(Section.ClassEHs.begin(), Section.ClassEHs.end());
        llvm::sort(Section.IVars.begin(), Section.IVars.end());
        llvm::sort(Section.WeakDefSymbols.begin(),
                   Section.WeakDefSymbols.end());
        llvm::sort(Section.TLVSymbols.begin(), Section.TLVSymbols.end());
        Exports.emplace_back(std::move(Section));
      }

      ArchSet.clear();
      SymbolToArchSet.clear();

      for (const auto *Symbol : File->undefineds()) {
        auto Architectures = Symbol->getArchitectures();
        SymbolToArchSet[Symbol] = Architectures;
        ArchSet.insert(Architectures);
      }

      for (auto Architectures : ArchSet) {
        UndefinedSection Section;
        Section.Architectures = Architectures;

        for (const auto &SymArch : SymbolToArchSet) {
          if (SymArch.second != Architectures)
            continue;

          const auto *Symbol = SymArch.first;
          switch (Symbol->getKind()) {
          case SymbolKind::GlobalSymbol:
            if (Symbol->isWeakReferenced())
              Section.WeakRefSymbols.emplace_back(Symbol->getName());
            else
              Section.Symbols.emplace_back(Symbol->getName());
            break;
          case SymbolKind::ObjectiveCClass:
            if (File->getFileType() != FileType::TBD_V3)
              Section.Classes.emplace_back(
                  copyString("_" + Symbol->getName().str()));
            else
              Section.Classes.emplace_back(Symbol->getName());
            break;
          case SymbolKind::ObjectiveCClassEHType:
            if (File->getFileType() != FileType::TBD_V3)
              Section.Symbols.emplace_back(
                  copyString("_OBJC_EHTYPE_$_" + Symbol->getName().str()));
            else
              Section.ClassEHs.emplace_back(Symbol->getName());
            break;
          case SymbolKind::ObjectiveCInstanceVariable:
            if (File->getFileType() != FileType::TBD_V3)
              Section.IVars.emplace_back(
                  copyString("_" + Symbol->getName().str()));
            else
              Section.IVars.emplace_back(Symbol->getName());
            break;
          }
        }
        llvm::sort(Section.Symbols.begin(), Section.Symbols.end());
        llvm::sort(Section.Classes.begin(), Section.Classes.end());
        llvm::sort(Section.ClassEHs.begin(), Section.ClassEHs.end());
        llvm::sort(Section.IVars.begin(), Section.IVars.end());
        llvm::sort(Section.WeakRefSymbols.begin(),
                   Section.WeakRefSymbols.end());
        Undefineds.emplace_back(std::move(Section));
      }
    }

    const InterfaceFile *denormalize(IO &IO) {
      auto Ctx = reinterpret_cast<TextAPIContext *>(IO.getContext());
      assert(Ctx);

      auto *File = new InterfaceFile;
      File->setPath(Ctx->Path);
      File->setFileType(Ctx->FileKind);
      for (auto &ID : UUIDs)
        File->addUUID(ID.first, ID.second);
      File->setPlatform(Platform);
      File->setArchitectures(Architectures);
      File->setInstallName(InstallName);
      File->setCurrentVersion(CurrentVersion);
      File->setCompatibilityVersion(CompatibilityVersion);
      File->setSwiftABIVersion(SwiftABIVersion);
      File->setObjCConstraint(ObjCConstraint);
      File->setParentUmbrella(ParentUmbrella);

      if (Ctx->FileKind == FileType::TBD_V1) {
        File->setTwoLevelNamespace();
        File->setApplicationExtensionSafe();
      } else {
        File->setTwoLevelNamespace(!(Flags & TBDFlags::FlatNamespace));
        File->setApplicationExtensionSafe(
            !(Flags & TBDFlags::NotApplicationExtensionSafe));
        File->setInstallAPI(Flags & TBDFlags::InstallAPI);
      }

      for (const auto &Section : Exports) {
        for (const auto &Library : Section.AllowableClients)
          File->addAllowableClient(Library, Section.Architectures);
        for (const auto &Library : Section.ReexportedLibraries)
          File->addReexportedLibrary(Library, Section.Architectures);

        for (const auto &Symbol : Section.Symbols) {
          if (Ctx->FileKind != FileType::TBD_V3 &&
              Symbol.value.startswith("_OBJC_EHTYPE_$_"))
            File->addSymbol(SymbolKind::ObjectiveCClassEHType,
                            Symbol.value.drop_front(15), Section.Architectures);
          else
            File->addSymbol(SymbolKind::GlobalSymbol, Symbol,
                            Section.Architectures);
        }
        for (auto &Symbol : Section.Classes) {
          auto Name = Symbol.value;
          if (Ctx->FileKind != FileType::TBD_V3)
            Name = Name.drop_front();
          File->addSymbol(SymbolKind::ObjectiveCClass, Name,
                          Section.Architectures);
        }
        for (auto &Symbol : Section.ClassEHs)
          File->addSymbol(SymbolKind::ObjectiveCClassEHType, Symbol,
                          Section.Architectures);
        for (auto &Symbol : Section.IVars) {
          auto Name = Symbol.value;
          if (Ctx->FileKind != FileType::TBD_V3)
            Name = Name.drop_front();
          File->addSymbol(SymbolKind::ObjectiveCInstanceVariable, Name,
                          Section.Architectures);
        }
        for (auto &Symbol : Section.WeakDefSymbols)
          File->addSymbol(SymbolKind::GlobalSymbol, Symbol,
                          Section.Architectures, SymbolFlags::WeakDefined);
        for (auto &Symbol : Section.TLVSymbols)
          File->addSymbol(SymbolKind::GlobalSymbol, Symbol,
                          Section.Architectures, SymbolFlags::ThreadLocalValue);
      }

      for (const auto &Section : Undefineds) {
        for (auto &Symbol : Section.Symbols) {
          if (Ctx->FileKind != FileType::TBD_V3 &&
              Symbol.value.startswith("_OBJC_EHTYPE_$_"))
            File->addSymbol(SymbolKind::ObjectiveCClassEHType,
                            Symbol.value.drop_front(15), Section.Architectures,
                            SymbolFlags::Undefined);
          else
            File->addSymbol(SymbolKind::GlobalSymbol, Symbol,
                            Section.Architectures, SymbolFlags::Undefined);
        }
        for (auto &Symbol : Section.Classes) {
          auto Name = Symbol.value;
          if (Ctx->FileKind != FileType::TBD_V3)
            Name = Name.drop_front();
          File->addSymbol(SymbolKind::ObjectiveCClass, Name,
                          Section.Architectures, SymbolFlags::Undefined);
        }
        for (auto &Symbol : Section.ClassEHs)
          File->addSymbol(SymbolKind::ObjectiveCClassEHType, Symbol,
                          Section.Architectures, SymbolFlags::Undefined);
        for (auto &Symbol : Section.IVars) {
          auto Name = Symbol.value;
          if (Ctx->FileKind != FileType::TBD_V3)
            Name = Name.drop_front();
          File->addSymbol(SymbolKind::ObjectiveCInstanceVariable, Name,
                          Section.Architectures, SymbolFlags::Undefined);
        }
        for (auto &Symbol : Section.WeakRefSymbols)
          File->addSymbol(SymbolKind::GlobalSymbol, Symbol,
                          Section.Architectures,
                          SymbolFlags::Undefined | SymbolFlags::WeakReferenced);
      }

      return File;
    }

    llvm::BumpPtrAllocator Allocator;
    StringRef copyString(StringRef String) {
      if (String.empty())
        return {};

      void *Ptr = Allocator.Allocate(String.size(), 1);
      memcpy(Ptr, String.data(), String.size());
      return StringRef(reinterpret_cast<const char *>(Ptr), String.size());
    }

    std::vector<Architecture> Architectures;
    std::vector<UUID> UUIDs;
    PlatformKind Platform{PlatformKind::unknown};
    StringRef InstallName;
    PackedVersion CurrentVersion;
    PackedVersion CompatibilityVersion;
    SwiftVersion SwiftABIVersion{0};
    ObjCConstraintType ObjCConstraint{ObjCConstraintType::None};
    TBDFlags Flags{TBDFlags::None};
    StringRef ParentUmbrella;
    std::vector<ExportSection> Exports;
    std::vector<UndefinedSection> Undefineds;
  };

  static void mapping(IO &IO, const InterfaceFile *&File) {
    auto *Ctx = reinterpret_cast<TextAPIContext *>(IO.getContext());
    assert((!Ctx || !IO.outputting() ||
            (Ctx && Ctx->FileKind != FileType::Invalid)) &&
           "File type is not set in YAML context");
    MappingNormalization<NormalizedTBD, const InterfaceFile *> Keys(IO, File);

    // prope file type when reading.
    if (!IO.outputting()) {
      if (IO.mapTag("!tapi-tbd-v2", false))
        Ctx->FileKind = FileType::TBD_V2;
      else if (IO.mapTag("!tapi-tbd-v3", false))
        Ctx->FileKind = FileType::TBD_V2;
      else if (IO.mapTag("!tapi-tbd-v1", false) ||
               IO.mapTag("tag:yaml.org,2002:map", false))
        Ctx->FileKind = FileType::TBD_V1;
      else {
        IO.setError("unsupported file type");
        return;
      }
    }

    // Set file tyoe when writing.
    if (IO.outputting()) {
      switch (Ctx->FileKind) {
      default:
        llvm_unreachable("unexpected file type");
      case FileType::TBD_V1:
        // Don't write the tag into the .tbd file for TBD v1.
        break;
      case FileType::TBD_V2:
        IO.mapTag("!tapi-tbd-v2", true);
        break;
      case FileType::TBD_V3:
        IO.mapTag("!tapi-tbd-v3", true);
        break;
      }
    }

    IO.mapRequired("archs", Keys->Architectures);
    if (Ctx->FileKind != FileType::TBD_V1)
      IO.mapOptional("uuids", Keys->UUIDs);
    IO.mapRequired("platform", Keys->Platform);
    if (Ctx->FileKind != FileType::TBD_V1)
      IO.mapOptional("flags", Keys->Flags, TBDFlags::None);
    IO.mapRequired("install-name", Keys->InstallName);
    IO.mapOptional("current-version", Keys->CurrentVersion,
                   PackedVersion(1, 0, 0));
    IO.mapOptional("compatibility-version", Keys->CompatibilityVersion,
                   PackedVersion(1, 0, 0));
    if (Ctx->FileKind != FileType::TBD_V3)
      IO.mapOptional("swift-version", Keys->SwiftABIVersion, SwiftVersion(0));
    else
      IO.mapOptional("swift-abi-version", Keys->SwiftABIVersion,
                     SwiftVersion(0));
    IO.mapOptional("objc-constraint", Keys->ObjCConstraint,
                   (Ctx->FileKind == FileType::TBD_V1)
                       ? ObjCConstraintType::None
                       : ObjCConstraintType::Retain_Release);
    if (Ctx->FileKind != FileType::TBD_V1)
      IO.mapOptional("parent-umbrella", Keys->ParentUmbrella, StringRef());
    IO.mapOptional("exports", Keys->Exports);
    if (Ctx->FileKind != FileType::TBD_V1)
      IO.mapOptional("undefineds", Keys->Undefineds);
  }
};

template <>
struct DocumentListTraits<std::vector<const MachO::InterfaceFile *>> {
  static size_t size(IO &IO, std::vector<const MachO::InterfaceFile *> &Seq) {
    return Seq.size();
  }
  static const InterfaceFile *&
  element(IO &IO, std::vector<const InterfaceFile *> &Seq, size_t Index) {
    if (Index >= Seq.size())
      Seq.resize(Index + 1);
    return Seq[Index];
  }
};

} // end namespace yaml.

namespace MachO {
static void DiagHandler(const SMDiagnostic &Diag, void *Context) {
  auto *File = static_cast<TextAPIContext *>(Context);
  SmallString<1024> Message;
  raw_svector_ostream S(Message);

  SMDiagnostic NewDiag(*Diag.getSourceMgr(), Diag.getLoc(), File->Path,
                       Diag.getLineNo(), Diag.getColumnNo(), Diag.getKind(),
                       Diag.getMessage(), Diag.getLineContents(),
                       Diag.getRanges(), Diag.getFixIts());

  NewDiag.print(nullptr, S);
  File->ErrorMessage = ("malformed file\n" + Message).str();
}

Expected<std::unique_ptr<InterfaceFile>>
TextAPIReader::get(std::unique_ptr<MemoryBuffer> InputBuffer) {
  TextAPIContext Ctx;
  Ctx.Path = InputBuffer->getBufferIdentifier();
  yaml::Input YAMLIn(InputBuffer->getBuffer(), &Ctx, DiagHandler, &Ctx);

  // Fill vector with interface file objects created by parsing the YAML file.
  std::vector<const InterfaceFile *> Files;
  YAMLIn >> Files;

  auto File = std::unique_ptr<InterfaceFile>(
      const_cast<InterfaceFile *>(Files.front()));

  if (YAMLIn.error())
    return make_error<StringError>(Ctx.ErrorMessage, YAMLIn.error());

  return std::move(File);
}

Error TextAPIWriter::writeToStream(raw_ostream &OS, const InterfaceFile &File) {
  TextAPIContext Ctx;
  Ctx.Path = File.getPath();
  Ctx.FileKind = File.getFileType();
  llvm::yaml::Output YAMLOut(OS, &Ctx, /*WrapColumn=*/80);

  std::vector<const InterfaceFile *> Files;
  Files.emplace_back(&File);

  // Stream out yaml.
  YAMLOut << Files;

  return Error::success();
}

} // end namespace MachO.
} // end namespace llvm.
