//===- TextStubV5.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements Text Stub JSON mappings.
//
//===----------------------------------------------------------------------===//
#include "TextStubCommon.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"

// clang-format off
/*

JSON Format specification.

All library level keys, accept target values and are defaulted if not specified. 

{
"tapi_tbd_version": 5,                            # Required: TBD version for all documents in file
"main_library": {                                 # Required: top level library
  "target_info": [                                # Required: target information 
    {
      "target": "x86_64-macos",
      "min_deployment": "10.14"                   # Required: minimum OS deployment version
    },
    {
      "target": "arm64-macos",
      "min_deployment": "10.14"
    },
    {
      "target": "arm64-maccatalyst",
      "min_deployment": "12.1"
    }],
  "flags":[{"attributes": ["flat_namespace"]}],     # Optional:
  "install_names":[{"name":"/S/L/F/Foo.fwk/Foo"}],  # Required: library install name 
  "current_versions":[{"version": "1.2"}],          # Optional: defaults to 1
  "compatibility_versions":[{ "version": "1.1"}],   # Optional: defaults to 1
  "rpaths": [                                       # Optional: 
    {
      "targets": ["x86_64-macos"],                  # Optional: defaults to targets in `target-info`
      "paths": ["@executable_path/.../Frameworks"]
    }],
  "parent_umbrellas": [{"umbrella": "System"}],
  "allowable_clients": [{"clients": ["ClientA"]}],
  "reexported_libraries": [{"names": ["/u/l/l/foo.dylib"]}],
  "exported_symbols": [{                            # List of export symbols section
      "targets": ["x86_64-macos", "arm64-macos"],   # Optional: defaults to targets in `target-info`
        "text": {                                   # List of Text segment symbols 
          "global": [ "_func" ],
          "weak": [],
          "thread_local": []
        },
        "data": { ... },                            # List of Data segment symbols
   }],
  "reexported_symbols": [{  ... }],                 # List of reexported symbols section
  "undefined_symbols": [{ ... }]                    # List of undefined symbols section
},
"libraries": [                                      # Optional: Array of inlined libraries
  {...}, {...}, {...}
]
}
*/
// clang-format on

using namespace llvm;
using namespace llvm::json;
using namespace llvm::MachO;

struct JSONSymbol {
  SymbolKind Kind;
  std::string Name;
  SymbolFlags Flags;
};

using AttrToTargets = std::map<std::string, TargetList>;
using TargetsToSymbols =
    SmallVector<std::pair<TargetList, std::vector<JSONSymbol>>>;

enum TBDKey : size_t {
  TBDVersion = 0U,
  MainLibrary,
  Documents,
  TargetInfo,
  Targets,
  Target,
  Deployment,
  Flags,
  Attributes,
  InstallName,
  CurrentVersion,
  CompatibilityVersion,
  Version,
  SwiftABI,
  ABI,
  ParentUmbrella,
  Umbrella,
  AllowableClients,
  Clients,
  ReexportLibs,
  Names,
  Name,
  Exports,
  Reexports,
  Undefineds,
  Data,
  Text,
  Weak,
  ThreadLocal,
  Globals,
  ObjCClass,
  ObjCEHType,
  ObjCIvar,
};

std::array<StringRef, 64> Keys = {
    "tapi_tbd_version",
    "main_library",
    "libraries",
    "target_info",
    "targets",
    "target",
    "min_deployment",
    "flags",
    "attributes",
    "install_names",
    "current_versions",
    "compatibility_versions",
    "version",
    "swift_abi",
    "abi",
    "parent_umbrellas",
    "umbrella",
    "allowable_clients",
    "clients",
    "reexported_libraries",
    "names",
    "name",
    "exported_symbols",
    "reexported_symbols",
    "undefined_symbols",
    "data",
    "text",
    "weak",
    "thread_local",
    "global",
    "objc_class",
    "objc_eh_type",
    "objc_ivar",
};

static llvm::SmallString<128> getParseErrorMsg(TBDKey Key) {
  return {"invalid ", Keys[Key], " section"};
}

class JSONStubError : public llvm::ErrorInfo<llvm::json::ParseError> {
public:
  JSONStubError(Twine ErrMsg) : Message(ErrMsg.str()) {}

  void log(llvm::raw_ostream &OS) const override { OS << Message << "\n"; }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

private:
  std::string Message;
};

template <typename JsonT, typename StubT = JsonT>
Expected<StubT> getRequiredValue(
    TBDKey Key, const Object *Obj,
    std::function<std::optional<JsonT>(const Object *, StringRef)> GetValue,
    std::function<std::optional<StubT>(JsonT)> Validate = nullptr) {
  std::optional<JsonT> Val = GetValue(Obj, Keys[Key]);
  if (!Val)
    return make_error<JSONStubError>(getParseErrorMsg(Key));

  if (Validate == nullptr)
    return static_cast<StubT>(*Val);

  std::optional<StubT> Result = Validate(*Val);
  if (!Result.has_value())
    return make_error<JSONStubError>(getParseErrorMsg(Key));
  return Result.value();
}

template <typename JsonT, typename StubT = JsonT>
Expected<StubT> getRequiredValue(
    TBDKey Key, const Object *Obj,
    std::function<std::optional<JsonT>(const Object *, StringRef)> GetValue,
    StubT DefaultValue, std::function<std::optional<StubT>(JsonT)> Validate) {
  std::optional<JsonT> Val = GetValue(Obj, Keys[Key]);
  if (!Val)
    return DefaultValue;

  std::optional<StubT> Result;
  Result = Validate(*Val);
  if (!Result.has_value())
    return make_error<JSONStubError>(getParseErrorMsg(Key));
  return Result.value();
}

Error collectFromArray(TBDKey Key, const Object *Obj,
                       std::function<void(StringRef)> Append,
                       bool IsRequired = false) {
  const auto *Values = Obj->getArray(Keys[Key]);
  if (!Values) {
    if (IsRequired)
      return make_error<JSONStubError>(getParseErrorMsg(Key));
    return Error::success();
  }

  for (Value Val : *Values) {
    auto ValStr = Val.getAsString();
    if (!ValStr.has_value())
      return make_error<JSONStubError>(getParseErrorMsg(Key));
    Append(ValStr.value());
  }

  return Error::success();
}

namespace StubParser {

Expected<FileType> getVersion(const Object *File) {
  auto VersionOrErr = getRequiredValue<int64_t, FileType>(
      TBDKey::TBDVersion, File, &Object::getInteger,
      [](int64_t Val) -> std::optional<FileType> {
        unsigned Result = Val;
        if (Result != 5)
          return std::nullopt;
        return FileType::TBD_V5;
      });

  if (!VersionOrErr)
    return VersionOrErr.takeError();
  return *VersionOrErr;
}

Expected<TargetList> getTargets(const Object *Section) {
  const auto *Targets = Section->getArray(Keys[TBDKey::Targets]);
  if (!Targets)
    return make_error<JSONStubError>(getParseErrorMsg(TBDKey::Targets));

  TargetList IFTargets;
  for (Value JSONTarget : *Targets) {
    auto TargetStr = JSONTarget.getAsString();
    if (!TargetStr.has_value())
      return make_error<JSONStubError>(getParseErrorMsg(TBDKey::Target));
    auto TargetOrErr = Target::create(TargetStr.value());
    if (!TargetOrErr)
      return make_error<JSONStubError>(getParseErrorMsg(TBDKey::Target));
    IFTargets.push_back(*TargetOrErr);
  }
  return std::move(IFTargets);
}

Expected<TargetList> getTargetsSection(const Object *Section) {
  const Array *Targets = Section->getArray(Keys[TBDKey::TargetInfo]);
  if (!Targets)
    return make_error<JSONStubError>(getParseErrorMsg(TBDKey::Targets));

  TargetList IFTargets;
  for (const Value &JSONTarget : *Targets) {
    const auto *Obj = JSONTarget.getAsObject();
    if (!Obj)
      return make_error<JSONStubError>(getParseErrorMsg(TBDKey::Target));
    auto TargetStr =
        getRequiredValue<StringRef>(TBDKey::Target, Obj, &Object::getString);
    if (!TargetStr)
      return make_error<JSONStubError>(getParseErrorMsg(TBDKey::Target));
    auto TargetOrErr = Target::create(*TargetStr);
    if (!TargetOrErr)
      return make_error<JSONStubError>(getParseErrorMsg(TBDKey::Target));
    IFTargets.push_back(*TargetOrErr);
    // TODO: Implement Deployment Version.
  }
  return std::move(IFTargets);
}

Error collectSymbolsFromSegment(const Object *Segment, TargetsToSymbols &Result,
                                SymbolFlags SectionFlag) {
  auto Err = collectFromArray(
      TBDKey::Globals, Segment, [&Result, &SectionFlag](StringRef Name) {
        JSONSymbol Sym = {SymbolKind::GlobalSymbol, Name.str(), SectionFlag};
        Result.back().second.emplace_back(Sym);
      });
  if (Err)
    return Err;

  Err = collectFromArray(
      TBDKey::ObjCClass, Segment, [&Result, &SectionFlag](StringRef Name) {
        JSONSymbol Sym = {SymbolKind::ObjectiveCClass, Name.str(), SectionFlag};
        Result.back().second.emplace_back(Sym);
      });
  if (Err)
    return Err;

  Err = collectFromArray(TBDKey::ObjCEHType, Segment,
                         [&Result, &SectionFlag](StringRef Name) {
                           JSONSymbol Sym = {SymbolKind::ObjectiveCClassEHType,
                                             Name.str(), SectionFlag};
                           Result.back().second.emplace_back(Sym);
                         });
  if (Err)
    return Err;

  Err = collectFromArray(
      TBDKey::ObjCIvar, Segment, [&Result, &SectionFlag](StringRef Name) {
        JSONSymbol Sym = {SymbolKind::ObjectiveCInstanceVariable, Name.str(),
                          SectionFlag};
        Result.back().second.emplace_back(Sym);
      });
  if (Err)
    return Err;

  SymbolFlags WeakFlag = SectionFlag | (SectionFlag == SymbolFlags::Undefined
                                            ? SymbolFlags::WeakReferenced
                                            : SymbolFlags::WeakDefined);
  Err = collectFromArray(TBDKey::Weak, Segment,
                         [&Result, WeakFlag](StringRef Name) {
                           JSONSymbol Sym = {
                               SymbolKind::GlobalSymbol,
                               Name.str(),
                               WeakFlag,
                           };
                           Result.back().second.emplace_back(Sym);
                         });
  if (Err)
    return Err;

  Err = collectFromArray(
      TBDKey::ThreadLocal, Segment, [&Result, SectionFlag](StringRef Name) {
        JSONSymbol Sym = {SymbolKind::GlobalSymbol, Name.str(),
                          SymbolFlags::ThreadLocalValue | SectionFlag};
        Result.back().second.emplace_back(Sym);
      });
  if (Err)
    return Err;

  return Error::success();
}

Expected<StringRef> getNameSection(const Object *File) {
  const Array *Section = File->getArray(Keys[TBDKey::InstallName]);
  if (!Section)
    return make_error<JSONStubError>(getParseErrorMsg(TBDKey::InstallName));

  assert(!Section->empty() && "unexpected missing install name");
  // TODO: Just take first for now.
  const auto *Obj = Section->front().getAsObject();
  if (!Obj)
    return make_error<JSONStubError>(getParseErrorMsg(TBDKey::InstallName));

  return getRequiredValue<StringRef>(TBDKey::Name, Obj, &Object::getString);
}

Expected<TargetsToSymbols> getSymbolSection(const Object *File, TBDKey Key,
                                            TargetList &Targets) {

  const Array *Section = File->getArray(Keys[Key]);
  if (!Section)
    return TargetsToSymbols();

  SymbolFlags SectionFlag;
  switch (Key) {
  case TBDKey::Reexports:
    SectionFlag = SymbolFlags::Rexported;
    break;
  case TBDKey::Undefineds:
    SectionFlag = SymbolFlags::Undefined;
    break;
  default:
    SectionFlag = SymbolFlags::None;
    break;
  };

  TargetsToSymbols Result;
  TargetList MappedTargets;
  for (auto Val : *Section) {
    auto *Obj = Val.getAsObject();
    if (!Obj)
      continue;

    auto TargetsOrErr = getTargets(Obj);
    if (!TargetsOrErr) {
      MappedTargets = Targets;
      consumeError(TargetsOrErr.takeError());
    } else {
      MappedTargets = *TargetsOrErr;
    }
    Result.emplace_back(std::make_pair(Targets, std::vector<JSONSymbol>()));

    auto *DataSection = Obj->getObject(Keys[TBDKey::Data]);
    auto *TextSection = Obj->getObject(Keys[TBDKey::Text]);
    // There should be at least one valid section.
    if (!DataSection && !TextSection)
      return make_error<JSONStubError>(getParseErrorMsg(Key));

    if (DataSection) {
      auto Err = collectSymbolsFromSegment(DataSection, Result, SectionFlag);
      if (Err)
        return std::move(Err);
    }
    if (TextSection) {
      auto Err = collectSymbolsFromSegment(TextSection, Result, SectionFlag);
      if (Err)
        return std::move(Err);
    }
  }

  return std::move(Result);
}

Expected<AttrToTargets> getLibSection(const Object *File, TBDKey Key,
                                      TBDKey SubKey,
                                      const TargetList &Targets) {
  auto *Section = File->getArray(Keys[Key]);
  if (!Section)
    return AttrToTargets();

  AttrToTargets Result;
  TargetList MappedTargets;
  for (auto Val : *Section) {
    auto *Obj = Val.getAsObject();
    if (!Obj)
      continue;

    auto TargetsOrErr = getTargets(Obj);
    if (!TargetsOrErr) {
      MappedTargets = Targets;
      consumeError(TargetsOrErr.takeError());
    } else {
      MappedTargets = *TargetsOrErr;
    }
    auto Err =
        collectFromArray(SubKey, Obj, [&Result, &MappedTargets](StringRef Key) {
          Result[Key.str()] = MappedTargets;
        });
    if (Err)
      return std::move(Err);
  }

  return std::move(Result);
}

Expected<AttrToTargets> getUmbrellaSection(const Object *File,
                                           const TargetList &Targets) {
  const auto *Umbrella = File->getArray(Keys[TBDKey::ParentUmbrella]);
  if (!Umbrella)
    return AttrToTargets();

  AttrToTargets Result;
  TargetList MappedTargets;
  for (auto Val : *Umbrella) {
    auto *Obj = Val.getAsObject();
    if (!Obj)
      return make_error<JSONStubError>(
          getParseErrorMsg(TBDKey::ParentUmbrella));

    // Get Targets section.
    auto TargetsOrErr = getTargets(Obj);
    if (!TargetsOrErr) {
      MappedTargets = Targets;
      consumeError(TargetsOrErr.takeError());
    } else {
      MappedTargets = *TargetsOrErr;
    }

    auto UmbrellaOrErr =
        getRequiredValue<StringRef>(TBDKey::Umbrella, Obj, &Object::getString);
    if (!UmbrellaOrErr)
      return UmbrellaOrErr.takeError();
    Result[UmbrellaOrErr->str()] = Targets;
  }
  return std::move(Result);
}

Expected<uint8_t> getSwiftVersion(const Object *File) {
  const Array *Versions = File->getArray(Keys[TBDKey::SwiftABI]);
  if (!Versions)
    return 0;

  for (const auto &Val : *Versions) {
    const auto *Obj = Val.getAsObject();
    if (!Obj)
      return make_error<JSONStubError>(getParseErrorMsg(TBDKey::SwiftABI));

    // TODO: Take first for now.
    return getRequiredValue<int64_t, uint8_t>(TBDKey::ABI, Obj,
                                              &Object::getInteger);
  }

  return 0;
}

Expected<PackedVersion> getPackedVersion(const Object *File, TBDKey Key) {
  const Array *Versions = File->getArray(Keys[Key]);
  if (!Versions)
    return PackedVersion(1, 0, 0);

  for (const auto &Val : *Versions) {
    const auto *Obj = Val.getAsObject();
    if (!Obj)
      return make_error<JSONStubError>(getParseErrorMsg(Key));

    auto ValidatePV = [](StringRef Version) -> std::optional<PackedVersion> {
      PackedVersion PV;
      auto [success, truncated] = PV.parse64(Version);
      if (!success || truncated)
        return std::nullopt;
      return PV;
    };
    // TODO: Take first for now.
    return getRequiredValue<StringRef, PackedVersion>(
        TBDKey::Version, Obj, &Object::getString, PackedVersion(1, 0, 0),
        ValidatePV);
  }

  return PackedVersion(1, 0, 0);
}

Expected<TBDFlags> getFlags(const Object *File) {
  TBDFlags Flags = TBDFlags::None;
  const Array *Section = File->getArray(Keys[TBDKey::Flags]);
  if (!Section)
    return Flags;

  for (auto &Val : *Section) {
    // TODO: Just take first for now.
    const auto *Obj = Val.getAsObject();
    if (!Obj)
      return make_error<JSONStubError>(getParseErrorMsg(TBDKey::Flags));

    auto FlagsOrErr =
        collectFromArray(TBDKey::Attributes, Obj, [&Flags](StringRef Flag) {
          TBDFlags TBDFlag =
              StringSwitch<TBDFlags>(Flag)
                  .Case("flat_namespace", TBDFlags::FlatNamespace)
                  .Case("not_app_extension_safe",
                        TBDFlags::NotApplicationExtensionSafe)
                  .Default(TBDFlags::None);
          Flags |= TBDFlag;
        });

    if (FlagsOrErr)
      return std::move(FlagsOrErr);

    return Flags;
  }

  return Flags;
}

using IFPtr = std::unique_ptr<InterfaceFile>;
Expected<IFPtr> parseToInterfaceFile(const Object *File) {
  auto TargetsOrErr = getTargetsSection(File);
  if (!TargetsOrErr)
    return TargetsOrErr.takeError();
  TargetList Targets = *TargetsOrErr;

  auto NameOrErr = getNameSection(File);
  if (!NameOrErr)
    return NameOrErr.takeError();
  StringRef Name = *NameOrErr;

  auto CurrVersionOrErr = getPackedVersion(File, TBDKey::CurrentVersion);
  if (!CurrVersionOrErr)
    return CurrVersionOrErr.takeError();
  PackedVersion CurrVersion = *CurrVersionOrErr;

  auto CompVersionOrErr = getPackedVersion(File, TBDKey::CompatibilityVersion);
  if (!CompVersionOrErr)
    return CompVersionOrErr.takeError();
  PackedVersion CompVersion = *CompVersionOrErr;

  auto SwiftABIOrErr = getSwiftVersion(File);
  if (!SwiftABIOrErr)
    return SwiftABIOrErr.takeError();
  uint8_t SwiftABI = *SwiftABIOrErr;

  auto FlagsOrErr = getFlags(File);
  if (!FlagsOrErr)
    return FlagsOrErr.takeError();
  TBDFlags Flags = *FlagsOrErr;

  auto UmbrellasOrErr = getUmbrellaSection(File, Targets);
  if (!UmbrellasOrErr)
    return UmbrellasOrErr.takeError();
  AttrToTargets Umbrellas = *UmbrellasOrErr;

  auto ClientsOrErr =
      getLibSection(File, TBDKey::AllowableClients, TBDKey::Clients, Targets);
  if (!ClientsOrErr)
    return ClientsOrErr.takeError();
  AttrToTargets Clients = *ClientsOrErr;

  auto RLOrErr =
      getLibSection(File, TBDKey::ReexportLibs, TBDKey::Names, Targets);
  if (!RLOrErr)
    return RLOrErr.takeError();
  AttrToTargets ReexportLibs = std::move(*RLOrErr);

  auto ExportsOrErr = getSymbolSection(File, TBDKey::Exports, Targets);
  if (!ExportsOrErr)
    return ExportsOrErr.takeError();
  TargetsToSymbols Exports = std::move(*ExportsOrErr);

  auto ReexportsOrErr = getSymbolSection(File, TBDKey::Reexports, Targets);
  if (!ReexportsOrErr)
    return ReexportsOrErr.takeError();
  TargetsToSymbols Reexports = std::move(*ReexportsOrErr);

  auto UndefinedsOrErr = getSymbolSection(File, TBDKey::Undefineds, Targets);
  if (!UndefinedsOrErr)
    return UndefinedsOrErr.takeError();
  TargetsToSymbols Undefineds = std::move(*UndefinedsOrErr);

  IFPtr F(new InterfaceFile);
  F->setInstallName(Name);
  F->setCurrentVersion(CurrVersion);
  F->setCompatibilityVersion(CompVersion);
  F->setSwiftABIVersion(SwiftABI);
  F->setTwoLevelNamespace(!(Flags & TBDFlags::FlatNamespace));
  F->setApplicationExtensionSafe(
      !(Flags & TBDFlags::NotApplicationExtensionSafe));
  for (auto &T : Targets)
    F->addTarget(T);
  for (auto &[Lib, Targets] : Clients)
    for (auto Target : Targets)
      F->addAllowableClient(Lib, Target);
  for (auto &[Lib, Targets] : ReexportLibs)
    for (auto Target : Targets)
      F->addReexportedLibrary(Lib, Target);
  for (auto &[Lib, Targets] : Umbrellas)
    for (auto Target : Targets)
      F->addParentUmbrella(Target, Lib);
  for (auto &[Targets, Symbols] : Exports)
    for (auto &Sym : Symbols)
      F->addSymbol(Sym.Kind, Sym.Name, Targets, Sym.Flags);
  for (auto &[Targets, Symbols] : Reexports)
    for (auto &Sym : Symbols)
      F->addSymbol(Sym.Kind, Sym.Name, Targets, Sym.Flags);
  for (auto &[Targets, Symbols] : Undefineds)
    for (auto &Sym : Symbols)
      F->addSymbol(Sym.Kind, Sym.Name, Targets, Sym.Flags);

  return std::move(F);
}

Expected<std::vector<IFPtr>> getInlinedLibs(const Object *File) {
  std::vector<IFPtr> IFs;
  const Array *Files = File->getArray(Keys[TBDKey::Documents]);
  if (!Files)
    return std::move(IFs);

  for (auto Lib : *Files) {
    auto IFOrErr = parseToInterfaceFile(Lib.getAsObject());
    if (!IFOrErr)
      return IFOrErr.takeError();
    auto IF = std::move(*IFOrErr);
    IFs.emplace_back(std::move(IF));
  }
  return std::move(IFs);
}

} // namespace StubParser

Expected<std::unique_ptr<InterfaceFile>>
MachO::getInterfaceFileFromJSON(StringRef JSON) {
  auto ValOrErr = parse(JSON);
  if (!ValOrErr)
    return ValOrErr.takeError();

  auto *Root = ValOrErr->getAsObject();
  auto VersionOrErr = StubParser::getVersion(Root);
  if (!VersionOrErr)
    return VersionOrErr.takeError();
  FileType Version = *VersionOrErr;

  Object *MainLib = Root->getObject(Keys[TBDKey::MainLibrary]);
  auto IFOrErr = StubParser::parseToInterfaceFile(MainLib);
  if (!IFOrErr)
    return IFOrErr.takeError();
  (*IFOrErr)->setFileType(Version);
  std::unique_ptr<InterfaceFile> IF(std::move(*IFOrErr));

  auto IFsOrErr = StubParser::getInlinedLibs(Root);
  if (!IFsOrErr)
    return IFsOrErr.takeError();
  for (auto &File : *IFsOrErr) {
    File->setFileType(Version);
    IF->addDocument(std::shared_ptr<InterfaceFile>(std::move(File)));
  }
  return std::move(IF);
}
