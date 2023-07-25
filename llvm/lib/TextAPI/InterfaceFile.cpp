//===- InterfaceFile.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Interface File.
//
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/InterfaceFile.h"
#include <iomanip>
#include <sstream>

using namespace llvm;
using namespace llvm::MachO;

void InterfaceFileRef::addTarget(const Target &Target) {
  addEntry(Targets, Target);
}

void InterfaceFile::addAllowableClient(StringRef InstallName,
                                       const Target &Target) {
  auto Client = addEntry(AllowableClients, InstallName);
  Client->addTarget(Target);
}

void InterfaceFile::addReexportedLibrary(StringRef InstallName,
                                         const Target &Target) {
  auto Lib = addEntry(ReexportedLibraries, InstallName);
  Lib->addTarget(Target);
}

void InterfaceFile::addParentUmbrella(const Target &Target_, StringRef Parent) {
  auto Iter = lower_bound(ParentUmbrellas, Target_,
                          [](const std::pair<Target, std::string> &LHS,
                             Target RHS) { return LHS.first < RHS; });

  if ((Iter != ParentUmbrellas.end()) && !(Target_ < Iter->first)) {
    Iter->second = std::string(Parent);
    return;
  }

  ParentUmbrellas.emplace(Iter, Target_, std::string(Parent));
}

void InterfaceFile::addRPath(const Target &InputTarget, StringRef RPath) {
  auto Iter = lower_bound(RPaths, InputTarget,
                          [](const std::pair<Target, std::string> &LHS,
                             Target RHS) { return LHS.first < RHS; });

  if ((Iter != RPaths.end()) && !(InputTarget < Iter->first)) {
    Iter->second = std::string(RPath);
    return;
  }

  RPaths.emplace(Iter, InputTarget, std::string(RPath));
}

void InterfaceFile::addTarget(const Target &Target) {
  addEntry(Targets, Target);
}

InterfaceFile::const_filtered_target_range
InterfaceFile::targets(ArchitectureSet Archs) const {
  std::function<bool(const Target &)> fn = [Archs](const Target &Target_) {
    return Archs.has(Target_.Arch);
  };
  return make_filter_range(Targets, fn);
}

void InterfaceFile::addDocument(std::shared_ptr<InterfaceFile> &&Document) {
  auto Pos = llvm::lower_bound(Documents, Document,
                               [](const std::shared_ptr<InterfaceFile> &LHS,
                                  const std::shared_ptr<InterfaceFile> &RHS) {
                                 return LHS->InstallName < RHS->InstallName;
                               });
  Document->Parent = this;
  Documents.insert(Pos, Document);
}

static bool isYAMLTextStub(const FileType &Kind) {
  return (Kind >= FileType::TBD_V1) && (Kind < FileType::TBD_V5);
}

bool InterfaceFile::operator==(const InterfaceFile &O) const {
  if (Targets != O.Targets)
    return false;
  if (InstallName != O.InstallName)
    return false;
  if ((CurrentVersion != O.CurrentVersion) ||
      (CompatibilityVersion != O.CompatibilityVersion))
    return false;
  if (SwiftABIVersion != O.SwiftABIVersion)
    return false;
  if (IsTwoLevelNamespace != O.IsTwoLevelNamespace)
    return false;
  if (IsAppExtensionSafe != O.IsAppExtensionSafe)
    return false;
  if (ParentUmbrellas != O.ParentUmbrellas)
    return false;
  if (AllowableClients != O.AllowableClients)
    return false;
  if (ReexportedLibraries != O.ReexportedLibraries)
    return false;
  if (*SymbolsSet != *O.SymbolsSet)
    return false;
  // Don't compare run search paths for older filetypes that cannot express
  // them.
  if (!(isYAMLTextStub(FileKind)) && !(isYAMLTextStub(O.FileKind))) {
    if (RPaths != O.RPaths)
      return false;
    if (mapToPlatformVersionSet(Targets) != mapToPlatformVersionSet(O.Targets))
      return false;
  }

  if (!std::equal(Documents.begin(), Documents.end(), O.Documents.begin(),
                  O.Documents.end(),
                  [](const std::shared_ptr<InterfaceFile> LHS,
                     const std::shared_ptr<InterfaceFile> RHS) {
                    return *LHS == *RHS;
                  }))
    return false;
  return true;
}
