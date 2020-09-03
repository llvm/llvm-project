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

#include "llvm/TextAPI/MachO/InterfaceFile.h"
#include <iomanip>
#include <sstream>

using namespace llvm;
using namespace llvm::MachO;

namespace {
template <typename C>
typename C::iterator addEntry(C &Container, StringRef InstallName) {
  auto I = partition_point(Container, [=](const InterfaceFileRef &O) {
    return O.getInstallName() < InstallName;
  });
  if (I != Container.end() && I->getInstallName() == InstallName)
    return I;

  return Container.emplace(I, InstallName);
}

template <typename C>
typename C::iterator addEntry(C &Container, const Target &Target_) {
  auto Iter =
      lower_bound(Container, Target_, [](const Target &LHS, const Target &RHS) {
        return LHS < RHS;
      });
  if ((Iter != std::end(Container)) && !(Target_ < *Iter))
    return Iter;

  return Container.insert(Iter, Target_);
}
} // end namespace

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
  return;
}

void InterfaceFile::addUUID(const Target &Target_, StringRef UUID) {
  auto Iter = lower_bound(UUIDs, Target_,
                          [](const std::pair<Target, std::string> &LHS,
                             Target RHS) { return LHS.first < RHS; });

  if ((Iter != UUIDs.end()) && !(Target_ < Iter->first)) {
    Iter->second = std::string(UUID);
    return;
  }

  UUIDs.emplace(Iter, Target_, std::string(UUID));
  return;
}

void InterfaceFile::addUUID(const Target &Target, uint8_t UUID[16]) {
  std::stringstream Stream;
  for (unsigned i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10)
      Stream << '-';
    Stream << std::setfill('0') << std::setw(2) << std::uppercase << std::hex
           << static_cast<int>(UUID[i]);
  }
  addUUID(Target, Stream.str());
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

void InterfaceFile::addSymbol(SymbolKind Kind, StringRef Name,
                              const TargetList &Targets, SymbolFlags Flags) {
  Name = copyString(Name);
  auto result = Symbols.try_emplace(SymbolsMapKey{Kind, Name}, nullptr);
  if (result.second)
    result.first->second = new (Allocator) Symbol{Kind, Name, Targets, Flags};
  else
    for (const auto &Target : Targets)
      result.first->second->addTarget(Target);
}
