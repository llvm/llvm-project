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

namespace llvm {
namespace MachO {
namespace detail {
template <typename C>
typename C::iterator addEntry(C &Container, StringRef InstallName) {
  auto I = partition_point(Container, [=](const InterfaceFileRef &O) {
    return O.getInstallName() < InstallName;
  });
  if (I != Container.end() && I->getInstallName() == InstallName)
    return I;

  return Container.emplace(I, InstallName);
}
} // end namespace detail.

void InterfaceFile::addAllowableClient(StringRef Name,
                                       ArchitectureSet Architectures) {
  auto Client = detail::addEntry(AllowableClients, Name);
  Client->addArchitectures(Architectures);
}

void InterfaceFile::addReexportedLibrary(StringRef InstallName,
                                         ArchitectureSet Architectures) {
  auto Lib = detail::addEntry(ReexportedLibraries, InstallName);
  Lib->addArchitectures(Architectures);
}

void InterfaceFile::addUUID(Architecture Arch, StringRef UUID) {
  auto I = partition_point(UUIDs,
                           [=](const std::pair<Architecture, std::string> &O) {
                             return O.first < Arch;
                           });

  if (I != UUIDs.end() && Arch == I->first) {
    I->second = UUID;
    return;
  }

  UUIDs.emplace(I, Arch, UUID);
  return;
}

void InterfaceFile::addUUID(Architecture Arch, uint8_t UUID[16]) {
  std::stringstream Stream;
  for (unsigned i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10)
      Stream << '-';
    Stream << std::setfill('0') << std::setw(2) << std::uppercase << std::hex
           << static_cast<int>(UUID[i]);
  }
  addUUID(Arch, Stream.str());
}

void InterfaceFile::addSymbol(SymbolKind Kind, StringRef Name,
                              ArchitectureSet Archs, SymbolFlags Flags) {
  Name = copyString(Name);
  auto result = Symbols.try_emplace(SymbolsMapKey{Kind, Name}, nullptr);
  if (result.second)
    result.first->second = new (Allocator) Symbol{Kind, Name, Archs, Flags};
  else
    result.first->second->addArchitectures(Archs);
}

} // end namespace MachO.
} // end namespace llvm.
