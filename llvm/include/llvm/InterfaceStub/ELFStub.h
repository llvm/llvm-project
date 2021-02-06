//===- ELFStub.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/
///
/// \file
/// This file defines an internal representation of an ELF stub.
///
//===-----------------------------------------------------------------------===/

#ifndef LLVM_INTERFACESTUB_ELFSTUB_H
#define LLVM_INTERFACESTUB_ELFSTUB_H

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/VersionTuple.h"
#include <set>
#include <vector>

namespace llvm {
namespace elfabi {

typedef uint16_t ELFArch;

enum class ELFSymbolType {
  NoType = ELF::STT_NOTYPE,
  Object = ELF::STT_OBJECT,
  Func = ELF::STT_FUNC,
  TLS = ELF::STT_TLS,

  // Type information is 4 bits, so 16 is safely out of range.
  Unknown = 16,
};

struct ELFSymbol {
  ELFSymbol(std::string SymbolName) : Name(SymbolName) {}
  std::string Name;
  uint64_t Size;
  ELFSymbolType Type;
  bool Undefined;
  bool Weak;
  Optional<std::string> Warning;
  bool operator<(const ELFSymbol &RHS) const { return Name < RHS.Name; }
};

// A cumulative representation of ELF stubs.
// Both textual and binary stubs will read into and write from this object.
class ELFStub {
  // TODO: Add support for symbol versioning.
public:
  VersionTuple TbeVersion;
  Optional<std::string> SoName;
  ELFArch Arch;
  std::vector<std::string> NeededLibs;
  std::set<ELFSymbol> Symbols;

  ELFStub() {}
  ELFStub(const ELFStub &Stub);
  ELFStub(ELFStub &&Stub);
};
} // end namespace elfabi
} // end namespace llvm

#endif // LLVM_INTERFACESTUB_ELFSTUB_H
