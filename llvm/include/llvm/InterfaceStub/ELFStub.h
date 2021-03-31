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
#include "llvm/Support/Error.h"
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

enum class ELFEndiannessType {
  Little = ELF::ELFDATA2LSB,
  Big = ELF::ELFDATA2MSB,

  // Endianness info is 1 bytes, 256 is safely out of rance.
  Unknown = 256,
};

enum class ELFBitWidthType {
  ELF32 = ELF::ELFCLASS32,
  ELF64 = ELF::ELFCLASS64,

  // Bit width info is 1 bytes, 256 is safely out of rance.
  Unknown = 256,
};

struct ELFSymbol {
  ELFSymbol() = default;
  explicit ELFSymbol(std::string SymbolName) : Name(std::move(SymbolName)) {}
  std::string Name;
  uint64_t Size;
  ELFSymbolType Type;
  bool Undefined;
  bool Weak;
  Optional<std::string> Warning;
  bool operator<(const ELFSymbol &RHS) const { return Name < RHS.Name; }
};

struct IFSTarget {
  Optional<std::string> Triple;
  Optional<std::string> ObjectFormat;
  Optional<ELFArch> Arch;
  Optional<std::string> ArchString;
  Optional<ELFEndiannessType> Endianness;
  Optional<ELFBitWidthType> BitWidth;
};

// A cumulative representation of ELF stubs.
// Both textual and binary stubs will read into and write from this object.
struct ELFStub {
  // TODO: Add support for symbol versioning.
  VersionTuple TbeVersion;
  Optional<std::string> SoName;
  IFSTarget Target;
  std::vector<std::string> NeededLibs;
  std::vector<ELFSymbol> Symbols;

  ELFStub() {}
  ELFStub(const ELFStub &Stub);
  ELFStub(ELFStub &&Stub);
};

// Create a alias class for ELFStub.
// LLVM's YAML library does not allow mapping a class with 2 traits,
// which prevents us using 'Target:' field with different definitions.
// This class makes it possible to map a second traits so the same data
// structure can be used for 2 different yaml schema.
struct ELFStubTriple : ELFStub {
  ELFStubTriple() {}
  ELFStubTriple(const ELFStub &Stub);
  ELFStubTriple(const ELFStubTriple &Stub);
  ELFStubTriple(ELFStubTriple &&Stub);
};

} // end namespace elfabi
} // end namespace llvm

#endif // LLVM_INTERFACESTUB_ELFSTUB_H
