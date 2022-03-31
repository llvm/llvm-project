//===- TapiFile.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Text-based Dynamcic Library Stub format.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/TapiFile.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TextAPI/ArchitectureSet.h"
#include "llvm/TextAPI/InterfaceFile.h"
#include "llvm/TextAPI/Platform.h"
#include "llvm/TextAPI/Symbol.h"

using namespace llvm;
using namespace MachO;
using namespace object;

static uint32_t getFlags(const Symbol *Sym) {
  uint32_t Flags = BasicSymbolRef::SF_Global;
  if (Sym->isUndefined())
    Flags |= BasicSymbolRef::SF_Undefined;
  else
    Flags |= BasicSymbolRef::SF_Exported;

  if (Sym->isWeakDefined() || Sym->isWeakReferenced())
    Flags |= BasicSymbolRef::SF_Weak;

  return Flags;
}

TapiFile::TapiFile(MemoryBufferRef Source, const InterfaceFile &interface,
                   Architecture Arch)
    : SymbolicFile(ID_TapiFile, Source), Arch(Arch) {
  for (const auto *Symbol : interface.symbols()) {
    if (!Symbol->getArchitectures().has(Arch))
      continue;

    switch (Symbol->getKind()) {
    case SymbolKind::GlobalSymbol:
      Symbols.emplace_back(StringRef(), Symbol->getName(), getFlags(Symbol));
      break;
    case SymbolKind::ObjectiveCClass:
      if (interface.getPlatforms().count(PLATFORM_MACOS) && Arch == AK_i386) {
        Symbols.emplace_back(ObjC1ClassNamePrefix, Symbol->getName(),
                             getFlags(Symbol));
      } else {
        Symbols.emplace_back(ObjC2ClassNamePrefix, Symbol->getName(),
                             getFlags(Symbol));
        Symbols.emplace_back(ObjC2MetaClassNamePrefix, Symbol->getName(),
                             getFlags(Symbol));
      }
      break;
    case SymbolKind::ObjectiveCClassEHType:
      Symbols.emplace_back(ObjC2EHTypePrefix, Symbol->getName(),
                           getFlags(Symbol));
      break;
    case SymbolKind::ObjectiveCInstanceVariable:
      Symbols.emplace_back(ObjC2IVarPrefix, Symbol->getName(),
                           getFlags(Symbol));
      break;
    }
  }
}

TapiFile::~TapiFile() = default;

void TapiFile::moveSymbolNext(DataRefImpl &DRI) const { DRI.d.a++; }

Error TapiFile::printSymbolName(raw_ostream &OS, DataRefImpl DRI) const {
  assert(DRI.d.a < Symbols.size() && "Attempt to access symbol out of bounds");
  const Symbol &Sym = Symbols[DRI.d.a];
  OS << Sym.Prefix << Sym.Name;
  return Error::success();
}

Expected<uint32_t> TapiFile::getSymbolFlags(DataRefImpl DRI) const {
  assert(DRI.d.a < Symbols.size() && "Attempt to access symbol out of bounds");
  return Symbols[DRI.d.a].Flags;
}

basic_symbol_iterator TapiFile::symbol_begin() const {
  DataRefImpl DRI;
  DRI.d.a = 0;
  return BasicSymbolRef{DRI, this};
}

basic_symbol_iterator TapiFile::symbol_end() const {
  DataRefImpl DRI;
  DRI.d.a = Symbols.size();
  return BasicSymbolRef{DRI, this};
}
