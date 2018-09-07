//===- NativeExeSymbol.cpp - native impl for PDBSymbolExe -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeExeSymbol.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeCompilandSymbol.h"
#include "llvm/DebugInfo/PDB/Native/NativeEnumModules.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/SymbolCache.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"

using namespace llvm;
using namespace llvm::pdb;

static DbiStream *getDbiStreamPtr(NativeSession &Session) {
  Expected<DbiStream &> DbiS = Session.getPDBFile().getPDBDbiStream();
  if (DbiS)
    return &DbiS.get();

  consumeError(DbiS.takeError());
  return nullptr;
}

NativeExeSymbol::NativeExeSymbol(NativeSession &Session, SymIndexId SymbolId)
    : NativeRawSymbol(Session, PDB_SymType::Exe, SymbolId),
      Dbi(getDbiStreamPtr(Session)) {}

std::unique_ptr<NativeRawSymbol> NativeExeSymbol::clone() const {
  return llvm::make_unique<NativeExeSymbol>(Session, SymbolId);
}

std::unique_ptr<IPDBEnumSymbols>
NativeExeSymbol::findChildren(PDB_SymType Type) const {
  switch (Type) {
  case PDB_SymType::Compiland: {
    return std::unique_ptr<IPDBEnumSymbols>(new NativeEnumModules(Session));
    break;
  }
  case PDB_SymType::Enum:
    return Session.getSymbolCache().createTypeEnumerator(codeview::LF_ENUM);
  default:
    break;
  }
  return nullptr;
}

uint32_t NativeExeSymbol::getAge() const {
  auto IS = Session.getPDBFile().getPDBInfoStream();
  if (IS)
    return IS->getAge();
  consumeError(IS.takeError());
  return 0;
}

std::string NativeExeSymbol::getSymbolsFileName() const {
  return Session.getPDBFile().getFilePath();
}

codeview::GUID NativeExeSymbol::getGuid() const {
  auto IS = Session.getPDBFile().getPDBInfoStream();
  if (IS)
    return IS->getGuid();
  consumeError(IS.takeError());
  return codeview::GUID{{0}};
}

bool NativeExeSymbol::hasCTypes() const {
  auto Dbi = Session.getPDBFile().getPDBDbiStream();
  if (Dbi)
    return Dbi->hasCTypes();
  consumeError(Dbi.takeError());
  return false;
}

bool NativeExeSymbol::hasPrivateSymbols() const {
  auto Dbi = Session.getPDBFile().getPDBDbiStream();
  if (Dbi)
    return !Dbi->isStripped();
  consumeError(Dbi.takeError());
  return false;
}
