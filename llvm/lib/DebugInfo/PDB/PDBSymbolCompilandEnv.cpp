//===- PDBSymbolCompilandEnv.cpp - compiland env variables ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolCompilandEnv.h"

#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymDumper.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"

#include <utility>

using namespace llvm;
using namespace llvm::pdb;

PDBSymbolCompilandEnv::PDBSymbolCompilandEnv(
    const IPDBSession &PDBSession, std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {
  assert(RawSymbol->getSymTag() == PDB_SymType::CompilandEnv);
}

std::string PDBSymbolCompilandEnv::getValue() const {
  Variant Value = RawSymbol->getValue();
  if (Value.Type != PDB_VariantType::String)
    return std::string();
  return std::string(Value.Value.String);
}

void PDBSymbolCompilandEnv::dump(PDBSymDumper &Dumper) const {
  Dumper.dump(*this);
}
