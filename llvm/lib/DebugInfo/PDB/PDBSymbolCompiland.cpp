//===- PDBSymbolCompiland.cpp - compiland details --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandEnv.h"

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

#include <utility>

using namespace llvm;
using namespace llvm::pdb;

PDBSymbolCompiland::PDBSymbolCompiland(const IPDBSession &PDBSession,
                                       std::unique_ptr<IPDBRawSymbol> Symbol)
    : PDBSymbol(PDBSession, std::move(Symbol)) {
  assert(RawSymbol->getSymTag() == PDB_SymType::Compiland);
}

void PDBSymbolCompiland::dump(PDBSymDumper &Dumper) const {
  Dumper.dump(*this);
}

std::string PDBSymbolCompiland::getSourceFileName() const
{
    std::string Result = RawSymbol->getSourceFileName();
    if (!Result.empty())
        return Result;
    auto Envs = findAllChildren<PDBSymbolCompilandEnv>();
    if (!Envs)
        return std::string();
    while (auto Env = Envs->getNext()) {
        std::string Var = Env->getName();
        if (Var != "src")
            continue;
        std::string Value = Env->getValue();
        return Value;
    }
    return std::string();
}
