//===- NativeBuiltinSymbol.cpp ------------------------------------ C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeBuiltinSymbol.h"

#include "llvm/DebugInfo/PDB/Native/NativeSession.h"

namespace llvm {
namespace pdb {

NativeBuiltinSymbol::NativeBuiltinSymbol(NativeSession &PDBSession,
                                         SymIndexId Id, PDB_BuiltinType T,
                                         uint64_t L)
    : NativeRawSymbol(PDBSession, Id), Session(PDBSession), Type(T), Length(L) {
}

NativeBuiltinSymbol::~NativeBuiltinSymbol() {}

std::unique_ptr<NativeRawSymbol> NativeBuiltinSymbol::clone() const {
  return llvm::make_unique<NativeBuiltinSymbol>(Session, SymbolId, Type, Length);
}

void NativeBuiltinSymbol::dump(raw_ostream &OS, int Indent) const {
  // TODO:  Apparently nothing needs this yet.
}

PDB_SymType NativeBuiltinSymbol::getSymTag() const {
  return PDB_SymType::BuiltinType;
}

PDB_BuiltinType NativeBuiltinSymbol::getBuiltinType() const { return Type; }

bool NativeBuiltinSymbol::isConstType() const { return false; }

uint64_t NativeBuiltinSymbol::getLength() const { return Length; }

bool NativeBuiltinSymbol::isUnalignedType() const { return false; }

bool NativeBuiltinSymbol::isVolatileType() const { return false; }

} // namespace pdb
} // namespace llvm
