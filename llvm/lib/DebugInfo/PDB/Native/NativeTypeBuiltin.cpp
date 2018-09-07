//===- NativeTypeBuiltin.cpp -------------------------------------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeTypeBuiltin.h"

namespace llvm {
namespace pdb {

NativeTypeBuiltin::NativeTypeBuiltin(NativeSession &PDBSession, SymIndexId Id,
                                     PDB_BuiltinType T, uint64_t L)
    : NativeRawSymbol(PDBSession, PDB_SymType::BuiltinType, Id),
      Session(PDBSession), Type(T), Length(L) {}

NativeTypeBuiltin::~NativeTypeBuiltin() {}

std::unique_ptr<NativeRawSymbol> NativeTypeBuiltin::clone() const {
  return llvm::make_unique<NativeTypeBuiltin>(Session, SymbolId, Type, Length);
}

void NativeTypeBuiltin::dump(raw_ostream &OS, int Indent) const {
  // TODO:  Apparently nothing needs this yet.
}

PDB_SymType NativeTypeBuiltin::getSymTag() const {
  return PDB_SymType::BuiltinType;
}

PDB_BuiltinType NativeTypeBuiltin::getBuiltinType() const { return Type; }

bool NativeTypeBuiltin::isConstType() const { return false; }

uint64_t NativeTypeBuiltin::getLength() const { return Length; }

bool NativeTypeBuiltin::isUnalignedType() const { return false; }

bool NativeTypeBuiltin::isVolatileType() const { return false; }

} // namespace pdb
} // namespace llvm
