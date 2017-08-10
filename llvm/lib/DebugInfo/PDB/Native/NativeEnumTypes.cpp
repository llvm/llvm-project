//==- NativeEnumTypes.cpp - Native Type Enumerator impl ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeEnumTypes.h"

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/Native/NativeEnumSymbol.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"

namespace llvm {
namespace pdb {

NativeEnumTypes::NativeEnumTypes(NativeSession &PDBSession,
                                 codeview::LazyRandomTypeCollection &Types,
                                 codeview::TypeLeafKind Kind)
    : Matches(), Index(0), Session(PDBSession), Kind(Kind) {
  for (auto Index = Types.getFirst(); Index;
       Index = Types.getNext(Index.getValue())) {
    if (Types.getType(Index.getValue()).kind() == Kind)
      Matches.push_back(Index.getValue());
  }
}

NativeEnumTypes::NativeEnumTypes(
    NativeSession &PDBSession, const std::vector<codeview::TypeIndex> &Matches,
    codeview::TypeLeafKind Kind)
    : Matches(Matches), Index(0), Session(PDBSession), Kind(Kind) {}

uint32_t NativeEnumTypes::getChildCount() const {
  return static_cast<uint32_t>(Matches.size());
}

std::unique_ptr<PDBSymbol>
NativeEnumTypes::getChildAtIndex(uint32_t Index) const {
  if (Index < Matches.size())
    return Session.createEnumSymbol(Matches[Index]);
  return nullptr;
}

std::unique_ptr<PDBSymbol> NativeEnumTypes::getNext() {
  return getChildAtIndex(Index++);
}

void NativeEnumTypes::reset() { Index = 0; }

NativeEnumTypes *NativeEnumTypes::clone() const {
  return new NativeEnumTypes(Session, Matches, Kind);
}

} // namespace pdb
} // namespace llvm
