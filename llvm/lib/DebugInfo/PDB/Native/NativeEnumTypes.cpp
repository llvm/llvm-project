//==- NativeEnumTypes.cpp - Native Type Enumerator impl ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeEnumTypes.h"

#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/NativeTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

NativeEnumTypes::NativeEnumTypes(NativeSession &PDBSession,
                                 LazyRandomTypeCollection &Types,
                                 TypeLeafKind Kind)
    : Matches(), Index(0), Session(PDBSession) {
  Optional<TypeIndex> TI = Types.getFirst();
  while (TI) {
    CVType CVT = Types.getType(*TI);
    TypeLeafKind K = CVT.kind();
    if (K == Kind)
      Matches.push_back(*TI);
    else if (K == TypeLeafKind::LF_MODIFIER) {
      ModifierRecord MR;
      if (auto EC = TypeDeserializer::deserializeAs<ModifierRecord>(CVT, MR)) {
        consumeError(std::move(EC));
      } else if (!MR.ModifiedType.isSimple()) {
        CVType UnmodifiedCVT = Types.getType(MR.ModifiedType);
        if (UnmodifiedCVT.kind() == Kind)
          Matches.push_back(*TI);
      }
    }
    TI = Types.getNext(*TI);
  }
}

NativeEnumTypes::NativeEnumTypes(NativeSession &PDBSession,
                                 const std::vector<TypeIndex> &Matches,
                                 TypeLeafKind Kind)
    : Matches(Matches), Index(0), Session(PDBSession) {}

uint32_t NativeEnumTypes::getChildCount() const {
  return static_cast<uint32_t>(Matches.size());
}

std::unique_ptr<PDBSymbol>
NativeEnumTypes::getChildAtIndex(uint32_t Index) const {
  if (Index < Matches.size()) {
    SymIndexId Id =
        Session.getSymbolCache().findSymbolByTypeIndex(Matches[Index]);
    return Session.getSymbolCache().getSymbolById(Id);
  }
  return nullptr;
}

std::unique_ptr<PDBSymbol> NativeEnumTypes::getNext() {
  return getChildAtIndex(Index++);
}

void NativeEnumTypes::reset() { Index = 0; }
