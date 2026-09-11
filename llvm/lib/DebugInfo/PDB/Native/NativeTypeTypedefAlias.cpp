//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeTypeTypedefAlias.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

NativeTypeTypedefAlias::NativeTypeTypedefAlias(NativeSession &Session,
                                               SymIndexId Id,
                                               TypeIndex /* TI */,
                                               codeview::AliasRecord Typedef)
    : NativeRawSymbol(Session, PDB_SymType::Typedef, Id),
      Record(std::move(Typedef)) {}

NativeTypeTypedefAlias::NativeTypeTypedefAlias(
    NativeSession &Session, SymIndexId Id,
    NativeTypeTypedefAlias &UnmodifiedType, codeview::ModifierRecord Modifier)
    : NativeRawSymbol(Session, PDB_SymType::Typedef, Id),
      UnmodifiedType(&UnmodifiedType), Modifiers(Modifier) {}

NativeTypeTypedefAlias::~NativeTypeTypedefAlias() = default;

void NativeTypeTypedefAlias::dump(raw_ostream &OS, int Indent,
                                  PdbSymbolIdField ShowIdFields,
                                  PdbSymbolIdField RecurseIdFields) const {
  NativeRawSymbol::dump(OS, Indent, ShowIdFields, RecurseIdFields);
  dumpSymbolField(OS, "name", getName(), Indent);
  dumpSymbolIdField(OS, "typeId", getTypeId(), Indent, Session,
                    PdbSymbolIdField::Type, ShowIdFields, RecurseIdFields);
  dumpSymbolField(OS, "constType", isConstType(), Indent);
  dumpSymbolField(OS, "unalignedType", isUnalignedType(), Indent);
  dumpSymbolField(OS, "volatileType", isVolatileType(), Indent);
}

std::string NativeTypeTypedefAlias::getName() const {
  if (UnmodifiedType)
    return UnmodifiedType->getName();
  return std::string(Record.Name);
}

SymIndexId NativeTypeTypedefAlias::getTypeId() const {
  if (UnmodifiedType)
    return UnmodifiedType->getTypeId();

  return Session.getSymbolCache().findSymbolByTypeIndex(Record.UnderlyingType);
}

SymIndexId NativeTypeTypedefAlias::getUnmodifiedTypeId() const {
  if (UnmodifiedType)
    return UnmodifiedType->getSymIndexId();

  return 0;
}

bool NativeTypeTypedefAlias::isConstType() const {
  if (!Modifiers)
    return false;
  return (Modifiers->Modifiers & ModifierOptions::Const) !=
         ModifierOptions::None;
}

bool NativeTypeTypedefAlias::isUnalignedType() const {
  if (!Modifiers)
    return false;
  return (Modifiers->Modifiers & ModifierOptions::Unaligned) !=
         ModifierOptions::None;
}

bool NativeTypeTypedefAlias::isVolatileType() const {
  if (!Modifiers)
    return false;
  return (Modifiers->Modifiers & ModifierOptions::Volatile) !=
         ModifierOptions::None;
}
