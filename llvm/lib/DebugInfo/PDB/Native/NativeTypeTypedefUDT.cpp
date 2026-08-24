//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeTypeTypedefUDT.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

NativeTypeTypedefUDT::NativeTypeTypedefUDT(NativeSession &Session, SymIndexId Id,
                                     codeview::UDTSym Typedef)
    : NativeRawSymbol(Session, PDB_SymType::Typedef, Id),
      Record(std::move(Typedef)) {}

NativeTypeTypedefUDT::~NativeTypeTypedefUDT() = default;

void NativeTypeTypedefUDT::dump(raw_ostream &OS, int Indent,
                             PdbSymbolIdField ShowIdFields,
                             PdbSymbolIdField RecurseIdFields) const {
  NativeRawSymbol::dump(OS, Indent, ShowIdFields, RecurseIdFields);
  dumpSymbolField(OS, "name", getName(), Indent);
  dumpSymbolIdField(OS, "typeId", getTypeId(), Indent, Session,
                    PdbSymbolIdField::Type, ShowIdFields, RecurseIdFields);
}

std::string NativeTypeTypedefUDT::getName() const {
  return std::string(Record.Name);
}

SymIndexId NativeTypeTypedefUDT::getTypeId() const {
  return Session.getSymbolCache().findSymbolByTypeIndex(Record.Type);
}
