//===- NativeEnumSymbol.cpp - info about enum type --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeEnumSymbol.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Native/NativeEnumTypes.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"

#include <cassert>

using namespace llvm;
using namespace llvm::pdb;

NativeEnumSymbol::NativeEnumSymbol(NativeSession &Session, SymIndexId Id,
                                   const codeview::CVType &CVT)
    : NativeRawSymbol(Session, Id), CV(CVT),
      Record(codeview::TypeRecordKind::Enum) {
  assert(CV.kind() == codeview::TypeLeafKind::LF_ENUM);
  cantFail(visitTypeRecord(CV, *this));
}

NativeEnumSymbol::~NativeEnumSymbol() {}

std::unique_ptr<NativeRawSymbol> NativeEnumSymbol::clone() const {
  return llvm::make_unique<NativeEnumSymbol>(Session, SymbolId, CV);
}

std::unique_ptr<IPDBEnumSymbols>
NativeEnumSymbol::findChildren(PDB_SymType Type) const {
  switch (Type) {
  case PDB_SymType::Data: {
    // TODO(amccarth):  Provide an actual implementation.
    return nullptr;
  }
  default:
    return nullptr;
  }
}

Error NativeEnumSymbol::visitKnownRecord(codeview::CVType &CVR,
                                         codeview::EnumRecord &ER) {
  Record = ER;
  return Error::success();
}

Error NativeEnumSymbol::visitKnownMember(codeview::CVMemberRecord &CVM,
                                         codeview::EnumeratorRecord &R) {
  return Error::success();
}

PDB_SymType NativeEnumSymbol::getSymTag() const { return PDB_SymType::Enum; }

uint32_t NativeEnumSymbol::getClassParentId() const { return 0xFFFFFFFF; }

uint32_t NativeEnumSymbol::getUnmodifiedTypeId() const { return 0; }

bool NativeEnumSymbol::hasConstructor() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasConstructorOrDestructor);
}

bool NativeEnumSymbol::hasAssignmentOperator() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasOverloadedAssignmentOperator);
}

bool NativeEnumSymbol::hasCastOperator() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasConversionOperator);
}

uint64_t NativeEnumSymbol::getLength() const {
  const auto Id = Session.findSymbolByTypeIndex(Record.getUnderlyingType());
  const auto UnderlyingType =
      Session.getConcreteSymbolById<PDBSymbolTypeBuiltin>(Id);
  return UnderlyingType ? UnderlyingType->getLength() : 0;
}

std::string NativeEnumSymbol::getName() const { return Record.getName(); }

bool NativeEnumSymbol::isNested() const {
  return bool(Record.getOptions() & codeview::ClassOptions::Nested);
}

bool NativeEnumSymbol::hasOverloadedOperator() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasOverloadedOperator);
}

bool NativeEnumSymbol::isPacked() const {
  return bool(Record.getOptions() & codeview::ClassOptions::Packed);
}

bool NativeEnumSymbol::isScoped() const {
  return bool(Record.getOptions() & codeview::ClassOptions::Scoped);
}

uint32_t NativeEnumSymbol::getTypeId() const {
  return Session.findSymbolByTypeIndex(Record.getUnderlyingType());
}
