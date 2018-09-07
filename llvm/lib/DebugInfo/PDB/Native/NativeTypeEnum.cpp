//===- NativeTypeEnum.cpp - info about enum type ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeTypeEnum.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Native/NativeEnumTypes.h"
#include "llvm/DebugInfo/PDB/Native/SymbolCache.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"

#include <cassert>

using namespace llvm;
using namespace llvm::pdb;

NativeTypeEnum::NativeTypeEnum(NativeSession &Session, SymIndexId Id,
                               const codeview::CVType &CVT)
    : NativeRawSymbol(Session, PDB_SymType::Enum, Id), CV(CVT),
      Record(codeview::TypeRecordKind::Enum) {
  assert(CV.kind() == codeview::TypeLeafKind::LF_ENUM);
  cantFail(visitTypeRecord(CV, *this));
}

NativeTypeEnum::~NativeTypeEnum() {}

std::unique_ptr<NativeRawSymbol> NativeTypeEnum::clone() const {
  return llvm::make_unique<NativeTypeEnum>(Session, SymbolId, CV);
}

std::unique_ptr<IPDBEnumSymbols>
NativeTypeEnum::findChildren(PDB_SymType Type) const {
  switch (Type) {
  case PDB_SymType::Data: {
    // TODO(amccarth):  Provide an actual implementation.
    return nullptr;
  }
  default:
    return nullptr;
  }
}

Error NativeTypeEnum::visitKnownRecord(codeview::CVType &CVR,
                                       codeview::EnumRecord &ER) {
  Record = ER;
  return Error::success();
}

Error NativeTypeEnum::visitKnownMember(codeview::CVMemberRecord &CVM,
                                       codeview::EnumeratorRecord &R) {
  return Error::success();
}

PDB_SymType NativeTypeEnum::getSymTag() const { return PDB_SymType::Enum; }

uint32_t NativeTypeEnum::getClassParentId() const { return 0xFFFFFFFF; }

uint32_t NativeTypeEnum::getUnmodifiedTypeId() const { return 0; }

bool NativeTypeEnum::hasConstructor() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasConstructorOrDestructor);
}

bool NativeTypeEnum::hasAssignmentOperator() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasOverloadedAssignmentOperator);
}

bool NativeTypeEnum::hasCastOperator() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasConversionOperator);
}

uint64_t NativeTypeEnum::getLength() const {
  const auto Id = Session.getSymbolCache().findSymbolByTypeIndex(
      Record.getUnderlyingType());
  const auto UnderlyingType =
      Session.getConcreteSymbolById<PDBSymbolTypeBuiltin>(Id);
  return UnderlyingType ? UnderlyingType->getLength() : 0;
}

std::string NativeTypeEnum::getName() const { return Record.getName(); }

bool NativeTypeEnum::isNested() const {
  return bool(Record.getOptions() & codeview::ClassOptions::Nested);
}

bool NativeTypeEnum::hasOverloadedOperator() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasOverloadedOperator);
}

bool NativeTypeEnum::isPacked() const {
  return bool(Record.getOptions() & codeview::ClassOptions::Packed);
}

bool NativeTypeEnum::isScoped() const {
  return bool(Record.getOptions() & codeview::ClassOptions::Scoped);
}

uint32_t NativeTypeEnum::getTypeId() const {
  return Session.getSymbolCache().findSymbolByTypeIndex(
      Record.getUnderlyingType());
}
