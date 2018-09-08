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

#include "llvm/Support/FormatVariadic.h"

#include <cassert>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

NativeTypeEnum::NativeTypeEnum(NativeSession &Session, SymIndexId Id,
                               const codeview::CVType &CVT)
    : NativeRawSymbol(Session, PDB_SymType::Enum, Id), CV(CVT),
      Record(codeview::TypeRecordKind::Enum) {
  assert(CV.kind() == codeview::TypeLeafKind::LF_ENUM);
  cantFail(visitTypeRecord(CV, *this));
}

NativeTypeEnum::~NativeTypeEnum() {}

void NativeTypeEnum::dump(raw_ostream &OS, int Indent) const {
  NativeRawSymbol::dump(OS, Indent);

  dumpSymbolField(OS, "baseType", static_cast<uint32_t>(getBuiltinType()),
                  Indent);
  dumpSymbolField(OS, "lexicalParentId", 0, Indent);
  dumpSymbolField(OS, "name", getName(), Indent);
  dumpSymbolField(OS, "typeId", getTypeId(), Indent);
  dumpSymbolField(OS, "length", getLength(), Indent);
  dumpSymbolField(OS, "constructor", hasConstructor(), Indent);
  dumpSymbolField(OS, "constType", isConstType(), Indent);
  dumpSymbolField(OS, "hasAssignmentOperator", hasAssignmentOperator(), Indent);
  dumpSymbolField(OS, "hasCastOperator", hasCastOperator(), Indent);
  dumpSymbolField(OS, "hasNestedTypes", hasNestedTypes(), Indent);
  dumpSymbolField(OS, "overloadedOperator", hasOverloadedOperator(), Indent);
  dumpSymbolField(OS, "isInterfaceUdt", isInterfaceUdt(), Indent);
  dumpSymbolField(OS, "intrinsic", isIntrinsic(), Indent);
  dumpSymbolField(OS, "nested", isNested(), Indent);
  dumpSymbolField(OS, "packed", isPacked(), Indent);
  dumpSymbolField(OS, "isRefUdt", isRefUdt(), Indent);
  dumpSymbolField(OS, "scoped", isScoped(), Indent);
  dumpSymbolField(OS, "unalignedType", isUnalignedType(), Indent);
  dumpSymbolField(OS, "isValueUdt", isValueUdt(), Indent);
  dumpSymbolField(OS, "volatileType", isVolatileType(), Indent);
}

std::unique_ptr<NativeRawSymbol> NativeTypeEnum::clone() const {
  return llvm::make_unique<NativeTypeEnum>(Session, SymbolId, CV);
}

std::unique_ptr<IPDBEnumSymbols>
NativeTypeEnum::findChildren(PDB_SymType Type) const {
  switch (Type) {
  case PDB_SymType::Data: {
    // TODO(amccarth) :  Provide an actual implementation.
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

PDB_BuiltinType NativeTypeEnum::getBuiltinType() const {
  Session.getSymbolCache().findSymbolByTypeIndex(Record.getUnderlyingType());

  codeview::TypeIndex Underlying = Record.getUnderlyingType();

  // This indicates a corrupt record.
  if (!Underlying.isSimple() ||
      Underlying.getSimpleMode() != SimpleTypeMode::Direct)
    return PDB_BuiltinType::None;

  switch (Underlying.getSimpleKind()) {
  case SimpleTypeKind::Boolean128:
  case SimpleTypeKind::Boolean64:
  case SimpleTypeKind::Boolean32:
  case SimpleTypeKind::Boolean16:
  case SimpleTypeKind::Boolean8:
    return PDB_BuiltinType::Bool;
  case SimpleTypeKind::NarrowCharacter:
  case SimpleTypeKind::UnsignedCharacter:
    return PDB_BuiltinType::Char;
  case SimpleTypeKind::WideCharacter:
    return PDB_BuiltinType::WCharT;
  case SimpleTypeKind::Character16:
    return PDB_BuiltinType::Char16;
  case SimpleTypeKind::Character32:
    return PDB_BuiltinType::Char32;
  case SimpleTypeKind::Int128:
  case SimpleTypeKind::Int128Oct:
  case SimpleTypeKind::Int16:
  case SimpleTypeKind::Int16Short:
  case SimpleTypeKind::Int32:
  case SimpleTypeKind::Int32Long:
  case SimpleTypeKind::Int64:
  case SimpleTypeKind::Int64Quad:
    return PDB_BuiltinType::Int;
  case SimpleTypeKind::UInt128:
  case SimpleTypeKind::UInt128Oct:
  case SimpleTypeKind::UInt16:
  case SimpleTypeKind::UInt16Short:
  case SimpleTypeKind::UInt32:
  case SimpleTypeKind::UInt32Long:
  case SimpleTypeKind::UInt64:
  case SimpleTypeKind::UInt64Quad:
    return PDB_BuiltinType::UInt;
  case SimpleTypeKind::HResult:
    return PDB_BuiltinType::HResult;
  case SimpleTypeKind::Complex16:
  case SimpleTypeKind::Complex32:
  case SimpleTypeKind::Complex32PartialPrecision:
  case SimpleTypeKind::Complex64:
  case SimpleTypeKind::Complex80:
  case SimpleTypeKind::Complex128:
    return PDB_BuiltinType::Complex;
  case SimpleTypeKind::Float16:
  case SimpleTypeKind::Float32:
  case SimpleTypeKind::Float32PartialPrecision:
  case SimpleTypeKind::Float48:
  case SimpleTypeKind::Float64:
  case SimpleTypeKind::Float80:
  case SimpleTypeKind::Float128:
    return PDB_BuiltinType::Float;
  default:
    return PDB_BuiltinType::None;
  }
  llvm_unreachable("Unreachable");
}

uint32_t NativeTypeEnum::getUnmodifiedTypeId() const {
  // FIXME: If this is const, volatile, or unaligned, we should return the
  // SymIndexId of the unmodified type here.
  return 0;
}

bool NativeTypeEnum::hasConstructor() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasConstructorOrDestructor);
}

bool NativeTypeEnum::hasAssignmentOperator() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::HasOverloadedAssignmentOperator);
}

bool NativeTypeEnum::hasNestedTypes() const {
  return bool(Record.getOptions() &
              codeview::ClassOptions::ContainsNestedClass);
}

bool NativeTypeEnum::isIntrinsic() const {
  return bool(Record.getOptions() & codeview::ClassOptions::Intrinsic);
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

bool NativeTypeEnum::isRefUdt() const { return false; }

bool NativeTypeEnum::isValueUdt() const { return false; }

bool NativeTypeEnum::isInterfaceUdt() const { return false; }
