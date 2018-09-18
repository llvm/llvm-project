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
#include "llvm/DebugInfo/PDB/Native/NativeSymbolEnumerator.h"
#include "llvm/DebugInfo/PDB/Native/NativeTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/SymbolCache.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"

#include "llvm/Support/FormatVariadic.h"

#include <cassert>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

namespace {
// Yea, this is a pretty terrible class name.  But if we have an enum:
//
// enum Foo {
//  A,
//  B
// };
//
// then A and B are the "enumerators" of the "enum" Foo.  And we need
// to enumerate them.
class NativeEnumEnumEnumerators : public IPDBEnumSymbols, TypeVisitorCallbacks {
public:
  NativeEnumEnumEnumerators(NativeSession &Session,
                            const NativeTypeEnum &ClassParent,
                            const codeview::EnumRecord &CVEnum);

  uint32_t getChildCount() const override;
  std::unique_ptr<PDBSymbol> getChildAtIndex(uint32_t Index) const override;
  std::unique_ptr<PDBSymbol> getNext() override;
  void reset() override;

private:
  Error visitKnownMember(CVMemberRecord &CVM,
                         EnumeratorRecord &Record) override;
  Error visitKnownMember(CVMemberRecord &CVM,
                         ListContinuationRecord &Record) override;

  NativeSession &Session;
  const NativeTypeEnum &ClassParent;
  const codeview::EnumRecord &CVEnum;
  std::vector<EnumeratorRecord> Enumerators;
  Optional<TypeIndex> ContinuationIndex;
  uint32_t Index = 0;
};
} // namespace

NativeEnumEnumEnumerators::NativeEnumEnumEnumerators(
    NativeSession &Session, const NativeTypeEnum &ClassParent,
    const codeview::EnumRecord &CVEnum)
    : Session(Session), ClassParent(ClassParent), CVEnum(CVEnum) {
  TpiStream &Tpi = cantFail(Session.getPDBFile().getPDBTpiStream());
  LazyRandomTypeCollection &Types = Tpi.typeCollection();

  ContinuationIndex = CVEnum.FieldList;
  while (ContinuationIndex) {
    CVType FieldList = Types.getType(*ContinuationIndex);
    assert(FieldList.kind() == LF_FIELDLIST);
    ContinuationIndex.reset();
    cantFail(visitMemberRecordStream(FieldList.data(), *this));
  }
}

Error NativeEnumEnumEnumerators::visitKnownMember(CVMemberRecord &CVM,
                                                  EnumeratorRecord &Record) {
  Enumerators.push_back(Record);
  return Error::success();
}

Error NativeEnumEnumEnumerators::visitKnownMember(
    CVMemberRecord &CVM, ListContinuationRecord &Record) {
  ContinuationIndex = Record.ContinuationIndex;
  return Error::success();
}

uint32_t NativeEnumEnumEnumerators::getChildCount() const {
  return Enumerators.size();
}

std::unique_ptr<PDBSymbol>
NativeEnumEnumEnumerators::getChildAtIndex(uint32_t Index) const {
  if (Index >= getChildCount())
    return nullptr;

  SymIndexId Id =
      Session.getSymbolCache()
          .getOrCreateFieldListMember<NativeSymbolEnumerator>(
              CVEnum.FieldList, Index, ClassParent, Enumerators[Index]);
  return Session.getSymbolCache().getSymbolById(Id);
}

std::unique_ptr<PDBSymbol> NativeEnumEnumEnumerators::getNext() {
  if (Index >= getChildCount())
    return nullptr;

  return getChildAtIndex(Index++);
}

void NativeEnumEnumEnumerators::reset() { Index = 0; }

NativeTypeEnum::NativeTypeEnum(NativeSession &Session, SymIndexId Id,
                               TypeIndex Index, EnumRecord Record)
    : NativeRawSymbol(Session, PDB_SymType::Enum, Id), Index(Index),
      Record(std::move(Record)) {}

NativeTypeEnum::NativeTypeEnum(NativeSession &Session, SymIndexId Id,
                               codeview::TypeIndex ModifierTI,
                               codeview::ModifierRecord Modifier,
                               codeview::EnumRecord EnumRecord)
    : NativeRawSymbol(Session, PDB_SymType::Enum, Id), Index(ModifierTI),
      Record(std::move(EnumRecord)), Modifiers(std::move(Modifier)) {}

NativeTypeEnum::~NativeTypeEnum() {}

void NativeTypeEnum::dump(raw_ostream &OS, int Indent) const {
  NativeRawSymbol::dump(OS, Indent);

  dumpSymbolField(OS, "baseType", static_cast<uint32_t>(getBuiltinType()),
                  Indent);
  dumpSymbolField(OS, "lexicalParentId", 0, Indent);
  dumpSymbolField(OS, "name", getName(), Indent);
  dumpSymbolField(OS, "typeId", getTypeId(), Indent);
  if (Modifiers.hasValue())
    dumpSymbolField(OS, "unmodifiedTypeId", getUnmodifiedTypeId(), Indent);
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

std::unique_ptr<IPDBEnumSymbols>
NativeTypeEnum::findChildren(PDB_SymType Type) const {
  if (Type != PDB_SymType::Data)
    return llvm::make_unique<NullEnumerator<PDBSymbol>>();

  const NativeTypeEnum *ClassParent = nullptr;
  if (!Modifiers)
    ClassParent = this;
  else {
    NativeRawSymbol &NRS =
        Session.getSymbolCache().getNativeSymbolById(getUnmodifiedTypeId());
    assert(NRS.getSymTag() == PDB_SymType::Enum);
    ClassParent = static_cast<NativeTypeEnum *>(&NRS);
  }
  return llvm::make_unique<NativeEnumEnumEnumerators>(Session, *ClassParent,
                                                      Record);
}

PDB_SymType NativeTypeEnum::getSymTag() const { return PDB_SymType::Enum; }

PDB_BuiltinType NativeTypeEnum::getBuiltinType() const {
  Session.getSymbolCache().findSymbolByTypeIndex(Record.getUnderlyingType());

  codeview::TypeIndex Underlying = Record.getUnderlyingType();

  // This indicates a corrupt record.
  if (!Underlying.isSimple() ||
      Underlying.getSimpleMode() != SimpleTypeMode::Direct) {
    return PDB_BuiltinType::None;
  }

  switch (Underlying.getSimpleKind()) {
  case SimpleTypeKind::Boolean128:
  case SimpleTypeKind::Boolean64:
  case SimpleTypeKind::Boolean32:
  case SimpleTypeKind::Boolean16:
  case SimpleTypeKind::Boolean8:
    return PDB_BuiltinType::Bool;
  case SimpleTypeKind::NarrowCharacter:
  case SimpleTypeKind::UnsignedCharacter:
  case SimpleTypeKind::SignedCharacter:
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

SymIndexId NativeTypeEnum::getUnmodifiedTypeId() const {
  if (!Modifiers)
    return 0;

  return Session.getSymbolCache().findSymbolByTypeIndex(
      Modifiers->ModifiedType);
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

SymIndexId NativeTypeEnum::getTypeId() const {
  return Session.getSymbolCache().findSymbolByTypeIndex(
      Record.getUnderlyingType());
}

bool NativeTypeEnum::isRefUdt() const { return false; }

bool NativeTypeEnum::isValueUdt() const { return false; }

bool NativeTypeEnum::isInterfaceUdt() const { return false; }

bool NativeTypeEnum::isConstType() const {
  if (!Modifiers)
    return false;
  return ((Modifiers->getModifiers() & ModifierOptions::Const) !=
          ModifierOptions::None);
}

bool NativeTypeEnum::isVolatileType() const {
  if (!Modifiers)
    return false;
  return ((Modifiers->getModifiers() & ModifierOptions::Volatile) !=
          ModifierOptions::None);
}

bool NativeTypeEnum::isUnalignedType() const {
  if (!Modifiers)
    return false;
  return ((Modifiers->getModifiers() & ModifierOptions::Unaligned) !=
          ModifierOptions::None);
}

const NativeTypeBuiltin &NativeTypeEnum::getUnderlyingBuiltinType() const {
  return Session.getSymbolCache().getNativeSymbolById<NativeTypeBuiltin>(
      getTypeId());
}
