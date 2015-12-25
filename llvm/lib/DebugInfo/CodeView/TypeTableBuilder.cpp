//===-- TypeTableBuilder.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/CodeView/FieldListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/MethodListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecordBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace codeview;

namespace {

const int PointerKindShift = 0;
const int PointerModeShift = 5;
const int PointerSizeShift = 13;

const int ClassHfaKindShift = 11;
const int ClassWindowsRTClassKindShift = 14;

void writePointerBase(TypeRecordBuilder &Builder,
                      const PointerRecordBase &Record) {
  Builder.writeTypeIndex(Record.getReferentType());
  uint32_t flags =
      static_cast<uint32_t>(Record.getOptions()) |
      (Record.getSize() << PointerSizeShift) |
      (static_cast<uint32_t>(Record.getMode()) << PointerModeShift) |
      (static_cast<uint32_t>(Record.getPointerKind()) << PointerKindShift);
  Builder.writeUInt32(flags);
}
}

TypeTableBuilder::TypeTableBuilder() {}

TypeTableBuilder::~TypeTableBuilder() {}

TypeIndex TypeTableBuilder::writeModifier(const ModifierRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::Modifier);

  Builder.writeTypeIndex(Record.getModifiedType());
  Builder.writeUInt16(static_cast<uint16_t>(Record.getOptions()));

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeProcedure(const ProcedureRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::Procedure);

  Builder.writeTypeIndex(Record.getReturnType());
  Builder.writeUInt8(static_cast<uint8_t>(Record.getCallConv()));
  Builder.writeUInt8(static_cast<uint8_t>(Record.getOptions()));
  Builder.writeUInt16(Record.getParameterCount());
  Builder.writeTypeIndex(Record.getArgumentList());

  return writeRecord(Builder);
}

TypeIndex
TypeTableBuilder::writeMemberFunction(const MemberFunctionRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::MemberFunction);

  Builder.writeTypeIndex(Record.getReturnType());
  Builder.writeTypeIndex(Record.getClassType());
  Builder.writeTypeIndex(Record.getThisType());
  Builder.writeUInt8(static_cast<uint8_t>(Record.getCallConv()));
  Builder.writeUInt8(static_cast<uint8_t>(Record.getOptions()));
  Builder.writeUInt16(Record.getParameterCount());
  Builder.writeTypeIndex(Record.getArgumentList());
  Builder.writeInt32(Record.getThisPointerAdjustment());

  return writeRecord(Builder);
}

TypeIndex
TypeTableBuilder::writeArgumentList(const ArgumentListRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::ArgumentList);

  Builder.writeUInt32(Record.getArgumentTypes().size());
  for (TypeIndex TI : Record.getArgumentTypes()) {
    Builder.writeTypeIndex(TI);
  }

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writePointer(const PointerRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::Pointer);

  writePointerBase(Builder, Record);

  return writeRecord(Builder);
}

TypeIndex
TypeTableBuilder::writePointerToMember(const PointerToMemberRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::Pointer);

  writePointerBase(Builder, Record);

  Builder.writeTypeIndex(Record.getContainingType());
  Builder.writeUInt16(static_cast<uint16_t>(Record.getRepresentation()));

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeArray(const ArrayRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::Array);

  Builder.writeTypeIndex(Record.getElementType());
  Builder.writeTypeIndex(Record.getIndexType());
  Builder.writeEncodedUnsignedInteger(Record.getSize());
  Builder.writeNullTerminatedString(Record.getName());

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeAggregate(const AggregateRecord &Record) {
  assert((Record.getKind() == TypeRecordKind::Structure) ||
         (Record.getKind() == TypeRecordKind::Class) ||
         (Record.getKind() == TypeRecordKind::Union));

  TypeRecordBuilder Builder(Record.getKind());

  Builder.writeUInt16(Record.getMemberCount());
  uint16_t Flags =
      static_cast<uint16_t>(Record.getOptions()) |
      (static_cast<uint16_t>(Record.getHfa()) << ClassHfaKindShift) |
      (static_cast<uint16_t>(Record.getWinRTKind())
       << ClassWindowsRTClassKindShift);
  Builder.writeUInt16(Flags);
  Builder.writeTypeIndex(Record.getFieldList());
  if (Record.getKind() != TypeRecordKind::Union) {
    Builder.writeTypeIndex(Record.getDerivationList());
    Builder.writeTypeIndex(Record.getVTableShape());
  } else {
    assert(Record.getDerivationList() == TypeIndex());
    assert(Record.getVTableShape() == TypeIndex());
  }
  Builder.writeEncodedUnsignedInteger(Record.getSize());
  Builder.writeNullTerminatedString(Record.getName());
  if ((Record.getOptions() & ClassOptions::HasUniqueName) !=
      ClassOptions::None) {
    Builder.writeNullTerminatedString(Record.getUniqueName());
  }

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeEnum(const EnumRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::Enum);

  Builder.writeUInt16(Record.getMemberCount());
  Builder.writeUInt16(static_cast<uint16_t>(Record.getOptions()));
  Builder.writeTypeIndex(Record.getUnderlyingType());
  Builder.writeTypeIndex(Record.getFieldList());
  Builder.writeNullTerminatedString(Record.getName());
  if ((Record.getOptions() & ClassOptions::HasUniqueName) !=
      ClassOptions::None) {
    Builder.writeNullTerminatedString(Record.getUniqueName());
  }

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeBitField(const BitFieldRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::BitField);

  Builder.writeTypeIndex(Record.getType());
  Builder.writeUInt8(Record.getBitSize());
  Builder.writeUInt8(Record.getBitOffset());

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeVirtualTableShape(
    const VirtualTableShapeRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::VirtualTableShape);

  ArrayRef<VirtualTableSlotKind> Slots = Record.getSlots();

  Builder.writeUInt16(Slots.size());
  for (size_t SlotIndex = 0; SlotIndex < Slots.size(); SlotIndex += 2) {
    uint8_t Byte = static_cast<uint8_t>(Slots[SlotIndex]) << 4;
    if ((SlotIndex + 1) < Slots.size()) {
      Byte |= static_cast<uint8_t>(Slots[SlotIndex + 1]);
    }
    Builder.writeUInt8(Byte);
  }

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeRecord(TypeRecordBuilder &Builder) {
  return writeRecord(Builder.str());
}

TypeIndex TypeTableBuilder::writeFieldList(FieldListRecordBuilder &FieldList) {
  // TODO: Split the list into multiple records if it's longer than 64KB, using
  // a subrecord of TypeRecordKind::Index to chain the records together.
  return writeRecord(FieldList.str());
}

TypeIndex
TypeTableBuilder::writeMethodList(MethodListRecordBuilder &MethodList) {
  // TODO: Split the list into multiple records if it's longer than 64KB, using
  // a subrecord of TypeRecordKind::Index to chain the records together.
  return writeRecord(MethodList.str());
}
