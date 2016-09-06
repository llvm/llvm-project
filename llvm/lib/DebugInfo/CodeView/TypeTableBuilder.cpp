//===-- TypeTableBuilder.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/FieldListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/MethodListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecordBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace codeview;

TypeTableBuilder::TypeTableBuilder() {}

TypeTableBuilder::~TypeTableBuilder() {}

TypeIndex TypeTableBuilder::writeKnownType(const ModifierRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());

  Builder.writeTypeIndex(Record.getModifiedType());
  Builder.writeUInt16(static_cast<uint16_t>(Record.getModifiers()));

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const ProcedureRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());

  Builder.writeTypeIndex(Record.getReturnType());
  Builder.writeUInt8(static_cast<uint8_t>(Record.getCallConv()));
  Builder.writeUInt8(static_cast<uint8_t>(Record.getOptions()));
  Builder.writeUInt16(Record.getParameterCount());
  Builder.writeTypeIndex(Record.getArgumentList());

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const MemberFunctionRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());

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

TypeIndex TypeTableBuilder::writeKnownType(const ArgListRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());

  Builder.writeUInt32(Record.getIndices().size());
  for (TypeIndex TI : Record.getIndices()) {
    Builder.writeTypeIndex(TI);
  }

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const PointerRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());

  Builder.writeTypeIndex(Record.getReferentType());
  uint32_t flags = static_cast<uint32_t>(Record.getOptions()) |
                   (Record.getSize() << PointerRecord::PointerSizeShift) |
                   (static_cast<uint32_t>(Record.getMode())
                    << PointerRecord::PointerModeShift) |
                   (static_cast<uint32_t>(Record.getPointerKind())
                    << PointerRecord::PointerKindShift);
  Builder.writeUInt32(flags);

  if (Record.isPointerToMember()) {
    const MemberPointerInfo &M = Record.getMemberInfo();
    Builder.writeTypeIndex(M.getContainingType());
    Builder.writeUInt16(static_cast<uint16_t>(M.getRepresentation()));
  }

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const ArrayRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());

  Builder.writeTypeIndex(Record.getElementType());
  Builder.writeTypeIndex(Record.getIndexType());
  Builder.writeEncodedUnsignedInteger(Record.getSize());
  Builder.writeNullTerminatedString(Record.getName());

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const ClassRecord &Record) {
  assert((Record.getKind() == TypeRecordKind::Struct) ||
         (Record.getKind() == TypeRecordKind::Class) ||
         (Record.getKind() == TypeRecordKind::Interface));

  TypeRecordBuilder Builder(Record.getKind());

  Builder.writeUInt16(Record.getMemberCount());
  uint16_t Flags =
      static_cast<uint16_t>(Record.getOptions()) |
      (static_cast<uint16_t>(Record.getHfa()) << ClassRecord::HfaKindShift) |
      (static_cast<uint16_t>(Record.getWinRTKind())
       << ClassRecord::WinRTKindShift);
  Builder.writeUInt16(Flags);
  Builder.writeTypeIndex(Record.getFieldList());
  Builder.writeTypeIndex(Record.getDerivationList());
  Builder.writeTypeIndex(Record.getVTableShape());
  Builder.writeEncodedUnsignedInteger(Record.getSize());
  Builder.writeNullTerminatedString(Record.getName());
  if ((Record.getOptions() & ClassOptions::HasUniqueName) !=
      ClassOptions::None) {
    Builder.writeNullTerminatedString(Record.getUniqueName());
  }

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const UnionRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::Union);
  Builder.writeUInt16(Record.getMemberCount());
  uint16_t Flags =
      static_cast<uint16_t>(Record.getOptions()) |
      (static_cast<uint16_t>(Record.getHfa()) << ClassRecord::HfaKindShift);
  Builder.writeUInt16(Flags);
  Builder.writeTypeIndex(Record.getFieldList());
  Builder.writeEncodedUnsignedInteger(Record.getSize());
  Builder.writeNullTerminatedString(Record.getName());
  if ((Record.getOptions() & ClassOptions::HasUniqueName) !=
      ClassOptions::None) {
    Builder.writeNullTerminatedString(Record.getUniqueName());
  }
  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const EnumRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());

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

TypeIndex TypeTableBuilder::writeKnownType(const BitFieldRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());

  Builder.writeTypeIndex(Record.getType());
  Builder.writeUInt8(Record.getBitSize());
  Builder.writeUInt8(Record.getBitOffset());

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const VFTableShapeRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());

  ArrayRef<VFTableSlotKind> Slots = Record.getSlots();

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

TypeIndex TypeTableBuilder::writeKnownType(const VFTableRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());
  Builder.writeTypeIndex(Record.getCompleteClass());
  Builder.writeTypeIndex(Record.getOverriddenVTable());
  Builder.writeUInt32(Record.getVFPtrOffset());

  // Sum up the lengths of the null-terminated names.
  size_t NamesLen = Record.getName().size() + 1;
  for (StringRef MethodName : Record.getMethodNames())
    NamesLen += MethodName.size() + 1;

  Builder.writeUInt32(NamesLen);
  Builder.writeNullTerminatedString(Record.getName());
  for (StringRef MethodName : Record.getMethodNames())
    Builder.writeNullTerminatedString(MethodName);

  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const StringIdRecord &Record) {
  TypeRecordBuilder Builder(TypeRecordKind::StringId);
  Builder.writeTypeIndex(Record.getId());
  Builder.writeNullTerminatedString(Record.getString());
  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const UdtSourceLineRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());
  Builder.writeTypeIndex(Record.getUDT());
  Builder.writeTypeIndex(Record.getSourceFile());
  Builder.writeUInt32(Record.getLineNumber());
  return writeRecord(Builder);
}

TypeIndex
TypeTableBuilder::writeKnownType(const UdtModSourceLineRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());
  Builder.writeTypeIndex(Record.getUDT());
  Builder.writeTypeIndex(Record.getSourceFile());
  Builder.writeUInt32(Record.getLineNumber());
  Builder.writeUInt16(Record.getModule());
  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const FuncIdRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());
  Builder.writeTypeIndex(Record.getParentScope());
  Builder.writeTypeIndex(Record.getFunctionType());
  Builder.writeNullTerminatedString(Record.getName());
  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const MemberFuncIdRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());
  Builder.writeTypeIndex(Record.getClassType());
  Builder.writeTypeIndex(Record.getFunctionType());
  Builder.writeNullTerminatedString(Record.getName());
  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const BuildInfoRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());
  assert(Record.getArgs().size() <= UINT16_MAX);
  Builder.writeUInt16(Record.getArgs().size());
  for (TypeIndex Arg : Record.getArgs())
    Builder.writeTypeIndex(Arg);
  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeRecord(TypeRecordBuilder &Builder) {
  return writeRecord(Builder.str());
}

TypeIndex TypeTableBuilder::writeFieldList(FieldListRecordBuilder &FieldList) {
  return FieldList.writeListRecord(*this);
}

TypeIndex
TypeTableBuilder::writeKnownType(const MethodOverloadListRecord &Record) {
  TypeRecordBuilder Builder(Record.getKind());
  for (const OneMethodRecord &Method : Record.getMethods()) {
    uint16_t Flags = static_cast<uint16_t>(Method.getAccess());
    Flags |= static_cast<uint16_t>(Method.getKind())
             << MemberAttributes::MethodKindShift;
    Flags |= static_cast<uint16_t>(Method.getOptions());
    Builder.writeUInt16(Flags);
    Builder.writeUInt16(0); // padding
    Builder.writeTypeIndex(Method.getType());
    if (Method.isIntroducingVirtual()) {
      assert(Method.getVFTableOffset() >= 0);
      Builder.writeInt32(Method.getVFTableOffset());
    } else {
      assert(Method.getVFTableOffset() == -1);
    }
  }

  // TODO: Split the list into multiple records if it's longer than 64KB, using
  // a subrecord of TypeRecordKind::Index to chain the records together.
  return writeRecord(Builder);
}

TypeIndex TypeTableBuilder::writeKnownType(const TypeServer2Record &Record) {
  TypeRecordBuilder Builder(Record.getKind());
  Builder.writeGuid(Record.getGuid());
  Builder.writeUInt32(Record.getAge());
  Builder.writeNullTerminatedString(Record.getName());
  return writeRecord(Builder);
}
