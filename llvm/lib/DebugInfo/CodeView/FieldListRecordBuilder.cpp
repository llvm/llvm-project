//===-- FieldListRecordBuilder.cpp ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/FieldListRecordBuilder.h"

using namespace llvm;
using namespace codeview;

FieldListRecordBuilder::FieldListRecordBuilder()
    : ListRecordBuilder(TypeRecordKind::FieldList) {}

void FieldListRecordBuilder::writeMemberType(const BaseClassRecord &Record) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::BaseClass);
  Builder.writeUInt16(static_cast<uint16_t>(Record.getAccess()));
  Builder.writeTypeIndex(Record.getBaseType());
  Builder.writeEncodedUnsignedInteger(Record.getBaseOffset());

  finishSubRecord();
}

void FieldListRecordBuilder::writeMemberType(const EnumeratorRecord &Record) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::Enumerator);
  Builder.writeUInt16(static_cast<uint16_t>(Record.getAccess()));
  // FIXME: Handle full APInt such as __int128.
  Builder.writeEncodedUnsignedInteger(Record.getValue().getZExtValue());
  Builder.writeNullTerminatedString(Record.getName());

  finishSubRecord();
}

void FieldListRecordBuilder::writeMemberType(const DataMemberRecord &Record) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(Record.getKind());
  Builder.writeUInt16(static_cast<uint16_t>(Record.getAccess()));
  Builder.writeTypeIndex(Record.getType());
  Builder.writeEncodedUnsignedInteger(Record.getFieldOffset());
  Builder.writeNullTerminatedString(Record.getName());

  finishSubRecord();
}

void FieldListRecordBuilder::writeMemberType(
    const OverloadedMethodRecord &Record) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::OverloadedMethod);
  Builder.writeUInt16(Record.getNumOverloads());
  Builder.writeTypeIndex(Record.getMethodList());
  Builder.writeNullTerminatedString(Record.getName());

  finishSubRecord();
}

void FieldListRecordBuilder::writeMemberType(const OneMethodRecord &Record) {
  TypeRecordBuilder &Builder = getBuilder();

  uint16_t Flags = static_cast<uint16_t>(Record.getAccess());
  Flags |= static_cast<uint16_t>(Record.getKind()) << MethodKindShift;
  Flags |= static_cast<uint16_t>(Record.getOptions());

  Builder.writeTypeRecordKind(TypeRecordKind::OneMethod);
  Builder.writeUInt16(Flags);
  Builder.writeTypeIndex(Record.getType());
  if (Record.isIntroducingVirtual()) {
    assert(Record.getVFTableOffset() >= 0);
    Builder.writeInt32(Record.getVFTableOffset());
  } else {
    assert(Record.getVFTableOffset() == -1);
  }

  Builder.writeNullTerminatedString(Record.getName());

  finishSubRecord();
}

void FieldListRecordBuilder::writeMemberType(const NestedTypeRecord &Record) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(Record.getKind());
  Builder.writeUInt16(0);
  Builder.writeTypeIndex(Record.getNestedType());
  Builder.writeNullTerminatedString(Record.getName());

  finishSubRecord();
}

void FieldListRecordBuilder::writeMemberType(
    const StaticDataMemberRecord &Record) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(Record.getKind());
  Builder.writeUInt16(static_cast<uint16_t>(Record.getAccess()));
  Builder.writeTypeIndex(Record.getType());
  Builder.writeNullTerminatedString(Record.getName());

  finishSubRecord();
}

void FieldListRecordBuilder::writeMemberType(
    const VirtualBaseClassRecord &Record) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(Record.getKind());
  Builder.writeUInt16(static_cast<uint16_t>(Record.getAccess()));
  Builder.writeTypeIndex(Record.getBaseType());
  Builder.writeTypeIndex(Record.getVBPtrType());
  Builder.writeEncodedInteger(Record.getVBPtrOffset());
  Builder.writeEncodedUnsignedInteger(Record.getVTableIndex());

  finishSubRecord();
}

void FieldListRecordBuilder::writeMemberType(const VFPtrRecord &Record) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::VFPtr);
  Builder.writeUInt16(0);
  Builder.writeTypeIndex(Record.getType());

  finishSubRecord();
}
