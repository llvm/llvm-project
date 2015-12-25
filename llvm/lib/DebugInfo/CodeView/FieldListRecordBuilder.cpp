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

void FieldListRecordBuilder::writeBaseClass(MemberAccess Access, TypeIndex Type,
                                            uint64_t Offset) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::BaseClass);
  Builder.writeUInt16(static_cast<uint16_t>(Access));
  Builder.writeTypeIndex(Type);
  Builder.writeEncodedUnsignedInteger(Offset);

  finishSubRecord();
}

void FieldListRecordBuilder::writeEnumerate(MemberAccess Access, uint64_t Value,
                                            StringRef Name) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::Enumerate);
  Builder.writeUInt16(static_cast<uint16_t>(Access));
  Builder.writeEncodedUnsignedInteger(Value);
  Builder.writeNullTerminatedString(Name);

  finishSubRecord();
}

void FieldListRecordBuilder::writeMember(MemberAccess Access, TypeIndex Type,
                                         uint64_t Offset, StringRef Name) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::Member);
  Builder.writeUInt16(static_cast<uint16_t>(Access));
  Builder.writeTypeIndex(Type);
  Builder.writeEncodedUnsignedInteger(Offset);
  Builder.writeNullTerminatedString(Name);

  finishSubRecord();
}

void FieldListRecordBuilder::writeMethod(uint16_t OverloadCount,
                                         TypeIndex MethodList, StringRef Name) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::Method);
  Builder.writeUInt16(OverloadCount);
  Builder.writeTypeIndex(MethodList);
  Builder.writeNullTerminatedString(Name);

  finishSubRecord();
}

void FieldListRecordBuilder::writeOneMethod(
    MemberAccess Access, MethodKind Kind, MethodOptions Options, TypeIndex Type,
    int32_t VTableSlotOffset, StringRef Name) {
  TypeRecordBuilder &Builder = getBuilder();

  uint16_t Flags = static_cast<uint16_t>(Access);
  Flags |= static_cast<uint16_t>(Kind) << MethodKindShift;
  Flags |= static_cast<uint16_t>(Options);

  Builder.writeTypeRecordKind(TypeRecordKind::OneMethod);
  Builder.writeUInt16(Flags);
  Builder.writeTypeIndex(Type);
  switch (Kind) {
  case MethodKind::IntroducingVirtual:
  case MethodKind::PureIntroducingVirtual:
    assert(VTableSlotOffset >= 0);
    Builder.writeInt32(VTableSlotOffset);
    break;

  default:
    assert(VTableSlotOffset == -1);
    break;
  }

  Builder.writeNullTerminatedString(Name);

  finishSubRecord();
}

void FieldListRecordBuilder::writeOneMethod(const MethodInfo &Method,
                                            StringRef Name) {
  writeOneMethod(Method.getAccess(), Method.getKind(), Method.getOptions(),
                 Method.getType(), Method.getVTableSlotOffset(), Name);
}

void FieldListRecordBuilder::writeNestedType(TypeIndex Type, StringRef Name) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::NestedType);
  Builder.writeUInt16(0);
  Builder.writeTypeIndex(Type);
  Builder.writeNullTerminatedString(Name);

  finishSubRecord();
}

void FieldListRecordBuilder::writeStaticMember(MemberAccess Access,
                                               TypeIndex Type, StringRef Name) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::StaticMember);
  Builder.writeUInt16(static_cast<uint16_t>(Access));
  Builder.writeTypeIndex(Type);
  Builder.writeNullTerminatedString(Name);

  finishSubRecord();
}

void FieldListRecordBuilder::writeIndirectVirtualBaseClass(
    MemberAccess Access, TypeIndex Type, TypeIndex VirtualBasePointerType,
    int64_t VirtualBasePointerOffset, uint64_t SlotIndex) {
  writeVirtualBaseClass(TypeRecordKind::IndirectVirtualBaseClass, Access, Type,
                        VirtualBasePointerType, VirtualBasePointerOffset,
                        SlotIndex);
}

void FieldListRecordBuilder::writeVirtualBaseClass(
    MemberAccess Access, TypeIndex Type, TypeIndex VirtualBasePointerType,
    int64_t VirtualBasePointerOffset, uint64_t SlotIndex) {
  writeVirtualBaseClass(TypeRecordKind::VirtualBaseClass, Access, Type,
                        VirtualBasePointerType, VirtualBasePointerOffset,
                        SlotIndex);
}

void FieldListRecordBuilder::writeVirtualBaseClass(
    TypeRecordKind Kind, MemberAccess Access, TypeIndex Type,
    TypeIndex VirtualBasePointerType, int64_t VirtualBasePointerOffset,
    uint64_t SlotIndex) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(Kind);
  Builder.writeUInt16(static_cast<uint16_t>(Access));
  Builder.writeTypeIndex(Type);
  Builder.writeTypeIndex(VirtualBasePointerType);
  Builder.writeEncodedInteger(VirtualBasePointerOffset);
  Builder.writeEncodedUnsignedInteger(SlotIndex);

  finishSubRecord();
}

void FieldListRecordBuilder::writeVirtualFunctionTablePointer(TypeIndex Type) {
  TypeRecordBuilder &Builder = getBuilder();

  Builder.writeTypeRecordKind(TypeRecordKind::VirtualFunctionTablePointer);
  Builder.writeUInt16(0);
  Builder.writeTypeIndex(Type);

  finishSubRecord();
}