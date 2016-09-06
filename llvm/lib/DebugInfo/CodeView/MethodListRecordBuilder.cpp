//===-- MethodListRecordBuilder.cpp ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/MethodListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/FieldListRecordBuilder.h"

using namespace llvm;
using namespace codeview;

MethodListRecordBuilder::MethodListRecordBuilder()
    : ListRecordBuilder(TypeRecordKind::MethodOverloadList) {}

void MethodListRecordBuilder::writeMethod(MemberAccess Access, MethodKind Kind,
                                          MethodOptions Options, TypeIndex Type,
                                          int32_t VTableSlotOffset) {
  TypeRecordBuilder &Builder = getBuilder();

  uint16_t Flags = static_cast<uint16_t>(Access);
  Flags |= static_cast<uint16_t>(Kind) << MethodKindShift;
  Flags |= static_cast<uint16_t>(Options);

  Builder.writeUInt16(Flags);
  Builder.writeUInt16(0);
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

  // TODO: Fail if too big?
}

void MethodListRecordBuilder::writeMethod(const MethodInfo &Method) {
  writeMethod(Method.getAccess(), Method.getKind(), Method.getOptions(),
              Method.getType(), Method.getVTableSlotOffset());
}
