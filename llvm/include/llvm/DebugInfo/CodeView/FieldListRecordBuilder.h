//===- FieldListRecordBuilder.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_FIELDLISTRECORDBUILDER_H
#define LLVM_DEBUGINFO_CODEVIEW_FIELDLISTRECORDBUILDER_H

#include "llvm/DebugInfo/CodeView/ListRecordBuilder.h"

namespace llvm {
namespace codeview {

class MethodInfo {
public:
  MethodInfo() : Access(), Kind(), Options(), Type(), VTableSlotOffset(-1) {}

  MethodInfo(MemberAccess Access, MethodKind Kind, MethodOptions Options,
             TypeIndex Type, int32_t VTableSlotOffset)
      : Access(Access), Kind(Kind), Options(Options), Type(Type),
        VTableSlotOffset(VTableSlotOffset) {}

  MemberAccess getAccess() const { return Access; }
  MethodKind getKind() const { return Kind; }
  MethodOptions getOptions() const { return Options; }
  TypeIndex getType() const { return Type; }
  int32_t getVTableSlotOffset() const { return VTableSlotOffset; }

private:
  MemberAccess Access;
  MethodKind Kind;
  MethodOptions Options;
  TypeIndex Type;
  int32_t VTableSlotOffset;
};

class FieldListRecordBuilder : public ListRecordBuilder {
private:
  FieldListRecordBuilder(const FieldListRecordBuilder &) = delete;
  void operator=(const FieldListRecordBuilder &) = delete;

public:
  FieldListRecordBuilder();

  void writeBaseClass(MemberAccess Access, TypeIndex Type, uint64_t Offset);
  void writeEnumerate(MemberAccess Access, uint64_t Value, StringRef Name);
  void writeIndirectVirtualBaseClass(MemberAccess Access, TypeIndex Type,
                                     TypeIndex VirtualBasePointerType,
                                     int64_t VirtualBasePointerOffset,
                                     uint64_t SlotIndex);
  void writeMember(MemberAccess Access, TypeIndex Type, uint64_t Offset,
                   StringRef Name);
  void writeOneMethod(MemberAccess Access, MethodKind Kind,
                      MethodOptions Options, TypeIndex Type,
                      int32_t VTableSlotOffset, StringRef Name);
  void writeOneMethod(const MethodInfo &Method, StringRef Name);
  void writeMethod(uint16_t OverloadCount, TypeIndex MethodList,
                   StringRef Name);
  void writeNestedType(TypeIndex Type, StringRef Name);
  void writeStaticMember(MemberAccess Access, TypeIndex Type, StringRef Name);
  void writeVirtualBaseClass(MemberAccess Access, TypeIndex Type,
                             TypeIndex VirtualBasePointerType,
                             int64_t VirtualBasePointerOffset,
                             uint64_t SlotIndex);
  void writeVirtualBaseClass(TypeRecordKind Kind, MemberAccess Access,
                             TypeIndex Type, TypeIndex VirtualBasePointerType,
                             int64_t VirtualBasePointerOffset,
                             uint64_t SlotIndex);
  void writeVirtualFunctionTablePointer(TypeIndex Type);
};
}
}

#endif
