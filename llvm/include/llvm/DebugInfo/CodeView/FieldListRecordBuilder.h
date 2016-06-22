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
#include "llvm/DebugInfo/CodeView/TypeRecord.h"

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

  void reset() { ListRecordBuilder::reset(); }

  void writeBaseClass(const BaseClassRecord &Record);
  void writeEnumerator(const EnumeratorRecord &Record);
  void writeDataMember(const DataMemberRecord &Record);
  void writeOneMethod(const OneMethodRecord &Record);
  void writeOverloadedMethod(const OverloadedMethodRecord &Record);
  void writeNestedType(const NestedTypeRecord &Record);
  void writeStaticDataMember(const StaticDataMemberRecord &Record);
  void writeVirtualBaseClass(const VirtualBaseClassRecord &Record);
  void writeVFPtr(const VFPtrRecord &Type);
};
}
}

#endif
