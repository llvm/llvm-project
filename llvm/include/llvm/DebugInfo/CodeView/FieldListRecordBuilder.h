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

  void writeMemberType(const BaseClassRecord &Record);
  void writeMemberType(const EnumeratorRecord &Record);
  void writeMemberType(const DataMemberRecord &Record);
  void writeMemberType(const OneMethodRecord &Record);
  void writeMemberType(const OverloadedMethodRecord &Record);
  void writeMemberType(const NestedTypeRecord &Record);
  void writeMemberType(const StaticDataMemberRecord &Record);
  void writeMemberType(const VirtualBaseClassRecord &Record);
  void writeMemberType(const VFPtrRecord &Type);

  using ListRecordBuilder::writeMemberType;
};
}
}

#endif
