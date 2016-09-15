//===- TypeSerializationVisitor.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPESERIALIZATIONVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPESERIALIZATIONVISITOR_H

#include "llvm/DebugInfo/CodeView/FieldListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/MemoryTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

class TypeSerializationVisitor : public TypeVisitorCallbacks {
public:
  TypeSerializationVisitor(FieldListRecordBuilder &FieldListBuilder,
                           MemoryTypeTableBuilder &TypeTableBuilder)
      : FieldListBuilder(FieldListBuilder), TypeTableBuilder(TypeTableBuilder) {
  }

  virtual Error visitTypeBegin(CVType &Record) override {
    if (Record.Type == TypeLeafKind::LF_FIELDLIST)
      FieldListBuilder.reset();
    return Error::success();
  }

  virtual Error visitTypeEnd(CVType &Record) override {
    // Since this visitor's purpose is to serialize the record, fill out the
    // fields of `Record` with the bytes of the record.
    if (Record.Type == TypeLeafKind::LF_FIELDLIST) {
      TypeTableBuilder.writeFieldList(FieldListBuilder);
      updateCVRecord(Record);
    }

    return Error::success();
  }

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  virtual Error visitKnownRecord(CVType &CVR, Name##Record &Record) override { \
    visitKnownRecordImpl(CVR, Record);                                         \
    return Error::success();                                                   \
  }
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  virtual Error visitKnownMember(CVMemberRecord &CVR, Name##Record &Record)    \
      override {                                                               \
    visitMemberRecordImpl(Record);                                             \
    return Error::success();                                                   \
  }
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

private:
  void updateCVRecord(CVType &Record) {
    StringRef S = TypeTableBuilder.getRecords().back();
    Record.RecordData = ArrayRef<uint8_t>(S.bytes_begin(), S.bytes_end());
  }
  template <typename RecordKind>
  void visitKnownRecordImpl(CVType &CVR, RecordKind &Record) {
    TypeTableBuilder.writeKnownType(Record);
    updateCVRecord(CVR);
  }
  template <typename RecordKind>
  void visitMemberRecordImpl(RecordKind &Record) {
    FieldListBuilder.writeMemberType(Record);
  }

  void visitKnownRecordImpl(CVType &CVR, FieldListRecord &FieldList) {}

  FieldListRecordBuilder &FieldListBuilder;
  MemoryTypeTableBuilder &TypeTableBuilder;
};
}
}

#endif
