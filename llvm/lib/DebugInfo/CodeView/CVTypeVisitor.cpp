//===- CVTypeVisitor.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"

using namespace llvm;
using namespace llvm::codeview;

template <typename T>
static Error takeObject(ArrayRef<uint8_t> &Data, const T *&Res) {
  if (Data.size() < sizeof(*Res))
    return llvm::make_error<CodeViewError>(cv_error_code::insufficient_buffer);
  Res = reinterpret_cast<const T *>(Data.data());
  Data = Data.drop_front(sizeof(*Res));
  return Error::success();
}

CVTypeVisitor::CVTypeVisitor(TypeVisitorCallbacks &Callbacks)
    : Callbacks(Callbacks) {}

Error CVTypeVisitor::visitTypeRecord(const CVRecord<TypeLeafKind> &Record) {
  ArrayRef<uint8_t> LeafData = Record.Data;
  if (auto EC = Callbacks.visitTypeBegin(Record))
    return EC;
  switch (Record.Type) {
  default:
    if (auto EC = Callbacks.visitUnknownType(Record))
      return EC;
    break;
  case LF_FIELDLIST:
    if (auto EC = Callbacks.visitFieldListBegin(Record))
      return EC;
    if (auto EC = visitFieldList(Record))
      return EC;
    if (auto EC = Callbacks.visitFieldListEnd(Record))
      return EC;
    break;
#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  case EnumName: {                                                             \
    TypeRecordKind RK = static_cast<TypeRecordKind>(EnumName);                 \
    auto Result = Name##Record::deserialize(RK, LeafData);                     \
    if (Result.getError())                                                     \
      return llvm::make_error<CodeViewError>(cv_error_code::corrupt_record);   \
    if (auto EC = Callbacks.visit##Name(*Result))                              \
      return EC;                                                               \
    break;                                                                     \
  }
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)                  \
  TYPE_RECORD(EnumVal, EnumVal, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"
  }
  if (auto EC = Callbacks.visitTypeEnd(Record))
    return EC;
  return Error::success();
}

/// Visits the type records in Data. Sets the error flag on parse failures.
Error CVTypeVisitor::visitTypeStream(const CVTypeArray &Types) {
  for (const auto &I : Types) {
    if (auto EC = visitTypeRecord(I))
      return EC;
  }
  return Error::success();
}

Error CVTypeVisitor::skipPadding(ArrayRef<uint8_t> &Data) {
  if (Data.empty())
    return Error::success();
  uint8_t Leaf = Data.front();
  if (Leaf < LF_PAD0)
    return Error::success();
  // Leaf is greater than 0xf0. We should advance by the number of bytes in
  // the low 4 bits.
  unsigned BytesToAdvance = Leaf & 0x0F;
  if (Data.size() < BytesToAdvance) {
    return llvm::make_error<CodeViewError>(cv_error_code::corrupt_record,
                                           "Invalid padding bytes!");
  }
  Data = Data.drop_front(BytesToAdvance);
  return Error::success();
}

/// Visits individual member records of a field list record. Member records do
/// not describe their own length, and need special handling.
Error CVTypeVisitor::visitFieldList(const CVRecord<TypeLeafKind> &Record) {
  ArrayRef<uint8_t> RecordData = Record.Data;
  while (!RecordData.empty()) {
    const ulittle16_t *LeafPtr;
    if (auto EC = takeObject(RecordData, LeafPtr))
      return EC;
    TypeLeafKind Leaf = TypeLeafKind(unsigned(*LeafPtr));
    switch (Leaf) {
    default:
      // Field list records do not describe their own length, so we cannot
      // continue parsing past an unknown member type.
      if (auto EC = Callbacks.visitUnknownMember(Record))
        return llvm::make_error<CodeViewError>(cv_error_code::corrupt_record);
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  case EnumName: {                                                             \
    TypeRecordKind RK = static_cast<TypeRecordKind>(EnumName);                 \
    auto Result = Name##Record::deserialize(RK, RecordData);                   \
    if (Result.getError())                                                     \
      return llvm::make_error<CodeViewError>(cv_error_code::corrupt_record);   \
    if (auto EC = Callbacks.visit##Name(*Result))                              \
      return EC;                                                               \
    break;                                                                     \
  }
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)                \
  MEMBER_RECORD(EnumVal, EnumVal, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"
    }
    if (auto EC = skipPadding(RecordData))
      return EC;
  }
  return Error::success();
}
