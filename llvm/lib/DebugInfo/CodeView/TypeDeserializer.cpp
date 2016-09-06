//===- TypeDeserializer.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"

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

Error TypeDeserializer::visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,
                                         FieldListRecord &Record) {
  ArrayRef<uint8_t> FieldListRecordData = CVR.Data;
  auto ExpectedRecord = FieldListRecord::deserialize(TypeRecordKind::FieldList,
                                                     FieldListRecordData);
  if (!ExpectedRecord)
    return ExpectedRecord.takeError();

  Record = *ExpectedRecord;
  ArrayRef<uint8_t> MemberData = Record.getFieldListData();

  while (!MemberData.empty()) {
    const ulittle16_t *LeafPtr;
    if (auto EC = takeObject(MemberData, LeafPtr))
      return EC;
    TypeLeafKind Leaf = TypeLeafKind(unsigned(*LeafPtr));
    switch (Leaf) {
    default:
      // Field list records do not describe their own length, so we cannot
      // continue parsing past a type that we don't know how to deserialize.
      if (auto EC = Recipient.visitUnknownMember(CVR))
        return EC;
      return llvm::make_error<CodeViewError>(
          cv_error_code::unknown_member_record);
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  case EnumName: {                                                             \
    TypeRecordKind RK = static_cast<TypeRecordKind>(Leaf);                     \
    Name##Record Member(RK);                                                   \
    if (auto EC = visitKnownMember(MemberData, Leaf, Member))                  \
      return EC;                                                               \
    break;                                                                     \
  }
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)                \
  MEMBER_RECORD(EnumVal, EnumVal, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"
    }
    if (auto EC = skipPadding(MemberData))
      return EC;
  }
  return Error::success();
}

Error TypeDeserializer::skipPadding(ArrayRef<uint8_t> &Data) {
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
