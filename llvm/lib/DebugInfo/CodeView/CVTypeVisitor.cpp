//===- CVTypeVisitor.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"

#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/MSF/ByteStream.h"

using namespace llvm;
using namespace llvm::codeview;

template <typename T>
static Expected<CVMemberRecord>
deserializeMemberRecord(FieldListDeserializer &Deserializer,
                        msf::StreamReader &Reader, TypeLeafKind Kind) {
  T MR(static_cast<TypeRecordKind>(Kind));
  CVMemberRecord CVR;
  CVR.Kind = Kind;

  if (auto EC = Deserializer.visitMemberBegin(CVR))
    return std::move(EC);
  if (auto EC = Deserializer.visitKnownMember(CVR, MR))
    return std::move(EC);
  if (auto EC = Deserializer.visitMemberEnd(CVR))
    return std::move(EC);

  return CVR;
}

CVTypeVisitor::CVTypeVisitor(TypeVisitorCallbacks &Callbacks)
    : Callbacks(Callbacks) {}

template <typename T>
static Error visitKnownRecord(CVType &Record, TypeVisitorCallbacks &Callbacks) {
  TypeRecordKind RK = static_cast<TypeRecordKind>(Record.Type);
  T KnownRecord(RK);
  if (auto EC = Callbacks.visitKnownRecord(Record, KnownRecord))
    return EC;
  return Error::success();
}

template <typename T>
static Error visitKnownMember(CVMemberRecord &Record,
                              TypeVisitorCallbacks &Callbacks) {
  TypeRecordKind RK = static_cast<TypeRecordKind>(Record.Kind);
  T KnownRecord(RK);
  if (auto EC = Callbacks.visitKnownMember(Record, KnownRecord))
    return EC;
  return Error::success();
}

Error CVTypeVisitor::visitTypeRecord(CVType &Record) {
  if (auto EC = Callbacks.visitTypeBegin(Record))
    return EC;

  switch (Record.Type) {
  default:
    if (auto EC = Callbacks.visitUnknownType(Record))
      return EC;
    break;
#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  case EnumName: {                                                             \
    if (auto EC = visitKnownRecord<Name##Record>(Record, Callbacks))           \
      return EC;                                                               \
    break;                                                                     \
  }
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)                  \
  TYPE_RECORD(EnumVal, EnumVal, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"
  }

  if (auto EC = Callbacks.visitTypeEnd(Record))
    return EC;

  return Error::success();
}

Error CVTypeVisitor::visitMemberRecord(CVMemberRecord &Record) {
  if (auto EC = Callbacks.visitMemberBegin(Record))
    return EC;

  switch (Record.Kind) {
  default:
    if (auto EC = Callbacks.visitUnknownMember(Record))
      return EC;
    break;
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  case EnumName: {                                                             \
    if (auto EC = visitKnownMember<Name##Record>(Record, Callbacks))           \
      return EC;                                                               \
    break;                                                                     \
  }
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)                \
  MEMBER_RECORD(EnumVal, EnumVal, AliasName)
#define TYPE_RECORD(EnumName, EnumVal, Name)
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"
  }

  if (auto EC = Callbacks.visitMemberEnd(Record))
    return EC;

  return Error::success();
}

/// Visits the type records in Data. Sets the error flag on parse failures.
Error CVTypeVisitor::visitTypeStream(const CVTypeArray &Types) {
  for (auto I : Types) {
    if (auto EC = visitTypeRecord(I))
      return EC;
  }
  return Error::success();
}

template <typename MR>
static Error visitKnownMember(FieldListDeserializer &Deserializer,
                              msf::StreamReader &Reader, TypeLeafKind Leaf,
                              TypeVisitorCallbacks &Callbacks) {
  MR Record(static_cast<TypeRecordKind>(Leaf));
  CVMemberRecord CVR;
  CVR.Kind = Leaf;

  if (auto EC = Callbacks.visitMemberBegin(CVR))
    return EC;
  if (auto EC = Callbacks.visitKnownMember(CVR, Record))
    return EC;
  if (auto EC = Callbacks.visitMemberEnd(CVR))
    return EC;
  return Error::success();
}

Error CVTypeVisitor::visitFieldListMemberStream(msf::StreamReader Reader) {
  FieldListDeserializer Deserializer(Reader);
  TypeVisitorCallbackPipeline Pipeline;
  Pipeline.addCallbackToPipeline(Deserializer);
  Pipeline.addCallbackToPipeline(Callbacks);

  TypeLeafKind Leaf;
  while (!Reader.empty()) {
    if (auto EC = Reader.readEnum(Leaf))
      return EC;

    CVType Record;
    switch (Leaf) {
    default:
      // Field list records do not describe their own length, so we cannot
      // continue parsing past a type that we don't know how to deserialize.
      return llvm::make_error<CodeViewError>(
          cv_error_code::unknown_member_record);
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  case EnumName: {                                                             \
    if (auto EC = visitKnownMember<Name##Record>(Deserializer, Reader, Leaf,   \
                                                 Pipeline))                    \
      return EC;                                                               \
    break;                                                                     \
  }
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)                \
  MEMBER_RECORD(EnumVal, EnumVal, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"
    }
  }
  return Error::success();
}

Error CVTypeVisitor::visitFieldListMemberStream(ArrayRef<uint8_t> Data) {
  msf::ByteStream S(Data);
  msf::StreamReader SR(S);
  return visitFieldListMemberStream(SR);
}
