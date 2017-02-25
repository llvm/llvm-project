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
#include "llvm/DebugInfo/CodeView/TypeDatabase.h"
#include "llvm/DebugInfo/CodeView/TypeDatabaseVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeRecordMapping.h"
#include "llvm/DebugInfo/CodeView/TypeServerHandler.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/MSF/BinaryByteStream.h"
#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"

using namespace llvm;
using namespace llvm::codeview;

CVTypeVisitor::CVTypeVisitor(TypeVisitorCallbacks &Callbacks)
    : Callbacks(Callbacks) {}

template <typename T>
static Error visitKnownRecord(CVTypeVisitor &Visitor, CVType &Record,
                              TypeVisitorCallbacks &Callbacks) {
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

static Expected<TypeServer2Record> deserializeTypeServerRecord(CVType &Record) {
  class StealTypeServerVisitor : public TypeVisitorCallbacks {
  public:
    explicit StealTypeServerVisitor(TypeServer2Record &TR) : TR(TR) {}

    Error visitKnownRecord(CVType &CVR, TypeServer2Record &Record) override {
      TR = Record;
      return Error::success();
    }

  private:
    TypeServer2Record &TR;
  };

  TypeServer2Record R(TypeRecordKind::TypeServer2);
  TypeDeserializer Deserializer;
  StealTypeServerVisitor Thief(R);
  TypeVisitorCallbackPipeline Pipeline;
  Pipeline.addCallbackToPipeline(Deserializer);
  Pipeline.addCallbackToPipeline(Thief);
  CVTypeVisitor Visitor(Pipeline);
  if (auto EC = Visitor.visitTypeRecord(Record))
    return std::move(EC);

  return R;
}

void CVTypeVisitor::addTypeServerHandler(TypeServerHandler &Handler) {
  Handlers.push_back(&Handler);
}

Error CVTypeVisitor::visitTypeRecord(CVType &Record) {
  if (Record.Type == TypeLeafKind::LF_TYPESERVER2 && !Handlers.empty()) {
    auto TS = deserializeTypeServerRecord(Record);
    if (!TS)
      return TS.takeError();

    for (auto Handler : Handlers) {
      auto ExpectedResult = Handler->handle(*TS, Callbacks);
      // If there was an error, return the error.
      if (!ExpectedResult)
        return ExpectedResult.takeError();

      // If the handler processed the record, return success.
      if (*ExpectedResult)
        return Error::success();

      // Otherwise keep searching for a handler, eventually falling out and
      // using the default record handler.
    }
  }

  if (auto EC = Callbacks.visitTypeBegin(Record))
    return EC;

  switch (Record.Type) {
  default:
    if (auto EC = Callbacks.visitUnknownType(Record))
      return EC;
    break;
#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  case EnumName: {                                                             \
    if (auto EC = visitKnownRecord<Name##Record>(*this, Record, Callbacks))    \
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

static Error visitMemberRecord(CVMemberRecord &Record,
                               TypeVisitorCallbacks &Callbacks) {
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

Error CVTypeVisitor::visitMemberRecord(CVMemberRecord &Record) {
  return ::visitMemberRecord(Record, Callbacks);
}

/// Visits the type records in Data. Sets the error flag on parse failures.
Error CVTypeVisitor::visitTypeStream(const CVTypeArray &Types) {
  for (auto I : Types) {
    if (auto EC = visitTypeRecord(I))
      return EC;
  }
  return Error::success();
}

Error CVTypeVisitor::visitTypeStream(CVTypeRange Types) {
  for (auto I : Types) {
    if (auto EC = visitTypeRecord(I))
      return EC;
  }
  return Error::success();
}

Error CVTypeVisitor::visitFieldListMemberStream(msf::StreamReader Reader) {
  FieldListDeserializer Deserializer(Reader);
  TypeVisitorCallbackPipeline Pipeline;
  Pipeline.addCallbackToPipeline(Deserializer);
  Pipeline.addCallbackToPipeline(Callbacks);

  TypeLeafKind Leaf;
  while (!Reader.empty()) {
    if (auto EC = Reader.readEnum(Leaf, llvm::support::little))
      return EC;

    CVMemberRecord Record;
    Record.Kind = Leaf;
    if (auto EC = ::visitMemberRecord(Record, Pipeline))
      return EC;
  }

  return Error::success();
}

Error CVTypeVisitor::visitFieldListMemberStream(ArrayRef<uint8_t> Data) {
  msf::ByteStream S(Data);
  msf::StreamReader SR(S);
  return visitFieldListMemberStream(SR);
}
