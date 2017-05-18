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
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"

using namespace llvm;
using namespace llvm::codeview;

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
  StealTypeServerVisitor Thief(R);
  if (auto EC = visitTypeRecord(Record, Thief))
    return std::move(EC);

  return R;
}

void CVTypeVisitor::addTypeServerHandler(TypeServerHandler &Handler) {
  Handlers.push_back(&Handler);
}

Expected<bool> CVTypeVisitor::handleTypeServer(CVType &Record) {
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
        return true;

      // Otherwise keep searching for a handler, eventually falling out and
      // using the default record handler.
    }
  }
  return false;
}

Error CVTypeVisitor::finishVisitation(CVType &Record) {
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

Error CVTypeVisitor::visitTypeRecord(CVType &Record, TypeIndex Index) {
  auto ExpectedResult = handleTypeServer(Record);
  if (!ExpectedResult)
    return ExpectedResult.takeError();
  if (*ExpectedResult)
    return Error::success();

  if (auto EC = Callbacks.visitTypeBegin(Record, Index))
    return EC;

  return finishVisitation(Record);
}

Error CVTypeVisitor::visitTypeRecord(CVType &Record) {
  auto ExpectedResult = handleTypeServer(Record);
  if (!ExpectedResult)
    return ExpectedResult.takeError();
  if (*ExpectedResult)
    return Error::success();

  if (auto EC = Callbacks.visitTypeBegin(Record))
    return EC;

  return finishVisitation(Record);
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

Error CVTypeVisitor::visitMemberRecord(CVMemberRecord Record) {
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

Error CVTypeVisitor::visitFieldListMemberStream(BinaryStreamReader Reader) {
  FieldListDeserializer Deserializer(Reader);
  TypeVisitorCallbackPipeline Pipeline;
  Pipeline.addCallbackToPipeline(Deserializer);
  Pipeline.addCallbackToPipeline(Callbacks);

  TypeLeafKind Leaf;
  while (!Reader.empty()) {
    if (auto EC = Reader.readEnum(Leaf))
      return EC;

    CVMemberRecord Record;
    Record.Kind = Leaf;
    if (auto EC = ::visitMemberRecord(Record, Pipeline))
      return EC;
  }

  return Error::success();
}

Error CVTypeVisitor::visitFieldListMemberStream(ArrayRef<uint8_t> Data) {
  BinaryByteStream S(Data, llvm::support::little);
  BinaryStreamReader SR(S);
  return visitFieldListMemberStream(SR);
}

namespace {
struct FieldListVisitHelper {
  FieldListVisitHelper(TypeVisitorCallbacks &Callbacks, ArrayRef<uint8_t> Data,
                       VisitorDataSource Source)
      : Stream(Data, llvm::support::little), Reader(Stream),
        Deserializer(Reader),
        Visitor((Source == VDS_BytesPresent) ? Pipeline : Callbacks) {
    if (Source == VDS_BytesPresent) {
      Pipeline.addCallbackToPipeline(Deserializer);
      Pipeline.addCallbackToPipeline(Callbacks);
    }
  }

  BinaryByteStream Stream;
  BinaryStreamReader Reader;
  FieldListDeserializer Deserializer;
  TypeVisitorCallbackPipeline Pipeline;
  CVTypeVisitor Visitor;
};

struct VisitHelper {
  VisitHelper(TypeVisitorCallbacks &Callbacks, VisitorDataSource Source,
              TypeServerHandler *TS)
      : Visitor((Source == VDS_BytesPresent) ? Pipeline : Callbacks) {
    if (TS)
      Visitor.addTypeServerHandler(*TS);
    if (Source == VDS_BytesPresent) {
      Pipeline.addCallbackToPipeline(Deserializer);
      Pipeline.addCallbackToPipeline(Callbacks);
    }
  }

  TypeDeserializer Deserializer;
  TypeVisitorCallbackPipeline Pipeline;
  CVTypeVisitor Visitor;
};
}

Error llvm::codeview::visitTypeRecord(CVType &Record, TypeIndex Index,
                                      TypeVisitorCallbacks &Callbacks,
                                      VisitorDataSource Source,
                                      TypeServerHandler *TS) {
  VisitHelper Helper(Callbacks, Source, TS);
  return Helper.Visitor.visitTypeRecord(Record, Index);
}

Error llvm::codeview::visitTypeRecord(CVType &Record,
                                      TypeVisitorCallbacks &Callbacks,
                                      VisitorDataSource Source,
                                      TypeServerHandler *TS) {
  VisitHelper Helper(Callbacks, Source, TS);
  return Helper.Visitor.visitTypeRecord(Record);
}

Error llvm::codeview::visitMemberRecordStream(ArrayRef<uint8_t> FieldList,
                                              TypeVisitorCallbacks &Callbacks) {
  CVTypeVisitor Visitor(Callbacks);
  return Visitor.visitFieldListMemberStream(FieldList);
}

Error llvm::codeview::visitMemberRecord(CVMemberRecord Record,
                                        TypeVisitorCallbacks &Callbacks,
                                        VisitorDataSource Source) {
  FieldListVisitHelper Helper(Callbacks, Record.Data, Source);
  return Helper.Visitor.visitMemberRecord(Record);
}

Error llvm::codeview::visitMemberRecord(TypeLeafKind Kind,
                                        ArrayRef<uint8_t> Record,
                                        TypeVisitorCallbacks &Callbacks) {
  CVMemberRecord R;
  R.Data = Record;
  R.Kind = Kind;
  return visitMemberRecord(R, Callbacks, VDS_BytesPresent);
}

Error llvm::codeview::visitTypeStream(const CVTypeArray &Types,
                                      TypeVisitorCallbacks &Callbacks,
                                      TypeServerHandler *TS) {
  VisitHelper Helper(Callbacks, VDS_BytesPresent, TS);
  return Helper.Visitor.visitTypeStream(Types);
}

Error llvm::codeview::visitTypeStream(CVTypeRange Types,
                                      TypeVisitorCallbacks &Callbacks,
                                      TypeServerHandler *TS) {
  VisitHelper Helper(Callbacks, VDS_BytesPresent, TS);
  return Helper.Visitor.visitTypeStream(Types);
}
