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
#include "llvm/DebugInfo/MSF/ByteStream.h"

using namespace llvm;
using namespace llvm::codeview;

CVTypeVisitor::CVTypeVisitor(TypeVisitorCallbacks &Callbacks)
    : Callbacks(Callbacks) {}

template <typename T>
static Error visitKnownRecord(const CVRecord<TypeLeafKind> &Record,
                              TypeVisitorCallbacks &Callbacks) {
  TypeRecordKind RK = static_cast<TypeRecordKind>(Record.Type);
  T KnownRecord(RK);
  if (auto EC = Callbacks.visitKnownRecord(Record, KnownRecord))
    return EC;
  return Error::success();
}

Error CVTypeVisitor::visitTypeRecord(const CVRecord<TypeLeafKind> &Record) {
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
