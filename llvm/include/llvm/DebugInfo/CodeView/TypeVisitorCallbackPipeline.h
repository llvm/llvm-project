//===- TypeVisitorCallbackPipeline.h -------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPEVISITORCALLBACKPIPELINE_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPEVISITORCALLBACKPIPELINE_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"

#include <vector>

namespace llvm {
namespace codeview {
class TypeVisitorCallbackPipeline : public TypeVisitorCallbacks {
public:
  TypeVisitorCallbackPipeline() {}

  virtual Error
  visitUnknownType(const CVRecord<TypeLeafKind> &Record) override {
    for (auto Visitor : Pipeline) {
      if (auto EC = Visitor->visitUnknownType(Record))
        return EC;
    }
    return Error::success();
  }

  virtual Error
  visitUnknownMember(const CVRecord<TypeLeafKind> &Record) override {
    for (auto Visitor : Pipeline) {
      if (auto EC = Visitor->visitUnknownMember(Record))
        return EC;
    }
    return Error::success();
  }

  virtual Expected<TypeLeafKind>
  visitTypeBegin(const CVRecord<TypeLeafKind> &Record) override {
    // An implementation can calculate of visitTypeBegin() can calculate the
    // kind based on an arbitrary factor, including the Type that is already
    // specified in the Record.  So, as we go through the pipeline invoking
    // each visitor, update the state in a copy of the record so that each
    // visitor in the pipeline sees the most recently value of the type.
    CVRecord<TypeLeafKind> RecordCopy = Record;
    for (auto Visitor : Pipeline) {
      if (auto ExpectedKind = Visitor->visitTypeBegin(RecordCopy)) {
        RecordCopy.Type = *ExpectedKind;
      } else
        return ExpectedKind.takeError();
    }
    return RecordCopy.Type;
  }
  virtual Error visitTypeEnd(const CVRecord<TypeLeafKind> &Record) override {
    for (auto Visitor : Pipeline) {
      if (auto EC = Visitor->visitTypeEnd(Record))
        return EC;
    }
    return Error::success();
  }

  void addCallbackToPipeline(TypeVisitorCallbacks &Callbacks) {
    Pipeline.push_back(&Callbacks);
  }

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  Error visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,                    \
                         Name##Record &Record) override {                      \
    for (auto Visitor : Pipeline) {                                            \
      if (auto EC = Visitor->visitKnownRecord(CVR, Record))                    \
        return EC;                                                             \
    }                                                                          \
    return Error::success();                                                   \
  }
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  TYPE_RECORD(EnumName, EnumVal, Name)
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

private:
  std::vector<TypeVisitorCallbacks *> Pipeline;
};
}
}

#endif
