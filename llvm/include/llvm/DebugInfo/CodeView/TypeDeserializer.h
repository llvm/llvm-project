//===- TypeDeserializer.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPEDESERIALIZER_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPEDESERIALIZER_H

#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {
class TypeDeserializer : public TypeVisitorCallbacks {
public:
  TypeDeserializer() {}

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  Error visitKnownRecord(CVType &CVR, Name##Record &Record) override {         \
    return defaultVisitKnownRecord(CVR, Record);                               \
  }
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownMember(CVMemberRecord &CVR, Name##Record &Record) override { \
    return defaultVisitKnownMember(CVR, Record);                               \
  }
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "TypeRecords.def"

protected:

  template <typename T>
  Error deserializeRecord(ArrayRef<uint8_t> &Data, TypeLeafKind Kind,
                          T &Record) const {
    TypeRecordKind RK = static_cast<TypeRecordKind>(Kind);
    auto ExpectedRecord = T::deserialize(RK, Data);
    if (!ExpectedRecord)
      return ExpectedRecord.takeError();
    Record = std::move(*ExpectedRecord);
    return Error::success();
  }

private:
  template <typename T> Error defaultVisitKnownRecord(CVType &CVR, T &Record) {
    ArrayRef<uint8_t> RD = CVR.content();
    if (auto EC = deserializeRecord(RD, CVR.Type, Record))
      return EC;
    return Error::success();
  }
  template <typename T>
  Error defaultVisitKnownMember(CVMemberRecord &CVMR, T &Record) {
    ArrayRef<uint8_t> RD = CVMR.Data;
    if (auto EC = deserializeRecord(RD, CVMR.Kind, Record))
      return EC;
    return Error::success();
  }
};
}
}

#endif
