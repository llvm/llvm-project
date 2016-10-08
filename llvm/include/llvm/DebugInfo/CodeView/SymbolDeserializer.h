//===- SymbolDeserializer.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_SYMBOLDESERIALIZER_H
#define LLVM_DEBUGINFO_CODEVIEW_SYMBOLDESERIALIZER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorDelegate.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {
class SymbolVisitorDelegate;
class SymbolDeserializer : public SymbolVisitorCallbacks {
public:
  explicit SymbolDeserializer(SymbolVisitorDelegate *Delegate)
      : Delegate(Delegate) {}

#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownRecord(CVSymbol &CVR, Name &Record) override {               \
    return defaultVisitKnownRecord(CVR, Record);                               \
  }
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "CVSymbolTypes.def"

protected:
  template <typename T>
  Error deserializeRecord(ArrayRef<uint8_t> &Data, SymbolKind Kind,
                          T &Record) const {
    uint32_t RecordOffset = Delegate ? Delegate->getRecordOffset(Data) : 0;
    SymbolRecordKind RK = static_cast<SymbolRecordKind>(Kind);
    auto ExpectedRecord = T::deserialize(RK, RecordOffset, Data);
    if (!ExpectedRecord)
      return ExpectedRecord.takeError();
    Record = std::move(*ExpectedRecord);
    return Error::success();
  }

private:
  template <typename T>
  Error defaultVisitKnownRecord(CVSymbol &CVR, T &Record) {
    ArrayRef<uint8_t> RD = CVR.content();
    if (auto EC = deserializeRecord(RD, CVR.Type, Record))
      return EC;
    return Error::success();
  }

  SymbolVisitorDelegate *Delegate;
};
}
}

#endif
