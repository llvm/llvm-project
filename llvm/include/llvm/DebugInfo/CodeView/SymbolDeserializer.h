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
#include "llvm/DebugInfo/CodeView/SymbolRecordMapping.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorDelegate.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {
class SymbolVisitorDelegate;
class SymbolDeserializer : public SymbolVisitorCallbacks {
  struct MappingInfo {
    explicit MappingInfo(ArrayRef<uint8_t> RecordData)
        : Stream(RecordData, llvm::support::little), Reader(Stream),
          Mapping(Reader) {}

    BinaryByteStream Stream;
    BinaryStreamReader Reader;
    SymbolRecordMapping Mapping;
  };

public:
  template <typename T> static Error deserializeAs(CVSymbol Symbol, T &Record) {
    SymbolDeserializer S(nullptr);
    if (auto EC = S.visitSymbolBegin(Symbol))
      return EC;
    if (auto EC = S.visitKnownRecord(Symbol, Record))
      return EC;
    if (auto EC = S.visitSymbolEnd(Symbol))
      return EC;
    return Error::success();
  }

  explicit SymbolDeserializer(SymbolVisitorDelegate *Delegate)
      : Delegate(Delegate) {}

  Error visitSymbolBegin(CVSymbol &Record) override {
    assert(!Mapping && "Already in a symbol mapping!");
    Mapping = llvm::make_unique<MappingInfo>(Record.content());
    return Mapping->Mapping.visitSymbolBegin(Record);
  }
  Error visitSymbolEnd(CVSymbol &Record) override {
    assert(Mapping && "Not in a symbol mapping!");
    auto EC = Mapping->Mapping.visitSymbolEnd(Record);
    Mapping.reset();
    return EC;
  }

#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownRecord(CVSymbol &CVR, Name &Record) override {               \
    return visitKnownRecordImpl(CVR, Record);                                  \
  }
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CodeViewSymbols.def"

private:
  template <typename T> Error visitKnownRecordImpl(CVSymbol &CVR, T &Record) {

    Record.RecordOffset =
        Delegate ? Delegate->getRecordOffset(Mapping->Reader) : 0;
    if (auto EC = Mapping->Mapping.visitKnownRecord(CVR, Record))
      return EC;
    return Error::success();
  }

  SymbolVisitorDelegate *Delegate;
  std::unique_ptr<MappingInfo> Mapping;
};
}
}

#endif
