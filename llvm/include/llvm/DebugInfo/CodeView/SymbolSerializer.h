//===- symbolSerializer.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_SYMBOLSERIALIZER_H
#define LLVM_DEBUGINFO_CODEVIEW_SYMBOLSERIALIZER_H

#include "llvm/DebugInfo/CodeView/SymbolRecordMapping.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/DebugInfo/MSF/BinaryByteStream.h"
#include "llvm/DebugInfo/MSF/BinaryStreamWriter.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"

namespace llvm {
class BinaryStreamWriter;
namespace codeview {

class SymbolSerializer : public SymbolVisitorCallbacks {
  uint32_t RecordStart = 0;
  BinaryStreamWriter &Writer;
  SymbolRecordMapping Mapping;
  Optional<SymbolKind> CurrentSymbol;

  Error writeRecordPrefix(SymbolKind Kind) {
    RecordPrefix Prefix;
    Prefix.RecordKind = Kind;
    Prefix.RecordLen = 0;
    if (auto EC = Writer.writeObject(Prefix))
      return EC;
    return Error::success();
  }

public:
  explicit SymbolSerializer(BinaryStreamWriter &Writer)
      : Writer(Writer), Mapping(Writer) {}

  virtual Error visitSymbolBegin(CVSymbol &Record) override {
    assert(!CurrentSymbol.hasValue() && "Already in a symbol mapping!");

    RecordStart = Writer.getOffset();
    if (auto EC = writeRecordPrefix(Record.kind()))
      return EC;

    CurrentSymbol = Record.kind();
    if (auto EC = Mapping.visitSymbolBegin(Record))
      return EC;

    return Error::success();
  }

  virtual Error visitSymbolEnd(CVSymbol &Record) override {
    assert(CurrentSymbol.hasValue() && "Not in a symbol mapping!");

    if (auto EC = Mapping.visitSymbolEnd(Record))
      return EC;

    uint32_t RecordEnd = Writer.getOffset();
    Writer.setOffset(RecordStart);
    uint16_t Length = RecordEnd - Writer.getOffset() - 2;
    if (auto EC = Writer.writeInteger(Length))
      return EC;

    Writer.setOffset(RecordEnd);
    CurrentSymbol.reset();

    return Error::success();
  }

#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  virtual Error visitKnownRecord(CVSymbol &CVR, Name &Record) override {       \
    return visitKnownRecordImpl(CVR, Record);                                  \
  }
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "CVSymbolTypes.def"

private:
  template <typename RecordKind>
  Error visitKnownRecordImpl(CVSymbol &CVR, RecordKind &Record) {
    return Mapping.visitKnownRecord(CVR, Record);
  }
};
}
}

#endif
