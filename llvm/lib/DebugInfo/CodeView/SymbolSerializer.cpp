//===- SymbolSerializer.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/SymbolSerializer.h"

using namespace llvm;
using namespace llvm::codeview;

SymbolSerializer::SymbolSerializer(BumpPtrAllocator &Allocator,
                                   CodeViewContainer Container)
    : Storage(Allocator), RecordBuffer(MaxRecordLength),
      Stream(RecordBuffer, llvm::support::little), Writer(Stream),
      Mapping(Writer, Container) {}

Error SymbolSerializer::visitSymbolBegin(CVSymbol &Record) {
  assert(!CurrentSymbol.hasValue() && "Already in a symbol mapping!");

  Writer.setOffset(0);

  if (auto EC = writeRecordPrefix(Record.kind()))
    return EC;

  CurrentSymbol = Record.kind();
  if (auto EC = Mapping.visitSymbolBegin(Record))
    return EC;

  return Error::success();
}

Error SymbolSerializer::visitSymbolEnd(CVSymbol &Record) {
  assert(CurrentSymbol.hasValue() && "Not in a symbol mapping!");

  if (auto EC = Mapping.visitSymbolEnd(Record))
    return EC;

  uint32_t RecordEnd = Writer.getOffset();
  uint16_t Length = RecordEnd - 2;
  Writer.setOffset(0);
  if (auto EC = Writer.writeInteger(Length))
    return EC;

  uint8_t *StableStorage = Storage.Allocate<uint8_t>(RecordEnd);
  ::memcpy(StableStorage, &RecordBuffer[0], RecordEnd);
  Record.RecordData = ArrayRef<uint8_t>(StableStorage, RecordEnd);
  CurrentSymbol.reset();

  return Error::success();
}
