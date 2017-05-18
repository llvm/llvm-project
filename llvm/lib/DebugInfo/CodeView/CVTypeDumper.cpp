//===-- CVTypeDumper.cpp - CodeView type info dumper ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/CVTypeDumper.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeDatabase.h"
#include "llvm/DebugInfo/CodeView/TypeDatabaseVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbackPipeline.h"
#include "llvm/Support/BinaryByteStream.h"

using namespace llvm;
using namespace llvm::codeview;

Error CVTypeDumper::dump(const CVType &Record, TypeVisitorCallbacks &Dumper) {
  TypeDatabaseVisitor DBV(TypeDB);
  TypeVisitorCallbackPipeline Pipeline;
  Pipeline.addCallbackToPipeline(DBV);
  Pipeline.addCallbackToPipeline(Dumper);

  CVType RecordCopy = Record;
  return codeview::visitTypeRecord(RecordCopy, Pipeline, VDS_BytesPresent,
                                   Handler);
}

Error CVTypeDumper::dump(const CVTypeArray &Types,
                         TypeVisitorCallbacks &Dumper) {
  TypeDatabaseVisitor DBV(TypeDB);
  TypeVisitorCallbackPipeline Pipeline;
  Pipeline.addCallbackToPipeline(DBV);
  Pipeline.addCallbackToPipeline(Dumper);

  return codeview::visitTypeStream(Types, Pipeline, Handler);
}

Error CVTypeDumper::dump(ArrayRef<uint8_t> Data, TypeVisitorCallbacks &Dumper) {
  BinaryByteStream Stream(Data, llvm::support::little);
  CVTypeArray Types;
  BinaryStreamReader Reader(Stream);
  if (auto EC = Reader.readArray(Types, Reader.getLength()))
    return EC;

  return dump(Types, Dumper);
}

void CVTypeDumper::printTypeIndex(ScopedPrinter &Printer, StringRef FieldName,
                                  TypeIndex TI, TypeDatabase &DB) {
  StringRef TypeName;
  if (!TI.isNoneType())
    TypeName = DB.getTypeName(TI);
  if (!TypeName.empty())
    Printer.printHex(FieldName, TypeName, TI.getIndex());
  else
    Printer.printHex(FieldName, TI.getIndex());
}
