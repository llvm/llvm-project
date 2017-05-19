//===-- TypeIndex.cpp - CodeView type index ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeIndex.h"

#include "llvm/DebugInfo/CodeView/TypeCollection.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::codeview;

void llvm::codeview::printTypeIndex(ScopedPrinter &Printer, StringRef FieldName,
                                    TypeIndex TI, TypeCollection &Types) {
  StringRef TypeName;
  if (!TI.isNoneType())
    TypeName = Types.getTypeName(TI);
  if (!TypeName.empty())
    Printer.printHex(FieldName, TypeName, TI.getIndex());
  else
    Printer.printHex(FieldName, TI.getIndex());
}
