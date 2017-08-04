//===-- COFFImportDumper.cpp - COFF import library dumper -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the COFF import library dumper for llvm-readobj.
///
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "ObjDumper.h"
#include "llvm-readobj.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/COFFImportFile.h"

using namespace llvm::object;

namespace llvm {

void dumpCOFFImportFile(const COFFImportFile *File) {
  outs() << '\n';
  outs() << "File: " << File->getFileName() << "\n";
  outs() << "Format: COFF-import-file\n";

  const coff_import_header *H = File->getCOFFImportHeader();
  switch (H->getType()) {
  case COFF::IMPORT_CODE:  outs() << "Type: code\n"; break;
  case COFF::IMPORT_DATA:  outs() << "Type: data\n"; break;
  case COFF::IMPORT_CONST: outs() << "Type: const\n"; break;
  }

  switch (H->getNameType()) {
  case COFF::IMPORT_ORDINAL: outs() << "Name type: ordinal\n"; break;
  case COFF::IMPORT_NAME: outs() << "Name type: name\n"; break;
  case COFF::IMPORT_NAME_NOPREFIX: outs() << "Name type: noprefix\n"; break;
  case COFF::IMPORT_NAME_UNDECORATE: outs() << "Name type: undecorate\n"; break;
  }

  for (const object::BasicSymbolRef &Sym : File->symbols()) {
    outs() << "Symbol: ";
    Sym.printName(outs());
    outs() << "\n";
  }
}

} // namespace llvm
