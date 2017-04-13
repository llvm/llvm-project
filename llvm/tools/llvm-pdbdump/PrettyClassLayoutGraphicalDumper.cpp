//===- PrettyClassLayoutGraphicalDumper.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PrettyClassLayoutGraphicalDumper.h"

using namespace llvm;
using namespace llvm::pdb;

PrettyClassLayoutGraphicalDumper::PrettyClassLayoutGraphicalDumper(
    LinePrinter &P)
    : PDBSymDumper(true) {}

bool PrettyClassLayoutGraphicalDumper::start(const ClassLayout &Layout) {
  return false;
}

void PrettyClassLayoutGraphicalDumper::dump(
    const PDBSymbolTypeBaseClass &Symbol) {}

void PrettyClassLayoutGraphicalDumper::dump(const PDBSymbolData &Symbol) {}

void PrettyClassLayoutGraphicalDumper::dump(const PDBSymbolTypeEnum &Symbol) {}

void PrettyClassLayoutGraphicalDumper::dump(const PDBSymbolFunc &Symbol) {}

void PrettyClassLayoutGraphicalDumper::dump(
    const PDBSymbolTypeTypedef &Symbol) {}

void PrettyClassLayoutGraphicalDumper::dump(const PDBSymbolTypeUDT &Symbol) {}

void PrettyClassLayoutGraphicalDumper::dump(const PDBSymbolTypeVTable &Symbol) {
}
