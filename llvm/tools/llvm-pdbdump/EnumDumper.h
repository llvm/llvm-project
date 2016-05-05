//===- EnumDumper.h - -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_ENUMDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_ENUMDUMPER_H

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

namespace llvm {
namespace pdb {

class LinePrinter;

class EnumDumper : public PDBSymDumper {
public:
  EnumDumper(LinePrinter &P);

  void start(const PDBSymbolTypeEnum &Symbol);

private:
  LinePrinter &Printer;
};
}
}
#endif
