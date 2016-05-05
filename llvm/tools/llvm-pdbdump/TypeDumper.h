//===- TypeDumper.h - PDBSymDumper implementation for types *- C++ ------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_TYPEDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_TYPEDUMPER_H

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

namespace llvm {
namespace pdb {
class LinePrinter;

class TypeDumper : public PDBSymDumper {
public:
  TypeDumper(LinePrinter &P);

  void start(const PDBSymbolExe &Exe);

  void dump(const PDBSymbolTypeEnum &Symbol) override;
  void dump(const PDBSymbolTypeTypedef &Symbol) override;
  void dump(const PDBSymbolTypeUDT &Symbol) override;

private:
  LinePrinter &Printer;
};
}
}
#endif
