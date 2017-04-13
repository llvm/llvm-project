//===- PrettyClassLayoutTextDumper.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_PRETTYCLASSLAYOUTTEXTDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_PRETTYCLASSLAYOUTTEXTDUMPER_H

#include "llvm/ADT/BitVector.h"

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

namespace llvm {

namespace pdb {

class ClassLayout;
class LinePrinter;

class PrettyClassLayoutTextDumper : public PDBSymDumper {
public:
  PrettyClassLayoutTextDumper(LinePrinter &P);

  bool start(const ClassLayout &Layout);

  void dump(const PDBSymbolTypeBaseClass &Symbol) override;
  void dump(const PDBSymbolData &Symbol) override;
  void dump(const PDBSymbolTypeEnum &Symbol) override;
  void dump(const PDBSymbolFunc &Symbol) override;
  void dump(const PDBSymbolTypeTypedef &Symbol) override;
  void dump(const PDBSymbolTypeUDT &Symbol) override;
  void dump(const PDBSymbolTypeVTable &Symbol) override;

private:
  bool DumpedAnything = false;
  LinePrinter &Printer;
};
}
}
#endif
