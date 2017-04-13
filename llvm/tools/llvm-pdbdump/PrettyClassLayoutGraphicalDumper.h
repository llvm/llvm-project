//===- PrettyClassLayoutGraphicalDumper.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_PRETTYCLASSLAYOUTGRAPHICALDUMPER_H
#define LLVM_TOOLS_LLVMPDBDUMP_PRETTYCLASSLAYOUTGRAPHICALDUMPER_H

#include "llvm/ADT/BitVector.h"

#include "llvm/DebugInfo/PDB/PDBSymDumper.h"

namespace llvm {

namespace pdb {

class ClassLayout;
class LinePrinter;

class PrettyClassLayoutGraphicalDumper : public PDBSymDumper {
public:
  explicit PrettyClassLayoutGraphicalDumper(LinePrinter &P);

  bool start(const ClassLayout &Layout);

  void dump(const PDBSymbolTypeBaseClass &Symbol) override;
  void dump(const PDBSymbolData &Symbol) override;
  void dump(const PDBSymbolTypeEnum &Symbol) override;
  void dump(const PDBSymbolFunc &Symbol) override;
  void dump(const PDBSymbolTypeTypedef &Symbol) override;
  void dump(const PDBSymbolTypeUDT &Symbol) override;
  void dump(const PDBSymbolTypeVTable &Symbol) override;
};
}
}
#endif
