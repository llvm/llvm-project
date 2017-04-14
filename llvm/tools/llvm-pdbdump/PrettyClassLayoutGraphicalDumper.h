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

class UDTLayoutBase;
class StorageItemBase;
class LinePrinter;

class PrettyClassLayoutGraphicalDumper : public PDBSymDumper {
public:
  PrettyClassLayoutGraphicalDumper(LinePrinter &P, uint32_t InitialOffset);

  bool start(const UDTLayoutBase &Layout);

  void dump(const PDBSymbolTypeBaseClass &Symbol) override;
  void dump(const PDBSymbolData &Symbol) override;
  void dump(const PDBSymbolTypeVTable &Symbol) override;

private:
  void printPaddingRow(uint32_t Amount);

  LinePrinter &Printer;

  StorageItemBase *CurrentItem = nullptr;
  uint32_t ClassOffsetZero = 0;
  uint32_t CurrentAbsoluteOffset = 0;
  bool DumpedAnything = false;
};
}
}
#endif
