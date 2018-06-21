//===- Diff.h - PDB diff utility --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_DIFF_H
#define LLVM_TOOLS_LLVMPDBDUMP_DIFF_H

#include "OutputStyle.h"

namespace llvm {
namespace pdb {
class PDBFile;
class DiffStyle : public OutputStyle {
public:
  explicit DiffStyle(PDBFile &File1, PDBFile &File2);

  Error dump() override;

private:
  Error diffSuperBlock();
  Error diffStreamDirectory();
  Error diffStringTable();
  Error diffFreePageMap();
  Error diffInfoStream();
  Error diffDbiStream();
  Error diffSectionContribs();
  Error diffSectionMap();
  Error diffFpoStream();
  Error diffTpiStream(int Index);
  Error diffModuleInfoStream(int Index);
  Error diffPublics();
  Error diffGlobals();

  PDBFile &File1;
  PDBFile &File2;
};
}
}

#endif
