//===- Analyze.h - PDB analysis functions -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_ANALYSIS_H
#define LLVM_TOOLS_LLVMPDBDUMP_ANALYSIS_H

#include "OutputStyle.h"

namespace llvm {
namespace pdb {
class PDBFile;
class AnalysisStyle : public OutputStyle {
public:
  explicit AnalysisStyle(PDBFile &File);

  Error dump() override;

private:
  PDBFile &File;
};
}
}

#endif
