//===- LLVMOutputStyle.h -------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_LLVMOUTPUTSTYLE_H
#define LLVM_TOOLS_LLVMPDBDUMP_LLVMOUTPUTSTYLE_H

#include "OutputStyle.h"

#include "llvm/DebugInfo/CodeView/TypeDumper.h"
#include "llvm/Support/ScopedPrinter.h"

namespace llvm {
namespace pdb {
class LLVMOutputStyle : public OutputStyle {
public:
  LLVMOutputStyle(PDBFile &File);

  Error dump() override;

private:
  Error dumpFileHeaders();
  Error dumpStreamSummary();
  Error dumpStreamBlocks();
  Error dumpStreamData();
  Error dumpInfoStream();
  Error dumpNamedStream();
  Error dumpTpiStream(uint32_t StreamIdx);
  Error dumpDbiStream();
  Error dumpSectionContribs();
  Error dumpSectionMap();
  Error dumpPublicsStream();
  Error dumpSectionHeaders();
  Error dumpFpoStream();

  void flush();

  PDBFile &File;
  ScopedPrinter P;
  codeview::CVTypeDumper TD;
};
}
}

#endif
