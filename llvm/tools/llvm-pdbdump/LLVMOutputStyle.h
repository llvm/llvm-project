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
class BitVector;
namespace pdb {
class LLVMOutputStyle : public OutputStyle {
public:
  LLVMOutputStyle(PDBFile &File);

  Error dump() override;

private:
  void discoverStreamPurposes();

  Error dumpFileHeaders();
  Error dumpStreamSummary();
  Error dumpFreePageMap();
  Error dumpBlockRanges();
  Error dumpStreamBytes();
  Error dumpStreamBlocks();
  Error dumpInfoStream();
  Error dumpTpiStream(uint32_t StreamIdx);
  Error dumpDbiStream();
  Error dumpSectionContribs();
  Error dumpSectionMap();
  Error dumpPublicsStream();
  Error dumpSectionHeaders();
  Error dumpFpoStream();

  void dumpBitVector(StringRef Name, const BitVector &V);

  void flush();

  PDBFile &File;
  ScopedPrinter P;
  codeview::CVTypeDumper Dumper;
  std::vector<std::string> StreamPurposes;
};
}
}

#endif
