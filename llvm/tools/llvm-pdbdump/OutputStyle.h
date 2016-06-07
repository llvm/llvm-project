//===- OutputStyle.h ------------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_OUTPUTSTYLE_H
#define LLVM_TOOLS_LLVMPDBDUMP_OUTPUTSTYLE_H

#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class PDBFile;

class OutputStyle {
public:
  virtual ~OutputStyle() {}
  virtual Error dumpFileHeaders() = 0;
  virtual Error dumpStreamSummary() = 0;
  virtual Error dumpStreamBlocks() = 0;
  virtual Error dumpStreamData() = 0;
  virtual Error dumpInfoStream() = 0;
  virtual Error dumpNamedStream() = 0;
  virtual Error dumpTpiStream(uint32_t StreamIdx) = 0;
  virtual Error dumpDbiStream() = 0;
  virtual Error dumpSectionContribs() = 0;
  virtual Error dumpSectionMap() = 0;
  virtual Error dumpPublicsStream() = 0;
  virtual Error dumpSectionHeaders() = 0;
  virtual Error dumpFpoStream() = 0;

  virtual void flush() = 0;
};
}
}

#endif
