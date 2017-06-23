//===- BytesOutputStyle.h ------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_BYTESOUTPUTSTYLE_H
#define LLVM_TOOLS_LLVMPDBDUMP_BYTESOUTPUTSTYLE_H

#include "LinePrinter.h"
#include "OutputStyle.h"

#include "llvm/Support/Error.h"

namespace llvm {

namespace pdb {

class PDBFile;

class BytesOutputStyle : public OutputStyle {
public:
  BytesOutputStyle(PDBFile &File);

  Error dump() override;

private:
  void dumpBlockRanges(uint32_t Min, uint32_t Max);
  void dumpStreamBytes();

  PDBFile &File;
  LinePrinter P;
  SmallVector<std::string, 8> StreamPurposes;
};
} // namespace pdb
} // namespace llvm

#endif
