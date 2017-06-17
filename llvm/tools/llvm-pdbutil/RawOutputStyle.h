//===- RawOutputStyle.h -------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_RAWOUTPUTSTYLE_H
#define LLVM_TOOLS_LLVMPDBDUMP_RAWOUTPUTSTYLE_H

#include "LinePrinter.h"
#include "OutputStyle.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace llvm {
class BitVector;

namespace codeview {
class LazyRandomTypeCollection;
}

namespace pdb {
class RawOutputStyle : public OutputStyle {
public:
  RawOutputStyle(PDBFile &File);

  Error dump() override;

private:
  Expected<codeview::LazyRandomTypeCollection &> initializeTypes(uint32_t SN);

  Error dumpFileSummary();
  Error dumpStreamSummary();
  Error dumpBlockRanges();
  Error dumpStreamBytes();
  Error dumpStringTable();
  Error dumpLines();
  Error dumpInlineeLines();
  Error dumpXmi();
  Error dumpXme();
  Error dumpTpiStream(uint32_t StreamIdx);
  Error dumpModules();
  Error dumpModuleFiles();
  Error dumpModuleSyms();
  Error dumpPublics();
  Error dumpSectionContribs();
  Error dumpSectionMap();

  PDBFile &File;
  LinePrinter P;
  std::unique_ptr<codeview::LazyRandomTypeCollection> TpiTypes;
  std::unique_ptr<codeview::LazyRandomTypeCollection> IpiTypes;
  SmallVector<std::string, 32> StreamPurposes;
};
} // namespace pdb
} // namespace llvm

#endif
