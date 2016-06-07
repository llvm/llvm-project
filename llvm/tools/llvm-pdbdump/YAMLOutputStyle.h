//===- YAMLOutputStyle.h -------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_YAMLOUTPUTSTYLE_H
#define LLVM_TOOLS_LLVMPDBDUMP_YAMLOUTPUTSTYLE_H

#include "OutputStyle.h"
#include "PdbYaml.h"

#include "llvm/DebugInfo/CodeView/TypeDumper.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace pdb {
class YAMLOutputStyle : public OutputStyle {
public:
  YAMLOutputStyle(PDBFile &File);

  Error dumpFileHeaders() override;
  Error dumpStreamSummary() override;
  Error dumpStreamBlocks() override;
  Error dumpStreamData() override;
  Error dumpInfoStream() override;
  Error dumpNamedStream() override;
  Error dumpTpiStream(uint32_t StreamIdx) override;
  Error dumpDbiStream() override;
  Error dumpSectionContribs() override;
  Error dumpSectionMap() override;
  Error dumpPublicsStream() override;
  Error dumpSectionHeaders() override;
  Error dumpFpoStream() override;

  void flush() override;

private:
  PDBFile &File;
  llvm::yaml::Output Out;

  yaml::PdbObject Obj;
};
} // namespace pdb
} // namespace llvm

#endif // LLVM_TOOLS_LLVMPDBDUMP_YAMLOUTPUTSTYLE_H
