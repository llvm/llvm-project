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

#include "llvm/DebugInfo/CodeView/CVTypeDumper.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace pdb {
class ModStream;

class YAMLOutputStyle : public OutputStyle {
public:
  YAMLOutputStyle(PDBFile &File);

  Error dump() override;

private:
  Expected<Optional<llvm::pdb::yaml::PdbSourceFileInfo>>
  getFileLineInfo(const pdb::ModStream &ModS);

  Error dumpStringTable();
  Error dumpFileHeaders();
  Error dumpStreamMetadata();
  Error dumpStreamDirectory();
  Error dumpPDBStream();
  Error dumpDbiStream();
  Error dumpTpiStream();
  Error dumpIpiStream();

  void flush();

  PDBFile &File;
  llvm::yaml::Output Out;

  yaml::PdbObject Obj;
};
} // namespace pdb
} // namespace llvm

#endif // LLVM_TOOLS_LLVMPDBDUMP_YAMLOUTPUTSTYLE_H
