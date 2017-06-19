//===- DbiModuleDescriptorBuilder.h - PDB module information ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_DBIMODULEDESCRIPTORBUILDER_H
#define LLVM_DEBUGINFO_PDB_RAW_DBIMODULEDESCRIPTORBUILDER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/DebugChecksumsSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugInlineeLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/DebugSubsectionRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class BinaryStreamWriter;

namespace codeview {
class DebugSubsectionRecordBuilder;
}

namespace msf {
class MSFBuilder;
struct MSFLayout;
}
namespace pdb {

class DbiModuleDescriptorBuilder {
  friend class DbiStreamBuilder;

public:
  DbiModuleDescriptorBuilder(StringRef ModuleName, uint32_t ModIndex,
                             msf::MSFBuilder &Msf);
  ~DbiModuleDescriptorBuilder();

  DbiModuleDescriptorBuilder(const DbiModuleDescriptorBuilder &) = delete;
  DbiModuleDescriptorBuilder &
  operator=(const DbiModuleDescriptorBuilder &) = delete;

  void setObjFileName(StringRef Name);
  void addSymbol(codeview::CVSymbol Symbol);

  void
  addDebugSubsection(std::shared_ptr<codeview::DebugSubsection> Subsection);

  void
  addDebugSubsection(const codeview::DebugSubsectionRecord &SubsectionContents);

  uint16_t getStreamIndex() const;
  StringRef getModuleName() const { return ModuleName; }
  StringRef getObjFileName() const { return ObjFileName; }

  unsigned getModuleIndex() const { return Layout.Mod; }

  ArrayRef<std::string> source_files() const {
    return makeArrayRef(SourceFiles);
  }

  uint32_t calculateSerializedLength() const;

  void finalize();
  Error finalizeMsfLayout();

  Error commit(BinaryStreamWriter &ModiWriter, const msf::MSFLayout &MsfLayout,
               WritableBinaryStreamRef MsfBuffer);

private:
  uint32_t calculateC13DebugInfoSize() const;

  void addSourceFile(StringRef Path);
  msf::MSFBuilder &MSF;

  uint32_t SymbolByteSize = 0;
  std::string ModuleName;
  std::string ObjFileName;
  std::vector<std::string> SourceFiles;
  std::vector<codeview::CVSymbol> Symbols;

  std::vector<std::unique_ptr<codeview::DebugSubsectionRecordBuilder>>
      C13Builders;

  ModuleInfoHeader Layout;
};

} // end namespace pdb

} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_DBIMODULEDESCRIPTORBUILDER_H
