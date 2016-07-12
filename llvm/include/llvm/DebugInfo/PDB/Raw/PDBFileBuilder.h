//===- PDBFileBuilder.h - PDB File Creation ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBFILEBUILDER_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBFILEBUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"

#include <memory>
#include <vector>

namespace llvm {
namespace codeview {
class StreamInterface;
}
namespace pdb {
class DbiStreamBuilder;
class InfoStreamBuilder;
class PDBFile;

class PDBFileBuilder {
public:
  explicit PDBFileBuilder(
      std::unique_ptr<codeview::StreamInterface> PdbFileBuffer);
  PDBFileBuilder(const PDBFileBuilder &) = delete;
  PDBFileBuilder &operator=(const PDBFileBuilder &) = delete;

  Error setSuperBlock(const PDBFile::SuperBlock &B);
  void setStreamSizes(ArrayRef<support::ulittle32_t> S);
  void setDirectoryBlocks(ArrayRef<support::ulittle32_t> D);
  void setStreamMap(const std::vector<ArrayRef<support::ulittle32_t>> &S);
  Error generateSimpleStreamMap();

  InfoStreamBuilder &getInfoBuilder();
  DbiStreamBuilder &getDbiBuilder();

  Expected<std::unique_ptr<PDBFile>> build();

private:
  std::unique_ptr<codeview::StreamInterface> PdbFileBuffer;
  std::unique_ptr<InfoStreamBuilder> Info;
  std::unique_ptr<DbiStreamBuilder> Dbi;

  std::unique_ptr<PDBFile> File;
};
}
}

#endif
