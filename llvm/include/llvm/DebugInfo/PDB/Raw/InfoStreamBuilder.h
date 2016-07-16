//===- InfoStreamBuilder.h - PDB Info Stream Creation -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBINFOSTREAMBUILDER_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBINFOSTREAMBUILDER_H

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"

#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/NameMapBuilder.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"

namespace llvm {
namespace pdb {
class PDBFile;

class InfoStreamBuilder {
public:
  InfoStreamBuilder();
  InfoStreamBuilder(const InfoStreamBuilder &) = delete;
  InfoStreamBuilder &operator=(const InfoStreamBuilder &) = delete;

  void setVersion(PdbRaw_ImplVer V);
  void setSignature(uint32_t S);
  void setAge(uint32_t A);
  void setGuid(PDB_UniqueId G);

  NameMapBuilder &getNamedStreamsBuilder();

  uint32_t calculateSerializedLength() const;

  Expected<std::unique_ptr<InfoStream>> build(PDBFile &File);

private:
  Optional<PdbRaw_ImplVer> Ver;
  Optional<uint32_t> Sig;
  Optional<uint32_t> Age;
  Optional<PDB_UniqueId> Guid;

  NameMapBuilder NamedStreams;
};
}
}

#endif
