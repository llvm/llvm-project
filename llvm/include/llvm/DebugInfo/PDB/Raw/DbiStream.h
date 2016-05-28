//===- DbiStream.h - PDB Dbi Stream (Stream 3) Access -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBDBISTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBDBISTREAM_H

#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/StreamRef.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/NameHashTable.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class PDBFile;

class DbiStream {
  struct HeaderInfo;

public:
  DbiStream(PDBFile &File);
  ~DbiStream();
  Error reload();

  PdbRaw_DbiVer getDbiVersion() const;
  uint32_t getAge() const;
  uint16_t getPublicSymbolStreamIndex() const;
  uint16_t getGlobalSymbolStreamIndex() const;

  bool isIncrementallyLinked() const;
  bool hasCTypes() const;
  bool isStripped() const;

  uint16_t getBuildMajorVersion() const;
  uint16_t getBuildMinorVersion() const;

  uint32_t getPdbDllVersion() const;

  uint32_t getSymRecordStreamIndex() const;

  PDB_Machine getMachineType() const;

  ArrayRef<ModuleInfoEx> modules() const;

  uint32_t getDebugStreamIndex(DbgHeaderType Type) const;

private:
  Error initializeFileInfo();

  PDBFile &Pdb;
  MappedBlockStream Stream;

  std::vector<ModuleInfoEx> ModuleInfos;
  NameHashTable ECNames;

  codeview::StreamRef ModInfoSubstream;
  codeview::StreamRef SecContrSubstream;
  codeview::StreamRef SecMapSubstream;
  codeview::StreamRef FileInfoSubstream;
  codeview::StreamRef TypeServerMapSubstream;
  codeview::StreamRef ECSubstream;

  codeview::FixedStreamArray<support::ulittle16_t> DbgStreams;

  const HeaderInfo *Header;
};
}
}

#endif
