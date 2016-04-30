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

#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/ByteStream.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace pdb {
class PDBFile;

class DbiStream {
  struct HeaderInfo;

public:
  DbiStream(PDBFile &File);
  ~DbiStream();
  std::error_code reload();

  PdbRaw_DbiVer getDbiVersion() const;
  uint32_t getAge() const;

  bool isIncrementallyLinked() const;
  bool hasCTypes() const;
  bool isStripped() const;

  uint16_t getBuildMajorVersion() const;
  uint16_t getBuildMinorVersion() const;

  uint32_t getPdbDllVersion() const;

  uint32_t getNumberOfSymbols() const;

  PDB_Machine getMachineType() const;

  ArrayRef<ModuleInfoEx> modules() const;

private:
  std::error_code initializeFileInfo();

  PDBFile &Pdb;
  MappedBlockStream Stream;

  std::vector<ModuleInfoEx> ModuleInfos;

  ByteStream ModInfoSubstream;
  ByteStream SecContrSubstream;
  ByteStream SecMapSubstream;
  ByteStream FileInfoSubstream;
  ByteStream TypeServerMapSubstream;
  ByteStream ECSubstream;
  ByteStream DbgHeader;

  std::unique_ptr<HeaderInfo> Header;
};
}
}

#endif
