//===- PDBDbiStream.h - PDB Dbi Stream (Stream 3) Access --------*- C++ -*-===//
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
#include "llvm/DebugInfo/PDB/Raw/PDBRawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/PDBStream.h"
#include "llvm/Support/Endian.h"

namespace llvm {
class PDBFile;

class PDBDbiStream {
  struct HeaderInfo;

public:
  PDBDbiStream(PDBFile &File);
  ~PDBDbiStream();
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

private:
  PDBFile &Pdb;
  PDBStream Stream;
  std::unique_ptr<HeaderInfo> Header;
};
}

#endif
