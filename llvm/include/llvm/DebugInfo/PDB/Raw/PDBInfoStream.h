//===- PDBInfoStream.h - PDB Info Stream (Stream 1) Access ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBINFOSTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBINFOSTREAM_H

#include "llvm/ADT/StringMap.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/PDBNameMap.h"
#include "llvm/DebugInfo/PDB/Raw/PDBRawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/PDBStream.h"

#include "llvm/Support/Endian.h"

namespace llvm {

class PDBInfoStream {
public:
  PDBInfoStream(PDBFile &File);

  std::error_code reload();

  PdbRaw_ImplVer getVersion() const;
  uint32_t getSignature() const;
  uint32_t getAge() const;
  PDB_UniqueId getGuid() const;

  uint32_t getNamedStreamIndex(llvm::StringRef Name) const;

  PDBFile &getFile() { return Pdb; }

private:
  PDBFile &Pdb;
  PDBStream Stream1;

  // PDB file format version.  We only support VC70.  See the enumeration
  // `PdbRaw_ImplVer` for the other possible values.
  uint32_t Version;

  // A 32-bit signature unique across all PDBs.  This is generated with
  // a call to time() when the PDB is written, but obviously this is not
  // universally unique.
  uint32_t Signature;

  // The number of times the PDB has been written.  Might also be used to
  // ensure that the PDB matches the executable.
  uint32_t Age;

  // Due to the aforementioned limitations with `Signature`, this is a new
  // signature present on VC70 and higher PDBs which is guaranteed to be
  // universally unique.
  PDB_UniqueId Guid;

  PDBNameMap NamedStreams;
};
}

#endif
