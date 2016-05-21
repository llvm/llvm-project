//===- SymbolStream.cpp - PDB Symbol Stream Access --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBSYMBOLSTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBSYMBOLSTREAM_H

#include "llvm/DebugInfo/CodeView/TypeStream.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/DebugInfo/PDB/Raw/ByteStream.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"

#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class PDBFile;

class SymbolStream {
public:
  SymbolStream(PDBFile &File, uint32_t StreamNum);
  ~SymbolStream();
  Error reload();

  Expected<std::string> getSymbolName(uint32_t Offset) const;

private:
  MappedBlockStream Stream;
};
}
}

#endif
