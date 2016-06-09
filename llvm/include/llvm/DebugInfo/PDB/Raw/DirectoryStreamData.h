//===- DirectoryStreamData.h ---------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_DIRECTORYSTREAMDATA_H
#define LLVM_DEBUGINFO_PDB_RAW_DIRECTORYSTREAMDATA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/PDB/Raw/IPDBStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace pdb {
class IPDBFile;

class DirectoryStreamData : public IPDBStreamData {
public:
  DirectoryStreamData(const PDBFile &File) : File(File) {}

  virtual uint32_t getLength() { return File.getNumDirectoryBytes(); }
  virtual llvm::ArrayRef<llvm::support::ulittle32_t> getStreamBlocks() {
    return File.getDirectoryBlockArray();
  }

private:
  const PDBFile &File;
};
}
}

#endif
