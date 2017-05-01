//===- ModuleDebugStream.h - PDB Module Info Stream Access ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_MODULEDEBUGSTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_MODULEDEBUGSTREAM_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/ModuleDebugFragmentRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class PDBFile;
class DbiModuleDescriptor;

class ModuleDebugStreamRef {
  typedef codeview::ModuleDebugFragmentArray::Iterator
      LinesAndChecksumsIterator;

public:
  ModuleDebugStreamRef(const DbiModuleDescriptor &Module,
                       std::unique_ptr<msf::MappedBlockStream> Stream);
  ~ModuleDebugStreamRef();

  Error reload();

  uint32_t signature() const { return Signature; }

  iterator_range<codeview::CVSymbolArray::Iterator>
  symbols(bool *HadError) const;

  llvm::iterator_range<LinesAndChecksumsIterator> linesAndChecksums() const;

  bool hasLineInfo() const;

  Error commit();

private:
  const DbiModuleDescriptor &Mod;

  uint32_t Signature;

  std::unique_ptr<msf::MappedBlockStream> Stream;

  codeview::CVSymbolArray SymbolsSubstream;
  BinaryStreamRef C11LinesSubstream;
  BinaryStreamRef C13LinesSubstream;
  BinaryStreamRef GlobalRefsSubstream;

  codeview::ModuleDebugFragmentArray LinesAndChecksums;
};
}
}

#endif
