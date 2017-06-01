//===- ModStream.h - PDB Module Info Stream Access ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_MODSTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_MODSTREAM_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/ModuleSubstream.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/MSF/StreamArray.h"
#include "llvm/DebugInfo/MSF/StreamRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {
class PDBFile;
class ModInfo;

class ModStream {
public:
  ModStream(const ModInfo &Module,
            std::unique_ptr<msf::MappedBlockStream> Stream);
  ~ModStream();

  Error reload();

  uint32_t signature() const { return Signature; }

  iterator_range<codeview::CVSymbolArray::Iterator>
  symbols(bool *HadError) const;

  iterator_range<codeview::ModuleSubstreamArray::Iterator>
  lines(bool *HadError) const;

  Error commit();

private:
  const ModInfo &Mod;

  uint32_t Signature;

  std::unique_ptr<msf::MappedBlockStream> Stream;

  codeview::CVSymbolArray SymbolsSubstream;
  msf::ReadableStreamRef LinesSubstream;
  msf::ReadableStreamRef C13LinesSubstream;
  msf::ReadableStreamRef GlobalRefsSubstream;

  codeview::ModuleSubstreamArray LineInfo;
};
}
}

#endif
