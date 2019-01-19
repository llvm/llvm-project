//===- SymbolStream.cpp - PDB Symbol Stream Access --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_PDBSYMBOLSTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_PDBSYMBOLSTREAM_H

#include "llvm/DebugInfo/CodeView/SymbolRecord.h"

#include "llvm/Support/Error.h"

namespace llvm {
namespace msf {
class MappedBlockStream;
}
namespace pdb {
class PDBFile;

class SymbolStream {
public:
  SymbolStream(std::unique_ptr<msf::MappedBlockStream> Stream);
  ~SymbolStream();
  Error reload();

  const codeview::CVSymbolArray &getSymbolArray() const {
    return SymbolRecords;
  }

  codeview::CVSymbol readRecord(uint32_t Offset) const;

  iterator_range<codeview::CVSymbolArray::Iterator>
  getSymbols(bool *HadError) const;

  Error commit();

private:
  codeview::CVSymbolArray SymbolRecords;
  std::unique_ptr<msf::MappedBlockStream> Stream;
};
}
}

#endif
