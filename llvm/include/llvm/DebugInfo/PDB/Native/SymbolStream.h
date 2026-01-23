//===- SymbolStream.cpp - PDB Symbol Stream Access --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_SYMBOLSTREAM_H
#define LLVM_DEBUGINFO_PDB_NATIVE_SYMBOLSTREAM_H

#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/Support/Compiler.h"

#include "llvm/Support/Error.h"

namespace llvm {
namespace msf {
class MappedBlockStream;
}
namespace pdb {

class SymbolStream {
public:
  LLVM_ABI SymbolStream(std::unique_ptr<msf::MappedBlockStream> Stream);
  LLVM_ABI ~SymbolStream();
  LLVM_ABI Error reload();

  const codeview::CVSymbolArray &getSymbolArray() const {
    return SymbolRecords;
  }

  LLVM_ABI codeview::CVSymbol readRecord(uint32_t Offset) const;

  LLVM_ABI iterator_range<codeview::CVSymbolArray::Iterator>
  getSymbols(bool *HadError) const;

  LLVM_ABI Error commit();

private:
  codeview::CVSymbolArray SymbolRecords;
  std::unique_ptr<msf::MappedBlockStream> Stream;
};
} // namespace pdb
}

#endif
