//===-- PdbUtil.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYMBOLFILENATIVEPDB_PDBUTIL_H
#define LLDB_PLUGINS_SYMBOLFILENATIVEPDB_PDBUTIL_H

#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"

#include <tuple>
#include <utility>

namespace lldb_private {
namespace npdb {

struct SegmentOffset {
  SegmentOffset() = default;
  SegmentOffset(uint16_t s, uint32_t o) : segment(s), offset(o) {}
  uint16_t segment = 0;
  uint32_t offset = 0;
};

struct SegmentOffsetLength {
  SegmentOffsetLength() = default;
  SegmentOffsetLength(uint16_t s, uint32_t o, uint32_t l)
      : so(s, o), length(l) {}
  SegmentOffset so;
  uint32_t length = 0;
};

llvm::pdb::PDB_SymType CVSymToPDBSym(llvm::codeview::SymbolKind kind);

bool SymbolHasAddress(const llvm::codeview::CVSymbol &sym);
bool SymbolIsCode(const llvm::codeview::CVSymbol &sym);

SegmentOffset GetSegmentAndOffset(const llvm::codeview::CVSymbol &sym);
SegmentOffsetLength
GetSegmentOffsetAndLength(const llvm::codeview::CVSymbol &sym);

template <typename RecordT> bool IsValidRecord(const RecordT &sym) {
  return true;
}

inline bool IsValidRecord(const llvm::codeview::ProcRefSym &sym) {
  // S_PROCREF symbols have 1-based module indices.
  return sym.Module > 0;
}

} // namespace npdb
} // namespace lldb_private

#endif
