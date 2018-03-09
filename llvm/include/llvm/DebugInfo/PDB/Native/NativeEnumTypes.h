//==- NativeEnumTypes.h - Native Type Enumerator impl ------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_NATIVEENUMTYPES_H
#define LLVM_DEBUGINFO_PDB_NATIVE_NATIVEENUMTYPES_H

#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"

#include <vector>

namespace llvm {
namespace pdb {

class NativeSession;

class NativeEnumTypes : public IPDBEnumChildren<PDBSymbol> {
public:
  NativeEnumTypes(NativeSession &Session,
                  codeview::LazyRandomTypeCollection &TypeCollection,
                  codeview::TypeLeafKind Kind);

  uint32_t getChildCount() const override;
  std::unique_ptr<PDBSymbol> getChildAtIndex(uint32_t Index) const override;
  std::unique_ptr<PDBSymbol> getNext() override;
  void reset() override;
  NativeEnumTypes *clone() const override;

private:
  NativeEnumTypes(NativeSession &Session,
                  const std::vector<codeview::TypeIndex> &Matches,
                  codeview::TypeLeafKind Kind);

  std::vector<codeview::TypeIndex> Matches;
  uint32_t Index;
  NativeSession &Session;
  codeview::TypeLeafKind Kind;
};

} // namespace pdb
} // namespace llvm

#endif
