//==- SymbolCache.h - Cache of native symbols and ids ------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_SYMBOLCACHE_H
#define LLVM_DEBUGINFO_PDB_NATIVE_SYMBOLCACHE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Native/NativeRawSymbol.h"
#include "llvm/Support/Allocator.h"

#include <memory>
#include <vector>

namespace llvm {
namespace pdb {
class DbiStream;
class PDBFile;

class SymbolCache {
  NativeSession &Session;
  DbiStream *Dbi = nullptr;

  std::vector<std::unique_ptr<NativeRawSymbol>> Cache;
  DenseMap<codeview::TypeIndex, SymIndexId> TypeIndexToSymbolId;
  std::vector<SymIndexId> Compilands;

  SymIndexId createSymbolPlaceholder() {
    SymIndexId Id = Cache.size();
    Cache.push_back(nullptr);
    return Id;
  }

public:
  SymbolCache(NativeSession &Session, DbiStream *Dbi);

  template <typename ConcreteSymbolT, typename... Args>
  SymIndexId createSymbol(Args &&... ConstructorArgs) {
    SymIndexId Id = Cache.size();
    std::unique_ptr<ConcreteSymbolT> Symbol =
        llvm::make_unique<ConcreteSymbolT>(
            Session, Id, std::forward<Args>(ConstructorArgs)...);
    std::unique_ptr<NativeRawSymbol> NRS = std::move(Symbol);
    Cache.push_back(std::move(NRS));
    return Id;
  }

  std::unique_ptr<PDBSymbolTypeEnum>
  createEnumSymbol(codeview::TypeIndex Index);

  std::unique_ptr<IPDBEnumSymbols>
  createTypeEnumerator(codeview::TypeLeafKind Kind);

  SymIndexId findSymbolByTypeIndex(codeview::TypeIndex TI);

  std::unique_ptr<PDBSymbolCompiland> getOrCreateCompiland(uint32_t Index);
  uint32_t getNumCompilands() const;

  std::unique_ptr<PDBSymbol> getSymbolById(SymIndexId SymbolId) const;

  NativeRawSymbol &getNativeSymbolById(SymIndexId SymbolId) const;

  template <typename ConcreteT>
  ConcreteT &getNativeSymbolById(SymIndexId SymbolId) const {
    return static_cast<ConcreteT &>(getNativeSymbolById(SymbolId));
  }
};

} // namespace pdb
} // namespace llvm

#endif
