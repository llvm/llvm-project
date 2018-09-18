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
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
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
  DenseMap<std::pair<codeview::TypeIndex, uint32_t>, SymIndexId>
      FieldListMembersToSymbolId;
  std::vector<SymIndexId> Compilands;

  SymIndexId createSymbolPlaceholder() {
    SymIndexId Id = Cache.size();
    Cache.push_back(nullptr);
    return Id;
  }

  template <typename ConcreteSymbolT, typename CVRecordT, typename... Args>
  SymIndexId createSymbolForType(codeview::TypeIndex TI, codeview::CVType CVT,
                                 Args &&... ConstructorArgs) {
    CVRecordT Record;
    if (auto EC =
            codeview::TypeDeserializer::deserializeAs<CVRecordT>(CVT, Record)) {
      consumeError(std::move(EC));
      return 0;
    }

    return createSymbol<ConcreteSymbolT>(
        TI, std::move(Record), std::forward<Args>(ConstructorArgs)...);
  }

  SymIndexId createSymbolForModifiedType(codeview::TypeIndex ModifierTI,
                                         codeview::CVType CVT);

  SymIndexId createSimpleType(codeview::TypeIndex TI,
                              codeview::ModifierOptions Mods);

public:
  SymbolCache(NativeSession &Session, DbiStream *Dbi);

  template <typename ConcreteSymbolT, typename... Args>
  SymIndexId createSymbol(Args &&... ConstructorArgs) {
    SymIndexId Id = Cache.size();

    auto Result = llvm::make_unique<ConcreteSymbolT>(
        Session, Id, std::forward<Args>(ConstructorArgs)...);
    Cache.push_back(std::move(Result));
    return Id;
  }

  std::unique_ptr<IPDBEnumSymbols>
  createTypeEnumerator(codeview::TypeLeafKind Kind);

  SymIndexId findSymbolByTypeIndex(codeview::TypeIndex TI);

  template <typename ConcreteSymbolT, typename... Args>
  SymIndexId getOrCreateFieldListMember(codeview::TypeIndex FieldListTI,
                                        uint32_t Index,
                                        Args &&... ConstructorArgs) {
    SymIndexId SymId = Cache.size();
    std::pair<codeview::TypeIndex, uint32_t> Key{FieldListTI, Index};
    auto Result = FieldListMembersToSymbolId.try_emplace(Key, SymId);
    if (Result.second) {
      auto NewSymbol = llvm::make_unique<ConcreteSymbolT>(
          Session, SymId, std::forward<Args>(ConstructorArgs)...);
      Cache.push_back(std::move(NewSymbol));
    } else {
      SymId = Result.first->second;
    }
    return SymId;
  }

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
