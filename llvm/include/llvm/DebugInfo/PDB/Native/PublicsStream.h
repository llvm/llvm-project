//===- PublicsStream.h - PDB Public Symbol Stream -------- ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_PUBLICSSTREAM_H
#define LLVM_DEBUGINFO_PDB_NATIVE_PUBLICSSTREAM_H

#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace msf {
class MappedBlockStream;
}
namespace codeview {
class PublicSym32;
}
namespace pdb {
struct PublicsStreamHeader;
struct SectionOffset;
class SymbolStream;

class PublicsStream {
public:
  LLVM_ABI PublicsStream(std::unique_ptr<msf::MappedBlockStream> Stream);
  LLVM_ABI ~PublicsStream();
  LLVM_ABI Error reload();

  LLVM_ABI uint32_t getSymHash() const;
  LLVM_ABI uint16_t getThunkTableSection() const;
  LLVM_ABI uint32_t getThunkTableOffset() const;
  const GSIHashTable &getPublicsTable() const { return PublicsTable; }
  FixedStreamArray<support::ulittle32_t> getAddressMap() const {
    return AddressMap;
  }
  FixedStreamArray<support::ulittle32_t> getThunkMap() const {
    return ThunkMap;
  }
  FixedStreamArray<SectionOffset> getSectionOffsets() const {
    return SectionOffsets;
  }

  /// Find a public symbol by a segment and offset.
  ///
  /// In case there is more than one symbol (for example due to ICF), the first
  /// one is returned.
  ///
  /// \return If a symbol was found, the symbol at the provided address is
  ///     returned as well as the index of this symbol in the address map. If
  ///     the binary was linked with ICF, there might be more symbols with the
  ///     same address after the returned one. If no symbol is found,
  ///     `std::nullopt` is returned.
  LLVM_ABI std::optional<std::pair<codeview::PublicSym32, size_t>>
  findByAddress(const SymbolStream &Symbols, uint16_t Segment,
                uint32_t Offset) const;

private:
  std::unique_ptr<msf::MappedBlockStream> Stream;
  GSIHashTable PublicsTable;
  FixedStreamArray<support::ulittle32_t> AddressMap;
  FixedStreamArray<support::ulittle32_t> ThunkMap;
  FixedStreamArray<SectionOffset> SectionOffsets;

  const PublicsStreamHeader *Header;
};
}
}

#endif
