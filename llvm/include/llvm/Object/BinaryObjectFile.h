//===- BinaryObjectFile.h - Binary object file implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the BinaryObjectFile class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_RAWOBJECTFILE_H
#define LLVM_OBJECT_RAWOBJECTFILE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <cstdint>

namespace llvm {

template <typename T> class SmallVectorImpl;

namespace object {

struct BinarySymbol {
  uint32_t Flags = 0;
  uint64_t Value = 0;
  StringRef Name;
};

struct BinaryRelocation {
  uint64_t Offset = 0;
  uint64_t Symbol = 0;
  uint64_t Type = 0;
};

struct BinarySection {
  uint64_t Offset = 0;
  uint64_t Index = 0;
  uint64_t Address = 0;
  uint64_t Size = 0;
  StringRef Name;
  std::vector<BinaryRelocation> Relocations;
};

class BinaryObjectFile : public ObjectFile {
private:
  std::vector<BinarySymbol> Symbols;
  std::vector<BinarySection> Sections;

public:
  BinaryObjectFile(MemoryBufferRef Source);

  bool is64Bit() const override;

  basic_symbol_iterator symbol_begin() const override;
  basic_symbol_iterator symbol_end() const override;
  section_iterator section_begin() const override;
  section_iterator section_end() const override;

  const BinarySymbol &getBinarySymbol(const DataRefImpl &Symb) const;
  const BinarySymbol &getBinarySymbol(const SymbolRef &Symb) const;

  void moveSymbolNext(DataRefImpl &Symb) const override;
  Expected<StringRef> getSymbolName(DataRefImpl Symb) const override;
  Expected<uint32_t> getSymbolFlags(DataRefImpl Symb) const override;
  Expected<uint64_t> getSymbolAddress(DataRefImpl Symb) const override;
  uint64_t getSymbolValueImpl(DataRefImpl Symb) const override;
  uint64_t getCommonSymbolSizeImpl(DataRefImpl Symb) const override;
  Expected<SymbolRef::Type> getSymbolType(DataRefImpl Symb) const override;
  Expected<section_iterator> getSymbolSection(DataRefImpl Symb) const override;

  const BinarySection &getBinarySection(const DataRefImpl Ref) const;
  const BinarySection &getBinarySection(const SectionRef &Section) const;

  void moveSectionNext(DataRefImpl &Sec) const override;
  Expected<StringRef> getSectionName(DataRefImpl Sec) const override;
  uint64_t getSectionAddress(DataRefImpl Sec) const override;
  uint64_t getSectionIndex(DataRefImpl Sec) const override;
  uint64_t getSectionSize(DataRefImpl Sec) const override;
  Expected<ArrayRef<uint8_t>>
  getSectionContents(DataRefImpl Sec) const override;
  uint64_t getSectionAlignment(DataRefImpl Sec) const override;
  bool isSectionCompressed(DataRefImpl Sec) const override;
  bool isSectionText(DataRefImpl Sec) const override;
  bool isSectionData(DataRefImpl Sec) const override;
  bool isSectionBSS(DataRefImpl Sec) const override;
  bool isSectionVirtual(DataRefImpl Sec) const override;

  relocation_iterator section_rel_begin(DataRefImpl Sec) const override;
  relocation_iterator section_rel_end(DataRefImpl Sec) const override;

  const BinaryRelocation &getBinaryRelocation(const RelocationRef &Ref) const;
  const BinaryRelocation &getBinaryRelocation(DataRefImpl Ref) const;

  // Overrides from RelocationRef.
  void moveRelocationNext(DataRefImpl &Rel) const override;
  uint64_t getRelocationOffset(DataRefImpl Rel) const override;
  symbol_iterator getRelocationSymbol(DataRefImpl Rel) const override;
  uint64_t getRelocationType(DataRefImpl Rel) const override;
  void getRelocationTypeName(DataRefImpl Rel,
                             SmallVectorImpl<char> &Result) const override;

  uint8_t getBytesInAddress() const override;
  StringRef getFileFormatName() const override;
  Triple::ArchType getArch() const override;
  Expected<SubtargetFeatures> getFeatures() const override;
  std::optional<StringRef> tryGetCPUName() const override;
  bool isRelocatableObject() const override;
};

} // end namespace object
} // end namespace llvm

#endif // LLVM_OBJECT_RAWOBJECTFILE_H
