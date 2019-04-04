//===- XCOFFObjectFile.h - XCOFF object file implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the XCOFFObjectFile class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_XCOFFOBJECTFILE_H
#define LLVM_OBJECT_XCOFFOBJECTFILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <system_error>

namespace llvm {
namespace object {

struct XCOFFFileHeader {
  support::ubig16_t Magic;
  support::ubig16_t NumberOfSections;

  // Unix time value, value of 0 indicates no timestamp.
  // Negative values are reserved.
  support::big32_t TimeStamp;

  support::ubig32_t SymbolTableOffset; // File offset to symbol table.
  support::big32_t NumberOfSymTableEntries;
  support::ubig16_t AuxHeaderSize;
  support::ubig16_t Flags;
};

class XCOFFObjectFile : public ObjectFile {
private:
  const XCOFFFileHeader *FileHdrPtr = nullptr;

public:
  void moveSymbolNext(DataRefImpl &Symb) const override;
  uint32_t getSymbolFlags(DataRefImpl Symb) const override;
  basic_symbol_iterator symbol_begin() const override;
  basic_symbol_iterator symbol_end() const override;

  Expected<StringRef> getSymbolName(DataRefImpl Symb) const override;
  Expected<uint64_t> getSymbolAddress(DataRefImpl Symb) const override;
  uint64_t getSymbolValueImpl(DataRefImpl Symb) const override;
  uint64_t getCommonSymbolSizeImpl(DataRefImpl Symb) const override;
  Expected<SymbolRef::Type> getSymbolType(DataRefImpl Symb) const override;
  Expected<section_iterator> getSymbolSection(DataRefImpl Symb) const override;

  void moveSectionNext(DataRefImpl &Sec) const override;
  std::error_code getSectionName(DataRefImpl Sec,
                                 StringRef &Res) const override;
  uint64_t getSectionAddress(DataRefImpl Sec) const override;
  uint64_t getSectionIndex(DataRefImpl Sec) const override;
  uint64_t getSectionSize(DataRefImpl Sec) const override;
  std::error_code getSectionContents(DataRefImpl Sec,
                                     StringRef &Res) const override;
  uint64_t getSectionAlignment(DataRefImpl Sec) const override;
  bool isSectionCompressed(DataRefImpl Sec) const override;
  bool isSectionText(DataRefImpl Sec) const override;
  bool isSectionData(DataRefImpl Sec) const override;
  bool isSectionBSS(DataRefImpl Sec) const override;

  bool isSectionVirtual(DataRefImpl Sec) const override;
  relocation_iterator section_rel_begin(DataRefImpl Sec) const override;
  relocation_iterator section_rel_end(DataRefImpl Sec) const override;

  void moveRelocationNext(DataRefImpl &Rel) const override;
  uint64_t getRelocationOffset(DataRefImpl Rel) const override;
  symbol_iterator getRelocationSymbol(DataRefImpl Rel) const override;
  uint64_t getRelocationType(DataRefImpl Rel) const override;
  void getRelocationTypeName(DataRefImpl Rel,
                             SmallVectorImpl<char> &Result) const override;

  section_iterator section_begin() const override;
  section_iterator section_end() const override;
  uint8_t getBytesInAddress() const override;
  StringRef getFileFormatName() const override;
  Triple::ArchType getArch() const override;
  SubtargetFeatures getFeatures() const override;
  bool isRelocatableObject() const override;

public:
  XCOFFObjectFile(MemoryBufferRef Object, std::error_code &EC);

  const XCOFFFileHeader *getFileHeader() const { return FileHdrPtr; }

}; // XCOFFObjectFile

} // namespace object
} // namespace llvm

#endif // LLVM_OBJECT_XCOFFOBJECTFILE_H
