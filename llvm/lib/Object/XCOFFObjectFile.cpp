//===--- XCOFFObjectFile.cpp - XCOFF object file implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the XCOFFObjectFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cstddef>
#include <cstring>

namespace llvm {
namespace object {

// Sets Obj unless any bytes in [addr, addr + size) fall outsize of m.
// Returns unexpected_eof on error.
template <typename T>
static std::error_code getObject(const T *&Obj, MemoryBufferRef M,
                                 const void *Ptr,
                                 const uint64_t Size = sizeof(T)) {
  uintptr_t Addr = uintptr_t(Ptr);
  if (std::error_code EC = Binary::checkOffset(M, Addr, Size))
    return EC;
  Obj = reinterpret_cast<const T *>(Addr);
  return std::error_code();
}

void XCOFFObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  llvm_unreachable("Not yet implemented!");
  return;
}

Expected<StringRef> XCOFFObjectFile::getSymbolName(DataRefImpl Symb) const {
  StringRef Result;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

Expected<uint64_t> XCOFFObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  uint64_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

uint64_t XCOFFObjectFile::getSymbolValueImpl(DataRefImpl Symb) const {
  uint64_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

uint64_t XCOFFObjectFile::getCommonSymbolSizeImpl(DataRefImpl Symb) const {
  uint64_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

Expected<SymbolRef::Type>
XCOFFObjectFile::getSymbolType(DataRefImpl Symb) const {
  llvm_unreachable("Not yet implemented!");
  return SymbolRef::ST_Other;
}

Expected<section_iterator>
XCOFFObjectFile::getSymbolSection(DataRefImpl Symb) const {
  llvm_unreachable("Not yet implemented!");
  return section_iterator(SectionRef());
}

void XCOFFObjectFile::moveSectionNext(DataRefImpl &Sec) const {
  llvm_unreachable("Not yet implemented!");
  return;
}

std::error_code XCOFFObjectFile::getSectionName(DataRefImpl Sec,
                                                StringRef &Res) const {
  llvm_unreachable("Not yet implemented!");
  return std::error_code();
}

uint64_t XCOFFObjectFile::getSectionAddress(DataRefImpl Sec) const {
  uint64_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

uint64_t XCOFFObjectFile::getSectionIndex(DataRefImpl Sec) const {
  uint64_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

uint64_t XCOFFObjectFile::getSectionSize(DataRefImpl Sec) const {
  uint64_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

std::error_code XCOFFObjectFile::getSectionContents(DataRefImpl Sec,
                                                    StringRef &Res) const {
  llvm_unreachable("Not yet implemented!");
  return std::error_code();
}

uint64_t XCOFFObjectFile::getSectionAlignment(DataRefImpl Sec) const {
  uint64_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

bool XCOFFObjectFile::isSectionCompressed(DataRefImpl Sec) const {
  bool Result = false;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

bool XCOFFObjectFile::isSectionText(DataRefImpl Sec) const {
  bool Result = false;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

bool XCOFFObjectFile::isSectionData(DataRefImpl Sec) const {
  bool Result = false;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

bool XCOFFObjectFile::isSectionBSS(DataRefImpl Sec) const {
  bool Result = false;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

bool XCOFFObjectFile::isSectionVirtual(DataRefImpl Sec) const {
  bool Result = false;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

relocation_iterator XCOFFObjectFile::section_rel_begin(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented!");
  return relocation_iterator(RelocationRef());
}

relocation_iterator XCOFFObjectFile::section_rel_end(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented!");
  return relocation_iterator(RelocationRef());
}

void XCOFFObjectFile::moveRelocationNext(DataRefImpl &Rel) const {
  llvm_unreachable("Not yet implemented!");
  return;
}

uint64_t XCOFFObjectFile::getRelocationOffset(DataRefImpl Rel) const {
  llvm_unreachable("Not yet implemented!");
  uint64_t Result = 0;
  return Result;
}

symbol_iterator XCOFFObjectFile::getRelocationSymbol(DataRefImpl Rel) const {
  llvm_unreachable("Not yet implemented!");
  return symbol_iterator(SymbolRef());
}

uint64_t XCOFFObjectFile::getRelocationType(DataRefImpl Rel) const {
  llvm_unreachable("Not yet implemented!");
  uint64_t Result = 0;
  return Result;
}

void XCOFFObjectFile::getRelocationTypeName(
    DataRefImpl Rel, SmallVectorImpl<char> &Result) const {
  llvm_unreachable("Not yet implemented!");
  return;
}

uint32_t XCOFFObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  uint32_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

basic_symbol_iterator XCOFFObjectFile::symbol_begin() const {
  llvm_unreachable("Not yet implemented!");
  return basic_symbol_iterator(SymbolRef());
}

basic_symbol_iterator XCOFFObjectFile::symbol_end() const {
  llvm_unreachable("Not yet implemented!");
  return basic_symbol_iterator(SymbolRef());
}

section_iterator XCOFFObjectFile::section_begin() const {
  llvm_unreachable("Not yet implemented!");
  return section_iterator(SectionRef());
}

section_iterator XCOFFObjectFile::section_end() const {
  llvm_unreachable("Not yet implemented!");
  return section_iterator(SectionRef());
}

uint8_t XCOFFObjectFile::getBytesInAddress() const {
  uint8_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

StringRef XCOFFObjectFile::getFileFormatName() const {
  llvm_unreachable("Not yet implemented!");
  return "";
}

Triple::ArchType XCOFFObjectFile::getArch() const {
  llvm_unreachable("Not yet implemented!");
  return Triple::UnknownArch;
}

SubtargetFeatures XCOFFObjectFile::getFeatures() const {
  llvm_unreachable("Not yet implemented!");
  return SubtargetFeatures();
}

bool XCOFFObjectFile::isRelocatableObject() const {
  bool Result = false;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

XCOFFObjectFile::XCOFFObjectFile(MemoryBufferRef Object, std::error_code &EC)
    : ObjectFile(Binary::ID_XCOFF32, Object) {

  // Current location within the file.
  uint64_t CurPtr = 0;

  if ((EC = getObject(FileHdrPtr, Data, base() + CurPtr)))
    return;
}

Expected<std::unique_ptr<ObjectFile>>
ObjectFile::createXCOFFObjectFile(MemoryBufferRef Object) {
  StringRef Data = Object.getBuffer();
  file_magic Type = identify_magic(Data);
  std::error_code EC;
  std::unique_ptr<ObjectFile> Ret;

  if (Type == file_magic::xcoff_object_32) {
    Ret.reset(new XCOFFObjectFile(Object, EC));
  } else {
    llvm_unreachable("Encountered an unexpected binary file type!");
  }

  if (EC)
    return errorCodeToError(EC);
  return std::move(Ret);
}

} // namespace object
} // namespace llvm
