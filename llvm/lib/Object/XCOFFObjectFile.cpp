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

enum { XCOFF32FileHeaderSize = 20 };
static_assert(sizeof(XCOFFFileHeader) == XCOFF32FileHeaderSize,
              "Wrong size for XCOFF file header.");

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

template <typename T> static const T *viewAs(uintptr_t in) {
  return reinterpret_cast<const T *>(in);
}

const XCOFFSectionHeader *XCOFFObjectFile::toSection(DataRefImpl Ref) const {
  auto Sec = viewAs<XCOFFSectionHeader>(Ref.p);
#ifndef NDEBUG
  if (Sec < SectionHdrTablePtr ||
      Sec >= (SectionHdrTablePtr + getNumberOfSections()))
    report_fatal_error("Section header outside of section header table.");

  uintptr_t Offset = uintptr_t(Sec) - uintptr_t(SectionHdrTablePtr);
  if (Offset % getSectionHeaderSize() != 0)
    report_fatal_error(
        "Section header pointer does not point to a valid section header.");
#endif
  return Sec;
}

// The next 2 functions are not exactly necessary yet, but they are useful to
// abstract over the size difference between XCOFF32 and XCOFF64 structure
// definitions.
size_t XCOFFObjectFile::getFileHeaderSize() const {
  return sizeof(XCOFFFileHeader);
}

size_t XCOFFObjectFile::getSectionHeaderSize() const {
  return sizeof(XCOFFSectionHeader);
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
  const char *Ptr = reinterpret_cast<const char *>(Sec.p);
  Sec.p = reinterpret_cast<uintptr_t>(Ptr + getSectionHeaderSize());
}

Expected<StringRef> XCOFFObjectFile::getSectionName(DataRefImpl Sec) const {
  const char *Name = toSection(Sec)->Name;
  auto NulCharPtr =
      static_cast<const char *>(memchr(Name, '\0', XCOFF::SectionNameSize));
  return NulCharPtr ? StringRef(Name, NulCharPtr - Name)
                    : StringRef(Name, XCOFF::SectionNameSize);
}

uint64_t XCOFFObjectFile::getSectionAddress(DataRefImpl Sec) const {
  return toSection(Sec)->VirtualAddress;
}

uint64_t XCOFFObjectFile::getSectionIndex(DataRefImpl Sec) const {
  // Section numbers in XCOFF are numbered beginning at 1. A section number of
  // zero is used to indicate that a symbol is being imported or is undefined.
  return toSection(Sec) - SectionHdrTablePtr + 1;
}

uint64_t XCOFFObjectFile::getSectionSize(DataRefImpl Sec) const {
  return toSection(Sec)->SectionSize;
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
  return toSection(Sec)->Flags & XCOFF::STYP_TEXT;
}

bool XCOFFObjectFile::isSectionData(DataRefImpl Sec) const {
  unsigned Flags = toSection(Sec)->Flags;
  return Flags & (XCOFF::STYP_DATA | XCOFF::STYP_TDATA);
}

bool XCOFFObjectFile::isSectionBSS(DataRefImpl Sec) const {
  unsigned Flags = toSection(Sec)->Flags;
  return Flags & (XCOFF::STYP_BSS | XCOFF::STYP_TBSS);
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
  DataRefImpl DRI;
  DRI.p = reinterpret_cast<uintptr_t>(SectionHdrTablePtr);
  return section_iterator(SectionRef(DRI, this));
}

section_iterator XCOFFObjectFile::section_end() const {
  DataRefImpl DRI;
  DRI.p =
      reinterpret_cast<uintptr_t>(SectionHdrTablePtr + getNumberOfSections());
  return section_iterator(SectionRef(DRI, this));
}

uint8_t XCOFFObjectFile::getBytesInAddress() const {
  // Only support 32-bit object files for now ...
  assert(getFileHeaderSize() ==  XCOFF32FileHeaderSize);
  return 4;
}

StringRef XCOFFObjectFile::getFileFormatName() const {
  assert(getFileHeaderSize() == XCOFF32FileHeaderSize);
  return "aixcoff-rs6000";
}

Triple::ArchType XCOFFObjectFile::getArch() const {
  assert(getFileHeaderSize() == XCOFF32FileHeaderSize);
  return Triple::ppc;
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

Expected<uint64_t> XCOFFObjectFile::getStartAddress() const {
  // TODO FIXME Should get from auxiliary_header->o_entry when support for the
  // auxiliary_header is added.
  return 0;
}

XCOFFObjectFile::XCOFFObjectFile(MemoryBufferRef Object, std::error_code &EC)
    : ObjectFile(Binary::ID_XCOFF32, Object) {

  // Current location within the file.
  uint64_t CurPtr = 0;

  if ((EC = getObject(FileHdrPtr, Data, base() + CurPtr)))
    return;

  CurPtr += getFileHeaderSize();
  // TODO FIXME we don't have support for an optional header yet, so just skip
  // past it.
  CurPtr += FileHdrPtr->AuxHeaderSize;

  if (getNumberOfSections() != 0) {
    if ((EC = getObject(SectionHdrTablePtr, Data, base() + CurPtr,
                        getNumberOfSections() * getSectionHeaderSize())))
      return;
  }
}

uint16_t XCOFFObjectFile::getMagic() const {
  return FileHdrPtr->Magic;
}

uint16_t XCOFFObjectFile::getNumberOfSections() const {
  return FileHdrPtr->NumberOfSections;
}

int32_t XCOFFObjectFile::getTimeStamp() const {
  return FileHdrPtr->TimeStamp;
}

uint32_t XCOFFObjectFile::getSymbolTableOffset() const {
  return FileHdrPtr->SymbolTableOffset;
}

int32_t XCOFFObjectFile::getNumberOfSymbolTableEntries() const {
  // As far as symbol table size is concerned, if this field is negative it is
  // to be treated as a 0. However since this field is also used for printing we
  // don't want to truncate any negative values.
  return FileHdrPtr->NumberOfSymTableEntries;
}

uint16_t XCOFFObjectFile::getOptionalHeaderSize() const {
  return FileHdrPtr->AuxHeaderSize;
}

uint16_t XCOFFObjectFile::getFlags() const {
  return FileHdrPtr->Flags;
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
