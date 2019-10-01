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

// Sets EC and returns false if there is less than 'Size' bytes left in the
// buffer at 'Offset'.
static bool checkSize(MemoryBufferRef M, std::error_code &EC, uint64_t Offset,
                      uint64_t Size) {
  if (M.getBufferSize() < Offset + Size) {
    EC = object_error::unexpected_eof;
    return false;
  }
  return true;
}

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

static StringRef generateStringRef(const char *Name, uint64_t Size) {
  auto NulCharPtr = static_cast<const char *>(memchr(Name, '\0', Size));
  return NulCharPtr ? StringRef(Name, NulCharPtr - Name)
                    : StringRef(Name, Size);
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

const XCOFFSymbolEntry *XCOFFObjectFile::toSymbolEntry(DataRefImpl Ref) const {
  assert(Ref.p != 0 && "Symbol table pointer can not be nullptr!");
  auto SymEntPtr = viewAs<XCOFFSymbolEntry>(Ref.p);
  return SymEntPtr;
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

uint16_t XCOFFObjectFile::getMagic() const { return FileHdrPtr->Magic; }

void XCOFFObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  const XCOFFSymbolEntry *SymEntPtr = toSymbolEntry(Symb);

  SymEntPtr += SymEntPtr->NumberOfAuxEntries + 1;
  Symb.p = reinterpret_cast<uintptr_t>(SymEntPtr);
}

Expected<StringRef> XCOFFObjectFile::getSymbolName(DataRefImpl Symb) const {
  const XCOFFSymbolEntry *SymEntPtr = toSymbolEntry(Symb);

  if (SymEntPtr->NameInStrTbl.Magic != XCOFFSymbolEntry::NAME_IN_STR_TBL_MAGIC)
    return generateStringRef(SymEntPtr->SymbolName, XCOFF::SymbolNameSize);

  // A storage class value with the high-order bit on indicates that the name is
  // a symbolic debugger stabstring.
  if (SymEntPtr->StorageClass & 0x80)
    return StringRef("Unimplemented Debug Name");

  uint32_t Offset = SymEntPtr->NameInStrTbl.Offset;
  // The byte offset is relative to the start of the string table
  // or .debug section. A byte offset value of 0 is a null or zero-length symbol
  // name. A byte offset in the range 1 to 3 (inclusive) points into the length
  // field; as a soft-error recovery mechanism, we treat such cases as having an
  // offset of 0.
  if (Offset < 4)
    return StringRef(nullptr, 0);

  if (StringTable.Data != nullptr && StringTable.Size > Offset)
    return (StringTable.Data + Offset);

  return make_error<GenericBinaryError>("Symbol Name parse failed",
                                        object_error::parse_failed);
}

Expected<uint64_t> XCOFFObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  uint64_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

uint64_t XCOFFObjectFile::getSymbolValueImpl(DataRefImpl Symb) const {
  return toSymbolEntry(Symb)->Value;
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
  const XCOFFSymbolEntry *SymEntPtr = toSymbolEntry(Symb);
  int16_t SectNum = SymEntPtr->SectionNumber;

  if (isReservedSectionNumber(SectNum))
    return section_end();

  const XCOFFSectionHeader *Sec;
  if (std::error_code EC = getSectionByNum(SectNum, Sec))
    return errorCodeToError(EC);

  DataRefImpl SecDRI;
  SecDRI.p = reinterpret_cast<uintptr_t>(Sec);

  return section_iterator(SectionRef(SecDRI, this));
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

Expected<ArrayRef<uint8_t>>
XCOFFObjectFile::getSectionContents(DataRefImpl Sec) const {
  llvm_unreachable("Not yet implemented!");
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
  DataRefImpl SymDRI;
  SymDRI.p = reinterpret_cast<uintptr_t>(SymbolTblPtr);
  return basic_symbol_iterator(SymbolRef(SymDRI, this));
}

basic_symbol_iterator XCOFFObjectFile::symbol_end() const {
  DataRefImpl SymDRI;
  SymDRI.p = reinterpret_cast<uintptr_t>(
      SymbolTblPtr + getLogicalNumberOfSymbolTableEntries());
  return basic_symbol_iterator(SymbolRef(SymDRI, this));
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
  assert(getFileHeaderSize() == XCOFF32FileHeaderSize);
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

std::error_code
XCOFFObjectFile::getSectionByNum(int16_t Num,
                                 const XCOFFSectionHeader *&Result) const {
  if (Num > 0 && static_cast<uint16_t>(Num) <= getNumberOfSections()) {
    Result = SectionHdrTablePtr + (Num - 1);
    return std::error_code();
  }

  return object_error::invalid_section_index;
}

Expected<StringRef>
XCOFFObjectFile::getSymbolSectionName(const XCOFFSymbolEntry *SymEntPtr) const {
  int16_t SectionNum = SymEntPtr->SectionNumber;

  switch (SectionNum) {
  case XCOFF::N_DEBUG:
    return "N_DEBUG";
  case XCOFF::N_ABS:
    return "N_ABS";
  case XCOFF::N_UNDEF:
    return "N_UNDEF";
  default: {
    const XCOFFSectionHeader *SectHeaderPtr;
    std::error_code EC;
    if ((EC = getSectionByNum(SectionNum, SectHeaderPtr)))
      return errorCodeToError(EC);
    else
      return generateStringRef(SectHeaderPtr->Name, XCOFF::SectionNameSize);
  }
  }
}

bool XCOFFObjectFile::isReservedSectionNumber(int16_t SectionNumber) {
  return (SectionNumber <= 0 && SectionNumber >= -2);
}

uint16_t XCOFFObjectFile::getNumberOfSections() const {
  return FileHdrPtr->NumberOfSections;
}

int32_t XCOFFObjectFile::getTimeStamp() const { return FileHdrPtr->TimeStamp; }

uint32_t XCOFFObjectFile::getSymbolTableOffset() const {
  return FileHdrPtr->SymbolTableOffset;
}

int32_t XCOFFObjectFile::getRawNumberOfSymbolTableEntries() const {
  return FileHdrPtr->NumberOfSymTableEntries;
}

uint32_t XCOFFObjectFile::getLogicalNumberOfSymbolTableEntries() const {
  return (FileHdrPtr->NumberOfSymTableEntries >= 0
              ? FileHdrPtr->NumberOfSymTableEntries
              : 0);
}

uint16_t XCOFFObjectFile::getOptionalHeaderSize() const {
  return FileHdrPtr->AuxHeaderSize;
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

  if (getLogicalNumberOfSymbolTableEntries() == 0)
    return;

  // Get pointer to the symbol table.
  CurPtr = FileHdrPtr->SymbolTableOffset;
  uint64_t SymbolTableSize = (uint64_t)(sizeof(XCOFFSymbolEntry)) *
                             getLogicalNumberOfSymbolTableEntries();

  if ((EC = getObject(SymbolTblPtr, Data, base() + CurPtr, SymbolTableSize)))
    return;

  // Move pointer to the string table.
  CurPtr += SymbolTableSize;

  if (CurPtr + 4 > Data.getBufferSize())
    return;

  StringTable.Size = support::endian::read32be(base() + CurPtr);

  if (StringTable.Size <= 4)
    return;

  // Check for whether the String table has the size indicated by length
  // field
  if (!checkSize(Data, EC, CurPtr, StringTable.Size))
    return;

  StringTable.Data = reinterpret_cast<const char *>(base() + CurPtr);
  if (StringTable.Data[StringTable.Size - 1] != '\0') {
    EC = object_error::string_table_non_null_end;
    return;
  }
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
