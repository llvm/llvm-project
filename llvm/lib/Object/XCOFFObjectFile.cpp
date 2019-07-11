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

// Checks that [Ptr, Ptr + Size) bytes fall inside the memory buffer
// 'M'. Returns a pointer to the underlying object on success.
template <typename T>
static Expected<const T *> getObject(MemoryBufferRef M, const void *Ptr,
                                     const uint64_t Size = sizeof(T)) {
  uintptr_t Addr = uintptr_t(Ptr);
  if (std::error_code EC = Binary::checkOffset(M, Addr, Size))
    return errorCodeToError(EC);
  return reinterpret_cast<const T *>(Addr);
}

static uintptr_t getWithOffset(uintptr_t Base, ptrdiff_t Offset) {
  return reinterpret_cast<uintptr_t>(reinterpret_cast<const char *>(Base) +
                                     Offset);
}

template <typename T> static const T *viewAs(uintptr_t in) {
  return reinterpret_cast<const T *>(in);
}

static StringRef generateStringRef(const char *Name, uint64_t Size) {
  auto NulCharPtr = static_cast<const char *>(memchr(Name, '\0', Size));
  return NulCharPtr ? StringRef(Name, NulCharPtr - Name)
                    : StringRef(Name, Size);
}

void XCOFFObjectFile::checkSectionAddress(uintptr_t Addr,
                                          uintptr_t TableAddress) const {
  if (Addr < TableAddress)
    report_fatal_error("Section header outside of section header table.");

  uintptr_t Offset = Addr - TableAddress;
  if (Offset >= getSectionHeaderSize() * getNumberOfSections())
    report_fatal_error("Section header outside of section header table.");

  if (Offset % getSectionHeaderSize() != 0)
    report_fatal_error(
        "Section header pointer does not point to a valid section header.");
}

const XCOFFSectionHeader32 *
XCOFFObjectFile::toSection32(DataRefImpl Ref) const {
  assert(!is64Bit() && "32-bit interface called on 64-bit object file.");
#ifndef NDEBUG
  checkSectionAddress(Ref.p, getSectionHeaderTableAddress());
#endif
  return viewAs<XCOFFSectionHeader32>(Ref.p);
}

const XCOFFSectionHeader64 *
XCOFFObjectFile::toSection64(DataRefImpl Ref) const {
  assert(is64Bit() && "64-bit interface called on a 32-bit object file.");
#ifndef NDEBUG
  checkSectionAddress(Ref.p, getSectionHeaderTableAddress());
#endif
  return viewAs<XCOFFSectionHeader64>(Ref.p);
}

const XCOFFSymbolEntry *XCOFFObjectFile::toSymbolEntry(DataRefImpl Ref) const {
  assert(!is64Bit() && "Symbol table support not implemented for 64-bit.");
  assert(Ref.p != 0 && "Symbol table pointer can not be nullptr!");
  auto SymEntPtr = viewAs<XCOFFSymbolEntry>(Ref.p);
  return SymEntPtr;
}

const XCOFFFileHeader32 *XCOFFObjectFile::fileHeader32() const {
  assert(!is64Bit() && "32-bit interface called on 64-bit object file.");
  return static_cast<const XCOFFFileHeader32 *>(FileHeader);
}

const XCOFFFileHeader64 *XCOFFObjectFile::fileHeader64() const {
  assert(is64Bit() && "64-bit interface called on a 32-bit object file.");
  return static_cast<const XCOFFFileHeader64 *>(FileHeader);
}

const XCOFFSectionHeader32 *
XCOFFObjectFile::sectionHeaderTable32() const {
  assert(!is64Bit() && "32-bit interface called on 64-bit object file.");
  return static_cast<const XCOFFSectionHeader32 *>(SectionHeaderTable);
}

const XCOFFSectionHeader64 *
XCOFFObjectFile::sectionHeaderTable64() const {
  assert(is64Bit() && "64-bit interface called on a 32-bit object file.");
  return static_cast<const XCOFFSectionHeader64 *>(SectionHeaderTable);
}

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

  Expected<DataRefImpl> ExpSec = getSectionByNum(SectNum);
  if (!ExpSec)
    return ExpSec.takeError();

  return section_iterator(SectionRef(ExpSec.get(), this));
}

void XCOFFObjectFile::moveSectionNext(DataRefImpl &Sec) const {
  const char *Ptr = reinterpret_cast<const char *>(Sec.p);
  Sec.p = reinterpret_cast<uintptr_t>(Ptr + getSectionHeaderSize());
}

Expected<StringRef> XCOFFObjectFile::getSectionName(DataRefImpl Sec) const {
  return generateStringRef(getSectionNameInternal(Sec), XCOFF::SectionNameSize);
}

uint64_t XCOFFObjectFile::getSectionAddress(DataRefImpl Sec) const {
  // Avoid ternary due to failure to convert the ubig32_t value to a unit64_t
  // with MSVC.
  if (is64Bit())
    return toSection64(Sec)->VirtualAddress;

  return toSection32(Sec)->VirtualAddress;
}

uint64_t XCOFFObjectFile::getSectionIndex(DataRefImpl Sec) const {
  // Section numbers in XCOFF are numbered beginning at 1. A section number of
  // zero is used to indicate that a symbol is being imported or is undefined.
  if (is64Bit())
    return toSection64(Sec) - sectionHeaderTable64() + 1;
  else
    return toSection32(Sec) - sectionHeaderTable32() + 1;
}

uint64_t XCOFFObjectFile::getSectionSize(DataRefImpl Sec) const {
  // Avoid ternary due to failure to convert the ubig32_t value to a unit64_t
  // with MSVC.
  if (is64Bit())
    return toSection64(Sec)->SectionSize;

  return toSection32(Sec)->SectionSize;
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
  return getSectionFlags(Sec) & XCOFF::STYP_TEXT;
}

bool XCOFFObjectFile::isSectionData(DataRefImpl Sec) const {
  uint32_t Flags = getSectionFlags(Sec);
  return Flags & (XCOFF::STYP_DATA | XCOFF::STYP_TDATA);
}

bool XCOFFObjectFile::isSectionBSS(DataRefImpl Sec) const {
  uint32_t Flags = getSectionFlags(Sec);
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
  assert(!is64Bit() && "64-bit support not implemented yet.");
  DataRefImpl SymDRI;
  SymDRI.p = reinterpret_cast<uintptr_t>(SymbolTblPtr);
  return basic_symbol_iterator(SymbolRef(SymDRI, this));
}

basic_symbol_iterator XCOFFObjectFile::symbol_end() const {
  assert(!is64Bit() && "64-bit support not implemented yet.");
  DataRefImpl SymDRI;
  SymDRI.p = reinterpret_cast<uintptr_t>(
      SymbolTblPtr + getLogicalNumberOfSymbolTableEntries32());
  return basic_symbol_iterator(SymbolRef(SymDRI, this));
}

section_iterator XCOFFObjectFile::section_begin() const {
  DataRefImpl DRI;
  DRI.p = getSectionHeaderTableAddress();
  return section_iterator(SectionRef(DRI, this));
}

section_iterator XCOFFObjectFile::section_end() const {
  DataRefImpl DRI;
  DRI.p = getWithOffset(getSectionHeaderTableAddress(),
                        getNumberOfSections() * getSectionHeaderSize());
  return section_iterator(SectionRef(DRI, this));
}

uint8_t XCOFFObjectFile::getBytesInAddress() const { return is64Bit() ? 8 : 4; }

StringRef XCOFFObjectFile::getFileFormatName() const {
  return is64Bit() ? "aix5coff64-rs6000" : "aixcoff-rs6000";
}

Triple::ArchType XCOFFObjectFile::getArch() const {
  return is64Bit() ? Triple::ppc64 : Triple::ppc;
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

size_t XCOFFObjectFile::getFileHeaderSize() const {
  return is64Bit() ? sizeof(XCOFFFileHeader64) : sizeof(XCOFFFileHeader32);
}

size_t XCOFFObjectFile::getSectionHeaderSize() const {
  return is64Bit() ? sizeof(XCOFFSectionHeader64) :
                     sizeof(XCOFFSectionHeader32);
}

bool XCOFFObjectFile::is64Bit() const {
  return Binary::ID_XCOFF64 == getType();
}

uint16_t XCOFFObjectFile::getMagic() const {
  return is64Bit() ? fileHeader64()->Magic : fileHeader32()->Magic;
}

Expected<DataRefImpl> XCOFFObjectFile::getSectionByNum(int16_t Num) const {
  if (Num <= 0 || Num > getNumberOfSections())
    return errorCodeToError(object_error::invalid_section_index);

  DataRefImpl DRI;
  DRI.p = getWithOffset(getSectionHeaderTableAddress(),
                        getSectionHeaderSize() * (Num - 1));
  return DRI;
}

Expected<StringRef>
XCOFFObjectFile::getSymbolSectionName(const XCOFFSymbolEntry *SymEntPtr) const {
  assert(!is64Bit() && "Symbol table support not implemented for 64-bit.");
  int16_t SectionNum = SymEntPtr->SectionNumber;

  switch (SectionNum) {
  case XCOFF::N_DEBUG:
    return "N_DEBUG";
  case XCOFF::N_ABS:
    return "N_ABS";
  case XCOFF::N_UNDEF:
    return "N_UNDEF";
  default:
    Expected<DataRefImpl> SecRef = getSectionByNum(SectionNum);
    if (SecRef)
      return generateStringRef(getSectionNameInternal(SecRef.get()),
                               XCOFF::SectionNameSize);
    return SecRef.takeError();
  }
}

bool XCOFFObjectFile::isReservedSectionNumber(int16_t SectionNumber) {
  return (SectionNumber <= 0 && SectionNumber >= -2);
}

uint16_t XCOFFObjectFile::getNumberOfSections() const {
  return is64Bit() ? fileHeader64()->NumberOfSections
                   : fileHeader32()->NumberOfSections;
}

int32_t XCOFFObjectFile::getTimeStamp() const {
  return is64Bit() ? fileHeader64()->TimeStamp : fileHeader32()->TimeStamp;
}

uint16_t XCOFFObjectFile::getOptionalHeaderSize() const {
  return is64Bit() ? fileHeader64()->AuxHeaderSize
                   : fileHeader32()->AuxHeaderSize;
}

uint32_t XCOFFObjectFile::getSymbolTableOffset32() const {
  return fileHeader32()->SymbolTableOffset;
}

int32_t XCOFFObjectFile::getRawNumberOfSymbolTableEntries32() const {
  // As far as symbol table size is concerned, if this field is negative it is
  // to be treated as a 0. However since this field is also used for printing we
  // don't want to truncate any negative values.
  return fileHeader32()->NumberOfSymTableEntries;
}

uint32_t XCOFFObjectFile::getLogicalNumberOfSymbolTableEntries32() const {
  return (fileHeader32()->NumberOfSymTableEntries >= 0
              ? fileHeader32()->NumberOfSymTableEntries
              : 0);
}

uint64_t XCOFFObjectFile::getSymbolTableOffset64() const {
  return fileHeader64()->SymbolTableOffset;
}

uint32_t XCOFFObjectFile::getNumberOfSymbolTableEntries64() const {
  return fileHeader64()->NumberOfSymTableEntries;
}

uint16_t XCOFFObjectFile::getFlags() const {
  return is64Bit() ? fileHeader64()->Flags : fileHeader32()->Flags;
}

const char *XCOFFObjectFile::getSectionNameInternal(DataRefImpl Sec) const {
  return is64Bit() ? toSection64(Sec)->Name : toSection32(Sec)->Name;
}

uintptr_t XCOFFObjectFile::getSectionHeaderTableAddress() const {
  return reinterpret_cast<uintptr_t>(SectionHeaderTable);
}

int32_t XCOFFObjectFile::getSectionFlags(DataRefImpl Sec) const {
  return is64Bit() ? toSection64(Sec)->Flags : toSection32(Sec)->Flags;
}

XCOFFObjectFile::XCOFFObjectFile(unsigned int Type, MemoryBufferRef Object)
    : ObjectFile(Type, Object) {
  assert(Type == Binary::ID_XCOFF32 || Type == Binary::ID_XCOFF64);
}

ArrayRef<XCOFFSectionHeader64> XCOFFObjectFile::sections64() const {
  assert(is64Bit() && "64-bit interface called for non 64-bit file.");
  const XCOFFSectionHeader64 *TablePtr = sectionHeaderTable64();
  return ArrayRef<XCOFFSectionHeader64>(TablePtr,
                                        TablePtr + getNumberOfSections());
}

ArrayRef<XCOFFSectionHeader32> XCOFFObjectFile::sections32() const {
  assert(!is64Bit() && "32-bit interface called for non 32-bit file.");
  const XCOFFSectionHeader32 *TablePtr = sectionHeaderTable32();
  return ArrayRef<XCOFFSectionHeader32>(TablePtr,
                                        TablePtr + getNumberOfSections());
}

Expected<XCOFFStringTable>
XCOFFObjectFile::parseStringTable(const XCOFFObjectFile *Obj, uint64_t Offset) {
  // If there is a string table, then the buffer must contain at least 4 bytes
  // for the string table's size. Not having a string table is not an error.
  if (auto EC = Binary::checkOffset(
          Obj->Data, reinterpret_cast<uintptr_t>(Obj->base() + Offset), 4))
    return XCOFFStringTable{0, nullptr};

  // Read the size out of the buffer.
  uint32_t Size = support::endian::read32be(Obj->base() + Offset);

  // If the size is less then 4, then the string table is just a size and no
  // string data.
  if (Size <= 4)
    return XCOFFStringTable{4, nullptr};

  auto StringTableOrErr =
      getObject<char>(Obj->Data, Obj->base() + Offset, Size);
  if (Error E = StringTableOrErr.takeError())
    return std::move(E);

  const char *StringTablePtr = StringTableOrErr.get();
  if (StringTablePtr[Size - 1] != '\0')
    return errorCodeToError(object_error::string_table_non_null_end);

  return XCOFFStringTable{Size, StringTablePtr};
}

Expected<std::unique_ptr<XCOFFObjectFile>>
XCOFFObjectFile::create(unsigned Type, MemoryBufferRef MBR) {
  // Can't use make_unique because of the private constructor.
  std::unique_ptr<XCOFFObjectFile> Obj;
  Obj.reset(new XCOFFObjectFile(Type, MBR));

  uint64_t CurOffset = 0;
  const auto *Base = Obj->base();
  MemoryBufferRef Data = Obj->Data;

  // Parse file header.
  auto FileHeaderOrErr =
      getObject<void>(Data, Base + CurOffset, Obj->getFileHeaderSize());
  if (Error E = FileHeaderOrErr.takeError())
    return std::move(E);
  Obj->FileHeader = FileHeaderOrErr.get();

  CurOffset += Obj->getFileHeaderSize();
  // TODO FIXME we don't have support for an optional header yet, so just skip
  // past it.
  CurOffset += Obj->getOptionalHeaderSize();

  // Parse the section header table if it is present.
  if (Obj->getNumberOfSections()) {
    auto SecHeadersOrErr = getObject<void>(Data, Base + CurOffset,
                                           Obj->getNumberOfSections() *
                                               Obj->getSectionHeaderSize());
    if (Error E = SecHeadersOrErr.takeError())
      return std::move(E);
    Obj->SectionHeaderTable = SecHeadersOrErr.get();
  }

  // 64-bit object supports only file header and section headers for now.
  if (Obj->is64Bit())
    return std::move(Obj);

  // If there is no symbol table we are done parsing the memory buffer.
  if (Obj->getLogicalNumberOfSymbolTableEntries32() == 0)
    return std::move(Obj);

  // Parse symbol table.
  CurOffset = Obj->fileHeader32()->SymbolTableOffset;
  uint64_t SymbolTableSize = (uint64_t)(sizeof(XCOFFSymbolEntry)) *
                             Obj->getLogicalNumberOfSymbolTableEntries32();
  auto SymTableOrErr =
      getObject<XCOFFSymbolEntry>(Data, Base + CurOffset, SymbolTableSize);
  if (Error E = SymTableOrErr.takeError())
    return std::move(E);
  Obj->SymbolTblPtr = SymTableOrErr.get();
  CurOffset += SymbolTableSize;

  // Parse String table.
  Expected<XCOFFStringTable> StringTableOrErr =
      parseStringTable(Obj.get(), CurOffset);
  if (Error E = StringTableOrErr.takeError())
    return std::move(E);
  Obj->StringTable = StringTableOrErr.get();

  return std::move(Obj);
}

Expected<std::unique_ptr<ObjectFile>>
ObjectFile::createXCOFFObjectFile(MemoryBufferRef MemBufRef,
                                  unsigned FileType) {
  return XCOFFObjectFile::create(FileType, MemBufRef);
}

StringRef XCOFFSectionHeader32::getName() const {
  return generateStringRef(Name, XCOFF::SectionNameSize);
}

StringRef XCOFFSectionHeader64::getName() const {
  return generateStringRef(Name, XCOFF::SectionNameSize);
}

} // namespace object
} // namespace llvm
