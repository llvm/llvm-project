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
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/DataExtractor.h"
#include <cstddef>
#include <cstring>

namespace llvm {

using namespace XCOFF;

namespace object {

static const uint8_t FunctionSym = 0x20;
static const uint8_t SymTypeMask = 0x07;
static const uint16_t NoRelMask = 0x0001;

// Checks that [Ptr, Ptr + Size) bytes fall inside the memory buffer
// 'M'. Returns a pointer to the underlying object on success.
template <typename T>
static Expected<const T *> getObject(MemoryBufferRef M, const void *Ptr,
                                     const uint64_t Size = sizeof(T)) {
  uintptr_t Addr = uintptr_t(Ptr);
  if (Error E = Binary::checkOffset(M, Addr, Size))
    return std::move(E);
  return reinterpret_cast<const T *>(Addr);
}

static uintptr_t getWithOffset(uintptr_t Base, ptrdiff_t Offset) {
  return reinterpret_cast<uintptr_t>(reinterpret_cast<const char *>(Base) +
                                     Offset);
}

template <typename T> static const T *viewAs(uintptr_t in) {
  return reinterpret_cast<const T *>(in);
}

static StringRef generateXCOFFFixedNameStringRef(const char *Name) {
  auto NulCharPtr =
      static_cast<const char *>(memchr(Name, '\0', XCOFF::NameSize));
  return NulCharPtr ? StringRef(Name, NulCharPtr - Name)
                    : StringRef(Name, XCOFF::NameSize);
}

template <typename T> StringRef XCOFFSectionHeader<T>::getName() const {
  const T &DerivedXCOFFSectionHeader = static_cast<const T &>(*this);
  return generateXCOFFFixedNameStringRef(DerivedXCOFFSectionHeader.Name);
}

template <typename T> uint16_t XCOFFSectionHeader<T>::getSectionType() const {
  const T &DerivedXCOFFSectionHeader = static_cast<const T &>(*this);
  return DerivedXCOFFSectionHeader.Flags & SectionFlagsTypeMask;
}

template <typename T>
bool XCOFFSectionHeader<T>::isReservedSectionType() const {
  return getSectionType() & SectionFlagsReservedMask;
}

bool XCOFFRelocation32::isRelocationSigned() const {
  return Info & XR_SIGN_INDICATOR_MASK;
}

bool XCOFFRelocation32::isFixupIndicated() const {
  return Info & XR_FIXUP_INDICATOR_MASK;
}

uint8_t XCOFFRelocation32::getRelocatedLength() const {
  // The relocation encodes the bit length being relocated minus 1. Add back
  // the 1 to get the actual length being relocated.
  return (Info & XR_BIASED_LENGTH_MASK) + 1;
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
#ifndef NDEBUG
  checkSymbolEntryPointer(Ref.p);
#endif
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
#ifndef NDEBUG
  // This function is used by basic_symbol_iterator, which allows to
  // point to the end-of-symbol-table address.
  if (reinterpret_cast<uintptr_t>(SymEntPtr) != getEndOfSymbolTableAddress())
    checkSymbolEntryPointer(reinterpret_cast<uintptr_t>(SymEntPtr));
#endif
  Symb.p = reinterpret_cast<uintptr_t>(SymEntPtr);
}

Expected<StringRef>
XCOFFObjectFile::getStringTableEntry(uint32_t Offset) const {
  // The byte offset is relative to the start of the string table.
  // A byte offset value of 0 is a null or zero-length symbol
  // name. A byte offset in the range 1 to 3 (inclusive) points into the length
  // field; as a soft-error recovery mechanism, we treat such cases as having an
  // offset of 0.
  if (Offset < 4)
    return StringRef(nullptr, 0);

  if (StringTable.Data != nullptr && StringTable.Size > Offset)
    return (StringTable.Data + Offset);

  return make_error<GenericBinaryError>("Bad offset for string table entry",
                                        object_error::parse_failed);
}

Expected<StringRef>
XCOFFObjectFile::getCFileName(const XCOFFFileAuxEnt *CFileEntPtr) const {
  if (CFileEntPtr->NameInStrTbl.Magic !=
      XCOFFSymbolEntry::NAME_IN_STR_TBL_MAGIC)
    return generateXCOFFFixedNameStringRef(CFileEntPtr->Name);
  return getStringTableEntry(CFileEntPtr->NameInStrTbl.Offset);
}

Expected<StringRef> XCOFFObjectFile::getSymbolName(DataRefImpl Symb) const {
  const XCOFFSymbolEntry *SymEntPtr = toSymbolEntry(Symb);

  // A storage class value with the high-order bit on indicates that the name is
  // a symbolic debugger stabstring.
  if (SymEntPtr->StorageClass & 0x80)
    return StringRef("Unimplemented Debug Name");

  if (SymEntPtr->NameInStrTbl.Magic != XCOFFSymbolEntry::NAME_IN_STR_TBL_MAGIC)
    return generateXCOFFFixedNameStringRef(SymEntPtr->SymbolName);

  return getStringTableEntry(SymEntPtr->NameInStrTbl.Offset);
}

Expected<uint64_t> XCOFFObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  assert(!is64Bit() && "Symbol table support not implemented for 64-bit.");
  return toSymbolEntry(Symb)->Value;
}

uint64_t XCOFFObjectFile::getSymbolValueImpl(DataRefImpl Symb) const {
  assert(!is64Bit() && "Symbol table support not implemented for 64-bit.");
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
  return generateXCOFFFixedNameStringRef(getSectionNameInternal(Sec));
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
  if (isSectionVirtual(Sec))
    return ArrayRef<uint8_t>();

  uint64_t OffsetToRaw;
  if (is64Bit())
    OffsetToRaw = toSection64(Sec)->FileOffsetToRawData;
  else
    OffsetToRaw = toSection32(Sec)->FileOffsetToRawData;

  const uint8_t * ContentStart = base() + OffsetToRaw;
  uint64_t SectionSize = getSectionSize(Sec);
  if (checkOffset(Data, uintptr_t(ContentStart), SectionSize))
    return make_error<BinaryError>();

  return makeArrayRef(ContentStart,SectionSize);
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
  return is64Bit() ? toSection64(Sec)->FileOffsetToRawData == 0
                   : toSection32(Sec)->FileOffsetToRawData == 0;
}

relocation_iterator XCOFFObjectFile::section_rel_begin(DataRefImpl Sec) const {
  if (is64Bit())
    report_fatal_error("64-bit support not implemented yet");
  const XCOFFSectionHeader32 *SectionEntPtr = toSection32(Sec);
  auto RelocationsOrErr = relocations(*SectionEntPtr);
  if (Error E = RelocationsOrErr.takeError())
    return relocation_iterator(RelocationRef());
  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(&*RelocationsOrErr.get().begin());
  return relocation_iterator(RelocationRef(Ret, this));
}

relocation_iterator XCOFFObjectFile::section_rel_end(DataRefImpl Sec) const {
  if (is64Bit())
    report_fatal_error("64-bit support not implemented yet");
  const XCOFFSectionHeader32 *SectionEntPtr = toSection32(Sec);
  auto RelocationsOrErr = relocations(*SectionEntPtr);
  if (Error E = RelocationsOrErr.takeError())
    return relocation_iterator(RelocationRef());
  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(&*RelocationsOrErr.get().end());
  return relocation_iterator(RelocationRef(Ret, this));
}

void XCOFFObjectFile::moveRelocationNext(DataRefImpl &Rel) const {
  Rel.p = reinterpret_cast<uintptr_t>(viewAs<XCOFFRelocation32>(Rel.p) + 1);
}

uint64_t XCOFFObjectFile::getRelocationOffset(DataRefImpl Rel) const {
  if (is64Bit())
    report_fatal_error("64-bit support not implemented yet");
  const XCOFFRelocation32 *Reloc = viewAs<XCOFFRelocation32>(Rel.p);
  const XCOFFSectionHeader32 *Sec32 = sectionHeaderTable32();
  const uint32_t RelocAddress = Reloc->VirtualAddress;
  const uint16_t NumberOfSections = getNumberOfSections();
  for (uint16_t i = 0; i < NumberOfSections; ++i) {
    // Find which section this relocation is belonging to, and get the
    // relocation offset relative to the start of the section.
    if (Sec32->VirtualAddress <= RelocAddress &&
        RelocAddress < Sec32->VirtualAddress + Sec32->SectionSize) {
      return RelocAddress - Sec32->VirtualAddress;
    }
    ++Sec32;
  }
  return InvalidRelocOffset;
}

symbol_iterator XCOFFObjectFile::getRelocationSymbol(DataRefImpl Rel) const {
  if (is64Bit())
    report_fatal_error("64-bit support not implemented yet");
  const XCOFFRelocation32 *Reloc = viewAs<XCOFFRelocation32>(Rel.p);
  const uint32_t Index = Reloc->SymbolIndex;

  if (Index >= getLogicalNumberOfSymbolTableEntries32())
    return symbol_end();

  DataRefImpl SymDRI;
  SymDRI.p = reinterpret_cast<uintptr_t>(getPointerToSymbolTable() + Index);
  return symbol_iterator(SymbolRef(SymDRI, this));
}

uint64_t XCOFFObjectFile::getRelocationType(DataRefImpl Rel) const {
  if (is64Bit())
    report_fatal_error("64-bit support not implemented yet");
  return viewAs<XCOFFRelocation32>(Rel.p)->Type;
}

void XCOFFObjectFile::getRelocationTypeName(
    DataRefImpl Rel, SmallVectorImpl<char> &Result) const {
  if (is64Bit())
    report_fatal_error("64-bit support not implemented yet");
  const XCOFFRelocation32 *Reloc = viewAs<XCOFFRelocation32>(Rel.p);
  StringRef Res = XCOFF::getRelocationTypeString(Reloc->Type);
  Result.append(Res.begin(), Res.end());
}

Expected<uint32_t> XCOFFObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  uint32_t Result = 0;
  llvm_unreachable("Not yet implemented!");
  return Result;
}

basic_symbol_iterator XCOFFObjectFile::symbol_begin() const {
  if (is64Bit())
    report_fatal_error("64-bit support not implemented yet");
  DataRefImpl SymDRI;
  SymDRI.p = reinterpret_cast<uintptr_t>(SymbolTblPtr);
  return basic_symbol_iterator(SymbolRef(SymDRI, this));
}

basic_symbol_iterator XCOFFObjectFile::symbol_end() const {
  if (is64Bit())
    report_fatal_error("64-bit support not implemented yet");
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
  return SubtargetFeatures();
}

bool XCOFFObjectFile::isRelocatableObject() const {
  if (is64Bit())
    report_fatal_error("64-bit support not implemented yet");
  return !(fileHeader32()->Flags & NoRelMask);
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
      return generateXCOFFFixedNameStringRef(
          getSectionNameInternal(SecRef.get()));
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

uintptr_t XCOFFObjectFile::getEndOfSymbolTableAddress() const {
  uint32_t NumberOfSymTableEntries =
      is64Bit() ? getNumberOfSymbolTableEntries64()
                : getLogicalNumberOfSymbolTableEntries32();
  return getWithOffset(reinterpret_cast<uintptr_t>(SymbolTblPtr),
                       XCOFF::SymbolTableEntrySize * NumberOfSymTableEntries);
}

void XCOFFObjectFile::checkSymbolEntryPointer(uintptr_t SymbolEntPtr) const {
  if (SymbolEntPtr < reinterpret_cast<uintptr_t>(SymbolTblPtr))
    report_fatal_error("Symbol table entry is outside of symbol table.");

  if (SymbolEntPtr >= getEndOfSymbolTableAddress())
    report_fatal_error("Symbol table entry is outside of symbol table.");

  ptrdiff_t Offset = reinterpret_cast<const char *>(SymbolEntPtr) -
                     reinterpret_cast<const char *>(SymbolTblPtr);

  if (Offset % XCOFF::SymbolTableEntrySize != 0)
    report_fatal_error(
        "Symbol table entry position is not valid inside of symbol table.");
}

uint32_t XCOFFObjectFile::getSymbolIndex(uintptr_t SymbolEntPtr) const {
  return (reinterpret_cast<const char *>(SymbolEntPtr) -
          reinterpret_cast<const char *>(SymbolTblPtr)) /
         XCOFF::SymbolTableEntrySize;
}

Expected<StringRef>
XCOFFObjectFile::getSymbolNameByIndex(uint32_t Index) const {
  if (is64Bit())
    report_fatal_error("64-bit symbol table support not implemented yet.");

  if (Index >= getLogicalNumberOfSymbolTableEntries32())
    return errorCodeToError(object_error::invalid_symbol_index);

  DataRefImpl SymDRI;
  SymDRI.p = reinterpret_cast<uintptr_t>(getPointerToSymbolTable() + Index);
  return getSymbolName(SymDRI);
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

// In an XCOFF32 file, when the field value is 65535, then an STYP_OVRFLO
// section header contains the actual count of relocation entries in the s_paddr
// field. STYP_OVRFLO headers contain the section index of their corresponding
// sections as their raw "NumberOfRelocations" field value.
Expected<uint32_t> XCOFFObjectFile::getLogicalNumberOfRelocationEntries(
    const XCOFFSectionHeader32 &Sec) const {

  uint16_t SectionIndex = &Sec - sectionHeaderTable32() + 1;

  if (Sec.NumberOfRelocations < XCOFF::RelocOverflow)
    return Sec.NumberOfRelocations;
  for (const auto &Sec : sections32()) {
    if (Sec.Flags == XCOFF::STYP_OVRFLO &&
        Sec.NumberOfRelocations == SectionIndex)
      return Sec.PhysicalAddress;
  }
  return errorCodeToError(object_error::parse_failed);
}

Expected<ArrayRef<XCOFFRelocation32>>
XCOFFObjectFile::relocations(const XCOFFSectionHeader32 &Sec) const {
  uintptr_t RelocAddr = getWithOffset(reinterpret_cast<uintptr_t>(FileHeader),
                                      Sec.FileOffsetToRelocationInfo);
  auto NumRelocEntriesOrErr = getLogicalNumberOfRelocationEntries(Sec);
  if (Error E = NumRelocEntriesOrErr.takeError())
    return std::move(E);

  uint32_t NumRelocEntries = NumRelocEntriesOrErr.get();

  assert(sizeof(XCOFFRelocation32) == XCOFF::RelocationSerializationSize32);
  auto RelocationOrErr =
      getObject<XCOFFRelocation32>(Data, reinterpret_cast<void *>(RelocAddr),
                                   NumRelocEntries * sizeof(XCOFFRelocation32));
  if (Error E = RelocationOrErr.takeError())
    return std::move(E);

  const XCOFFRelocation32 *StartReloc = RelocationOrErr.get();

  return ArrayRef<XCOFFRelocation32>(StartReloc, StartReloc + NumRelocEntries);
}

Expected<XCOFFStringTable>
XCOFFObjectFile::parseStringTable(const XCOFFObjectFile *Obj, uint64_t Offset) {
  // If there is a string table, then the buffer must contain at least 4 bytes
  // for the string table's size. Not having a string table is not an error.
  if (Error E = Binary::checkOffset(
          Obj->Data, reinterpret_cast<uintptr_t>(Obj->base() + Offset), 4)) {
    consumeError(std::move(E));
    return XCOFFStringTable{0, nullptr};
  }

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
  // Can't use std::make_unique because of the private constructor.
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

XCOFF::StorageClass XCOFFSymbolRef::getStorageClass() const {
  return OwningObjectPtr->toSymbolEntry(SymEntDataRef)->StorageClass;
}

uint8_t XCOFFSymbolRef::getNumberOfAuxEntries() const {
  return OwningObjectPtr->toSymbolEntry(SymEntDataRef)->NumberOfAuxEntries;
}

// TODO: The function needs to return an error if there is no csect auxiliary
// entry.
const XCOFFCsectAuxEnt32 *XCOFFSymbolRef::getXCOFFCsectAuxEnt32() const {
  assert(!OwningObjectPtr->is64Bit() &&
         "32-bit interface called on 64-bit object file.");
  assert(hasCsectAuxEnt() && "No Csect Auxiliary Entry is found.");

  // In XCOFF32, the csect auxilliary entry is always the last auxiliary
  // entry for the symbol.
  uintptr_t AuxAddr = getWithOffset(
      SymEntDataRef.p, XCOFF::SymbolTableEntrySize * getNumberOfAuxEntries());

#ifndef NDEBUG
  OwningObjectPtr->checkSymbolEntryPointer(AuxAddr);
#endif

  return reinterpret_cast<const XCOFFCsectAuxEnt32 *>(AuxAddr);
}

uint16_t XCOFFSymbolRef::getType() const {
  return OwningObjectPtr->toSymbolEntry(SymEntDataRef)->SymbolType;
}

int16_t XCOFFSymbolRef::getSectionNumber() const {
  return OwningObjectPtr->toSymbolEntry(SymEntDataRef)->SectionNumber;
}

// TODO: The function name needs to be changed to express the purpose of the
// function.
bool XCOFFSymbolRef::hasCsectAuxEnt() const {
  XCOFF::StorageClass SC = getStorageClass();
  return (SC == XCOFF::C_EXT || SC == XCOFF::C_WEAKEXT ||
          SC == XCOFF::C_HIDEXT);
}

bool XCOFFSymbolRef::isFunction() const {
  if (OwningObjectPtr->is64Bit())
    report_fatal_error("64-bit support is unimplemented yet.");

  if (getType() & FunctionSym)
    return true;

  if (!hasCsectAuxEnt())
    return false;

  const XCOFFCsectAuxEnt32 *CsectAuxEnt = getXCOFFCsectAuxEnt32();

  // A function definition should be a label definition.
  if ((CsectAuxEnt->SymbolAlignmentAndType & SymTypeMask) != XCOFF::XTY_LD)
    return false;

  if (CsectAuxEnt->StorageMappingClass != XCOFF::XMC_PR)
    return false;

  int16_t SectNum = getSectionNumber();
  Expected<DataRefImpl> SI = OwningObjectPtr->getSectionByNum(SectNum);
  if (!SI)
    return false;

  return (OwningObjectPtr->getSectionFlags(SI.get()) & XCOFF::STYP_TEXT);
}

// Explictly instantiate template classes.
template struct XCOFFSectionHeader<XCOFFSectionHeader32>;
template struct XCOFFSectionHeader<XCOFFSectionHeader64>;

bool doesXCOFFTracebackTableBegin(ArrayRef<uint8_t> Bytes) {
  if (Bytes.size() < 4)
    return false;

  return support::endian::read32be(Bytes.data()) == 0;
}

TBVectorExt::TBVectorExt(StringRef TBvectorStrRef) {
  const uint8_t *Ptr = reinterpret_cast<const uint8_t *>(TBvectorStrRef.data());
  Data = support::endian::read16be(Ptr);
  VecParmsInfo = support::endian::read32be(Ptr + 2);
}

#define GETVALUEWITHMASK(X) (Data & (TracebackTable::X))
#define GETVALUEWITHMASKSHIFT(X, S)                                            \
  ((Data & (TracebackTable::X)) >> (TracebackTable::S))
uint8_t TBVectorExt::getNumberOfVRSaved() const {
  return GETVALUEWITHMASKSHIFT(NumberOfVRSavedMask, NumberOfVRSavedShift);
}

bool TBVectorExt::isVRSavedOnStack() const {
  return GETVALUEWITHMASK(IsVRSavedOnStackMask);
}

bool TBVectorExt::hasVarArgs() const {
  return GETVALUEWITHMASK(HasVarArgsMask);
}
uint8_t TBVectorExt::getNumberOfVectorParms() const {
  return GETVALUEWITHMASKSHIFT(NumberOfVectorParmsMask,
                               NumberOfVectorParmsShift);
}

bool TBVectorExt::hasVMXInstruction() const {
  return GETVALUEWITHMASK(HasVMXInstructionMask);
}
#undef GETVALUEWITHMASK
#undef GETVALUEWITHMASKSHIFT

SmallString<32> TBVectorExt::getVectorParmsInfoString() const {
  SmallString<32> ParmsType;
  uint32_t Value = VecParmsInfo;
  for (uint8_t I = 0; I < getNumberOfVectorParms(); ++I) {
    if (I != 0)
      ParmsType += ", ";
    switch (Value & TracebackTable::ParmTypeMask) {
    case TracebackTable::ParmTypeIsVectorCharBit:
      ParmsType += "vc";
      break;

    case TracebackTable::ParmTypeIsVectorShortBit:
      ParmsType += "vs";
      break;

    case TracebackTable::ParmTypeIsVectorIntBit:
      ParmsType += "vi";
      break;

    case TracebackTable::ParmTypeIsVectorFloatBit:
      ParmsType += "vf";
      break;
    }
    Value <<= 2;
  }
  return ParmsType;
}

static SmallString<32> parseParmsTypeWithVecInfo(uint32_t Value,
                                                 unsigned int ParmsNum) {
  SmallString<32> ParmsType;
  unsigned I = 0;
  bool Begin = false;
  while (I < ParmsNum || Value) {
    if (Begin)
      ParmsType += ", ";
    else
      Begin = true;

    switch (Value & TracebackTable::ParmTypeMask) {
    case TracebackTable::ParmTypeIsFixedBits:
      ParmsType += "i";
      ++I;
      break;
    case TracebackTable::ParmTypeIsVectorBits:
      ParmsType += "v";
      break;
    case TracebackTable::ParmTypeIsFloatingBits:
      ParmsType += "f";
      ++I;
      break;
    case TracebackTable::ParmTypeIsDoubleBits:
      ParmsType += "d";
      ++I;
      break;
    default:
      assert(false && "Unrecognized bits in ParmsType.");
    }
    Value <<= 2;
  }
  assert(I == ParmsNum &&
         "The total parameters number of fixed-point or floating-point "
         "parameters not equal to the number in the parameter type!");
  return ParmsType;
}

static SmallString<32> parseParmsType(uint32_t Value, unsigned ParmsNum) {
  SmallString<32> ParmsType;
  for (unsigned I = 0; I < ParmsNum; ++I) {
    if (I != 0)
      ParmsType += ", ";
    if ((Value & TracebackTable::ParmTypeIsFloatingBit) == 0) {
      // Fixed parameter type.
      ParmsType += "i";
      Value <<= 1;
    } else {
      if ((Value & TracebackTable::ParmTypeFloatingIsDoubleBit) == 0)
        // Float parameter type.
        ParmsType += "f";
      else
        // Double parameter type.
        ParmsType += "d";

      Value <<= 2;
    }
  }
  return ParmsType;
}

Expected<XCOFFTracebackTable> XCOFFTracebackTable::create(const uint8_t *Ptr,
                                                          uint64_t &Size) {
  Error Err = Error::success();
  XCOFFTracebackTable TBT(Ptr, Size, Err);
  if (Err)
    return std::move(Err);
  return TBT;
}

XCOFFTracebackTable::XCOFFTracebackTable(const uint8_t *Ptr, uint64_t &Size,
                                         Error &Err)
    : TBPtr(Ptr) {
  ErrorAsOutParameter EAO(&Err);
  DataExtractor DE(ArrayRef<uint8_t>(Ptr, Size), /*IsLittleEndian=*/false,
                   /*AddressSize=*/0);
  DataExtractor::Cursor Cur(/*Offset=*/0);

  // Skip 8 bytes of mandatory fields.
  DE.getU64(Cur);

  // Begin to parse optional fields.
  if (Cur) {
    unsigned ParmNum = getNumberOfFixedParms() + getNumberOfFPParms();

    // As long as there are no "fixed-point" or floating-point parameters, this
    // field remains not present even when hasVectorInfo gives true and
    // indicates the presence of vector parameters.
    if (ParmNum > 0) {
      uint32_t ParamsTypeValue = DE.getU32(Cur);
      if (Cur)
        ParmsType = hasVectorInfo()
                        ? parseParmsTypeWithVecInfo(ParamsTypeValue, ParmNum)
                        : parseParmsType(ParamsTypeValue, ParmNum);
    }
  }

  if (Cur && hasTraceBackTableOffset())
    TraceBackTableOffset = DE.getU32(Cur);

  if (Cur && isInterruptHandler())
    HandlerMask = DE.getU32(Cur);

  if (Cur && hasControlledStorage()) {
    NumOfCtlAnchors = DE.getU32(Cur);
    if (Cur && NumOfCtlAnchors) {
      SmallVector<uint32_t, 8> Disp;
      Disp.reserve(NumOfCtlAnchors.getValue());
      for (uint32_t I = 0; I < NumOfCtlAnchors && Cur; ++I)
        Disp.push_back(DE.getU32(Cur));
      if (Cur)
        ControlledStorageInfoDisp = std::move(Disp);
    }
  }

  if (Cur && isFuncNamePresent()) {
    uint16_t FunctionNameLen = DE.getU16(Cur);
    if (Cur)
      FunctionName = DE.getBytes(Cur, FunctionNameLen);
  }

  if (Cur && isAllocaUsed())
    AllocaRegister = DE.getU8(Cur);

  if (Cur && hasVectorInfo()) {
    StringRef VectorExtRef = DE.getBytes(Cur, 6);
    if (Cur)
      VecExt = TBVectorExt(VectorExtRef);
  }

  if (Cur && hasExtensionTable())
    ExtensionTable = DE.getU8(Cur);

  if (!Cur)
    Err = Cur.takeError();
  Size = Cur.tell();
}

#define GETBITWITHMASK(P, X)                                                   \
  (support::endian::read32be(TBPtr + (P)) & (TracebackTable::X))
#define GETBITWITHMASKSHIFT(P, X, S)                                           \
  ((support::endian::read32be(TBPtr + (P)) & (TracebackTable::X)) >>           \
   (TracebackTable::S))

uint8_t XCOFFTracebackTable::getVersion() const {
  return GETBITWITHMASKSHIFT(0, VersionMask, VersionShift);
}

uint8_t XCOFFTracebackTable::getLanguageID() const {
  return GETBITWITHMASKSHIFT(0, LanguageIdMask, LanguageIdShift);
}

bool XCOFFTracebackTable::isGlobalLinkage() const {
  return GETBITWITHMASK(0, IsGlobaLinkageMask);
}

bool XCOFFTracebackTable::isOutOfLineEpilogOrPrologue() const {
  return GETBITWITHMASK(0, IsOutOfLineEpilogOrPrologueMask);
}

bool XCOFFTracebackTable::hasTraceBackTableOffset() const {
  return GETBITWITHMASK(0, HasTraceBackTableOffsetMask);
}

bool XCOFFTracebackTable::isInternalProcedure() const {
  return GETBITWITHMASK(0, IsInternalProcedureMask);
}

bool XCOFFTracebackTable::hasControlledStorage() const {
  return GETBITWITHMASK(0, HasControlledStorageMask);
}

bool XCOFFTracebackTable::isTOCless() const {
  return GETBITWITHMASK(0, IsTOClessMask);
}

bool XCOFFTracebackTable::isFloatingPointPresent() const {
  return GETBITWITHMASK(0, IsFloatingPointPresentMask);
}

bool XCOFFTracebackTable::isFloatingPointOperationLogOrAbortEnabled() const {
  return GETBITWITHMASK(0, IsFloatingPointOperationLogOrAbortEnabledMask);
}

bool XCOFFTracebackTable::isInterruptHandler() const {
  return GETBITWITHMASK(0, IsInterruptHandlerMask);
}

bool XCOFFTracebackTable::isFuncNamePresent() const {
  return GETBITWITHMASK(0, IsFunctionNamePresentMask);
}

bool XCOFFTracebackTable::isAllocaUsed() const {
  return GETBITWITHMASK(0, IsAllocaUsedMask);
}

uint8_t XCOFFTracebackTable::getOnConditionDirective() const {
  return GETBITWITHMASKSHIFT(0, OnConditionDirectiveMask,
                             OnConditionDirectiveShift);
}

bool XCOFFTracebackTable::isCRSaved() const {
  return GETBITWITHMASK(0, IsCRSavedMask);
}

bool XCOFFTracebackTable::isLRSaved() const {
  return GETBITWITHMASK(0, IsLRSavedMask);
}

bool XCOFFTracebackTable::isBackChainStored() const {
  return GETBITWITHMASK(4, IsBackChainStoredMask);
}

bool XCOFFTracebackTable::isFixup() const {
  return GETBITWITHMASK(4, IsFixupMask);
}

uint8_t XCOFFTracebackTable::getNumOfFPRsSaved() const {
  return GETBITWITHMASKSHIFT(4, FPRSavedMask, FPRSavedShift);
}

bool XCOFFTracebackTable::hasExtensionTable() const {
  return GETBITWITHMASK(4, HasExtensionTableMask);
}

bool XCOFFTracebackTable::hasVectorInfo() const {
  return GETBITWITHMASK(4, HasVectorInfoMask);
}

uint8_t XCOFFTracebackTable::getNumOfGPRsSaved() const {
  return GETBITWITHMASKSHIFT(4, GPRSavedMask, GPRSavedShift);
}

uint8_t XCOFFTracebackTable::getNumberOfFixedParms() const {
  return GETBITWITHMASKSHIFT(4, NumberOfFixedParmsMask,
                             NumberOfFixedParmsShift);
}

uint8_t XCOFFTracebackTable::getNumberOfFPParms() const {
  return GETBITWITHMASKSHIFT(4, NumberOfFloatingPointParmsMask,
                             NumberOfFloatingPointParmsShift);
}

bool XCOFFTracebackTable::hasParmsOnStack() const {
  return GETBITWITHMASK(4, HasParmsOnStackMask);
}

#undef GETBITWITHMASK
#undef GETBITWITHMASKSHIFT
} // namespace object
} // namespace llvm
